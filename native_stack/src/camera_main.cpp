#include "kaivuri/frame_store.hpp"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <librealsense2/rs.hpp>
#include <memory>
#include <signal.h>
#include <thread>
#include <unistd.h>
using namespace kaivuri;
static double monotonic_seconds() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static volatile sig_atomic_t stopped = 0;
static void stop_signal(int) { stopped = 1; }
struct Camera {
    std::string serial, path;
    int index = 1, sync = 0;
    rs2::sensor sensor;
    std::unique_ptr<FrameWriter> writer;
    std::atomic<uint64_t> received{0}, dropped{0}, last_ns{0};
    std::atomic<uint64_t> last_device_counter{0}, counter_gaps{0}, counter_resets{0};
    std::atomic<int> error{0};
    bool started = false, opened = false;
    ~Camera() {
        try {
            if (started)
                sensor.stop();
            if (opened)
                sensor.close();
        } catch (...) {
        }
    }
};
static uint64_t mono_ns() { return static_cast<uint64_t>(monotonic_seconds() * 1e9); }
int main(int argc, char **argv) {
    try {
        int width = 640, height = 480, fps = 30;
        double duration = 0, exposure = -1, gain = -1;
        bool list = false, synthetic = false, global_time = true, require_metadata = true;
        std::string output = "/tmp/kaivuri-native-" + std::to_string(geteuid()) + "/cameras",
                    csv_path;
        std::vector<std::string> specs;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto value = [&]() {
                if (++i == argc)
                    throw std::runtime_error("missing option argument");
                return std::string(argv[i]);
            };
            if (a == "--camera")
                specs.push_back(value());
            else if (a == "--width")
                width = std::stoi(value());
            else if (a == "--height")
                height = std::stoi(value());
            else if (a == "--fps")
                fps = std::stoi(value());
            else if (a == "--duration-s")
                duration = std::stod(value());
            else if (a == "--exposure-us")
                exposure = std::stod(value());
            else if (a == "--gain")
                gain = std::stod(value());
            else if (a == "--output-dir")
                output = value();
            else if (a == "--metadata-csv")
                csv_path = value();
            else if (a == "--list")
                list = true;
            else if (a == "--synthetic")
                synthetic = true;
            else if (a == "--hardware-clock")
                global_time = false;
            else if (a == "--allow-missing-metadata")
                require_metadata = false;
            else if (a == "--help") {
                std::puts(
                    "kaivuri-camera --list\nkaivuri-camera --camera SERIAL:IR_INDEX[:SYNC_MODE] "
                    "[--camera SECOND:1]\n  [--width 640 --height 480 --fps 30] [--duration-s SEC] "
                    "[--exposure-us US --gain N]\n  [--hardware-clock] [--allow-missing-metadata] "
                    "[--output-dir DIR] [--metadata-csv FILE]\n  [--synthetic] (hardware-free test "
                    "source; never a physical camera)\nIR index 1 or 2. Sync mode 0 standalone, 1 "
                    "master, 2 slave; requires a sync cable.\nEmitter disabled; no depth/RGB, "
                    "conversion, encoding, or CUDA context.");
                return 0;
            } else
                throw std::runtime_error("unknown camera option: " + a);
        }
        if (width < 1 || height < 1 || uint64_t(width) * height > 16 * 1024 * 1024 || fps < 1 ||
            fps > 90 || duration < 0 || !std::isfinite(duration) || !std::isfinite(exposure) ||
            !std::isfinite(gain))
            throw std::runtime_error("invalid camera options");
        // Context is not constructed in synthetic mode, avoiding USB enumeration and SDK worker
        // threads.
        std::unique_ptr<rs2::context> context;
        std::vector<rs2::device> devices;
        if (!synthetic)
            context = std::make_unique<rs2::context>();
        // Keep one enumeration alive for both cameras. Re-enumerating after
        // starting the first RSUSB stream can reopen its claimed USB interfaces.
        if (context)
            for (auto device : context->query_devices())
                devices.push_back(device);
        if (list) {
            if (synthetic)
                throw std::runtime_error("--list requires real devices");
            for (auto d : devices)
                std::printf("%s  %s  firmware=%s usb=%s port=%s\n",
                            d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER),
                            d.get_info(RS2_CAMERA_INFO_NAME),
                            d.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION),
                            d.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)
                                ? d.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)
                                : "unknown",
                            d.supports(RS2_CAMERA_INFO_PHYSICAL_PORT)
                                ? d.get_info(RS2_CAMERA_INFO_PHYSICAL_PORT)
                                : "unknown");
            return 0;
        }
        if (specs.empty() || specs.size() > 2)
            throw std::runtime_error(
                "select one or two cameras explicitly with --camera SERIAL:IR_INDEX");
        std::filesystem::create_directories(output);
        std::filesystem::permissions(output, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace);
        std::vector<std::unique_ptr<Camera>> cameras;
        for (const auto &spec : specs) {
            auto camera = std::make_unique<Camera>();
            auto first = spec.find(':');
            if (first == std::string::npos)
                throw std::runtime_error("camera syntax SERIAL:IR_INDEX[:SYNC_MODE]");
            auto second = spec.find(':', first + 1);
            camera->serial = spec.substr(0, first);
            camera->index = std::stoi(spec.substr(first + 1, second - first - 1));
            if (second != std::string::npos)
                camera->sync = std::stoi(spec.substr(second + 1));
            if (camera->serial.empty() ||
                camera->serial.find_first_not_of(
                    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_") !=
                    std::string::npos ||
                camera->index < 1 || camera->index > 2 || camera->sync < 0 || camera->sync > 2)
                throw std::runtime_error("invalid camera spec");
            for (const auto &previous : cameras)
                if (previous->serial == camera->serial)
                    throw std::runtime_error("select one IR stream per physical camera");
            camera->path =
                output + "/" + camera->serial + "-ir" + std::to_string(camera->index) + ".frame";
            camera->writer =
                std::make_unique<FrameWriter>(camera->path, width, height, camera->index);
            if (!synthetic) {
                rs2::device selected;
                bool found = false;
                for (auto device : devices)
                    if (camera->serial == device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
                        selected = device;
                        found = true;
                        break;
                    }
                if (!found)
                    throw std::runtime_error("camera serial not found: " + camera->serial);
                bool profile_found = false;
                rs2::stream_profile profile;
                for (auto sensor : selected.query_sensors())
                    for (auto p : sensor.get_stream_profiles()) {
                        auto video = p.as<rs2::video_stream_profile>();
                        if (video && p.stream_type() == RS2_STREAM_INFRARED &&
                            p.stream_index() == camera->index && p.format() == RS2_FORMAT_Y8 &&
                            video.width() == width && video.height() == height && p.fps() == fps) {
                            camera->sensor = sensor;
                            profile = p;
                            profile_found = true;
                            break;
                        }
                    }
                if (!profile_found)
                    throw std::runtime_error("requested Y8 IR profile unavailable: " + spec);
                auto &sensor = camera->sensor;
                if (sensor.supports(RS2_OPTION_EMITTER_ENABLED))
                    sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0);
                if (sensor.supports(RS2_OPTION_FRAMES_QUEUE_SIZE))
                    sensor.set_option(RS2_OPTION_FRAMES_QUEUE_SIZE, 1);
                if (sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED))
                    sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, global_time ? 1 : 0);
                else if (global_time)
                    throw std::runtime_error(
                        "global time option unavailable; use --hardware-clock");
                if (camera->sync) {
                    if (!sensor.supports(RS2_OPTION_INTER_CAM_SYNC_MODE))
                        throw std::runtime_error("camera sync unsupported");
                    sensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, camera->sync);
                } else if (sensor.supports(RS2_OPTION_INTER_CAM_SYNC_MODE))
                    sensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, 0);
                if (exposure >= 0) {
                    if (!sensor.supports(RS2_OPTION_EXPOSURE) ||
                        !sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE))
                        throw std::runtime_error("manual exposure unsupported");
                    sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
                    sensor.set_option(RS2_OPTION_EXPOSURE, exposure);
                }
                if (gain >= 0) {
                    if (!sensor.supports(RS2_OPTION_GAIN))
                        throw std::runtime_error("gain unsupported");
                    sensor.set_option(RS2_OPTION_GAIN, gain);
                }
                sensor.open(profile);
                camera->opened = true;
                Camera *c = camera.get();
                sensor.start([c, require_metadata](rs2::frame frame) {
                    try {
                        auto v = frame.as<rs2::video_frame>();
                        if (!v)
                            return;
                        FrameMetadata m{};
                        m.sequence = ++c->received;
                        m.host_monotonic_ns = mono_ns();
                        c->last_ns = m.host_monotonic_ns;
                        m.sdk_timestamp_ms = frame.get_timestamp();
                        if (!std::isfinite(m.sdk_timestamp_ms) || m.sdk_timestamp_ms < 0) {
                            c->error = 2;
                            return;
                        }
                        m.timestamp_domain = frame.get_frame_timestamp_domain();
                        m.width = v.get_width();
                        m.height = v.get_height();
                        m.stride = v.get_stride_in_bytes();
                        m.ir_index = c->index;
                        m.pixel_bytes = m.width * m.height;
                        const rs2_frame_metadata_value keys[]{
                            RS2_FRAME_METADATA_SENSOR_TIMESTAMP, RS2_FRAME_METADATA_FRAME_TIMESTAMP,
                            RS2_FRAME_METADATA_FRAME_COUNTER, RS2_FRAME_METADATA_ACTUAL_EXPOSURE,
                            RS2_FRAME_METADATA_GAIN_LEVEL};
                        uint64_t *destinations[]{&m.sensor_timestamp_us, &m.frame_timestamp_us,
                                                 &m.frame_counter, &m.actual_exposure_us,
                                                 &m.gain_level};
                        for (int i = 0; i < 5; ++i)
                            if (frame.supports_frame_metadata(keys[i])) {
                                *destinations[i] =
                                    static_cast<uint64_t>(frame.get_frame_metadata(keys[i]));
                                m.valid_metadata |= 1ULL << i;
                            }
                        if (m.valid_metadata & 4) {
                            const auto previous = c->last_device_counter.exchange(m.frame_counter);
                            if (previous && m.frame_counter > previous + 1)
                                c->counter_gaps += m.frame_counter - previous - 1;
                            else if (previous && m.frame_counter <= previous)
                                ++c->counter_resets;
                        }
                        if (require_metadata && (m.valid_metadata & 3) != 3) {
                            c->error = 1;
                            return;
                        }
                        if (!c->writer->publish(m, v.get_data()))
                            ++c->dropped;
                    } catch (...) {
                        c->error = 2;
                    }
                });
                camera->started = true;
            }
            std::fprintf(stderr, "%sIR%d serial=%s -> %s (Y8 %dx%d@%d, sync=%d)\n",
                         synthetic ? "SYNTHETIC " : "", camera->index, camera->serial.c_str(),
                         camera->path.c_str(), width, height, fps, camera->sync);
            cameras.push_back(std::move(camera));
        }
        signal(SIGINT, stop_signal);
        signal(SIGTERM, stop_signal);
        std::ofstream csv;
        if (!csv_path.empty()) {
            csv.open(csv_path);
            if (!csv)
                throw std::runtime_error("cannot open metadata CSV");
            csv << "serial,ir_index,sequence,host_monotonic_ns,sdk_timestamp_ms,domain,sensor_"
                   "timestamp_us,frame_timestamp_us,frame_counter,exposure_us,gain,valid_bits\n";
        }
        std::vector<void *> readers;
        for (auto &c : cameras) {
            void *reader = kaivuri_frame_open(c->path.c_str());
            if (!reader)
                throw std::runtime_error("open frame store reader");
            readers.push_back(reader);
        }
        struct Cleanup {
            std::vector<void *> &handles;
            ~Cleanup() {
                for (void *h : handles)
                    kaivuri_frame_close(h);
            }
        } cleanup{readers};
        std::vector<uint8_t> pixels(size_t(width) * height, 0);
        std::vector<uint64_t> logged(cameras.size(), 0);
        double begin = monotonic_seconds(), next = begin;
        bool failed = false;
        while (!stopped) {
            double now = monotonic_seconds();
            if (duration > 0 && now - begin >= duration)
                break;
            if (synthetic && now >= next) {
                next += 1. / fps;
                for (auto &c : cameras) {
                    FrameMetadata m{};
                    m.sequence = ++c->received;
                    m.host_monotonic_ns = mono_ns();
                    c->last_ns = m.host_monotonic_ns;
                    m.sensor_timestamp_us = static_cast<uint64_t>((now - begin) * 1e6);
                    m.frame_timestamp_us = m.sensor_timestamp_us + 500;
                    m.frame_counter = m.sequence;
                    m.actual_exposure_us = 1000;
                    m.valid_metadata = 31;
                    m.sdk_timestamp_ms = m.sensor_timestamp_us / 1000.;
                    m.timestamp_domain = RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK;
                    m.width = width;
                    m.height = height;
                    m.stride = width;
                    m.ir_index = c->index;
                    m.pixel_bytes = pixels.size();
                    std::fill(pixels.begin(), pixels.end(), static_cast<uint8_t>(m.sequence % 256));
                    if (!c->writer->publish(m, pixels.data()))
                        ++c->dropped;
                }
            }
            for (size_t i = 0; i < cameras.size(); ++i) {
                auto &c = cameras[i];
                if (c->error ||
                    (now - begin > 2 && (!c->last_ns || mono_ns() - c->last_ns > 1000000000ULL))) {
                    std::fprintf(stderr, "Camera %s failed: %s\n", c->serial.c_str(),
                                 c->error == 1   ? "required timestamp metadata missing"
                                 : c->error == 2 ? "capture callback failure"
                                                 : "no fresh frame for one second");
                    failed = true;
                    stopped = 1;
                    break;
                }
                if (csv.is_open()) {
                    FrameMetadata m{};
                    if (kaivuri_frame_read(readers[i], &m, nullptr, 0) == 1 &&
                        m.sequence != logged[i]) {
                        logged[i] = m.sequence;
                        csv.precision(17);
                        csv << c->serial << ',' << m.ir_index << ',' << m.sequence << ','
                            << m.host_monotonic_ns << ',' << m.sdk_timestamp_ms << ','
                            << m.timestamp_domain << ',' << m.sensor_timestamp_us << ','
                            << m.frame_timestamp_us << ',' << m.frame_counter << ','
                            << m.actual_exposure_us << ',' << m.gain_level << ','
                            << m.valid_metadata << '\n';
                        if (!csv)
                            throw std::runtime_error("metadata CSV write failed");
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        for (auto &c : cameras)
            std::fprintf(
                stderr,
                "serial=%s received=%llu busy_drops=%llu counter_gaps=%llu counter_resets=%llu\n",
                c->serial.c_str(), (unsigned long long)c->received.load(),
                (unsigned long long)c->dropped.load(), (unsigned long long)c->counter_gaps.load(),
                (unsigned long long)c->counter_resets.load());
        return failed ? 1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Camera failed: %s\n", e.what());
        return 1;
    }
}
