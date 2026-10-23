#include "kaivuri/frame_store.hpp"
#include <algorithm>
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
#include <zstd.h>
using namespace kaivuri;
static double monotonic_seconds() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static volatile sig_atomic_t stopped = 0;
static void stop_signal(int) {
    stopped = 1;
}
struct Camera {
    std::string serial, path;
    int index = 1, sync = 0;
    rs2::sensor sensor;
    std::unique_ptr<FrameWriter> writer;
    std::atomic<uint64_t> received{0}, dropped{0}, last_ns{0};
    std::atomic<uint64_t> last_device_counter{0}, counter_gaps{0}, counter_resets{0};
    std::atomic<int> error{0};
    bool started = false, opened = false, rotate_180 = false;
    std::vector<uint8_t> rotated;
    // Optional Z16 depth from the same stereo sensor: computed on the camera, aligned to IR1.
    bool depth = false;
    std::string depth_path;
    std::unique_ptr<FrameWriter> depth_writer;
    std::atomic<uint64_t> depth_received{0}, depth_dropped{0}, depth_last_ns{0};
    std::atomic<uint64_t> depth_last_counter{0}, depth_counter_gaps{0};
    std::vector<uint16_t> depth_rotated;
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
static uint64_t mono_ns() {
    return static_cast<uint64_t>(monotonic_seconds() * 1e9);
}
// Shared by IR and depth. Returns false when the SDK timestamp is invalid.
static bool read_metadata(const rs2::frame &frame, const rs2::video_frame &v, FrameMetadata &m) {
    m.sdk_timestamp_ms = frame.get_timestamp();
    if (!std::isfinite(m.sdk_timestamp_ms) || m.sdk_timestamp_ms < 0)
        return false;
    m.timestamp_domain = frame.get_frame_timestamp_domain();
    m.width = v.get_width();
    m.height = v.get_height();
    m.stride = v.get_stride_in_bytes();
    m.pixel_bytes = m.stride * m.height;
    const rs2_frame_metadata_value keys[]{
        RS2_FRAME_METADATA_SENSOR_TIMESTAMP, RS2_FRAME_METADATA_FRAME_TIMESTAMP,
        RS2_FRAME_METADATA_FRAME_COUNTER, RS2_FRAME_METADATA_ACTUAL_EXPOSURE,
        RS2_FRAME_METADATA_GAIN_LEVEL};
    uint64_t *destinations[]{&m.sensor_timestamp_us, &m.frame_timestamp_us, &m.frame_counter,
                             &m.actual_exposure_us, &m.gain_level};
    for (int i = 0; i < 5; ++i)
        if (frame.supports_frame_metadata(keys[i])) {
            *destinations[i] = static_cast<uint64_t>(frame.get_frame_metadata(keys[i]));
            m.valid_metadata |= 1ULL << i;
        }
    return true;
}
// Pair depth with IR by sensor_timestamp_us: both streams of one exposure share it exactly,
// while their frame counters differ by one (+1 against IR1, -1 against IR2, firmware 5.17).
static void publish_depth(Camera *c, const rs2::frame &frame, const rs2::video_frame &v,
                          bool require_metadata) {
    FrameMetadata m{};
    m.host_monotonic_ns = mono_ns();
    if (!read_metadata(frame, v, m)) {
        c->error = 2;
        return;
    }
    m.sequence = ++c->depth_received;
    c->depth_last_ns = m.host_monotonic_ns;
    m.ir_index = 0;
    if (m.valid_metadata & 4) {
        const auto previous = c->depth_last_counter.exchange(m.frame_counter);
        if (previous && m.frame_counter > previous + 1)
            c->depth_counter_gaps += m.frame_counter - previous - 1;
    }
    if (require_metadata && (m.valid_metadata & 3) != 3) {
        c->error = 1;
        return;
    }
    const void *pixels = v.get_data();
    if (c->rotate_180) {
        const size_t n = size_t(m.width) * m.height;
        if (m.stride != m.width * 2 || n != c->depth_rotated.size()) {
            c->error = 2;
            return;
        }
        const auto *source = static_cast<const uint16_t *>(pixels);
        std::reverse_copy(source, source + n, c->depth_rotated.begin());
        pixels = c->depth_rotated.data();
    }
    if (!c->depth_writer->publish(m, pixels))
        ++c->depth_dropped;
}
static void write_json(const std::string &path, const std::string &text) {
    std::ofstream f(path + ".new");
    f << text;
    f.close();
    if (!f)
        throw std::runtime_error("cannot write " + path);
    std::filesystem::rename(path + ".new", path);
}
static std::string intrinsics_json(const rs2_intrinsics &i) {
    char b[512];
    std::snprintf(b, sizeof b,
                  "{\"width\":%d,\"height\":%d,\"fx\":%.9g,\"fy\":%.9g,\"ppx\":%.9g,"
                  "\"ppy\":%.9g,\"model\":\"%s\",\"coeffs\":[%.9g,%.9g,%.9g,%.9g,%.9g]}",
                  i.width, i.height, i.fx, i.fy, i.ppx, i.ppy, rs2_distortion_to_string(i.model),
                  i.coeffs[0], i.coeffs[1], i.coeffs[2], i.coeffs[3], i.coeffs[4]);
    return b;
}
int main(int argc, char **argv) {
    try {
        int width = 640, height = 480, fps = 30;
        double duration = 0, exposure = -1, gain = -1, depth_log_hz = 2;
        bool list = false, synthetic = false, global_time = true, require_metadata = true;
        std::string output = "/tmp/kaivuri-native-" + std::to_string(geteuid()) + "/cameras",
                    csv_path, depth_log;
        std::vector<std::string> specs, rotate_serials, depth_serials;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto value = [&]() {
                if (++i == argc)
                    throw std::runtime_error("missing option argument");
                return std::string(argv[i]);
            };
            if (a == "--camera")
                specs.push_back(value());
            else if (a == "--rotate-180")
                rotate_serials.push_back(value());
            else if (a == "--depth")
                depth_serials.push_back(value());
            else if (a == "--depth-log")
                depth_log = value();
            else if (a == "--depth-log-hz")
                depth_log_hz = std::stod(value());
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
                    "[--output-dir DIR] [--metadata-csv FILE]\n  [--rotate-180 SERIAL] [--depth "
                    "SERIAL] [--depth-log DIR --depth-log-hz 2]\n  [--synthetic] (hardware-free "
                    "test source; never a physical camera)\nIR index 1 or 2. Sync mode 0 "
                    "standalone, 1 master, 2 slave; requires a sync cable.\n--depth adds same-size "
                    "Z16 from the camera's stereo ASIC to SERIAL-depth.frame (+ SERIAL-depth.json\n"
                    "calibration); --depth-log appends zstd snapshots at the given rate. Emitter "
                    "disabled;\nno RGB, host conversion, encoding, or CUDA context.");
                return 0;
            } else
                throw std::runtime_error("unknown camera option: " + a);
        }
        if (width < 1 || height < 1 || uint64_t(width) * height > 16 * 1024 * 1024 || fps < 1 ||
            fps > 90 || duration < 0 || !std::isfinite(duration) || !std::isfinite(exposure) ||
            !std::isfinite(gain) || !std::isfinite(depth_log_hz) || depth_log_hz <= 0 ||
            (!depth_log.empty() && depth_serials.empty()))
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
            camera->rotate_180 = std::find(rotate_serials.begin(), rotate_serials.end(),
                                           camera->serial) != rotate_serials.end();
            if (camera->rotate_180)
                camera->rotated.resize(size_t(width) * height);
            camera->path =
                output + "/" + camera->serial + "-ir" + std::to_string(camera->index) + ".frame";
            camera->writer =
                std::make_unique<FrameWriter>(camera->path, width, height, camera->index);
            camera->depth = std::find(depth_serials.begin(), depth_serials.end(), camera->serial) !=
                            depth_serials.end();
            std::string calibration;
            if (camera->depth) {
                camera->depth_path = output + "/" + camera->serial + "-depth.frame";
                if (camera->rotate_180)
                    camera->depth_rotated.resize(size_t(width) * height);
                if (synthetic)
                    calibration = "{\"synthetic\":true,\"depth_scale_m\":0.001}\n";
            }
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
                rs2::stream_profile depth_profile;
                if (camera->depth) {
                    for (auto p : camera->sensor.get_stream_profiles()) {
                        auto video = p.as<rs2::video_stream_profile>();
                        if (video && p.stream_type() == RS2_STREAM_DEPTH &&
                            p.format() == RS2_FORMAT_Z16 && video.width() == width &&
                            video.height() == height && p.fps() == fps) {
                            depth_profile = p;
                            break;
                        }
                    }
                    // IR and depth on one stereo sensor must share resolution and rate.
                    if (!depth_profile)
                        throw std::runtime_error("matching Z16 depth profile unavailable: " + spec);
                    const auto e = depth_profile.get_extrinsics_to(profile);
                    char extrinsics[512];
                    std::snprintf(extrinsics, sizeof extrinsics,
                                  "{\"rotation_column_major\":[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                                  "%.9g,%.9g],\"translation_m\":[%.9g,%.9g,%.9g]}",
                                  e.rotation[0], e.rotation[1], e.rotation[2], e.rotation[3],
                                  e.rotation[4], e.rotation[5], e.rotation[6], e.rotation[7],
                                  e.rotation[8], e.translation[0], e.translation[1],
                                  e.translation[2]);
                    char scale[64];
                    std::snprintf(scale, sizeof scale, "%.9g",
                                  camera->sensor.as<rs2::depth_sensor>().get_depth_scale());
                    calibration =
                        std::string("{\"synthetic\":false,\"serial\":\"") + camera->serial +
                        "\",\"depth_scale_m\":" + scale +
                        ",\"ir_index\":" + std::to_string(camera->index) +
                        ",\"rotation_deg\":" + (camera->rotate_180 ? "180" : "0") +
                        ",\"note\":\"intrinsics/extrinsics are SDK values for unrotated pixels\""
                        ",\"depth_intrinsics\":" +
                        intrinsics_json(
                            depth_profile.as<rs2::video_stream_profile>().get_intrinsics()) +
                        ",\"ir_intrinsics\":" +
                        intrinsics_json(profile.as<rs2::video_stream_profile>().get_intrinsics()) +
                        ",\"depth_to_ir_extrinsics\":" + extrinsics + "}\n";
                }
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
                if (camera->depth)
                    sensor.open(std::vector<rs2::stream_profile>{profile, depth_profile});
                else
                    sensor.open(profile);
                camera->opened = true;
                // Calibration and the depth store exist before the first callback can publish.
                if (camera->depth) {
                    write_json(output + "/" + camera->serial + "-depth.json", calibration);
                    camera->depth_writer =
                        std::make_unique<FrameWriter>(camera->depth_path, width, height, 0, 2);
                }
                Camera *c = camera.get();
                sensor.start([c, require_metadata](rs2::frame frame) {
                    try {
                        auto v = frame.as<rs2::video_frame>();
                        if (!v)
                            return;
                        if (c->depth && frame.get_profile().stream_type() == RS2_STREAM_DEPTH) {
                            publish_depth(c, frame, v, require_metadata);
                            return;
                        }
                        FrameMetadata m{};
                        m.sequence = ++c->received;
                        m.host_monotonic_ns = mono_ns();
                        c->last_ns = m.host_monotonic_ns;
                        if (!read_metadata(frame, v, m)) {
                            c->error = 2;
                            return;
                        }
                        m.ir_index = c->index;
                        m.pixel_bytes = m.width * m.height;
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
                        const void *pixels = v.get_data();
                        if (c->rotate_180) {
                            if (m.stride != m.width || m.pixel_bytes != c->rotated.size()) {
                                c->error = 2;
                                return;
                            }
                            const auto *source = static_cast<const uint8_t *>(pixels);
                            std::reverse_copy(source, source + m.pixel_bytes, c->rotated.begin());
                            pixels = c->rotated.data();
                        }
                        if (!c->writer->publish(m, pixels))
                            ++c->dropped;
                    } catch (...) {
                        c->error = 2;
                    }
                });
                camera->started = true;
            } else if (camera->depth) {
                write_json(output + "/" + camera->serial + "-depth.json", calibration);
                camera->depth_writer =
                    std::make_unique<FrameWriter>(camera->depth_path, width, height, 0, 2);
            }
            std::fprintf(stderr, "%sIR%d serial=%s -> %s (Y8 %dx%d@%d, sync=%d, rotation=%d)\n",
                         synthetic ? "SYNTHETIC " : "", camera->index, camera->serial.c_str(),
                         camera->path.c_str(), width, height, fps, camera->sync,
                         camera->rotate_180 ? 180 : 0);
            if (camera->depth)
                std::fprintf(stderr, "%sdepth serial=%s -> %s (Z16 %dx%d@%d)\n",
                             synthetic ? "SYNTHETIC " : "", camera->serial.c_str(),
                             camera->depth_path.c_str(), width, height, fps);
            cameras.push_back(std::move(camera));
        }
        for (const auto &serial : rotate_serials)
            if (std::none_of(cameras.begin(), cameras.end(),
                             [&](const auto &c) { return c->serial == serial; }))
                throw std::runtime_error("rotation serial is not a selected camera: " + serial);
        for (const auto &serial : depth_serials)
            if (std::none_of(cameras.begin(), cameras.end(),
                             [&](const auto &c) { return c->serial == serial; }))
                throw std::runtime_error("depth serial is not a selected camera: " + serial);
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
        // Low-rate depth snapshots (e.g. 2 Hz during policy runs), read back from the store.
        struct DepthLog {
            Camera *camera;
            void *reader;
            std::ofstream data;
            uint64_t offset = 0, last = 0;
        };
        std::vector<std::unique_ptr<DepthLog>> depth_logs;
        struct DepthLogCleanup {
            std::vector<std::unique_ptr<DepthLog>> &logs;
            ~DepthLogCleanup() {
                for (auto &l : logs)
                    kaivuri_frame_close(l->reader);
            }
        } depth_log_cleanup{depth_logs};
        std::ofstream depth_csv;
        std::vector<uint8_t> depth_pixels(size_t(width) * height * 2), depth_packed;
        struct Compressor {
            ZSTD_CCtx *p = ZSTD_createCCtx();
            ~Compressor() { ZSTD_freeCCtx(p); }
        } zstd;
        double next_depth_log = monotonic_seconds();
        if (!depth_log.empty()) {
            std::filesystem::create_directories(depth_log);
            depth_csv.open(depth_log + "/depth-log.csv");
            depth_csv << "serial,sequence,host_monotonic_ns,sdk_timestamp_ms,sensor_timestamp_us,"
                         "frame_counter,valid_fraction,zstd_offset,zstd_bytes\n";
            depth_packed.resize(ZSTD_compressBound(depth_pixels.size()));
            for (auto &c : cameras)
                if (c->depth) {
                    auto log = std::make_unique<DepthLog>();
                    log->camera = c.get();
                    log->reader = kaivuri_frame_open(c->depth_path.c_str());
                    if (!log->reader)
                        throw std::runtime_error("open depth store reader");
                    depth_logs.push_back(std::move(log));
                    auto &l = *depth_logs.back();
                    l.data.open(depth_log + "/" + c->serial + "-depth.zst", std::ios::binary);
                    std::filesystem::copy_file(output + "/" + c->serial + "-depth.json",
                                               depth_log + "/" + c->serial + "-depth.json",
                                               std::filesystem::copy_options::overwrite_existing);
                    if (!depth_csv || !l.data)
                        throw std::runtime_error("cannot open depth log");
                }
        }
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
                    const uint8_t *image = pixels.data();
                    if (c->rotate_180) {
                        std::reverse_copy(pixels.begin(), pixels.end(), c->rotated.begin());
                        image = c->rotated.data();
                    }
                    if (!c->writer->publish(m, image))
                        ++c->dropped;
                    if (c->depth) {
                        // Same exposure as the IR frame: identical sensor timestamp. A horizontal
                        // ramp makes 180 degree rotation observable.
                        std::vector<uint16_t> z(pixels.size());
                        for (size_t k = 0; k < z.size(); ++k)
                            z[k] = uint16_t(400 + m.sequence % 600 + k % width);
                        if (c->rotate_180)
                            std::reverse(z.begin(), z.end());
                        FrameMetadata d = m;
                        d.sequence = ++c->depth_received;
                        d.ir_index = 0;
                        d.stride = width * 2;
                        d.pixel_bytes = z.size() * 2;
                        c->depth_last_ns = d.host_monotonic_ns;
                        if (!c->depth_writer->publish(d, z.data()))
                            ++c->depth_dropped;
                    }
                }
            }
            for (size_t i = 0; i < cameras.size(); ++i) {
                auto &c = cameras[i];
                const bool depth_stale =
                    c->depth && now - begin > 2 &&
                    (!c->depth_last_ns || mono_ns() - c->depth_last_ns > 1000000000ULL);
                if (c->error || depth_stale ||
                    (now - begin > 2 && (!c->last_ns || mono_ns() - c->last_ns > 1000000000ULL))) {
                    std::fprintf(stderr, "Camera %s failed: %s\n", c->serial.c_str(),
                                 c->error == 1   ? "required timestamp metadata missing"
                                 : c->error == 2 ? "capture callback failure"
                                 : depth_stale   ? "no fresh depth frame for one second"
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
            if (!depth_logs.empty() && now >= next_depth_log) {
                next_depth_log += 1. / depth_log_hz;
                if (next_depth_log < now)
                    next_depth_log = now + 1. / depth_log_hz;
                for (auto &log : depth_logs) {
                    FrameMetadata m{};
                    if (kaivuri_frame_read(log->reader, &m, depth_pixels.data(),
                                           depth_pixels.size()) != 1 ||
                        m.sequence == log->last)
                        continue;
                    log->last = m.sequence;
                    const auto *z = reinterpret_cast<const uint16_t *>(depth_pixels.data());
                    const size_t n = m.pixel_bytes / 2;
                    size_t valid = 0;
                    for (size_t k = 0; k < n; ++k)
                        valid += z[k] != 0;
                    const size_t bytes =
                        ZSTD_compressCCtx(zstd.p, depth_packed.data(), depth_packed.size(),
                                          depth_pixels.data(), m.pixel_bytes, 1);
                    if (ZSTD_isError(bytes))
                        throw std::runtime_error("depth zstd compression failed");
                    log->data.write(reinterpret_cast<const char *>(depth_packed.data()), bytes);
                    log->data.flush();
                    depth_csv.precision(17);
                    depth_csv << log->camera->serial << ',' << m.sequence << ','
                              << m.host_monotonic_ns << ',' << m.sdk_timestamp_ms << ','
                              << m.sensor_timestamp_us << ',' << m.frame_counter << ','
                              << double(valid) / std::max<size_t>(1, n) << ',' << log->offset << ','
                              << bytes << '\n';
                    depth_csv.flush();
                    log->offset += bytes;
                    if (!log->data || !depth_csv)
                        throw std::runtime_error("depth log write failed");
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
        for (auto &c : cameras)
            if (c->depth)
                std::fprintf(stderr,
                             "depth_serial=%s depth_received=%llu depth_busy_drops=%llu "
                             "depth_counter_gaps=%llu\n",
                             c->serial.c_str(), (unsigned long long)c->depth_received.load(),
                             (unsigned long long)c->depth_dropped.load(),
                             (unsigned long long)c->depth_counter_gaps.load());
        return failed ? 1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Camera failed: %s\n", e.what());
        return 1;
    }
}
