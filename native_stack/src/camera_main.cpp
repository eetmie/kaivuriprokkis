// RealSense IR (and optional same-sensor Z16 depth) capture into the shared frame stores.
#include "kaivuri/frame_store.hpp"
#include "kaivuri/util.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <librealsense2/rs.hpp>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <zstd.h>
using namespace kaivuri;

namespace {
double monotonic_seconds() {
    return monotonic_ns() * 1e-9;
}
__attribute__((format(printf, 1, 2))) std::string format(const char *pattern, ...) {
    char text[512];
    va_list values;
    va_start(values, pattern);
    std::vsnprintf(text, sizeof text, pattern, values);
    va_end(values);
    return text;
}

// Read handle on a frame store; closes on destruction.
using Reader = std::unique_ptr<void, void (*)(void *)>;
Reader open_reader(const std::string &path, const char *what) {
    Reader reader(kaivuri_frame_open(path.c_str()), kaivuri_frame_close);
    require(bool(reader), what);
    return reader;
}

// One published stream (IR Y8 or depth Z16) and its counters, shared with the SDK callback.
struct Stream {
    std::string path;
    uint32_t bytes_per_pixel = 1;
    std::unique_ptr<FrameWriter> writer;
    std::atomic<uint64_t> received{0}, dropped{0}, last_ns{0};
    std::atomic<uint64_t> last_counter{0}, counter_gaps{0}, counter_resets{0};
    std::vector<uint16_t> rotated; // width*height pixels when rotating 180 degrees, else empty
};

struct Camera {
    std::string serial;
    int index = 1, sync = 0;
    rs2::sensor sensor;
    Stream ir;
    // Optional Z16 depth from the same stereo sensor: computed on the camera, aligned to IR1.
    bool with_depth = false;
    Stream depth;
    std::atomic<int> error{0}; // 1 = required timestamp metadata missing, 2 = callback failure
    bool started = false, opened = false, rotate_180 = false;
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

// Publishes one packed frame, rotated 180 degrees when configured. False on a layout the
// rotation cannot handle.
bool publish(Stream &s, const FrameMetadata &m, const void *pixels) {
    if (!s.rotated.empty()) {
        const size_t n = size_t(m.width) * m.height;
        if (m.stride != m.width * s.bytes_per_pixel || n != s.rotated.size())
            return false;
        if (s.bytes_per_pixel == 1) {
            const auto *source = static_cast<const uint8_t *>(pixels);
            std::reverse_copy(source, source + n, reinterpret_cast<uint8_t *>(s.rotated.data()));
        } else {
            const auto *source = static_cast<const uint16_t *>(pixels);
            std::reverse_copy(source, source + n, s.rotated.data());
        }
        pixels = s.rotated.data();
    }
    if (!s.writer->publish(m, pixels))
        ++s.dropped;
    return true;
}

// Returns false when the SDK timestamp is invalid.
bool read_metadata(const rs2::frame &frame, const rs2::video_frame &v, FrameMetadata &m) {
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

// SDK callback body for IR (ir_index 1/2) and depth (ir_index 0). Depth pairs with IR by
// sensor_timestamp_us: both streams of one exposure share it exactly, while their frame
// counters differ by one (+1 against IR1, -1 against IR2, firmware 5.17).
void on_frame(Camera &c, Stream &s, const rs2::frame &frame, const rs2::video_frame &v,
              uint32_t ir_index, bool require_metadata) {
    FrameMetadata m{};
    m.sequence = ++s.received;
    m.host_monotonic_ns = monotonic_ns();
    s.last_ns = m.host_monotonic_ns;
    if (!read_metadata(frame, v, m)) {
        c.error = 2;
        return;
    }
    m.ir_index = ir_index;
    if (ir_index)
        m.pixel_bytes = m.width * m.height; // IR is published packed
    if (m.valid_metadata & 4) {
        const auto previous = s.last_counter.exchange(m.frame_counter);
        if (previous && m.frame_counter > previous + 1)
            s.counter_gaps += m.frame_counter - previous - 1;
        else if (previous && m.frame_counter <= previous)
            ++s.counter_resets;
    }
    if (require_metadata && (m.valid_metadata & 3) != 3)
        c.error = 1;
    else if (!publish(s, m, v.get_data()))
        c.error = 2;
}

void write_json(const std::string &path, const std::string &text) {
    std::ofstream f(path + ".new");
    f << text;
    f.close();
    require(bool(f), "cannot write " + path);
    std::filesystem::rename(path + ".new", path);
}

std::string intrinsics_json(const rs2_intrinsics &i) {
    return format("{\"width\":%d,\"height\":%d,\"fx\":%.9g,\"fy\":%.9g,\"ppx\":%.9g,"
                  "\"ppy\":%.9g,\"model\":\"%s\",\"coeffs\":[%.9g,%.9g,%.9g,%.9g,%.9g]}",
                  i.width, i.height, i.fx, i.fy, i.ppx, i.ppy, rs2_distortion_to_string(i.model),
                  i.coeffs[0], i.coeffs[1], i.coeffs[2], i.coeffs[3], i.coeffs[4]);
}

bool contains(const std::vector<std::string> &list, const std::string &item) {
    return std::find(list.begin(), list.end(), item) != list.end();
}

const char *usage = "kaivuri-camera --list\nkaivuri-camera --camera SERIAL:IR_INDEX[:SYNC_MODE] "
                    "[--camera SECOND:1]\n  [--width 640 --height 480 --fps 30] [--duration-s SEC] "
                    "[--exposure-us US --gain N]\n  [--hardware-clock] [--allow-missing-metadata] "
                    "[--output-dir DIR] [--metadata-csv FILE]\n  [--rotate-180 SERIAL] [--depth "
                    "SERIAL] [--depth-log DIR --depth-log-hz 2]\n  [--synthetic] (hardware-free "
                    "test source; never a physical camera)\nIR index 1 or 2. Sync mode 0 "
                    "standalone, 1 master, 2 slave; requires a sync cable.\n--depth adds same-size "
                    "Z16 from the camera's stereo ASIC to SERIAL-depth.frame (+ SERIAL-depth.json\n"
                    "calibration); --depth-log appends zstd snapshots at the given rate. Emitter "
                    "disabled;\nno RGB, host conversion, encoding, or CUDA context.";
} // namespace

int main(int argc, char **argv) {
    try {
        int width = 640, height = 480, fps = 30;
        double duration = 0, exposure = -1, gain = -1, depth_log_hz = 2;
        bool list = false, synthetic = false, global_time = true, require_metadata = true;
        std::string output = (runtime_dir() / "cameras").string(), csv_path, depth_log;
        std::vector<std::string> specs, rotate_serials, depth_serials;
        for (Args args(argc, argv); args.next();) {
            if (args.is("--camera"))
                specs.push_back(args.value());
            else if (args.is("--rotate-180"))
                rotate_serials.push_back(args.value());
            else if (args.is("--depth"))
                depth_serials.push_back(args.value());
            else if (args.is("--depth-log"))
                depth_log = args.value();
            else if (args.is("--depth-log-hz"))
                depth_log_hz = args.number();
            else if (args.is("--width"))
                width = args.integer();
            else if (args.is("--height"))
                height = args.integer();
            else if (args.is("--fps"))
                fps = args.integer();
            else if (args.is("--duration-s"))
                duration = args.number();
            else if (args.is("--exposure-us"))
                exposure = args.number();
            else if (args.is("--gain"))
                gain = args.number();
            else if (args.is("--output-dir"))
                output = args.value();
            else if (args.is("--metadata-csv"))
                csv_path = args.value();
            else if (args.is("--list"))
                list = true;
            else if (args.is("--synthetic"))
                synthetic = true;
            else if (args.is("--hardware-clock"))
                global_time = false;
            else if (args.is("--allow-missing-metadata"))
                require_metadata = false;
            else if (args.is("--help")) {
                std::puts(usage);
                return 0;
            } else
                args.unknown();
        }
        require(width >= 1 && height >= 1 && uint64_t(width) * height <= 16 * 1024 * 1024 &&
                    fps >= 1 && fps <= 90 && duration >= 0 && std::isfinite(duration) &&
                    std::isfinite(exposure) && std::isfinite(gain) && std::isfinite(depth_log_hz) &&
                    depth_log_hz > 0 && (depth_log.empty() || !depth_serials.empty()),
                "invalid camera options");
        // Synthetic mode constructs no context: no USB enumeration or SDK worker threads.
        // Otherwise keep one enumeration alive for both cameras: re-enumerating after starting
        // the first RSUSB stream can reopen its claimed USB interfaces.
        std::unique_ptr<rs2::context> context;
        std::vector<rs2::device> devices;
        if (!synthetic) {
            context = std::make_unique<rs2::context>();
            for (auto device : context->query_devices())
                devices.push_back(device);
        }
        if (list) {
            require(!synthetic, "--list requires real devices");
            auto info = [](const rs2::device &d, rs2_camera_info key) {
                return d.supports(key) ? d.get_info(key) : "unknown";
            };
            for (const auto &d : devices)
                std::printf("%s  %s  firmware=%s usb=%s port=%s\n",
                            d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER),
                            d.get_info(RS2_CAMERA_INFO_NAME),
                            d.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION),
                            info(d, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR),
                            info(d, RS2_CAMERA_INFO_PHYSICAL_PORT));
            return 0;
        }
        require(!specs.empty() && specs.size() <= 2,
                "select one or two cameras explicitly with --camera SERIAL:IR_INDEX");
        std::filesystem::create_directories(output);
        std::filesystem::permissions(output, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace);
        const size_t frame_pixels = size_t(width) * height;
        // First matching video profile of a sensor at the requested size and rate (index -1: any).
        auto find_profile = [&](const rs2::sensor &sensor, rs2_stream type, int index,
                                rs2_format pixel_format) {
            for (auto p : sensor.get_stream_profiles()) {
                auto video = p.as<rs2::video_stream_profile>();
                if (video && p.stream_type() == type && (index < 0 || p.stream_index() == index) &&
                    p.format() == pixel_format && video.width() == width &&
                    video.height() == height && p.fps() == fps)
                    return p;
            }
            return rs2::stream_profile();
        };
        std::vector<std::unique_ptr<Camera>> cameras;
        for (const auto &spec : specs) {
            auto camera = std::make_unique<Camera>();
            auto first = spec.find(':');
            require(first != std::string::npos, "camera syntax SERIAL:IR_INDEX[:SYNC_MODE]");
            auto second = spec.find(':', first + 1);
            camera->serial = spec.substr(0, first);
            camera->index = std::stoi(spec.substr(first + 1, second - first - 1));
            if (second != std::string::npos)
                camera->sync = std::stoi(spec.substr(second + 1));
            require(!camera->serial.empty() &&
                        camera->serial.find_first_not_of("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                         "abcdefghijklmnopqrstuvwxyz-_") ==
                            std::string::npos &&
                        camera->index >= 1 && camera->index <= 2 && camera->sync >= 0 &&
                        camera->sync <= 2,
                    "invalid camera spec");
            for (const auto &previous : cameras)
                require(previous->serial != camera->serial,
                        "select one IR stream per physical camera");
            const std::string base = output + "/" + camera->serial;
            camera->rotate_180 = contains(rotate_serials, camera->serial);
            camera->with_depth = contains(depth_serials, camera->serial);
            camera->ir.path = base + "-ir" + std::to_string(camera->index) + ".frame";
            camera->ir.writer =
                std::make_unique<FrameWriter>(camera->ir.path, width, height, camera->index);
            camera->depth.path = base + "-depth.frame";
            camera->depth.bytes_per_pixel = 2;
            if (camera->rotate_180) {
                camera->ir.rotated.resize(frame_pixels);
                if (camera->with_depth)
                    camera->depth.rotated.resize(frame_pixels);
            }
            std::string calibration = "{\"synthetic\":true,\"depth_scale_m\":0.001}\n";
            if (!synthetic) {
                auto selected = std::find_if(devices.begin(), devices.end(), [&](const auto &d) {
                    return camera->serial == d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
                });
                require(selected != devices.end(), "camera serial not found: " + camera->serial);
                // The last sensor offering the profile wins (in practice there is only one).
                rs2::stream_profile profile, depth_profile;
                for (auto sensor : selected->query_sensors())
                    if (auto p = find_profile(sensor, RS2_STREAM_INFRARED, camera->index,
                                              RS2_FORMAT_Y8)) {
                        camera->sensor = sensor;
                        profile = p;
                    }
                require(bool(profile), "requested Y8 IR profile unavailable: " + spec);
                auto &sensor = camera->sensor;
                if (camera->with_depth) {
                    // IR and depth on one stereo sensor must share resolution and rate.
                    depth_profile = find_profile(sensor, RS2_STREAM_DEPTH, -1, RS2_FORMAT_Z16);
                    require(bool(depth_profile), "matching Z16 depth profile unavailable: " + spec);
                    const auto e = depth_profile.get_extrinsics_to(profile);
                    calibration =
                        "{\"synthetic\":false,\"serial\":\"" + camera->serial +
                        "\",\"depth_scale_m\":" +
                        format("%.9g", sensor.as<rs2::depth_sensor>().get_depth_scale()) +
                        ",\"ir_index\":" + std::to_string(camera->index) +
                        ",\"rotation_deg\":" + (camera->rotate_180 ? "180" : "0") +
                        ",\"note\":\"intrinsics/extrinsics are SDK values for unrotated pixels\""
                        ",\"depth_intrinsics\":" +
                        intrinsics_json(
                            depth_profile.as<rs2::video_stream_profile>().get_intrinsics()) +
                        ",\"ir_intrinsics\":" +
                        intrinsics_json(profile.as<rs2::video_stream_profile>().get_intrinsics()) +
                        ",\"depth_to_ir_extrinsics\":" +
                        format("{\"rotation_column_major\":[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                               "%.9g,%.9g],\"translation_m\":[%.9g,%.9g,%.9g]}",
                               e.rotation[0], e.rotation[1], e.rotation[2], e.rotation[3],
                               e.rotation[4], e.rotation[5], e.rotation[6], e.rotation[7],
                               e.rotation[8], e.translation[0], e.translation[1],
                               e.translation[2]) +
                        "}\n";
                }
                if (sensor.supports(RS2_OPTION_EMITTER_ENABLED))
                    sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0);
                if (sensor.supports(RS2_OPTION_FRAMES_QUEUE_SIZE))
                    sensor.set_option(RS2_OPTION_FRAMES_QUEUE_SIZE, 1);
                if (sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED))
                    sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, global_time ? 1 : 0);
                else
                    require(!global_time, "global time option unavailable; use --hardware-clock");
                if (sensor.supports(RS2_OPTION_INTER_CAM_SYNC_MODE))
                    sensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, camera->sync);
                else
                    require(!camera->sync, "camera sync unsupported");
                if (exposure >= 0) {
                    require(sensor.supports(RS2_OPTION_EXPOSURE) &&
                                sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE),
                            "manual exposure unsupported");
                    sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
                    sensor.set_option(RS2_OPTION_EXPOSURE, exposure);
                }
                if (gain >= 0) {
                    require(sensor.supports(RS2_OPTION_GAIN), "gain unsupported");
                    sensor.set_option(RS2_OPTION_GAIN, gain);
                }
                if (camera->with_depth)
                    sensor.open(std::vector<rs2::stream_profile>{profile, depth_profile});
                else
                    sensor.open(profile);
                camera->opened = true;
            }
            // Calibration and the depth store exist before the first callback can publish.
            if (camera->with_depth) {
                write_json(base + "-depth.json", calibration);
                camera->depth.writer =
                    std::make_unique<FrameWriter>(camera->depth.path, width, height, 0, 2);
            }
            if (!synthetic) {
                Camera *c = camera.get();
                c->sensor.start([c, require_metadata](rs2::frame frame) {
                    try {
                        auto v = frame.as<rs2::video_frame>();
                        if (!v)
                            return;
                        if (c->with_depth && frame.get_profile().stream_type() == RS2_STREAM_DEPTH)
                            on_frame(*c, c->depth, frame, v, 0, require_metadata);
                        else
                            on_frame(*c, c->ir, frame, v, c->index, require_metadata);
                    } catch (...) {
                        c->error = 2;
                    }
                });
                camera->started = true;
            }
            const char *source = synthetic ? "SYNTHETIC " : "";
            std::fprintf(stderr, "%sIR%d serial=%s -> %s (Y8 %dx%d@%d, sync=%d, rotation=%d)\n",
                         source, camera->index, camera->serial.c_str(), camera->ir.path.c_str(),
                         width, height, fps, camera->sync, camera->rotate_180 ? 180 : 0);
            if (camera->with_depth)
                std::fprintf(stderr, "%sdepth serial=%s -> %s (Z16 %dx%d@%d)\n", source,
                             camera->serial.c_str(), camera->depth.path.c_str(), width, height,
                             fps);
            cameras.push_back(std::move(camera));
        }
        auto selected = [&](const std::string &serial) {
            return std::any_of(cameras.begin(), cameras.end(),
                               [&](const auto &c) { return c->serial == serial; });
        };
        for (const auto &serial : rotate_serials)
            require(selected(serial), "rotation serial is not a selected camera: " + serial);
        for (const auto &serial : depth_serials)
            require(selected(serial), "depth serial is not a selected camera: " + serial);
        install_stop_handlers();
        std::ofstream csv;
        if (!csv_path.empty()) {
            csv.open(csv_path);
            require(bool(csv), "cannot open metadata CSV");
            csv.precision(17);
            csv << "serial,ir_index,sequence,host_monotonic_ns,sdk_timestamp_ms,domain,sensor_"
                   "timestamp_us,frame_timestamp_us,frame_counter,exposure_us,gain,valid_bits\n";
        }
        std::vector<Reader> readers;
        readers.reserve(cameras.size());
        for (auto &c : cameras)
            readers.push_back(open_reader(c->ir.path, "open frame store reader"));
        std::vector<uint64_t> logged(cameras.size(), 0);
        // Low-rate depth snapshots (e.g. 2 Hz during policy runs), read back from the store.
        struct DepthLog {
            Camera *camera;
            Reader reader;
            std::ofstream data;
            uint64_t offset = 0, last = 0;
        };
        std::vector<DepthLog> depth_logs;
        std::ofstream depth_csv;
        std::vector<uint8_t> depth_pixels(frame_pixels * 2), depth_packed;
        std::unique_ptr<ZSTD_CCtx, size_t (*)(ZSTD_CCtx *)> zstd(nullptr, ZSTD_freeCCtx);
        double next_depth_log = monotonic_seconds();
        if (!depth_log.empty()) {
            zstd.reset(ZSTD_createCCtx());
            require(bool(zstd), "zstd context allocation failed");
            std::filesystem::create_directories(depth_log);
            depth_csv.open(depth_log + "/depth-log.csv");
            depth_csv.precision(17);
            depth_csv << "serial,sequence,host_monotonic_ns,sdk_timestamp_ms,sensor_timestamp_us,"
                         "frame_counter,valid_fraction,zstd_offset,zstd_bytes\n";
            depth_packed.resize(ZSTD_compressBound(depth_pixels.size()));
            for (auto &c : cameras)
                if (c->with_depth) {
                    depth_logs.push_back(DepthLog{
                        c.get(), open_reader(c->depth.path, "open depth store reader"), {}});
                    auto &l = depth_logs.back();
                    l.data.open(depth_log + "/" + c->serial + "-depth.zst", std::ios::binary);
                    std::filesystem::copy_file(output + "/" + c->serial + "-depth.json",
                                               depth_log + "/" + c->serial + "-depth.json",
                                               std::filesystem::copy_options::overwrite_existing);
                    require(depth_csv && l.data, "cannot open depth log");
                }
        }
        // Synthetic source buffers, reused every frame.
        std::vector<uint8_t> pixels(synthetic ? frame_pixels : 0);
        std::vector<uint16_t> ramp(synthetic ? frame_pixels : 0);
        const double begin = monotonic_seconds();
        double next = begin;
        bool failed = false;
        while (!stop_requested) {
            const double now = monotonic_seconds();
            if (duration > 0 && now - begin >= duration)
                break;
            if (synthetic && now >= next) {
                next += 1. / fps;
                for (auto &c : cameras) {
                    FrameMetadata m{};
                    m.sequence = ++c->ir.received;
                    m.host_monotonic_ns = monotonic_ns();
                    c->ir.last_ns = m.host_monotonic_ns;
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
                    publish(c->ir, m, pixels.data());
                    if (c->with_depth) {
                        // Same exposure as the IR frame: identical sensor timestamp. A horizontal
                        // ramp makes 180 degree rotation observable.
                        for (size_t k = 0; k < ramp.size(); ++k)
                            ramp[k] = uint16_t(400 + m.sequence % 600 + k % width);
                        FrameMetadata d = m;
                        d.sequence = ++c->depth.received;
                        d.ir_index = 0;
                        d.stride = width * 2;
                        d.pixel_bytes = ramp.size() * 2;
                        c->depth.last_ns = d.host_monotonic_ns;
                        publish(c->depth, d, ramp.data());
                    }
                }
            }
            auto stale = [&](const Stream &s) {
                return now - begin > 2 &&
                       (!s.last_ns || monotonic_ns() - s.last_ns > 1000000000ULL);
            };
            for (size_t i = 0; i < cameras.size(); ++i) {
                auto &c = *cameras[i];
                const bool depth_stale = c.with_depth && stale(c.depth);
                if (c.error || depth_stale || stale(c.ir)) {
                    std::fprintf(stderr, "Camera %s failed: %s\n", c.serial.c_str(),
                                 c.error == 1   ? "required timestamp metadata missing"
                                 : c.error == 2 ? "capture callback failure"
                                 : depth_stale  ? "no fresh depth frame for one second"
                                                : "no fresh frame for one second");
                    failed = true;
                    stop_requested = 1;
                    break;
                }
                FrameMetadata m{};
                if (!csv.is_open() || kaivuri_frame_read(readers[i].get(), &m, nullptr, 0) != 1 ||
                    m.sequence == logged[i])
                    continue;
                logged[i] = m.sequence;
                csv << c.serial << ',' << m.ir_index << ',' << m.sequence << ','
                    << m.host_monotonic_ns << ',' << m.sdk_timestamp_ms << ',' << m.timestamp_domain
                    << ',' << m.sensor_timestamp_us << ',' << m.frame_timestamp_us << ','
                    << m.frame_counter << ',' << m.actual_exposure_us << ',' << m.gain_level << ','
                    << m.valid_metadata << '\n';
                require(bool(csv), "metadata CSV write failed");
            }
            if (!depth_logs.empty() && now >= next_depth_log) {
                next_depth_log += 1. / depth_log_hz;
                if (next_depth_log < now)
                    next_depth_log = now + 1. / depth_log_hz;
                for (auto &log : depth_logs) {
                    FrameMetadata m{};
                    if (kaivuri_frame_read(log.reader.get(), &m, depth_pixels.data(),
                                           depth_pixels.size()) != 1 ||
                        m.sequence == log.last)
                        continue;
                    log.last = m.sequence;
                    const auto *z = reinterpret_cast<const uint16_t *>(depth_pixels.data());
                    const size_t n = m.pixel_bytes / 2;
                    const size_t valid = n - std::count(z, z + n, uint16_t(0));
                    const size_t bytes =
                        ZSTD_compressCCtx(zstd.get(), depth_packed.data(), depth_packed.size(),
                                          depth_pixels.data(), m.pixel_bytes, 1);
                    require(!ZSTD_isError(bytes), "depth zstd compression failed");
                    log.data.write(reinterpret_cast<const char *>(depth_packed.data()), bytes);
                    log.data.flush();
                    depth_csv << log.camera->serial << ',' << m.sequence << ','
                              << m.host_monotonic_ns << ',' << m.sdk_timestamp_ms << ','
                              << m.sensor_timestamp_us << ',' << m.frame_counter << ','
                              << double(valid) / std::max<size_t>(1, n) << ',' << log.offset << ','
                              << bytes << '\n';
                    depth_csv.flush();
                    log.offset += bytes;
                    require(log.data && depth_csv, "depth log write failed");
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        using ull = unsigned long long;
        for (auto &c : cameras)
            std::fprintf(
                stderr,
                "serial=%s received=%llu busy_drops=%llu counter_gaps=%llu counter_resets=%llu\n",
                c->serial.c_str(), ull(c->ir.received), ull(c->ir.dropped), ull(c->ir.counter_gaps),
                ull(c->ir.counter_resets));
        for (auto &c : cameras)
            if (c->with_depth)
                std::fprintf(stderr,
                             "depth_serial=%s depth_received=%llu depth_busy_drops=%llu "
                             "depth_counter_gaps=%llu\n",
                             c->serial.c_str(), ull(c->depth.received), ull(c->depth.dropped),
                             ull(c->depth.counter_gaps));
        return failed ? 1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Camera failed: %s\n", e.what());
        return 1;
    }
}
