#include "kaivuri/mlp_record.hpp"
#include <chrono>
#include <cmath>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace kaivuri::mlp {
namespace fs = std::filesystem;
const char *drive_source_name(DriveSource source) {
    switch (source) {
    case DriveSource::Manual:
        return "manual";
    case DriveSource::Controller:
        return "controller";
    case DriveSource::Idle:
        return "idle";
    }
    throw std::invalid_argument("invalid drive source");
}
namespace {
constexpr const char *roles[]{"base", "boom", "arm", "bucket"};
constexpr const char *channels[]{"rotate", "lift", "tilt", "scoop"};
constexpr const char *joints[]{"slew", "boom", "arm", "bucket"};
constexpr const char *arm_joints[]{"boom", "arm", "bucket"};

void sync_file(const fs::path &p, bool directory = false) {
    Fd fd(::open(p.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0)));
    require(bool(fd), "cannot open for sync: " + p.string());
    require(fsync(fd.get()) == 0, "sync failed: " + p.string());
}
// Every output is written as NAME.partial and renamed to NAME on commit.
std::string committed(const fs::path &partial) {
    auto name = partial.string();
    name.resize(name.size() - std::char_traits<char>::length(".partial"));
    return name;
}
void scalar(std::ostream &out, double x) {
    if (std::isfinite(x))
        out << x;
    else
        out << "null";
}
// Writes write(item) for each item, comma separated.
template <class Items, class Write> void join(std::ostream &out, const Items &items, Write write) {
    bool first = true;
    for (const auto &item : items) {
        if (!first)
            out << ',';
        first = false;
        write(item);
    }
}
// Appends ",<prefix><name>" for every prefix, then name.
template <class Names>
void columns(std::ostream &out, std::initializer_list<const char *> prefixes, const Names &names) {
    for (const char *prefix : prefixes)
        for (const char *name : names)
            out << ',' << prefix << name;
}
// The first four (arm and slew) channels of a command vector.
void arm_channels(std::ostream &out, const Commands &c) {
    for (unsigned i = 0; i < 4; ++i)
        out << ',' << c[i];
}
std::string drive_header(bool controller) {
    std::ostringstream h;
    h << "timestamp,sample_idx,scheduled_ns,start_ns,post_write_ns,lateness_ms,work_ms,cmd_stale,"
         "cmd_age_s,state_age_s,state_imu_ts_us,imu_device_ts_us,imu_sequence,imu_epoch,imu_valid,"
         "input_valid,pump_on,write_ok,dry_pwm";
    columns(h,
            {"manual_cmd_", "sine_cmd_", "combined_cmd_", "command_clipped_",
             "effective_excitation_cmd_"},
            channels);
    columns(h, {"written_cmd_"}, command_names);
    columns(h, {"joint_pos_"}, joints);
    columns(h, {"joint_vel_"}, arm_joints);
    h << ",vel_age_s";
    columns(h, {"joint_gyro_vel_"}, arm_joints);
    for (auto role : roles)
        for (auto field : {"gx", "gy", "gz", "ax", "ay", "az"})
            h << ",imu_" << field << '_' << role;
    for (auto prefix : {",pwm_us_", ",pwm_count_"})
        for (unsigned i = 0; i < 16; ++i)
            h << prefix << i;
    h << ",sine_enabled,sine_target,sine_seed,excitation_mode,excitation_version,"
         "excitation_block,excitation_elapsed_s,excitation_noise_tick,excitation_stage,"
         "missed_periods,checksum_errors,invalid_imu_frames,imu_resets";
    if (controller) {
        h << ",control_source,control_source_code,controller_valid";
        columns(h, {"assist_cmd_", "constraint_cmd_"}, channels);
    }
    h << '\n';
    return h.str();
}
std::string raw_header() {
    std::ostringstream h;
    h << "device_ts_us,raw_device_ts_us,host_read_ns,sequence,epoch,packet_type";
    for (auto role : roles)
        for (auto field :
             {"qw", "qx", "qy", "qz", "gx_dps", "gy_dps", "gz_dps", "ax_g", "ay_g", "az_g"})
            h << ",imu_" << role << '_' << field;
    h << ",gyro_range_dps,accel_range_g\n";
    return h.str();
}
} // namespace

struct MlpWriter::Impl {
    fs::path out;
    RecordingMetadata metadata;
    std::shared_ptr<RecordingAttachment> attachment;
    std::vector<fs::path> attachment_paths;
    const size_t capacity;
    std::vector<Record> queue;
    alignas(64) std::atomic<uint64_t> head{0};
    alignas(64) std::atomic<uint64_t> tail{0};
    std::atomic<WriterState> state{WriterState::Idle};
    std::atomic<bool> quit{false};
    std::atomic<uint64_t> dropped{0};
    std::atomic<size_t> high_water{0};
    std::atomic<uint64_t> transition{0};
    std::thread thread;
    // Everything below is owned by the writer thread.
    const std::string drive_columns, raw_columns;
    std::ofstream drive, raw, excitation;
    fs::path drive_path, raw_path, excitation_path, session_path;
    uint64_t number = 0, first_tick = 0, rows = 0, packets = 0, last_packet = 0;
    uint64_t drops_at_start = 0, last_block = 0;
    bool first_block = true, have_tick = false, open = false;
    DriveSample final{};

    Impl(fs::path destination, RecordingMetadata m, size_t n,
         std::shared_ptr<RecordingAttachment> a)
        : out(std::move(destination)), metadata(std::move(m)), attachment(std::move(a)),
          capacity(n), queue(n), drive_columns(drive_header(bool(attachment))),
          raw_columns(raw_header()) {
        if (n < 16 || n > 65536 || (n & (n - 1)))
            throw std::invalid_argument("queue capacity must be a power of two in 16..65536");
        fs::create_directories(out);
        thread = std::thread([this] { run(); });
    }
    // Producer side: a copy into a preallocated slot, never a lock or allocation.
    bool push(const Record &r, size_t reserve = 4) {
        const auto h = head.load(std::memory_order_relaxed),
                   t = tail.load(std::memory_order_acquire);
        if (h - t >= capacity - reserve)
            return false;
        queue[h & (capacity - 1)] = r;
        head.store(h + 1, std::memory_order_release);
        const size_t used = size_t(h + 1 - t);
        if (used > high_water.load(std::memory_order_relaxed))
            high_water.store(used, std::memory_order_relaxed);
        return true;
    }
    void begin(const Record &request) {
        if (open)
            close(false);
        const auto now = std::chrono::system_clock::now();
        const auto stamp = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
        localtime_r(&stamp, &tm);
        std::ostringstream id;
        id << std::put_time(&tm, "%Y%m%d_%H%M%S") << '_'
           << std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch())
                      .count() %
                  1000000
           << '_' << getpid() << '_' << ++number;
        if (!metadata.suffix.empty())
            id << '_' << metadata.suffix;
        const auto name = id.str();
        attachment_paths.clear();
        if (attachment) {
            attachment->begin(out, name);
            attachment_paths = attachment->paths();
            for (const auto &path : attachment_paths)
                require(fs::absolute(path).lexically_normal().parent_path() ==
                                fs::absolute(out).lexically_normal() &&
                            path.extension() == ".partial",
                        "invalid recording attachment path");
        }
        drive_path = out / ("drive_log_" + name + ".csv.partial");
        raw_path = out / ("imu_raw_" + name + ".csv.partial");
        excitation_path = out / ("excitation_" + name + ".json.partial");
        session_path = out / ("session_" + name + ".json.partial");
        require(!fs::exists(drive_path) && !fs::exists(raw_path), "recording already exists");
        const std::pair<std::ofstream *, const fs::path *> files[]{
            {&drive, &drive_path}, {&raw, &raw_path}, {&excitation, &excitation_path}};
        for (auto [file, path] : files) {
            file->clear();
            file->open(*path);
        }
        for (auto [file, path] : files) {
            file->exceptions(std::ios::badbit | std::ios::failbit);
            *file << std::setprecision(17);
        }
        open = true;
        rows = packets = last_packet = 0;
        have_tick = false;
        final = {};
        drops_at_start = request.drops_at_start;
        last_block = 0;
        first_block = true;
        excitation << "{\"version\":" << excitation_version
                   << ",\"rng\":\"splitmix64-box-muller-v1\",\"blocks\":[";
        drive << drive_columns;
        raw << raw_columns;
        // An incomplete manifest exists even if capture or the machine is interrupted.
        write_manifest(false);
        if (transition.load(std::memory_order_acquire) == request.transition) {
            auto expected = WriterState::Opening;
            state.compare_exchange_strong(expected, WriterState::Recording,
                                          std::memory_order_release);
        }
        std::cout << "[MLP] Recording " << drive_path.filename() << std::endl;
    }
    void write_raw(const DriveSample &s) {
        const auto &imu = s.imu;
        if (imu.sequence == last_packet)
            return;
        last_packet = imu.sequence;
        raw << imu.device_us << ',' << imu.raw_device_us << ',' << uint64_t(imu.received * 1e9)
            << ',' << imu.sequence << ',' << imu.epoch << ',' << unsigned(imu.packet_type);
        for (const auto &joint : metadata.config.joints) {
            for (double x : imu.quats[joint.sensor])
                raw << ',' << x;
            // Packet type 1 carries no accelerometer.
            for (unsigned j = 0; j < 6; ++j)
                raw << ',' << (imu.packet_type == 1 && j >= 3 ? NAN : imu.motion[joint.sensor][j]);
        }
        raw << ',' << (s.gyro_range_dps > 0 ? s.gyro_range_dps : NAN) << ','
            << (s.accel_range_g > 0 ? s.accel_range_g : NAN) << '\n';
        ++packets;
    }
    void write_drive(const DriveSample &s) {
        if (!have_tick) {
            first_tick = s.tick;
            have_tick = true;
        }
        final = s;
        const auto &config = metadata.config;
        const bool stale = !s.write_ok || !s.input_valid || !s.imu_valid || !s.controller_valid;
        const int64_t imu_us = s.imu.valid ? int64_t(s.imu.device_us) : -1;
        drive << double(s.post_write_ns) / 1e9 << ',' << s.tick - first_tick << ','
              << s.scheduled_ns << ',' << s.start_ns << ',' << s.post_write_ns << ','
              << double(int64_t(s.start_ns) - int64_t(s.scheduled_ns)) / 1e6 << ','
              << double(s.post_write_ns - s.start_ns) / 1e6 << ',' << stale << ','
              << s.command_age_s << ',' << s.state_age_s << ',' << imu_us << ',' << imu_us << ','
              << s.imu.sequence << ',' << s.imu.epoch << ',' << s.imu_valid << ',' << s.input_valid
              << ',' << s.pump_on << ',' << s.write_ok << ',' << metadata.dry;
        arm_channels(drive, s.manual);
        arm_channels(drive, s.excitation);
        arm_channels(drive, s.command);
        for (unsigned i = 0; i < 4; ++i)
            drive << ',' << (std::abs(s.manual[i] + s.assist[i] + s.excitation[i]) > 1 + 1e-9);
        for (unsigned i = 0; i < 4; ++i)
            drive << ',' << s.command[i] - s.manual[i] - s.assist[i] - s.constraint[i];
        for (double x : s.command)
            drive << ',' << x;
        for (double x : s.angles)
            drive << ',' << (s.imu_valid ? x : NAN);
        // Historical joint_vel_* came from a separate finite-difference cache.
        // Same-packet gyro rates have explicit column names below.
        drive << ",nan,nan,nan,nan";
        for (unsigned i = 1; i < 4; ++i) {
            const auto &j = config.joints[i];
            drive << ','
                  << (s.imu_valid ? (s.imu.motion[j.sensor][1] - s.imu.motion[j.parent_sensor][1]) *
                                        pi / 180
                                  : NAN);
        }
        for (const auto &joint : config.joints)
            for (unsigned j = 0; j < 6; ++j)
                drive << ','
                      << (s.imu_valid && !(s.imu.packet_type == 1 && j >= 3)
                              ? s.imu.motion[joint.sensor][j] * (j < 3 ? pi / 180 : 9.80665)
                              : NAN);
        for (double x : s.pulses)
            drive << ',' << x;
        for (double x : s.pulses)
            drive << ',' << pulse_count(x, config.frequency);
        drive << ',' << s.enabled << ',' << target_name(s.target) << ',' << s.seed << ','
              << waveform_name(s.waveform) << ',' << excitation_version << ',' << s.block_id << ','
              << s.excitation_elapsed_s << ',' << s.noise_tick << ',' << stage_name(s.stage) << ','
              << s.missed_periods << ',' << s.checksum_errors << ',' << s.invalid_frames << ','
              << s.resets;
        if (attachment) {
            drive << ',' << drive_source_name(s.source) << ',' << static_cast<unsigned>(s.source)
                  << ',' << s.controller_valid;
            arm_channels(drive, s.assist);
            arm_channels(drive, s.constraint);
        }
        drive << '\n';
        ++rows;
        if (attachment)
            attachment->drive(s);
    }
    void write_manifest(bool complete) {
        std::ofstream manifest(session_path);
        manifest.exceptions(std::ios::badbit | std::ios::failbit);
        manifest << std::setprecision(17) << std::boolalpha
                 << "{\"format\":\"kaivuri-native-mlp-v1\",\"complete\":" << complete
                 << ",\"control_hz\":100,\"dry_pwm\":" << metadata.dry
                 << ",\"collection_id\":" << json_quote(metadata.collection_id)
                 << ",\"collector\":" << json_quote(metadata.collector)
                 << ",\"enable_slew\":" << metadata.slew << ",\"enable_tracks\":" << metadata.tracks
                 << ",\"rows\":" << rows << ",\"raw_packets\":" << packets
                 << ",\"queue_capacity\":" << capacity << ",\"queue_high_water\":" << high_water
                 << ",\"dropped\":" << dropped.load() - drops_at_start
                 << ",\"missed_periods\":" << final.missed_periods
                 << ",\"checksum_errors\":" << final.checksum_errors
                 << ",\"invalid_imu_frames\":" << final.invalid_frames
                 << ",\"imu_resets\":" << final.resets
                 << ",\"source_revision\":" << json_quote(metadata.source_revision)
                 << ",\"executable_sha256\":" << json_quote(metadata.executable_sha256)
                 << ",\"suffix\":" << json_quote(metadata.suffix) << ",\"options\":"
                 << (metadata.options_json.empty() ? "{}" : metadata.options_json)
                 << ",\"files\":{";
        join(manifest, metadata.files, [&](const auto &file) {
            const auto &[name, bytes] = file;
            manifest << json_quote(name)
                     << ":{\"sha256\":" << json_quote(sha256_hex(bytes.data(), bytes.size()))
                     << ",\"content\":" << json_quote(bytes) << '}';
        });
        manifest << "},\"sensor_contract\":{\"version\":1,\"roles\":[\"base\",\"boom\",\"arm\","
                    "\"bucket\"],\"mount_pitch_rad\":[";
        // Fixed training frame, also used by hydraulic_controller.sensors and bundle export.
        const std::array<double, 4> training_mount{0, 2 * std::atan2(.12057, .9927048),
                                                   2 * std::atan2(.0087265, .9999619), 0};
        join(manifest, training_mount, [&](double x) { scalar(manifest, x); });
        manifest << "],\"runtime_mount_pitch_rad\":[";
        const auto &joints = metadata.config.joints;
        bool supported = joints[0].parent_sensor == -1;
        for (unsigned i = 0; i < 4; ++i) {
            if (i)
                manifest << ',';
            const auto &q = metadata.config.offsets_inv[joints[i].sensor];
            scalar(manifest, -2 * std::atan2(q[2], q[0]));
            supported = supported && std::abs(q[1]) + std::abs(q[3]) < 1e-6;
            if (i > 0)
                supported = supported && joints[i].parent_sensor == joints[i - 1].sensor &&
                            (joints[i].pitch_delta || joints[i].extraction_axis == Vec3{0, 1, 0});
        }
        manifest << "],\"carriage_pitch_zero_rad\":-0.004821687005460262,\"axis_aligned_y\":"
                 << supported << "},\"outputs\":{";
        // Output hashes are only meaningful once every stream is closed.
        auto hashes = [&](const std::vector<fs::path> &paths) {
            if (complete)
                join(manifest, paths, [&](const fs::path &p) {
                    manifest << json_quote(fs::path(committed(p)).filename().string()) << ':'
                             << json_quote(sha256_file(p));
                });
        };
        hashes({drive_path, raw_path, excitation_path});
        manifest << "},\"attachments\":{";
        hashes(attachment_paths);
        manifest << "}}\n";
    }
    void write_block(const ExcitationBlock &b) {
        if (!first_block && b.id == last_block)
            return;
        if (!first_block)
            excitation << ',';
        first_block = false;
        last_block = b.id;
        const auto &o = b.options;
        excitation << "{\"id\":" << b.id << ",\"seed\":" << o.seed
                   << ",\"target\":" << json_quote(target_name(o.target))
                   << ",\"mode\":" << json_quote(waveform_name(o.waveform))
                   << ",\"start_hz\":" << o.start_hz << ",\"end_hz\":" << o.end_hz
                   << ",\"sweep_s\":" << o.sweep_s << ",\"stroke_s\":" << o.stroke_s
                   << ",\"margin_deg\":" << o.margin_deg << ",\"gain_per_deg\":" << o.gain_per_deg
                   << ",\"valve_limit\":" << o.valve_limit
                   << ",\"ramp_s\":1,\"noise_hz\":100,\"noise_clip_sigma\":3,\"chirp_hold_s\":"
                   << (o.waveform == Waveform::Stroke ? 0 : 10)
                   << ",\"noise_seed\":" << b.noise_seed << ",\"initial_direction\":[";
        join(excitation, b.initial_direction, [&](double d) { excitation << d; });
        excitation << "],\"limits_deg\":[";
        join(excitation, b.limits_deg,
             [&](const auto &l) { excitation << '[' << l[0] << ',' << l[1] << ']'; });
        excitation << "],\"parameters\":[";
        join(excitation, b.parameters, [&](const WaveParameters &p) {
            excitation << "{\"env\":" << p.env << ",\"carrier\":" << p.carrier
                       << ",\"rate\":" << p.rate << ",\"depth\":" << p.depth
                       << ",\"cutoff\":" << p.cutoff << ",\"noise\":" << p.noise
                       << ",\"amplitude\":" << p.amplitude << ",\"phi1\":" << p.phi1
                       << ",\"phi2\":" << p.phi2 << '}';
        });
        excitation << "]}";
    }
    void close(bool complete) {
        if (!open)
            return;
        excitation << "]}\n";
        for (auto *file : {&drive, &raw, &excitation})
            file->flush();
        for (auto *file : {&drive, &raw, &excitation})
            file->close();
        if (attachment)
            attachment->close();
        complete = complete && dropped.load(std::memory_order_acquire) == drops_at_start &&
                   rows > 0 && packets > 0;
        write_manifest(complete);
        for (const auto &p : {drive_path, raw_path, excitation_path, session_path})
            sync_file(p);
        for (const auto &p : attachment_paths)
            sync_file(p);
        if (complete) {
            // The manifest is the final commit marker for the entire trio.
            std::vector<fs::path> paths{raw_path, excitation_path, drive_path};
            paths.insert(paths.end(), attachment_paths.begin(), attachment_paths.end());
            paths.push_back(session_path);
            for (const auto &p : paths) {
                const auto final_name = committed(p);
                require(!fs::exists(final_name), "output exists");
                fs::rename(p, final_name);
            }
            sync_file(out, true);
        }
        open = false;
        std::cout << "[MLP] " << (complete ? "Saved " : "Incomplete ") << rows << " rows, "
                  << packets << " IMU packets\n"
                  << std::flush;
    }
    void fail(const std::exception &e) noexcept {
        if (attachment)
            attachment->abort();
        for (auto *file : {&drive, &raw, &excitation})
            try {
                if (file->is_open())
                    file->close();
            } catch (...) {
            }
        open = false;
        state.store(WriterState::Failed, std::memory_order_release);
        std::cerr << "[MLP] Writer failed; files remain incomplete: " << e.what() << std::endl;
    }
    void run() {
        while (!quit.load(std::memory_order_acquire) || tail.load() != head.load()) {
            const auto t = tail.load(std::memory_order_relaxed);
            if (t == head.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            // Copy out before releasing the slot back to the producer.
            const Record r = queue[t & (capacity - 1)];
            tail.store(t + 1, std::memory_order_release);
            try {
                if (r.kind == RecordKind::Start)
                    begin(r);
                else if (r.kind == RecordKind::Finish) {
                    close(r.complete);
                    if (transition.load(std::memory_order_acquire) == r.transition) {
                        auto expected = WriterState::Closing;
                        state.compare_exchange_strong(expected, WriterState::Idle,
                                                      std::memory_order_release);
                    }
                } else if (open) {
                    if (r.kind == RecordKind::Raw)
                        write_raw(r.sample);
                    else if (r.kind == RecordKind::Drive)
                        write_drive(r.sample);
                    else if (r.kind == RecordKind::Block)
                        write_block(r.block);
                }
            } catch (const std::exception &e) {
                fail(e);
            }
        }
        if (open) {
            try {
                close(false);
            } catch (...) {
            }
        }
    }
};
MlpWriter::MlpWriter(const fs::path &out, RecordingMetadata metadata, size_t capacity,
                     std::shared_ptr<RecordingAttachment> attachment)
    : impl_(std::make_unique<Impl>(out, std::move(metadata), capacity, std::move(attachment))) {}
MlpWriter::~MlpWriter() {
    shutdown();
}
bool MlpWriter::start() {
    auto old = impl_->state.load(std::memory_order_acquire);
    // A new take may be queued behind an in-flight Finish. Transition IDs keep
    // that Finish from overwriting Opening with Idle when its fsync completes.
    if (old != WriterState::Idle && old != WriterState::Failed && old != WriterState::Closing)
        return false;
    impl_->state.store(WriterState::Opening, std::memory_order_release);
    Record r;
    r.kind = RecordKind::Start;
    r.transition = ++impl_->transition;
    r.drops_at_start = impl_->dropped.load();
    if (!impl_->push(r, 0)) {
        impl_->state = old;
        return false;
    }
    return true;
}
bool MlpWriter::push(const Record &r) {
    const auto s = state();
    if (s != WriterState::Recording && s != WriterState::Opening)
        return false;
    if (impl_->push(r))
        return true;
    ++impl_->dropped;
    return false;
}
bool MlpWriter::rotate() {
    Record end;
    end.kind = RecordKind::Finish;
    Record begin;
    begin.kind = RecordKind::Start;
    begin.drops_at_start = impl_->dropped.load();
    const auto generation = impl_->transition.fetch_add(2, std::memory_order_acq_rel);
    end.transition = generation + 1;
    begin.transition = generation + 2;
    impl_->state.store(WriterState::Opening, std::memory_order_release);
    if (!impl_->push(end, 0) || !impl_->push(begin, 0)) {
        ++impl_->dropped;
        return false;
    }
    return true;
}
void MlpWriter::finish(bool complete) {
    const auto s = state();
    if (s != WriterState::Recording && s != WriterState::Opening)
        return;
    impl_->state.store(WriterState::Closing, std::memory_order_release);
    Record r;
    r.kind = RecordKind::Finish;
    r.complete = complete;
    r.transition = ++impl_->transition;
    if (!impl_->push(r, 0)) {
        ++impl_->dropped;
        impl_->state = WriterState::Failed;
    }
}
void MlpWriter::shutdown() {
    if (!impl_ || !impl_->thread.joinable())
        return;
    impl_->quit.store(true, std::memory_order_release);
    impl_->thread.join();
}
WriterState MlpWriter::state() const {
    return impl_->state.load(std::memory_order_acquire);
}
uint64_t MlpWriter::dropped() const {
    return impl_->dropped.load();
}
size_t MlpWriter::high_water() const {
    return impl_->high_water.load();
}
void RawCapture::packet(const ImuSample &sample) {
    if (!recording)
        return;
    if (sample.epoch != epoch) {
        // A new parser epoch (IMU reset) closes this take and queues the next one.
        if (!writer->rotate()) {
            failed = true;
            recording = false;
            return;
        }
        epoch = sample.epoch;
        last_raw = 0;
        reset = true;
    }
    if (sample.sequence == last_raw)
        return;
    Record record;
    record.kind = RecordKind::Raw;
    record.sample.imu = sample;
    record.sample.gyro_range_dps = parser->gyro_range_dps;
    record.sample.accel_range_g = parser->accel_range_g;
    if (!writer->push(record)) {
        failed = true;
        recording = false;
        writer->finish(false);
    } else
        last_raw = sample.sequence;
}
} // namespace kaivuri::mlp
