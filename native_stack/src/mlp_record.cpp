#include "kaivuri/mlp_record.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/evp.h>
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
void sync_file(const fs::path &p, bool directory = false) {
    int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
    if (fd < 0)
        throw std::runtime_error("cannot open for sync: " + p.string());
    const int result = fsync(fd);
    ::close(fd);
    if (result != 0)
        throw std::runtime_error("sync failed: " + p.string());
}
void scalar(std::ostream &out, double x) {
    if (std::isfinite(x))
        out << x;
    else
        out << "null";
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
    std::ofstream drive, raw, excitation;
    fs::path drive_path, raw_path, excitation_path, session_path;
    uint64_t last_block = 0;
    bool first_block = true;
    uint64_t number = 0, first_tick = 0, rows = 0, packets = 0, last_packet = 0;
    uint64_t drops_at_start = 0;
    bool have_tick = false, open = false;
    DriveSample final{};
    Impl(fs::path destination, RecordingMetadata m, size_t n,
         std::shared_ptr<RecordingAttachment> a)
        : out(std::move(destination)), metadata(std::move(m)), attachment(std::move(a)),
          capacity(n), queue(n) {
        if (n < 16 || n > 65536 || (n & (n - 1)))
            throw std::invalid_argument("queue capacity must be a power of two in 16..65536");
        fs::create_directories(out);
        thread = std::thread([this] { run(); });
    }
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
                if (fs::absolute(path).lexically_normal().parent_path() !=
                        fs::absolute(out).lexically_normal() ||
                    path.extension() != ".partial")
                    throw std::runtime_error("invalid recording attachment path");
        }
        drive_path = out / ("drive_log_" + name + ".csv.partial");
        raw_path = out / ("imu_raw_" + name + ".csv.partial");
        excitation_path = out / ("excitation_" + name + ".json.partial");
        session_path = out / ("session_" + name + ".json.partial");
        if (fs::exists(drive_path) || fs::exists(raw_path))
            throw std::runtime_error("recording already exists");
        drive.clear();
        raw.clear();
        excitation.clear();
        drive.open(drive_path);
        raw.open(raw_path);
        excitation.open(excitation_path);
        drive.exceptions(std::ios::badbit | std::ios::failbit);
        raw.exceptions(std::ios::badbit | std::ios::failbit);
        excitation.exceptions(std::ios::badbit | std::ios::failbit);
        open = true;
        rows = packets = last_packet = 0;
        have_tick = false;
        final = {};
        drops_at_start = request.drops_at_start;
        last_block = 0;
        first_block = true;
        excitation << std::setprecision(17) << "{\"version\":" << excitation_version
                   << ",\"rng\":\"splitmix64-box-muller-v1\",\"blocks\":[";
        drive << std::setprecision(17);
        raw << std::setprecision(17);
        drive << "timestamp,sample_idx,scheduled_ns,start_ns,post_write_ns,lateness_ms,work_ms,cmd_"
                 "stale,cmd_age_s,state_age_s,state_imu_ts_us,imu_device_ts_us,imu_sequence,imu_"
                 "epoch,imu_valid,input_valid,pump_on,write_ok,dry_pwm";
        for (auto prefix : {"manual_cmd_", "sine_cmd_", "combined_cmd_", "command_clipped_",
                            "effective_excitation_cmd_"})
            for (auto channel : channels)
                drive << ',' << prefix << channel;
        for (auto name : command_names)
            drive << ",written_cmd_" << name;
        for (auto name : {"slew", "boom", "arm", "bucket"})
            drive << ",joint_pos_" << name;
        for (auto name : {"boom", "arm", "bucket"})
            drive << ",joint_vel_" << name;
        drive << ",vel_age_s";
        for (auto name : {"boom", "arm", "bucket"})
            drive << ",joint_gyro_vel_" << name;
        for (auto role : roles)
            for (auto field : {"gx", "gy", "gz", "ax", "ay", "az"})
                drive << ",imu_" << field << '_' << role;
        for (unsigned i = 0; i < 16; ++i)
            drive << ",pwm_us_" << i;
        for (unsigned i = 0; i < 16; ++i)
            drive << ",pwm_count_" << i;
        drive << ",sine_enabled,sine_target,sine_seed,excitation_mode,excitation_version,"
                 "excitation_block,excitation_elapsed_s,excitation_noise_tick,excitation_stage,"
                 "missed_periods,checksum_errors,invalid_imu_frames,imu_resets";
        if (attachment) {
            drive << ",control_source,control_source_code,controller_valid";
            for (auto prefix : {"assist_cmd_", "constraint_cmd_"})
                for (auto channel : channels)
                    drive << ',' << prefix << channel;
        }
        drive << '\n';
        raw << "device_ts_us,raw_device_ts_us,host_read_ns,sequence,epoch,packet_type";
        for (auto role : roles)
            for (auto field :
                 {"qw", "qx", "qy", "qz", "gx_dps", "gy_dps", "gz_dps", "ax_g", "ay_g", "az_g"})
                raw << ",imu_" << role << '_' << field;
        raw << ",gyro_range_dps,accel_range_g\n";
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
        for (unsigned i = 0; i < 4; ++i) {
            const int sensor = metadata.config.joints[i].sensor;
            for (double x : imu.quats[sensor])
                raw << ',' << x;
            for (unsigned j = 0; j < 6; ++j)
                raw << ',' << (imu.packet_type == 1 && j >= 3 ? NAN : imu.motion[sensor][j]);
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
        const bool stale = !s.write_ok || !s.input_valid || !s.imu_valid || !s.controller_valid;
        drive << double(s.post_write_ns) / 1e9 << ',' << s.tick - first_tick << ','
              << s.scheduled_ns << ',' << s.start_ns << ',' << s.post_write_ns << ','
              << double(int64_t(s.start_ns) - int64_t(s.scheduled_ns)) / 1e6 << ','
              << double(s.post_write_ns - s.start_ns) / 1e6 << ',' << stale << ','
              << s.command_age_s << ',' << s.state_age_s << ',';
        if (s.imu.valid)
            drive << s.imu.device_us;
        else
            drive << -1;
        drive << ',';
        if (s.imu.valid)
            drive << s.imu.device_us;
        else
            drive << -1;
        drive << ',' << s.imu.sequence << ',' << s.imu.epoch << ',' << s.imu_valid << ','
              << s.input_valid << ',' << s.pump_on << ',' << s.write_ok << ',' << metadata.dry;
        for (const auto *values : {&s.manual, &s.excitation, &s.command})
            for (unsigned i = 0; i < 4; ++i)
                drive << ',' << (*values)[i];
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
            const auto &j = metadata.config.joints[i];
            drive << ','
                  << (s.imu_valid ? (s.imu.motion[j.sensor][1] - s.imu.motion[j.parent_sensor][1]) *
                                        pi / 180
                                  : NAN);
        }
        for (unsigned i = 0; i < 4; ++i)
            for (unsigned j = 0; j < 6; ++j)
                drive << ','
                      << (s.imu_valid && !(s.imu.packet_type == 1 && j >= 3)
                              ? s.imu.motion[metadata.config.joints[i].sensor][j] *
                                    (j < 3 ? pi / 180 : 9.80665)
                              : NAN);
        for (double x : s.pulses)
            drive << ',' << x;
        for (double x : s.pulses)
            drive << ',' << pulse_count(x, metadata.config.frequency);
        drive << ',' << s.enabled << ',' << target_name(s.target) << ',' << s.seed << ','
              << waveform_name(s.waveform) << ',' << excitation_version << ',' << s.block_id << ','
              << s.excitation_elapsed_s << ',' << s.noise_tick << ',' << stage_name(s.stage) << ','
              << s.missed_periods << ',' << s.checksum_errors << ',' << s.invalid_frames << ','
              << s.resets;
        if (attachment) {
            drive << ',' << drive_source_name(s.source) << ',' << static_cast<unsigned>(s.source)
                  << ',' << s.controller_valid;
            for (const auto *values : {&s.assist, &s.constraint})
                for (unsigned i = 0; i < 4; ++i)
                    drive << ',' << (*values)[i];
        }
        drive << '\n';
        ++rows;
        if (attachment)
            attachment->drive(s);
    }
    void write_manifest(bool complete) {
        std::ofstream manifest(session_path);
        manifest.exceptions(std::ios::badbit | std::ios::failbit);
        manifest << std::setprecision(17) << "{\"format\":\"kaivuri-native-mlp-v1\",\"complete\":"
                 << (complete ? "true" : "false")
                 << ",\"control_hz\":100,\"dry_pwm\":" << (metadata.dry ? "true" : "false")
                 << ",\"collection_id\":" << json_quote(metadata.collection_id)
                 << ",\"collector\":" << json_quote(metadata.collector)
                 << ",\"enable_slew\":" << (metadata.slew ? "true" : "false")
                 << ",\"enable_tracks\":" << (metadata.tracks ? "true" : "false")
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
        bool first = true;
        for (const auto &[name, bytes] : metadata.files) {
            if (!first)
                manifest << ',';
            first = false;
            manifest << json_quote(name)
                     << ":{\"sha256\":" << json_quote(sha256_hex(bytes.data(), bytes.size()))
                     << ",\"content\":" << json_quote(bytes) << '}';
        }
        manifest << "},\"sensor_contract\":{\"version\":1,\"roles\":[\"base\",\"boom\",\"arm\","
                    "\"bucket\"],\"mount_pitch_rad\":[";
        // Fixed training frame, also used by hydraulic_controller.sensors and bundle export.
        const std::array<double, 4> training_mount{0, 2 * std::atan2(.12057, .9927048),
                                                   2 * std::atan2(.0087265, .9999619), 0};
        for (unsigned i = 0; i < 4; ++i) {
            if (i)
                manifest << ',';
            scalar(manifest, training_mount[i]);
        }
        manifest << "],\"runtime_mount_pitch_rad\":[";
        bool supported = metadata.config.joints[0].parent_sensor == -1;
        for (unsigned i = 0; i < 4; ++i) {
            if (i)
                manifest << ',';
            const auto &q = metadata.config.offsets_inv[metadata.config.joints[i].sensor];
            scalar(manifest, -2 * std::atan2(q[2], q[0]));
            supported = supported && std::abs(q[1]) + std::abs(q[3]) < 1e-6;
            if (i > 0) {
                const auto &j = metadata.config.joints[i];
                supported = supported && j.parent_sensor == metadata.config.joints[i - 1].sensor &&
                            (j.pitch_delta || j.extraction_axis == Vec3{0, 1, 0});
            }
        }
        manifest << "],\"carriage_pitch_zero_rad\":-0.004821687005460262,\"axis_aligned_y\":"
                 << (supported ? "true" : "false") << "},\"outputs\":{";
        first = true;
        if (complete)
            for (const auto &p : {drive_path, raw_path, excitation_path}) {
                if (!first)
                    manifest << ',';
                first = false;
                std::string name = p.filename().string();
                name.resize(name.size() - 8);
                manifest << json_quote(name) << ':' << json_quote(sha256_file(p));
            }
        manifest << "},\"attachments\":{";
        first = true;
        if (complete)
            for (const auto &path : attachment_paths) {
                if (!first)
                    manifest << ',';
                first = false;
                auto name = path.filename().string();
                name.resize(name.size() - 8);
                manifest << json_quote(name) << ':' << json_quote(sha256_file(path));
            }
        manifest << "}}\n";
    }
    void write_block(const ExcitationBlock &b) {
        if (!first_block && b.id == last_block)
            return;
        last_block = b.id;
        auto &out_file = excitation;
        if (!first_block)
            out_file << ',';
        first_block = false;
        const auto &o = b.options;
        out_file << "{\"id\":" << b.id << ",\"seed\":" << o.seed
                 << ",\"target\":" << json_quote(target_name(o.target))
                 << ",\"mode\":" << json_quote(waveform_name(o.waveform))
                 << ",\"start_hz\":" << o.start_hz << ",\"end_hz\":" << o.end_hz
                 << ",\"sweep_s\":" << o.sweep_s << ",\"stroke_s\":" << o.stroke_s
                 << ",\"margin_deg\":" << o.margin_deg << ",\"gain_per_deg\":" << o.gain_per_deg
                 << ",\"valve_limit\":" << o.valve_limit
                 << ",\"ramp_s\":1,\"noise_hz\":100,\"noise_clip_sigma\":3,\"chirp_hold_s\":"
                 << (o.waveform == Waveform::Stroke ? 0 : 10) << ",\"noise_seed\":" << b.noise_seed
                 << ",\"initial_direction\":[";
        for (unsigned i = 0; i < 4; ++i) {
            if (i)
                out_file << ',';
            out_file << b.initial_direction[i];
        }
        out_file << "],\"limits_deg\":[";
        for (unsigned i = 0; i < 4; ++i) {
            if (i)
                out_file << ',';
            out_file << '[' << b.limits_deg[i][0] << ',' << b.limits_deg[i][1] << ']';
        }
        out_file << "],\"parameters\":[";
        for (unsigned i = 0; i < 4; ++i) {
            if (i)
                out_file << ',';
            const auto &p = b.parameters[i];
            out_file << "{\"env\":" << p.env << ",\"carrier\":" << p.carrier
                     << ",\"rate\":" << p.rate << ",\"depth\":" << p.depth
                     << ",\"cutoff\":" << p.cutoff << ",\"noise\":" << p.noise
                     << ",\"amplitude\":" << p.amplitude << ",\"phi1\":" << p.phi1
                     << ",\"phi2\":" << p.phi2 << '}';
        }
        out_file << "]}";
    }
    void close(bool complete) {
        if (!open)
            return;
        excitation << "]}\n";
        drive.flush();
        raw.flush();
        excitation.flush();
        drive.close();
        raw.close();
        excitation.close();
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
                std::string final_name = p.string();
                final_name.resize(final_name.size() - 8);
                if (fs::exists(final_name))
                    throw std::runtime_error("output exists");
                fs::rename(p, final_name);
            }
            sync_file(out, true);
        }
        open = false;
        std::cout << "[MLP] " << (complete ? "Saved " : "Incomplete ") << rows << " rows, "
                  << packets << " IMU packets\n"
                  << std::flush;
    }
    void run() {
        while (!quit.load(std::memory_order_acquire) || tail.load() != head.load()) {
            const auto t = tail.load(std::memory_order_relaxed);
            if (t == head.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
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
                if (attachment)
                    attachment->abort();
                try {
                    if (drive.is_open())
                        drive.close();
                } catch (...) {
                }
                try {
                    if (raw.is_open())
                        raw.close();
                } catch (...) {
                }
                try {
                    if (excitation.is_open())
                        excitation.close();
                } catch (...) {
                }
                open = false;
                state.store(WriterState::Failed, std::memory_order_release);
                std::cerr << "[MLP] Writer failed; files remain incomplete: " << e.what()
                          << std::endl;
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
