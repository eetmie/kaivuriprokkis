#include "kaivuri/pid_collect.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace kaivuri::pidcollect {
namespace fs = std::filesystem;
namespace {
constexpr std::array<const char *, 3> joints{"boom", "arm", "bucket"};
template <class T> void csv_values(std::ostream &stream, const T &values) {
    for (const auto &v : values)
        stream << ',' << v;
}
template <class T> void json_values(std::ostream &stream, const T &values) {
    bool first = true;
    stream << '[';
    for (const auto &v : values) {
        if (!first)
            stream << ',';
        first = false;
        stream << v;
    }
    stream << ']';
}
} // namespace
struct PidAttachment::Impl {
    const std::string header;
    const size_t capacity;
    std::vector<Frame> queue;
    alignas(64) std::atomic<uint64_t> head{0};
    alignas(64) std::atomic<uint64_t> tail{0};
    std::ofstream csv, protocol;
    std::vector<fs::path> files;
    uint64_t last_segment = 0;
    bool first_segment = true;
    Impl(std::string text, size_t n) : header(std::move(text)), capacity(n), queue(n) {
        if (n < 16 || n > 65536 || (n & (n - 1)))
            throw std::invalid_argument("invalid PID telemetry queue capacity");
    }
    bool push(const Frame &frame) {
        const auto h = head.load(std::memory_order_relaxed),
                   t = tail.load(std::memory_order_acquire);
        if (h - t == capacity)
            return false;
        queue[h & (capacity - 1)] = frame;
        head.store(h + 1, std::memory_order_release);
        return true;
    }
    Frame take(uint64_t tick) {
        while (true) {
            const auto t = tail.load(std::memory_order_relaxed);
            if (t == head.load(std::memory_order_acquire))
                throw std::runtime_error("missing PID telemetry for drive row");
            const auto frame = queue[t & (capacity - 1)];
            if (frame.tick > tick)
                throw std::runtime_error("PID telemetry/drive clock mismatch");
            tail.store(t + 1, std::memory_order_release);
            if (frame.tick == tick)
                return frame;
            // A failed base-queue push may leave an orphan from an incomplete take.
        }
    }
    void segment(const collection::Segment &s) {
        if (!first_segment && s.id == last_segment)
            return;
        if (!first_segment)
            protocol << ',';
        first_segment = false;
        last_segment = s.id;
        protocol << "{\"id\":" << s.id << ",\"begin_ns\":" << s.begin_ns
                 << ",\"end_ns_requested\":" << s.end_ns
                 << ",\"duration_requested_s\":" << s.duration
                 << ",\"kind\":" << mlp::json_quote(collection::kind_name(s.kind))
                 << ",\"goal_cell\":" << s.goal_cell << ",\"goal_q_rad\":";
        json_values(protocol, s.goal_q);
        protocol << ",\"velocity_xz_m_s\":";
        json_values(protocol, s.velocity);
        protocol << ",\"pitch_target_rad\":" << s.pitch << ",\"pid_rate_requested_hz\":";
        json_values(protocol, s.pid_rates);
        protocol << ",\"gains_requested\":[";
        for (unsigned j = 0; j < 3; ++j) {
            if (j)
                protocol << ',';
            const auto &g = s.gains[j];
            protocol << "{\"kp\":" << g.kp << ",\"ki\":" << g.ki << ",\"kd\":" << g.kd
                     << ",\"tau_s\":" << g.tau << '}';
        }
        protocol << "]}";
    }
};
PidAttachment::PidAttachment(std::string header, size_t capacity)
    : impl_(std::make_unique<Impl>(std::move(header), capacity)) {}
PidAttachment::~PidAttachment() {
    abort();
}
bool PidAttachment::push(const Frame &frame) {
    return impl_->push(frame);
}
void PidAttachment::begin(const fs::path &directory, const std::string &stem) {
    auto &p = *impl_;
    abort();
    p.files = {directory / ("pid_control_" + stem + ".csv.partial"),
               directory / ("pid_protocol_" + stem + ".json.partial")};
    for (const auto &path : p.files)
        if (fs::exists(path))
            throw std::runtime_error("PID output already exists");
    p.csv.clear();
    p.protocol.clear();
    p.csv.open(p.files[0]);
    p.protocol.open(p.files[1]);
    p.csv.exceptions(std::ios::failbit | std::ios::badbit);
    p.protocol.exceptions(std::ios::failbit | std::ios::badbit);
    p.csv << std::setprecision(17);
    p.protocol << std::setprecision(17) << p.header;
    p.first_segment = true;
    p.last_segment = 0;
    p.csv << "tick,post_write_ns,imu_epoch,control_source,active,controller_valid,reason,segment_"
             "id,segment_kind,coverage_cell,recovery";
    for (const auto *prefix : {"task", "task_ref"})
        for (const auto *coordinate : {"x_m", "z_m", "pitch_rad"})
            p.csv << ',' << prefix << '_' << coordinate;
    for (const auto *prefix :
         {"joint_ref_rad", "pid_rate_requested_hz", "pid_updated", "pid_dt_s", "kp_applied",
          "ki_applied", "kd_applied", "tau_applied_s", "assist_requested", "excitation_requested",
          "command_before_guard", "guard_scale", "guard_predicted_q_rad", "guard_limited"})
        for (const auto *joint : joints)
            p.csv << ',' << prefix << '_' << joint;
    p.csv << ",guard_valid,guard_reason,ik_position_error_m,ik_pitch_error_rad,ik_iterations\n";
}
void PidAttachment::drive(const mlp::DriveSample &sample) {
    auto &p = *impl_;
    const auto f = p.take(sample.tick);
    const auto &o = f.output;
    const auto &s = f.segment;
    if (f.active && s.id != 0)
        p.segment(s);
    p.csv << sample.tick << ',' << sample.post_write_ns << ',' << sample.imu.epoch << ','
          << mlp::drive_source_name(sample.source) << ',' << f.active << ','
          << sample.controller_valid << ',' << f.reason << ',' << (f.active ? s.id : 0) << ','
          << (f.active ? collection::kind_name(s.kind) : "idle") << ',' << o.cell << ','
          << o.recovery;
    csv_values(p.csv, o.measured.pose);
    csv_values(p.csv, o.reference);
    csv_values(p.csv, o.joint_reference);
    csv_values(p.csv, f.active ? s.pid_rates : collection::Angles{});
    csv_values(p.csv, o.pid_updated);
    csv_values(p.csv, o.pid_dt);
    for (unsigned term = 0; term < 4; ++term)
        for (const auto &g : o.applied_gains)
            p.csv << ',' << (term == 0 ? g.kp : term == 1 ? g.ki : term == 2 ? g.kd : g.tau);
    csv_values(p.csv, o.assist);
    csv_values(p.csv, o.excitation);
    csv_values(p.csv, o.command);
    csv_values(p.csv, f.guard.scale);
    csv_values(p.csv, f.guard.predicted_q);
    csv_values(p.csv, f.guard.limited);
    p.csv << ',' << f.guard.valid << ',' << f.guard.reason << ',' << o.ik.position_error << ','
          << o.ik.pitch_error << ',' << o.ik.iterations << '\n';
}
void PidAttachment::close() {
    auto &p = *impl_;
    p.protocol << "]}\n";
    p.csv.flush();
    p.protocol.flush();
    p.csv.close();
    p.protocol.close();
}
void PidAttachment::abort() noexcept {
    for (auto *stream : {&impl_->csv, &impl_->protocol}) {
        stream->exceptions(std::ios::goodbit);
        if (stream->is_open())
            stream->close();
    }
}
std::vector<fs::path> PidAttachment::paths() const {
    return impl_->files;
}
} // namespace kaivuri::pidcollect
