#include "kaivuri/pid_collect.hpp"
#include "kaivuri/util.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace kaivuri::pidcollect {
namespace fs = std::filesystem;
namespace {
using collection::joint_names;
// ",a,b,c" for every element of every argument, in order.
template <class... T> void csv_values(std::ostream &stream, const T &...arrays) {
    auto values = [&](const auto &array) {
        for (const auto &v : array)
            stream << ',' << v;
    };
    (values(arrays), ...);
}
template <class T> void json_values(std::ostream &stream, const T &values) {
    const char *separator = "[";
    for (const auto &v : values) {
        stream << separator << v;
        separator = ",";
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
            require(t != head.load(std::memory_order_acquire),
                    "missing PID telemetry for drive row");
            const auto frame = queue[t & (capacity - 1)];
            require(frame.tick <= tick, "PID telemetry/drive clock mismatch");
            tail.store(t + 1, std::memory_order_release);
            if (frame.tick == tick)
                return frame;
            // A failed base-queue push may leave an orphan from an incomplete take.
        }
    }
    void segment(const collection::Segment &s) {
        if (!first_segment && s.id == last_segment)
            return;
        protocol << (first_segment ? "" : ",") << "{\"id\":" << s.id
                 << ",\"begin_ns\":" << s.begin_ns << ",\"end_ns_requested\":" << s.end_ns
                 << ",\"duration_requested_s\":" << s.duration
                 << ",\"kind\":" << json_quote(collection::kind_name(s.kind))
                 << ",\"goal_cell\":" << s.goal_cell << ",\"goal_q_rad\":";
        first_segment = false;
        last_segment = s.id;
        json_values(protocol, s.goal_q);
        protocol << ",\"velocity_xz_m_s\":";
        json_values(protocol, s.velocity);
        protocol << ",\"pitch_target_rad\":" << s.pitch << ",\"pid_rate_requested_hz\":";
        json_values(protocol, s.pid_rates);
        protocol << ",\"gains_requested\":";
        const char *separator = "[";
        for (const auto &g : s.gains) {
            protocol << separator << "{\"kp\":" << g.kp << ",\"ki\":" << g.ki << ",\"kd\":" << g.kd
                     << ",\"tau_s\":" << g.tau << '}';
            separator = ",";
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
        require(!fs::exists(path), "PID output already exists");
    p.csv.clear();
    p.protocol.clear();
    p.csv.open(p.files[0]);
    p.protocol.open(p.files[1]);
    for (auto *stream : {&p.csv, &p.protocol}) {
        stream->exceptions(std::ios::failbit | std::ios::badbit);
        *stream << std::setprecision(17);
    }
    p.protocol << p.header;
    p.first_segment = true;
    p.last_segment = 0;
    p.csv << "tick,post_write_ns,imu_epoch,control_source,active,controller_valid,reason,"
             "segment_id,segment_kind,coverage_cell,recovery";
    for (const auto *prefix : {"task", "task_ref"})
        for (const auto *coordinate : {"x_m", "z_m", "pitch_rad"})
            p.csv << ',' << prefix << '_' << coordinate;
    for (const auto *prefix :
         {"joint_ref_rad", "pid_rate_requested_hz", "pid_updated", "pid_dt_s", "kp_applied",
          "ki_applied", "kd_applied", "tau_applied_s", "assist_requested", "excitation_requested",
          "command_before_guard", "guard_scale", "guard_predicted_q_rad", "guard_limited"})
        for (const auto *joint : joint_names)
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
    csv_values(p.csv, o.measured.pose, o.reference, o.joint_reference,
               f.active ? s.pid_rates : collection::Angles{}, o.pid_updated, o.pid_dt);
    for (auto term : {&collection::Gains::kp, &collection::Gains::ki, &collection::Gains::kd,
                      &collection::Gains::tau})
        for (const auto &g : o.applied_gains)
            p.csv << ',' << g.*term;
    csv_values(p.csv, o.assist, o.excitation, o.command, f.guard.scale, f.guard.predicted_q,
               f.guard.limited);
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
