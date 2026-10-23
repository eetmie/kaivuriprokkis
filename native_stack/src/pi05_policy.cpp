#include "kaivuri/pi05_policy.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <ctime>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/un.h>

namespace kaivuri::pi05 {
std::vector<double> normalize(const std::vector<float> &x, const Quantiles &q) {
    require(q.q01.size() >= x.size() && q.q99.size() >= x.size(), "state quantiles too short");
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = (double(x[i]) - q.q01[i]) / (q.q99[i] - q.q01[i] + 1e-6) * 2.0 - 1.0;
    return out;
}
double unnormalize(float x, double q01, double q99) {
    // numpy: float32 array + Python scalar stays float32; the float64 stats promote after.
    return double((x + 1.f) / 2.f) * (q99 - q01 + 1e-6) + q01;
}
int state_bin(double x) {
    require(!std::isnan(x), "NaN state");
    // Edges -1 + k/128 are exact in binary, as numpy's linspace produces them.
    auto edge = [](int k) { return -1.0 + k * 0.0078125; };
    if (x < edge(0))
        return -1;
    int k = int(std::clamp(std::floor((x + 1.0) * 128.0), 0.0, 255.0));
    while (k < 255 && edge(k + 1) <= x)
        ++k;
    while (k > 0 && edge(k) > x)
        --k;
    return k;
}
namespace {
struct Taps {
    std::vector<int> first, count;
    std::vector<int32_t> weights; // count per output, padded to ksize
    int ksize = 0;
};
constexpr int precision_bits = 32 - 8 - 2;
// Pillow Resample.c precompute_coeffs + normalize_coeffs_8bpc for the bilinear filter.
Taps pil_taps(int in_size, int out_size) {
    const double scale = double(in_size) / out_size, filterscale = std::max(scale, 1.0),
                 support = filterscale;
    Taps t;
    t.ksize = int(std::ceil(support)) * 2 + 1;
    t.weights.assign(size_t(out_size) * t.ksize, 0);
    std::vector<double> w(t.ksize);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale, ss = 1.0 / filterscale;
        int xmin = std::max(int(center - support + 0.5), 0);
        int xmax = std::min(int(center + support + 0.5), in_size) - xmin;
        double ww = 0;
        for (int x = 0; x < xmax; ++x) {
            double d = std::abs((x + xmin - center + 0.5) * ss);
            w[x] = d < 1.0 ? 1.0 - d : 0.0;
            ww += w[x];
        }
        for (int x = 0; x < xmax; ++x) {
            double v = ww != 0.0 ? w[x] / ww : w[x];
            t.weights[size_t(xx) * t.ksize + x] = v < 0 ? int32_t(-0.5 + v * (1 << precision_bits))
                                                        : int32_t(0.5 + v * (1 << precision_bits));
        }
        t.first.push_back(xmin);
        t.count.push_back(xmax);
    }
    return t;
}
uint8_t clip8(int64_t v) {
    if (v >= (int64_t(1) << precision_bits << 8))
        return 255;
    return v <= 0 ? 0 : uint8_t(v >> precision_bits);
}
std::vector<int64_t> ids(const YAML::Node &node, const std::string &what) {
    require(node && node.IsSequence() && node.size() > 0, "bundle " + what + " token ids");
    std::vector<int64_t> out;
    for (auto v : node)
        out.push_back(v.as<int64_t>());
    return out;
}
Quantiles quantiles(const YAML::Node &stats, size_t dims, const std::string &what) {
    require(stats && stats["q01"] && stats["q99"], "norm_stats." + what + " lacks q01/q99");
    Quantiles q;
    for (auto v : stats["q01"])
        q.q01.push_back(v.as<double>());
    for (auto v : stats["q99"])
        q.q99.push_back(v.as<double>());
    require(q.q01.size() == q.q99.size() && q.q01.size() >= dims,
            "norm_stats." + what + " quantiles shorter than the bundle dimension");
    q.q01.resize(dims);
    q.q99.resize(dims);
    for (size_t i = 0; i < dims; ++i)
        require(std::isfinite(q.q01[i]) && std::isfinite(q.q99[i]) && q.q99[i] >= q.q01[i],
                "invalid norm_stats." + what + " quantiles");
    return q;
}
// Single centered row: the arming hold before the first chunk (observation 0).
Plan centered_hold() {
    return Plan{{Commands{}}, 30, 0, 0};
}
} // namespace
std::vector<uint8_t> pil_bilinear(const uint8_t *input, int width, int height, size_t row_stride,
                                  int pixel_step, int out_width, int out_height) {
    require(input && width > 0 && height > 0 && out_width > 0 && out_height > 0 && pixel_step > 0 &&
                row_stride >= size_t(width) * pixel_step,
            "invalid resize");
    auto h = pil_taps(width, out_width), v = pil_taps(height, out_height);
    std::vector<uint8_t> temp(size_t(height) * out_width), out(size_t(out_width) * out_height);
    for (int y = 0; y < height; ++y)
        for (int xx = 0; xx < out_width; ++xx) {
            int32_t sum = 1 << (precision_bits - 1);
            const int32_t *k = &h.weights[size_t(xx) * h.ksize];
            const uint8_t *row = input + size_t(y) * row_stride;
            for (int x = 0; x < h.count[xx]; ++x)
                sum += int32_t(row[size_t(x + h.first[xx]) * pixel_step]) * k[x];
            temp[size_t(y) * out_width + xx] = clip8(sum);
        }
    for (int yy = 0; yy < out_height; ++yy) {
        const int32_t *k = &v.weights[size_t(yy) * v.ksize];
        for (int xx = 0; xx < out_width; ++xx) {
            int32_t sum = 1 << (precision_bits - 1);
            for (int y = 0; y < v.count[yy]; ++y)
                sum += int32_t(temp[size_t(y + v.first[yy]) * out_width + xx]) * k[y];
            out[size_t(yy) * out_width + xx] = clip8(sum);
        }
    }
    return out;
}
std::vector<int64_t> StatePrompt::compose(const std::string &task,
                                          const std::vector<int> &bins) const {
    auto head = heads.find(task);
    require(head != heads.end(), "task has no token head in this bundle: " + task);
    std::vector<int64_t> ids = head->second;
    for (int bin : bins) {
        require(bin >= -1 && bin <= 255, "state bin outside -1..255");
        ids.push_back(bin < 0 ? neg : sep);
        for (char c : std::to_string(std::abs(bin)))
            ids.push_back(digits[c - '0']);
    }
    ids.insert(ids.end(), tail.begin(), tail.end());
    return ids;
}
size_t StatePrompt::longest(size_t state_dim) const {
    size_t head = 0;
    for (auto &entry : heads)
        head = std::max(head, entry.second.size());
    return head + 4 * state_dim + tail.size();
}
int joint_index(const std::string &name) {
    static const std::array<const char *, 4> names{"slew", "lift", "tilt", "scoop"};
    for (int i = 0; i < 4; ++i)
        if (name == names[i])
            return i;
    throw std::runtime_error("unknown state joint " + name + " (slew, lift, tilt, scoop)");
}
std::vector<int> Contract::bins(const std::vector<float> &state) const {
    require(int(state.size()) == state_dim, "state dimension mismatch");
    for (float v : state)
        require(std::isfinite(v), "nonfinite state");
    std::vector<int> out;
    for (double v : normalize(state, this->state))
        out.push_back(state_bin(v));
    return out;
}
std::vector<int64_t> Contract::tokens(const std::string &task,
                                      const std::vector<float> &state) const {
    return prompt.compose(task, bins(state));
}
Contract load_contract(const YAML::Node &b, const YAML::Node &norm) {
    require(b["policy"].as<std::string>() == "pi05_openpi_state", "not a pi05_openpi_state bundle");
    Contract c;
    c.state_dim = b["state_dim"].as<int>();
    c.robot_dim = b["robot_action_dim"].as<int>();
    require(c.state_dim > 0 && c.state_dim <= 4, "state_dim must be 1..4 joint angles");
    require((c.robot_dim == 4 || c.robot_dim == 6) && c.robot_dim <= b["action_dim"].as<int>(),
            "robot_action_dim must be 4 or 6");
    auto tok = b["tokenizer"];
    require(bool(tok), "bundle lacks tokenizer pieces");
    c.prompt.sep = tok["sep"].as<int64_t>();
    c.prompt.neg = tok["neg"].as<int64_t>();
    for (int d = 0; d < 10; ++d)
        c.prompt.digits[d] = tok["pieces"][std::to_string(d)].as<int64_t>();
    c.prompt.tail = ids(tok["tail"], "tail");
    require(b["tasks"] && b["tasks"].IsMap() && b["tasks"].size() > 0, "bundle lacks tasks");
    for (auto entry : b["tasks"])
        c.prompt.heads[entry.first.as<std::string>()] =
            ids(entry.second["head"], "task head " + entry.first.as<std::string>());
    auto k = b["kaivuri"];
    if (k && k["state"] && k["state"]["names"])
        for (auto name : k["state"]["names"])
            c.state_names.push_back(name.as<std::string>());
    else
        c.state_names = {"lift", "tilt", "scoop"};
    require(int(c.state_names.size()) == c.state_dim, "state names do not match state_dim");
    for (auto &name : c.state_names)
        c.state_joints.push_back(joint_index(name));
    if (k && k["action"] && k["action"]["names"]) {
        static const std::array<const char *, 6> order{"slew",  "lift",   "tilt",
                                                       "scoop", "trackL", "trackR"};
        auto names = k["action"]["names"];
        require(int(names.size()) == c.robot_dim, "action names do not match robot_action_dim");
        for (int i = 0; i < c.robot_dim; ++i)
            require(names[i].as<std::string>() == order[i],
                    "action order must be slew, lift, tilt, scoop(, trackL, trackR)");
    }
    if (k && k["fps"])
        c.fps = k["fps"].as<double>();
    require(c.fps >= 0 && c.fps <= 500, "kaivuri.fps outside 0..500");
    auto stats = norm["norm_stats"];
    require(bool(stats), "norm_stats.json lacks norm_stats");
    c.state = quantiles(stats["state"], c.state_dim, "state");
    c.actions = quantiles(stats["actions"], c.robot_dim, "actions");
    size_t longest = c.prompt.longest(c.state_dim);
    require(longest <= b["prompt_capacity"].as<size_t>() &&
                longest + 512 <= b["prefix_tokens"].as<size_t>(),
            "longest state prompt exceeds the bundle's prompt capacity");
    return c;
}
RobotState robot_state(const std::string &json, const std::vector<int> &joints, uint64_t now,
                       uint64_t max_age_ns) {
    auto s = YAML::Load(json);
    RobotState r;
    r.safety_state = s["state"].as<std::string>();
    r.reason = s["reason"].as<std::string>();
    r.live_pwm = s["live_pwm"].as<bool>();
    r.enable_slew = s["enable_slew"].as<bool>();
    r.enable_tracks = s["enable_tracks"].as<bool>();
    require(s["imu_healthy"].as<bool>() && s["fk_valid"].as<bool>() &&
                !s["firmware_blocked"].as<bool>(),
            "controller reports unhealthy IMUs/FK; no state");
    r.imu_ns = s["imu_host_monotonic_ns"].as<uint64_t>();
    require(r.imu_ns > 0 && r.imu_ns <= now + 10000000ULL, "IMU timestamp in the future");
    require(now <= r.imu_ns || now - r.imu_ns <= max_age_ns, "stale IMU state");
    auto angles = s["angles_rad"];
    require(angles.size() == 4, "angles_rad must have four joints");
    // np.rad2deg on float32 multiplies by float32(180) / float32(pi).
    const float to_degrees = 180.f / float(pi);
    for (int j : joints) {
        float radians = float(angles[j].as<double>());
        require(std::isfinite(radians), "nonfinite joint angle");
        r.degrees.push_back(radians * to_degrees);
    }
    return r;
}
uint64_t Plan::end_ns() const {
    if (rows.size() < 2)
        return t0_ns;
    return t0_ns + uint64_t(std::llround((rows.size() - 1) * 1e9 / fps));
}
Plan make_plan(const std::vector<double> &robot, int horizon, int robot_dim, double fps,
               uint64_t t0_ns, uint64_t observation_ns, const ActionGate &gate, int &clamped) {
    require(horizon > 0 && size_t(horizon) <= max_steps && (robot_dim == 4 || robot_dim == 6) &&
                robot.size() == size_t(horizon) * robot_dim,
            "chunk shape");
    require(std::isfinite(fps) && fps > 0 && fps <= 500, "plan fps");
    require(gate.tolerance >= 0, "action tolerance must be >= 0");
    Plan plan;
    plan.fps = fps;
    plan.t0_ns = t0_ns;
    plan.observation_ns = observation_ns;
    clamped = 0;
    for (int r = 0; r < horizon; ++r) {
        Commands row{};
        for (int j = 0; j < robot_dim; ++j) {
            double v = robot[size_t(r) * robot_dim + j];
            require(std::isfinite(v), "nonfinite policy action");
            require(std::abs(v) <= 1 + gate.tolerance,
                    "policy action " + std::to_string(v) + " outside [-1,1] beyond tolerance");
            if (std::abs(v) > 1) {
                v = std::clamp(v, -1.0, 1.0);
                ++clamped;
            }
            bool enabled = j == 0 ? gate.slew : j >= 4 ? gate.tracks : true;
            row[j] = enabled ? v : 0;
        }
        plan.rows.push_back(row);
    }
    return plan;
}
std::vector<uint8_t> encode(Operation op, uint32_t flags, uint64_t sequence, uint64_t sent_ns,
                            const Plan *plan) {
    const size_t count = op == Operation::Drive && plan ? plan->rows.size() : 0;
    require(op != Operation::Drive || (count > 0 && count <= max_steps), "drive needs 1..128 rows");
    std::vector<uint8_t> data(wire_header + count * 24);
    std::memcpy(data.data(), "KVR1", 4);
    auto put = [&](size_t offset, auto value) {
        std::memcpy(data.data() + offset, &value, sizeof(value));
    };
    put(4, uint16_t(1));
    put(6, uint16_t(op));
    put(8, flags);
    put(12, sequence);
    put(20, sent_ns);
    put(28, uint32_t(count));
    put(32, float(count ? plan->fps : 0));
    put(36, uint64_t(count ? plan->t0_ns : 0));
    put(44, uint64_t(count ? plan->observation_ns : 0));
    for (size_t r = 0; r < count; ++r)
        for (int j = 0; j < 6; ++j) {
            float v = float(plan->rows[r][j]);
            require(std::isfinite(v) && std::abs(v) <= 1, "drive value outside [-1,1]");
            put(wire_header + r * 24 + j * 4, v);
        }
    return data;
}
void PlanSupervisor::hold(uint64_t now) {
    plan_ = centered_hold();
    holding_ = pending_ = true;
    hold_started_ = now;
}
std::string PlanSupervisor::offer(Plan plan, uint64_t now) {
    if (stopped_)
        return "producer stopped: " + stop_reason_;
    if (plan.rows.empty() || plan.rows.size() > max_steps || !(plan.fps > 0 && plan.fps <= 500))
        return "invalid plan shape/fps";
    if (!plan.observation_ns || !plan.t0_ns || plan.observation_ns > now + 10000000ULL ||
        plan.t0_ns > now + 10000000ULL)
        return "plan observation/start time missing or in the future";
    if (!holding_ && plan.observation_ns <= plan_.observation_ns)
        return "plan observation is not newer than the current plan's";
    if (now > plan.observation_ns && now - plan.observation_ns > cfg_.max_observation_age_ns)
        return "observation already too old";
    if (plan.end_ns() <= now + cfg_.end_margin_ns)
        return "plan already exhausted";
    plan_ = std::move(plan);
    holding_ = false;
    pending_ = true;
    return {};
}
PlanSupervisor::Decision PlanSupervisor::next(uint64_t now, std::string &reason) {
    if (!stopped_) {
        if (holding_ && now - hold_started_ > cfg_.first_plan_timeout_ns)
            stop_reason_ = "no policy chunk within the first-plan timeout";
        else if (!holding_ && !plan_.observation_ns)
            stop_reason_ = "no plan";
        else if (!holding_ && now > plan_.observation_ns &&
                 now - plan_.observation_ns > cfg_.max_observation_age_ns)
            stop_reason_ = "observation age limit (inference stalled)";
        else if (!holding_ && now + cfg_.end_margin_ns >= plan_.end_ns())
            stop_reason_ = "plan exhausted (inference stalled)";
        stopped_ = !stop_reason_.empty();
    }
    if (stopped_) {
        reason = stop_reason_;
        return Decision::Stop;
    }
    return pending_ || now - last_sent_ >= cfg_.renew_ns ? Decision::Send : Decision::Idle;
}
ControlLink::ControlLink(const std::string &path)
    : fd_(socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0)) {
    require(bool(fd_), "control socket failed");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    require(path.size() < sizeof(address.sun_path), "cannot connect to controller at " + path);
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    require(connect(fd_.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0,
            "cannot connect to controller at " + path);
}
void ControlLink::send(Operation op, uint32_t flags, const Plan *plan) {
    auto data = encode(op, flags, ++sequence_, monotonic_ns(), plan);
    require(::send(fd_.get(), data.data(), data.size(), MSG_DONTWAIT | MSG_NOSIGNAL) ==
                ssize_t(data.size()),
            "control send failed");
}
std::string ControlLink::status() {
    send(Operation::Status);
    pollfd p{fd_.get(), POLLIN, 0};
    int ready;
    do {
        ready = poll(&p, 1, 100);
    } while (ready < 0 && errno == EINTR);
    require(ready == 1, "controller status timeout");
    char bytes[8192];
    ssize_t count;
    do {
        count = recv(fd_.get(), bytes, sizeof(bytes), MSG_TRUNC);
    } while (count < 0 && errno == EINTR);
    require(count > 0 && count <= ssize_t(sizeof(bytes)), "invalid/truncated status");
    return std::string(bytes, count);
}
Producer::Producer(ProducerOptions options)
    : options_(std::move(options)), link_(options_.socket), supervisor_(options_.supervisor) {
    auto s = YAML::Load(link_.status());
    const auto state = s["state"].as<std::string>();
    require(state == "disarmed", "controller must start disarmed (is " + state + ": " +
                                     s["reason"].as<std::string>() + "); disarm it explicitly");
    require(!s["live_pwm"].as<bool>() || options_.allow_live,
            "controller writes live PWM; driving it requires --allow-live");
    require(!options_.gate.slew || s["enable_slew"].as<bool>(),
            "--enable-slew also requires kaivuri-control --enable-slew");
    require(!options_.gate.tracks || s["enable_tracks"].as<bool>(),
            "--enable-tracks also requires kaivuri-control --enable-tracks");
}
Producer::~Producer() {
    stop("shutdown");
    try {
        link_.status(); // Drain the disarm before closing the owning connection.
    } catch (...) {
    }
}
void Producer::arm(double timeout_s) {
    require(!thread_.joinable(), "producer already armed once; restart to re-arm");
    const uint32_t pump = options_.pump ? pump_flag : 0;
    const uint64_t begin = monotonic_ns();
    supervisor_.hold(begin);
    const Plan centered = centered_hold();
    for (;;) {
        link_.send(Operation::Drive, deadman_flag | arm_flag | pump, &centered);
        auto s = YAML::Load(link_.status());
        const auto state = s["state"].as<std::string>();
        if (state == "armed")
            break;
        if (state == "fault" || monotonic_ns() - begin > uint64_t(timeout_s * 1e9)) {
            link_.send(Operation::Disarm);
            throw std::runtime_error("arming failed: " + state + " " +
                                     s["reason"].as<std::string>());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    supervisor_.sent(monotonic_ns());
    running_ = true;
    thread_ = std::thread([this] { loop(); });
}
bool Producer::post(Plan plan, std::string &reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        reason = "producer stopped: " + reason_;
        return false;
    }
    reason = supervisor_.offer(std::move(plan), monotonic_ns());
    wake_.notify_all();
    return reason.empty();
}
void Producer::loop() {
    const uint32_t pump = options_.pump ? pump_flag : 0;
    uint64_t last_status = 0, last_send = monotonic_ns();
    try {
        for (;;) {
            Plan send;
            bool fresh = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait_for(lock, std::chrono::duration<double>(options_.poll_s));
                if (stopping_)
                    return;
                std::string why;
                const uint64_t now = monotonic_ns();
                auto decision = supervisor_.next(now, why);
                if (decision == PlanSupervisor::Decision::Stop) {
                    lock.unlock();
                    finish(why);
                    return;
                }
                if (decision == PlanSupervisor::Decision::Send) {
                    send = supervisor_.plan();
                    fresh = supervisor_.fresh();
                    supervisor_.sent(now);
                }
            }
            if (!send.rows.empty()) {
                // Renewals resend the same rows, t0 and observation time; only the
                // sequence and send time change.
                link_.send(Operation::Drive, deadman_flag | pump, &send);
                const uint64_t now = monotonic_ns();
                std::lock_guard<std::mutex> lock(mutex_);
                counters_.max_send_gap_ms =
                    std::max(counters_.max_send_gap_ms, (now - last_send) / 1e6);
                last_send = now;
                if (send.observation_ns == 0)
                    ++counters_.holds;
                else if (fresh)
                    ++counters_.plans;
                else
                    ++counters_.renewals;
            }
            const uint64_t now = monotonic_ns();
            if (now - last_status >= uint64_t(options_.status_s * 1e9)) {
                last_status = now;
                auto s = YAML::Load(link_.status());
                const auto state = s["state"].as<std::string>();
                if (state != "armed") {
                    finish("controller " + state + ": " + s["reason"].as<std::string>());
                    return;
                }
            }
        }
    } catch (const std::exception &e) {
        finish(std::string("producer error: ") + e.what());
    }
}
void Producer::finish(const std::string &reason) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (reason_.empty())
            reason_ = reason;
        running_ = false;
        if (disarmed_)
            return;
        disarmed_ = true;
    }
    try {
        link_.send(Operation::Disarm);
    } catch (...) {
    }
}
void Producer::stop(const std::string &reason) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        if (reason_.empty() && running_)
            reason_ = reason;
    }
    wake_.notify_all();
    if (thread_.joinable())
        thread_.join();
    finish(reason);
}
bool Producer::running() {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}
std::string Producer::stop_reason() {
    std::lock_guard<std::mutex> lock(mutex_);
    return reason_;
}
ProducerCounters Producer::counters() {
    std::lock_guard<std::mutex> lock(mutex_);
    return counters_;
}
} // namespace kaivuri::pi05
