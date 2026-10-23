#include "kaivuri/ik_collection.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace kaivuri::collection {
namespace {
bool finite(const Angles &a) {
    return std::all_of(a.begin(), a.end(), [](double v) { return std::isfinite(v); });
}
Angles solve(Matrix a, Angles b) {
    for (unsigned i = 0; i < 3; ++i) {
        unsigned pivot = i;
        for (unsigned j = i + 1; j < 3; ++j)
            if (std::abs(a[j][i]) > std::abs(a[pivot][i]))
                pivot = j;
        if (std::abs(a[pivot][i]) < 1e-15)
            throw std::runtime_error("singular DLS solve");
        std::swap(a[i], a[pivot]);
        std::swap(b[i], b[pivot]);
        for (unsigned j = i + 1; j < 3; ++j) {
            const double ratio = a[j][i] / a[i][i];
            for (unsigned k = i; k < 3; ++k)
                a[j][k] -= ratio * a[i][k];
            b[j] -= ratio * b[i];
        }
    }
    Angles x{};
    for (int i = 2; i >= 0; --i) {
        double v = b[i];
        for (unsigned j = unsigned(i) + 1; j < 3; ++j)
            v -= a[i][j] * x[j];
        x[i] = v / a[i][i];
    }
    return x;
}
void check_gains(const Gains &g) {
    for (double v : {g.kp, g.ki, g.kd, g.tau})
        if (!std::isfinite(v) || v < 0)
            throw std::invalid_argument("invalid PID gains");
}
} // namespace

Robot load_robot(const std::filesystem::path &profile) {
    Robot r;
    r.hardware = load_hardware_config(profile.string());
    const auto p = YAML::LoadFile((profile / "profile.yaml").string());
    const auto c = YAML::LoadFile((profile / p["control_config_file"].as<std::string>()).string());
    r.pitch_offset = c["robot"]["tool"]["pitch_offset_rad"]
                         ? c["robot"]["tool"]["pitch_offset_rad"].as<double>()
                         : 0;
    for (unsigned i = 0; i < 3; ++i) {
        const auto l = c["ik"]["joint_limits_relative"][i + 1];
        r.limits[i] = {l[0].as<double>() * pi / 180, l[1].as<double>() * pi / 180};
        const auto g = c["pid"]["joint" + std::to_string(i + 1)];
        r.gains[i] = {g["kp"].as<double>(), g["ki"].as<double>(), g["kd"].as<double>(), .1};
        check_gains(r.gains[i]);
    }
    return r;
}
Kinematics::Kinematics(Robot robot) : robot_(std::move(robot)) {
    if (!std::isfinite(robot_.pitch_offset))
        throw std::invalid_argument("invalid tool pitch");
    for (unsigned i = 0; i < 3; ++i) {
        const auto &axis = robot_.hardware.joints[i + 1].axis;
        if (std::abs(axis[0]) > 1e-10 || std::abs(axis[1] - 1) > 1e-10 || std::abs(axis[2]) > 1e-10)
            throw std::invalid_argument("collection requires aligned positive-Y arm joints");
        const auto &l = robot_.limits[i];
        if (!std::isfinite(l[0]) || !std::isfinite(l[1]) || l[0] >= l[1])
            throw std::invalid_argument("invalid collection joint limits");
    }
}
TaskState Kinematics::state(const Angles &q) const {
    if (!finite(q))
        throw std::invalid_argument("nonfinite joint angles");
    const auto fk = forward_kinematic_state(robot_.hardware, {0, q[0], q[1], q[2]});
    TaskState result;
    result.pose = {fk.upper.position[0], fk.upper.position[2],
                   wrap(fk.bucket_pitch + robot_.pitch_offset)};
    for (unsigned i = 0; i < 3; ++i) {
        result.jacobian[0][i] = fk.upper.position[2] - fk.joint_positions_upper[i + 1][2];
        result.jacobian[1][i] = -(fk.upper.position[0] - fk.joint_positions_upper[i + 1][0]);
        result.jacobian[2][i] = 1;
    }
    return result;
}
bool Kinematics::inside(const Angles &q, double margin) const {
    if (!finite(q) || !std::isfinite(margin) || margin < 0)
        return false;
    for (unsigned i = 0; i < 3; ++i)
        if (q[i] < robot_.limits[i][0] + margin || q[i] > robot_.limits[i][1] - margin)
            return false;
    return true;
}
IkResult Kinematics::inverse(const Angles &target, const Angles &seed, double margin) const {
    IkResult r;
    r.q = seed;
    if (!finite(target) || !inside(seed, margin))
        return r;
    const Angles weights{1, 1, .2};
    for (unsigned iteration = 0; iteration <= 20; ++iteration) {
        const auto s = state(r.q);
        Angles error{target[0] - s.pose[0], target[1] - s.pose[1], wrap(target[2] - s.pose[2])};
        r.position_error = std::hypot(error[0], error[1]);
        r.pitch_error = std::abs(error[2]);
        r.iterations = iteration;
        r.reached = r.position_error < .0005 && r.pitch_error < .005;
        if (r.reached || iteration == 20)
            break;
        Matrix normal{};
        Angles rhs{};
        for (unsigned i = 0; i < 3; ++i) {
            normal[i][i] = 1e-6;
            for (unsigned k = 0; k < 3; ++k) {
                rhs[i] += s.jacobian[k][i] * weights[k] * weights[k] * error[k];
                for (unsigned j = 0; j < 3; ++j)
                    normal[i][j] += s.jacobian[k][i] * s.jacobian[k][j] * weights[k] * weights[k];
            }
        }
        const auto dq = solve(normal, rhs);
        for (unsigned i = 0; i < 3; ++i)
            r.q[i] = std::clamp(r.q[i] + std::clamp(dq[i], -.1, .1), robot_.limits[i][0] + margin,
                                robot_.limits[i][1] - margin);
    }
    return r;
}

Pid::Pid(Gains gains, double limit) : limit_(limit) {
    if (!std::isfinite(limit) || limit <= 0 || limit > 1)
        throw std::invalid_argument("invalid PID limit");
    configure(gains, false);
}
void Pid::configure(Gains gains, bool preserve) {
    check_gains(gains);
    if (preserve && gains.kp == gains_.kp && gains.ki == gains_.ki && gains.kd == gains_.kd &&
        gains.tau == gains_.tau)
        return;
    const double contribution = gains_.ki * integral_;
    gains_ = gains;
    integral_ = preserve && gains.ki != 0
                    ? std::clamp(contribution / gains.ki, -limit_ / gains.ki, limit_ / gains.ki)
                    : 0;
}
void Pid::reset() {
    integral_ = last_ = derivative_ = 0;
    have_last_ = false;
}
double Pid::compute(double setpoint, double measured, double dt, double excitation,
                    bool blocked_positive, bool blocked_negative) {
    for (double v : {setpoint, measured, dt, excitation})
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite PID input");
    if (dt < 0)
        throw std::invalid_argument("negative PID dt");
    dt = std::max(dt, .001);
    const double error = setpoint - measured;
    const double raw_d = have_last_ ? -(measured - last_) / dt : 0;
    derivative_ =
        gains_.tau == 0 ? raw_d : derivative_ + dt / (gains_.tau + dt) * (raw_d - derivative_);
    const double bound =
        gains_.ki != 0 ? limit_ / gains_.ki : std::numeric_limits<double>::infinity();
    const double tentative = std::clamp(integral_ + error * dt, -bound, bound);
    const double unsaturated = gains_.kp * error + gains_.ki * tentative + gains_.kd * derivative_;
    const double output = std::clamp(unsaturated, -limit_, limit_);
    const bool blocked =
        (error > 0 && ((unsaturated > limit_) || output + excitation > 1 || blocked_positive)) ||
        (error < 0 && ((unsaturated < -limit_) || output + excitation < -1 || blocked_negative));
    if (!blocked)
        integral_ = tentative;
    last_ = measured;
    have_last_ = true;
    return output;
}

JointGuard::JointGuard(const Robot &robot, GuardOptions options)
    : limits_(robot.limits), options_(options) {
    if (!std::isfinite(options.margin) || options.margin < 0 || !std::isfinite(options.taper) ||
        options.taper <= 0 || !std::isfinite(options.velocity_lookahead) ||
        options.velocity_lookahead < 0)
        throw std::invalid_argument("invalid joint guard parameters");
    for (unsigned i = 0; i < 3; ++i) {
        const auto &l = limits_[i];
        if (!std::isfinite(l[0]) || !std::isfinite(l[1]) ||
            l[1] - l[0] <= 2 * (options.margin + options.taper) ||
            (options.command_sign[i] != 1 && options.command_sign[i] != -1))
            throw std::invalid_argument("invalid guard joint range or command sign");
    }
}
GuardResult JointGuard::apply(const Angles &q, const Angles &velocity, const Angles &command,
                              bool fresh) const {
    GuardResult result;
    if (!fresh || !finite(q) || !finite(velocity) || !finite(command))
        return result;
    result.valid = true;
    result.reason = "ok";
    for (unsigned i = 0; i < 3; ++i) {
        const double bounded = std::clamp(command[i], -1., 1.);
        const double direction = bounded * options_.command_sign[i];
        // Only velocity toward this command's boundary consumes headroom.
        const double projected_v = direction > 0   ? std::max(0., velocity[i])
                                   : direction < 0 ? std::min(0., velocity[i])
                                                   : 0;
        result.predicted_q[i] = q[i] + projected_v * options_.velocity_lookahead;
        const double distance =
            direction > 0   ? limits_[i][1] - options_.margin - result.predicted_q[i]
            : direction < 0 ? result.predicted_q[i] - limits_[i][0] - options_.margin
                            : options_.taper;
        result.scale[i] = std::clamp(distance / options_.taper, 0., 1.);
        result.command[i] = bounded * result.scale[i];
        result.limited[i] = result.command[i] != command[i];
        if (result.limited[i])
            result.reason = "joint_limit_guard";
    }
    for (unsigned i = 0; i < 3; ++i)
        if (q[i] < limits_[i][0] + options_.margin || q[i] > limits_[i][1] - options_.margin)
            result.reason = "inward_recovery";
    return result;
}
double guarded_valve_pulse(const Valve &v, double command, bool limited, double now,
                           RampState &state) {
    if (!std::isfinite(command) || !std::isfinite(now))
        throw std::invalid_argument("nonfinite guarded PWM input");
    if (command == 0) {
        state = {v.center, now, true};
        return v.center;
    }
    const double direction = command * v.direction;
    if (limited || (state.initialized && (state.pulse - v.center) * direction < 0))
        state = {};
    double pulse;
    if (limited) {
        auto bounded = v;
        bounded.ramp = bounded.dither = false;
        pulse = valve_pulse(bounded, command, now, state);
    } else
        pulse = valve_pulse(v, command, now, state);
    // Neither old ramp state nor dither may reverse the guarded direction.
    return direction > 0 ? std::max(v.center, pulse) : std::min(v.center, pulse);
}

const char *kind_name(Kind k) {
    constexpr const char *names[]{"free", "coverage", "reverse", "pitch", "hold", "recovery"};
    return names[static_cast<unsigned>(k)];
}
RandomCollection::RandomCollection(Robot robot, Options options)
    : kin_(robot), options_(std::move(options)), rng_(options_.seed),
      pid_{Pid(robot.gains[0], options_.valve_limit), Pid(robot.gains[1], options_.valve_limit),
           Pid(robot.gains[2], options_.valve_limit)} {
    const auto &o = options_;
    for (double v : {o.min_seconds, o.max_seconds, o.min_speed, o.max_speed, o.max_pitch_change,
                     o.pitch_rate, o.gain_min, o.gain_max, o.joint_margin, o.max_lead,
                     o.coverage_bin, o.goal_seconds, o.max_gap, o.recovery_limit, o.recovery_inset})
        if (!std::isfinite(v) || v <= 0)
            throw std::invalid_argument("invalid random collection options");
    if (o.min_seconds > o.max_seconds || o.min_speed > o.max_speed || o.gain_min > o.gain_max ||
        o.max_seconds > 86400 || o.max_pitch_change > pi || o.max_gap < .01 ||
        o.recovery_limit > o.valve_limit ||
        std::ceil(o.min_seconds * 100) > std::floor(o.max_seconds * 100) || o.pid_rates.empty())
        throw std::invalid_argument("invalid random collection ranges");
    for (double hz : o.pid_rates)
        if (!std::isfinite(hz) || hz <= 0 || hz > 100 || !std::isfinite(100 / hz) ||
            1e9 / hz >= double(std::numeric_limits<int64_t>::max()) ||
            std::abs(100 / hz - std::round(100 / hz)) > 1e-9)
            throw std::invalid_argument("PID rates must divide the fixed 100 Hz output clock");
    for (double sign : o.command_sign)
        if (sign != 1 && sign != -1)
            throw std::invalid_argument("invalid PID command growth sign");
    for (const auto &g : robot.gains)
        check_gains({g.kp * o.gain_max, g.ki * o.gain_max, g.kd * o.gain_max, g.tau});
    size_t size = 1;
    for (unsigned i = 0; i < 3; ++i) {
        const auto &l = robot.limits[i];
        if (l[1] - l[0] <= 2 * (o.joint_margin + o.recovery_inset))
            throw std::invalid_argument("joint margin and recovery inset consume range");
        const double count = std::ceil((l[1] - l[0]) / o.coverage_bin);
        if (!std::isfinite(count) || count > 1000000)
            throw std::invalid_argument("coverage grid too large");
        bins_[i] = size_t(count);
        if (bins_[i] > 1000000 / size)
            throw std::invalid_argument("coverage grid too large");
        size *= bins_[i];
        // Count bins with nonzero overlap with the working interval. Exact
        // boundary points belong to its nearest interior bin for this statistic.
        const auto first = size_t(std::floor(o.joint_margin / o.coverage_bin + 1e-10));
        const auto end = size_t(std::ceil((l[1] - l[0] - o.joint_margin) / o.coverage_bin - 1e-10));
        working_begin_[i] = first;
        working_end_[i] = end;
        working_cells_ *= end - first;
    }
    dwell_.resize(size);
    moving_.resize(size);
    working_visited_.resize(size);
}
double RandomCollection::uniform(double lo, double hi) {
    uint64_t z = (rng_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return lo + (hi - lo) * (double(z >> 11) + .5) * 0x1.0p-53;
}
double RandomCollection::logarithmic(double lo, double hi) {
    return std::exp(uniform(std::log(lo), std::log(hi)));
}
void RandomCollection::reset() {
    initialized_ = have_goal_ = recovering_ = false;
    reset_pids(options_.valve_limit);
}
void RandomCollection::reset_pids(double limit) {
    held_ = {};
    have_pid_.fill(false);
    block_positive_.fill(false);
    block_negative_.fill(false);
    for (unsigned i = 0; i < 3; ++i)
        pid_[i] = Pid(kin_.robot().gains[i], limit);
}
void RandomCollection::feedback_limited(const Angles &requested, const Angles &actual) {
    if (!finite(requested) || !finite(actual))
        throw std::invalid_argument("nonfinite guard feedback");
    for (unsigned i = 0; i < 3; ++i) {
        const double lost = (requested[i] - actual[i]) * options_.command_sign[i];
        block_positive_[i] = block_positive_[i] || lost > 1e-12;
        block_negative_[i] = block_negative_[i] || lost < -1e-12;
    }
}
size_t RandomCollection::cell(const Angles &q) const {
    if (!finite(q))
        throw std::invalid_argument("nonfinite coverage pose");
    size_t result = 0;
    for (unsigned i = 0; i < 3; ++i) {
        const double offset = (q[i] - kin_.robot().limits[i][0]) / options_.coverage_bin;
        const size_t index = size_t(std::clamp(std::floor(offset), 0., double(bins_[i] - 1)));
        result = result * bins_[i] + index;
    }
    return result;
}
void RandomCollection::choose_goal(uint64_t now, const Angles &q) {
    const auto pose = kin_.state(q).pose;
    double best = -1;
    for (unsigned attempt = 0; attempt < 64; ++attempt) {
        Angles candidate{};
        for (unsigned i = 0; i < 3; ++i)
            candidate[i] = uniform(kin_.robot().limits[i][0] + options_.joint_margin,
                                   kin_.robot().limits[i][1] - options_.joint_margin);
        const auto target = kin_.state(candidate).pose;
        const double distance = std::hypot(target[0] - pose[0], target[1] - pose[1]);
        const double score = 1 / ((1 + moving_[cell(candidate)]) * (1 + distance));
        if (score > best) {
            best = score;
            goal_q_ = candidate;
        }
    }
    have_goal_ = true;
    goal_begin_ = now;
}
void RandomCollection::choose_segment(uint64_t now, const Angles &q) {
    const auto pose = kin_.state(q).pose;
    if (!have_goal_ || double(now - goal_begin_) / 1e9 >= options_.goal_seconds ||
        cell(q) == cell(goal_q_))
        choose_goal(now, q);
    const auto previous_velocity = segment_.velocity;
    segment_ = {};
    segment_.id = ++segment_id_;
    segment_.begin_ns = now;
    segment_.goal_cell = cell(goal_q_);
    segment_.goal_q = goal_q_;
    const double draw = uniform(0, 1);
    segment_.kind = draw < .40   ? Kind::Coverage
                    : draw < .75 ? Kind::Free
                    : draw < .85 ? Kind::Reverse
                    : draw < .95 ? Kind::Pitch
                                 : Kind::Hold;
    double seconds = logarithmic(options_.min_seconds, options_.max_seconds);
    if (segment_.kind == Kind::Coverage)
        seconds = uniform(std::max(options_.min_seconds, options_.max_seconds * .66),
                          options_.max_seconds);
    const auto ticks =
        uint64_t(std::clamp(std::ceil(seconds * 100), std::ceil(options_.min_seconds * 100),
                            std::floor(options_.max_seconds * 100)));
    segment_.duration = double(ticks) / 100;
    segment_.end_ns = now + ticks * 10000000;
    const double direction = uniform(-pi, pi),
                 speed = uniform(options_.min_speed, options_.max_speed);
    segment_.velocity = {speed * std::cos(direction), speed * std::sin(direction)};
    segment_.pitch = wrap(pose[2] + uniform(-options_.max_pitch_change, options_.max_pitch_change));
    if (segment_.kind == Kind::Coverage) {
        const auto goal = kin_.state(goal_q_).pose;
        const double dx = goal[0] - pose[0], dz = goal[1] - pose[1], length = std::hypot(dx, dz);
        segment_.velocity = length > 1e-9
                                ? std::array<double, 2>{speed * dx / length, speed * dz / length}
                                : std::array<double, 2>{};
        segment_.pitch =
            wrap(pose[2] + std::clamp(wrap(goal[2] - pose[2]), -options_.max_pitch_change,
                                      options_.max_pitch_change));
    } else if (segment_.kind == Kind::Reverse) {
        const double length = std::hypot(previous_velocity[0], previous_velocity[1]);
        if (length > 1e-9)
            segment_.velocity = {-speed * previous_velocity[0] / length,
                                 -speed * previous_velocity[1] / length};
    } else if (segment_.kind == Kind::Pitch || segment_.kind == Kind::Hold) {
        segment_.velocity = {};
        if (segment_.kind == Kind::Hold)
            segment_.pitch = pose[2];
    }
    for (unsigned i = 0; i < 3; ++i) {
        const auto index = std::min(size_t(uniform(0, double(options_.pid_rates.size()))),
                                    options_.pid_rates.size() - 1);
        segment_.pid_rates[i] = options_.pid_rates[index];
        const auto &g = kin_.robot().gains[i];
        segment_.gains[i] = {g.kp * uniform(options_.gain_min, options_.gain_max),
                             g.ki * uniform(options_.gain_min, options_.gain_max),
                             g.kd * uniform(options_.gain_min, options_.gain_max), g.tau};
    }
}
void RandomCollection::begin_recovery(uint64_t now, const Angles &q) {
    recovering_ = true;
    have_goal_ = false;
    reset_pids(options_.recovery_limit);
    segment_ = {};
    segment_.id = ++segment_id_;
    segment_.begin_ns = now;
    segment_.kind = Kind::Recovery;
    // A recovery has no random duration: it ends on measured re-entry. Fixed
    // profile gains at 100 Hz avoid randomizing the boundary recovery response.
    for (unsigned i = 0; i < 3; ++i) {
        const auto &l = kin_.robot().limits[i];
        segment_.goal_q[i] =
            std::clamp(q[i], l[0] + options_.joint_margin + options_.recovery_inset,
                       l[1] - options_.joint_margin - options_.recovery_inset);
        segment_.pid_rates[i] = 100;
        segment_.gains[i] = kin_.robot().gains[i];
    }
    segment_.goal_cell = cell(segment_.goal_q);
    reference_ = kin_.state(segment_.goal_q).pose;
    segment_.pitch = reference_[2];
}
Output RandomCollection::step(uint64_t now, const Angles &q, const Angles &excitation, bool fresh,
                              uint64_t scheduled_ns) {
    const auto schedule = scheduled_ns == UINT64_MAX ? now : scheduled_ns;
    Output out;
    out.pid_dt.fill(std::numeric_limits<double>::quiet_NaN());
    if (!fresh || !finite(q) || !finite(excitation)) {
        reset();
        out.reason = "unavailable";
        return out;
    }
    out.measured = kin_.state(q);
    out.cell = kin_.inside(q) ? cell(q) : UINT64_MAX;
    double dt = 0;
    if (initialized_) {
        if (now <= previous_ns_ || schedule <= previous_schedule_ns_ ||
            double(now - previous_ns_) / 1e9 > options_.max_gap) {
            reset();
            out.reason = "clock_gap";
            return out;
        }
        dt = double(now - previous_ns_) / 1e9;
        if (kin_.inside(previous_q_) && kin_.inside(q)) {
            const auto index = cell(previous_q_);
            if (dwell_[index] == 0)
                ++visited_cells_;
            dwell_[index] += dt;
            double movement = 0;
            for (unsigned i = 0; i < 3; ++i)
                movement = std::max(movement, std::abs(q[i] - previous_q_[i]) / dt);
            if (movement >= pi / 180)
                moving_[index] += dt;
            if (kin_.inside(previous_q_, options_.joint_margin) &&
                kin_.inside(q, options_.joint_margin)) {
                size_t working_index = 0;
                for (unsigned i = 0; i < 3; ++i) {
                    const double offset =
                        (previous_q_[i] - kin_.robot().limits[i][0]) / options_.coverage_bin;
                    const auto bin =
                        size_t(std::clamp(std::floor(offset), double(working_begin_[i]),
                                          double(working_end_[i] - 1)));
                    working_index = working_index * bins_[i] + bin;
                }
                if (!working_visited_[working_index]) {
                    working_visited_[working_index] = 1;
                    ++visited_working_cells_;
                }
            }
        }
    } else {
        reference_ = out.measured.pose;
        initialized_ = true;
        segment_.end_ns = 0;
    }
    previous_ns_ = now;
    previous_schedule_ns_ = schedule;
    previous_q_ = q;
    if (recovering_ && kin_.inside(q, options_.joint_margin + .5 * options_.recovery_inset)) {
        recovering_ = false;
        reset_pids(options_.valve_limit);
        reference_ = out.measured.pose;
        segment_.end_ns = 0;
        dt = 0;
    }
    if (!recovering_ && !kin_.inside(q, options_.joint_margin)) {
        begin_recovery(now, q);
        out.new_segment = true;
    }
    out.recovery = recovering_;
    if (recovering_) {
        out.reference = reference_;
        out.joint_reference = segment_.goal_q;
        out.ik.q = segment_.goal_q;
        out.ik.reached = true;
        // No excitation until every measured joint is back inside the range.
        // This is a direct joint target, so it does not seed IK outside limits.
    } else {
        out.excitation = excitation;
        if (now >= segment_.end_ns) {
            choose_segment(now, q);
            out.new_segment = true;
        }
        reference_[0] += segment_.velocity[0] * dt;
        reference_[1] += segment_.velocity[1] * dt;
        const double lead =
            std::hypot(reference_[0] - out.measured.pose[0], reference_[1] - out.measured.pose[1]);
        if (lead > options_.max_lead)
            for (unsigned i = 0; i < 2; ++i)
                reference_[i] = out.measured.pose[i] +
                                (reference_[i] - out.measured.pose[i]) * options_.max_lead / lead;
        reference_[2] =
            wrap(reference_[2] + std::clamp(wrap(segment_.pitch - reference_[2]),
                                            -options_.pitch_rate * dt, options_.pitch_rate * dt));
        reference_[2] = wrap(out.measured.pose[2] +
                             std::clamp(wrap(reference_[2] - out.measured.pose[2]), -.25, .25));
        out.reference = reference_;
        out.ik = kin_.inverse(reference_, q, options_.joint_margin);
        if (!out.ik.reached) {
            reference_ = out.measured.pose;
            reset_pids(options_.valve_limit);
            segment_.end_ns = now + 10000000;
            out.reason = "ik_unreachable";
            return out;
        }
        out.joint_reference = out.ik.q;
    }
    for (unsigned i = 0; i < 3; ++i) {
        const auto period = uint64_t(std::llround(1e9 / segment_.pid_rates[i]));
        if (!have_pid_[i] || schedule - pid_schedule_ns_[i] >= period) {
            out.pid_dt[i] = have_pid_[i] ? double(now - pid_ns_[i]) / 1e9 : .01;
            pid_[i].configure(segment_.gains[i]);
            held_[i] = pid_[i].compute(out.joint_reference[i], q[i], out.pid_dt[i],
                                       out.excitation[i] * options_.command_sign[i],
                                       block_positive_[i], block_negative_[i]);
            pid_ns_[i] = now;
            pid_schedule_ns_[i] = schedule;
            have_pid_[i] = true;
            out.pid_updated[i] = true;
            block_positive_[i] = block_negative_[i] = false;
        }
        out.assist[i] = held_[i] * options_.command_sign[i];
        out.applied_gains[i] = pid_[i].gains();
        const double sum = out.assist[i] + out.excitation[i];
        out.command[i] = std::clamp(sum, -1., 1.);
        out.clipped[i] = out.command[i] != sum;
        block_positive_[i] = block_positive_[i] || sum * options_.command_sign[i] > 1;
        block_negative_[i] = block_negative_[i] || sum * options_.command_sign[i] < -1;
    }
    out.valid = true;
    out.reason = recovering_ ? "joint_recovery" : "ok";
    return out;
}
} // namespace kaivuri::collection
