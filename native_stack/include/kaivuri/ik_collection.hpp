#pragma once
#include "kaivuri/core.hpp"
#include <filesystem>
#include <vector>

namespace kaivuri::collection {
using Angles = std::array<double, 3>;
using Matrix = std::array<Angles, 3>;
struct Gains {
    double kp = 0, ki = 0, kd = 0, tau = .1;
};
struct Robot {
    Config hardware;
    std::array<std::array<double, 2>, 3> limits{};
    std::array<Gains, 3> gains{};
    double pitch_offset = 0;
};
Robot load_robot(const std::filesystem::path &profile);

struct TaskState {
    Angles pose{}; // carriage-relative x [m], z [m], cutting-lip pitch [rad]
    Matrix jacobian{};
};
struct IkResult {
    Angles q{};
    double position_error = 0, pitch_error = 0;
    unsigned iterations = 0;
    bool reached = false;
};
class Kinematics {
    Robot robot_;

  public:
    explicit Kinematics(Robot robot);
    TaskState state(const Angles &) const;
    bool inside(const Angles &, double margin = 0) const;
    IkResult inverse(const Angles &target, const Angles &seed, double margin = 0) const;
    const Robot &robot() const { return robot_; }
};

// Same explicit-dt arithmetic as reference/modules/pid.py; optional combined-command
// saturation also prevents windup when an excitation overlay consumes headroom.
class Pid {
    Gains gains_{};
    double limit_ = 1, integral_ = 0, last_ = 0, derivative_ = 0;
    bool have_last_ = false;

  public:
    explicit Pid(Gains gains = {}, double limit = 1);
    void configure(Gains, bool preserve_i_contribution = true);
    double compute(double setpoint, double measured, double dt, double excitation = 0,
                   bool blocked_positive = false, bool blocked_negative = false);
    void reset();
    const Gains &gains() const { return gains_; }
    double integral() const { return integral_; }
    double derivative() const { return derivative_; }
};

struct GuardOptions {
    double margin = pi / 18, taper = pi / 36, velocity_lookahead = .1;
    // Normalized valve command -> increasing measured joint angle. This is
    // independent of the PWM electrical direction in servo_config.yaml.
    Angles command_sign{1, 1, 1};
};
struct GuardResult {
    Angles command{}, scale{}, predicted_q{};
    std::array<bool, 3> limited{};
    bool valid = false;
    const char *reason = "unavailable";
};
// A command-direction guard, not a braking model or collision certificate.
// Outside either limit, only the direction back into the working range passes.
class JointGuard {
    std::array<std::array<double, 2>, 3> limits_{};
    GuardOptions options_;

  public:
    JointGuard(const Robot &, GuardOptions = {});
    GuardResult apply(const Angles &q, const Angles &velocity, const Angles &command,
                      bool fresh = true) const;
    const GuardOptions &options() const { return options_; }
};
double guarded_valve_pulse(const Valve &, double command, bool limited, double now, RampState &);

struct Options {
    uint64_t seed = 42;
    double min_seconds = .1, max_seconds = 3;
    double min_speed = .005, max_speed = .020;
    double max_pitch_change = pi / 4, pitch_rate = .4;
    double gain_min = .5, gain_max = 1.5;
    std::vector<double> pid_rates{25, 50, 100};
    Angles command_sign{1, 1, 1};
    double joint_margin = pi / 18, max_lead = .03, valve_limit = .5;
    double recovery_limit = .25, recovery_inset = pi / 180;
    double coverage_bin = pi / 18, goal_seconds = 30, max_gap = .03;
};
enum class Kind { Free, Coverage, Reverse, Pitch, Hold, Recovery };
const char *kind_name(Kind);
struct Segment {
    uint64_t id = 0, begin_ns = 0, end_ns = 0, goal_cell = 0;
    Kind kind = Kind::Free;
    double duration = 0, pitch = 0;
    std::array<double, 2> velocity{};
    Angles pid_rates{}, goal_q{};
    std::array<Gains, 3> gains{};
};
struct Output {
    TaskState measured{};
    Angles reference{}, joint_reference{}, assist{}, excitation{}, command{}, pid_dt{};
    std::array<Gains, 3> applied_gains{};
    std::array<bool, 3> pid_updated{}, clipped{};
    IkResult ik{};
    uint64_t cell = UINT64_MAX;
    bool valid = false, new_segment = false, recovery = false;
    const char *reason = "initializing";
};

// Hardware-free command/controller core used by preview and the separate PID
// collector. Joint limits alone are NOT a collision certificate; automatic
// full-reach routing needs checked geometry and physical acceptance.
class RandomCollection {
    Kinematics kin_;
    Options options_;
    uint64_t rng_ = 0, previous_ns_ = 0, previous_schedule_ns_ = 0, goal_begin_ = 0,
             segment_id_ = 0;
    bool initialized_ = false, have_goal_ = false, recovering_ = false;
    Angles previous_q_{}, reference_{}, goal_q_{}, held_{};
    std::array<Pid, 3> pid_;
    std::array<uint64_t, 3> pid_ns_{}, pid_schedule_ns_{};
    std::array<bool, 3> have_pid_{}, block_positive_{}, block_negative_{};
    std::array<size_t, 3> bins_{}, working_begin_{}, working_end_{};
    std::vector<double> dwell_, moving_;
    std::vector<uint8_t> working_visited_;
    size_t visited_cells_ = 0, working_cells_ = 1, visited_working_cells_ = 0;
    Segment segment_{};
    double uniform(double lo, double hi);
    double logarithmic(double lo, double hi);
    void choose_goal(uint64_t now, const Angles &q);
    void choose_segment(uint64_t now, const Angles &q);
    void reset_pids(double limit);
    void begin_recovery(uint64_t now, const Angles &q);

  public:
    RandomCollection(Robot, Options = {});
    // Supply the output grid separately on hardware: scheduling uses that grid,
    // while PID dt and stale-clock checks always use actual monotonic time.
    Output step(uint64_t now_ns, const Angles &q, const Angles &excitation = {}, bool fresh = true,
                uint64_t scheduled_ns = UINT64_MAX);
    void reset();
    void feedback_limited(const Angles &requested, const Angles &actual);
    size_t cell(const Angles &) const;
    size_t visited_cells() const { return visited_cells_; }
    size_t working_cells() const { return working_cells_; }
    size_t visited_working_cells() const { return visited_working_cells_; }
    double coverage_percent() const {
        return 100. * double(visited_working_cells_) / double(working_cells_);
    }
    bool recovering() const { return recovering_; }
    const std::vector<double> &dwell() const { return dwell_; }
    const std::vector<double> &moving_dwell() const { return moving_; }
    const Segment &segment() const { return segment_; }
    const Kinematics &kinematics() const { return kin_; }
    const Options &options() const { return options_; }
};
} // namespace kaivuri::collection
