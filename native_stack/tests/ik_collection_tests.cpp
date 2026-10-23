#include "kaivuri/ik_collection.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
using namespace kaivuri;
using namespace kaivuri::collection;
namespace {
unsigned checks = 0;
void expect(bool condition, const char *message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}
bool near(double a, double b, double tolerance = 1e-10) {
    return std::abs(a - b) < tolerance;
}
Angles center(const Robot &r) {
    Angles q{};
    for (unsigned j = 0; j < 3; ++j)
        q[j] = (r.limits[j][0] + r.limits[j][1]) / 2;
    return q;
}
void math(const Robot &robot) {
    Kinematics k(robot);
    for (unsigned n = 0; n < 90; ++n) {
        Angles q{};
        for (unsigned j = 0; j < 3; ++j) {
            const auto &l = robot.limits[j];
            q[j] = l[0] + (l[1] - l[0]) * (.5 + .35 * std::sin(n * .31 + j));
        }
        const auto s = k.state(q);
        expect(near(wrap(s.pose[2] - q[0] - q[1] - q[2] - robot.pitch_offset), 0),
               "cutting-lip pitch");
        for (unsigned j = 0; j < 3; ++j) {
            Angles lo = q, hi = q;
            lo[j] -= 1e-6;
            hi[j] += 1e-6;
            const auto a = k.state(lo).pose, b = k.state(hi).pose;
            for (unsigned row = 0; row < 3; ++row) {
                const double diff = row == 2 ? wrap(b[row] - a[row]) : b[row] - a[row];
                expect(near(diff / 2e-6, s.jacobian[row][j], 2e-9),
                       "analytic Jacobian vs finite differences");
            }
        }
        auto seed = q;
        for (auto &v : seed)
            v += .015;
        const auto result = k.inverse(s.pose, seed);
        expect(result.reached && result.position_error < .0005 && result.pitch_error < .005,
               "IK recovers position and pitch near its branch");
    }
    const auto q = center(robot);
    auto pose = k.state(q).pose;
    // Position alone is exact, but joint constraints cannot realize an arbitrary extra half-turn.
    pose[2] = wrap(pose[2] + pi);
    const auto result = k.inverse(pose, q);
    expect(!result.reached || (result.pitch_error < .005 && result.position_error < .0005),
           "wrong pitch cannot pass the position-only acceptance check");
    pose[0] = 10;
    expect(!k.inverse(pose, q).reached, "unreachable position rejected");
    auto bad = q;
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    expect(!k.inside(bad) && !k.inverse(pose, bad).reached, "nonfinite IK rejected");
}
void pid() {
    Pid p({2, 1, .1, .1}, .5);
    for (unsigned n = 0; n < 100; ++n)
        p.compute(2, 0, .01);
    expect(p.integral() == 0, "own saturation blocks windup");
    Pid overlay({0, 1, 0, .1}, .5);
    overlay.compute(.1, 0, .01, .0001);
    expect(near(overlay.integral(), .001),
           "unclipped overlay cannot falsely block integral through roundoff");
    Pid combined({1, 1, 0, .1}, .5);
    combined.compute(.4, 0, .01, .8);
    expect(combined.integral() == 0, "combined saturation blocks windup");
    combined.compute(.1, 0, .01, 0, true, false);
    expect(combined.integral() == 0, "clipping during held ticks remembered at PID update");
    combined.compute(-.1, 0, .01, 0, true, false);
    expect(combined.integral() < 0, "opposite error can unwind");
    Pid retune({0, 1, 0, .1}, .5);
    retune.compute(.1, 0, .1);
    const double contribution = retune.integral();
    retune.configure({0, .5, 0, .1});
    expect(near(retune.integral() * retune.gains().ki, contribution),
           "retuning preserves integral contribution");
    Pid derivative({0, 0, 1, .1});
    derivative.compute(0, .2, .01);
    derivative.compute(1, .2, .01);
    expect(derivative.derivative() == 0, "setpoint step has no derivative kick");
    derivative.compute(1, .21, .01);
    expect(derivative.derivative() < 0, "derivative acts on measured motion");
    derivative.reset();
    expect(derivative.compute(0, .8, .01) == 0, "reset discards previous derivative");
}
void scheduling(const Robot &robot) {
    Options o;
    o.min_seconds = o.max_seconds = 3;
    o.pid_rates = {25};
    RandomCollection c(robot, o);
    const auto q = center(robot);
    auto out = c.step(0, q);
    expect(out.valid && out.new_segment, "first tick synchronizes to measured pose");
    const auto held = out.assist;
    out = c.step(10000000, q, {.1, -.2, .3});
    for (unsigned j = 0; j < 3; ++j) {
        expect(!out.pid_updated[j] && out.assist[j] == held[j], "25 Hz PID holds between updates");
        expect(near(out.command[j], held[j] + out.excitation[j]),
               "overlay still changes every 100 Hz tick");
    }
    c.step(20000000, q);
    c.step(30000000, q);
    out = c.step(40000000, q);
    expect(out.valid && out.pid_updated[0] && near(out.pid_dt[0], .04),
           "PID uses actual elapsed time");
    out = c.step(90000000, q);
    expect(!out.valid && std::string(out.reason) == "clock_gap" && out.command == Angles{},
           "gap resets and suppresses output");
    out = c.step(100000000, q);
    expect(out.valid && out.pid_updated[0] && out.reference == out.measured.pose,
           "restart does not catch up a stale trajectory");
    out = c.step(110000000, q, {}, false);
    expect(!out.valid && out.command == Angles{}, "stale state suppresses assist and overlay");
    auto outside = q;
    outside[0] = robot.limits[0][0];
    out = c.step(120000000, outside);
    expect(out.valid && out.recovery && out.assist[0] > 0,
           "outside margin starts inward PID recovery");
    Options jitter_options = o;
    jitter_options.pid_rates = {50};
    RandomCollection jitter(robot, jitter_options);
    jitter.step(0, q);
    jitter.step(11000000, q);
    out = jitter.step(22000000, q);
    expect(out.pid_updated[0] && near(out.pid_dt[0], .022), "jitter records actual PID dt");
    RandomCollection grid(robot, o);
    grid.step(1000000, q, {}, true, 0);
    grid.step(11010000, q, {}, true, 10000000);
    grid.step(21000000, q, {}, true, 20000000);
    grid.step(31000000, q, {}, true, 30000000);
    out = grid.step(40900000, q, {}, true, 40000000);
    expect(out.pid_updated[0] && near(out.pid_dt[0], .0399),
           "hardware PID scheduling follows output grid while dt uses actual jittered times");
    auto invalid = o;
    invalid.pid_rates = {60};
    bool rejected = false;
    try {
        RandomCollection bad(robot, invalid);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    expect(rejected, "nondivisor PID rate rejected instead of mislabeling delivered frequency");
}
void limits(const Robot &robot) {
    JointGuard guard(robot);
    auto q = center(robot);
    const Angles u{.8, -.7, .6};
    auto result = guard.apply(q, {}, u);
    expect(result.valid && result.command == u, "interior guard preserves fast excitation exactly");
    for (unsigned joint = 0; joint < 3; ++joint) {
        for (double sign : {-1., 1.}) {
            q = center(robot);
            const auto &l = robot.limits[joint];
            q[joint] = sign > 0 ? l[1] - guard.options().margin : l[0] + guard.options().margin;
            Angles outward{}, inward{};
            outward[joint] = sign * .9;
            inward[joint] = -sign * .9;
            result = guard.apply(q, {}, outward);
            expect(result.valid && std::abs(result.command[joint]) < 1e-14,
                   "outward PID plus overlay blocked at soft boundary");
            result = guard.apply(q, {}, inward);
            expect(result.command == inward, "inward recovery remains available");
            q[joint] -= sign * guard.options().taper * .5;
            result = guard.apply(q, {}, outward);
            expect(near(result.scale[joint], .5), "outward command tapers before margin");
            Angles velocity{};
            velocity[joint] = sign;
            result = guard.apply(q, velocity, outward);
            expect(result.command[joint] == 0,
                   "outward measured velocity consumes boundary headroom");
        }
    }
    q = center(robot);
    q[0] = robot.limits[0][1] + .001;
    result = guard.apply(q, {}, u);
    expect(result.valid && result.command[0] == 0 && result.command[1] == u[1],
           "beyond hard limit blocks outward command while preserving other joints");
    expect(guard.apply(q, {}, {-.4, 0, 0}).command[0] == -.4,
           "inward command remains available beyond hard limit");
    q = center(robot);
    expect(!guard.apply(q, {}, u, false).valid, "stale feedback cannot drive the limit guard");
    auto nan_velocity = Angles{};
    nan_velocity[1] = NAN;
    expect(!guard.apply(q, nan_velocity, u).valid, "nonfinite gyro feedback is rejected");
    GuardOptions reversed;
    reversed.command_sign[0] = -1;
    JointGuard inverse_sign(robot, reversed);
    q[0] = robot.limits[0][1] - reversed.margin;
    expect(std::abs(inverse_sign.apply(q, {}, {-1, 0, 0}).command[0]) < 1e-14,
           "command growth sign is separate from PWM direction");
    Valve valve;
    valve.min = 1000;
    valve.max = 2000;
    valve.center = 1500;
    valve.ramp = true;
    valve.ramp_limit = 10;
    valve.dither = true;
    valve.dither_amp = 50;
    valve.dither_hz = 33;
    RampState history{1900, 1, true};
    expect(guarded_valve_pulse(valve, 0, true, 1.01, history) == 1500,
           "neutral guard bypasses retained ramp and dither");
    history = {1900, 1, true};
    expect(guarded_valve_pulse(valve, -.2, false, 1.01, history) <= 1500,
           "inward recovery cannot retain old outward ramp direction");
    history = {1900, 1, true};
    expect(near(guarded_valve_pulse(valve, .1, true, 1.01, history), 1550),
           "tapered guard command cannot be enlarged by ramp or dither");
    Options positive_options, negative_options;
    negative_options.command_sign = {-1, -1, -1};
    RandomCollection positive(robot, positive_options), negative(robot, negative_options);
    q = center(robot);
    for (uint64_t tick = 0; tick < 300; ++tick) {
        const auto a = positive.step(tick * 10000000, q), b = negative.step(tick * 10000000, q);
        for (unsigned j = 0; j < 3; ++j)
            expect(a.assist[j] == -b.assist[j],
                   "PID output follows configured physical command growth sign");
    }
    RandomCollection c(robot);
    q = center(robot);
    for (uint64_t tick = 0; tick < 20000; ++tick) {
        const auto output = c.step(tick * 10000000, q, {.8, -.8, .8});
        if (output.valid)
            expect(c.kinematics().inside(output.joint_reference, c.options().joint_margin),
                   "all accepted random IK references obey configured joint margins");
    }
}
void recovery(const Robot &robot) {
    Options options;
    RandomCollection c(robot, options);
    JointGuard guard(robot);
    Angles q{-50.15 * pi / 180, 92.78 * pi / 180, -119.2 * pi / 180};
    auto out = c.step(0, q, {1, -1, 1});
    expect(out.valid && out.recovery && out.new_segment && out.cell == UINT64_MAX,
           "parked pose outside hard limit starts labelled recovery without a false coverage cell");
    expect(c.segment().kind == Kind::Recovery && c.segment().duration == 0,
           "recovery lasts until measured re-entry rather than a random timeout");
    expect(out.assist[0] > 0 && out.assist[2] > 0 && out.excitation == Angles{},
           "recovery drives parked boom and bucket inward with no overlay");
    for (unsigned i = 0; i < 3; ++i) {
        expect(c.segment().pid_rates[i] == 100 && near(out.applied_gains[i].kp, robot.gains[i].kp),
               "recovery uses fixed profile gains at 100 Hz");
        expect(std::abs(out.assist[i]) <= options.recovery_limit,
               "recovery output bounded separately");
    }
    Options reversed = options;
    reversed.command_sign = {-1, -1, -1};
    RandomCollection inverse(robot, reversed);
    const auto negative = inverse.step(0, q, {1, -1, 1});
    for (unsigned i = 0; i < 3; ++i)
        expect(out.assist[i] == -negative.assist[i],
               "recovery respects configured command growth sign");
    uint64_t tick = 1;
    for (; tick < 10000 && out.recovery; ++tick) {
        const auto bounded = guard.apply(q, {}, out.command);
        for (unsigned i = 0; i < 3; ++i) {
            const auto &l = robot.limits[i];
            if (q[i] < l[0] + options.joint_margin)
                expect(bounded.command[i] >= 0, "lower-bound recovery only inward");
            if (q[i] > l[1] - options.joint_margin)
                expect(bounded.command[i] <= 0, "upper-bound recovery only inward");
            q[i] += bounded.command[i] * .01; // test-only plant, no hardware
        }
        c.feedback_limited(out.command, bounded.command);
        out = c.step(tick * 10000000, q, {1, -1, 1});
        if (out.recovery) {
            expect(out.excitation == Angles{}, "overlay stays absent for the entire recovery");
            expect(c.kinematics().inside(out.joint_reference, options.joint_margin),
                   "recovery targets stay inside working range");
        }
    }
    expect(tick < 10000 && out.valid && !out.recovery && out.new_segment,
           "measured re-entry resumes randomized collection");
    expect(out.reference == out.measured.pose && c.segment().kind != Kind::Recovery,
           "random control starts from actual re-entry pose without catching up a trajectory");
    q[0] = robot.limits[0][1] + .01;
    out = c.step(tick++ * 10000000, q);
    expect(out.valid && out.recovery && out.assist[0] < 0,
           "upper hard-limit breach also recovers inward");
    out = c.step(tick++ * 10000000, q, {}, false);
    expect(!out.valid && out.command == Angles{}, "stale feedback stops recovery");
    out = c.step(tick * 10000000, q);
    expect(out.valid && out.recovery && out.pid_updated[0],
           "fresh restart rebuilds recovery PID state");
}
void coverage(const Robot &robot) {
    RandomCollection c(robot);
    expect(c.coverage_percent() == 0 && c.visited_working_cells() == 0, "coverage begins empty");
    std::array<std::vector<double>, 3> samples;
    const auto &o = c.options();
    for (unsigned i = 0; i < 3; ++i) {
        const auto &l = robot.limits[i];
        for (unsigned bin = 0; bin < unsigned(std::ceil((l[1] - l[0]) / o.coverage_bin)); ++bin) {
            const double lo = std::max(l[0] + bin * o.coverage_bin, l[0] + o.joint_margin);
            const double hi = std::min(l[0] + (bin + 1) * o.coverage_bin, l[1] - o.joint_margin);
            if (hi - lo > 1e-10)
                samples[i].push_back((lo + hi) / 2);
        }
    }
    expect(c.working_cells() == samples[0].size() * samples[1].size() * samples[2].size(),
           "percentage denominator includes only bins overlapping the working range");
    uint64_t tick = 0;
    for (double a : samples[0])
        for (double b : samples[1])
            for (double d : samples[2]) {
                c.step(tick++ * 10000000, {a, b, d});
                c.step(tick++ * 10000000, {a, b, d});
                expect(c.coverage_percent() <= 100, "coverage cannot exceed 100 percent");
            }
    expect(c.visited_working_cells() == c.working_cells() && near(c.coverage_percent(), 100),
           "constant-time visit counter agrees with exhaustive measured grid coverage");
    c.reset();
    expect(near(c.coverage_percent(), 100), "pauses and strip changes retain this-run coverage");
    const auto count = c.visited_cells();
    auto outside = center(robot);
    outside[0] = robot.limits[0][0] - .01;
    c.step(tick++ * 10000000, outside);
    c.step(tick++ * 10000000, outside);
    expect(c.visited_cells() == count,
           "out-of-range parked pose cannot fabricate a coverage visit");
}
void diversity(const Robot &robot) {
    RandomCollection first(robot), second(robot);
    const auto q = center(robot);
    std::set<Kind> kinds;
    std::set<double> rates;
    double min_duration = 4, max_duration = 0, min_gain = 2, max_gain = 0;
    for (uint64_t tick = 0; tick < 60000; ++tick) {
        const auto a = first.step(tick * 10000000, q), b = second.step(tick * 10000000, q);
        expect(a.command == b.command && a.reference == b.reference && a.valid == b.valid,
               "seeded run deterministic");
        if (!a.new_segment)
            continue;
        const auto &s = first.segment();
        kinds.insert(s.kind);
        expect(s.duration >= .1 && s.duration <= 3, "segment duration bounds");
        min_duration = std::min(min_duration, s.duration);
        max_duration = std::max(max_duration, s.duration);
        for (unsigned j = 0; j < 3; ++j) {
            rates.insert(s.pid_rates[j]);
            const double scale = s.gains[j].kp / robot.gains[j].kp;
            min_gain = std::min(min_gain, scale);
            max_gain = std::max(max_gain, scale);
            expect(scale >= .5 && scale <= 1.5, "PID gain scale bounds");
            expect(s.gains[j].kd == 0, "zero base D stays zero without an invented tuning");
        }
        expect(s.goal_q == second.segment().goal_q && s.pid_rates == second.segment().pid_rates,
               "goals and controller rates reproducible");
    }
    expect(kinds.size() == 5 && rates.size() == 3, "all segment kinds and PID rates exercised");
    expect(min_duration <= .12 && max_duration >= 2.9 && min_gain < .55 && max_gain > 1.45,
           "duration and gain ranges actually covered");
    expect(first.visited_cells() == 1, "scheduled goals never count as visited poses");
    double dwell = 0, moving = 0;
    for (double v : first.dwell())
        dwell += v;
    for (double v : first.moving_dwell())
        moving += v;
    expect(near(dwell, 599.99, 1e-7) && moving == 0,
           "measured dwell accounting excludes fabricated motion");
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("expected profile directory");
        const auto robot = load_robot(argv[1]);
        math(robot);
        pid();
        scheduling(robot);
        limits(robot);
        recovery(robot);
        coverage(robot);
        diversity(robot);
        std::cout << checks << " IK collection checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
