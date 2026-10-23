#include "kaivuri/gamepad.hpp"
#include "kaivuri/pid_collect.hpp"
#include "kaivuri/pwm.hpp"
#include "kaivuri/util.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

using namespace kaivuri;
using namespace kaivuri::pidcollect;
using collection::output_period_ns;
namespace {
constexpr uint64_t coverage_report_ns = 30000000000ULL;

// Edge-triggered requests gathered from the gamepad or scripted stdin during one tick.
struct Requests {
    bool record = false, save = false, pump = false, assist = false, excitation = false;
};
double seconds(uint64_t ns) {
    return double(ns) / 1e9;
}
bool moving_sticks(const Commands &manual) {
    return std::any_of(manual.begin(), manual.end(), [](double v) { return std::abs(v) > 1e-6; });
}
// A (record), B (overlay), X (pump), Start (PID).
std::array<bool, 4> buttons(const Pad &pad) {
    return {pad.keys[BTN_SOUTH], pad.keys[BTN_EAST], pad.keys[BTN_NORTH], pad.keys[BTN_START]};
}
void scripted_command(const std::string &text, Requests &requests, Commands &manual) {
    if (text == "start")
        requests.record = true;
    else if (text == "save")
        requests.save = true;
    else if (text == "pump")
        requests.pump = true;
    else if (text == "assist")
        requests.assist = true;
    else if (text == "excitation")
        requests.excitation = true;
    else if (text == "quit")
        stop_requested = 1;
    else if (text.rfind("manual ", 0) == 0) {
        std::istringstream input(text.substr(7));
        manual = {};
        for (unsigned j = 1; j <= 3; ++j)
            require(bool(input >> manual[j]) && std::isfinite(manual[j]) &&
                        std::abs(manual[j]) <= 1,
                    "manual requires three finite values in [-1,1]");
        input >> std::ws;
        require(input.eof(), "trailing manual command data");
    } else
        require(text.empty(), "unknown scripted input: " + text);
}
// Nonblocking line reader for scripted stdin; at most 1024 bytes per tick.
class Script {
    std::array<char, 1024> line_{};
    size_t used_ = 0;

  public:
    Script() {
        require(fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK) >= 0,
                "cannot make scripted input nonblocking");
    }
    void poll(Requests &requests, Commands &manual) {
        char c;
        for (unsigned count = 0; count < 1024 && ::read(STDIN_FILENO, &c, 1) == 1; ++count) {
            if (c != '\n') {
                require(used_ + 1 < line_.size(), "scripted command too long");
                line_[used_++] = c;
                continue;
            }
            const std::string text(line_.data(), used_);
            used_ = 0;
            scripted_command(text, requests, manual);
        }
    }
};
void print_coverage(const collection::RandomCollection &controller) {
    std::ostringstream message;
    message << std::fixed << std::setprecision(1)
            << "[PID] Joint coverage: " << controller.coverage_percent() << "% ("
            << controller.visited_working_cells() << '/' << controller.working_cells()
            << " working-range cells visited during PID; this run)\n";
    std::cout << message.str() << std::flush;
}
} // namespace

int main(int argc, char **argv) {
    try {
        const auto o = parse_options(argc, argv);
        const auto runtime = load_runtime(o);
        const auto &hardware = runtime.robot.hardware;
        collection::RandomCollection controller(runtime.robot, o.planning);
        collection::JointGuard guard(runtime.robot, o.guard);
        mlp::Excitation excitation(o.excitation, runtime.limits);
        ControlLock lock(runtime_dir());
        Pwm pwm(runtime.metadata.config, !o.dry);
        const auto serial = open_serial(o.serial);
        std::unique_ptr<Pad> pad;
        if (!o.scripted)
            pad = std::make_unique<Pad>(o.gamepad);
        auto attachment = std::make_shared<PidAttachment>(runtime.protocol_json, o.queue);
        mlp::MlpWriter writer(o.out, runtime.metadata, o.queue, attachment);
        ImuParser parser(4);
        mlp::RawCapture capture{&writer, &parser};
        parser.set_sample_sink(mlp::RawCapture::sink, &capture);
        install_stop_handlers();
        std::unique_ptr<Script> script;
        if (o.scripted)
            script = std::make_unique<Script>();
        // Command index (0..5) driving each valve; 6 = none, held at zero.
        std::vector<unsigned> valve_command(hardware.valves.size(), 6);
        for (unsigned i = 0; i < hardware.valves.size(); ++i)
            for (unsigned j = 0; j < 6; ++j)
                if (hardware.valves[i].name == command_names[j])
                    valve_command[i] = j;
        bool active = false, pump = false, input_valid = true;
        Commands manual{};
        std::array<bool, 4> previous{};
        if (pad)
            previous = buttons(*pad);
        std::array<RampState, 8> ramps{};
        std::array<uint8_t, 8192> bytes{};
        uint64_t tick = 0, missed = 0, first = monotonic_ns(), next = first, record_begin = 0,
                 saved_block = UINT64_MAX, next_coverage = first + coverage_report_ns;
        auto pause = [&] {
            active = false;
            excitation.disable();
            controller.reset();
        };
        auto halt = [&] {
            pause();
            pump = false;
        };
        auto finish = [&](bool complete) {
            halt();
            manual = {};
            capture.recording = false;
            writer.finish(complete);
        };
        auto begin = [&] {
            if (!writer.start())
                return;
            pause();
            capture.failed = false;
            capture.recording = true;
            capture.epoch = parser.sample().epoch;
            capture.last_raw = 0;
            record_begin = monotonic_ns();
            saved_block = UINT64_MAX;
            if (parser.sample().valid)
                capture.packet(parser.sample());
        };
        std::cout << "PID collector: 100 Hz, PWM " << (o.dry ? "DRY" : "LIVE") << ", seed "
                  << o.planning.seed
                  << "; A=record/save Start=PID B=overlay X=pump; joint guards on\n"
                  << std::flush;
        if (o.record_on_start)
            begin();
        try {
            while (!stop_requested) {
                const auto start_ns = monotonic_ns();
                if (o.duration > 0 && seconds(start_ns - first) >= o.duration)
                    break;
                for (size_t total = 0; total < bytes.size();) {
                    const auto n = ::read(serial.get(), bytes.data(),
                                          std::min(o.read_chunk, bytes.size() - total));
                    if (n <= 0) {
                        require(n == 0 || errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR,
                                "serial read failed");
                        break;
                    }
                    parser.feed(bytes.data(), size_t(n), seconds(monotonic_ns()));
                    total += size_t(n);
                }
                Requests requests;
                const char *event_reason = nullptr;
                if (pad && input_valid) {
                    try {
                        pad->poll_events();
                        manual = pad->action(false, false);
                        const auto keys = buttons(*pad);
                        requests.record = keys[0] && !previous[0];
                        requests.excitation = keys[1] && !previous[1];
                        requests.pump = keys[2] && !previous[2];
                        requests.assist = keys[3] && !previous[3];
                        previous = keys;
                    } catch (const std::exception &e) {
                        input_valid = false;
                        manual = {};
                        halt();
                        std::cerr << "[PID] " << e.what() << "; output centered, PID cancelled\n";
                    }
                }
                if (script)
                    script->poll(requests, manual);
                if (stop_requested)
                    break;
                if (capture.reset) {
                    capture.reset = false;
                    halt();
                    record_begin = monotonic_ns();
                    saved_block = UINT64_MAX;
                    event_reason = "imu_epoch_reset";
                }
                require(!capture.failed && writer.state() != mlp::WriterState::Failed,
                        "PID recording failed; incomplete strip retained, restart collector");
                if (requests.save || (requests.record && capture.recording))
                    finish(true);
                else if (requests.record)
                    begin();
                if (capture.recording && seconds(monotonic_ns() - record_begin) >= o.record_seconds)
                    finish(true);
                if (requests.pump) {
                    pump = !pump;
                    if (!pump)
                        pause();
                }
                const double now = seconds(monotonic_ns());
                const auto imu = parser.sample();
                const bool fresh = imu.valid && !parser.firmware_blocked && now >= imu.received &&
                                   now - imu.received <= .03;
                std::array<double, 4> angles{};
                collection::Angles q{}, velocity{};
                if (fresh) {
                    angles = joint_angles(hardware, imu.quats);
                    for (unsigned j = 0; j < 3; ++j) {
                        q[j] = angles[j + 1];
                        const auto &joint = hardware.joints[j + 1];
                        velocity[j] =
                            (imu.motion[joint.sensor][1] - imu.motion[joint.parent_sensor][1]) *
                            pi / 180;
                    }
                }
                // Everything that must hold for PID or any output other than neutral.
                const bool ready = fresh && input_valid && capture.recording && pump;
                const bool sticks = moving_sticks(manual);
                if (!ready || sticks)
                    pause();
                if (requests.assist) {
                    if (active)
                        pause();
                    else if (ready && !sticks) {
                        controller.reset();
                        active = true;
                    } else {
                        event_reason = "pid_start_requirements";
                        std::cerr << "[PID] Start requires a recording, pump on, fresh state and "
                                     "centered sticks\n";
                    }
                }
                if (requests.excitation) {
                    if (excitation.enabled())
                        excitation.disable();
                    else if (active && !controller.recovering())
                        excitation.enable(now);
                }
                Frame frame;
                frame.tick = tick;
                frame.active = active;
                frame.output.pid_dt.fill(NAN);
                if (fresh) {
                    frame.output.measured = controller.kinematics().state(q);
                    frame.output.cell =
                        controller.kinematics().inside(q) ? controller.cell(q) : UINT64_MAX;
                }
                Commands signal{}, assist{}, combined{};
                if (active) {
                    const bool was_recovering = controller.recovering();
                    signal = excitation.sample(now, angles, fresh);
                    frame.output = controller.step(monotonic_ns(), q,
                                                   {signal[1], signal[2], signal[3]}, fresh, next);
                    frame.output.measured = controller.kinematics().state(q);
                    frame.segment = controller.segment();
                    frame.reason = frame.output.reason;
                    for (unsigned j = 0; j < 3; ++j) {
                        assist[j + 1] = frame.output.assist[j];
                        signal[j + 1] = frame.output.excitation[j];
                    }
                    if (frame.output.recovery) {
                        excitation.disable();
                        if (!was_recovering)
                            std::cout
                                << "[PID] Inward recovery; fixed gains at 100 Hz, output limit "
                                << o.planning.recovery_limit << ", overlay off\n"
                                << std::flush;
                    } else if (was_recovering && frame.output.valid)
                        std::cout << "[PID] Working range reached; randomized commands active\n"
                                  << std::flush;
                    if (!frame.output.valid && std::string(frame.output.reason) != "ik_unreachable")
                        pause();
                } else
                    frame.reason = event_reason   ? event_reason
                                   : !input_valid ? "input_lost"
                                   : !fresh       ? "stale_feedback"
                                   : sticks       ? "manual_takeover"
                                                  : "idle";
                for (unsigned i = 0; i < 6; ++i)
                    combined[i] = std::clamp(manual[i] + assist[i] + signal[i], -1., 1.);
                const bool allow = ready && (!frame.active || frame.output.valid);
                frame.guard =
                    guard.apply(q, velocity,
                                allow ? collection::Angles{combined[1], combined[2], combined[3]}
                                      : collection::Angles{},
                                fresh);
                if (frame.active && frame.output.valid && frame.guard.valid)
                    controller.feedback_limited(frame.output.command, frame.guard.command);
                if (!frame.guard.valid) {
                    pause();
                    if (fresh)
                        pump = false;
                }
                Commands command{};
                for (unsigned j = 0; j < 3; ++j)
                    command[j + 1] = frame.guard.command[j];
                const auto computed = monotonic_ns();
                for (unsigned i = 0; i < hardware.valves.size(); ++i) {
                    const unsigned index = valve_command[i];
                    const bool limited = index >= 1 && index <= 3 && frame.guard.limited[index - 1];
                    pwm.pulses[hardware.valves[i].channel] = collection::guarded_valve_pulse(
                        hardware.valves[i], index < 6 ? command[index] : 0, limited,
                        seconds(computed), ramps[i]);
                }
                pwm.pulses[hardware.pump_channel] = pump ? hardware.pump_run : hardware.pump_min;
                pwm.flush();
                const auto written = monotonic_ns();
                if (capture.recording) {
                    const auto &b = excitation.block();
                    if (saved_block != b.id) {
                        mlp::Record block;
                        block.kind = mlp::RecordKind::Block;
                        block.block = b;
                        if (!writer.push(block))
                            capture.failed = true;
                        saved_block = b.id;
                    }
                    mlp::Record record;
                    record.kind = mlp::RecordKind::Drive;
                    auto &s = record.sample;
                    s.imu = imu;
                    s.manual = manual;
                    s.excitation = signal;
                    s.assist = assist;
                    s.command = command;
                    for (unsigned j = 0; j < 6; ++j)
                        s.constraint[j] = command[j] - combined[j];
                    s.angles = angles;
                    s.pulses = pwm.pulses;
                    s.tick = tick;
                    s.scheduled_ns = next;
                    s.start_ns = start_ns;
                    s.post_write_ns = written;
                    s.missed_periods = missed;
                    s.command_age_s = seconds(written - computed);
                    s.state_age_s =
                        imu.valid ? std::max(0., seconds(written) - imu.received) : INFINITY;
                    s.imu_valid = fresh && s.state_age_s <= .03;
                    s.input_valid = input_valid;
                    s.write_ok = true;
                    s.pump_on = pump;
                    s.controller_valid = frame.guard.valid && (!frame.active || frame.output.valid);
                    s.source = frame.active ? mlp::DriveSource::Controller
                               : sticks     ? mlp::DriveSource::Manual
                                            : mlp::DriveSource::Idle;
                    s.checksum_errors = parser.checksum_errors;
                    s.invalid_frames = parser.invalid_frames;
                    s.resets = parser.resets;
                    s.block_id = b.id;
                    s.seed = b.options.seed;
                    s.target = b.options.target;
                    s.waveform = b.options.waveform;
                    s.enabled = excitation.enabled();
                    s.stage = excitation.stage();
                    s.excitation_elapsed_s = excitation.elapsed(now);
                    s.noise_tick = excitation.noise_tick();
                    if (!attachment->push(frame) || !writer.push(record))
                        capture.failed = true;
                    require(!capture.failed,
                            "PID recording queue failed; incomplete strip retained");
                }
                ++tick;
                next += output_period_ns;
                const auto end = monotonic_ns();
                if (end >= next_coverage) {
                    print_coverage(controller);
                    next_coverage = end + coverage_report_ns;
                }
                if (end >= next) {
                    const auto skipped = (end - next) / output_period_ns + 1;
                    missed += skipped;
                    tick += skipped;
                    next += skipped * output_period_ns;
                }
                timespec wake{time_t(next / 1000000000), long(next % 1000000000)};
                while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr) == EINTR &&
                       !stop_requested) {
                }
            }
            pwm.neutral();
            finish(!capture.failed);
            writer.shutdown();
        } catch (...) {
            try {
                pwm.neutral();
            } catch (...) {
            }
            finish(false);
            writer.shutdown();
            throw;
        }
        print_coverage(controller);
        std::cout << "[PID] Finished: ticks=" << tick << " missed=" << missed
                  << " dropped=" << writer.dropped() << " queue_high_water=" << writer.high_water()
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "PID collector: " << e.what() << '\n';
        return 1;
    }
}
