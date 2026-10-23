#include "kaivuri/gamepad.hpp"
#include "kaivuri/pid_collect.hpp"
#include "kaivuri/pwm.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

using namespace kaivuri;
using namespace kaivuri::pidcollect;
namespace {
volatile sig_atomic_t stopped = 0;
void stop_signal(int) {
    stopped = 1;
}
uint64_t ns() {
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return uint64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}
struct Serial {
    int fd = -1;
    explicit Serial(const std::filesystem::path &path) {
        fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_NOCTTY | O_CLOEXEC);
        if (fd < 0)
            throw std::runtime_error("cannot open serial: " + path.string());
        if (isatty(fd)) {
            termios t{};
            if (tcgetattr(fd, &t) != 0) {
                ::close(fd);
                fd = -1;
                throw std::runtime_error("serial termios failed");
            }
            cfmakeraw(&t);
            t.c_cflag |= CLOCAL | CREAD;
            if (tcsetattr(fd, TCSANOW, &t) != 0 || ioctl(fd, TIOCEXCL) != 0) {
                ::close(fd);
                fd = -1;
                throw std::runtime_error("serial setup/exclusive lock failed");
            }
        }
    }
    ~Serial() {
        if (fd >= 0)
            ::close(fd);
    }
};
bool moving_sticks(const Commands &manual) {
    return std::any_of(manual.begin(), manual.end(), [](double v) { return std::abs(v) > 1e-6; });
}
} // namespace

int main(int argc, char **argv) {
    try {
        const auto o = parse_options(argc, argv);
        const auto runtime = load_runtime(o);
        collection::RandomCollection controller(runtime.robot, o.planning);
        collection::JointGuard guard(runtime.robot, o.guard);
        mlp::Excitation excitation(o.excitation, runtime.limits);
        ControlLock lock("/tmp/kaivuri-native-" + std::to_string(geteuid()));
        Pwm pwm(runtime.metadata.config, !o.dry);
        Serial serial(o.serial);
        std::unique_ptr<Pad> pad;
        if (!o.scripted)
            pad = std::make_unique<Pad>(o.gamepad);
        auto attachment = std::make_shared<PidAttachment>(runtime.protocol_json, o.queue);
        mlp::MlpWriter writer(o.out, runtime.metadata, o.queue, attachment);
        ImuParser parser(4);
        mlp::RawCapture capture{&writer, &parser};
        parser.set_sample_sink(mlp::RawCapture::sink, &capture);
        signal(SIGINT, stop_signal);
        signal(SIGTERM, stop_signal);
        if (o.scripted &&
            fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK) < 0)
            throw std::runtime_error("cannot make scripted input nonblocking");
        bool active = false, pump = false, input_valid = true;
        Commands manual{};
        std::array<bool, 4> previous{};
        if (pad)
            previous = {pad->keys[BTN_SOUTH], pad->keys[BTN_EAST], pad->keys[BTN_NORTH],
                        pad->keys[BTN_START]};
        std::array<RampState, 8> ramps{};
        std::array<uint8_t, 8192> bytes{};
        std::array<char, 1024> line{};
        size_t used = 0;
        uint64_t tick = 0, missed = 0, first = ns(), next = first, record_begin = 0,
                 saved_block = UINT64_MAX;
        uint64_t next_coverage = first + 30000000000ULL;
        auto print_coverage = [&] {
            std::ostringstream message;
            message << std::fixed << std::setprecision(1)
                    << "[PID] Joint coverage: " << controller.coverage_percent() << "% ("
                    << controller.visited_working_cells() << '/' << controller.working_cells()
                    << " working-range cells visited during PID; this run)\n";
            std::cout << message.str() << std::flush;
        };
        auto pause = [&] {
            active = false;
            excitation.disable();
            controller.reset();
        };
        auto finish = [&](bool complete) {
            pause();
            pump = false;
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
            record_begin = ns();
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
            while (!stopped) {
                const auto start_ns = ns();
                if (o.duration > 0 && double(start_ns - first) / 1e9 >= o.duration)
                    break;
                size_t total = 0;
                while (total < bytes.size()) {
                    const auto n = ::read(serial.fd, bytes.data(),
                                          std::min(o.read_chunk, bytes.size() - total));
                    if (n > 0) {
                        parser.feed(bytes.data(), size_t(n), double(ns()) / 1e9);
                        total += size_t(n);
                    } else {
                        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                            throw std::runtime_error("serial read failed");
                        break;
                    }
                }
                bool start_record = false, save = false, toggle_pump = false, toggle_assist = false,
                     toggle_excitation = false;
                const char *event_reason = nullptr;
                if (pad && input_valid) {
                    try {
                        pad->poll_events();
                        manual = pad->action(false, false);
                        const std::array<bool, 4> keys{pad->keys[BTN_SOUTH], pad->keys[BTN_EAST],
                                                       pad->keys[BTN_NORTH], pad->keys[BTN_START]};
                        start_record = keys[0] && !previous[0];
                        toggle_excitation = keys[1] && !previous[1];
                        toggle_pump = keys[2] && !previous[2];
                        toggle_assist = keys[3] && !previous[3];
                        previous = keys;
                    } catch (const std::exception &e) {
                        input_valid = false;
                        manual = {};
                        pause();
                        pump = false;
                        std::cerr << "[PID] " << e.what() << "; output centered, PID cancelled\n";
                    }
                }
                if (o.scripted) {
                    char c;
                    for (unsigned count = 0; count < 1024 && ::read(STDIN_FILENO, &c, 1) == 1;
                         ++count) {
                        if (c != '\n') {
                            if (used + 1 >= line.size())
                                throw std::runtime_error("scripted command too long");
                            line[used++] = c;
                            continue;
                        }
                        const std::string text(line.data(), used);
                        used = 0;
                        if (text == "start")
                            start_record = true;
                        else if (text == "save")
                            save = true;
                        else if (text == "pump")
                            toggle_pump = true;
                        else if (text == "assist")
                            toggle_assist = true;
                        else if (text == "excitation")
                            toggle_excitation = true;
                        else if (text == "quit")
                            stopped = 1;
                        else if (text.rfind("manual ", 0) == 0) {
                            std::istringstream input(text.substr(7));
                            manual = {};
                            for (unsigned j = 1; j <= 3; ++j)
                                if (!(input >> manual[j]) || !std::isfinite(manual[j]) ||
                                    std::abs(manual[j]) > 1)
                                    throw std::runtime_error(
                                        "manual requires three finite values in [-1,1]");
                            input >> std::ws;
                            if (!input.eof())
                                throw std::runtime_error("trailing manual command data");
                        } else if (!text.empty())
                            throw std::runtime_error("unknown scripted input: " + text);
                    }
                }
                if (stopped)
                    break;
                if (capture.reset) {
                    capture.reset = false;
                    pause();
                    pump = false;
                    record_begin = ns();
                    saved_block = UINT64_MAX;
                    event_reason = "imu_epoch_reset";
                }
                if (capture.failed || writer.state() == mlp::WriterState::Failed) {
                    throw std::runtime_error(
                        "PID recording failed; incomplete strip retained, restart collector");
                }
                if (save || (start_record && capture.recording))
                    finish(true);
                else if (start_record)
                    begin();
                if (capture.recording && double(ns() - record_begin) / 1e9 >= o.record_seconds)
                    finish(true);
                if (toggle_pump) {
                    pump = !pump;
                    if (!pump)
                        pause();
                }
                const double now = double(ns()) / 1e9;
                const auto imu = parser.sample();
                const bool fresh = imu.valid && !parser.firmware_blocked && now >= imu.received &&
                                   now - imu.received <= .03;
                std::array<double, 4> angles{};
                collection::Angles q{}, velocity{};
                if (fresh) {
                    angles = joint_angles(runtime.robot.hardware, imu.quats);
                    for (unsigned j = 0; j < 3; ++j) {
                        q[j] = angles[j + 1];
                        const auto &joint = runtime.robot.hardware.joints[j + 1];
                        velocity[j] =
                            (imu.motion[joint.sensor][1] - imu.motion[joint.parent_sensor][1]) *
                            pi / 180;
                    }
                }
                if (!fresh || !input_valid || !capture.recording || !pump || moving_sticks(manual))
                    pause();
                if (toggle_assist) {
                    if (active)
                        pause();
                    else if (fresh && input_valid && capture.recording && pump &&
                             !moving_sticks(manual)) {
                        controller.reset();
                        active = true;
                    } else {
                        event_reason = "pid_start_requirements";
                        std::cerr << "[PID] Start requires a recording, pump on, fresh state and "
                                     "centered sticks\n";
                    }
                }
                if (toggle_excitation) {
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
                    frame.output =
                        controller.step(ns(), q, {signal[1], signal[2], signal[3]}, fresh, next);
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
                    frame.reason = event_reason            ? event_reason
                                   : !input_valid          ? "input_lost"
                                   : !fresh                ? "stale_feedback"
                                   : moving_sticks(manual) ? "manual_takeover"
                                                           : "idle";
                for (unsigned i = 0; i < 6; ++i)
                    combined[i] = std::clamp(manual[i] + assist[i] + signal[i], -1., 1.);
                const bool allow = fresh && input_valid && capture.recording && pump &&
                                   (!frame.active || frame.output.valid);
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
                const auto computed = ns();
                for (unsigned i = 0; i < runtime.robot.hardware.valves.size(); ++i) {
                    const auto &v = runtime.robot.hardware.valves[i];
                    unsigned index = 6;
                    for (unsigned j = 0; j < 6; ++j)
                        if (v.name == command_names[j])
                            index = j;
                    const double value = index < 6 ? command[index] : 0;
                    const bool limited = index >= 1 && index <= 3 && frame.guard.limited[index - 1];
                    pwm.pulses[v.channel] = collection::guarded_valve_pulse(
                        v, value, limited, double(computed) / 1e9, ramps[i]);
                }
                pwm.pulses[runtime.robot.hardware.pump_channel] =
                    pump ? runtime.robot.hardware.pump_run : runtime.robot.hardware.pump_min;
                pwm.flush();
                const auto written = ns();
                if (capture.recording) {
                    if (saved_block != excitation.block().id) {
                        mlp::Record block;
                        block.kind = mlp::RecordKind::Block;
                        block.block = excitation.block();
                        if (!writer.push(block))
                            capture.failed = true;
                        saved_block = excitation.block().id;
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
                    s.command_age_s = double(written - computed) / 1e9;
                    s.state_age_s =
                        imu.valid ? std::max(0., double(written) / 1e9 - imu.received) : INFINITY;
                    s.imu_valid = fresh && s.state_age_s <= .03;
                    s.input_valid = input_valid;
                    s.write_ok = true;
                    s.pump_on = pump;
                    s.controller_valid = frame.guard.valid && (!frame.active || frame.output.valid);
                    s.source = frame.active            ? mlp::DriveSource::Controller
                               : moving_sticks(manual) ? mlp::DriveSource::Manual
                                                       : mlp::DriveSource::Idle;
                    s.checksum_errors = parser.checksum_errors;
                    s.invalid_frames = parser.invalid_frames;
                    s.resets = parser.resets;
                    const auto &b = excitation.block();
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
                    if (capture.failed)
                        throw std::runtime_error(
                            "PID recording queue failed; incomplete strip retained");
                }
                ++tick;
                next += 10000000;
                const auto end = ns();
                if (end >= next_coverage) {
                    print_coverage();
                    next_coverage = end + 30000000000ULL;
                }
                if (end >= next) {
                    const auto skipped = (end - next) / 10000000 + 1;
                    missed += skipped;
                    tick += skipped;
                    next += skipped * 10000000;
                }
                timespec wake{time_t(next / 1000000000), long(next % 1000000000)};
                while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr) == EINTR &&
                       !stopped) {
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
        print_coverage();
        std::cout << "[PID] Finished: ticks=" << tick << " missed=" << missed
                  << " dropped=" << writer.dropped() << " queue_high_water=" << writer.high_water()
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "PID collector: " << e.what() << '\n';
        return 1;
    }
}
