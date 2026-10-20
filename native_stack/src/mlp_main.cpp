#include "kaivuri/gamepad.hpp"
#include "kaivuri/mlp_collect.hpp"
#include "kaivuri/pwm.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

using namespace kaivuri;
using namespace kaivuri::mlp;
namespace {
volatile sig_atomic_t stopped = 0;
void stop_signal(int) {
    stopped = 1;
}
uint64_t ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
struct Fd {
    int value;
    explicit Fd(int v) : value(v) {}
    ~Fd() {
        if (value >= 0)
            ::close(value);
    }
};
} // namespace
int main(int argc, char **argv) {
    try {
        const auto o = parse_options(argc, argv);
        auto runtime = load_runtime(o);
        auto excitation = std::make_unique<Excitation>(o.excitation, runtime.limits);
        ControlLock lock("/tmp/kaivuri-native-" + std::to_string(geteuid()));
        auto pwm = std::make_unique<Pwm>(runtime.config, !o.dry);
        Fd serial(::open(o.serial.c_str(), O_RDONLY | O_NONBLOCK | O_NOCTTY | O_CLOEXEC));
        if (serial.value < 0)
            throw std::runtime_error("cannot open serial: " + o.serial.string());
        if (isatty(serial.value)) {
            termios t{};
            if (tcgetattr(serial.value, &t) != 0)
                throw std::runtime_error("serial termios read failed");
            cfmakeraw(&t);
            t.c_cflag |= CLOCAL | CREAD;
            if (tcsetattr(serial.value, TCSANOW, &t) != 0 || ioctl(serial.value, TIOCEXCL) != 0)
                throw std::runtime_error("serial setup/exclusive lock failed");
        }
        std::unique_ptr<Pad> pad;
        if (!o.scripted)
            pad = std::make_unique<Pad>(o.gamepad);
        auto writer = std::make_unique<MlpWriter>(o.out, runtime.metadata, o.queue);
        ImuParser parser(4);
        RawCapture capture{writer.get(), &parser};
        parser.set_sample_sink(RawCapture::sink, &capture);
        signal(SIGINT, stop_signal);
        signal(SIGTERM, stop_signal);
        if (o.scripted)
            fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK);
        std::array<bool, 6> previous{};
        if (pad)
            previous = {pad->keys[BTN_SOUTH], pad->keys[BTN_EAST],    pad->keys[BTN_NORTH],
                        pad->keys[BTN_WEST],  pad->keys[BTN_DPAD_UP], pad->keys[BTN_DPAD_DOWN]};
        Commands manual{};
        bool input_valid = true, pump = false;
        uint64_t tick = 0, missed = 0, first = ns(), next_tick = first, record_begin = 0,
                 block_saved = 0;
        std::array<RampState, 8> ramps{};
        std::array<char, 1024> command_buffer{};
        size_t command_used = 0;
        std::cout << "MLP collector: 100 Hz, PWM " << (o.dry ? "DRY" : "LIVE") << ", seed "
                  << o.excitation.seed << "; A=record B=excitation X=pump Y=reload D-pad=target\n"
                  << std::flush;
        std::array<uint8_t, 8192> bytes{};
        auto finish = [&](bool complete) {
            excitation->disable();
            capture.recording = false;
            writer->finish(complete);
        };
        auto begin = [&] {
            if (!writer->start())
                return;
            const auto stamp = ns();
            excitation->reseed(o.fixed_seed ? o.excitation.seed : uint32_t(stamp ^ (stamp >> 32)),
                               double(stamp) / 1e9);
            capture.epoch = parser.sample().epoch;
            capture.last_raw = 0;
            capture.recording = true;
            record_begin = ns();
            block_saved = 0;
            if (parser.sample().valid)
                capture.packet(parser.sample());
        };
        if (o.record_on_start)
            begin();
        try {
            while (!stopped) {
                const uint64_t start_ns = ns();
                if (o.duration > 0 && double(start_ns - first) / 1e9 >= o.duration)
                    break;
                size_t read_total = 0;
                while (read_total < bytes.size()) {
                    const auto n = ::read(serial.value, bytes.data(),
                                          std::min(o.read_chunk, bytes.size() - read_total));
                    if (n > 0) {
                        parser.feed(bytes.data(), size_t(n), double(ns()) / 1e9);
                        read_total += size_t(n);
                    } else {
                        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                            throw std::runtime_error("serial read failed");
                        break;
                    }
                }
                const double now = double(ns()) / 1e9;
                bool start_record = false, save = false, toggle_excitation = false,
                     toggle_pump = false, reload = false;
                int target_step = 0;
                if (pad && input_valid) {
                    try {
                        pad->poll_events();
                        manual = pad->action(o.slew, o.tracks);
                        std::array<bool, 6> keys{pad->keys[BTN_SOUTH],   pad->keys[BTN_EAST],
                                                 pad->keys[BTN_NORTH],   pad->keys[BTN_WEST],
                                                 pad->keys[BTN_DPAD_UP], pad->keys[BTN_DPAD_DOWN]};
                        start_record = keys[0] && !previous[0];
                        toggle_excitation = keys[1] && !previous[1];
                        toggle_pump = keys[2] && !previous[2];
                        reload = keys[3] && !previous[3];
                        target_step = int(keys[4] && !previous[4]) - int(keys[5] && !previous[5]);
                        previous = keys;
                    } catch (const std::exception &e) {
                        input_valid = false;
                        manual = {};
                        excitation->disable();
                        std::cerr << "[MLP] " << e.what() << "; inputs centered\n";
                    }
                }
                if (o.scripted) {
                    char c;
                    for (unsigned n = 0; n < 1024 && ::read(STDIN_FILENO, &c, 1) == 1; ++n) {
                        if (c != '\n') {
                            if (command_used + 1 < command_buffer.size())
                                command_buffer[command_used++] = c;
                            continue;
                        }
                        const std::string command(command_buffer.data(), command_used);
                        command_used = 0;
                        if (command == "start")
                            start_record = true;
                        else if (command == "save")
                            save = true;
                        else if (command == "pump")
                            toggle_pump = true;
                        else if (command == "excitation")
                            toggle_excitation = true;
                        else if (command == "reload")
                            reload = true;
                        else if (command == "quit")
                            stopped = 1;
                        else if (command.rfind("target ", 0) == 0) {
                            const unsigned target = target_mask(command.substr(7));
                            if (target == 1 && !o.slew)
                                throw std::runtime_error("slew excitation requires --enable-slew");
                            excitation->select(target, now);
                        } else if (command.rfind("command ", 0) == 0) {
                            std::istringstream in(command.substr(8));
                            for (auto &value : manual)
                                if (!(in >> value) || !std::isfinite(value) || std::abs(value) > 1)
                                    throw std::runtime_error("invalid scripted command");
                        } else if (!command.empty())
                            throw std::runtime_error("unknown scripted input");
                    }
                }
                if (!o.slew)
                    manual[0] = 0;
                if (!o.tracks)
                    manual[4] = manual[5] = 0;
                if (capture.reset) {
                    capture.reset = false;
                    excitation->disable();
                    block_saved = 0;
                    record_begin = ns();
                }
                if (capture.failed || writer->state() == WriterState::Failed) {
                    capture.failed = false;
                    finish(false);
                }
                if (save || (start_record && capture.recording))
                    finish(true);
                else if (start_record)
                    begin();
                if (toggle_pump)
                    pump = !pump;
                if (toggle_excitation) {
                    if (excitation->enabled())
                        excitation->disable();
                    else
                        excitation->enable(now);
                }
                if (target_step) {
                    const std::array<unsigned, 8> targets{14, 2, 4, 8, 6, 10, 12, 1};
                    const int count = o.slew && o.excitation.waveform != Waveform::Stroke ? 8 : 7;
                    const auto index = std::find(targets.begin(), targets.end(),
                                                 excitation->block().options.target) -
                                       targets.begin();
                    excitation->select(targets[(int(index) + target_step + count) % count], now);
                }
                if (reload && !capture.recording && writer->state() == WriterState::Idle) {
                    pwm->neutral();
                    pump = false;
                    auto replacement = load_runtime(o);
                    auto next_excitation =
                        std::make_unique<Excitation>(o.excitation, replacement.limits);
                    writer->shutdown();
                    writer.reset();
                    pwm.reset();
                    runtime = std::move(replacement);
                    excitation = std::move(next_excitation);
                    pwm = std::make_unique<Pwm>(runtime.config, !o.dry);
                    writer = std::make_unique<MlpWriter>(o.out, runtime.metadata, o.queue);
                    capture.writer = writer.get();
                    ramps = {};
                }
                const auto &imu = parser.sample();
                const bool fresh =
                    imu.valid && !parser.firmware_blocked && now - imu.received <= .03;
                std::array<double, 4> angles{};
                if (fresh)
                    angles = joint_angles(runtime.config, imu.quats);
                const Commands signal = excitation->sample(now, angles, fresh);
                Commands combined{};
                for (unsigned i = 0; i < 6; ++i)
                    combined[i] = std::clamp(manual[i] + signal[i], -1., 1.);
                const uint64_t computed = ns();
                for (unsigned i = 0; i < runtime.config.valves.size(); ++i) {
                    const auto &v = runtime.config.valves[i];
                    double command = 0;
                    for (unsigned j = 0; j < 6; ++j)
                        if (v.name == command_names[j])
                            command = combined[j];
                    pwm->pulses[v.channel] =
                        valve_pulse(v, command, double(computed) / 1e9, ramps[i]);
                }
                pwm->pulses[runtime.config.pump_channel] =
                    pump ? runtime.config.pump_run : runtime.config.pump_min;
                bool ok = true;
                try {
                    pwm->flush();
                } catch (...) {
                    ok = false;
                }
                const uint64_t written = ns();
                if (capture.recording) {
                    if (block_saved != excitation->block().id) {
                        Record block;
                        block.kind = RecordKind::Block;
                        block.block = excitation->block();
                        if (!writer->push(block))
                            capture.failed = true;
                        block_saved = excitation->block().id;
                    }
                    Record r;
                    r.kind = RecordKind::Drive;
                    auto &s = r.sample;
                    s.imu = imu;
                    s.manual = manual;
                    s.excitation = signal;
                    s.command = combined;
                    s.angles = angles;
                    s.pulses = pwm->pulses;
                    s.tick = tick;
                    s.scheduled_ns = next_tick;
                    s.start_ns = start_ns;
                    s.post_write_ns = written;
                    s.command_age_s = double(written - computed) / 1e9;
                    s.state_age_s =
                        imu.valid ? std::max(0., double(written) / 1e9 - imu.received) : INFINITY;
                    s.imu_valid = fresh && s.state_age_s <= .03;
                    s.input_valid = input_valid;
                    s.write_ok = ok;
                    s.pump_on = pump;
                    s.missed_periods = missed;
                    s.checksum_errors = parser.checksum_errors;
                    s.invalid_frames = parser.invalid_frames;
                    s.resets = parser.resets;
                    const auto &b = excitation->block();
                    s.block_id = b.id;
                    s.seed = b.options.seed;
                    s.target = b.options.target;
                    s.waveform = b.options.waveform;
                    s.enabled = excitation->enabled();
                    s.stage = excitation->stage();
                    s.excitation_elapsed_s = excitation->elapsed(now);
                    s.noise_tick = excitation->noise_tick();
                    if (!writer->push(r))
                        capture.failed = true;
                }
                if (!ok)
                    throw std::runtime_error("PCA9685 write failed");
                if (capture.recording && written >= record_begin &&
                    double(written - record_begin) / 1e9 >= o.record_seconds) {
                    pwm->neutral();
                    pump = false;
                    finish(true);
                }
                ++tick;
                next_tick += 10000000;
                const uint64_t end = ns();
                if (end >= next_tick) {
                    const uint64_t skipped = (end - next_tick) / 10000000 + 1;
                    missed += skipped;
                    tick += skipped;
                    next_tick += skipped * 10000000;
                }
                timespec wake{time_t(next_tick / 1000000000), long(next_tick % 1000000000)};
                while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr) == EINTR &&
                       !stopped) {
                }
            }
            pwm->neutral();
            pump = false;
            finish(!capture.failed);
            writer->shutdown();
        } catch (...) {
            try {
                pwm->neutral();
            } catch (...) {
            }
            finish(false);
            writer->shutdown();
            throw;
        }
        std::cout << "[MLP] Finished: ticks=" << tick << " missed=" << missed
                  << " dropped=" << writer->dropped()
                  << " queue_high_water=" << writer->high_water() << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "MLP collector: " << e.what() << '\n';
        return 1;
    }
}
