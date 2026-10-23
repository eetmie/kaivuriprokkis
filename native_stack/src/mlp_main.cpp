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
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>

using namespace kaivuri;
using namespace kaivuri::mlp;
namespace {
constexpr uint64_t period_ns = 10000000; // 100 Hz
double seconds(uint64_t ns) {
    return double(ns) / 1e9;
}
// A, B, X, Y, D-pad up, D-pad down.
std::array<bool, 6> buttons(const Pad &pad) {
    return {pad.keys[BTN_SOUTH], pad.keys[BTN_EAST],    pad.keys[BTN_NORTH],
            pad.keys[BTN_WEST],  pad.keys[BTN_DPAD_UP], pad.keys[BTN_DPAD_DOWN]};
}
// Command index driving each valve (-1: none), resolved once per profile load
// instead of comparing valve names every tick.
using ValveCommands = std::array<int, std::tuple_size_v<decltype(Config::valves)>>;
ValveCommands valve_commands(const Config &config) {
    ValveCommands map;
    map.fill(-1);
    for (unsigned i = 0; i < map.size(); ++i)
        for (unsigned j = 0; j < command_names.size(); ++j)
            if (config.valves[i].name == command_names[j])
                map[i] = int(j);
    return map;
}
} // namespace
int main(int argc, char **argv) {
    try {
        const auto o = parse_options(argc, argv);
        auto runtime = load_runtime(o);
        auto excitation = std::make_unique<Excitation>(o.excitation, runtime.limits);
        ControlLock lock(runtime_dir().string());
        auto pwm = std::make_unique<Pwm>(runtime.config, !o.dry);
        auto valves = valve_commands(runtime.config);
        const Fd serial = open_serial(o.serial);
        std::unique_ptr<Pad> pad;
        if (!o.scripted)
            pad = std::make_unique<Pad>(o.gamepad);
        auto writer = std::make_unique<MlpWriter>(o.out, runtime.metadata, o.queue);
        ImuParser parser(4);
        RawCapture capture{writer.get(), &parser};
        parser.set_sample_sink(RawCapture::sink, &capture);
        install_stop_handlers();
        if (o.scripted)
            fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK);
        std::array<bool, 6> previous{};
        if (pad)
            previous = buttons(*pad);
        Commands manual{};
        bool input_valid = true, pump = false;
        uint64_t tick = 0, missed = 0, first = monotonic_ns(), next_tick = first, record_begin = 0,
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
            const auto stamp = monotonic_ns();
            excitation->reseed(o.fixed_seed ? o.excitation.seed : uint32_t(stamp ^ (stamp >> 32)),
                               seconds(stamp));
            capture.epoch = parser.sample().epoch;
            capture.last_raw = 0;
            capture.recording = true;
            record_begin = monotonic_ns();
            block_saved = 0;
            if (parser.sample().valid)
                capture.packet(parser.sample());
        };
        if (o.record_on_start)
            begin();
        try {
            while (!stop_requested) {
                const uint64_t start_ns = monotonic_ns();
                if (o.duration > 0 && seconds(start_ns - first) >= o.duration)
                    break;
                for (size_t read_total = 0; read_total < bytes.size();) {
                    const auto n = ::read(serial.get(), bytes.data(),
                                          std::min(o.read_chunk, bytes.size() - read_total));
                    if (n <= 0) {
                        require(n == 0 || errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR,
                                "serial read failed");
                        break;
                    }
                    parser.feed(bytes.data(), size_t(n), seconds(monotonic_ns()));
                    read_total += size_t(n);
                }
                const double now = seconds(monotonic_ns());
                bool start_record = false, save = false, toggle_excitation = false,
                     toggle_pump = false, reload = false;
                int target_step = 0;
                if (pad && input_valid) {
                    try {
                        pad->poll_events();
                        manual = pad->action(o.slew, o.tracks);
                        const auto keys = buttons(*pad);
                        auto pressed = [&](unsigned i) { return keys[i] && !previous[i]; };
                        start_record = pressed(0);
                        toggle_excitation = pressed(1);
                        toggle_pump = pressed(2);
                        reload = pressed(3);
                        target_step = int(pressed(4)) - int(pressed(5));
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
                            stop_requested = 1;
                        else if (command.rfind("target ", 0) == 0) {
                            const unsigned target = target_mask(command.substr(7));
                            require(target != 1 || o.slew,
                                    "slew excitation requires --enable-slew");
                            excitation->select(target, now);
                        } else if (command.rfind("command ", 0) == 0) {
                            std::istringstream in(command.substr(8));
                            for (auto &value : manual)
                                require(in >> value && std::isfinite(value) && std::abs(value) <= 1,
                                        "invalid scripted command");
                        } else
                            require(command.empty(), "unknown scripted input");
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
                    record_begin = monotonic_ns();
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
                    // Slew is last in the cycle and skipped unless enabled (never for stroke).
                    const int count = o.slew && o.excitation.waveform != Waveform::Stroke ? 8 : 7;
                    const auto index = std::find(target_masks.begin(), target_masks.end(),
                                                 excitation->block().options.target) -
                                       target_masks.begin();
                    excitation->select(target_masks[(int(index) + target_step + count) % count],
                                       now);
                }
                if (reload && !capture.recording && writer->state() == WriterState::Idle) {
                    pwm->neutral();
                    pump = false;
                    // Load and validate before tearing anything down.
                    auto replacement = load_runtime(o);
                    auto next_excitation =
                        std::make_unique<Excitation>(o.excitation, replacement.limits);
                    writer->shutdown();
                    writer.reset();
                    pwm.reset();
                    runtime = std::move(replacement);
                    excitation = std::move(next_excitation);
                    pwm = std::make_unique<Pwm>(runtime.config, !o.dry);
                    valves = valve_commands(runtime.config);
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
                const uint64_t computed = monotonic_ns();
                for (unsigned i = 0; i < valves.size(); ++i) {
                    const auto &v = runtime.config.valves[i];
                    const double command = valves[i] < 0 ? 0 : combined[valves[i]];
                    pwm->pulses[v.channel] = valve_pulse(v, command, seconds(computed), ramps[i]);
                }
                pwm->pulses[runtime.config.pump_channel] =
                    pump ? runtime.config.pump_run : runtime.config.pump_min;
                bool ok = true;
                try {
                    pwm->flush();
                } catch (...) {
                    ok = false;
                }
                const uint64_t written = monotonic_ns();
                if (capture.recording) {
                    const auto &b = excitation->block();
                    if (block_saved != b.id) {
                        Record block;
                        block.kind = RecordKind::Block;
                        block.block = b;
                        if (!writer->push(block))
                            capture.failed = true;
                        block_saved = b.id;
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
                    s.command_age_s = seconds(written - computed);
                    s.state_age_s =
                        imu.valid ? std::max(0., seconds(written) - imu.received) : INFINITY;
                    s.imu_valid = fresh && s.state_age_s <= .03;
                    s.input_valid = input_valid;
                    s.write_ok = ok;
                    s.pump_on = pump;
                    s.missed_periods = missed;
                    s.checksum_errors = parser.checksum_errors;
                    s.invalid_frames = parser.invalid_frames;
                    s.resets = parser.resets;
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
                require(ok, "PCA9685 write failed");
                if (capture.recording && written >= record_begin &&
                    seconds(written - record_begin) >= o.record_seconds) {
                    pwm->neutral();
                    pump = false;
                    finish(true);
                }
                ++tick;
                next_tick += period_ns;
                const uint64_t end = monotonic_ns();
                if (end >= next_tick) {
                    const uint64_t skipped = (end - next_tick) / period_ns + 1;
                    missed += skipped;
                    tick += skipped;
                    next_tick += skipped * period_ns;
                }
                timespec wake{time_t(next_tick / 1000000000), long(next_tick % 1000000000)};
                while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr) == EINTR &&
                       !stop_requested) {
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
