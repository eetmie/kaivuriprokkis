#include "kaivuri/core.hpp"
#include "kaivuri/pwm.hpp"
#include "kaivuri/util.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>
using namespace kaivuri;

namespace {
constexpr const char *usage =
    "kaivuri-control --profile DIR --safety FILE [--serial DEVICE | --simulate-imu]\n"
    "  [--socket PATH] [--live-pwm] [--neutralize] [--enable-slew] [--enable-tracks]\n"
    "  [--duration-s SEC] [--cpu N] [--rt-priority 1..80] [--lock-memory] [--timing-csv FILE] "
    "[--control-hz 50..500]\n"
    "Timing requires a finite duration; records in memory, writes after neutral.\n"
    "Defaults: dry PWM, real IMUs, disarmed; simulation is forbidden with live PWM.";

struct Client {
    Fd fd;
    uint64_t sequence = 0;
    bool seen = false;
};
struct TickTiming {
    double scheduled, start, finish, fk_work_us;
};
// One coherent observation per tick, reused by every status client on this tick. Unhealthy
// ticks keep the defaults (zero angles, identity FK).
struct Tick {
    double now = 0, fk_work_us = 0;
    bool healthy = false;
    Observation obs;
    std::array<Quat, 4> imu_raw{}, imu_corrected{}; // per joint, from that joint's IMU
    std::array<Vec3, 4> imu_gyro{}, imu_accel{};
};
const char *boolean(bool value) {
    return value ? "true" : "false";
}
// Stack-only status serialization: bounded storage and no allocations in the control loop. On
// overflow ok turns false and the reply is dropped.
struct StatusReply {
    char bytes[8192]{};
    size_t length = 0;
    bool ok = true;
    void append(const char *text) {
        const size_t n = std::strlen(text);
        ok = ok && n < sizeof(bytes) - length;
        if (ok) {
            std::memcpy(bytes + length, text, n + 1);
            length += n;
        }
    }
    template <class... T> void append(const char *format, T... args) {
        if (!ok)
            return;
        const size_t room = sizeof(bytes) - length;
        const int n = std::snprintf(bytes + length, room, format, args...);
        ok = n >= 0 && static_cast<size_t>(n) < room;
        if (ok)
            length += static_cast<size_t>(n);
    }
    void key(const char *name) { append(",\"%s\":", name); }
    template <class T> void list(const T *values, size_t count, const char *format = "%.9f") {
        append("[");
        for (size_t i = 0; i < count; ++i) {
            if (i)
                append(",");
            append(format, static_cast<double>(values[i]));
        }
        append("]");
    }
    void scalar(const char *name, double value, bool valid) {
        key(name);
        if (valid)
            append("%.9f", value);
        else
            append("null");
    }
    template <class T, size_t N>
    void vector(const char *name, const std::array<T, N> &values, bool valid = true) {
        key(name);
        if (valid)
            list(values.data(), N);
        else
            append("null");
    }
    template <class T, size_t N>
    void matrix(const char *name, const std::array<std::array<T, N>, 4> &rows, bool valid) {
        key(name);
        if (!valid)
            return append("null");
        append("[");
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i)
                append(",");
            list(rows[i].data(), N);
        }
        append("]");
    }
};
} // namespace

int main(int argc, char **argv) {
    try {
        std::string profile = "../configuration_files/profiles/jetson",
                    safety = "config/safety.yaml", serial = "/dev/ttyACM0", socket_path,
                    timing_path;
        bool simulate = false, live = false, neutralize = false, slew = false, tracks = false,
             lock_memory = false;
        double duration = 0, control_hz = 0;
        int cpu = -1, priority = 0;
        for (Args args(argc, argv); args.next();) {
            if (args.is("--profile"))
                profile = args.value();
            else if (args.is("--safety"))
                safety = args.value();
            else if (args.is("--socket"))
                socket_path = args.value();
            else if (args.is("--serial"))
                serial = args.value();
            else if (args.is("--simulate-imu"))
                simulate = true;
            else if (args.is("--live-pwm"))
                live = true;
            else if (args.is("--neutralize"))
                neutralize = true;
            else if (args.is("--enable-slew"))
                slew = true;
            else if (args.is("--enable-tracks"))
                tracks = true;
            else if (args.is("--lock-memory"))
                lock_memory = true;
            else if (args.is("--duration-s"))
                duration = args.number();
            else if (args.is("--timing-csv"))
                timing_path = args.value();
            else if (args.is("--control-hz"))
                control_hz = args.number();
            else if (args.is("--cpu"))
                cpu = args.integer();
            else if (args.is("--rt-priority"))
                priority = args.integer();
            else if (args.is("--help")) {
                std::puts(usage);
                return 0;
            } else
                args.unknown();
        }
        require(!(simulate && live), "simulated IMUs cannot drive physical PWM");
        require(duration >= 0 && std::isfinite(duration) && cpu < CPU_SETSIZE && cpu >= -1 &&
                    priority >= 0 && priority <= 80 && std::isfinite(control_hz) &&
                    (control_hz == 0 || (control_hz >= 50 && control_hz <= 500)),
                "invalid runtime option");
        Config cfg = load_config(profile, safety);
        if (control_hz != 0)
            cfg.control_hz = control_hz;
        cfg.enable_slew = slew;
        cfg.enable_tracks = tracks;
        // Timing is recorded in memory and written after neutral: no file writes in the loop.
        std::vector<TickTiming> timings;
        std::ofstream timing_file;
        if (!timing_path.empty()) {
            const double count = std::ceil(duration * cfg.control_hz) + 256;
            require(duration > 0 && count <= 1000000,
                    "timing CSV requires a bounded duration (at most one million ticks)");
            require(!std::filesystem::exists(timing_path), "timing CSV already exists");
            timing_file.open(timing_path);
            require(bool(timing_file), "cannot open timing CSV");
            timings.resize(static_cast<size_t>(count)); // allocate and touch before mlockall
        }
        size_t timing_count = 0;
        ControlCore core(cfg);
        const auto runtime = runtime_dir();
        if (socket_path.empty())
            socket_path = (runtime / "control.sock").string();
        // Hardware lock is independent of the socket path: two native daemons cannot own PWM.
        // It also creates and checks the runtime directory.
        ControlLock ownership(runtime.string());
        Pwm pwm(cfg, live);
        if (neutralize) {
            pwm.neutral();
            std::puts(live ? "Physical neutral written."
                           : "Dry neutral only; use --live-pwm for hardware.");
            return 0;
        }
        const Fd imu = simulate ? Fd() : open_serial(serial, 115200);
        Fd server(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
        require_errno(bool(server), "socket");
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        require(socket_path.size() < sizeof(addr.sun_path), "socket path too long");
        std::strcpy(addr.sun_path, socket_path.c_str());
        ::unlink(socket_path.c_str());
        require_errno(bind(server.get(), reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0,
                      "bind socket");
        require_errno(chmod(socket_path.c_str(), 0600) == 0, "socket permissions");
        require_errno(listen(server.get(), 8) == 0, "listen");
        struct SocketCleanup {
            std::string path;
            ~SocketCleanup() { ::unlink(path.c_str()); }
        } socket_cleanup{socket_path};
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            require_errno(sched_setaffinity(0, sizeof(set), &set) == 0, "CPU affinity");
        }
        if (priority) {
            sched_param p{};
            p.sched_priority = priority;
            require_errno(sched_setscheduler(0, SCHED_FIFO, &p) == 0, "realtime priority");
        }
        if (lock_memory)
            require_errno(mlockall(MCL_CURRENT | MCL_FUTURE) == 0, "lock memory");
        install_stop_handlers();
        std::signal(SIGPIPE, SIG_IGN);
        std::fprintf(stderr, "Control %s; %s; socket %s; slew=%d tracks=%d\n",
                     live ? "LIVE PWM" : "DRY PWM", simulate ? "SIMULATED IMU" : "real IMU",
                     socket_path.c_str(), slew, tracks);

        // Everything below runs in the realtime loop: stack and preallocated storage only.
        ImuParser parser(4);
        std::array<Client, 8> clients{};
        std::array<RampState, 8> ramps{};
        std::array<uint8_t, max_packet> bytes{};
        int owner = -1; // client index that holds the drive lease
        Commands applied_command{};
        uint64_t applied_sequence = 0, applied_monotonic_ns = 0;
        uint64_t ticks = 0, last_resets = 0, last_invalid = 0, last_command = 0;
        double worst_lateness = 0, begin = monotonic_seconds(), next = begin, previous = begin;
        const auto close_client = [&](int index) {
            if (owner == index) {
                core.fault("producer_disconnected");
                owner = -1;
            }
            clients[index] = Client{};
        };
        const auto status = [&](StatusReply &reply, const Tick &t) {
            const auto &sample = parser.sample();
            const auto &fk = t.obs.fk;
            const auto &att = t.obs.attitude;
            double imu_age = -1;
            if (simulate)
                imu_age = 0;
            else if (sample.received)
                imu_age = t.now - sample.received;
            using ull = unsigned long long;
            reply.append("{\"state\":\"%s\",\"reason\":\"%s\",\"live_pwm\":%s,\"simulated_imu\":%s,"
                         "\"imu_healthy\":%s,\"imu_age_s\":%.6f,\"imu_sequence\":%llu,"
                         "\"imu_device_us\":%llu,\"checksum_errors\":%llu,"
                         "\"invalid_imu_frames\":%llu,\"pump_on\":%s,\"command_sequence\":%llu,"
                         "\"ticks\":%llu,\"worst_lateness_s\":%.6f,\"host_monotonic_ns\":%llu,"
                         "\"imu_host_monotonic_ns\":%llu,\"firmware_blocked\":%s",
                         state_name(core.state()), core.reason(), boolean(live), boolean(simulate),
                         boolean(t.healthy), imu_age, ull(sample.sequence), ull(sample.device_us),
                         ull(parser.checksum_errors), ull(parser.invalid_frames),
                         boolean(core.pump_on()), ull(last_command), ull(ticks), worst_lateness,
                         ull(t.now * 1e9), ull((simulate ? t.now : sample.received) * 1e9),
                         boolean(parser.firmware_blocked));
            reply.vector("angles_rad", t.obs.angles);
            reply.vector("fk_xyz_m", fk.base.position);
            reply.vector("fk_quat_wxyz", fk.base.orientation);
            reply.key("pulses_us");
            reply.list(pwm.pulses.data(), 9, "%.3f"); // PCA9685 channels 0-8
            reply.append(",\"control_hz\":%.3f,\"fk_work_us\":%.3f,\"fk_valid\":%s,"
                         "\"fk_frame\":\"base\",\"fk_upper_frame\":\"upper_carriage\","
                         "\"joint_names\":[\"slew\",\"boom\",\"arm\",\"bucket\"],"
                         "\"imu_sensor_indices\":[%d,%d,%d,%d]",
                         cfg.control_hz, t.fk_work_us, boolean(t.healthy), cfg.joints[0].sensor,
                         cfg.joints[1].sensor, cfg.joints[2].sensor, cfg.joints[3].sensor);
            reply.vector("applied_action", applied_command);
            reply.append(",\"applied_command_sequence\":%llu,\"applied_host_monotonic_ns\":%llu,"
                         "\"enable_slew\":%s,\"enable_tracks\":%s",
                         ull(applied_sequence), ull(applied_monotonic_ns), boolean(cfg.enable_slew),
                         boolean(cfg.enable_tracks));
            const bool valid = t.healthy;
            reply.vector("fk_upper_xyz_m", fk.upper.position, valid);
            reply.vector("fk_upper_xz_m",
                         std::array<double, 2>{fk.upper.position[0], fk.upper.position[2]}, valid);
            reply.vector("fk_upper_quat_wxyz", fk.upper.orientation, valid);
            reply.scalar("slew_yaw_rad", fk.slew_yaw, valid);
            reply.scalar("bucket_pitch_rad", fk.bucket_pitch, valid);
            reply.matrix("fk_joint_xyz_m", fk.joint_positions_base, valid);
            reply.matrix("fk_joint_xyz_upper_m", fk.joint_positions_upper, valid);
            reply.matrix("fk_joint_quat_wxyz", fk.joint_rotations, valid);
            reply.matrix("fk_link_quat_wxyz", fk.link_orientations_base, valid);
            reply.matrix("fk_link_quat_upper_wxyz", fk.link_orientations_upper, valid);
            reply.append(",\"fk_cabin_tilt\":%s,\"fk_link_roll\":%s", boolean(cfg.level.cabin_tilt),
                         boolean(cfg.level.link_roll));
            reply.scalar("cabin_roll_rad", att.cabin_roll, valid);
            reply.scalar("cabin_pitch_rad", att.cabin_pitch, valid);
            reply.vector("cabin_tilt_quat_wxyz", att.tilt, valid);
            reply.vector("link_roll_rad", att.link_roll, valid);
            reply.matrix("imu_quat_wxyz", t.imu_raw, valid);
            reply.matrix("imu_corrected_quat_wxyz", t.imu_corrected, valid);
            reply.matrix("imu_gyro_dps", t.imu_gyro, valid);
            reply.matrix("imu_accel_g", t.imu_accel, valid);
            reply.append("}");
        };
        // Handles up to 8 queued messages from one client.
        const auto serve = [&](int ci, const Tick &t) {
            auto &client = clients[ci];
            for (int message = 0; message < 8; ++message) {
                const ssize_t n =
                    recv(client.fd.get(), bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_TRUNC);
                if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR))
                    return close_client(ci);
                if (n < 0)
                    return;
                Packet p;
                if (static_cast<size_t>(n) > bytes.size() ||
                    !decode_packet(bytes.data(), n, t.now, cfg, p) ||
                    (client.seen && p.sequence <= client.sequence)) {
                    if (owner == ci)
                        core.fault("invalid_or_replayed_packet");
                    continue;
                }
                client.seen = true;
                client.sequence = p.sequence;
                if (p.op == Operation::Status) {
                    StatusReply reply;
                    status(reply, t);
                    if (reply.ok)
                        send(client.fd.get(), reply.bytes, reply.length,
                             MSG_DONTWAIT | MSG_NOSIGNAL);
                    continue;
                }
                if (p.op == Operation::Drive) {
                    if (owner >= 0 && owner != ci)
                        continue;
                    owner = ci;
                    last_command = p.sequence;
                }
                core.receive(p, t.now, t.healthy);
                if (p.op == Operation::Disarm)
                    owner = -1;
            }
        };

        while (!stop_requested) {
            Tick t;
            t.now = monotonic_seconds();
            const double now = t.now;
            if (duration > 0 && now - begin >= duration)
                break;
            worst_lateness = std::max(worst_lateness, now - next);
            if (now - previous > cfg.deadline)
                core.fault("control_deadline_missed");
            previous = now;
            if (imu) {
                uint8_t chunk[512];
                for (int j = 0; j < 4; ++j) {
                    const ssize_t n = ::read(imu.get(), chunk, sizeof(chunk));
                    if (n <= 0) {
                        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                            core.fault("serial_io_error");
                        break;
                    }
                    parser.feed(chunk, n, now);
                }
            }
            const auto &sample = parser.sample();
            t.healthy = simulate || (!parser.firmware_blocked && sample.valid &&
                                     now - sample.received <= cfg.imu_timeout);
            if (t.healthy) {
                const double fk_begin = monotonic_seconds();
                auto raw = sample.quats;
                if (simulate)
                    raw.fill({1, 0, 0, 0});
                t.obs = observe(cfg, raw);
                for (int i = 0; i < 4; ++i) {
                    const int sensor = cfg.joints[i].sensor;
                    t.imu_raw[i] = raw[sensor];
                    t.imu_corrected[i] = t.obs.corrected[sensor];
                    if (simulate)
                        continue;
                    for (int j = 0; j < 3; ++j) {
                        t.imu_gyro[i][j] = sample.motion[sensor][j];
                        t.imu_accel[i][j] = sample.motion[sensor][j + 3];
                    }
                }
                t.fk_work_us = (monotonic_seconds() - fk_begin) * 1e6;
            }
            if (parser.resets != last_resets || parser.invalid_frames != last_invalid) {
                if (core.state() != SafetyState::Disarmed)
                    core.fault("imu_reset_or_invalid_sample");
                last_resets = parser.resets;
                last_invalid = parser.invalid_frames;
            }
            // Check expiry before processing a renewal arriving on this tick.
            core.tick(now, t.healthy);
            for (int accepted = 0; accepted < 8; ++accepted) {
                Fd fd(accept4(server.get(), nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC));
                if (!fd)
                    break;
                ucred cred{};
                socklen_t len = sizeof(cred);
                if (getsockopt(fd.get(), SOL_SOCKET, SO_PEERCRED, &cred, &len) != 0 ||
                    cred.uid != geteuid())
                    continue; // only this user's processes may connect
                for (auto &client : clients)
                    if (!client.fd) {
                        client.fd = std::move(fd);
                        break;
                    }
            }
            for (int ci = 0; ci < 8; ++ci)
                if (clients[ci].fd)
                    serve(ci, t);
            const Commands command = core.tick(now, t.healthy);
            const bool active =
                core.state() == SafetyState::Armed || core.state() == SafetyState::Decaying;
            for (size_t i = 0; i < cfg.valves.size(); ++i) {
                const auto &v = cfg.valves[i];
                double value = 0;
                for (int j = 0; j < 6; ++j)
                    if (v.name == command_names[j])
                        value = command[j];
                if (active)
                    pwm.pulses[v.channel] = valve_pulse(v, value, now, ramps[i]);
                else {
                    pwm.pulses[v.channel] = v.center;
                    ramps[i] = {v.center, now, true};
                }
            }
            pwm.pulses[cfg.pump_channel] = core.pump_on() ? cfg.pump_run : cfg.pump_min;
            try {
                pwm.flush();
            } catch (...) {
                core.fault("pwm_io_error");
                pwm.neutral();
                throw;
            }
            applied_command = active ? command : Commands{};
            applied_sequence = last_command;
            applied_monotonic_ns = static_cast<uint64_t>(monotonic_seconds() * 1e9);
            if (timing_count < timings.size())
                timings[timing_count++] = {next, now, monotonic_seconds(), t.fk_work_us};
            ++ticks;
            next += 1 / cfg.control_hz;
            if (next < now)
                next = now + 1 / cfg.control_hz;
            const timespec wake{static_cast<time_t>(next),
                                static_cast<long>((next - std::floor(next)) * 1e9)};
            int slept;
            do
                slept = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr);
            while (slept == EINTR && !stop_requested);
        }
        core.disarm();
        pwm.neutral();
        if (timing_file.is_open()) {
            timing_file << "tick,scheduled_ns,start_ns,finish_ns,lateness_ms,work_ms,fk_work_us\n"
                        << std::setprecision(12);
            for (size_t i = 0; i < timing_count; ++i) {
                const auto &t = timings[i];
                timing_file << i << ',' << static_cast<uint64_t>(t.scheduled * 1e9) << ','
                            << static_cast<uint64_t>(t.start * 1e9) << ','
                            << static_cast<uint64_t>(t.finish * 1e9) << ','
                            << std::max(0., t.start - t.scheduled) * 1000 << ','
                            << (t.finish - t.start) * 1000 << ',' << t.fk_work_us << '\n';
            }
            require(bool(timing_file), "timing CSV write failed");
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Control failed: %s\n", e.what());
        return 1;
    }
}
