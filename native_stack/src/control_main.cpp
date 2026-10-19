#include "kaivuri/core.hpp"
#include "kaivuri/pwm.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <linux/i2c-dev.h>
#include <sched.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <termios.h>
#include <unistd.h>
#include <vector>
using namespace kaivuri;
static volatile sig_atomic_t stopped = 0;
static void stop_signal(int) {
    stopped = 1;
}
static void check(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}
struct Fd {
    int n = -1;
    Fd() = default;
    explicit Fd(int x) : n(x) {}
    ~Fd() {
        if (n >= 0)
            ::close(n);
    }
    Fd(const Fd &) = delete;
    Fd &operator=(const Fd &) = delete;
};
struct Client {
    int fd = -1;
    uint64_t sequence = 0;
    bool seen = false;
};
// Stack-only status serialization: bounded storage and no allocations in the control loop.
struct StatusReply {
    char bytes[8192]{};
    size_t length = 0;
    bool ok = true;
    void append(const char *text) {
        if (!ok)
            return;
        const size_t n = std::strlen(text);
        if (n >= sizeof(bytes) - length) {
            ok = false;
            return;
        }
        std::memcpy(bytes + length, text, n + 1);
        length += n;
    }
    template <class... Args> void append(const char *format, Args... args) {
        if (!ok)
            return;
        const int n = std::snprintf(bytes + length, sizeof(bytes) - length, format, args...);
        if (n < 0 || static_cast<size_t>(n) >= sizeof(bytes) - length) {
            ok = false;
            return;
        }
        length += static_cast<size_t>(n);
    }
    template <class T, size_t N>
    void vector(const char *key, const std::array<T, N> &values, bool valid) {
        append(",\"%s\":", key);
        if (!valid) {
            append("null");
            return;
        }
        append("[");
        for (size_t j = 0; j < N; ++j)
            append(j ? ",%.9f" : "%.9f", static_cast<double>(values[j]));
        append("]");
    }
    template <class T, size_t N>
    void matrix(const char *key, const std::array<std::array<T, N>, 4> &rows, bool valid) {
        append(",\"%s\":", key);
        if (!valid) {
            append("null");
            return;
        }
        append("[");
        for (size_t i = 0; i < rows.size(); ++i) {
            append(i ? ",[" : "[");
            for (size_t j = 0; j < N; ++j)
                append(j ? ",%.9f" : "%.9f", static_cast<double>(rows[i][j]));
            append("]");
        }
        append("]");
    }
};
int main(int argc, char **argv) {
    try {
        std::string profile = "../configuration_files/profiles/jetson",
                    safety = "config/safety.yaml", serial = "/dev/ttyACM0";
        std::string runtime = "/tmp/kaivuri-native-" + std::to_string(geteuid()), socket_path,
                    timing_path;
        bool simulate = false, live = false, neutralize = false, slew = false, tracks = false,
             lock_memory = false;
        double duration = 0, control_hz = 0;
        int cpu = -1, priority = 0;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto value = [&]() {
                if (++i == argc)
                    throw std::runtime_error("missing argument: " + a);
                return std::string(argv[i]);
            };
            if (a == "--profile")
                profile = value();
            else if (a == "--safety")
                safety = value();
            else if (a == "--socket")
                socket_path = value();
            else if (a == "--serial")
                serial = value();
            else if (a == "--simulate-imu")
                simulate = true;
            else if (a == "--live-pwm")
                live = true;
            else if (a == "--neutralize")
                neutralize = true;
            else if (a == "--enable-slew")
                slew = true;
            else if (a == "--enable-tracks")
                tracks = true;
            else if (a == "--lock-memory")
                lock_memory = true;
            else if (a == "--duration-s")
                duration = std::stod(value());
            else if (a == "--timing-csv")
                timing_path = value();
            else if (a == "--control-hz")
                control_hz = std::stod(value());
            else if (a == "--cpu")
                cpu = std::stoi(value());
            else if (a == "--rt-priority")
                priority = std::stoi(value());
            else if (a == "--help") {
                std::puts(
                    "kaivuri-control --profile DIR --safety FILE [--serial DEVICE | "
                    "--simulate-imu]\n  [--socket PATH] [--live-pwm] [--neutralize] "
                    "[--enable-slew] [--enable-tracks]\n  [--duration-s SEC] [--cpu N] "
                    "[--rt-priority 1..80] [--lock-memory] [--timing-csv FILE] [--control-hz "
                    "50..500]\n"
                    "Timing requires a finite duration; records in memory, writes after neutral.\n"
                    "Defaults: dry PWM, real IMUs, "
                    "disarmed; simulation is forbidden with live PWM.");
                return 0;
            } else
                throw std::runtime_error("unknown option: " + a);
        }
        if (simulate && live)
            throw std::runtime_error("simulated IMUs cannot drive physical PWM");
        if (duration < 0 || !std::isfinite(duration) || cpu >= CPU_SETSIZE || cpu < -1 ||
            priority < 0 || priority > 80 || !std::isfinite(control_hz) ||
            (control_hz != 0 && (control_hz < 50 || control_hz > 500)))
            throw std::runtime_error("invalid runtime option");
        Config cfg = load_config(profile, safety);
        if (control_hz != 0)
            cfg.control_hz = control_hz;
        struct TickTiming {
            double scheduled, start, finish, fk_work_us;
        };
        std::vector<TickTiming> timings;
        std::ofstream timing_file;
        if (!timing_path.empty()) {
            const double count = std::ceil(duration * cfg.control_hz) + 256;
            if (duration <= 0 || count > 1000000)
                throw std::runtime_error(
                    "timing CSV requires a bounded duration (at most one million ticks)");
            if (std::filesystem::exists(timing_path))
                throw std::runtime_error("timing CSV already exists");
            timing_file.open(timing_path);
            if (!timing_file)
                throw std::runtime_error("cannot open timing CSV");
            // Allocate and touch before optional mlockall; no file writes in the loop.
            timings.resize(static_cast<size_t>(count));
        }
        size_t timing_count = 0;
        cfg.enable_slew = slew;
        cfg.enable_tracks = tracks;
        ControlCore core(cfg);
        if (socket_path.empty())
            socket_path = runtime + "/control.sock";
        std::filesystem::create_directories(runtime);
        struct stat runtime_stat{};
        check(lstat(runtime.c_str(), &runtime_stat) == 0 && S_ISDIR(runtime_stat.st_mode) &&
                  runtime_stat.st_uid == geteuid(),
              "unsafe runtime directory");
        check(chmod(runtime.c_str(), 0700) == 0, "runtime permissions");
        // Hardware lock is independent of the socket path: two native daemons cannot own PWM.
        ControlLock ownership(runtime);
        Pwm pwm(cfg, live);
        if (neutralize) {
            pwm.neutral();
            std::puts(live ? "Physical neutral written."
                           : "Dry neutral only; use --live-pwm for hardware.");
            return 0;
        }
        Fd imu;
        if (!simulate) {
            imu.n = ::open(serial.c_str(), O_RDONLY | O_NONBLOCK | O_NOCTTY | O_CLOEXEC);
            check(imu.n >= 0, "open IMU serial");
            check(ioctl(imu.n, TIOCEXCL) == 0, "exclusive IMU serial");
            termios t{};
            check(tcgetattr(imu.n, &t) == 0, "get serial settings");
            cfmakeraw(&t);
            cfsetispeed(&t, B115200);
            cfsetospeed(&t, B115200);
            t.c_cflag |= CLOCAL | CREAD;
            check(tcsetattr(imu.n, TCSANOW, &t) == 0, "set serial settings");
        }
        Fd server(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
        check(server.n >= 0, "socket");
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        if (socket_path.size() >= sizeof(addr.sun_path))
            throw std::runtime_error("socket path too long");
        std::strcpy(addr.sun_path, socket_path.c_str());
        ::unlink(socket_path.c_str());
        check(bind(server.n, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0,
              "bind socket");
        check(chmod(socket_path.c_str(), 0600) == 0, "socket permissions");
        check(listen(server.n, 8) == 0, "listen");
        struct SocketCleanup {
            std::string path;
            ~SocketCleanup() { ::unlink(path.c_str()); }
        } socket_cleanup{socket_path};
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            check(sched_setaffinity(0, sizeof(set), &set) == 0, "CPU affinity");
        }
        if (priority) {
            sched_param p{};
            p.sched_priority = priority;
            check(sched_setscheduler(0, SCHED_FIFO, &p) == 0, "realtime priority");
        }
        if (lock_memory)
            check(mlockall(MCL_CURRENT | MCL_FUTURE) == 0, "lock memory");
        signal(SIGINT, stop_signal);
        signal(SIGTERM, stop_signal);
        signal(SIGPIPE, SIG_IGN);
        std::fprintf(stderr, "Control %s; %s; socket %s; slew=%d tracks=%d\n",
                     live ? "LIVE PWM" : "DRY PWM", simulate ? "SIMULATED IMU" : "real IMU",
                     socket_path.c_str(), slew, tracks);
        ImuParser parser(4);
        std::array<Client, 8> clients{};
        std::array<RampState, 8> ramps{};
        std::array<uint8_t, max_packet> bytes{};
        int owner = -1;
        Commands applied_command{};
        uint64_t applied_sequence = 0, applied_monotonic_ns = 0;
        uint64_t ticks = 0, last_resets = 0, last_invalid = 0, last_command = 0;
        double worst_lateness = 0, begin = monotonic_seconds(), next = begin, previous = begin;
        auto close_client = [&](int index) {
            if (owner == index) {
                core.fault("producer_disconnected");
                owner = -1;
            }
            ::close(clients[index].fd);
            clients[index] = Client{};
        };
        while (!stopped) {
            double now = monotonic_seconds();
            if (duration > 0 && now - begin >= duration)
                break;
            worst_lateness = std::max(worst_lateness, std::max(0., now - next));
            if (now - previous > cfg.deadline)
                core.fault("control_deadline_missed");
            previous = now;
            if (!simulate) {
                uint8_t chunk[512];
                for (int j = 0; j < 4; ++j) {
                    ssize_t n = ::read(imu.n, chunk, sizeof(chunk));
                    if (n > 0)
                        parser.feed(chunk, n, now);
                    else {
                        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                            core.fault("serial_io_error");
                        break;
                    }
                }
            }
            const auto &sample = parser.sample();
            bool healthy = simulate || (!parser.firmware_blocked && sample.valid &&
                                        now - sample.received <= cfg.imu_timeout);
            // One coherent observation per tick, reused by every status client on this tick.
            std::array<double, 4> angles{};
            KinematicState fk;
            std::array<Quat, 4> imu_raw{}, imu_corrected{};
            std::array<Vec3, 4> imu_gyro{}, imu_accel{};
            double fk_work_us = 0;
            if (healthy) {
                const double fk_begin = monotonic_seconds();
                auto raw = sample.quats;
                if (simulate)
                    for (auto &q : raw)
                        q = {1, 0, 0, 0};
                angles = joint_angles(cfg, raw);
                fk = forward_kinematic_state(cfg, angles);
                const auto corrected = corrected_imu_quaternions(cfg, raw);
                for (int i = 0; i < 4; ++i) {
                    const int sensor = cfg.joints[i].sensor;
                    imu_raw[i] = raw[sensor];
                    imu_corrected[i] = corrected[sensor];
                    for (int j = 0; j < 3; ++j) {
                        imu_gyro[i][j] = simulate ? 0 : sample.motion[sensor][j];
                        imu_accel[i][j] = simulate ? 0 : sample.motion[sensor][j + 3];
                    }
                }
                fk_work_us = (monotonic_seconds() - fk_begin) * 1e6;
            }
            if (parser.resets != last_resets || parser.invalid_frames != last_invalid) {
                if (core.state() != SafetyState::Disarmed)
                    core.fault("imu_reset_or_invalid_sample");
                last_resets = parser.resets;
                last_invalid = parser.invalid_frames;
            }
            // Check expiry before processing a renewal arriving on this tick.
            core.tick(now, healthy);
            for (int accepted = 0; accepted < 8; ++accepted) {
                int fd = accept4(server.n, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
                if (fd < 0)
                    break;
                ucred cred{};
                socklen_t len = sizeof(cred);
                bool placed = false;
                if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) == 0 &&
                    cred.uid == geteuid())
                    for (auto &client : clients)
                        if (client.fd < 0) {
                            client.fd = fd;
                            placed = true;
                            break;
                        }
                if (!placed)
                    ::close(fd);
            }
            for (int ci = 0; ci < 8; ++ci)
                if (clients[ci].fd >= 0)
                    for (int message = 0; message < 8; ++message) {
                        ssize_t n = recv(clients[ci].fd, bytes.data(), bytes.size(),
                                         MSG_DONTWAIT | MSG_TRUNC);
                        if (n == 0) {
                            close_client(ci);
                            break;
                        }
                        if (n < 0) {
                            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                                close_client(ci);
                            break;
                        }
                        Packet p;
                        if (static_cast<size_t>(n) > bytes.size() ||
                            !decode_packet(bytes.data(), n, now, cfg, p) ||
                            (clients[ci].seen && p.sequence <= clients[ci].sequence)) {
                            if (owner == ci)
                                core.fault("invalid_or_replayed_packet");
                            continue;
                        }
                        clients[ci].seen = true;
                        clients[ci].sequence = p.sequence;
                        if (p.op == Operation::Status) {
                            const auto &pose = fk.base;
                            StatusReply reply;
                            reply.append(
                                "{\"state\":\"%s\",\"reason\":\"%s\",\"live_pwm\":%s,\"simulated_"
                                "imu\":%s,\"imu_healthy\":%s,\"imu_age_s\":%.6f,\"imu_sequence\":%"
                                "llu,\"imu_device_us\":%llu,\"checksum_errors\":%llu,\"invalid_imu_"
                                "frames\":%llu,\"pump_on\":%s,\"command_sequence\":%llu,\"ticks\":%"
                                "llu,\"worst_lateness_s\":%.6f,\"host_monotonic_ns\":%llu,\"imu_"
                                "host_monotonic_ns\":%llu,\"firmware_blocked\":%s,\"angles_rad\":[%"
                                ".9f,%.9f,%.9f,%.9f]"
                                ",\"fk_xyz_m\":[%.9f,%.9f,%.9f],\"fk_quat_wxyz\":[%.9f,%.9f,%.9f,%."
                                "9f],\"pulses_us\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]",
                                state_name(core.state()), core.reason(), live ? "true" : "false",
                                simulate ? "true" : "false", healthy ? "true" : "false",
                                simulate          ? 0
                                : sample.received ? now - sample.received
                                                  : -1.,
                                (unsigned long long)sample.sequence,
                                (unsigned long long)sample.device_us,
                                (unsigned long long)parser.checksum_errors,
                                (unsigned long long)parser.invalid_frames,
                                core.pump_on() ? "true" : "false", (unsigned long long)last_command,
                                (unsigned long long)ticks, worst_lateness,
                                (unsigned long long)(now * 1e9),
                                (unsigned long long)((simulate ? now : sample.received) * 1e9),
                                parser.firmware_blocked ? "true" : "false", angles[0], angles[1],
                                angles[2], angles[3], pose.position[0], pose.position[1],
                                pose.position[2], pose.orientation[0], pose.orientation[1],
                                pose.orientation[2], pose.orientation[3], pwm.pulses[0],
                                pwm.pulses[1], pwm.pulses[2], pwm.pulses[3], pwm.pulses[4],
                                pwm.pulses[5], pwm.pulses[6], pwm.pulses[7], pwm.pulses[8]);
                            reply.append(",\"control_hz\":%.3f,\"fk_work_us\":%.3f,\"fk_valid\":%s,"
                                         "\"fk_frame\":\"base\","
                                         "\"fk_upper_frame\":\"upper_carriage\","
                                         "\"joint_names\":[\"slew\",\"boom\",\"arm\",\"bucket\"],"
                                         "\"imu_sensor_indices\":[%d,%d,%d,%d]",
                                         cfg.control_hz, fk_work_us, healthy ? "true" : "false",
                                         cfg.joints[0].sensor, cfg.joints[1].sensor,
                                         cfg.joints[2].sensor, cfg.joints[3].sensor);
                            reply.vector("applied_action", applied_command, true);
                            reply.append(
                                ",\"applied_command_sequence\":%llu,\"applied_host_monotonic_ns\":%"
                                "llu,\"enable_slew\":%s,\"enable_tracks\":%s",
                                (unsigned long long)applied_sequence,
                                (unsigned long long)applied_monotonic_ns,
                                cfg.enable_slew ? "true" : "false",
                                cfg.enable_tracks ? "true" : "false");
                            reply.vector("fk_upper_xyz_m", fk.upper.position, healthy);
                            reply.vector(
                                "fk_upper_xz_m",
                                std::array<double, 2>{fk.upper.position[0], fk.upper.position[2]},
                                healthy);
                            reply.vector("fk_upper_quat_wxyz", fk.upper.orientation, healthy);
                            if (healthy)
                                reply.append(",\"slew_yaw_rad\":%.9f,\"bucket_pitch_rad\":%.9f",
                                             fk.slew_yaw, fk.bucket_pitch);
                            else
                                reply.append(",\"slew_yaw_rad\":null,\"bucket_pitch_rad\":null");
                            reply.matrix("fk_joint_xyz_m", fk.joint_positions_base, healthy);
                            reply.matrix("fk_joint_xyz_upper_m", fk.joint_positions_upper, healthy);
                            reply.matrix("fk_joint_quat_wxyz", fk.joint_rotations, healthy);
                            reply.matrix("fk_link_quat_wxyz", fk.link_orientations_base, healthy);
                            reply.matrix("fk_link_quat_upper_wxyz", fk.link_orientations_upper,
                                         healthy);
                            reply.matrix("imu_quat_wxyz", imu_raw, healthy);
                            reply.matrix("imu_corrected_quat_wxyz", imu_corrected, healthy);
                            reply.matrix("imu_gyro_dps", imu_gyro, healthy);
                            reply.matrix("imu_accel_g", imu_accel, healthy);
                            reply.append("}");
                            if (reply.ok)
                                send(clients[ci].fd, reply.bytes, reply.length,
                                     MSG_DONTWAIT | MSG_NOSIGNAL);
                            continue;
                        }
                        if (p.op == Operation::Drive) {
                            if (owner >= 0 && owner != ci)
                                continue;
                            owner = ci;
                            last_command = p.sequence;
                        }
                        core.receive(p, now, healthy);
                        if (p.op == Operation::Disarm)
                            owner = -1;
                    }
            const Commands command = core.tick(now, healthy);
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
                timings[timing_count++] = {next, now, monotonic_seconds(), fk_work_us};
            ++ticks;
            next += 1 / cfg.control_hz;
            if (next < now)
                next = now + 1 / cfg.control_hz;
            timespec wake{static_cast<time_t>(next),
                          static_cast<long>((next - std::floor(next)) * 1e9)};
            int result;
            do {
                result = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr);
            } while (result == EINTR && !stopped);
        }
        core.disarm();
        pwm.neutral();
        if (timing_file.is_open()) {
            timing_file << "tick,scheduled_ns,start_ns,finish_ns,lateness_ms,work_ms,fk_work_us\n";
            timing_file << std::setprecision(12);
            for (size_t i = 0; i < timing_count; ++i) {
                const auto &t = timings[i];
                timing_file << i << ',' << static_cast<uint64_t>(t.scheduled * 1e9) << ','
                            << static_cast<uint64_t>(t.start * 1e9) << ','
                            << static_cast<uint64_t>(t.finish * 1e9) << ','
                            << std::max(0., t.start - t.scheduled) * 1000 << ','
                            << (t.finish - t.start) * 1000 << ',' << t.fk_work_us << '\n';
            }
            if (!timing_file)
                throw std::runtime_error("timing CSV write failed");
        }
        for (auto &client : clients)
            if (client.fd >= 0)
                ::close(client.fd);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Control failed: %s\n", e.what());
        return 1;
    }
}
