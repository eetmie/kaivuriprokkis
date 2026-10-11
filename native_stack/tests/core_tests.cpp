#include "kaivuri/core.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace kaivuri;
static int checks = 0;
static void expect(bool v, const char *label) {
    ++checks;
    if (!v)
        throw std::runtime_error(label);
}
static bool close(double a, double b, double tol = 1e-8) {
    return std::abs(a - b) < tol;
}
template <class T> static void put(std::vector<uint8_t> &b, size_t at, T value) {
    std::memcpy(b.data() + at, &value, sizeof(T));
}
static std::vector<uint8_t> imu_packet(uint32_t timestamp, bool nan = false) {
    std::vector<uint8_t> b(170);
    b[0] = 0xaa;
    b[1] = 0x55;
    b[2] = 4;
    b[3] = 4;
    put(b, 4, timestamp);
    for (int s = 0; s < 4; ++s)
        put(b, 8 + s * 40, nan && s == 0 ? float(NAN) : 1.f);
    uint16_t sum = 0;
    for (size_t i = 2; i < b.size() - 2; ++i)
        sum += b[i];
    put(b, b.size() - 2, sum);
    return b;
}
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("profile directory required");
        Config c = load_config(argv[1], std::string(NATIVE_SOURCE) + "/config/safety.yaml");
        expect(c.frequency == 100 && c.pump_channel == 8, "profile calibration loaded");
        expect(close(c.valves[2].center, 1637.5), "None center uses midpoint");
        for (const auto &v : c.valves) {
            RampState r;
            auto plain = v;
            plain.dither = false;
            plain.ramp = false;
            expect(close(valve_pulse(plain, 0, 1, r), v.center), "zero is calibrated center");
            expect(close(valve_pulse(plain, 1, 1, r), v.direction > 0 ? v.max : v.min),
                   "full positive respects direction");
            expect(close(valve_pulse(plain, -1, 1, r), v.direction > 0 ? v.min : v.max),
                   "full negative respects direction");
            for (int i = -100; i <= 100; ++i) {
                double p = valve_pulse(v, i / 100., 1.234, r);
                expect(p >= v.min && p <= v.max, "dither remains bounded");
            }
        }
        expect(pulse_count(1500, 100) == 614, "matches existing truncation and 4095 multiplier");
        Pose p = forward_kinematics(c, {0, 0, 0, 0});
        expect(close(p.position[0], .7655) && close(p.position[2], -.0075),
               "zero FK analytical geometry");
        p = forward_kinematics(c, {pi / 2, 0, 0, 0});
        expect(close(p.position[0], 0) && close(p.position[1], .7655),
               "slew rotates child offsets");
        std::array<Quat, 4> mounted{};
        for (int i = 0; i < 4; ++i)
            mounted[i] = conjugate(c.offsets_inv[i]);
        auto angles = joint_angles(c, mounted);
        for (double a : angles)
            expect(close(a, 0), "mounting correction right multiplies");
        mounted[c.joints[1].sensor] =
            multiply(axis_angle({0, 1, 0}, .3), conjugate(c.offsets_inv[c.joints[1].sensor]));
        angles = joint_angles(c, mounted);
        expect(close(angles[1], .3) && close(angles[2], -.3), "pitch relative to parent");
        ImuParser parser(4);
        auto b = imu_packet(0xfffffff0U);
        for (auto byte : b)
            parser.feed(&byte, 1, 1);
        expect(parser.sample().valid, "fragmented IMU decoded");
        b = imu_packet(0x20);
        parser.feed(b.data(), b.size(), 1.01);
        expect(parser.sample().valid && parser.sample().device_us == 0x100000020ULL,
               "device timestamp rollover");
        parser.feed(b.data(), b.size(), 1.02);
        expect(!parser.sample().valid && parser.resets == 1,
               "duplicate timestamp invalidates data");
        b = imu_packet(1000, true);
        parser.feed(b.data(), b.size(), 1.03);
        expect(!parser.sample().valid, "NaN rejected before normalization");
        b = imu_packet(2000);
        b.back() ^= 1;
        parser.feed(b.data(), b.size(), 1.04);
        expect(parser.checksum_errors > 0, "checksum rejected");
        b = imu_packet(3000);
        parser.feed(b.data(), b.size(), 1.05);
        expect(parser.sample().valid, "resynchronization after corrupt packet");
        std::vector<uint8_t> wait_frame{0xaa, 0x55, 2, 7, 9, 0};
        parser.feed(wait_frame.data(), wait_frame.size(), 1.06);
        b = imu_packet(4000);
        parser.feed(b.data(), b.size(), 1.07);
        expect(parser.firmware_blocked, "CAL_WAIT cannot be hidden by subsequent data");
        std::vector<uint8_t> report(28 + 4 * 72);
        report[0] = 0xaa;
        report[1] = 0x55;
        report[2] = 5;
        report[3] = 4;
        report[8] = 1;
        uint16_t report_sum = 0;
        for (size_t i = 2; i < report.size() - 2; ++i)
            report_sum += report[i];
        put(report, report.size() - 2, report_sum);
        parser.feed(report.data(), report.size(), 1.08);
        expect(!parser.firmware_blocked, "accepted calibration unblocks streaming");
        report[8] = 0;
        report_sum = 0;
        for (size_t i = 2; i < report.size() - 2; ++i)
            report_sum += report[i];
        put(report, report.size() - 2, report_sum);
        parser.feed(report.data(), report.size(), 1.09);
        expect(parser.firmware_blocked, "failed calibration blocks arming");
        std::vector<uint8_t> wire(wire_header + 24);
        std::memcpy(wire.data(), "KVR1", 4);
        put(wire, 4, uint16_t(1));
        put(wire, 6, uint16_t(1));
        put(wire, 8, uint32_t(3));
        put(wire, 12, uint64_t(1));
        put(wire, 20, uint64_t(1000000000));
        put(wire, 28, uint32_t(1));
        put(wire, 32, 30.f);
        Packet decoded;
        expect(decode_packet(wire.data(), wire.size(), 1, c, decoded),
               "wire point packet accepted");
        put(wire, wire_header, float(NAN));
        expect(!decode_packet(wire.data(), wire.size(), 1, c, decoded),
               "nonfinite action rejected");
        put(wire, wire_header, 0.f);
        expect(!decode_packet(wire.data(), wire.size(), 2, c, decoded), "stale packet rejected");
        put(wire, 44, uint64_t(100000000));
        put(wire, 20, uint64_t(1700000000));
        expect(!decode_packet(wire.data(), wire.size(), 1.7, c, decoded),
               "old point observation rejected");
        put(wire, 44, uint64_t(0));
        put(wire, 20, uint64_t(1000000000));
        std::vector<uint8_t> chunk_wire = wire;
        chunk_wire.resize(wire_header + 48);
        put(chunk_wire, 28, uint32_t(2));
        put(chunk_wire, 44, uint64_t(1000000000));
        expect(decode_packet(chunk_wire.data(), chunk_wire.size(), 1, c, decoded),
               "chunk wire format accepted with fresh observation");
        put(chunk_wire, 44, uint64_t(0));
        expect(!decode_packet(chunk_wire.data(), chunk_wire.size(), 1, c, decoded),
               "chunk requires observation provenance");
        put(chunk_wire, 44, uint64_t(1000000000));
        put(chunk_wire, 32, .1f);
        expect(!decode_packet(chunk_wire.data(), chunk_wire.size(), 1, c, decoded),
               "overlong chunk horizon rejected");
        // Deterministic malformed byte streams exercise parser bounds under sanitizers.
        ImuParser fuzz(4);
        std::vector<uint8_t> noise(100000);
        uint32_t seed = 123;
        for (auto &byte : noise) {
            seed = 1664525 * seed + 1013904223;
            byte = seed >> 24;
        }
        fuzz.feed(noise.data(), noise.size(), 2);
        b = imu_packet(10000);
        fuzz.feed(b.data(), b.size(), 2.01);
        expect(fuzz.sample().valid, "parser recovers after a bounded random byte stream");
        ControlCore core(c);
        Packet arm;
        arm.op = Operation::Drive;
        arm.flags = deadman_flag | arm_flag | pump_flag;
        arm.count = 1;
        arm.fps = 30;
        arm.t0 = 1;
        core.receive(arm, 1, true);
        expect(core.state() == SafetyState::Arming && !core.pump_on(), "arming never runs pump");
        core.receive(arm, 1.26, true);
        expect(core.state() == SafetyState::Armed && core.pump_on(), "centered hold arms");
        Packet move = arm;
        move.flags = deadman_flag | pump_flag;
        move.steps[0] = {.7, .5, -.4, .3, .8, .9};
        move.t0 = 1.27;
        core.receive(move, 1.27, true);
        auto out = core.tick(1.27, true);
        expect(close(out[1], .5) && out[0] == 0 && out[4] == 0 && out[5] == 0,
               "slew and tracks gated");
        core.tick(1.78, true);
        expect(core.state() == SafetyState::Decaying && !core.pump_on(),
               "lease expiry stops pump immediately");
        out = core.tick(1.905, true);
        expect(close(out[1], .25), "timeout decays valves");
        core.tick(2.04, true);
        expect(core.state() == SafetyState::Fault, "timeout fault latches");
        core.receive(arm, 2.1, true);
        expect(core.state() == SafetyState::Fault, "arming cannot clear fault");
        core.disarm();
        core.receive(arm, 3, true);
        core.receive(arm, 3.26, true);
        core.tick(3.27, false);
        expect(core.state() == SafetyState::Fault && !core.pump_on(),
               "IMU loss immediately faults");
        core.disarm();
        core.receive(arm, 4, true);
        core.receive(arm, 4.26, true);
        Packet chunk = move;
        chunk.count = 2;
        chunk.fps = 10;
        chunk.t0 = 4.27;
        chunk.steps[0][1] = 0;
        chunk.steps[1][1] = 1;
        core.receive(chunk, 4.27, true);
        expect(close(core.tick(4.32, true)[1], .5), "native action interpolation");
        core.tick(4.38, true);
        expect(core.state() == SafetyState::Decaying, "exhausted chunk cannot hold final action");
        core.disarm();
        core.receive(arm, 5, true);
        core.receive(arm, 5.26, true);
        move.t0 = 5.27;
        move.observation = 4;
        core.receive(move, 5.27, true);
        core.tick(5.51, true);
        expect(core.state() == SafetyState::Fault &&
                   std::string(core.reason()) == "observation_stale",
               "accepted observation still expires during execution");
        std::cout << checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
