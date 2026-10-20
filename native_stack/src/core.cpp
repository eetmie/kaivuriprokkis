#include "kaivuri/core.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace kaivuri {
static double dot(const Vec3 &a, const Vec3 &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Quat normalized(Quat q) {
    double n = 0;
    for (double v : q) {
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite quaternion");
        n += v * v;
    }
    if (n < 1e-12)
        throw std::invalid_argument("zero quaternion");
    n = std::sqrt(n);
    for (double &v : q)
        v /= n;
    return q;
}
Quat multiply(const Quat &a, const Quat &b) {
    return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
Quat conjugate(Quat q) {
    for (int i = 1; i < 4; ++i)
        q[i] = -q[i];
    return q;
}
Vec3 rotate(const Quat &q, const Vec3 &v) {
    Vec3 u{q[1], q[2], q[3]}, c = cross(u, v), cc = cross(u, c), out{};
    for (int i = 0; i < 3; ++i)
        out[i] = v[i] + 2 * q[0] * c[i] + 2 * cc[i];
    return out;
}
Quat axis_angle(Vec3 axis, double angle) {
    const double n = std::sqrt(dot(axis, axis));
    if (n < 1e-12)
        throw std::invalid_argument("zero axis");
    const double s = std::sin(angle / 2) / n;
    return {std::cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s};
}
double wrap(double a) {
    return a - 2 * pi * std::floor((a + pi) / (2 * pi));
}
double gravity_pitch(const Quat &q) {
    return std::atan2(-2 * (q[1] * q[3] - q[0] * q[2]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
}
double gravity_roll(const Quat &q) {
    // Gravity's sensor-Y component is sin(roll) whatever the pitch (unlike atan2(y, z)).
    return std::asin(std::clamp(2 * (q[2] * q[3] + q[0] * q[1]), -1., 1.));
}
double twist(const Quat &q, Vec3 axis) {
    const double n = std::sqrt(dot(axis, axis));
    if (n < 1e-12)
        throw std::invalid_argument("zero twist axis");
    return wrap(2 * std::atan2((q[1] * axis[0] + q[2] * axis[1] + q[3] * axis[2]) / n, q[0]));
}
double monotonic_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double number(const YAML::Node &n, const char *key) {
    const double v = n[key].as<double>();
    if (!std::isfinite(v))
        throw std::runtime_error(std::string("nonfinite config: ") + key);
    return v;
}
static Vec3 vec3(const YAML::Node &n) {
    if (!n.IsSequence() || n.size() != 3)
        throw std::runtime_error("expected 3-vector");
    Vec3 v{};
    for (int i = 0; i < 3; ++i) {
        v[i] = n[i].as<double>();
        if (!std::isfinite(v[i]))
            throw std::runtime_error("nonfinite vector");
    }
    return v;
}
Config load_hardware_config(const std::string &profile_dir) {
    const auto dir = std::filesystem::path(profile_dir);
    const auto profile = YAML::LoadFile((dir / "profile.yaml").string());
    if (profile["board"].as<std::string>() != "jetson")
        throw std::runtime_error("native v1 supports Jetson profiles only");
    const auto servo =
        YAML::LoadFile((dir / profile["servo_config_file"].as<std::string>()).string());
    const auto control =
        YAML::LoadFile((dir / profile["control_config_file"].as<std::string>()).string());
    Config c;
    c.frequency = servo["pwm_frequency"].as<int>();
    if (c.frequency < 24 || c.frequency > 1526)
        throw std::runtime_error("invalid PWM frequency");
    constexpr std::array<const char *, 8> names{"trackL", "trackR", "slew",   "boom",
                                                "arm",    "bucket", "extra1", "extra2"};
    std::set<int> channels;
    for (size_t i = 0; i < names.size(); ++i) {
        auto n = servo["CHANNEL_CONFIGS"][names[i]];
        auto &v = c.valves[i];
        v.name = names[i];
        v.channel = n["output_channel"].as<int>();
        v.direction = n["direction"].as<int>();
        v.min = number(n, "pulse_min");
        v.max = number(n, "pulse_max");
        auto center = n["center"];
        const auto text = center.IsScalar() ? center.Scalar() : "";
        v.center = (!center || center.IsNull() || text == "None" || text == "none")
                       ? (v.min + v.max) / 2
                       : number(n, "center");
        v.deadzone = number(n, "deadzone") / 100;
        v.gamma = number(n, "gamma");
        v.deadband_pos = number(n, "deadband_us_pos");
        v.deadband_neg = number(n, "deadband_us_neg");
        v.dither = n["dither_enable"].as<bool>();
        v.dither_amp = number(n, "dither_amp_us");
        v.dither_hz = number(n, "dither_hz");
        v.taper = n["dither_taper"].as<bool>();
        v.taper_us = number(n, "dither_taper_us");
        v.ramp = n["ramp_enable"].as<bool>();
        v.ramp_limit = number(n, "ramp_limit");
        v.skip_deadband = n["ramp_skip_deadband"].as<bool>();
        if (v.channel < 0 || v.channel > 15 || !channels.insert(v.channel).second ||
            v.direction * v.direction != 1 || v.min <= 0 || v.min >= v.center ||
            v.center >= v.max || v.max >= 1e6 / c.frequency || v.deadzone < 0 || v.deadzone >= 1 ||
            v.gamma <= 0 || v.deadband_pos < 0 || v.deadband_neg < 0 ||
            v.center + v.deadband_pos > v.max || v.center - v.deadband_neg < v.min ||
            v.dither_amp < 0 || v.dither_hz < 0 || v.taper_us < 0 || v.ramp_limit < 0)
            throw std::runtime_error("invalid valve calibration: " + v.name);
    }
    auto pump = servo["CHANNEL_CONFIGS"]["pump"];
    c.pump_channel = pump["output_channel"].as<int>();
    c.pump_min = number(pump, "pulse_min");
    c.pump_max = number(pump, "pulse_max");
    c.pump_run = number(pump, "static_pulse_us");
    if (c.pump_channel < 0 || c.pump_channel > 15 || !channels.insert(c.pump_channel).second ||
        c.pump_min <= 0 || c.pump_min > c.pump_run || c.pump_run > c.pump_max ||
        c.pump_max >= 1e6 / c.frequency)
        throw std::runtime_error("invalid pump calibration");
    auto mapping = control["imu"]["imu_mapping"];
    std::set<int> sensors;
    for (const auto &entry : mapping) {
        const auto role = entry.first.as<std::string>();
        int index = entry.second.as<int>();
        if (index < 0 || index >= 4 || !sensors.insert(index).second)
            throw std::runtime_error("invalid IMU mapping");
        c.offsets_inv[index] = {1, 0, 0, 0};
        auto n = control["imu"]["mounting_offsets_quat"][role];
        if (n) {
            if (n.size() != 4)
                throw std::runtime_error("invalid mounting quaternion");
            Quat q{};
            for (int j = 0; j < 4; ++j)
                q[j] = n[j].as<double>();
            c.offsets_inv[index] = conjugate(normalized(q));
        }
    }
    if (sensors.size() != 4)
        throw std::runtime_error("native v1 requires four mapped IMUs");
    if (const auto level = control["robot"]["level_frame"]) {
        c.level.cabin_tilt = level["cabin_tilt"].as<bool>(c.level.cabin_tilt);
        c.level.link_roll = level["link_roll"].as<bool>(c.level.link_roll);
    }
    const auto joints = control["robot"]["joints"], chain = control["imu"]["chain"];
    if (joints.size() != 4 || chain.size() != 4)
        throw std::runtime_error("native v1 requires four joints and extraction rules");
    std::set<int> rules;
    for (int i = 0; i < 4; ++i) {
        auto n = joints[i];
        c.joints[i].name = n["name"].as<std::string>();
        if (c.joints[i].name != command_names[i])
            throw std::runtime_error("unexpected joint order");
        c.joints[i].axis = vec3(n["axis"]);
        c.joints[i].offset = vec3(n["parent_to_joint_xyz"]);
        axis_angle(c.joints[i].axis, 0);
    }
    for (const auto &n : chain) {
        int i = n["output_index"].as<int>();
        if (i < 0 || i >= 4 || !rules.insert(i).second)
            throw std::runtime_error("invalid extraction index");
        auto &j = c.joints[i];
        j.sensor = mapping[n["role"].as<std::string>()].as<int>();
        auto parent = n["parent_role"];
        if (parent && !parent.IsNull())
            j.parent_sensor = mapping[parent.as<std::string>()].as<int>();
        auto method = n["extraction"].as<std::string>();
        j.pitch_delta = method == "gravity_pitch_delta";
        if (!j.pitch_delta && method != "relative_axis_twist")
            throw std::runtime_error("unsupported extraction method");
        j.extraction_axis = j.axis;
        if (n["axis"]) {
            auto axis = n["axis"].as<std::string>();
            if (axis != "x" && axis != "y" && axis != "z")
                throw std::runtime_error("invalid extraction axis");
            j.extraction_axis = {axis == "x" ? 1. : 0., axis == "y" ? 1. : 0.,
                                 axis == "z" ? 1. : 0.};
        }
    }
    c.tip = vec3(control["robot"]["tool"]["parent_to_tip_xyz"]);
    return c;
}
Config load_config(const std::string &profile_dir, const std::string &safety_file) {
    Config c = load_hardware_config(profile_dir);
    const auto safety = YAML::LoadFile(safety_file);
    c.control_hz = number(safety, "control_hz");
    c.imu_timeout = number(safety, "imu_timeout_s");
    c.command_lease = number(safety, "command_lease_s");
    c.arm_duration = number(safety, "arming_s");
    c.decay = number(safety, "decay_s");
    c.deadline = number(safety, "deadline_fault_s");
    c.max_packet_age = number(safety, "max_packet_age_s");
    c.max_observation_age = number(safety, "max_observation_age_s");
    c.max_horizon = number(safety, "max_chunk_horizon_s");
    if (c.control_hz < 50 || c.control_hz > 500 || c.imu_timeout <= 0 || c.imu_timeout > 1 ||
        c.command_lease <= 0 || c.command_lease > 5 || c.arm_duration < 0.1 || c.decay < 0 ||
        c.decay > 1 || c.deadline <= 1 / c.control_hz || c.max_packet_age <= 0 ||
        c.max_observation_age <= 0 || c.max_horizon <= 0 || c.max_horizon > 10)
        throw std::runtime_error("invalid safety timing");
    return c;
}
std::array<Quat, 4> corrected_imu_quaternions(const Config &c, const std::array<Quat, 4> &raw) {
    std::array<Quat, 4> corrected{};
    for (int i = 0; i < 4; ++i)
        corrected[i] = normalized(multiply(normalized(raw[i]), c.offsets_inv[i]));
    return corrected;
}
std::array<double, 4> joint_angles(const Config &c, const std::array<Quat, 4> &raw) {
    const auto corrected = corrected_imu_quaternions(c, raw);
    std::array<double, 4> out{};
    for (int i = 0; i < 4; ++i) {
        const auto &j = c.joints[i];
        const auto &child = corrected[j.sensor];
        if (j.pitch_delta)
            out[i] = wrap(gravity_pitch(child) -
                          (j.parent_sensor < 0 ? 0 : gravity_pitch(corrected[j.parent_sensor])));
        else
            out[i] = twist(j.parent_sensor < 0
                               ? child
                               : normalized(multiply(conjugate(corrected[j.parent_sensor]), child)),
                           j.extraction_axis);
    }
    return out;
}
Attitude attitude(const Config &c, const std::array<Quat, 4> &raw) {
    const auto q = corrected_imu_quaternions(c, raw);
    const auto &slew = c.joints[0];
    const Quat cabin = q[slew.sensor];
    Attitude a;
    // Swing-twist about the slew axis: the twist is the slew angle, the swing is the tilt.
    if (c.level.cabin_tilt)
        a.tilt = normalized(multiply(
            cabin, conjugate(axis_angle(slew.extraction_axis, twist(cabin, slew.extraction_axis)))));
    a.cabin_roll = gravity_roll(cabin);
    a.cabin_pitch = gravity_pitch(cabin);
    // Link roll: the turn about the child's X between the gravity it measures and the gravity the
    // parent IMU plus the hinge angle predict. Only the Y-Z part of gravity sees it, so a link
    // within about 11 degrees of vertical reports 0 (unobservable).
    const auto angles = joint_angles(c, raw);
    for (int i = 1; i < 4; ++i) {
        const auto &j = c.joints[i];
        if (!j.pitch_delta || j.parent_sensor < 0)
            continue;
        const auto expected = rotate(conjugate(multiply(q[j.parent_sensor], axis_angle(j.axis, angles[i]))), {0, 0, 1});
        const auto measured = rotate(conjugate(q[j.sensor]), {0, 0, 1});
        const double norm = expected[1] * expected[1] + expected[2] * expected[2];
        if (norm > .04)
            a.link_roll[i] = std::atan2(measured[1] * expected[2] - measured[2] * expected[1],
                                        measured[1] * expected[1] + measured[2] * expected[2]);
    }
    return a;
}
KinematicState forward_kinematic_state(const Config &c, const std::array<double, 4> &angles,
                                       const Quat &tilt, const std::array<double, 4> &link_roll) {
    KinematicState state;
    // Evaluate the arm in the carriage frame directly: slew drift never enters this chain.
    const Quat slew = axis_angle(c.joints[0].axis, angles[0]);
    state.joint_rotations[0] = slew;
    state.link_orientations_upper[0] = {1, 0, 0, 0};
    for (int i = 1; i < 4; ++i) {
        const auto d = rotate(state.upper.orientation, c.joints[i].offset);
        for (int j = 0; j < 3; ++j)
            state.upper.position[j] += d[j];
        state.joint_positions_upper[i] = state.upper.position;
        state.joint_rotations[i] = axis_angle(c.joints[i].axis, angles[i]);
        if (link_roll[i] != 0) // measured roll about the link's own X, after the hinge
            state.joint_rotations[i] =
                multiply(state.joint_rotations[i], axis_angle({1, 0, 0}, link_roll[i]));
        state.upper.orientation =
            normalized(multiply(state.upper.orientation, state.joint_rotations[i]));
        state.link_orientations_upper[i] = state.upper.orientation;
    }
    const auto tip = rotate(state.upper.orientation, c.tip);
    for (int j = 0; j < 3; ++j)
        state.upper.position[j] += tip[j];
    // Carriage to level base: slew, then the measured tilt about the ground reference origin.
    const Quat carriage = normalized(multiply(tilt, slew));
    state.base.orientation = normalized(multiply(carriage, state.upper.orientation));
    const auto to_base = [&](const Vec3 &upper) {
        Vec3 p = rotate(carriage, upper);
        const auto pivot = rotate(tilt, c.joints[0].offset);
        for (int j = 0; j < 3; ++j)
            p[j] += pivot[j];
        return p;
    };
    state.base.position = to_base(state.upper.position);
    for (int i = 0; i < 4; ++i) {
        state.joint_positions_base[i] = to_base(state.joint_positions_upper[i]);
        state.link_orientations_base[i] =
            normalized(multiply(carriage, state.link_orientations_upper[i]));
    }
    state.slew_yaw = angles[0];
    // Signed planar tool-axis angle, including angles beyond Euler pitch's +/-90 deg range.
    state.bucket_pitch = gravity_pitch(state.upper.orientation);
    return state;
}
Pose forward_kinematics(const Config &c, const std::array<double, 4> &angles) {
    return forward_kinematic_state(c, angles).base;
}
double valve_pulse(const Valve &v, double command, double now, RampState &state) {
    if (!std::isfinite(command) || !std::isfinite(now))
        throw std::invalid_argument("nonfinite valve input");
    double value = std::clamp(command, -1., 1.);
    const double a = std::abs(value);
    value = a < v.deadzone ? 0 : std::copysign((a - v.deadzone) / (1 - v.deadzone), value);
    value = std::copysign(std::pow(std::abs(value), v.gamma), value);
    const double sign = value * v.direction;
    double base = sign == 0
                      ? v.center
                      : (sign > 0 ? v.center + v.deadband_pos +
                                        std::abs(value) * (v.max - v.center - v.deadband_pos)
                                  : v.center - v.deadband_neg -
                                        std::abs(value) * (v.center - v.deadband_neg - v.min));
    if (v.ramp && v.ramp_limit > 0 && state.initialized) {
        double last = state.pulse;
        if (v.skip_deadband) {
            if (last == v.center && base != v.center)
                last = base > v.center ? v.center + v.deadband_pos : v.center - v.deadband_neg;
            else if (last > v.center && base < v.center)
                last = v.center - v.deadband_neg;
            else if (last < v.center && base > v.center)
                last = v.center + v.deadband_pos;
        }
        const double allowed = v.ramp_limit * std::clamp(now - state.time, 0., 0.05);
        base = last + std::clamp(base - last, -allowed, allowed);
        if (v.skip_deadband && sign == 0 && base > v.center - v.deadband_neg &&
            base < v.center + v.deadband_pos)
            base = v.center;
    }
    state = {base, now, true};
    if (v.dither && v.dither_amp != 0 && v.dither_hz != 0 && std::abs(base - v.center) > 1e-6) {
        auto amplitude = [&](double pulse) {
            if (!v.taper)
                return v.dither_amp;
            const double distance = pulse > v.center ? pulse - v.center - v.deadband_pos
                                                     : v.center - v.deadband_neg - pulse;
            return v.dither_amp * std::clamp(distance / std::max(v.taper_us, 1e-6), 0., 1.);
        };
        double amp = amplitude(base), lo = v.min + amp, hi = v.max - amp;
        base = lo > hi ? (v.min + v.max) / 2 : std::clamp(base, lo, hi);
        amp = amplitude(base);
        base += amp * std::sin(2 * pi * v.dither_hz * now + v.channel * 1.0471975512);
    }
    return std::clamp(base, v.min, v.max);
}
uint16_t pulse_count(double us, int frequency) {
    if (!std::isfinite(us) || frequency <= 0)
        throw std::invalid_argument("invalid PWM");
    return static_cast<uint16_t>(std::clamp(us * 4095 * frequency / 1e6, 0., 4095.));
}
template <class T> static T read_le(const uint8_t *p) {
    T v{};
    std::memcpy(&v, p, sizeof(v));
    return v;
} // supported hosts: little-endian ARM64/x86_64
void ImuParser::consume(size_t n) {
    size_ -= n;
    std::memmove(buffer_.data(), buffer_.data() + n, size_);
}
void ImuParser::feed(const uint8_t *bytes, size_t count, double now) {
    for (size_t i = 0; i < count; ++i) {
        if (size_ == buffer_.size()) {
            consume(1);
            ++invalid_frames;
        }
        buffer_[size_++] = bytes[i];
        parse(now);
    }
}
void ImuParser::parse(double now) {
    while (size_ >= 4) {
        if (buffer_[0] != 0xaa || buffer_[1] != 0x55) {
            consume(1);
            continue;
        }
        const int type = buffer_[2], count = buffer_[3];
        size_t len = 0;
        if (type == 2)
            len = 6;
        else if (count < 1 || count > 4) {
            consume(1);
            ++invalid_frames;
            continue;
        } else if (type == 1 || type == 4)
            len = 10 + count * (type == 4 ? 40 : 28);
        else if (type == 3 || type == 6)
            len = 8 + (type == 6 ? 8 : 0) + count * 2;
        else if (type == 5)
            len = 28 + count * 72;
        else {
            consume(1);
            ++invalid_frames;
            continue;
        }
        if (size_ < len)
            return;
        uint16_t sum = 0;
        for (size_t i = 2; i < len - 2; ++i)
            sum = static_cast<uint16_t>(sum + buffer_[i]);
        if (sum != read_le<uint16_t>(&buffer_[len - 2])) {
            consume(1);
            ++checksum_errors;
            continue;
        }
        if (type == 2) {
            control_message = count;
            if (count >= 5 && count <= 8)
                firmware_blocked = true;
            sample_.valid = false;
            ++resets;
        } else if (type == 3 || type == 6) {
            if (type == 6) {
                const double gyro = read_le<float>(&buffer_[6]);
                const double accel = read_le<float>(&buffer_[10]);
                if (std::isfinite(gyro) && gyro > 0)
                    gyro_range_dps = gyro;
                if (std::isfinite(accel) && accel > 0)
                    accel_range_g = accel;
            }
            sample_.valid = false;
            have_ts_ = false;
            ++resets;
        } else if (type == 5) {
            firmware_blocked = count != expected_ || (buffer_[8] & 1) == 0;
            sample_.valid = false;
            ++resets;
        } else if (type == 1 || type == 4) {
            bool valid = count == expected_;
            ImuSample next{};
            const int floats = type == 4 ? 10 : 7;
            for (int s = 0; s < count; ++s) {
                double norm = 0;
                for (int j = 0; j < floats; ++j) {
                    float v = read_le<float>(&buffer_[8 + (s * floats + j) * 4]);
                    valid = valid && std::isfinite(v);
                    if (j < 4) {
                        next.quats[s][j] = v;
                        norm += v * v;
                    } else
                        next.motion[s][j - 4] = v;
                }
                valid = valid && norm >= 0.9025 && norm <= 1.1025;
            }
            uint32_t ts = read_le<uint32_t>(&buffer_[4]);
            if (have_ts_) {
                uint32_t delta = ts - previous_ts_;
                if (delta == 0 || delta > 1000000) {
                    valid = false;
                    have_ts_ = false;
                    ++resets;
                } else
                    extended_ts_ += delta;
            } else {
                extended_ts_ = ts;
                have_ts_ = true;
            }
            previous_ts_ = ts;
            next.device_us = extended_ts_;
            next.raw_device_us = ts;
            next.packet_type = static_cast<uint8_t>(type);
            next.epoch = resets;
            next.sequence = sample_.sequence + 1;
            next.received = now;
            next.valid = valid;
            sample_ = next;
            if (!valid)
                ++invalid_frames;
            else if (sink_)
                sink_(sink_context_, sample_);
        }
        consume(len);
    }
}
bool decode_packet(const uint8_t *b, size_t n, double now, const Config &c, Packet &p) {
    if (n < wire_header || std::memcmp(b, "KVR1", 4) != 0 || read_le<uint16_t>(b + 4) != 1)
        return false;
    const auto op = read_le<uint16_t>(b + 6);
    if (op < 1 || op > 4)
        return false;
    p.op = static_cast<Operation>(op);
    p.flags = read_le<uint32_t>(b + 8);
    p.sequence = read_le<uint64_t>(b + 12);
    p.sent = read_le<uint64_t>(b + 20) * 1e-9;
    p.count = read_le<uint32_t>(b + 28);
    p.fps = read_le<float>(b + 32);
    p.t0 = read_le<uint64_t>(b + 36) * 1e-9;
    p.observation = read_le<uint64_t>(b + 44) * 1e-9;
    if (p.flags & ~7u || now - p.sent > c.max_packet_age || p.sent - now > 0.01)
        return false;
    if (p.op != Operation::Drive)
        return p.count == 0 && n == wire_header;
    if (p.count < 1 || p.count > max_steps || n != wire_header + p.count * 24 ||
        !std::isfinite(p.fps) || p.fps <= 0 || p.fps > 500)
        return false;
    if ((p.count > 1 || p.observation > 0) &&
        (p.observation <= 0 || now - p.observation > c.max_observation_age ||
         p.observation - now > 0.01))
        return false;
    if (p.t0 <= 0)
        p.t0 = now;
    if (p.t0 - now > 0.01 || now - p.t0 > c.max_horizon || (p.count - 1) / p.fps > c.max_horizon)
        return false;
    for (size_t i = 0; i < p.count; ++i)
        for (int j = 0; j < 6; ++j) {
            float v = read_le<float>(b + wire_header + i * 24 + j * 4);
            if (!std::isfinite(v) || std::abs(v) > 1)
                return false;
            p.steps[i][j] = v;
        }
    return true;
}
const char *state_name(SafetyState s) {
    switch (s) {
    case SafetyState::Disarmed:
        return "disarmed";
    case SafetyState::Arming:
        return "arming";
    case SafetyState::Armed:
        return "armed";
    case SafetyState::Decaying:
        return "decaying";
    case SafetyState::Fault:
        return "fault";
    }
    return "unknown";
}
void ControlCore::fault(const char *reason) {
    state_ = SafetyState::Fault;
    reason_ = reason;
    pump_ = false;
    previous_ = {};
    plan_.count = 0;
}
void ControlCore::disarm() {
    state_ = SafetyState::Disarmed;
    reason_ = "explicit_disarm";
    pump_ = false;
    previous_ = {};
    plan_.count = 0;
}
void ControlCore::receive(const Packet &p, double now, bool healthy) {
    if (p.op == Operation::Disarm) {
        disarm();
        return;
    }
    if (p.op == Operation::Estop) {
        fault("emergency_stop");
        return;
    }
    if (p.op != Operation::Drive)
        return;
    if (state_ == SafetyState::Fault || state_ == SafetyState::Decaying)
        return;
    if (!(p.flags & deadman_flag)) {
        disarm();
        return;
    }
    if (!healthy) {
        fault("imu_unhealthy");
        return;
    }
    if (state_ != SafetyState::Armed) {
        bool centered = p.count == 1;
        for (int j = 0; j < 6; ++j)
            centered = centered && std::abs(p.steps[0][j]) <= 0.01;
        if (!(p.flags & arm_flag) || !centered) {
            disarm();
            return;
        }
        if (state_ != SafetyState::Arming) {
            state_ = SafetyState::Arming;
            reason_ = "centered_arming";
            arm_started_ = now;
        }
        if (now - arm_started_ >= cfg_.arm_duration) {
            state_ = SafetyState::Armed;
            reason_ = "ready";
        }
    }
    plan_ = p;
    updated_ = now;
    pump_ = (p.flags & pump_flag) != 0;
}
Commands ControlCore::evaluate(double now) const {
    Commands out{};
    if (!plan_.count)
        return out;
    const double pos =
        std::clamp((now - plan_.t0) * plan_.fps, 0., static_cast<double>(plan_.count - 1));
    const size_t i = static_cast<size_t>(pos),
                 next = std::min(i + 1, static_cast<size_t>(plan_.count - 1));
    for (int j = 0; j < 6; ++j)
        out[j] = plan_.steps[i][j] + (plan_.steps[next][j] - plan_.steps[i][j]) * (pos - i);
    if (!cfg_.enable_slew)
        out[0] = 0;
    if (!cfg_.enable_tracks)
        out[4] = out[5] = 0;
    return out;
}
Commands ControlCore::tick(double now, bool healthy) {
    if (state_ == SafetyState::Armed && plan_.observation > 0 &&
        now - plan_.observation > cfg_.max_observation_age)
        fault("observation_stale");
    if ((state_ == SafetyState::Armed || state_ == SafetyState::Arming ||
         state_ == SafetyState::Decaying) &&
        !healthy)
        fault("imu_unhealthy");
    if (state_ == SafetyState::Arming && now - updated_ > cfg_.command_lease)
        disarm();
    if (state_ == SafetyState::Armed) {
        const bool exhausted = plan_.count > 1 && now > plan_.t0 + (plan_.count - 1) / plan_.fps;
        if (now - updated_ > cfg_.command_lease || exhausted) {
            decay_from_ = previous_;
            decay_started_ = now;
            state_ = SafetyState::Decaying;
            reason_ = exhausted ? "chunk_exhausted" : "producer_timeout";
            pump_ = false;
        } else {
            previous_ = evaluate(now);
            return previous_;
        }
    }
    if (state_ == SafetyState::Decaying) {
        const double scale =
            cfg_.decay > 0 ? std::clamp(1 - (now - decay_started_) / cfg_.decay, 0., 1.) : 0;
        if (scale == 0) {
            fault(reason_);
            return {};
        }
        Commands out = decay_from_;
        for (double &v : out)
            v *= scale;
        return out;
    }
    return {};
}
} // namespace kaivuri
