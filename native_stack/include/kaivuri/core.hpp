#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace kaivuri {
constexpr double pi = 3.14159265358979323846;
using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;     // w,x,y,z
using Commands = std::array<double, 6>; // slew,boom,arm,bucket,trackL,trackR
constexpr std::array<const char *, 6> command_names{"slew",   "boom",   "arm",
                                                    "bucket", "trackL", "trackR"};
Quat normalized(Quat q);
Quat multiply(const Quat &a, const Quat &b);
Quat conjugate(Quat q);
Vec3 rotate(const Quat &q, const Vec3 &v);
Quat axis_angle(Vec3 axis, double angle);
double wrap(double angle);
double gravity_pitch(const Quat &q);
double twist(const Quat &q, Vec3 axis);
double monotonic_seconds();

struct Valve {
    std::string name;
    int channel = -1, direction = 1;
    double min = 1000, max = 2000, center = 1500, deadzone = 0;
    double deadband_pos = 0, deadband_neg = 0, gamma = 1;
    bool dither = false, taper = false, ramp = false, skip_deadband = false;
    double dither_amp = 0, dither_hz = 0, taper_us = 0, ramp_limit = 0;
};
struct Joint {
    std::string name;
    Vec3 axis{}, offset{};
    int sensor = -1, parent_sensor = -1;
    bool pitch_delta = false;
    Vec3 extraction_axis{};
};
struct Config {
    std::array<Valve, 8> valves{};
    std::array<Joint, 4> joints{};
    std::array<Quat, 4> offsets_inv{};
    Vec3 tip{};
    int sensor_count = 4, frequency = 100, i2c_bus = 7, address = 0x40;
    int pump_channel = 8;
    double pump_min = 1000, pump_max = 1650, pump_run = 1550;
    double control_hz = 100, imu_timeout = 0.1, command_lease = 0.5;
    double arm_duration = 0.25, decay = 0.25, deadline = 0.04;
    double max_packet_age = 0.1, max_observation_age = 1.5, max_horizon = 5;
    bool enable_slew = false, enable_tracks = false;
};
Config load_hardware_config(const std::string &profile_dir);
Config load_config(const std::string &profile_dir, const std::string &safety_file);
struct Pose {
    Vec3 position{};
    Quat orientation{1, 0, 0, 0};
};
struct KinematicState {
    Pose base, upper; // Upper frame: slew pivot origin, axes rotate with the carriage.
    std::array<Vec3, 4> joint_positions_base{}, joint_positions_upper{};
    std::array<Quat, 4> joint_rotations{}, link_orientations_base{}, link_orientations_upper{};
    double slew_yaw = 0, bucket_pitch = 0; // radians; pitch is positive about upper +Y.
};
std::array<Quat, 4> corrected_imu_quaternions(const Config &cfg, const std::array<Quat, 4> &raw);
std::array<double, 4> joint_angles(const Config &cfg, const std::array<Quat, 4> &raw);
KinematicState forward_kinematic_state(const Config &cfg, const std::array<double, 4> &angles);
Pose forward_kinematics(const Config &cfg, const std::array<double, 4> &angles);
struct RampState {
    double pulse = 0, time = 0;
    bool initialized = false;
};
double valve_pulse(const Valve &v, double command, double now, RampState &state);
uint16_t pulse_count(double pulse_us, int frequency);

struct ImuSample {
    std::array<Quat, 4> quats{};
    std::array<std::array<float, 6>, 4> motion{}; // gyro dps, accel g
    uint64_t device_us = 0, sequence = 0, epoch = 0;
    uint32_t raw_device_us = 0;
    uint8_t packet_type = 0;
    double received = 0;
    bool valid = false;
};
class ImuParser {
  public:
    using SampleSink = void (*)(void *, const ImuSample &);
    explicit ImuParser(int expected) : expected_(expected) {}
    void feed(const uint8_t *bytes, size_t count, double now);
    const ImuSample &sample() const { return sample_; }
    void set_sample_sink(SampleSink sink, void *context) {
        sink_ = sink;
        sink_context_ = context;
    }
    double gyro_range_dps = 0, accel_range_g = 0;
    uint64_t checksum_errors = 0, invalid_frames = 0, resets = 0;
    int control_message = 0;
    bool firmware_blocked = false;

  private:
    std::array<uint8_t, 2048> buffer_{};
    size_t size_ = 0;
    int expected_;
    ImuSample sample_{};
    SampleSink sink_ = nullptr;
    void *sink_context_ = nullptr;
    uint32_t previous_ts_ = 0;
    uint64_t extended_ts_ = 0;
    bool have_ts_ = false;
    void consume(size_t n);
    void parse(double now);
};

constexpr size_t max_steps = 128, wire_header = 52, max_packet = wire_header + max_steps * 6 * 4;
enum class Operation : uint16_t { Drive = 1, Disarm = 2, Estop = 3, Status = 4 };
constexpr uint32_t deadman_flag = 1, arm_flag = 2, pump_flag = 4;
struct Packet {
    Operation op = Operation::Status;
    uint32_t flags = 0, count = 0;
    uint64_t sequence = 0;
    double sent = 0, fps = 0, t0 = 0, observation = 0;
    std::array<Commands, max_steps> steps{};
};
bool decode_packet(const uint8_t *bytes, size_t size, double now, const Config &cfg, Packet &out);
enum class SafetyState { Disarmed, Arming, Armed, Decaying, Fault };
const char *state_name(SafetyState s);
class ControlCore {
  public:
    explicit ControlCore(Config cfg) : cfg_(std::move(cfg)) {}
    void receive(const Packet &p, double now, bool imu_healthy);
    Commands tick(double now, bool imu_healthy);
    void fault(const char *reason);
    void disarm();
    SafetyState state() const { return state_; }
    const char *reason() const { return reason_; }
    bool pump_on() const { return state_ == SafetyState::Armed && pump_; }
    const Config &config() const { return cfg_; }

  private:
    Config cfg_;
    SafetyState state_ = SafetyState::Disarmed;
    const char *reason_ = "boot";
    bool pump_ = false;
    double arm_started_ = 0, updated_ = 0, decay_started_ = 0;
    Packet plan_{};
    Commands previous_{}, decay_from_{};
    Commands evaluate(double now) const;
};
} // namespace kaivuri
