#pragma once
// CUDA-free parts of the fine-tuned OpenPI pi0.5 runtime (bundle policy pi05_openpi_state):
// per-step state prompt, OpenPI quantile normalization, controller state, and the
// policy -> controller drive-plan producer. See notes/PI05_FINETUNE.md.
#include "kaivuri/core.hpp"
#include "kaivuri/util.hpp"
#include <array>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace kaivuri::pi05 {

struct Quantiles {
    std::vector<double> q01, q99;
};
// OpenPI Normalize/Unnormalize with use_quantiles, in numpy's dtype order: float32 data,
// float64 stats, +1e-6 in the denominator. Bit-exact against the smoke9 fixtures.
std::vector<double> normalize(const std::vector<float> &x, const Quantiles &q);
double unnormalize(float x, double q01, double q99);
// np.digitize(x, np.linspace(-1, 1, 257)[:-1]) - 1: -1 below -1, 255 from 1 - 1/128.
int state_bin(double normalized);

// Pillow's Image.resize(BILINEAR) on 8-bit data, exactly (22-bit fixed-point taps,
// horizontal pass first): OpenPI ResizeImages via openpi_client.image_tools. One channel,
// read every pixel_step bytes.
std::vector<uint8_t> pil_bilinear(const uint8_t *input, int width, int height, size_t row_stride,
                                  int pixel_step, int out_width, int out_height);

struct StatePrompt {
    std::map<std::string, std::vector<int64_t>> heads; // ids("<bos>Task: T, State:")
    int64_t sep = -1, neg = -1;                        // ids("▁"), ids("▁-")
    std::array<int64_t, 10> digits{};
    std::vector<int64_t> tail; // ids(";\nAction: ")
    // head + per bin [neg if bin < 0 else sep] + digits of |bin| + tail.
    std::vector<int64_t> compose(const std::string &task, const std::vector<int> &bins) const;
    size_t longest(size_t state_dim) const; // worst case: 4 tokens per state value
};

// Index of a dataset joint name in the controller's angles_rad [slew, boom, arm, bucket].
int joint_index(const std::string &dataset_name);
struct Contract {
    std::vector<std::string> state_names; // dataset order, e.g. lift, tilt, scoop
    std::vector<int> state_joints;        // index into angles_rad
    Quantiles state, actions;
    StatePrompt prompt;
    int state_dim = 0, robot_dim = 0;
    double fps = 0; // kaivuri.fps; 0 when the bundle has none
    std::vector<int64_t> tokens(const std::string &task, const std::vector<float> &state) const;
    std::vector<int> bins(const std::vector<float> &state) const;
};
// Reads bundle.json fields and the bundle's norm_stats.json (OpenPI layout).
Contract load_contract(const YAML::Node &bundle, const YAML::Node &norm_stats);

struct RobotState {
    std::vector<float> degrees; // float32 like the exporter's np.rad2deg(float32)
    uint64_t imu_ns = 0;        // host arrival of the IMU sample behind these angles
    std::string safety_state, reason;
    bool live_pwm = false, enable_slew = false, enable_tracks = false;
};
// Parses a kaivuri-control status reply. Throws unless the IMUs are healthy, FK is valid
// and the IMU sample is at most max_age_ns old at now.
RobotState robot_state(const std::string &status_json, const std::vector<int> &joints, uint64_t now,
                       uint64_t max_age_ns);

struct Plan {
    std::vector<Commands> rows; // controller order: slew, boom, arm, bucket, trackL, trackR
    double fps = 0;
    uint64_t t0_ns = 0, observation_ns = 0; // observation 0 = centered arming hold
    uint64_t end_ns() const;
};
struct ActionGate {
    bool slew = false, tracks = false; // producer-side flags; also required by the daemon
    // Out-of-range values are clamped to [-1, 1] (the controller clamps too); a finite
    // tolerance turns |v| > 1 + tolerance into a stop instead.
    double tolerance = std::numeric_limits<double>::infinity();
};
// Robot-unit chunk [horizon][robot_dim] in [slew, lift, tilt, scoop(, trackL, trackR)].
// Throws on non-finite values (or beyond a finite 1 + tolerance); disabled joints become 0.
Plan make_plan(const std::vector<double> &robot, int horizon, int robot_dim, double fps,
               uint64_t t0_ns, uint64_t observation_ns, const ActionGate &gate, int &clamped);
// One PROTOCOL.md packet. plan == nullptr encodes a non-drive operation.
std::vector<uint8_t> encode(Operation op, uint32_t flags, uint64_t sequence, uint64_t sent_ns,
                            const Plan *plan);

// Renewal/stop rules, independent of sockets and clocks.
struct SupervisorConfig {
    uint64_t renew_ns = 100000000ULL; // well inside the 500 ms lease
    // A plan must stay valid until the next chunk lands, i.e. about two inference periods
    // (~1.3 s at ~640 ms); kept just below the daemon's 1.5 s observation fault.
    uint64_t max_observation_age_ns = 1400000000ULL;
    uint64_t end_margin_ns = 0;                     // stop this long before the last row
    uint64_t first_plan_timeout_ns = 5000000000ULL; // centered hold before the first chunk
};
class PlanSupervisor {
  public:
    enum class Decision { Idle, Send, Stop };
    explicit PlanSupervisor(SupervisorConfig cfg) : cfg_(cfg) {}
    void hold(uint64_t now); // centered single-row plan until the first chunk
    // Empty string = accepted; otherwise the refusal reason. Never accepts an older
    // observation than the current plan's.
    std::string offer(Plan plan, uint64_t now);
    Decision next(uint64_t now, std::string &reason);
    void sent(uint64_t now) {
        last_sent_ = now;
        pending_ = false;
    }
    const Plan &plan() const { return plan_; }
    bool holding() const { return holding_; }
    bool fresh() const { return pending_; } // the next Send is a new plan, not a renewal

  private:
    SupervisorConfig cfg_;
    Plan plan_;
    bool holding_ = false, pending_ = false, stopped_ = false;
    uint64_t last_sent_ = 0, hold_started_ = 0;
    std::string stop_reason_;
};

class ControlLink {
  public:
    explicit ControlLink(const std::string &socket_path);
    void send(Operation op, uint32_t flags = 0, const Plan *plan = nullptr);
    std::string status();

  private:
    Fd fd_; // move-only, so the link is too
    uint64_t sequence_ = 0;
};

struct ProducerOptions {
    std::string socket;
    bool allow_live = false, pump = false;
    ActionGate gate;
    SupervisorConfig supervisor;
    double poll_s = 0.005, status_s = 0.05;
};
struct ProducerCounters {
    uint64_t plans = 0, renewals = 0, holds = 0;
    double max_send_gap_ms = 0;
};
// Owns the drive connection on its own thread: arms with a centered hold, sends each
// chunk once, renews it with the same t0/observation until a newer chunk or a stop rule,
// and disarms on any stop, fault or error. Never re-arms by itself.
class Producer {
  public:
    explicit Producer(ProducerOptions options);
    ~Producer();
    void arm(double timeout_s = 2); // blocks until armed or throws
    bool post(Plan plan, std::string &reason);
    void stop(const std::string &reason);
    bool running();
    std::string stop_reason();
    ProducerCounters counters();

  private:
    ProducerOptions options_;
    ControlLink link_;
    PlanSupervisor supervisor_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread thread_;
    bool running_ = false, stopping_ = false, disarmed_ = false;
    std::string reason_;
    ProducerCounters counters_;
    void loop();
    void finish(const std::string &reason);
};
} // namespace kaivuri::pi05
