#pragma once
#include "kaivuri/ik_collection.hpp"
#include "kaivuri/mlp_record.hpp"

namespace kaivuri::pidcollect {
struct AppOptions {
    std::filesystem::path profile, settings, out, serial = "/dev/ttyACM0";
    std::string gamepad, suffix;
    bool dry = false, scripted = false, record_on_start = false, fixed_seed = false;
    double duration = 0, record_seconds = 600;
    size_t queue = 4096, read_chunk = 512;
    collection::Options planning;
    collection::GuardOptions guard;
    mlp::ExcitationOptions excitation;
};
struct Runtime {
    collection::Robot robot;
    mlp::RecordingMetadata metadata;
    std::array<std::array<double, 2>, 4> limits{};
    std::string protocol_json;
};
AppOptions parse_options(int argc, char **argv);
Runtime load_runtime(const AppOptions &);

struct Frame {
    uint64_t tick = 0;
    bool active = false;
    const char *reason = "idle";
    collection::Output output;
    collection::Segment segment;
    collection::GuardResult guard;
};

// Dedicated PID telemetry and protocol writer. Its preallocated SPSC queue is
// used only by the PID executable. The common writer commits its sidecars.
class PidAttachment final : public mlp::RecordingAttachment {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    PidAttachment(std::string protocol_json, size_t capacity = 4096);
    ~PidAttachment() override;
    bool push(const Frame &);
    void begin(const std::filesystem::path &, const std::string &) override;
    void drive(const mlp::DriveSample &) override;
    void close() override;
    void abort() noexcept override;
    std::vector<std::filesystem::path> paths() const override;
};
} // namespace kaivuri::pidcollect
