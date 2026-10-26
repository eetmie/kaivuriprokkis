#pragma once
// CPU reference and integration contract. No CUDA, robot I/O or actuator authority.
// All tensors are one batch of NORMALIZED, PADDED model-space actions, row-major.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kaivuri::flow {
using Tensor = std::vector<float>;
enum class Convention { OpenPi, Groot17 };
struct Step { float time, dt; };
// OpenPI: noise time 1 -> 0. GR00T: noise time 0 -> 1. Steps are bundle metadata.
std::vector<Step> schedule(Convention convention, int steps);

struct Grid {
    uint64_t origin_ns = 0;
    double fps = 0;
    uint64_t time_ns(uint64_t tick) const;
    uint64_t floor_tick(uint64_t time_ns) const;
    uint64_t ceil_tick(uint64_t time_ns) const;
};
struct Chunk {
    Tensor actions;
    size_t horizon = 0, width = 0;
    Grid grid;
    uint64_t first_tick = 0, observation_ns = 0;
    uint64_t task_epoch = 0;
    // Includes checkpoint, stats, units, padding and action representation identity.
    std::string contract_id;
};
struct Request {
    size_t horizon = 0, width = 0;
    Grid grid;
    // Model row zero is on the tick containing the current observation, NOT launch.
    // launch may be later; the frozen interval includes that observation age.
    uint64_t first_tick = 0, observation_ns = 0, launched_ns = 0;
    uint64_t latency_budget_ns = 0, task_epoch = 0;
    std::string contract_id;
};
struct Prefix {
    Tensor actions; // request shape; zeros past overlap, never treated as constraints
    size_t overlap = 0, frozen = 0;
    uint64_t deadline_ns = 0;
};
// Caller supplies the chunk actually accepted for execution, before unnormalization.
// Common action grid avoids interpolation of valve commands. Mismatches fail closed.
// For observation-relative actions the caller MUST rebase to the new observation
// before calling; initial excavator valve actions are observation-independent.
Prefix align_prefix(const Chunk &accepted, const Request &request);
// Refuse a late completion: it exceeded the prefix protected against inference delay.
void check_completion(const Prefix &prefix, uint64_t ready_ns);

struct GrootSeed { Tensor actions, velocity_strength; };
// NVIDIA's experimental overlap initialization and normalized exponential ramp.
GrootSeed groot_seed(const Tensor &noise, size_t horizon, size_t width,
                     const Prefix &prefix, float ramp_rate);
Tensor euler(const Tensor &actions, const Tensor &velocity, float dt,
             const Tensor *velocity_strength = nullptr);
// Adapter for a GR00T engine whose output already includes the unmasked Euler step.
// FP32 reconstruction can round differently from masking velocity before integration.
Tensor groot_post_step(const Tensor &actions, const Tensor &unmasked_next,
                       const Tensor &velocity_strength);

enum class PrefixSchedule { Linear, Exp, Ones, Zeros };
Tensor prefix_weights(size_t frozen, size_t overlap, size_t horizon,
                      PrefixSchedule schedule = PrefixSchedule::Exp);
// VJP of clean-action estimate D(x,s) = x - s*v_openpi(x,s), not of v alone.
// Forward-only TensorRT engines do not implement this. No identity approximation.
using EndpointVjp = std::function<Tensor(const Tensor &x, float noise_time,
                                         const Tensor &cotangent)>;
Tensor openpi_guided_velocity(const Tensor &x, const Tensor &velocity, float noise_time,
                             const Prefix &prefix, size_t width,
                             PrefixSchedule weights, float max_guidance,
                             const EndpointVjp &endpoint_vjp);
} // namespace kaivuri::flow
