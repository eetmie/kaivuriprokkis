#pragma once
#include "kaivuri/core.hpp"
#include <algorithm>
#include <limits>

namespace kaivuri::mlp {
enum class Waveform { Sine, Chirp, Stroke };
enum class Stage { Off, Run, Up, Down, Rest, Stroke, StrokeChirp };
const char *waveform_name(Waveform);
const char *stage_name(Stage);
const char *target_name(unsigned mask);
unsigned target_mask(const std::string &);
struct ExcitationOptions {
    Waveform waveform = Waveform::Sine;
    uint32_t seed = 42;
    unsigned target = 14;
    double amplitude = std::numeric_limits<double>::quiet_NaN();
    double start_hz = .05, end_hz = .9, sweep_s = 60;
    double stroke_s = 8, margin_deg = 10, gain_per_deg = .05, valve_limit = .5;
};
struct WaveParameters {
    double env = 0, carrier = 0, rate = 0, depth = 0, cutoff = 0;
    double noise = 0, amplitude = 0, phi1 = 0, phi2 = 0;
};
struct ExcitationBlock {
    ExcitationOptions options{};
    uint64_t id = 0, noise_seed = 0;
    std::array<double, 4> initial_direction{};
    std::array<std::array<double, 2>, 4> limits_deg{};
    std::array<WaveParameters, 4> parameters{};
};
class Excitation {
    ExcitationBlock block_{};
    std::array<std::array<double, 2>, 4> limits_{};
    std::array<double, 4> noise1_{}, noise2_{}, reference_{}, direction_{};
    std::array<std::array<double, 3>, 4> noise_coefficients_{};
    std::array<unsigned, 4> strokes_{};
    std::array<bool, 4> have_reference_{};
    uint64_t noise_rng_ = 0, noise_tick_ = 0;
    double started_ = 0, previous_ = 0;
    bool enabled_ = false, have_previous_ = false;
    Stage stage_ = Stage::Off;
    void begin(double now);
    void noise_coefficients();

  public:
    Excitation(const ExcitationOptions &, const std::array<std::array<double, 2>, 4> &limits);
    void enable(double now);
    void disable();
    void select(unsigned target, double now);
    void reseed(uint32_t seed, double now);
    Commands sample(double now, const std::array<double, 4> &angles_rad, bool fresh);
    double phase(double elapsed) const;
    bool enabled() const { return enabled_; }
    Stage stage() const { return stage_; }
    double elapsed(double now) const { return enabled_ ? std::max(0., now - started_) : 0; }
    uint64_t noise_tick() const { return noise_tick_; }
    const ExcitationBlock &block() const { return block_; }
    // Used for offline replay and analytic cross-language fixtures.
    void restore_parameters(const std::array<WaveParameters, 4> &p) {
        block_.parameters = p;
        noise_coefficients();
    }
};
} // namespace kaivuri::mlp
