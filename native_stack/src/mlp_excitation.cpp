#include "kaivuri/mlp_excitation.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kaivuri::mlp {
namespace {
uint64_t next(uint64_t &state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
double uniform(uint64_t &state) {
    return (double(next(state) >> 11) + .5) * 0x1.0p-53;
}
double normal(uint64_t &state) {
    const double u = uniform(state), v = uniform(state);
    return std::sqrt(-2 * std::log(u)) * std::cos(2 * pi * v);
}
double ramp(double t) {
    return .5 - .5 * std::cos(pi * std::clamp(t, 0., 1.));
}
} // namespace
const char *waveform_name(Waveform w) {
    return w == Waveform::Sine ? "sine" : w == Waveform::Chirp ? "chirp" : "stroke";
}
const char *stage_name(Stage s) {
    constexpr const char *names[]{"off", "run", "up", "down", "rest", "stroke", "stroke+chirp"};
    return names[static_cast<unsigned>(s)];
}
const char *target_name(unsigned m) {
    constexpr const char *names[]{"all",       "lift",       "tilt",       "scoop",
                                  "lift+tilt", "lift+scoop", "tilt+scoop", "slew"};
    for (unsigned i = 0; i < target_masks.size(); ++i)
        if (target_masks[i] == m)
            return names[i];
    throw std::invalid_argument("invalid excitation target");
}
unsigned target_mask(const std::string &name) {
    for (unsigned m : target_masks)
        if (name == target_name(m))
            return m;
    throw std::invalid_argument("invalid excitation target: " + name);
}
Excitation::Excitation(const ExcitationOptions &o,
                       const std::array<std::array<double, 2>, 4> &limits)
    : limits_(limits) {
    block_.options = o;
    block_.limits_deg = limits;
    target_name(o.target);
    for (double v :
         {o.start_hz, o.end_hz, o.sweep_s, o.stroke_s, o.margin_deg, o.gain_per_deg, o.valve_limit})
        if (!std::isfinite(v) || v <= 0)
            throw std::invalid_argument("invalid excitation parameter");
    if (o.start_hz > o.end_hz || o.end_hz > 15 || o.sweep_s < 2 || o.valve_limit > 1 ||
        (!std::isnan(o.amplitude) &&
         (!std::isfinite(o.amplitude) || o.amplitude <= 0 || o.amplitude > 1)))
        throw std::invalid_argument("invalid excitation range");
    if (o.waveform == Waveform::Stroke) {
        if (o.target == 1)
            throw std::invalid_argument("stroke does not drive slew");
        for (unsigned i = 1; i < 4; ++i)
            if (!std::isfinite(limits[i][0]) || !std::isfinite(limits[i][1]) ||
                limits[i][1] - limits[i][0] <= 2 * o.margin_deg)
                throw std::invalid_argument("joint range does not fit stroke margin");
    }
    begin(0);
}
void Excitation::begin(double now) {
    ++block_.id;
    uint64_t rng = uint64_t(block_.options.seed) ^ (block_.id * 0x9e3779b97f4a7c15ULL);
    noise_rng_ = next(rng);
    block_.noise_seed = noise_rng_;
    for (auto &p : block_.parameters) {
        auto draw = [&](double low, double high) { return low + (high - low) * uniform(rng); };
        p.env = .03 * draw(.60, 1.50);
        // Equal weight per frequency ratio retains slow strokes and fast corrections.
        p.carrier = std::exp(draw(std::log(sine_carrier_min_hz), std::log(sine_carrier_max_hz)));
        p.rate = .04 * draw(.70, 1.10);
        p.depth = std::min(.99 * draw(.70, 1.20),
                           std::max(0., (sine_max_instant_hz / p.carrier - 1) / (2 * pi * p.rate)));
        p.cutoff = .25 * draw(.70, 1.30);
        p.noise = block_.options.waveform == Waveform::Sine ? .10 * draw(.70, 1.30) : 0;
        p.amplitude = std::isnan(block_.options.amplitude)
                          ? (block_.options.waveform == Waveform::Sine     ? draw(.35, 1.)
                             : block_.options.waveform == Waveform::Stroke ? .4
                                                                           : .35)
                          : block_.options.amplitude;
        p.phi1 = draw(0, 2 * pi);
        p.phi2 = draw(0, 2 * pi);
    }
    for (auto &d : direction_)
        d = uniform(rng) < .5 ? 1 : -1;
    block_.initial_direction = direction_;
    noise_coefficients();
    noise1_.fill(0);
    noise2_.fill(0);
    strokes_.fill(0);
    have_reference_.fill(false);
    noise_tick_ = 0;
    started_ = now;
    previous_ = now;
    have_previous_ = false;
}
void Excitation::noise_coefficients() {
    for (unsigned i = 0; i < 4; ++i) {
        const double alpha = std::exp(-2 * pi * block_.parameters[i].cutoff / 100);
        noise_coefficients_[i] = {alpha, std::sqrt(1 - alpha * alpha),
                                  (1 - alpha * alpha) / std::sqrt(1 + alpha * alpha)};
    }
}
void Excitation::enable(double now) {
    enabled_ = true;
    begin(now);
}
void Excitation::disable() {
    enabled_ = false;
    stage_ = Stage::Off;
    have_previous_ = false;
}
void Excitation::select(unsigned target, double now) {
    target_name(target);
    if (block_.options.waveform == Waveform::Stroke && target == 1)
        throw std::invalid_argument("stroke does not drive slew");
    block_.options.target = target;
    begin(now);
}
void Excitation::reseed(uint32_t seed, double now) {
    block_.options.seed = seed;
    block_.id = 0;
    begin(now);
}
double Excitation::phase(double t) const {
    const auto &o = block_.options;
    const double k = std::log(o.end_hz / o.start_hz) / o.sweep_s;
    if (std::abs(k) < 1e-12)
        return 2 * pi * o.start_hz * t;
    if (t <= o.sweep_s)
        return 2 * pi * o.start_hz * std::expm1(k * t) / k;
    return 2 * pi *
           (o.start_hz * std::expm1(k * o.sweep_s) / k +
            o.end_hz * std::expm1(-k * (t - o.sweep_s)) / -k);
}
Commands Excitation::sample(double now, const std::array<double, 4> &q, bool fresh) {
    Commands result{};
    if (!enabled_) {
        stage_ = Stage::Off;
        return result;
    }
    const auto &o = block_.options;
    const double e = std::max(0., now - started_);
    if (o.waveform == Waveform::Stroke && !fresh) {
        have_previous_ = false;
        stage_ = Stage::Off;
        return result;
    }
    if (o.waveform == Waveform::Sine) {
        const auto tick = static_cast<uint64_t>(std::floor(e * 100 + 1e-7));
        while (noise_tick_ < tick) {
            for (unsigned i = 0; i < 4; ++i) {
                const auto &c = noise_coefficients_[i];
                noise1_[i] = std::clamp(c[0] * noise1_[i] + c[1] * normal(noise_rng_), -3., 3.);
                noise2_[i] = std::clamp(c[0] * noise2_[i] + c[2] * noise1_[i], -3., 3.);
            }
            ++noise_tick_;
        }
        stage_ = Stage::Run;
    }
    const double dt = have_previous_ ? std::clamp(now - previous_, 0., .1) : 0;
    previous_ = now;
    have_previous_ = true;
    bool chirping = false;
    for (unsigned i = 0; i < 4; ++i) {
        if (!(o.target & (1U << i)))
            continue;
        const auto &p = block_.parameters[i];
        double value = 0;
        if (o.waveform == Waveform::Sine) {
            const double env = std::sin(2 * pi * p.env * e + p.phi1);
            const double carrier = std::sin(
                2 * pi * p.carrier * (e + p.depth * std::sin(2 * pi * p.rate * e)) + p.phi2);
            const double limit = std::isnan(o.amplitude) ? 1 : o.amplitude;
            value = std::clamp(ramp(e) * p.amplitude * (env * carrier + p.noise * noise2_[i]),
                               -limit, limit);
        } else if (o.waveform == Waveform::Chirp) {
            const double t = std::fmod(e, 2 * o.sweep_s + 10);
            stage_ = t < o.sweep_s ? Stage::Up : t < 2 * o.sweep_s ? Stage::Down : Stage::Rest;
            if (t < 2 * o.sweep_s)
                value =
                    p.amplitude * ramp(t) * ramp(2 * o.sweep_s - t) * std::sin(phase(t) + p.phi2);
        } else {
            const double angle = q[i] * 180 / pi;
            if (!std::isfinite(angle) || angle < limits_[i][0] || angle > limits_[i][1]) {
                disable();
                return {};
            }
            const double lo = limits_[i][0] + o.margin_deg, hi = limits_[i][1] - o.margin_deg;
            if (!have_reference_[i]) {
                reference_[i] = std::clamp(angle, lo, hi);
                have_reference_[i] = true;
            }
            reference_[i] += direction_[i] * (hi - lo) / o.stroke_s * dt;
            if (reference_[i] < lo || reference_[i] > hi) {
                reference_[i] = std::clamp(reference_[i], lo, hi);
                direction_[i] *= -1;
                ++strokes_[i];
            }
            value = ramp(e) * std::clamp(o.gain_per_deg * (reference_[i] - angle), -o.valve_limit,
                                         o.valve_limit);
            if (strokes_[i] % 2 == 0) {
                chirping = true;
                value +=
                    p.amplitude * ramp(e) * std::sin(phase(std::fmod(e, 2 * o.sweep_s)) + p.phi2);
            }
        }
        result[i] = std::clamp(value, -1., 1.);
    }
    if (o.waveform == Waveform::Stroke)
        stage_ = chirping ? Stage::StrokeChirp : Stage::Stroke;
    return result;
}
} // namespace kaivuri::mlp
