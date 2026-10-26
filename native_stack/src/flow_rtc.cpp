#include "kaivuri/flow_rtc.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kaivuri::flow {
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}
size_t count(size_t h, size_t w) {
    require(h > 0 && w > 0 && h <= 128 && w <= 4096, "invalid action shape");
    return h * w;
}
void finite(const Tensor &x) {
    for (float v : x) require(std::isfinite(v), "non-finite model action/velocity");
}
void grid_valid(const Grid &g) {
    require(g.origin_ns > 0 && std::isfinite(g.fps) && g.fps > 0 && g.fps <= 500,
            "invalid action grid");
}
void prefix_valid(const Prefix &p, size_t h, size_t w) {
    require(p.actions.size() == count(h, w), "prefix tensor shape mismatch");
    require(p.frozen <= p.overlap && p.overlap <= h, "invalid RTC overlap/frozen");
    finite(p.actions);
}
uint64_t as_uint(long double value) {
    require(value >= 0 && value < static_cast<long double>(UINT64_MAX), "timestamp overflow");
    return static_cast<uint64_t>(value);
}
} // namespace

std::vector<Step> schedule(Convention convention, int steps) {
    require(steps > 0 && steps <= 1000, "invalid denoising steps");
    std::vector<Step> result;
    for (int i = 0; i < steps; ++i) {
        result.push_back(convention == Convention::OpenPi
                             ? Step{float(1. + i * (-1. / steps)), float(-1. / steps)}
                             : Step{float(double(i) / steps), float(1. / steps)});
    }
    return result;
}
uint64_t Grid::time_ns(uint64_t tick) const {
    grid_valid(*this);
    return as_uint(static_cast<long double>(origin_ns) +
                   std::round(static_cast<long double>(tick) * 1e9L / fps));
}
uint64_t Grid::floor_tick(uint64_t ns) const {
    grid_valid(*this);
    require(ns >= origin_ns, "time precedes action grid");
    uint64_t tick = as_uint(std::floor(static_cast<long double>(ns - origin_ns) * fps / 1e9L));
    // Match time_ns's rounded integer-nanosecond representation at exact boundaries.
    if (time_ns(tick) > ns && tick > 0) --tick;
    if (time_ns(tick + 1) <= ns) ++tick;
    return tick;
}
uint64_t Grid::ceil_tick(uint64_t ns) const {
    uint64_t tick = floor_tick(ns);
    return time_ns(tick) == ns ? tick : tick + 1;
}
Prefix align_prefix(const Chunk &old, const Request &r) {
    require(old.actions.size() == count(old.horizon, old.width), "old chunk shape mismatch");
    count(r.horizon, r.width);
    finite(old.actions);
    grid_valid(old.grid);
    grid_valid(r.grid);
    require(!r.contract_id.empty() && old.contract_id == r.contract_id,
            "RTC checkpoint/action contract changed: reset required");
    require(old.task_epoch == r.task_epoch, "RTC task epoch changed: reset required");
    require(old.width == r.width && old.horizon == r.horizon, "RTC action shape changed");
    require(old.grid.origin_ns == r.grid.origin_ns && old.grid.fps == r.grid.fps,
            "RTC requires a common action grid");
    require(old.observation_ns > 0 && r.observation_ns > old.observation_ns &&
                r.observation_ns <= r.launched_ns, "RTC observation must be newer and available");
    require(r.first_tick >= old.first_tick, "RTC cannot move action time backwards");
    require(r.grid.floor_tick(r.observation_ns) == r.first_tick,
            "new chunk must start on the action tick containing its observation");
    require(r.latency_budget_ns > 0 && r.latency_budget_ns <= 5000000000ULL &&
                r.launched_ns <= UINT64_MAX - r.latency_budget_ns, "invalid latency budget");
    uint64_t shift = r.first_tick - old.first_tick;
    require(shift < old.horizon, "previous chunk exhausted");
    Prefix p;
    p.actions.assign(count(r.horizon, r.width), 0.f);
    p.overlap = std::min(r.horizon, old.horizon - size_t(shift));
    uint64_t commit_tick = r.grid.ceil_tick(r.launched_ns + r.latency_budget_ns);
    uint64_t frozen = commit_tick - r.first_tick;
    require(frozen <= p.overlap, "insufficient previous actions to cover inference budget");
    p.frozen = size_t(frozen);
    p.deadline_ns = r.grid.time_ns(commit_tick);
    std::copy_n(old.actions.begin() + size_t(shift) * r.width,
                p.overlap * r.width, p.actions.begin());
    return p;
}
void check_completion(const Prefix &p, uint64_t ready_ns) {
    require(p.deadline_ns > 0 && ready_ns < p.deadline_ns,
            "RTC inference missed the protected prefix deadline");
}
GrootSeed groot_seed(const Tensor &noise, size_t h, size_t w, const Prefix &p, float rate) {
    prefix_valid(p, h, w);
    require(noise.size() == count(h, w), "noise shape mismatch");
    require(std::isfinite(rate) && rate > 0, "RTC ramp rate must be positive");
    finite(noise);
    GrootSeed result{noise, Tensor(noise.size(), 1.f)};
    std::copy_n(p.actions.begin(), p.overlap * w, result.actions.begin());
    size_t middle = p.overlap - p.frozen;
    for (size_t row = 0; row < p.overlap; ++row) {
        float strength = 0;
        if (row >= p.frozen) {
            double t = double(row - p.frozen + 1) / double(middle + 1);
            strength = float(-std::expm1(-double(rate) * t) / -std::expm1(-double(rate)));
        }
        std::fill_n(result.velocity_strength.begin() + row * w, w, strength);
    }
    return result;
}
Tensor euler(const Tensor &x, const Tensor &v, float dt, const Tensor *strength) {
    require(!x.empty() && x.size() == v.size() && std::isfinite(dt), "Euler shape/dt mismatch");
    finite(x);
    finite(v);
    if (strength) {
        require(strength->size() == x.size(), "velocity strength shape mismatch");
        finite(*strength);
        for (float s : *strength) require(s >= 0 && s <= 1, "invalid velocity strength");
    }
    Tensor out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float delta = dt * v[i];
        if (strength) delta *= (*strength)[i];
        out[i] = x[i] + delta;
    }
    finite(out);
    return out;
}
Tensor prefix_weights(size_t frozen, size_t overlap, size_t h, PrefixSchedule mode) {
    require(h > 0 && h <= 128 && frozen <= overlap && overlap <= h,
            "invalid prefix weight range");
    Tensor weights(h, 0.f);
    for (size_t i = 0; i < overlap; ++i) {
        double w = 0;
        switch (mode) {
        case PrefixSchedule::Ones: w = 1; break;
        case PrefixSchedule::Zeros: w = i < frozen ? 1 : 0; break;
        case PrefixSchedule::Linear:
        case PrefixSchedule::Exp:
            w = std::clamp((double(frozen) - 1 - double(i)) /
                               double(overlap - frozen + 1) + 1, 0., 1.);
            if (mode == PrefixSchedule::Exp) w = w * std::expm1(w) / std::expm1(1.);
            break;
        }
        weights[i] = float(w);
    }
    return weights;
}
Tensor groot_post_step(const Tensor &x, const Tensor &next, const Tensor &strength) {
    require(!x.empty() && x.size() == next.size() && x.size() == strength.size(),
            "GR00T post-step shape mismatch");
    finite(x); finite(next); finite(strength);
    Tensor result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float s = strength[i];
        require(s >= 0 && s <= 1, "invalid GR00T velocity strength");
        result[i] = s == 0 ? x[i] : s == 1 ? next[i] : x[i] + s * (next[i] - x[i]);
    }
    finite(result);
    return result;
}
Tensor openpi_guided_velocity(const Tensor &x, const Tensor &v, float s, const Prefix &p,
                             size_t width, PrefixSchedule mode, float max_guidance,
                             const EndpointVjp &vjp) {
    require(width > 0 && !x.empty() && x.size() % width == 0 && x.size() == v.size(),
            "OpenPI guidance shape mismatch");
    size_t h = x.size() / width;
    prefix_valid(p, h, width);
    require(std::isfinite(s) && s > 0 && s <= 1, "OpenPI noise time must be in (0,1]");
    require(std::isfinite(max_guidance) && max_guidance > 0, "invalid guidance cap");
    require(bool(vjp), "OpenPI RTC requires an endpoint VJP backend; forward-only engine supplied");
    finite(x);
    finite(v);
    Tensor weights = prefix_weights(p.frozen, p.overlap, h, mode), error(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        error[i] = (p.actions[i] - (x[i] - s * v[i])) * weights[i / width];
    Tensor correction = vjp(x, s, error);
    require(correction.size() == x.size(), "endpoint VJP shape mismatch");
    finite(correction);
    // Paper uses progress t=1-s and velocity -v. Flip BOTH time and guidance sign.
    double progress = 1. - s;
    double gamma = progress == 0 ? max_guidance
                                : std::min(double(max_guidance),
                                           (progress * progress + double(s) * s) / (s * progress));
    Tensor result(x.size());
    for (size_t i = 0; i < x.size(); ++i) result[i] = v[i] - float(gamma) * correction[i];
    finite(result);
    return result;
}
} // namespace kaivuri::flow
