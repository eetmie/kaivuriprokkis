#include "kaivuri/flow_rtc.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace kaivuri::flow;
static int checks = 0;
static void expect(bool ok, const char *label) {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}
template<class F> static void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::exception &) { threw = true; }
    expect(threw, "invalid request was accepted");
}
static void near(float a, float b, float tolerance = 1e-5f) {
    expect(std::abs(a - b) <= tolerance, "numerical mismatch");
}
int main() {
    try {
        for (auto convention : {Convention::OpenPi, Convention::Groot17}) {
            auto steps = schedule(convention, 10);
            Tensor x{1, 2}, v = convention == Convention::OpenPi ? Tensor{2, 3} : Tensor{-2, -3};
            for (auto s : steps) x = euler(x, v, s.dt);
            near(x[0], -1);
            near(x[1], -1);
            near(steps.front().time, convention == Convention::OpenPi ? 1.f : 0.f);
            expect(steps.size() == 10, "wrong number of flow evaluations");
        }
        rejects([] { schedule(Convention::OpenPi, 0); });
        rejects([] { euler({1}, {2, 3}, .1); });
        rejects([] { euler({1}, {NAN}, .1); });
        Grid grid{1000000000ULL, 30};
        for (uint64_t i = 0; i < 1000; ++i) {
            expect(grid.floor_tick(grid.time_ns(i)) == i, "floor at exact rounded tick");
            expect(grid.ceil_tick(grid.time_ns(i)) == i, "ceil at exact rounded tick");
        }
        expect(grid.floor_tick(grid.time_ns(100) - 1) == 99, "floor before boundary");
        expect(grid.ceil_tick(grid.time_ns(100) + 1) == 101, "ceil after boundary");
        rejects([&] { grid.floor_tick(1); });
        Chunk old{Tensor(40 * 4), 40, 4, grid, 0, grid.origin_ns, 7, "checkpoint:stats:valves"};
        for (size_t i = 0; i < old.actions.size(); ++i) old.actions[i] = float(i) / 100;
        Request request{40, 4, grid, 10, grid.time_ns(10) + 500000,
                        grid.time_ns(10) + 1000000, 270000000, 7, old.contract_id};
        auto prefix = align_prefix(old, request);
        expect(prefix.overlap == 30 && prefix.frozen == 9, "latency/overlap alignment");
        near(prefix.actions[0], old.actions[40]);
        near(prefix.actions[119], old.actions[159]);
        near(prefix.actions[120], 0);
        check_completion(prefix, request.launched_ns + 270000000);
        rejects([&] { check_completion(prefix, prefix.deadline_ns); });
        auto bad = request;
        ++bad.task_epoch;
        rejects([&] { align_prefix(old, bad); });
        bad = request; bad.contract_id = "different stats";
        rejects([&] { align_prefix(old, bad); });
        bad = request; bad.width = 3;
        rejects([&] { align_prefix(old, bad); });
        bad = request; bad.grid.origin_ns++;
        rejects([&] { align_prefix(old, bad); });
        bad = request; bad.observation_ns = old.observation_ns;
        rejects([&] { align_prefix(old, bad); });
        bad = request; bad.latency_budget_ns = 1500000000;
        rejects([&] { align_prefix(old, bad); });
        bad = request; bad.first_tick = 40; bad.launched_ns = grid.time_ns(40);
        rejects([&] { align_prefix(old, bad); });
        // 10 ms into a 30 Hz tick + 30 ms budget crosses TWO action rows.
        bad = request; bad.launched_ns = grid.time_ns(10) + 10000000;
        bad.latency_budget_ns = 30000000;
        expect(align_prefix(old, bad).frozen == 2, "partial action tick omitted from delay");
        bad = request; bad.first_tick = 9; bad.observation_ns = grid.time_ns(9) + 2000000;
        auto aged = align_prefix(old, bad);
        expect(aged.overlap == 31 && aged.frozen == 10,
               "observation age omitted from protected prefix");
        near(aged.actions[0], old.actions[36]);
        Tensor noise(160, -2.f);
        auto seed = groot_seed(noise, 40, 4, prefix, 2.f);
        auto x = seed.actions;
        for (auto step : schedule(Convention::Groot17, 4))
            x = euler(x, Tensor(160, 1.f), step.dt, &seed.velocity_strength);
        for (size_t i = 0; i < prefix.frozen * 4; ++i)
            expect(x[i] == prefix.actions[i], "GR00T committed prefix changed");
        near(x[120], -1.f);
        auto plain_next = euler(seed.actions, Tensor(160, 1.f), .25f);
        auto post = groot_post_step(seed.actions, plain_next, seed.velocity_strength);
        auto direct = euler(seed.actions, Tensor(160, 1.f), .25f, &seed.velocity_strength);
        for (size_t i = 0; i < post.size(); ++i) near(post[i], direct[i], 2e-7f);
        expect(groot_post_step(noise, plain_next, Tensor(160, 1)) == plain_next,
               "no-RTC fused output changed");
        expect(groot_post_step(noise, plain_next, Tensor(160, 0)) == noise,
               "frozen fused output changed");
        for (size_t r = prefix.frozen; r < prefix.overlap; ++r) {
            double u = double(r - prefix.frozen + 1) / (prefix.overlap - prefix.frozen + 1);
            near(seed.velocity_strength[r * 4], float((1 - std::exp(-2 * u)) / (1 - std::exp(-2.))));
        }
        rejects([&] { groot_seed(noise, 40, 4, prefix, 0); });
        Prefix full{Tensor(160, .5f), 40, 40, 1};
        auto frozen = groot_seed(noise, 40, 4, full, 2);
        expect(euler(frozen.actions, Tensor(160, 3), .25f, &frozen.velocity_strength) == full.actions,
               "all-frozen GR00T should be immutable");
        Prefix empty{Tensor(160, 0), 0, 0, 1};
        auto plain = groot_seed(noise, 40, 4, empty, 2);
        expect(plain.actions == noise && plain.velocity_strength == Tensor(160, 1), "no-RTC parity");
        auto weights = prefix_weights(2, 6, 10, PrefixSchedule::Linear);
        Tensor expected{1, 1, .8f, .6f, .4f, .2f, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i) near(weights[i], expected[i]);
        // Coupled velocity v(x)=A*x+b. VJP must include off-diagonal derivatives:
        // J_D = I-s*A. Identity/elementwise stand-ins fail this test.
        Tensor input{.2f, -.3f}, v{-.2f, -.7f}; // A=[[2,1],[-1,3]], b=[-.3,.4]
        Prefix target{{.7f, -.2f}, 2, 1, 1};
        float s = .6f;
        EndpointVjp vjp = [](const Tensor &, float t, const Tensor &e) {
            return Tensor{(1 - 2 * t) * e[0] + t * e[1],
                          -t * e[0] + (1 - 3 * t) * e[1]};
        };
        auto guided = openpi_guided_velocity(input, v, s, target, 1, PrefixSchedule::Linear, 5, vjp);
        // Independent finite-difference gradient of .5 * weighted squared endpoint error.
        auto loss = [](double a, double b) {
            double da = a - .6 * (2 * a + b - .3) - .7;
            double db = b - .6 * (-a + 3 * b + .4) + .2;
            return .5 * (da * da + .5 * db * db);
        };
        double epsilon = 1e-6, gamma = (.4 * .4 + .6 * .6) / (.4 * .6);
        double grad0 = (loss(input[0] + epsilon, input[1]) - loss(input[0] - epsilon, input[1])) / (2 * epsilon);
        double grad1 = (loss(input[0], input[1] + epsilon) - loss(input[0], input[1] - epsilon)) / (2 * epsilon);
        near(guided[0], float(v[0] + gamma * grad0));
        near(guided[1], float(v[1] + gamma * grad1));
        rejects([&] { openpi_guided_velocity(input, v, s, target, 1, PrefixSchedule::Exp, 5, {}); });
        rejects([&] { openpi_guided_velocity(input, v, 0, target, 1, PrefixSchedule::Exp, 5, vjp); });
        auto first = openpi_guided_velocity(input, v, 1, target, 1, PrefixSchedule::Exp, 5, vjp);
        for (float value : first) expect(std::isfinite(value), "singularity at noise time 1");
        // The guidance sign must move a simple zero-velocity sample towards its target.
        EndpointVjp identity = [](const Tensor &, float, const Tensor &e) { return e; };
        auto direction = openpi_guided_velocity({0}, {0}, .5, {{1}, 1, 1, 1}, 1,
                                               PrefixSchedule::Exp, 5, identity);
        expect(euler({0}, direction, -.1f)[0] > 0, "OpenPI guidance sign reversed");
        std::cout << checks << " flow/RTC checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n';
        return 1;
    }
}
