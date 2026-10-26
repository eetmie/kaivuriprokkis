// Executable integration example with an analytic velocity field, not model weights.
// Prints model-space sampler traces; never connects to cameras or control sockets.
#include "kaivuri/flow_rtc.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace kaivuri::flow;
int main(int argc, char **argv) {
    try {
        std::string mode = argc == 2 ? argv[1] : "";
        if (mode != "groot17" && mode != "openpi")
            throw std::invalid_argument("usage: flow-rtc-probe groot17|openpi (analytic field only)");
        bool groot = mode == "groot17";
        Grid grid{1000000000ULL, 30};
        Chunk old{Tensor(40 * 4, .1f), 40, 4, grid, 0, grid.origin_ns, 1, "analytic-fixture"};
        Request request{40, 4, grid, 8, grid.time_ns(8), grid.time_ns(8),
                        300000000, 1, old.contract_id};
        auto prefix = align_prefix(old, request);
        Tensor noise(160, -.5f), x = noise;
        GrootSeed seed;
        if (groot) { seed = groot_seed(noise, 40, 4, prefix, 2); x = seed.actions; }
        std::cout << "{\"analytic_only\":true,\"actuator_authority\":false,\"mode\":\""
                  << mode << "\",\"overlap\":" << prefix.overlap << ",\"frozen\":"
                  << prefix.frozen << ",\"trace\":[";
        bool first = true;
        for (auto step : schedule(groot ? Convention::Groot17 : Convention::OpenPi, groot ? 4 : 10)) {
            // Progress velocity u(x)=.3-.2*x. OpenPI uses v=-u.
            Tensor velocity(x.size());
            for (size_t i = 0; i < x.size(); ++i) velocity[i] = (groot ? 1 : -1) * (.3f - .2f * x[i]);
            if (!groot) {
                EndpointVjp vjp = [](const Tensor &, float s, const Tensor &e) {
                    Tensor out(e);
                    for (float &v : out) v *= 1 - .2f * s;
                    return out;
                };
                velocity = openpi_guided_velocity(x, velocity, step.time, prefix, 4,
                                                   PrefixSchedule::Exp, 5, vjp);
            }
            x = euler(x, velocity, step.dt, groot ? &seed.velocity_strength : nullptr);
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"time\":" << step.time << ",\"dt\":" << step.dt
                      << ",\"first_action\":" << x.front() << '}';
        }
        check_completion(prefix, request.launched_ns + request.latency_budget_ns - 1);
        std::cout << "],\"deadline_check\":\"PASS\",\"actions\":[";
        for (size_t i = 0; i < x.size(); ++i) std::cout << (i ? "," : "") << x[i];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
