#include "kaivuri/core.hpp"
#include <cmath>
#include <cstdio>
using namespace kaivuri;
int main(int argc, char **argv) {
    if (argc != 2)
        return 1;
    Config c = load_config(argv[1], std::string(NATIVE_SOURCE) + "/config/safety.yaml");
    for (const auto &valve : c.valves)
        for (int variant = 0; variant < 3; ++variant) {
            Valve v = valve;
            if (variant == 1) {
                v.ramp = true;
                v.skip_deadband = true;
                v.ramp_limit = 500;
            }
            if (variant == 2) {
                v.taper = true;
                v.taper_us = 50;
                v.gamma = 1.5;
            }
            RampState state;
            for (int step = 0; step < 161; ++step) {
                double value = step < 81 ? (-1 + step * .025) : (1 - (step - 80) * .025);
                double now = 10 + step * .013;
                double pulse = valve_pulse(v, value, now, state);
                std::printf("{\"type\":\"pwm\",\"name\":\"%s\",\"variant\":%d,\"value\":%.17g,"
                            "\"now\":%.17g,\"pulse\":%.17g,\"count\":%u}\n",
                            v.name.c_str(), variant, value, now, pulse,
                            pulse_count(pulse, c.frequency));
            }
        }
    for (int i = 0; i < 200; ++i) {
        std::array<double, 4> angles{.01 * i - 1, .73 * std::sin(i), .44 * std::cos(i),
                                     .91 * std::sin(i * .3)};
        Pose pose = forward_kinematics(c, angles);
        std::printf("{\"type\":\"fk\",\"angles\":[%.17g,%.17g,%.17g,%.17g],\"position\":[%.17g,%."
                    "17g,%.17g],\"orientation\":[%.17g,%.17g,%.17g,%.17g]}\n",
                    angles[0], angles[1], angles[2], angles[3], pose.position[0], pose.position[1],
                    pose.position[2], pose.orientation[0], pose.orientation[1], pose.orientation[2],
                    pose.orientation[3]);
    }
}
