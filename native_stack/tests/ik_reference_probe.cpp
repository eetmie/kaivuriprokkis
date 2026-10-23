#include "kaivuri/ik_collection.hpp"
#include <cmath>
#include <cstdio>
using namespace kaivuri;
using namespace kaivuri::collection;
int main(int argc, char **argv) {
    if (argc != 2)
        return 1;
    const auto robot = load_robot(argv[1]);
    const Kinematics k(robot);
    for (unsigned n = 0; n < 400; ++n) {
        Angles q{};
        for (unsigned j = 0; j < 3; ++j) {
            const auto &l = robot.limits[j];
            q[j] = l[0] + (l[1] - l[0]) * (.5 + .5 * std::sin(n * .31 + j));
        }
        const auto pose = k.state(q).pose;
        std::printf("{\"type\":\"fk\",\"q\":[%.17g,%.17g,%.17g],\"pose\":[%.17g,%.17g,%.17g]}\n",
                    q[0], q[1], q[2], pose[0], pose[1], pose[2]);
    }
    for (unsigned scenario = 0; scenario < 6; ++scenario) {
        const Gains g{scenario == 0 ? 0. : 2., scenario == 1 ? 0. : .4, scenario == 2 ? 0. : .3,
                      scenario == 3 ? 0. : .1};
        Pid pid(g, .5);
        for (unsigned n = 0; n < 500; ++n) {
            const double dt = n % 17 == 0 ? .0001 : .01 + .01 * (n % 4);
            const double measured = .4 * std::sin(n * .08);
            const double target = scenario == 4 ? (n < 250 ? 2. : -.2) : .7 * std::cos(n * .013);
            const double out = pid.compute(target, measured, dt);
            std::printf("{\"type\":\"pid\",\"scenario\":%u,\"gains\":[%.17g,%.17g,%.17g,%.17g],"
                        "\"dt\":%.17g,\"target\":%.17g,\"measured\":%.17g,\"output\":%.17g,"
                        "\"integral\":%.17g,\"derivative\":%.17g}\n",
                        scenario, g.kp, g.ki, g.kd, g.tau, dt, target, measured, out,
                        pid.integral(), pid.derivative());
        }
    }
}
