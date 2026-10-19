#pragma once
#include "kaivuri/mlp_record.hpp"

namespace kaivuri::mlp {
struct Options {
    std::filesystem::path profile, out, serial = "/dev/ttyACM0";
    std::string gamepad, suffix;
    bool dry = false, scripted = false, slew = false, tracks = false, record_on_start = false;
    bool fixed_seed = false;
    double duration = 0, record_seconds = 600;
    size_t queue = 4096, read_chunk = 512;
    ExcitationOptions excitation{};
};
struct Runtime {
    Config config;
    std::array<std::array<double, 2>, 4> limits{};
    RecordingMetadata metadata;
};
Options parse_options(int argc, char **argv);
Runtime load_runtime(const Options &);
} // namespace kaivuri::mlp
