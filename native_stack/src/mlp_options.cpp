#include "kaivuri/mlp_collect.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

namespace kaivuri::mlp {
namespace fs = std::filesystem;
Options parse_options(int argc, char **argv) {
    Options o;
    o.profile = fs::path(NATIVE_SOURCE).parent_path() / "configuration_files/profiles/jetson";
    bool seed_given = false, start_given = false, end_given = false, sweep_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() {
            if (++i >= argc)
                throw std::invalid_argument("missing value for " + a);
            return std::string(argv[i]);
        };
        auto real = [&]() {
            const auto value = val();
            size_t used = 0;
            const double number = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(number))
                throw std::invalid_argument("invalid number for " + a);
            return number;
        };
        auto integer = [&]() {
            const auto value = val();
            size_t used = 0;
            const auto number = std::stoull(value, &used);
            if (value.empty() || value[0] == '-' || used != value.size())
                throw std::invalid_argument("invalid integer for " + a);
            return number;
        };
        if (a == "--out")
            o.out = val();
        else if (a == "--profile")
            o.profile = val();
        else if (a == "--serial")
            o.serial = val();
        else if (a == "--gamepad")
            o.gamepad = val();
        else if (a == "--suffix")
            o.suffix = val();
        else if (a == "--dry")
            o.dry = true;
        else if (a == "--scripted")
            o.scripted = true;
        else if (a == "--enable-slew")
            o.slew = true;
        else if (a == "--enable-tracks")
            o.tracks = true;
        else if (a == "--record-on-start")
            o.record_on_start = true;
        else if (a == "--duration-s")
            o.duration = real();
        else if (a == "--record-seconds")
            o.record_seconds = real();
        else if (a == "--queue-records")
            o.queue = integer();
        else if (a == "--read-chunk-bytes")
            o.read_chunk = integer();
        else if (a == "--excitation") {
            const auto value = val();
            if (value != "sine" && value != "chirp" && value != "stroke")
                throw std::invalid_argument("excitation must be sine, chirp or stroke");
            o.excitation.waveform = value == "sine"    ? Waveform::Sine
                                    : value == "chirp" ? Waveform::Chirp
                                                       : Waveform::Stroke;
        } else if (a == "--excitation-target")
            o.excitation.target = target_mask(val());
        else if (a == "--excitation-seed") {
            const auto seed = integer();
            if (seed > UINT32_MAX)
                throw std::invalid_argument("seed must be uint32");
            o.excitation.seed = uint32_t(seed);
            seed_given = true;
        } else if (a == "--excitation-amplitude")
            o.excitation.amplitude = real();
        else if (a == "--chirp-start-hz") {
            o.excitation.start_hz = real();
            start_given = true;
        } else if (a == "--chirp-end-hz") {
            o.excitation.end_hz = real();
            end_given = true;
        } else if (a == "--chirp-seconds") {
            o.excitation.sweep_s = real();
            sweep_given = true;
        } else if (a == "--stroke-seconds")
            o.excitation.stroke_s = real();
        else if (a == "--stroke-margin-deg")
            o.excitation.margin_deg = real();
        else if (a == "--help") {
            std::cout
                << "kaivuri-mlp-collect --out DIR --gamepad /dev/input/eventN [--profile DIR]\n"
                   "  [--serial /dev/ttyACM0 --suffix LABEL --dry --enable-slew --enable-tracks]\n"
                   "  [--excitation sine|chirp|stroke --excitation-target "
                   "all|lift|tilt|scoop|lift+tilt|lift+scoop|tilt+scoop|slew]\n"
                   "  [--excitation-seed N --excitation-amplitude A --chirp-start-hz F "
                   "--chirp-end-hz F --chirp-seconds S]\n"
                   "  [--stroke-seconds 8 --stroke-margin-deg 10 --record-seconds 600]\n"
                   "  [--scripted --record-on-start --duration-s S --queue-records 4096 "
                   "--read-chunk-bytes 512]\n"
                   "PWM is LIVE by default. --dry skips I2C. No arming/lease state.\n"
                   "A=record/save B=excitation X=pump Y=reload between strips D-pad=target.\n"
                   "Scripted stdin: start, save, pump, excitation, target NAME, command "
                   "SIX_VALUES, quit.\n";
            std::exit(0);
        } else
            throw std::invalid_argument("unknown option: " + a);
    }
    if (o.out.empty() || (!o.scripted && o.gamepad.empty()) || (o.scripted && !o.gamepad.empty()) ||
        !std::isfinite(o.duration) || o.duration < 0 || !std::isfinite(o.record_seconds) ||
        o.record_seconds <= 0 || o.read_chunk == 0 || o.read_chunk > 8192 || o.queue < 16 ||
        o.queue > 65536 || (o.queue & (o.queue - 1)))
        throw std::invalid_argument("invalid output, input, duration or queue options");
    if (o.excitation.target == 1 && !o.slew)
        throw std::invalid_argument("slew excitation requires --enable-slew");
    for (char &c : o.suffix)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '-' || c == '_'))
            c = '_';
    if (o.suffix.size() > 100)
        throw std::invalid_argument("suffix too long");
    if (!seed_given)
        o.excitation.seed = std::random_device{}();
    o.fixed_seed = seed_given;
    if (o.excitation.waveform == Waveform::Stroke) {
        if (!start_given)
            o.excitation.start_hz = .5;
        if (!end_given)
            o.excitation.end_hz = 12;
        if (!sweep_given)
            o.excitation.sweep_s = 20;
    }
    return o;
}
Runtime load_runtime(const Options &o) {
    Runtime r;
    r.config = load_hardware_config(o.profile.string());
    if (r.config.frequency != 100)
        throw std::runtime_error("MLP collection requires a 100 Hz PWM profile");
    r.config.control_hz = 100;
    r.metadata.config = r.config;
    r.metadata.dry = o.dry;
    r.metadata.slew = o.slew;
    r.metadata.tracks = o.tracks;
    r.metadata.suffix = o.suffix;
    r.metadata.source_revision = NATIVE_GIT_REVISION;
    r.metadata.collection_id =
        std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count()) +
        "-" + std::to_string(getpid());
    r.metadata.executable_sha256 = sha256_file("/proc/self/exe");
    std::ostringstream effective;
    effective << std::setprecision(17)
              << "{\"profile\":" << json_quote(fs::absolute(o.profile).string())
              << ",\"serial\":" << json_quote(o.serial.string())
              << ",\"gamepad\":" << json_quote(o.gamepad)
              << ",\"scripted\":" << (o.scripted ? "true" : "false")
              << ",\"duration_s\":" << o.duration << ",\"record_seconds\":" << o.record_seconds
              << ",\"read_chunk_bytes\":" << o.read_chunk
              << ",\"fixed_seed\":" << (o.fixed_seed ? "true" : "false")
              << ",\"pwm_frequency_hz\":" << r.config.frequency << '}';
    r.metadata.options_json = effective.str();
    auto read = [](const fs::path &path) {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("cannot read " + path.string());
        return std::string(std::istreambuf_iterator<char>(in), {});
    };
    const auto p = YAML::LoadFile((o.profile / "profile.yaml").string());
    const auto control =
        YAML::LoadFile((o.profile / p["control_config_file"].as<std::string>()).string());
    const auto limits = control["ik"]["joint_limits_relative"];
    for (unsigned i = 0; i < 4; ++i) {
        r.limits[i] = {-180, 180};
        if (limits && limits.size() > i && limits[i] && !limits[i].IsNull())
            r.limits[i] = {limits[i][0].as<double>(), limits[i][1].as<double>()};
        else if (i > 0 && o.excitation.waveform == Waveform::Stroke)
            throw std::runtime_error(
                "stroke requires joint_limits_relative for boom, arm and bucket");
    }
    for (auto name : {std::string("profile.yaml"), p["control_config_file"].as<std::string>(),
                      p["servo_config_file"].as<std::string>()})
        r.metadata.files.emplace_back("profile/" + name, read(o.profile / name));
    for (const auto &entry : fs::directory_iterator(fs::path(NATIVE_SOURCE) / "src"))
        if (entry.path().extension() == ".cpp")
            r.metadata.files.emplace_back("source/" + entry.path().filename().string(),
                                          read(entry.path()));
    for (const auto &entry : fs::directory_iterator(fs::path(NATIVE_SOURCE) / "include/kaivuri"))
        if (entry.path().extension() == ".hpp")
            r.metadata.files.emplace_back("include/kaivuri/" + entry.path().filename().string(),
                                          read(entry.path()));
    r.metadata.files.emplace_back("CMakeLists.txt",
                                  read(std::string(NATIVE_SOURCE) + "/CMakeLists.txt"));
    return r;
}
} // namespace kaivuri::mlp
