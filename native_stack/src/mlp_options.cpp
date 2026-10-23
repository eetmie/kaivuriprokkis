#include "kaivuri/mlp_collect.hpp"
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

namespace kaivuri::mlp {
namespace fs = std::filesystem;
namespace {
constexpr const char *usage =
    "kaivuri-mlp-collect --out DIR --gamepad /dev/input/eventN [--profile DIR]\n"
    "  [--serial /dev/ttyACM0 --suffix LABEL --dry --enable-slew --enable-tracks]\n"
    "  [--excitation sine|chirp|stroke --excitation-target "
    "all|lift|tilt|scoop|lift+tilt|lift+scoop|tilt+scoop|slew]\n"
    "  [--excitation-seed N --excitation-amplitude A --chirp-start-hz F "
    "--chirp-end-hz F --chirp-seconds S]\n"
    "  [--stroke-seconds 8 --stroke-margin-deg 10 --record-seconds 600]\n"
    "  [--scripted --record-on-start --duration-s S --queue-records 4096 "
    "--read-chunk-bytes 512]\n"
    "Defaults: sine carriers 0.2-10 Hz (log-uniform), 12 Hz instantaneous limit;\n"
    "  chirp 0.05-12 Hz, 60 s each way, amplitude 0.35; stroke 0.5-12 Hz.\n"
    "PWM is LIVE by default. --dry skips I2C. No arming/lease state.\n"
    "A=record/save B=excitation X=pump Y=reload between strips D-pad=target.\n"
    "Scripted stdin: start, save, pump, excitation, target NAME, command "
    "SIX_VALUES, quit.\n";

Waveform waveform(const std::string &name) {
    for (auto w : {Waveform::Sine, Waveform::Chirp, Waveform::Stroke})
        if (name == waveform_name(w))
            return w;
    throw std::invalid_argument("excitation must be sine, chirp or stroke");
}
std::string slurp(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot read " + path.string());
    return std::string(std::istreambuf_iterator<char>(in), {});
}
} // namespace

Options parse_options(int argc, char **argv) {
    Options o;
    auto &e = o.excitation;
    o.profile = fs::path(NATIVE_SOURCE).parent_path() / "configuration_files/profiles/jetson";
    std::optional<double> start_hz, sweep_s;
    Args args(argc, argv);
    while (args.next()) {
        if (args.is("--out"))
            o.out = args.value();
        else if (args.is("--profile"))
            o.profile = args.value();
        else if (args.is("--serial"))
            o.serial = args.value();
        else if (args.is("--gamepad"))
            o.gamepad = args.value();
        else if (args.is("--suffix"))
            o.suffix = args.value();
        else if (args.is("--dry"))
            o.dry = true;
        else if (args.is("--scripted"))
            o.scripted = true;
        else if (args.is("--enable-slew"))
            o.slew = true;
        else if (args.is("--enable-tracks"))
            o.tracks = true;
        else if (args.is("--record-on-start"))
            o.record_on_start = true;
        else if (args.is("--duration-s"))
            o.duration = args.finite();
        else if (args.is("--record-seconds"))
            o.record_seconds = args.finite();
        else if (args.is("--queue-records"))
            o.queue = args.natural();
        else if (args.is("--read-chunk-bytes"))
            o.read_chunk = args.natural();
        else if (args.is("--excitation"))
            e.waveform = waveform(args.value());
        else if (args.is("--excitation-target"))
            e.target = target_mask(args.value());
        else if (args.is("--excitation-seed")) {
            const auto seed = args.natural();
            require(seed <= UINT32_MAX, "seed must be uint32");
            e.seed = uint32_t(seed);
            o.fixed_seed = true;
        } else if (args.is("--excitation-amplitude"))
            e.amplitude = args.finite();
        else if (args.is("--chirp-start-hz"))
            start_hz = args.finite();
        else if (args.is("--chirp-end-hz"))
            e.end_hz = args.finite();
        else if (args.is("--chirp-seconds"))
            sweep_s = args.finite();
        else if (args.is("--stroke-seconds"))
            e.stroke_s = args.finite();
        else if (args.is("--stroke-margin-deg"))
            e.margin_deg = args.finite();
        else if (args.is("--help")) {
            std::cout << usage;
            std::exit(0);
        } else
            args.unknown();
    }
    require(!o.out.empty() && o.scripted == o.gamepad.empty() && o.duration >= 0 &&
                o.record_seconds > 0 && o.read_chunk > 0 && o.read_chunk <= 8192 && o.queue >= 16 &&
                o.queue <= 65536 && !(o.queue & (o.queue - 1)),
            "invalid output, input, duration or queue options");
    require(e.target != 1 || o.slew, "slew excitation requires --enable-slew");
    for (char &c : o.suffix)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_')
            c = '_';
    require(o.suffix.size() <= 100, "suffix too long");
    if (!o.fixed_seed)
        e.seed = std::random_device{}();
    // Stroke sweeps start at 0.5 Hz over 20 s unless given; the 12 Hz end is shared.
    const bool stroke = e.waveform == Waveform::Stroke;
    e.start_hz = start_hz.value_or(stroke ? .5 : e.start_hz);
    e.sweep_s = sweep_s.value_or(stroke ? 20 : e.sweep_s);
    return o;
}

Runtime load_runtime(const Options &o) {
    Runtime r;
    r.config = load_hardware_config(o.profile.string());
    require(r.config.frequency == 100, "MLP collection requires a 100 Hz PWM profile");
    r.config.control_hz = 100;
    auto &m = r.metadata;
    m.config = r.config;
    m.dry = o.dry;
    m.slew = o.slew;
    m.tracks = o.tracks;
    m.suffix = o.suffix;
    m.source_revision = NATIVE_GIT_REVISION;
    m.collection_id = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count()) +
                      "-" + std::to_string(getpid());
    m.executable_sha256 = sha256_file("/proc/self/exe");
    std::ostringstream effective;
    effective << std::setprecision(17) << std::boolalpha
              << "{\"profile\":" << json_quote(fs::absolute(o.profile).string())
              << ",\"serial\":" << json_quote(o.serial.string())
              << ",\"gamepad\":" << json_quote(o.gamepad) << ",\"scripted\":" << o.scripted
              << ",\"duration_s\":" << o.duration << ",\"record_seconds\":" << o.record_seconds
              << ",\"read_chunk_bytes\":" << o.read_chunk << ",\"fixed_seed\":" << o.fixed_seed
              << ",\"pwm_frequency_hz\":" << r.config.frequency << '}';
    m.options_json = effective.str();

    const auto profile = YAML::LoadFile((o.profile / "profile.yaml").string());
    const auto control_file = profile["control_config_file"].as<std::string>();
    const auto limits =
        YAML::LoadFile((o.profile / control_file).string())["ik"]["joint_limits_relative"];
    for (unsigned i = 0; i < 4; ++i) {
        r.limits[i] = {-180, 180};
        if (limits && limits.size() > i && limits[i] && !limits[i].IsNull())
            r.limits[i] = {limits[i][0].as<double>(), limits[i][1].as<double>()};
        else
            require(i == 0 || o.excitation.waveform != Waveform::Stroke,
                    "stroke requires joint_limits_relative for boom, arm and bucket");
    }
    // Snapshot the profile and the collector source so every take is reproducible.
    for (const auto &name : {std::string("profile.yaml"), control_file,
                             profile["servo_config_file"].as<std::string>()})
        m.files.emplace_back("profile/" + name, slurp(o.profile / name));
    const fs::path source = NATIVE_SOURCE;
    for (auto [directory, extension, prefix] :
         {std::tuple{"src", ".cpp", "source/"},
          std::tuple{"include/kaivuri", ".hpp", "include/kaivuri/"}})
        for (const auto &entry : fs::directory_iterator(source / directory))
            if (entry.path().extension() == extension)
                m.files.emplace_back(prefix + entry.path().filename().string(),
                                     slurp(entry.path()));
    m.files.emplace_back("CMakeLists.txt", slurp(source / "CMakeLists.txt"));
    return r;
}
} // namespace kaivuri::mlp
