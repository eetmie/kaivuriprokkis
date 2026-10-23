#include "kaivuri/pid_collect.hpp"
#include "kaivuri/util.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

namespace kaivuri::pidcollect {
namespace fs = std::filesystem;
namespace {
constexpr const char *usage =
    "kaivuri-pid-collect --out DIR --gamepad /dev/input/eventN [--dry]\n"
    "  --profile DIR --settings YAML --serial DEVICE --suffix LABEL --seed UINT64\n"
    "  --excitation sine|chirp --excitation-target all|lift|tilt|scoop|PAIR\n"
    "  --excitation-amplitude 0.35 --chirp-seconds 60 --record-seconds 600\n"
    "  --scripted --record-on-start --duration-s S --queue-records 4096 --read-chunk-bytes 512\n"
    "Independent PID collector; PWM is LIVE by default. --dry suppresses I2C.\n"
    "A=record/save Start=PID on/off B=overlay X=pump; sticks cancel PID and overlay.\n"
    "Scripted input: start, save, pump, assist, excitation, manual BOOM ARM BUCKET, quit.\n"
    "Joint targets/commands are bounded; obstacle/full-reach routing is not implemented.\n";

// Args::number() accepts nan/inf and Args::integer() is an int; these options need a finite
// double and an unsigned 64-bit value.
double finite(Args &args) {
    const double v = args.number();
    require(std::isfinite(v), "invalid value for " + args.flag() + ": " + std::to_string(v));
    return v;
}
uint64_t natural(Args &args) {
    const auto text = args.value();
    size_t used = 0;
    uint64_t v = 0;
    try {
        if (text[0] != '-')
            v = std::stoull(text, &used);
    } catch (const std::exception &) {
    }
    require(used > 0 && used == text.size(), "invalid value for " + args.flag() + ": " + text);
    return v;
}
std::string read(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot read " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}
double radians(const YAML::Node &degrees) {
    return degrees.as<double>() * pi / 180;
}
void settings(AppOptions &o) {
    const auto y = YAML::LoadFile(o.settings.string());
    require(y["version"].as<int>() == 1, "unsupported PID settings version");
    const auto command = y["command"], pid = y["pid"], limits = y["limits"],
               coverage = y["coverage"];
    auto &p = o.planning;
    const auto duration = command["duration_s"].as<std::array<double, 2>>();
    const auto speed = command["tip_speed_m_s"].as<std::array<double, 2>>();
    const auto gain = pid["gain_multiplier"].as<std::array<double, 2>>();
    p.min_seconds = duration[0];
    p.max_seconds = duration[1];
    p.min_speed = speed[0];
    p.max_speed = speed[1];
    p.max_pitch_change = radians(command["pitch_change_max_deg"]);
    p.pitch_rate = command["pitch_rate_rad_s"].as<double>();
    p.max_lead = command["reference_lead_m"].as<double>();
    p.pid_rates = pid["update_hz"].as<std::vector<double>>();
    p.gain_min = gain[0];
    p.gain_max = gain[1];
    p.valve_limit = pid["base_output_limit"].as<double>();
    p.joint_margin = radians(limits["joint_margin_deg"]);
    o.guard.margin = p.joint_margin;
    o.guard.taper = radians(limits["taper_deg"]);
    o.guard.velocity_lookahead = limits["velocity_lookahead_s"].as<double>();
    p.recovery_limit = limits["recovery_output_limit"].as<double>(.25);
    p.recovery_inset = limits["recovery_inset_deg"].as<double>(1.) * pi / 180;
    o.guard.command_sign = p.command_sign = limits["command_sign"].as<collection::Angles>();
    p.coverage_bin = radians(coverage["joint_bin_deg"]);
    p.goal_seconds = coverage["goal_lifetime_s"].as<double>();
}
} // namespace
AppOptions parse_options(int argc, char **argv) {
    AppOptions o;
    o.profile = fs::path(NATIVE_SOURCE).parent_path() / "configuration_files/profiles/jetson";
    o.settings = fs::path(NATIVE_SOURCE) / "config/pid_collection.yaml";
    o.excitation.waveform = mlp::Waveform::Chirp;
    o.excitation.amplitude = .35;
    Args args(argc, argv);
    while (args.next()) {
        if (args.is("--out"))
            o.out = args.value();
        else if (args.is("--profile"))
            o.profile = args.value();
        else if (args.is("--settings"))
            o.settings = args.value();
        else if (args.is("--serial"))
            o.serial = args.value();
        else if (args.is("--gamepad"))
            o.gamepad = args.value();
        else if (args.is("--suffix"))
            o.suffix = args.value();
        else if (args.is("--seed")) {
            o.planning.seed = natural(args);
            o.fixed_seed = true;
        } else if (args.is("--dry"))
            o.dry = true;
        else if (args.is("--scripted"))
            o.scripted = true;
        else if (args.is("--record-on-start"))
            o.record_on_start = true;
        else if (args.is("--duration-s"))
            o.duration = finite(args);
        else if (args.is("--record-seconds"))
            o.record_seconds = finite(args);
        else if (args.is("--queue-records"))
            o.queue = natural(args);
        else if (args.is("--read-chunk-bytes"))
            o.read_chunk = natural(args);
        else if (args.is("--excitation-amplitude"))
            o.excitation.amplitude = finite(args);
        else if (args.is("--chirp-seconds"))
            o.excitation.sweep_s = finite(args);
        else if (args.is("--excitation")) {
            const auto v = args.value();
            require(v == "sine" || v == "chirp", "PID overlay supports sine or chirp");
            o.excitation.waveform = v == "sine" ? mlp::Waveform::Sine : mlp::Waveform::Chirp;
        } else if (args.is("--excitation-target")) {
            o.excitation.target = mlp::target_mask(args.value());
            require(!(o.excitation.target & ~14U), "PID overlay targets arm joints only");
        } else if (args.is("--help")) {
            std::cout << usage;
            std::exit(0);
        } else
            args.unknown();
    }
    require(!(o.out.empty() || o.scripted != o.gamepad.empty() || o.duration < 0 ||
              o.duration > 86400 || o.record_seconds <= 0 || o.record_seconds > 86400 ||
              o.queue < 16 || o.queue > 65536 || (o.queue & (o.queue - 1)) || o.read_chunk == 0 ||
              o.read_chunk > 8192),
            "invalid PID input/output, duration or queue options");
    require(o.suffix.size() <= 100, "suffix too long");
    for (char &c : o.suffix)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '-' || c == '_'))
            c = '_';
    if (!o.fixed_seed)
        o.planning.seed = (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    o.excitation.seed = uint32_t(o.planning.seed ^ (o.planning.seed >> 32));
    settings(o);
    return o;
}
Runtime load_runtime(const AppOptions &o) {
    Runtime r;
    r.robot = collection::load_robot(o.profile);
    require(r.robot.hardware.frequency == 100, "PID collection requires 100 Hz PWM");
    // Construct/validate every option before opening serial, PWM or output files.
    collection::RandomCollection validate(r.robot, o.planning);
    collection::JointGuard validate_guard(r.robot, o.guard);
    r.limits = collection::excitation_limits_deg(r.robot);
    mlp::Excitation validate_excitation(o.excitation, r.limits);
    const auto settings_yaml = read(o.settings);
    auto &m = r.metadata;
    m.config = r.robot.hardware;
    m.config.control_hz = 100;
    m.dry = o.dry;
    m.suffix = o.suffix;
    m.collector = "kaivuri-pid-collect";
    m.source_revision = NATIVE_GIT_REVISION;
    m.executable_sha256 = sha256_file("/proc/self/exe");
    m.collection_id = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count()) +
                      "-pid-" + std::to_string(getpid());
    std::ostringstream options;
    options << std::setprecision(17)
            << "{\"profile\":" << json_quote(fs::absolute(o.profile).string())
            << ",\"settings\":" << json_quote(fs::absolute(o.settings).string())
            << ",\"serial\":" << json_quote(o.serial.string())
            << ",\"gamepad\":" << json_quote(o.gamepad)
            << ",\"scripted\":" << (o.scripted ? "true" : "false")
            << ",\"seed\":" << o.planning.seed << ",\"duration_s\":" << o.duration
            << ",\"record_seconds\":" << o.record_seconds
            << ",\"read_chunk_bytes\":" << o.read_chunk
            << ",\"settings_yaml\":" << json_quote(settings_yaml) << '}';
    m.options_json = options.str();
    r.protocol_json =
        "{\"version\":1,\"controller\":\"random-pitch-velocity-ik-pid-v1\",\"seed\":" +
        std::to_string(o.planning.seed) +
        ",\"rng_scope\":\"process; pauses reset control state without rewinding "
        "draws\",\"output_hz\":100,\"joint_order\":[\"boom\",\"arm\",\"bucket\"],"
        "\"collision_checked\":false,\"settings_yaml\":" +
        json_quote(settings_yaml) + ",\"segments\":[";
    for (const auto &name : collection::profile_files(o.profile))
        m.files.emplace_back("profile/" + name, read(o.profile / name));
    m.files.emplace_back("pid_collection.yaml", settings_yaml);
    const fs::path source = NATIVE_SOURCE;
    for (const auto &[directory, prefix] :
         {std::pair{source / "src", "source/"},
          std::pair{source / "include/kaivuri", "include/kaivuri/"}})
        for (const auto &entry : fs::directory_iterator(directory))
            if (entry.path().extension() == ".cpp" || entry.path().extension() == ".hpp")
                m.files.emplace_back(prefix + entry.path().filename().string(), read(entry.path()));
    m.files.emplace_back("CMakeLists.txt", read(source / "CMakeLists.txt"));
    return r;
}
} // namespace kaivuri::pidcollect
