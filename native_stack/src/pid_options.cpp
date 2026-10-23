#include "kaivuri/pid_collect.hpp"
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

namespace kaivuri::pidcollect {
namespace fs = std::filesystem;
namespace {
std::string read(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}
void settings(AppOptions &o) {
    const auto y = YAML::LoadFile(o.settings.string());
    if (y["version"].as<int>() != 1)
        throw std::invalid_argument("unsupported PID settings version");
    const auto command = y["command"], pid = y["pid"], limits = y["limits"],
               coverage = y["coverage"];
    const auto duration = command["duration_s"].as<std::array<double, 2>>();
    const auto speed = command["tip_speed_m_s"].as<std::array<double, 2>>();
    const auto gain = pid["gain_multiplier"].as<std::array<double, 2>>();
    o.planning.min_seconds = duration[0];
    o.planning.max_seconds = duration[1];
    o.planning.min_speed = speed[0];
    o.planning.max_speed = speed[1];
    o.planning.max_pitch_change = command["pitch_change_max_deg"].as<double>() * pi / 180;
    o.planning.pitch_rate = command["pitch_rate_rad_s"].as<double>();
    o.planning.max_lead = command["reference_lead_m"].as<double>();
    o.planning.pid_rates = pid["update_hz"].as<std::vector<double>>();
    o.planning.gain_min = gain[0];
    o.planning.gain_max = gain[1];
    o.planning.valve_limit = pid["base_output_limit"].as<double>();
    o.planning.joint_margin = limits["joint_margin_deg"].as<double>() * pi / 180;
    o.guard.margin = o.planning.joint_margin;
    o.guard.taper = limits["taper_deg"].as<double>() * pi / 180;
    o.guard.velocity_lookahead = limits["velocity_lookahead_s"].as<double>();
    o.planning.recovery_limit = limits["recovery_output_limit"].as<double>(.25);
    o.planning.recovery_inset = limits["recovery_inset_deg"].as<double>(1.) * pi / 180;
    o.guard.command_sign = limits["command_sign"].as<collection::Angles>();
    o.planning.command_sign = o.guard.command_sign;
    o.planning.coverage_bin = coverage["joint_bin_deg"].as<double>() * pi / 180;
    o.planning.goal_seconds = coverage["goal_lifetime_s"].as<double>();
}
} // namespace
AppOptions parse_options(int argc, char **argv) {
    AppOptions o;
    o.profile = fs::path(NATIVE_SOURCE).parent_path() / "configuration_files/profiles/jetson";
    o.settings = fs::path(NATIVE_SOURCE) / "config/pid_collection.yaml";
    o.excitation.waveform = mlp::Waveform::Chirp;
    o.excitation.amplitude = .35;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() {
            if (++i == argc)
                throw std::invalid_argument("missing value for " + a);
            return std::string(argv[i]);
        };
        auto real = [&]() {
            const auto text = value();
            size_t end = 0;
            const double v = std::stod(text, &end);
            if (end != text.size() || !std::isfinite(v))
                throw std::invalid_argument("invalid number for " + a);
            return v;
        };
        auto integer = [&]() {
            const auto text = value();
            size_t end = 0;
            if (text.empty() || text[0] == '-')
                throw std::invalid_argument("invalid integer for " + a);
            const auto v = std::stoull(text, &end);
            if (end != text.size())
                throw std::invalid_argument("invalid integer for " + a);
            return v;
        };
        if (a == "--out")
            o.out = value();
        else if (a == "--profile")
            o.profile = value();
        else if (a == "--settings")
            o.settings = value();
        else if (a == "--serial")
            o.serial = value();
        else if (a == "--gamepad")
            o.gamepad = value();
        else if (a == "--suffix")
            o.suffix = value();
        else if (a == "--seed") {
            o.planning.seed = integer();
            o.fixed_seed = true;
        } else if (a == "--dry")
            o.dry = true;
        else if (a == "--scripted")
            o.scripted = true;
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
        else if (a == "--excitation-amplitude")
            o.excitation.amplitude = real();
        else if (a == "--chirp-seconds")
            o.excitation.sweep_s = real();
        else if (a == "--excitation") {
            const auto v = value();
            if (v != "sine" && v != "chirp")
                throw std::invalid_argument("PID overlay supports sine or chirp");
            o.excitation.waveform = v == "sine" ? mlp::Waveform::Sine : mlp::Waveform::Chirp;
        } else if (a == "--excitation-target") {
            o.excitation.target = mlp::target_mask(value());
            if (o.excitation.target & ~14U)
                throw std::invalid_argument("PID overlay targets arm joints only");
        } else if (a == "--help") {
            std::cout
                << "kaivuri-pid-collect --out DIR --gamepad /dev/input/eventN [--dry]\n"
                   "  --profile DIR --settings YAML --serial DEVICE --suffix LABEL --seed UINT64\n"
                   "  --excitation sine|chirp --excitation-target all|lift|tilt|scoop|PAIR\n"
                   "  --excitation-amplitude 0.35 --chirp-seconds 60 --record-seconds 600\n"
                   "  --scripted --record-on-start --duration-s S --queue-records 4096 "
                   "--read-chunk-bytes 512\n"
                   "Independent PID collector; PWM is LIVE by default. --dry suppresses I2C.\n"
                   "A=record/save Start=PID on/off B=overlay X=pump; sticks cancel PID and "
                   "overlay.\n"
                   "Scripted input: start, save, pump, assist, excitation, manual BOOM ARM BUCKET, "
                   "quit.\n"
                   "Joint targets/commands are bounded; obstacle/full-reach routing is not "
                   "implemented.\n";
            std::exit(0);
        } else
            throw std::invalid_argument("unknown PID collector option: " + a);
    }
    if (o.out.empty() || (!o.scripted && o.gamepad.empty()) || (o.scripted && !o.gamepad.empty()) ||
        o.duration < 0 || o.duration > 86400 || o.record_seconds <= 0 || o.record_seconds > 86400 ||
        o.queue < 16 || o.queue > 65536 || (o.queue & (o.queue - 1)) || o.read_chunk == 0 ||
        o.read_chunk > 8192)
        throw std::invalid_argument("invalid PID input/output, duration or queue options");
    if (o.suffix.size() > 100)
        throw std::invalid_argument("suffix too long");
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
    if (r.robot.hardware.frequency != 100)
        throw std::invalid_argument("PID collection requires 100 Hz PWM");
    // Construct/validate every option before opening serial, PWM or output files.
    collection::RandomCollection validate(r.robot, o.planning);
    collection::JointGuard validate_guard(r.robot, o.guard);
    r.limits[0] = {-180, 180};
    for (unsigned j = 0; j < 3; ++j)
        for (unsigned b = 0; b < 2; ++b)
            r.limits[j + 1][b] = r.robot.limits[j][b] * 180 / pi;
    mlp::Excitation validate_excitation(o.excitation, r.limits);
    auto &m = r.metadata;
    m.config = r.robot.hardware;
    m.config.control_hz = 100;
    m.dry = o.dry;
    m.suffix = o.suffix;
    m.collector = "kaivuri-pid-collect";
    m.source_revision = NATIVE_GIT_REVISION;
    m.executable_sha256 = mlp::sha256_file("/proc/self/exe");
    m.collection_id = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count()) +
                      "-pid-" + std::to_string(getpid());
    std::ostringstream options;
    options << std::setprecision(17)
            << "{\"profile\":" << mlp::json_quote(fs::absolute(o.profile).string())
            << ",\"settings\":" << mlp::json_quote(fs::absolute(o.settings).string())
            << ",\"serial\":" << mlp::json_quote(o.serial.string())
            << ",\"gamepad\":" << mlp::json_quote(o.gamepad)
            << ",\"scripted\":" << (o.scripted ? "true" : "false")
            << ",\"seed\":" << o.planning.seed << ",\"duration_s\":" << o.duration
            << ",\"record_seconds\":" << o.record_seconds
            << ",\"read_chunk_bytes\":" << o.read_chunk
            << ",\"settings_yaml\":" << mlp::json_quote(read(o.settings)) << '}';
    m.options_json = options.str();
    std::ostringstream protocol;
    protocol << "{\"version\":1,\"controller\":\"random-pitch-velocity-ik-pid-v1\",\"seed\":"
             << o.planning.seed
             << ",\"rng_scope\":\"process; pauses reset control state without rewinding "
                "draws\",\"output_hz\":100,"
                "\"joint_order\":[\"boom\",\"arm\",\"bucket\"],\"collision_checked\":false,"
                "\"settings_yaml\":"
             << mlp::json_quote(read(o.settings)) << ",\"segments\":[";
    r.protocol_json = protocol.str();
    const auto profile = YAML::LoadFile((o.profile / "profile.yaml").string());
    for (const auto &name :
         {std::string("profile.yaml"), profile["control_config_file"].as<std::string>(),
          profile["servo_config_file"].as<std::string>()})
        m.files.emplace_back("profile/" + name, read(o.profile / name));
    m.files.emplace_back("pid_collection.yaml", read(o.settings));
    for (const auto &directory :
         {fs::path(NATIVE_SOURCE) / "src", fs::path(NATIVE_SOURCE) / "include/kaivuri"})
        for (const auto &entry : fs::directory_iterator(directory))
            if (entry.path().extension() == ".cpp" || entry.path().extension() == ".hpp")
                m.files.emplace_back(
                    (directory.filename() == "src" ? "source/" : "include/kaivuri/") +
                        entry.path().filename().string(),
                    read(entry.path()));
    m.files.emplace_back("CMakeLists.txt", read(fs::path(NATIVE_SOURCE) / "CMakeLists.txt"));
    return r;
}
} // namespace kaivuri::pidcollect
