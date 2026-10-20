#include "kaivuri/ik_collection.hpp"
#include "kaivuri/mlp_excitation.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

using namespace kaivuri;
using namespace kaivuri::collection;
namespace fs = std::filesystem;
namespace {
constexpr std::array<const char *, 3> joints{"boom", "arm", "bucket"};
constexpr uint64_t tick_ns = 10000000, block_ticks = 2000;
double number(const std::string &text) {
    size_t end = 0;
    const double value = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(value)) throw std::invalid_argument("expected a finite number: " + text);
    return value;
}
uint64_t integer(const std::string &text) {
    size_t end = 0;
    if (text.empty() || text[0] == '-') throw std::invalid_argument("expected a nonnegative integer");
    const auto value = std::stoull(text, &end);
    if (end != text.size()) throw std::invalid_argument("expected an integer: " + text);
    return value;
}
template <class T> YAML::Node sequence(const T &values) {
    YAML::Node n(YAML::NodeType::Sequence);
    for (const auto &value : values) n.push_back(value);
    return n;
}
YAML::Node gains(const std::array<Gains, 3> &g) {
    YAML::Node n(YAML::NodeType::Sequence);
    for (const auto &v : g) {
        YAML::Node joint;
        joint["kp"] = v.kp; joint["ki"] = v.ki; joint["kd"] = v.kd; joint["derivative_tau_s"] = v.tau;
        n.push_back(joint);
    }
    return n;
}
void write_yaml(const fs::path &file, const YAML::Node &node) {
    YAML::Emitter emitter;
    emitter.SetDoublePrecision(17); emitter << node;
    if (!emitter.good()) throw std::runtime_error("cannot encode " + file.string());
    std::ofstream stream(file);
    stream << emitter.c_str() << '\n';
    if (!stream) throw std::runtime_error("cannot write " + file.string());
}
struct Csv {
    std::ofstream stream;
    explicit Csv(const fs::path &file) : stream(file) {
        if (!stream) throw std::runtime_error("cannot open preview CSV");
        stream << std::setprecision(17);
        stream << "timestamp,tick,segment_id,segment_kind,excitation_block,excitation_mode,excitation_stage,valid,reason,coverage_cell";
        for (const auto *prefix : {"q_rad", "q_ref_rad", "task", "task_ref", "assist", "excitation_requested",
                                   "command", "excitation_effective", "pid_rate_requested_hz", "pid_updated",
                                   "pid_dt_s", "kp_applied", "ki_applied", "kd_applied", "clipped"})
            if (std::string(prefix) == "task" || std::string(prefix) == "task_ref") {
                for (const auto *name : {"x_m", "z_m", "pitch_rad"}) stream << ',' << prefix << '_' << name;
            } else {
                for (const auto *name : joints) stream << ',' << prefix << '_' << name;
            }
        stream << ",ik_position_error_m,ik_pitch_error_rad,ik_iterations,step_work_us\n";
    }
    template <class T> void values(const T &a) { for (const auto &v : a) stream << ',' << v; }
    void row(uint64_t tick, const Angles &q, const Output &o, const Segment &s, uint64_t block,
             const char *mode, const char *stage, double work_us) {
        stream << double(tick) / 100 << ',' << tick << ',' << s.id << ',' << kind_name(s.kind) << ','
               << block << ',' << mode << ',' << stage << ',' << o.valid << ',' << o.reason << ',' << o.cell;
        values(q); values(o.joint_reference); values(o.measured.pose); values(o.reference);
        values(o.assist); values(o.excitation); values(o.command);
        Angles effective{};
        for (unsigned j = 0; j < 3; ++j) effective[j] = o.command[j] - o.assist[j];
        values(effective); values(s.pid_rates); values(o.pid_updated); values(o.pid_dt);
        for (unsigned term = 0; term < 3; ++term)
            for (const auto &g : o.applied_gains) stream << ',' << (term == 0 ? g.kp : term == 1 ? g.ki : g.kd);
        values(o.clipped);
        stream << ',' << o.ik.position_error << ',' << o.ik.pitch_error << ',' << o.ik.iterations << ',' << work_us << '\n';
    }
};
YAML::Node segment_record(const Segment &s) {
    YAML::Node n;
    n["id"] = s.id; n["begin_ns"] = s.begin_ns; n["end_ns_requested"] = s.end_ns;
    n["duration_requested_s"] = s.duration; n["kind"] = kind_name(s.kind);
    n["velocity_xz_m_s"] = sequence(s.velocity); n["pitch_target_rad"] = s.pitch;
    n["goal_joint_cell"] = s.goal_cell; n["goal_q_rad"] = sequence(s.goal_q);
    n["pid_rate_requested_hz"] = sequence(s.pid_rates); n["pid_gains_requested"] = gains(s.gains);
    return n;
}
YAML::Node block_record(uint64_t index, const mlp::Excitation *e) {
    YAML::Node n;
    n["index"] = index; n["begin_ns"] = index * block_ticks * tick_ns;
    n["end_ns_requested"] = (index + 1) * block_ticks * tick_ns;
    n["enabled"] = e != nullptr;
    if (!e) { n["mode"] = "clean"; return n; }
    const auto &b = e->block();
    n["mode"] = mlp::waveform_name(b.options.waveform);
    n["seed"] = b.options.seed; n["noise_seed"] = b.noise_seed; n["target_mask"] = b.options.target;
    n["amplitude"] = b.options.amplitude; n["start_hz"] = b.options.start_hz;
    n["end_hz"] = b.options.end_hz; n["sweep_s_each_direction"] = b.options.sweep_s;
    for (unsigned j = 0; j < 4; ++j) {
        YAML::Node p;
        const auto &v = b.parameters[j];
        p["env_hz"] = v.env; p["carrier_hz"] = v.carrier; p["fm_rate_hz"] = v.rate;
        p["fm_depth_s"] = v.depth; p["noise_cutoff_hz"] = v.cutoff; p["noise_gain"] = v.noise;
        p["amplitude"] = v.amplitude; p["env_phase_rad"] = v.phi1; p["carrier_phase_rad"] = v.phi2;
        n["parameters_slew_boom_arm_bucket"].push_back(p);
    }
    return n;
}
} // namespace

int main(int argc, char **argv) {
    try {
        Options o;
        fs::path out, profile = fs::path(NATIVE_SOURCE) / "../configuration_files/profiles/jetson_bucket";
        double seconds = 120;
        bool excite = true;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Hardware-free IK/PID synthetic preview; never connects to robot devices.\n"
                    "Usage: kaivuri-ik-preview --out NEW_DIR [--profile PROFILE_DIR] [--seconds 120] [--seed 42]\n"
                    "  --min-seconds 0.1 --max-seconds 3 --pid-rates 25,50,100\n"
                    "  --gain-min 0.5 --gain-max 1.5 --min-speed-mm-s 5 --max-speed-mm-s 20\n"
                    "  --no-excitation   Disable alternating clean/sine/chirp preview blocks.\n"
                    "Joint PID rates vary independently. Output samples stay at 100 Hz.\n"
                    "This is a synthetic plant, with no collision or physical tracking acceptance.\n";
                return 0;
            }
            if (arg == "--no-excitation") { excite = false; continue; }
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--out") out = value;
            else if (arg == "--profile") profile = value;
            else if (arg == "--seed") o.seed = integer(value);
            else if (arg == "--seconds") seconds = number(value);
            else if (arg == "--min-seconds") o.min_seconds = number(value);
            else if (arg == "--max-seconds") o.max_seconds = number(value);
            else if (arg == "--gain-min") o.gain_min = number(value);
            else if (arg == "--gain-max") o.gain_max = number(value);
            else if (arg == "--min-speed-mm-s") o.min_speed = number(value) / 1000;
            else if (arg == "--max-speed-mm-s") o.max_speed = number(value) / 1000;
            else if (arg == "--pid-rates") {
                o.pid_rates.clear();
                std::istringstream rates(value);
                std::string rate;
                while (std::getline(rates, rate, ',')) o.pid_rates.push_back(number(rate));
                if (value.empty() || value.back() == ',') throw std::invalid_argument("empty PID rate");
            } else throw std::invalid_argument("unknown option: " + arg);
        }
        if (out.empty()) throw std::invalid_argument("--out NEW_DIR is required");
        if (seconds <= 0 || seconds > 86400) throw std::invalid_argument("--seconds must be in (0, 86400]");
        const auto robot = load_robot(profile);
        RandomCollection controller(robot, o);
        if (!out.parent_path().empty()) fs::create_directories(out.parent_path());
        if (!fs::create_directory(out)) throw std::invalid_argument("output directory already exists");
        fs::create_directory(out / "profile");
        const auto p = YAML::LoadFile((profile / "profile.yaml").string());
        for (const std::string &file : {std::string("profile.yaml"), p["control_config_file"].as<std::string>(),
                                      p["servo_config_file"].as<std::string>()})
            fs::copy_file(profile / file, out / "profile" / file);

        YAML::Node protocol, segments(YAML::NodeType::Sequence), blocks(YAML::NodeType::Sequence);
        protocol["format_version"] = 1; protocol["source_revision"] = NATIVE_GIT_REVISION;
        protocol["source_note"] = "Revision can include worktree changes; retain frozen source manifest and executable SHA-256.";
        protocol["synthetic_only"] = true; protocol["collision_checked"] = false;
        protocol["profile"] = fs::canonical(profile).string(); protocol["seed"] = o.seed;
        protocol["frame"] = "slew-pivot-relative x/z, cutting-lip pitch, radians";
        protocol["joint_order"] = sequence(joints); protocol["output_hz"] = 100;
        protocol["requested_seconds"] = seconds;
        protocol["segment_seconds_min_max"] = sequence(std::array<double, 2>{o.min_seconds, o.max_seconds});
        protocol["duration_sampling"] = "log uniform, quantized to 10ms; coverage segments use upper third";
        protocol["kind_probabilities"]["coverage"] = .4; protocol["kind_probabilities"]["free"] = .35;
        protocol["kind_probabilities"]["reverse"] = .1; protocol["kind_probabilities"]["pitch"] = .1;
        protocol["kind_probabilities"]["hold"] = .05;
        protocol["speed_m_s_min_max"] = sequence(std::array<double, 2>{o.min_speed, o.max_speed});
        protocol["gain_scale_min_max"] = sequence(std::array<double, 2>{o.gain_min, o.gain_max});
        protocol["gain_sampling"] = "independent per joint and P/I/D term; zero base terms stay zero";
        protocol["pid_rate_choices_hz"] = sequence(o.pid_rates); protocol["base_gains"] = gains(robot.gains);
        protocol["pitch_change_max_rad"] = o.max_pitch_change; protocol["pitch_rate_rad_s"] = o.pitch_rate;
        protocol["reference_lead_max_m"] = o.max_lead; protocol["reference_pitch_lead_max_rad"] = .25;
        protocol["joint_margin_rad"] = o.joint_margin; protocol["base_valve_limit"] = o.valve_limit;
        protocol["coverage_bin_rad"] = o.coverage_bin; protocol["coverage_goal_lifetime_s"] = o.goal_seconds;
        protocol["maximum_output_gap_s"] = o.max_gap; protocol["excitation_generator_version"] = mlp::excitation_version;
        protocol["excitation_block_s"] = 20; protocol["excitation_enabled"] = excite;
        protocol["synthetic_plant"]["velocity_lag_s"] = .15;
        protocol["synthetic_plant"]["command_deadband"] = .05;
        protocol["synthetic_plant"]["max_joint_speed_rad_s"] = 1.;
        protocol["synthetic_plant"]["position_clamp"] = "profile limits with configured joint margin";
        protocol["synthetic_plant"]["interpretation"] = "controller exercise only; not a hydraulic model or motion-clearance proof";

        Csv csv(out / "preview.csv");
        Angles q{}, velocity{};
        for (unsigned j = 0; j < 3; ++j) q[j] = (robot.limits[j][0] + robot.limits[j][1]) / 2;
        std::array<std::array<double, 2>, 4> limits{};
        limits[0] = {-180, 180};
        for (unsigned j = 0; j < 3; ++j)
            for (unsigned side = 0; side < 2; ++side) limits[j + 1][side] = robot.limits[j][side] * 180 / pi;
        std::unique_ptr<mlp::Excitation> excitation;
        uint64_t block = 0, valid = 0, clipped = 0, boundaries = 0;
        std::array<uint64_t, 3> updates{};
        std::map<std::string, uint64_t> invalid;
        double maximum_work_us = 0, total_work_us = 0;
        const uint64_t ticks = uint64_t(std::ceil(seconds * 100 - 1e-9));
        for (uint64_t tick = 0; tick < ticks; ++tick) {
            if (tick % block_ticks == 0) {
                block = tick / block_ticks;
                excitation.reset();
                if (excite && block % 2) {
                    mlp::ExcitationOptions e;
                    e.seed = uint32_t(o.seed + block * 0x9e3779b9ULL);
                    e.waveform = (block / 2) % 2 ? mlp::Waveform::Chirp : mlp::Waveform::Sine;
                    constexpr unsigned targets[]{2, 4, 8, 6, 10, 12, 14};
                    e.target = targets[(block / 2 + o.seed % 7) % 7];
                    e.amplitude = .2 + .2 * (double(e.seed % 1001) / 1000);
                    e.start_hz = .05; e.end_hz = 12; e.sweep_s = 5;
                    excitation = std::make_unique<mlp::Excitation>(e, limits);
                    excitation->enable(double(tick) / 100);
                }
                blocks.push_back(block_record(block, excitation.get()));
            }
            Angles overlay{};
            if (excitation) {
                const auto u = excitation->sample(double(tick) / 100, {0, q[0], q[1], q[2]}, true);
                for (unsigned j = 0; j < 3; ++j) overlay[j] = u[j + 1];
            }
            const auto begin = std::chrono::steady_clock::now();
            const auto output = controller.step(tick * tick_ns, q, overlay);
            const double work_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
            maximum_work_us = std::max(maximum_work_us, work_us); total_work_us += work_us;
            if (output.new_segment) segments.push_back(segment_record(controller.segment()));
            if (output.valid) ++valid; else ++invalid[output.reason];
            for (unsigned j = 0; j < 3; ++j) {
                updates[j] += output.pid_updated[j]; clipped += output.clipped[j];
            }
            csv.row(tick, q, output, controller.segment(), block,
                    excitation ? mlp::waveform_name(excitation->block().options.waveform) : "clean",
                    excitation ? mlp::stage_name(excitation->stage()) : "off", work_us);
            for (unsigned j = 0; j < 3; ++j) {
                const double u = output.command[j];
                const double requested_v = std::copysign(std::max(0., std::abs(u) - .05) / .95, u);
                velocity[j] += .01 / (.15 + .01) * (requested_v - velocity[j]);
                const double next = q[j] + .01 * velocity[j];
                q[j] = std::clamp(next, robot.limits[j][0] + o.joint_margin + 1e-9,
                                  robot.limits[j][1] - o.joint_margin - 1e-9);
                if (next != q[j]) { velocity[j] = 0; ++boundaries; }
            }
        }
        csv.stream.close();
        if (!csv.stream) throw std::runtime_error("preview CSV write failed");
        protocol["segments"] = segments; protocol["excitation_blocks"] = blocks;
        write_yaml(out / "protocol.yaml", protocol);
        std::ofstream coverage(out / "coverage.csv");
        coverage << std::setprecision(17) << "cell,dwell_s,moving_dwell_s\n";
        for (size_t i = 0; i < controller.dwell().size(); ++i)
            if (controller.dwell()[i] > 0) coverage << i << ',' << controller.dwell()[i] << ',' << controller.moving_dwell()[i] << '\n';
        coverage.close();
        if (!coverage) throw std::runtime_error("coverage write failed");
        YAML::Node summary;
        summary["synthetic_only"] = true; summary["collision_checked"] = false;
        summary["rows"] = ticks; summary["valid_rows"] = valid; summary["segments"] = segments.size();
        summary["visited_joint_cells"] = controller.visited_cells(); summary["joint_cells_total"] = controller.dwell().size();
        summary["measured_dwell_s"] = std::accumulate(controller.dwell().begin(), controller.dwell().end(), 0.);
        summary["measured_moving_dwell_s"] = std::accumulate(controller.moving_dwell().begin(), controller.moving_dwell().end(), 0.);
        summary["pid_updates"] = sequence(updates); summary["clipped_joint_rows"] = clipped;
        summary["synthetic_joint_boundary_contacts"] = boundaries;
        summary["controller_work_us_mean"] = total_work_us / ticks; summary["controller_work_us_max"] = maximum_work_us;
        for (const auto &[reason, count] : invalid) summary["invalid_reasons"][reason] = count;
        write_yaml(out / "summary.yaml", summary);
        std::cout << "Synthetic IK/PID preview: " << ticks << " rows, " << segments.size() << " segments, "
                  << controller.visited_cells() << " measured joint cells; " << out << '\n';
        std::cout << "No robot devices accessed. Collision and powered tracking acceptance remain open.\n";
    } catch (const std::exception &e) { std::cerr << "ik-preview: " << e.what() << '\n'; return 1; }
}
