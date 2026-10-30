// kaivuri-inference: bench, fixture validation and the standalone C++ drive loop around
// the pi0.5 runtime (pi05_runtime.cpp).
// Without --live, model actions are recorded only. With --live (pi05_openpi_state only),
// chunks go to kaivuri-control as drive plans through pi05::Producer; no I2C path here.
#include "kaivuri/frame_store.hpp"
#include "kaivuri/pi05_runtime.hpp"
#include "kaivuri/util.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <yaml-cpp/yaml.h>
namespace fs = std::filesystem;
namespace kp = kaivuri::pi05;
using kaivuri::monotonic_ns;
using kaivuri::require;
using kaivuri::stop_requested;

namespace {
// The node's value, or the fallback when the key is absent.
template <class T> T value_or(const YAML::Node &node, const T &fallback) {
    return node ? node.as<T>() : fallback;
}
// Values separated by sep, printed with the given stream precision.
template <class T> std::string join(const std::vector<T> &values, char sep, int precision = 6) {
    std::ostringstream out;
    out << std::setprecision(precision);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            out << sep;
        out << values[i];
    }
    return out.str();
}
struct FrameReader {
    void *handle = nullptr;
    std::vector<uint8_t> pixels = std::vector<uint8_t>(640 * 480);
    explicit FrameReader(const fs::path &path) {
        handle = kaivuri_frame_open(path.c_str());
        require(handle, "cannot open camera frame: " + path.string());
    }
    ~FrameReader() {
        if (handle)
            kaivuri_frame_close(handle);
    }
    FrameMetadata read() {
        FrameMetadata metadata{};
        uint64_t deadline = monotonic_ns() + 1000000000ULL;
        while (!stop_requested && monotonic_ns() < deadline) {
            if (kaivuri_frame_read(handle, &metadata, pixels.data(), pixels.size()) == 1) {
                auto now = monotonic_ns();
                if (metadata.host_monotonic_ns && metadata.host_monotonic_ns <= now &&
                    now - metadata.host_monotonic_ns <= 200000000ULL)
                    return metadata;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error("camera source missing/stale for one second");
    }
};
// Training camera order: frame files are <serial>-ir<index>.frame (kaivuri-camera). Rotation
// is applied by kaivuri-camera --rotate-180 and cannot be read back here. Returns the IR
// index each slot's frames must carry.
std::vector<uint32_t> bundle_cameras(const YAML::Node &info, const std::vector<fs::path> &frames) {
    auto cameras = info["kaivuri"]["cameras"];
    require(cameras && cameras.size() == 2, "bundle lacks kaivuri.cameras");
    std::vector<uint32_t> ir_indices;
    for (size_t k = 0; k < 2; ++k) {
        auto spec = cameras[k]["camera"].as<std::string>();
        auto colon = spec.find(':');
        require(colon != std::string::npos, "camera spec SERIAL:IR_INDEX");
        auto serial = spec.substr(0, colon), index = spec.substr(colon + 1);
        require(frames[k].stem().string() == serial + "-ir" + index,
                "camera " + std::to_string(k) + " must be " + spec + " (" + serial + "-ir" + index +
                    ".frame) in the bundle's slot order");
        ir_indices.push_back(std::stoul(index));
        const int rotation = value_or(cameras[k]["rotation_deg"], 0);
        if (rotation != 0)
            std::cerr << "[cameras] " << spec << " was trained rotated " << rotation
                      << " deg: kaivuri-camera must run with --rotate-180 " << serial << '\n';
    }
    return ir_indices;
}
double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    double pos = p * (values.size() - 1);
    size_t lo = size_t(pos), hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}
constexpr const char *usage =
    "kaivuri-inference --bundle OPENPI_BUNDLE --cache VALIDATED_ENGINE_CACHE\n"
    "  --fixtures PREPARED_FIXTURES [--out NEW_REPORT.json] [--validate-only]\n"
    "  [--camera-frame FIRST.frame --camera-frame SECOND.frame]\n"
    "  [--task EXPORTED_TASK --duration-s 60 --no-graph]\n"
    "  [--rate-hz 0] (0 = unrestricted; wait before reading fresh frames)\n"
    "  pi05_openpi_state: [--control-socket PATH --max-state-age-ms 50]\n"
    "    [--parity-cosine 0.9998 --parity-max-abs 0.01]\n"
    "  Driving (pi05_openpi_state only):\n"
    "    --live [--enable-slew] [--enable-tracks]\n"
    "    [--action-tolerance inf --renew-ms 100 --max-observation-age-ms 1400 "
    "--action-offset-ms 0]\n"
    "--live sends plans with live PWM and pump permission; without it actions never drive\n"
    "hardware. Without --out nothing is written (report NEW.json + timing NEW.csv).\n";
constexpr const char *csv_header =
    "sample,start_ns,finish_ns,frame0_sequence,frame1_sequence,frame0_arrival_ns,"
    "frame1_arrival_ns,frame0_sensor_us,frame1_sensor_us,frame0_exposure_us,frame1_exposure_us,"
    "frame0_valid_bits,frame1_valid_bits,preprocess_ms,vision_ms,language_ms,denoise_ms,"
    "inference_ms,iteration_ms,observation_age_start_ms,observation_to_result_ms,"
    "camera_arrival_skew_ms,rate_wait_ms,rate_lateness_ms,state_imu_ns,state_age_ms,state_deg,"
    "state_bins,prompt_tokens,prompt_ms,plan_t0_ns,plan_observation_ns,plan_sent,plan_clamped\n";
} // namespace

int main(int argc, char **argv) {
    try {
        fs::path bundle, cache, fixtures, out;
        std::vector<fs::path> frames;
        std::string task = "pick up the red block";
        std::string control_socket = (kaivuri::runtime_dir() / "control.sock").string();
        double duration = 60, rate_hz = 0, max_state_age_ms = 50, action_offset_ms = 0;
        // One inference on live inputs before timing starts; --live arms only after it, from
        // its observation-to-plan time. Validation has already warmed the engines.
        constexpr int warmup = 1;
        bool only = false, graph = true, drive = false;
        kp::ProducerOptions producer_options;
        auto &supervisor = producer_options.supervisor;
        kp::ParityGate gate;
        kaivuri::Args args(argc, argv);
        while (args.next()) {
            if (args.is("--bundle"))
                bundle = args.value();
            else if (args.is("--cache"))
                cache = args.value();
            else if (args.is("--fixtures"))
                fixtures = args.value();
            else if (args.is("--out"))
                out = args.value();
            else if (args.is("--camera-frame"))
                frames.push_back(args.value());
            else if (args.is("--task"))
                task = args.value();
            else if (args.is("--duration-s"))
                duration = args.number();
            else if (args.is("--rate-hz"))
                rate_hz = args.number();
            else if (args.is("--validate-only"))
                only = true;
            else if (args.is("--no-graph"))
                graph = false;
            else if (args.is("--control-socket"))
                control_socket = args.value();
            else if (args.is("--max-state-age-ms"))
                max_state_age_ms = args.number();
            else if (args.is("--parity-cosine"))
                gate.cosine = args.number(), gate.overridden = true;
            else if (args.is("--parity-max-abs"))
                gate.max_abs = args.number(), gate.overridden = true;
            else if (args.is("--live"))
                drive = producer_options.allow_live = producer_options.pump = true;
            else if (args.is("--enable-slew"))
                producer_options.gate.slew = true;
            else if (args.is("--enable-tracks"))
                producer_options.gate.tracks = true;
            else if (args.is("--action-tolerance"))
                producer_options.gate.tolerance = args.number();
            else if (args.is("--renew-ms"))
                supervisor.renew_ns = uint64_t(args.number() * 1e6);
            else if (args.is("--max-observation-age-ms"))
                supervisor.max_observation_age_ns = uint64_t(args.number() * 1e6);
            else if (args.is("--action-offset-ms"))
                action_offset_ms = args.number();
            else if (args.is("--help")) {
                std::cout << usage;
                return 0;
            } else
                args.unknown();
        }
        require(!bundle.empty() && !cache.empty() && !fixtures.empty(),
                "bundle, cache and fixtures required");
        const bool logging = !out.empty();
        require(duration > 0 && std::isfinite(duration) && (only || frames.size() == 2),
                "positive duration and two camera frames required");
        require(std::isfinite(rate_hz) && rate_hz >= 0 && rate_hz <= 500 &&
                    (rate_hz == 0 || rate_hz >= .001),
                "rate must be 0 (unrestricted) or 0.001..500 Hz");
        require(max_state_age_ms > 0 && max_state_age_ms <= 200, "state age limit 0..200 ms");
        require(std::abs(action_offset_ms) <= 100, "action offset must be within +-100 ms");
        require(supervisor.renew_ns >= 10000000ULL && supervisor.renew_ns <= 250000000ULL &&
                    supervisor.max_observation_age_ns >= 200000000ULL &&
                    supervisor.max_observation_age_ns <= 1450000000ULL,
                "renewal 10..250 ms (lease 500 ms); observation age 200..1450 ms (daemon 1500)");
        require(!logging || !fs::exists(out), "report already exists; choose a new path");
        kaivuri::install_stop_handlers();
        uint64_t load_start = monotonic_ns();
        kp::Runtime model(bundle, cache, graph);
        double load_ms = (monotonic_ns() - load_start) / 1e6;
        std::unique_ptr<kp::Contract> contract;
        if (model.policy_name() == "pi05_openpi_state") {
            contract = std::make_unique<kp::Contract>(
                kp::load_contract(model.bundle_info(), kp::bundle_norm_stats(bundle)));
            require(contract->prompt.longest(contract->state_dim) + 512 <=
                        size_t(model.prefix_size()),
                    "longest state prompt exceeds the compact prefix");
        }
        require(contract || !gate.overridden, "parity overrides apply to pi05_openpi_state only");
        if (contract && !gate.overridden)
            gate = {.9998, .01, false};
        require(!drive || contract, "--live requires a pi05_openpi_state bundle");
        require(!drive || contract->fps > 0, "--live requires kaivuri.fps in bundle.json");
        std::string parity = kp::validate(model, bundle, fixtures, contract.get(), gate);
        // Without --out the streams stay closed and every write to them is a no-op.
        std::ofstream summary, csv;
        if (logging) {
            if (!out.parent_path().empty())
                fs::create_directories(out.parent_path());
            summary.open(out);
            require(bool(summary), "cannot create report");
        }
        summary << std::setprecision(12)
                << "{\"model\":" << kaivuri::json_quote(model.policy_name())
                << ",\"runtime\":\"C++ TensorRT " << (model.native_cache() ? "native" : "+ AOT")
                << "\",\"actuator_output\":" << (drive ? "true" : "false")
                << ",\"cuda_graph\":" << (graph ? "true" : "false") << ",\"load_ms\":" << load_ms
                << ",\"action_horizon\":" << model.action_horizon()
                << ",\"action_dim\":" << model.action_dim()
                << ",\"denoising_steps\":" << model.denoising_steps() << ",\"parity\":" << parity
                << ",\"parity_gate\":{\"cosine\":" << gate.cosine << ",\"max_abs\":" << gate.max_abs
                << '}';
        if (only) {
            summary << ",\"status\":\"PASS\"}\n";
            std::cout << "PASS validate-only\n";
            return 0;
        }
        std::vector<uint32_t> ir_indices;
        if (contract) {
            require(contract->prompt.heads.count(task), "task has no token head in this bundle");
            ir_indices = bundle_cameras(model.bundle_info(), frames);
        } else
            model.set_prompt(model.task_tokens(task));
        FrameReader first(frames[0]), second(frames[1]);
        // Status-only connection for live state; the producer owns a separate one.
        std::unique_ptr<kp::ControlLink> state_link;
        if (contract)
            state_link = std::make_unique<kp::ControlLink>(control_socket);
        std::unique_ptr<kp::Producer> producer;
        if (drive) {
            producer_options.socket = control_socket;
            producer = std::make_unique<kp::Producer>(producer_options);
        }
        std::mt19937 random(42);
        std::normal_distribution<float> gaussian(0, 1);
        std::vector<float> noise(model.noise_count());
        if (logging) {
            fs::path csv_path = out;
            csv_path.replace_extension(".csv");
            csv.open(csv_path);
            require(bool(csv), "cannot create timing CSV");
            csv << csv_header;
        }
        std::vector<double> latencies, delays, prompt_times, warmup_delays, waits;
        int count = 0;
        uint64_t begin = 0;
        const uint64_t period_ns = rate_hz ? static_cast<uint64_t>(std::llround(1e9 / rate_hz)) : 0;
        uint64_t next_start = 0, rate_overruns = 0;
        std::vector<float> last_action;
        std::vector<double> last_robot;
        auto running = [&] {
            return !stop_requested && (!begin || (monotonic_ns() - begin) / 1e9 < duration);
        };
        while (running()) {
            const uint64_t wait_begin = monotonic_ns();
            if (next_start && wait_begin > next_start)
                ++rate_overruns;
            while (!stop_requested && next_start && monotonic_ns() < next_start) {
                // Bounded absolute sleeps remain responsive to shutdown, even at low rates.
                const uint64_t wake_ns =
                    std::min<uint64_t>(next_start, monotonic_ns() + 50000000ULL);
                timespec wake{static_cast<time_t>(wake_ns / 1000000000ULL),
                              static_cast<long>(wake_ns % 1000000000ULL)};
                const int result = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr);
                require(result == 0 || result == EINTR, "inference pacing sleep failed");
            }
            if (!running())
                break;
            // Arm after warm-up, then hold centered until the first chunk.
            if (producer && count == warmup) {
                // Each chunk must outlive the gap until the next one lands: one start period
                // (the rate, or back-to-back) plus the slowest warm-up observation-to-plan time.
                const double worst = *std::max_element(warmup_delays.begin(), warmup_delays.end());
                const double gap = (period_ns ? period_ns / 1e6 : worst) + worst;
                const double plan_ms = (model.action_horizon() - 1) * 1000. / contract->fps;
                const double age_ms = supervisor.max_observation_age_ns / 1e6;
                std::cerr << "[drive] next chunk after <= " << gap << " ms; plan covers " << plan_ms
                          << " ms; observation limit " << age_ms << " ms\n";
                require(gap + 100 < plan_ms && gap + 50 < age_ms,
                        "chunk coverage too short for this latency/rate; refusing to arm");
                producer->arm();
            }
            if (producer && count > warmup && !producer->running())
                throw std::runtime_error("producer stopped: " + producer->stop_reason());
            uint64_t start = monotonic_ns();
            const double rate_wait_ms = (start - wait_begin) / 1e6;
            const double rate_lateness_ms =
                next_start && start > next_start ? (start - next_start) / 1e6 : 0;
            // An overrun starts one fresh job now; no queued observations or catch-up burst.
            next_start = count >= warmup && period_ns ? start + period_ns : 0;
            auto a = first.read(), b = second.read();
            require(a.width == 640 && a.height == 480 && b.width == 640 && b.height == 480 &&
                        a.stride == a.width && b.stride == b.width && (a.valid_metadata & 3) == 3 &&
                        (b.valid_metadata & 3) == 3,
                    "640x480 Y8 with timestamp metadata required");
            require(ir_indices.empty() ||
                        (a.ir_index == ir_indices[0] && b.ir_index == ir_indices[1]),
                    "camera IR index differs from the bundle's");
            uint64_t oldest = std::min(a.host_monotonic_ns, b.host_monotonic_ns);
            kp::RobotState robot;
            std::vector<int> bins;
            size_t prompt_tokens = 0;
            double prompt_ms = 0;
            if (contract) {
                // State sampled right after the frames; never older than max_state_age_ms.
                robot = kp::robot_state(state_link->status(), contract->state_joints,
                                        monotonic_ns(), uint64_t(max_state_age_ms * 1e6));
                oldest = std::min(oldest, robot.imu_ns);
                const uint64_t prompt_start = monotonic_ns();
                bins = contract->bins(robot.degrees);
                auto tokens = contract->prompt.compose(task, bins);
                prompt_tokens = tokens.size();
                model.set_prompt(tokens);
                prompt_ms = (monotonic_ns() - prompt_start) / 1e6;
            }
            uint64_t observed = monotonic_ns();
            require(observed >= oldest && observed - oldest <= 200000000ULL,
                    "oldest observation exceeds 200ms freshness limit");
            auto prep = contract ? kp::preprocess_openpi : kp::preprocess_libero;
            auto image0 = prep(first.pixels.data(), a.width, a.height, a.stride, 1);
            auto image1 = prep(second.pixels.data(), b.width, b.height, b.stride, 1);
            double preprocess_ms = (monotonic_ns() - observed) / 1e6;
            for (auto &value : noise)
                value = gaussian(random);
            kp::Timing timing;
            last_action = model.infer(image0.data(), image1.data(), noise.data(), timing);
            for (auto value : last_action)
                require(std::isfinite(value), "nonfinite model action");
            kp::Plan plan;
            int clamped = 0;
            bool sent = false;
            if (contract) {
                const int h = model.action_horizon(), r = contract->robot_dim;
                last_robot = kp::robot_chunk(last_action, h, model.action_dim(), *contract);
                // Row 0 belongs to the observation instant; the controller starts at the
                // row matching its own clock, which skips the inference delay.
                const double fps = contract->fps > 0 ? contract->fps : 30;
                const uint64_t t0 = uint64_t(int64_t(oldest) + int64_t(action_offset_ms * 1e6));
                try {
                    plan = kp::make_plan(last_robot, h, r, fps, t0, oldest, producer_options.gate,
                                         clamped);
                } catch (const std::runtime_error &) {
                    if (producer)
                        throw;
                    clamped = -1; // Record-only: report an out-of-range chunk, keep running.
                }
                if (producer && clamped > 0) {
                    double worst = 0;
                    for (auto v : last_robot)
                        worst = std::max(worst, std::abs(v));
                    std::cerr << "[drive] warning: sample " << count << " clamped " << clamped
                              << " actions to [-1,1] (max |v| " << worst << ")\n";
                }
                if (producer && count >= warmup) {
                    std::string reason;
                    require(producer->post(plan, reason), "drive plan refused: " + reason);
                    sent = true;
                }
            }
            uint64_t finish = monotonic_ns();
            if (count < warmup)
                warmup_delays.push_back((finish - oldest) / 1e6);
            if (count++ < warmup)
                continue;
            if (!begin) {
                begin = start;
                std::cerr << "[bench] READY\n";
            }
            double iteration = (finish - start) / 1e6, delay = (finish - oldest) / 1e6;
            latencies.push_back(timing.total_ms);
            delays.push_back(delay);
            waits.push_back(rate_wait_ms);
            prompt_times.push_back(prompt_ms);
            double skew = std::abs(double(a.host_monotonic_ns) - double(b.host_monotonic_ns)) / 1e6;
            csv << std::setprecision(12) << latencies.size() << ',' << start << ',' << finish << ','
                << a.sequence << ',' << b.sequence << ',' << a.host_monotonic_ns << ','
                << b.host_monotonic_ns << ',' << a.sensor_timestamp_us << ','
                << b.sensor_timestamp_us << ',' << a.actual_exposure_us << ','
                << b.actual_exposure_us << ',' << a.valid_metadata << ',' << b.valid_metadata << ','
                << preprocess_ms << ',' << timing.vision_ms << ',' << timing.language_ms << ','
                << timing.denoise_ms << ',' << timing.total_ms << ',' << iteration << ','
                << (observed - oldest) / 1e6 << ',' << delay << ',' << skew << ',' << rate_wait_ms
                << ',' << rate_lateness_ms << ',' << robot.imu_ns << ','
                << (robot.imu_ns ? (observed - std::min(observed, robot.imu_ns)) / 1e6 : 0) << ','
                << join(robot.degrees, ' ', 9) << ',' << join(bins, ' ') << ',' << prompt_tokens
                << ',' << prompt_ms << ',' << plan.t0_ns << ',' << plan.observation_ns << ','
                << sent << ',' << clamped << '\n';
            require(!logging || bool(csv), "timing write failed");
        }
        std::string producer_json;
        if (producer) {
            producer->stop(stop_requested ? "signal" : "duration complete");
            auto c = producer->counters();
            std::ostringstream p;
            p << std::setprecision(12) << ",\"producer\":{\"plans\":" << c.plans
              << ",\"renewals\":" << c.renewals << ",\"holds\":" << c.holds
              << ",\"max_send_gap_ms\":" << c.max_send_gap_ms
              << ",\"stop_reason\":" << kaivuri::json_quote(producer->stop_reason()) << '}';
            producer_json = p.str();
            producer.reset();
        }
        require(!latencies.empty(), "no inference samples");
        summary << ",\"task\":" << kaivuri::json_quote(task) << ",\"views\":2,\"input\":"
                << (contract ? "\"IR Y8 replicated to RGB; joint-angle state prompt\""
                             : "\"IR Y8 replicated to RGB; unfinetuned LIBERO bench\"")
                << ",\"samples\":" << latencies.size()
                << ",\"elapsed_s\":" << (monotonic_ns() - begin) / 1e9 << ",\"rate_hz\":" << rate_hz
                << ",\"rate_overruns\":" << rate_overruns
                << ",\"rate_wait_p50_ms\":" << percentile(waits, .5)
                << ",\"inference_p50_ms\":" << percentile(latencies, .5)
                << ",\"inference_p95_ms\":" << percentile(latencies, .95)
                << ",\"inference_p99_ms\":" << percentile(latencies, .99)
                << ",\"inference_max_ms\":" << percentile(latencies, 1)
                << ",\"prompt_p50_ms\":" << percentile(prompt_times, .5)
                << ",\"prompt_max_ms\":" << percentile(prompt_times, 1)
                << ",\"arrival_to_result_p50_ms\":" << percentile(delays, .5)
                << ",\"arrival_to_result_p99_ms\":" << percentile(delays, .99) << producer_json
                << ",\"last_normalized_chunk\":[" << join(last_action, ',', 12) << ']';
        if (contract)
            summary << ",\"last_robot_chunk\":[" << join(last_robot, ',', 12) << ']';
        summary << ",\"status\":\"PASS\"}\n";
        require(!logging || bool(summary), "summary write failed");
        std::cout << "PASS samples=" << latencies.size() << " p50_ms=" << percentile(latencies, .5)
                  << " p99_ms=" << percentile(latencies, .99);
        if (logging)
            std::cout << " report=" << out;
        std::cout << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "kaivuri-inference: " << e.what() << '\n';
        return 1;
    }
}
