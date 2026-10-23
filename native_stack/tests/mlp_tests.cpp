#include "kaivuri/core.hpp"
#include "kaivuri/mlp_excitation.hpp"
#include "kaivuri/mlp_record.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <vector>
#include <yaml-cpp/yaml.h>
using namespace kaivuri;
using namespace kaivuri::mlp;
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
std::vector<uint8_t> frame(uint32_t timestamp) {
    std::vector<uint8_t> b(170, 0);
    b[0] = 0xaa;
    b[1] = 0x55;
    b[2] = 4;
    b[3] = 4;
    std::memcpy(b.data() + 4, &timestamp, 4);
    const float one = 1;
    for (unsigned i = 0; i < 4; ++i)
        std::memcpy(b.data() + 8 + i * 40, &one, 4);
    uint16_t sum = 0;
    for (size_t i = 2; i < b.size() - 2; ++i)
        sum += b[i];
    std::memcpy(b.data() + b.size() - 2, &sum, 2);
    return b;
}
void reference_fixtures(const char *path) {
    const auto fixture = YAML::LoadFile(path);
    const std::array<std::array<double, 2>, 4> limits{
        {{-180, 180}, {-50, 30}, {28, 144}, {-120, 20}}};
    for (const auto &test : fixture["cases"]) {
        const auto mode = test["mode"].as<std::string>();
        ExcitationOptions o;
        o.seed = 42;
        o.waveform = mode == "sine"     ? Waveform::Sine
                     : mode == "stroke" ? Waveform::Stroke
                                        : Waveform::Chirp;
        o.amplitude = test["amplitude"].as<double>();
        o.start_hz = test["start_hz"].as<double>();
        o.end_hz = test["end_hz"].as<double>();
        o.sweep_s = test["sweep_s"].as<double>();
        Excitation generator(o, limits);
        generator.enable(0);
        auto parameters = generator.block().parameters;
        const auto &p = test["parameters"];
        for (auto &v : parameters) {
            v.env = p["f_env"].as<double>();
            v.carrier = p["f_car"].as<double>();
            v.rate = p["f_rate"].as<double>();
            v.depth = p["depth"].as<double>();
            v.cutoff = p["f_noise"].as<double>();
            v.noise = 0;
            v.amplitude = o.amplitude;
            v.phi1 = p["phi1"].as<double>();
            v.phi2 = p["phi2"].as<double>();
        }
        generator.restore_parameters(parameters);
        if (mode == "stroke")
            for (unsigned i = 0; i < 4; ++i)
                check(generator.block().initial_direction[i] == test["directions"][i].as<double>(),
                      "reference stroke direction changed");
        for (const auto &row : test["samples"]) {
            std::array<double, 4> q{};
            for (unsigned i = 0; i < 4; ++i)
                q[i] = row[2][i].as<double>() * pi / 180;
            const auto u = generator.sample(row[0].as<double>(), q, row[1].as<bool>());
            for (unsigned i = 0; i < 4; ++i)
                check(std::abs(u[i] - row[3][i].as<double>()) < 1e-10,
                      "frozen Python waveform mismatch");
            check(generator.enabled() == row[4].as<bool>(),
                  "frozen Python stroke disable mismatch");
        }
    }
}
void default_frequency_coverage() {
    const std::array<std::array<double, 2>, 4> limits{
        {{-180, 180}, {-50, 30}, {28, 144}, {-120, 20}}};
    unsigned slow = 0, fast = 0;
    double maximum = 0;
    for (uint32_t seed = 0; seed < 64; ++seed) {
        ExcitationOptions options;
        options.seed = seed;
        Excitation generator(options, limits);
        for (const auto &p : generator.block().parameters) {
            check(p.carrier >= .2 && p.carrier <= 10, "sine carrier range");
            check(p.carrier * (1 + 2 * pi * p.rate * p.depth) <= 12 + 1e-9,
                  "sine instantaneous frequency limit");
            slow += p.carrier < 1;
            fast += p.carrier >= 5;
            maximum = std::max(maximum, p.carrier);
        }
    }
    check(slow > 38 && fast > 25 && maximum > 8, "default sine misses slow or fast motion");
    ExcitationOptions options;
    options.waveform = Waveform::Chirp;
    Excitation chirp(options, limits);
    const double h = 1e-4;
    for (double t : {59., 61.}) {
        const double frequency = (chirp.phase(t + h) - chirp.phase(t - h)) / (4 * pi * h);
        check(frequency > 10 && frequency < 12, "default chirp misses fast motion");
    }
}
void queue_overflow() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("mlp-queue-" + std::to_string(getpid()));
    fs::remove_all(root);
    RecordingMetadata metadata;
    metadata.dry = true;
    for (unsigned i = 0; i < 4; ++i) {
        metadata.config.joints[i].sensor = i;
        metadata.config.joints[i].parent_sensor = int(i) - 1;
        metadata.config.joints[i].pitch_delta = true;
        metadata.config.offsets_inv[i] = {1, 0, 0, 0};
    }
    // A large startup snapshot keeps the consumer busy while the tiny queue fills.
    metadata.files.emplace_back("test", std::string(100000, 'x'));
    MlpWriter writer(root, metadata, 16);
    check(writer.start(), "writer did not start");
    Record raw;
    raw.kind = RecordKind::Raw;
    raw.sample.imu.valid = true;
    raw.sample.imu.sequence = 1;
    bool overflow = false;
    for (unsigned i = 0; i < 10000; ++i) {
        raw.sample.imu.sequence = i + 1;
        if (!writer.push(raw)) {
            overflow = true;
            break;
        }
    }
    check(overflow && writer.dropped() > 0, "tiny recorder queue did not report overflow");
    writer.finish(false);
    writer.shutdown();
    bool partial = false;
    for (const auto &entry : fs::directory_iterator(root)) {
        check(entry.path().extension() == ".partial", "overflow committed an output");
        if (entry.path().filename().string().find("session_") == 0) {
            const auto manifest = YAML::LoadFile(entry.path().string());
            check(!manifest["complete"].as<bool>() && manifest["dropped"].as<unsigned>() > 0,
                  "overflow lost its incomplete marker");
            partial = true;
        }
    }
    check(partial, "overflow left no incomplete manifest");
    fs::remove_all(root);
}
void queued_restart() {
    namespace fs = std::filesystem;
    struct ClosingBarrier : RecordingAttachment {
        std::atomic<bool> entered{false}, release{false};
        void begin(const fs::path &, const std::string &) override {}
        void drive(const DriveSample &) override {}
        void close() override {
            entered.store(true, std::memory_order_release);
            while (!release.load(std::memory_order_acquire))
                std::this_thread::yield();
        }
        void abort() noexcept override { release.store(true, std::memory_order_release); }
        std::vector<fs::path> paths() const override { return {}; }
    };
    const auto root = fs::temp_directory_path() / ("mlp-restart-" + std::to_string(getpid()));
    fs::remove_all(root);
    RecordingMetadata metadata;
    metadata.dry = true;
    for (unsigned i = 0; i < 4; ++i) {
        metadata.config.joints[i].sensor = i;
        metadata.config.joints[i].parent_sensor = int(i) - 1;
        metadata.config.joints[i].pitch_delta = true;
        metadata.config.offsets_inv[i] = {1, 0, 0, 0};
    }
    auto barrier = std::make_shared<ClosingBarrier>();
    MlpWriter writer(root, metadata, 16, barrier);
    auto samples = [&](uint64_t tick) {
        Record raw;
        raw.kind = RecordKind::Raw;
        raw.sample.imu.valid = true;
        raw.sample.imu.sequence = tick + 1;
        raw.sample.imu.device_us = (tick + 1) * 10000;
        check(writer.push(raw), "restart raw queue failed");
        Record drive;
        drive.kind = RecordKind::Drive;
        drive.sample = raw.sample;
        drive.sample.tick = tick;
        drive.sample.start_ns = drive.sample.post_write_ns = 1000000000 + tick * 10000000;
        check(writer.push(drive), "restart drive queue failed");
    };
    check(writer.start(), "first restart take failed");
    samples(0);
    writer.finish();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!barrier->entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const bool entered = barrier->entered.load(std::memory_order_acquire);
    const bool accepted = entered && writer.start();
    // Release even when start was rejected, so a regression reports failure
    // instead of hanging in the writer destructor.
    barrier->release.store(true, std::memory_order_release);
    check(entered && accepted, "start during previous close was lost");
    samples(1);
    writer.finish();
    writer.shutdown();
    unsigned sessions = 0;
    for (const auto &entry : fs::directory_iterator(root))
        if (entry.path().filename().string().find("session_") == 0) {
            check(entry.path().extension() == ".json", "queued restart left an incomplete take");
            const auto m = YAML::LoadFile(entry.path().string());
            check(m["complete"].as<bool>() && m["rows"].as<unsigned>() == 1,
                  "queued take lost rows or completion");
            ++sessions;
        }
    check(sessions == 2 && writer.state() == WriterState::Idle,
          "queued restart did not commit both takes");
    fs::remove_all(root);
}
int main(int argc, char **argv) {
    try {
        check(argc == 2, "missing waveform reference fixture");
        reference_fixtures(argv[1]);
        default_frequency_coverage();
        queue_overflow();
        queued_restart();
        ImuParser parser(4);
        std::vector<ImuSample> samples;
        parser.set_sample_sink(
            [](void *p, const ImuSample &s) {
                static_cast<std::vector<ImuSample> *>(p)->push_back(s);
            },
            &samples);
        auto a = frame(0xfffffff0U), b = frame(0x20U);
        a.insert(a.end(), b.begin(), b.end());
        parser.feed(a.data(), a.size(), 1);
        check(samples.size() == 2 && samples[1].device_us == 0x100000020ULL,
              "sink lost burst or wrap");
        check(samples[0].raw_device_us == 0xfffffff0U && samples[1].packet_type == 4,
              "packet identity lost");
        parser.feed(b.data(), b.size(), 1.01);
        check(samples.size() == 2 && parser.resets == 1, "invalid duplicate reached sink");
        std::array<std::array<double, 2>, 4> limits{
            {{-180, 180}, {-50, 30}, {28, 144}, {-120, 20}}};
        ExcitationOptions options;
        options.waveform = Waveform::Chirp;
        options.target = 2;
        options.amplitude = .35;
        Excitation chirp(options, limits);
        chirp.enable(10);
        auto parameters = chirp.block().parameters;
        for (auto &p : parameters)
            p.phi2 = .3;
        chirp.restore_parameters(parameters);
        for (double t : {0., .2, 1., 30., 59., 60., 61., 90., 119., 120., 125., 130.}) {
            auto u = chirp.sample(10 + t, {}, true);
            check(u[0] == 0 && u[2] == 0 && u[3] == 0 && std::abs(u[1]) <= .35,
                  "chirp target or bounds");
            if (t >= 120 && t < 130)
                check(u[1] == 0, "chirp rest interval");
            if (t < 120) {
                auto ramp = [](double x) { return .5 - .5 * std::cos(pi * std::clamp(x, 0., 1.)); };
                double expected = .35 * ramp(t) * ramp(120 - t) * std::sin(chirp.phase(t) + .3);
                check(std::abs(u[1] - expected) < 1e-12, "analytic Python chirp formula mismatch");
            }
        }
        ExcitationOptions sine_options;
        sine_options.seed = 71;
        Excitation s1(sine_options, limits), s2(sine_options, limits);
        s1.enable(0);
        s2.enable(100);
        for (unsigned i = 0; i <= 200; ++i)
            s1.sample(i * .01, {}, true);
        auto x = s1.sample(2, {}, true), y = s2.sample(102, {}, true);
        for (unsigned i = 0; i < 4; ++i)
            check(std::abs(x[i] - y[i]) < 1e-12, "noise depends on polling history");
        options.waveform = Waveform::Stroke;
        options.start_hz = .5;
        options.end_hz = 12;
        options.sweep_s = 20;
        Excitation stroke(options, limits);
        stroke.enable(0);
        const std::array<double, 4> q{0, -.3, 1.2, -.7};
        stroke.sample(0, q, true);
        auto u = stroke.sample(.5, q, true);
        check(std::abs(u[1]) <= 1, "stroke bounds");
        u = stroke.sample(.51, q, false);
        check(u == Commands{}, "stale stroke not zero");
        auto outside = q;
        outside[1] = 2;
        stroke.sample(1, outside, true);
        check(!stroke.enabled(), "out-of-range stroke not disabled");
        std::cout << "MLP parser emission and excitation checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
