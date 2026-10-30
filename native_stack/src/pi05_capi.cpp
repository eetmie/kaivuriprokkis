// C API over the pi0.5 runtime for tools/vla_pi05.py (ctypes): one fine-tuned
// pi05_openpi_state bundle per handle. Every call reports failure as a nonzero return (or
// NULL) with the message in `err`; no exception crosses the boundary.
#include "kaivuri/pi05_runtime.hpp"
#include "kaivuri/util.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>

namespace kp = kaivuri::pi05;
using kaivuri::require;

namespace {
struct Handle {
    std::unique_ptr<kp::Runtime> model;
    kp::Contract contract;
};
void put(char *buffer, size_t size, const std::string &text) {
    if (!buffer || !size)
        return;
    const size_t n = std::min(size - 1, text.size());
    std::memcpy(buffer, text.data(), n);
    buffer[n] = 0;
}
template <class F> int guarded(char *err, size_t err_len, F &&f) {
    try {
        f();
        return 0;
    } catch (const std::exception &e) {
        put(err, err_len, e.what());
    } catch (...) {
        put(err, err_len, "unknown error");
    }
    return 1;
}
std::string quoted_list(const std::vector<std::string> &items) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < items.size(); ++i)
        out << (i ? "," : "") << kaivuri::json_quote(items[i]);
    out << ']';
    return out.str();
}
} // namespace

extern "C" {

// Loads the engines, then runs every prepared fixture through them (the parity gate). The
// per-fixture report goes to `report`. cosine/max_abs <= 0 keep the defaults (0.9998 / 0.01).
void *kvr_pi05_open(const char *bundle, const char *cache, const char *fixtures, double cosine,
                    double max_abs, char *report, size_t report_len, char *err, size_t err_len) {
    auto h = std::make_unique<Handle>();
    const int failed = guarded(err, err_len, [&] {
        h->model = std::make_unique<kp::Runtime>(bundle, cache, true);
        require(h->model->policy_name() == "pi05_openpi_state",
                "the live runner takes fine-tuned pi05_openpi_state bundles");
        h->contract = kp::load_contract(h->model->bundle_info(), kp::bundle_norm_stats(bundle));
        require(h->contract.prompt.longest(h->contract.state_dim) + 512 <=
                    size_t(h->model->prefix_size()),
                "longest state prompt exceeds the compact prefix");
        kp::ParityGate gate{.9998, .01, false};
        if (cosine > 0)
            gate.cosine = cosine, gate.overridden = true;
        if (max_abs > 0)
            gate.max_abs = max_abs, gate.overridden = true;
        put(report, report_len, kp::validate(*h->model, bundle, fixtures, &h->contract, gate));
    });
    return failed ? nullptr : h.release();
}

void kvr_pi05_close(void *handle) { delete static_cast<Handle *>(handle); }

// JSON: horizon, action_dim, robot_dim, state_dim, fps, state_joints, tasks, cameras.
int kvr_pi05_info(void *handle, char *json, size_t json_len, char *err, size_t err_len) {
    return guarded(err, err_len, [&] {
        auto &h = *static_cast<Handle *>(handle);
        const auto &c = h.contract;
        std::vector<std::string> tasks;
        for (auto &head : c.prompt.heads)
            tasks.push_back(head.first);
        std::vector<std::string> cameras;
        for (auto cam : h.model->bundle_info()["kaivuri"]["cameras"])
            cameras.push_back(cam["camera"].as<std::string>());
        std::ostringstream out;
        out << "{\"horizon\":" << h.model->action_horizon()
            << ",\"action_dim\":" << h.model->action_dim() << ",\"robot_dim\":" << c.robot_dim
            << ",\"state_dim\":" << c.state_dim << ",\"fps\":" << c.fps << ",\"state_joints\":[";
        for (size_t i = 0; i < c.state_joints.size(); ++i)
            out << (i ? "," : "") << c.state_joints[i];
        out << "],\"state_names\":" << quoted_list(c.state_names)
            << ",\"tasks\":" << quoted_list(tasks) << ",\"cameras\":" << quoted_list(cameras)
            << ",\"native_cache\":" << (h.model->native_cache() ? "true" : "false") << '}';
        const auto text = out.str();
        require(text.size() < json_len, "info buffer too small");
        put(json, json_len, text);
    });
}

// One step: angles_rad [slew, boom, arm, bucket] -> float32 degrees -> state prompt for `task`;
// two Y8 frames (Pillow BILINEAR preprocessing) and noise [horizon][action_dim] -> the robot
// chunk [horizon][robot_dim] (OpenPI unnormalized, controller order). degrees_out gets the
// state the prompt used; timings_ms: prompt, preprocess, vision, language, denoise, total.
int kvr_pi05_infer(void *handle, const char *task, const double *angles_rad, const uint8_t *y8_0,
                   const uint8_t *y8_1, int width, int height, const float *noise,
                   double *robot_out, float *degrees_out, double *timings_ms, int *prompt_tokens,
                   char *err, size_t err_len) {
    return guarded(err, err_len, [&] {
        auto &h = *static_cast<Handle *>(handle);
        const auto &c = h.contract;
        require(c.prompt.heads.count(task), std::string("task has no token head: ") + task);
        const uint64_t start = kaivuri::monotonic_ns();
        auto degrees = kp::state_degrees({angles_rad, angles_rad + 4}, c.state_joints);
        auto tokens = c.tokens(task, degrees);
        h.model->set_prompt(tokens);
        const uint64_t prompted = kaivuri::monotonic_ns();
        auto image0 = kp::preprocess_openpi(y8_0, width, height, width, 1);
        auto image1 = kp::preprocess_openpi(y8_1, width, height, width, 1);
        const uint64_t prepared = kaivuri::monotonic_ns();
        kp::Timing t;
        auto normalized = h.model->infer(image0.data(), image1.data(), noise, t);
        for (float v : normalized)
            require(std::isfinite(v), "non-finite model output");
        auto robot = kp::robot_chunk(normalized, h.model->action_horizon(),
                                     h.model->action_dim(), c);
        std::copy(robot.begin(), robot.end(), robot_out);
        std::copy(degrees.begin(), degrees.end(), degrees_out);
        *prompt_tokens = int(tokens.size());
        const double total = (kaivuri::monotonic_ns() - start) / 1e6;
        const double values[6] = {(prompted - start) / 1e6, (prepared - prompted) / 1e6,
                                  t.vision_ms, t.language_ms, t.denoise_ms, total};
        std::copy(values, values + 6, timings_ms);
    });
}

} // extern "C"
