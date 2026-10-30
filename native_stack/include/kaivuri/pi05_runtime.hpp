// OpenPI pi0.5 TensorRT runtime (TensorRT/CUDA stay inside pi05_runtime.cpp). Used by
// kaivuri-inference and, through the C API in pi05_capi.cpp, by tools/vla_pi05.py.
#pragma once
#include "kaivuri/pi05_policy.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace kaivuri::pi05 {

struct Timing {
    double vision_ms = 0, language_ms = 0, denoise_ms = 0, total_ms = 0;
};

// Full normalized chunk vs the exporter's FP16 split reference. pi05_libero (AOT kernel in
// the exported order) keeps 0.99999 / 0.005. pi05_openpi_state on native TensorRT attention
// defaults to 0.9998 / 0.01: the exporter's own compact-vs-padded FP16 paths already differ
// by 0.99994 / 0.004, and GB10 TRT 10.14 measured 0.99990 / 0.0070 worst on smoke9.
struct ParityGate {
    double cosine = .99999, max_abs = .005;
    bool overridden = false;
};

// One 224x224 model image, [3][224][224] FP16 bit patterns.
using Image = std::vector<uint16_t>;

class Runtime {
  public:
    Runtime(const std::filesystem::path &bundle, const std::filesystem::path &cache,
            bool cuda_graph);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    const YAML::Node &bundle_info() const;
    const std::string &policy_name() const;
    bool native_cache() const;
    int prefix_size() const;
    int noise_count() const;
    int action_horizon() const;
    int action_dim() const;
    int denoising_steps() const;
    // pi05_libero: the bundle's fixed token table for a task.
    std::vector<int64_t> task_tokens(const std::string &task) const;
    // Uploads prompt rows; the first call also primes and captures the CUDA graph.
    void set_prompt(const std::vector<int64_t> &tokens);
    // Two preprocessed images and [horizon][action_dim] noise -> the normalized chunk.
    std::vector<float> infer(const void *image0, const void *image1, const float *noise,
                             Timing &timing);
    // Training-time RTC (Mode C; bundle kaivuri.rtc.row_time, action templates with per-row
    // modulation [1, horizon, hidden]). The first `prefix_rows` rows are the committed actions:
    // `prefix` is [prefix_rows][prefix_dims] normalized model units (the robot dims), padded
    // with zeros to action_dim, exactly as training sees a clean row. Those rows start clean,
    // get the t = 0 style in every step and are reset to the prefix after every Euler step, so
    // they come back bit-exact. prefix_rows = 0 is the plain sampler on the same graph.
    std::vector<float> infer(const void *image0, const void *image1, const float *noise,
                             Timing &timing, int prefix_rows, const float *prefix,
                             int prefix_dims);
    // Mode C bundle (per-row action templates), and the most prefix rows it was trained for.
    bool row_time() const;
    int max_prefix() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// pi05_libero: OpenPI's antialiased linear resize. pi05_openpi_state: Pillow BILINEAR
// (openpi_client ResizeImages). Both pad to 224x224 and map to [-1, 1]; gray is replicated.
Image preprocess_libero(const uint8_t *input, int width, int height, int stride, int channels);
Image preprocess_openpi(const uint8_t *input, int width, int height, int stride, int channels);
// OpenPI Unnormalize of a normalized [horizon][width] chunk: the first robot_dim columns.
std::vector<double> robot_chunk(const std::vector<float> &normalized, int horizon, int width,
                                const Contract &contract);
// norm_stats.json, checked against the bundle's MANIFEST.sha256.
YAML::Node bundle_norm_stats(const std::filesystem::path &bundle);
// Every prepared fixture (prepare_pi05.py) through `model`; throws on the first failure,
// otherwise returns the per-fixture JSON report.
std::string validate(Runtime &model, const std::filesystem::path &bundle,
                     const std::filesystem::path &prepared, const Contract *contract,
                     const ParityGate &gate);

} // namespace kaivuri::pi05
