// OpenPI pi0.5 TensorRT runtime: the pi05_libero bench bundle and the fine-tuned
// pi05_openpi_state excavator bundle (joint angles in a per-step prompt). The device chain
// follows the validated jetson-orin-nano-vla/bench/vendor/pi05_trt.py implementation.
// Used by kaivuri-inference and, through pi05_capi.cpp, by tools/vla_pi05.py.
#include "kaivuri/pi05_runtime.hpp"
#include "kaivuri/util.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace fs = std::filesystem;
namespace kp = kaivuri::pi05;
using namespace nvinfer1;
using kaivuri::monotonic_ns;
using kaivuri::require;

namespace {
void ck(cudaError_t code, const char *operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
std::vector<char> read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    require(bool(in), "cannot open " + path.string());
    auto size = in.tellg();
    require(size > 0, "empty file " + path.string());
    std::vector<char> bytes(static_cast<size_t>(size));
    in.seekg(0);
    in.read(bytes.data(), size);
    require(bool(in), "short read " + path.string());
    return bytes;
}
// The node's value, or the fallback when the key is absent.
template <class T> T value_or(const YAML::Node &node, const T &fallback) {
    return node ? node.as<T>() : fallback;
}
struct Npy {
    std::shared_ptr<void> mapping;
    size_t file_bytes = 0, bytes = 0, count = 1;
    const uint8_t *data = nullptr;
    std::string dtype;
    std::vector<int64_t> shape;
    explicit Npy(const fs::path &path, const std::string &expected_sha = "") {
        kaivuri::Fd fd(open(path.c_str(), O_RDONLY | O_CLOEXEC));
        require(bool(fd), "cannot open array " + path.string());
        struct stat st{};
        require(fstat(fd.get(), &st) == 0 && st.st_size >= 10, "invalid array " + path.string());
        file_bytes = size_t(st.st_size);
        void *address = mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd.get(), 0);
        require(address != MAP_FAILED, "cannot map " + path.string());
        mapping = std::shared_ptr<void>(address, [size = file_bytes](void *p) { munmap(p, size); });
        auto p = static_cast<const uint8_t *>(address);
        require(std::memcmp(p, "\x93NUMPY", 6) == 0, "not NPY: " + path.string());
        size_t offset = p[6] == 1 ? 10 : 12;
        require(p[6] >= 1 && p[6] <= 3 && file_bytes >= offset, "NPY version");
        size_t length = size_t(p[8]) | size_t(p[9]) << 8;
        if (offset == 12)
            length |= size_t(p[10]) << 16 | size_t(p[11]) << 24;
        require(length <= file_bytes - offset, "NPY header length");
        std::string header(reinterpret_cast<const char *>(p + offset), length);
        require(header.find("'fortran_order': False") != std::string::npos,
                "requires C-contiguous NPY array");
        std::smatch match;
        require(std::regex_search(header, match, std::regex("'descr':\\s*'([^']+)'")), "NPY dtype");
        dtype = match[1];
        size_t item = dtype == "<f2"                       ? 2
                      : dtype == "<f4"                     ? 4
                      : (dtype == "<i8" || dtype == "<f8") ? 8
                      : (dtype == "|b1" || dtype == "|u1") ? 1
                                                           : 0;
        require(item != 0, "unsupported NPY dtype " + dtype);
        require(std::regex_search(header, match, std::regex("'shape':\\s*\\(([^)]*)\\)")),
                "NPY shape");
        std::string dims = match[1];
        const std::regex numbers("[0-9]+");
        for (std::sregex_iterator it(dims.begin(), dims.end(), numbers), end; it != end; ++it) {
            auto n = std::stoll(it->str());
            require(n > 0 && size_t(n) <= SIZE_MAX / count, "NPY dimension overflow");
            shape.push_back(n);
            count *= n;
        }
        require(count <= SIZE_MAX / item, "NPY size overflow");
        bytes = count * item;
        require(bytes == file_bytes - offset - length, "NPY payload length");
        data = p + offset + length;
        if (!expected_sha.empty())
            require(kaivuri::sha256_hex(p, file_bytes) == expected_sha,
                    "array checksum mismatch: " + path.string());
    }
};
size_t item_size(DataType type) {
    switch (type) {
    case DataType::kFLOAT:
    case DataType::kINT32:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT64:
        return 8;
    case DataType::kBOOL:
    case DataType::kUINT8:
    case DataType::kINT8:
        return 1;
    default:
        throw std::runtime_error("unsupported TensorRT dtype");
    }
}
struct Buffer {
    void *ptr = nullptr, *host = nullptr;
    size_t bytes;
    std::vector<int64_t> shape;
    DataType type;
    Buffer(std::vector<int64_t> dims, DataType dtype) : shape(std::move(dims)), type(dtype) {
        bytes = item_size(type);
        for (auto n : shape) {
            require(n > 0 && size_t(n) <= SIZE_MAX / bytes, "invalid static tensor shape");
            bytes *= n;
        }
        ck(cudaMalloc(&ptr, bytes), "cudaMalloc buffer");
        ck(cudaMemset(ptr, 0, bytes), "initialize buffer");
    }
    ~Buffer() {
        if (host)
            cudaFreeHost(host);
        if (ptr)
            cudaFree(ptr);
    }
    Buffer(const Buffer &) = delete;
    void *staging() {
        if (!host)
            ck(cudaMallocHost(&host, bytes), "pinned staging");
        return host;
    }
};
struct Logger : ILogger {
    void log(Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << message << '\n';
    }
};
struct Allocator : IGpuAllocator {
    void *allocate(uint64_t size, uint64_t, AllocatorFlags) noexcept override {
        void *p = nullptr;
        return size && cudaMalloc(&p, size) == cudaSuccess ? p : nullptr;
    }
    bool deallocate(void *p) noexcept override { return !p || cudaFree(p) == cudaSuccess; }
    void *allocateAsync(uint64_t size, uint64_t alignment, AllocatorFlags flags,
                        cudaStream_t) noexcept override {
        return allocate(size, alignment, flags);
    }
    bool deallocateAsync(void *p, cudaStream_t) noexcept override { return deallocate(p); }
    void *reallocate(void *, uint64_t, uint64_t) noexcept override { return nullptr; }
};
struct Engine {
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
};
// Prompt row element: FP16 as rounded, or widened for an FP32 residual stream.
void store(__half &target, __half value) {
    target = value;
}
void store(float &target, __half value) {
    target = __half2float(value);
}
} // namespace

namespace kaivuri::pi05 {
class Runtime::Impl {
    Logger logger;
    Allocator allocator;
    std::unique_ptr<IRuntime> runtime;
    std::map<std::string, Engine> engines;
    std::vector<std::unique_ptr<Buffer>> storage;
    std::unique_ptr<Buffer> scratch;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph = nullptr;
    std::array<cudaEvent_t, 4> events{};
    YAML::Node info;
    Npy embedding, rope_cos, rope_sin, conditioning;
    Buffer *images[2]{}, *vis[2]{}, *features{}, *prefix[2]{}, *rows{}, *p_mask{}, *p_cos{},
        *p_sin{}, *expert[2]{}, *s_mask{}, *s_cos{}, *s_sin{}, *actions[2]{}, *velocity{},
        *styles{};
    std::vector<std::pair<Buffer *, Buffer *>> kv;
    std::map<std::string, std::string> kinds; // component label -> template kind
    int vision_layers = 0, language_layers = 0, action_layers = 0, horizon, width, steps,
        prefix_tokens, valid_tokens = -1;
    int style_sites;
    size_t style_row_bytes = 0, prompt_bytes = 0;
    std::vector<int64_t> current_tokens;
    bool graph_enabled, capturing = false, primed = false, native = false;
    std::string euler, policy;
    // Mode C (training-time RTC). Per step, op_rows writes every site's per-row modulation
    // into `mods` ([sites][horizon][hidden]: the t = 0 style on rows < d, the step's style
    // below), and op_prefix resets rows < d of the Euler result to `anchor`. d lives only in
    // the `rtc_mask` contents ([1][horizon][1], 1 on prefix rows), uploaded every inference,
    // so one captured graph serves every d, including 0.
    bool row_time_ = false;
    int max_rows = 0;
    size_t mod_bytes = 0; // one site's [1][horizon][hidden] FP16 block
    Buffer *clean{}, *rtc_mask{}, *anchor{}, *mods{}, *stepped{};
    const std::string op_rows = "op_rtc_rows", op_prefix = "op_rtc_prefix";
    Buffer *make(std::vector<int64_t> shape, DataType type) {
        storage.push_back(std::make_unique<Buffer>(std::move(shape), type));
        return storage.back().get();
    }
    Buffer *for_tensor(const std::string &name, const char *tensor) {
        auto &e = *engines.at(name).engine;
        auto shape = e.getTensorShape(tensor);
        require(shape.nbDims > 0, "missing tensor " + name + "/" + tensor);
        return make(std::vector<int64_t>(shape.d, shape.d + shape.nbDims),
                    e.getTensorDataType(tensor));
    }
    // The buffer's pinned staging copy as count elements of T, filled in place before push().
    template <class T> T *staged(Buffer *buffer, size_t count) {
        require(count * sizeof(T) == buffer->bytes, "upload size mismatch");
        return static_cast<T *>(buffer->staging());
    }
    void push(Buffer *buffer) {
        ck(cudaMemcpyAsync(buffer->ptr, buffer->host, buffer->bytes, cudaMemcpyHostToDevice,
                           stream),
           "upload");
    }
    void upload(Buffer *buffer, const void *data, size_t size) {
        std::memcpy(staged<uint8_t>(buffer, size), data, size);
        push(buffer);
    }
    void copy(void *dst, const void *src, size_t bytes) {
        ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream), "device copy");
    }
    static void *offset(Buffer *buf, size_t bytes) {
        require(bytes <= buf->bytes, "buffer offset");
        return static_cast<uint8_t *>(buf->ptr) + bytes;
    }
    void run(const std::string &name,
             std::initializer_list<std::pair<const char *, void *>> bindings) {
        auto &e = engines.at(name);
        require(size_t(e.engine->getNbIOTensors()) == bindings.size(),
                "incomplete bindings: " + name);
        for (auto &binding : bindings)
            require(e.context->setTensorAddress(binding.first, binding.second),
                    "binding failed: " + name + "/" + binding.first);
        require(e.context->enqueueV3(stream), "enqueue failed: " + name);
    }
    static std::string layer(const char *prefix, int n) {
        char name[64];
        std::snprintf(name, sizeof(name), "%s_%02d", prefix, n);
        return name;
    }
    void record(int index) {
        ck(capturing ? cudaEventRecordWithFlags(events[index], stream, cudaEventRecordExternal)
                     : cudaEventRecord(events[index], stream),
           "stage event");
    }
    void enqueue() {
        record(0);
        for (int camera = 0; camera < 2; ++camera) {
            run("stem", {{"image", images[camera]->ptr}, {"hidden", vis[0]->ptr}});
            for (int i = 0; i < vision_layers; ++i)
                run(layer("vision", i),
                    {{"hidden", vis[i % 2]->ptr}, {"output", vis[1 - i % 2]->ptr}});
            run("vision_tail",
                {{"hidden", vis[vision_layers % 2]->ptr}, {"output", features->ptr}});
            copy(offset(prefix[0], camera * features->bytes), features->ptr, features->bytes);
        }
        copy(offset(prefix[0], 2 * features->bytes), rows->ptr, prompt_bytes);
        record(1);
        for (int i = 0; i < language_layers; ++i) {
            auto name = layer("language", i);
            if (kinds.at(name) == "language_kv") // last layer: K/V only, no attention/MLP
                run(name, {{"hidden", prefix[i % 2]->ptr},
                           {"cos", p_cos->ptr},
                           {"sin", p_sin->ptr},
                           {"key", kv[i].first->ptr},
                           {"value", kv[i].second->ptr}});
            else
                run(name, {{"hidden", prefix[i % 2]->ptr},
                           {"output", prefix[1 - i % 2]->ptr},
                           {"mask", p_mask->ptr},
                           {"cos", p_cos->ptr},
                           {"sin", p_sin->ptr},
                           {"key", kv[i].first->ptr},
                           {"value", kv[i].second->ptr}});
        }
        record(2);
        for (int s = 0; s < steps; ++s) {
            // Per step: two style rows per action layer, then the final one.
            auto style = [&](int site) {
                return offset(styles, (s * style_sites + site) * style_row_bytes);
            };
            // Mode C: this step's per-row modulation, one [1][horizon][hidden] block per site.
            if (row_time_)
                run(op_rows, {{"step", style(0)},
                              {"clean", clean->ptr},
                              {"mask", rtc_mask->ptr},
                              {"out", mods->ptr}});
            auto mod = [&](int site) {
                return row_time_ ? offset(mods, size_t(site) * mod_bytes) : style(site);
            };
            run("action_input", {{"actions", actions[s % 2]->ptr}, {"hidden", expert[0]->ptr}});
            for (int i = 0; i < action_layers; ++i)
                run(layer("action", i), {{"hidden", expert[i % 2]->ptr},
                                         {"output", expert[1 - i % 2]->ptr},
                                         {"mask", s_mask->ptr},
                                         {"cos", s_cos->ptr},
                                         {"sin", s_sin->ptr},
                                         {"prefix_k", kv[i].first->ptr},
                                         {"prefix_v", kv[i].second->ptr},
                                         {"mod_in", mod(2 * i)},
                                         {"mod_post", mod(2 * i + 1)}});
            run("action_output", {{"hidden", expert[action_layers % 2]->ptr},
                                  {"mod_final", mod(style_sites - 1)},
                                  {"velocity", velocity->ptr}});
            if (row_time_) {
                // Euler into `stepped`, then the prefix rows back to the committed actions.
                run(euler, {{"a", actions[s % 2]->ptr}, {"b", velocity->ptr}, {"out", stepped->ptr}});
                run(op_prefix, {{"x", stepped->ptr},
                                {"anchor", anchor->ptr},
                                {"mask", rtc_mask->ptr},
                                {"out", actions[(s + 1) % 2]->ptr}});
            } else
                run(euler, {{"a", actions[s % 2]->ptr},
                            {"b", velocity->ptr},
                            {"out", actions[(s + 1) % 2]->ptr}});
        }
        record(3);
    }

    // Euler update out = a + c * b in FP32, as bench/vendor/trt_ops.py build_op("axpy").
    std::vector<char> build_axpy(const Dims &shape, float c) {
        std::cerr << "[load] building the Euler axpy op (no cached op engine)\n";
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
        require(bool(builder), "TensorRT builder");
        std::unique_ptr<INetworkDefinition> net(
            builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        Dims ones = shape;
        for (int i = 0; i < ones.nbDims; ++i)
            ones.d[i] = 1;
        auto a = net->addInput("a", DataType::kFLOAT, shape);
        auto b = net->addInput("b", DataType::kFLOAT, shape);
        auto k = net->addConstant(ones, Weights{DataType::kFLOAT, &c, 1});
        auto cb = net->addElementWise(*b, *k->getOutput(0), ElementWiseOperation::kPROD);
        auto out = net->addElementWise(*a, *cb->getOutput(0), ElementWiseOperation::kSUM);
        out->getOutput(0)->setName("out");
        net->markOutput(*out->getOutput(0));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*net, *config));
        require(plan && plan->size(), "axpy op build failed");
        auto bytes = static_cast<const char *>(plan->data());
        return std::vector<char>(bytes, bytes + plan->size());
    }

    static Dims dims3(int64_t a, int64_t b, int64_t c) {
        Dims d{};
        d.nbDims = 3;
        d.d[0] = a, d.d[1] = b, d.d[2] = c;
        return d;
    }
    // Mode C op: out = (mask > 0.5) ? then : otherwise, broadcast over the three inputs. A
    // select only copies, so every output element is bit-identical to its source.
    std::vector<char> build_select(const std::string &what, const char *then_name,
                                   const char *else_name, DataType type, const Dims &then_shape,
                                   const Dims &else_shape, const Dims &mask_shape) {
        std::cerr << "[load] building the Mode C " << what << " op\n";
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
        require(bool(builder), "TensorRT builder");
        std::unique_ptr<INetworkDefinition> net(
            builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        auto then_in = net->addInput(then_name, type, then_shape);
        auto else_in = net->addInput(else_name, type, else_shape);
        auto mask = net->addInput("mask", DataType::kFLOAT, mask_shape);
        static const float half = 0.5f;
        require(then_in && else_in && mask, what + " op: inputs");
        auto k = net->addConstant(dims3(1, 1, 1), Weights{DataType::kFLOAT, &half, 1});
        require(k != nullptr, what + " op: constant");
        auto cond = net->addElementWise(*mask, *k->getOutput(0), ElementWiseOperation::kGREATER);
        require(cond != nullptr, what + " op: comparison");
        auto select = net->addSelect(*cond->getOutput(0), *then_in, *else_in);
        require(select != nullptr, what + " op: select");
        select->getOutput(0)->setName("out");
        net->markOutput(*select->getOutput(0));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*net, *config));
        require(plan && plan->size(), what + " op build failed");
        auto bytes = static_cast<const char *>(plan->data());
        return std::vector<char>(bytes, bytes + plan->size());
    }
    void download(Buffer *buffer) {
        ck(cudaMemcpyAsync(buffer->staging(), buffer->ptr, buffer->bytes, cudaMemcpyDeviceToHost,
                           stream),
           "download");
        ck(cudaStreamSynchronize(stream), "download synchronize");
    }
    // Load-time check of the Mode C ops on this device: op_rows takes the clean table on mask
    // rows and the step's styles elsewhere, op_prefix the anchor on mask rows and x elsewhere,
    // all bit for bit. Leaves the mask and anchor zero (d = 0).
    void check_rtc_ops(const Npy &clean_table) {
        const int n = noise_count();
        float *m = staged<float>(rtc_mask, size_t(horizon));
        std::fill(m, m + horizon, 0.f);
        m[0] = m[2] = 1.f;
        push(rtc_mask);
        run(op_rows,
            {{"step", styles->ptr}, {"clean", clean->ptr}, {"mask", rtc_mask->ptr}, {"out", mods->ptr}});
        download(mods);
        const size_t row = style_row_bytes;
        auto out = static_cast<const uint8_t *>(mods->host);
        for (int site = 0; site < style_sites; ++site)
            for (int r = 0; r < horizon; ++r) {
                const uint8_t *want = (r == 0 || r == 2) ? clean_table.data + site * row
                                                         : conditioning.data + site * row;
                require(std::memcmp(out + site * mod_bytes + r * row, want, row) == 0,
                        "Mode C row-style op picked the wrong source at site " +
                            std::to_string(site) + ", row " + std::to_string(r));
            }
        float *x = staged<float>(stepped, size_t(n)), *a = staged<float>(anchor, size_t(n));
        for (int i = 0; i < n; ++i)
            x[i] = float(i) + .25f, a[i] = -float(i) - .5f;
        push(stepped);
        push(anchor);
        run(op_prefix, {{"x", stepped->ptr},
                        {"anchor", anchor->ptr},
                        {"mask", rtc_mask->ptr},
                        {"out", actions[1]->ptr}});
        download(actions[1]);
        auto y = static_cast<const float *>(actions[1]->host);
        for (int i = 0; i < n; ++i) {
            const int r = i / width;
            const float want = (r == 0 || r == 2) ? a[i] : x[i];
            require(std::memcmp(&y[i], &want, 4) == 0,
                    "Mode C prefix op picked the wrong source at row " + std::to_string(r));
        }
        std::fill(m, m + horizon, 0.f);
        std::fill(a, a + n, 0.f);
        push(rtc_mask);
        push(anchor);
        ck(cudaStreamSynchronize(stream), "Mode C op check");
    }

  public:
    explicit Impl(const fs::path &bundle, const fs::path &cache, bool use_graph)
        : info(YAML::LoadFile((bundle / "bundle.json").string())),
          embedding(bundle / "embedding_fp16.npy", info["embedding_sha256"].as<std::string>()),
          rope_cos(bundle / "rope_cos_fp16.npy"), rope_sin(bundle / "rope_sin_fp16.npy"),
          conditioning(bundle /
                       value_or(info["conditioning"], std::string("conditioning_10step_fp16.npy"))),
          graph_enabled(use_graph) {
        policy = info["policy"].as<std::string>();
        require(info["complete"].as<bool>() &&
                    (policy == "pi05_libero" || policy == "pi05_openpi_state") &&
                    (value_or(info["valid_camera_compute_only"], false) ||
                     value_or(info["layout"], std::string()) == "compact"),
                "requires a compact OpenPI pi05_libero or pi05_openpi_state bundle");
        const auto residual = value_or(info["language_residual"], std::string("float16"));
        require(residual == "float16" || residual == "float32", "language_residual");
        horizon = info["action_horizon"].as<int>();
        width = info["action_dim"].as<int>();
        steps = info["num_steps"].as<int>();
        prefix_tokens = info["prefix_tokens"].as<int>();
        std::set<std::string> needed;
        for (auto component : info["components"]) {
            auto label = component["label"].as<std::string>(),
                 kind = component["kind"].as<std::string>();
            kinds[label] = kind;
            needed.insert(kind);
            vision_layers += kind == "vision";
            action_layers += kind == "action";
            language_layers += label.rfind("language_", 0) == 0;
        }
        for (int i = 0; i < language_layers; ++i) {
            auto kind = kinds.count(layer("language", i)) ? kinds.at(layer("language", i)) : "";
            require(kind == "language" || (kind == "language_kv" && i == language_layers - 1),
                    "language layers must be language_00.. with an optional K/V-only last layer");
        }
        require(info["valid_cameras"].as<int>() == 2 && prefix_tokens >= 512 && vision_layers > 0 &&
                    action_layers == language_layers && steps > 0 && horizon > 0 && width > 0,
                "unsupported bundle dimensions");
        std::map<std::string, std::vector<char>> plans;
        // AOT cache (candidate.json, attention plugin compiled into the action plan), or the
        // native template cache: one <kind>.engine per kind, keyed by <kind>.onnx.sha256.
        native = !fs::exists(cache / "candidate.json");
        if (!native) {
            auto candidate = YAML::LoadFile((cache / "candidate.json").string());
            require(fs::canonical(candidate["bundle"].as<std::string>()) == fs::canonical(bundle),
                    "engine cache belongs to another bundle");
            for (auto entry : candidate["template_sha256"])
                require(
                    kaivuri::sha256_file(bundle / "templates" / entry.first.as<std::string>()) ==
                        entry.second.as<std::string>(),
                    "template checksum mismatch");
            for (auto entry : candidate["engine_sha256"]) {
                auto name = entry.first.as<std::string>();
                auto bytes = read_file(cache / name);
                require(kaivuri::sha256_hex(bytes.data(), bytes.size()) ==
                            entry.second.as<std::string>(),
                        "engine checksum mismatch");
                plans[fs::path(name).stem().string()] = std::move(bytes);
            }
        } else
            for (auto &kind : needed) {
                std::ifstream key(cache / (kind + ".onnx.sha256"));
                std::string onnx_sha;
                require(bool(key >> onnx_sha), "native cache lacks " + kind + ".onnx.sha256");
                require(onnx_sha == kaivuri::sha256_file(bundle / "templates" / (kind + ".onnx")),
                        "native " + kind + " plan was built from another template");
                plans[kind] = read_file(cache / (kind + ".engine"));
            }
        std::cerr << "[load] " << (native ? "native TensorRT" : "AOT attention") << " cache, "
                  << residual << " language residual\n";
        require(initLibNvInferPlugins(&logger, ""), "plugin registration failed");
        runtime.reset(createInferRuntime(logger));
        require(bool(runtime), "TensorRT runtime");
        runtime->setGpuAllocator(&allocator);
        size_t scratch_bytes = 0;
        for (auto component : info["components"]) {
            auto name = component["label"].as<std::string>(),
                 kind = component["kind"].as<std::string>();
            require(plans.count(kind), "cache lacks the " + kind + " plan");
            auto &plan = plans.at(kind);
            auto &item = engines[name];
            item.engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
            require(bool(item.engine), "cannot deserialize " + name);
            fs::path directory = bundle / "weights" / name;
            auto index = YAML::LoadFile((directory / "index.json").string());
            std::map<std::string, YAML::Node> by_name;
            for (YAML::Node wt : index["weights"])
                by_name[wt["name"].as<std::string>()] = wt;
            std::unique_ptr<IRefitter> refitter(createInferRefitter(*item.engine, logger));
            require(bool(refitter), "cannot create refitter");
            int n = refitter->getAllWeights(0, nullptr);
            std::vector<const char *> names(n);
            refitter->getAllWeights(n, names.data());
            std::vector<std::unique_ptr<Npy>> arrays;
            for (auto key : names) {
                auto wt = by_name.at(key);
                arrays.push_back(std::make_unique<Npy>(directory / wt["file"].as<std::string>(),
                                                       wt["sha256"].as<std::string>()));
                auto &a = *arrays.back();
                require(a.dtype == "<f2" || a.dtype == "<f4", "refit dtype");
                require(refitter->setNamedWeights(
                            key, Weights{a.dtype == "<f2" ? DataType::kHALF : DataType::kFLOAT,
                                         a.data, int64_t(a.count)}),
                        "refit rejected " + name + "/" + key);
            }
            require(refitter->getMissingWeights(0, nullptr) == 0 && refitter->refitCudaEngine(),
                    "refit failed: " + name);
            scratch_bytes = std::max(scratch_bytes, size_t(item.engine->getDeviceMemorySizeV2()));
            std::cerr << "[load] " << name << '\n';
        }
        plans.clear();
        // The cached Euler op must match this bundle's action shape; its constant is
        // checked numerically below. Without one, it is built here (seconds, FP32).
        const auto action_shape = engines.at("action_input").engine->getTensorShape("actions");
        const float euler_dt = float(-1.0 / steps);
        std::vector<std::pair<std::string, std::vector<char>>> ops;
        for (auto &path : fs::directory_iterator(cache))
            if (path.path().filename().string().find("op_axpy_") == 0 &&
                path.path().extension() == ".engine")
                ops.emplace_back(path.path().stem().string(), read_file(path.path()));
        if (ops.empty())
            ops.emplace_back("op_axpy_built", build_axpy(action_shape, euler_dt));
        for (auto &op : ops) {
            std::unique_ptr<ICudaEngine> engine(
                runtime->deserializeCudaEngine(op.second.data(), op.second.size()));
            require(bool(engine), "cannot deserialize Euler op " + op.first);
            auto shape = engine->getTensorShape("a");
            if (shape.nbDims != action_shape.nbDims ||
                !std::equal(shape.d, shape.d + shape.nbDims, action_shape.d))
                continue;
            require(euler.empty(), "ambiguous Euler op cache");
            euler = op.first;
            scratch_bytes = std::max(scratch_bytes, size_t(engine->getDeviceMemorySizeV2()));
            engines[euler].engine = std::move(engine);
        }
        require(!euler.empty(), "no cached Euler op matches the action shape");
        // Mode C: kaivuri.rtc.row_time must agree with what the action templates take, one
        // modulation row (broadcast) or one per action row, at every modulation input.
        const YAML::Node &meta = info;
        const YAML::Node rtc_meta = meta["kaivuri"] ? meta["kaivuri"]["rtc"] : YAML::Node();
        row_time_ = rtc_meta && value_or(rtc_meta["row_time"], false);
        require(conditioning.shape.size() == 3, "conditioning contract");
        const int64_t hidden = conditioning.shape[2], sites = conditioning.shape[1];
        const int64_t mod_rows = row_time_ ? horizon : 1;
        auto check_modulation = [&](const std::string &engine, const char *tensor) {
            auto d = engines.at(engine).engine->getTensorShape(tensor);
            require(d.nbDims == 3 && d.d[0] == 1 && d.d[2] == hidden,
                    "modulation input " + engine + "/" + tensor);
            require(d.d[1] == mod_rows,
                    row_time_ ? "the bundle declares kaivuri.rtc.row_time, but " + engine + "/" +
                                    tensor + " takes one modulation row (re-export with --row-time)"
                              : engine + "/" + tensor +
                                    " takes per-row modulation, but the bundle does not declare "
                                    "kaivuri.rtc.row_time");
        };
        for (int i = 0; i < action_layers; ++i) {
            check_modulation(layer("action", i), "mod_in");
            check_modulation(layer("action", i), "mod_post");
        }
        check_modulation("action_output", "mod_final");
        if (row_time_) {
            max_rows = rtc_meta["max_prefix"].as<int>();
            require(max_rows > 0 && max_rows < horizon, "kaivuri.rtc.max_prefix outside (0, horizon)");
            require(action_shape.nbDims == 3 && action_shape.d[0] == 1 &&
                        action_shape.d[1] == horizon && action_shape.d[2] == width,
                    "action buffer is not [1][horizon][action_dim]");
            const auto mask_shape = dims3(1, horizon, 1);
            auto add_op = [&](const std::string &name, std::vector<char> plan) {
                std::unique_ptr<ICudaEngine> engine(
                    runtime->deserializeCudaEngine(plan.data(), plan.size()));
                require(bool(engine), "cannot deserialize " + name);
                scratch_bytes = std::max(scratch_bytes, size_t(engine->getDeviceMemorySizeV2()));
                engines[name].engine = std::move(engine);
            };
            add_op(op_rows, build_select("row-style", "clean", "step", DataType::kHALF,
                                         dims3(sites, 1, hidden), dims3(sites, 1, hidden),
                                         mask_shape));
            add_op(op_prefix, build_select("prefix", "anchor", "x", DataType::kFLOAT, action_shape,
                                           action_shape, mask_shape));
            const auto out = engines.at(op_rows).engine->getTensorShape("out");
            require(out.nbDims == 3 && out.d[0] == sites && out.d[1] == horizon && out.d[2] == hidden,
                    "Mode C row-style op output shape");
        }
        scratch = std::make_unique<Buffer>(
            std::vector<int64_t>{int64_t(std::max(size_t(1), scratch_bytes))}, DataType::kUINT8);
        for (auto &entry : engines) {
            entry.second.context.reset(entry.second.engine->createExecutionContext(
                ExecutionContextAllocationStrategy::kUSER_MANAGED));
            require(bool(entry.second.context), "cannot create context " + entry.first);
            entry.second.context->setDeviceMemoryV2(scratch->ptr, scratch->bytes);
        }
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
        for (auto &event : events)
            ck(cudaEventCreate(&event), "event create");
        for (int i = 0; i < 2; ++i) {
            images[i] = for_tensor("stem", "image");
            vis[i] = for_tensor("vision_00", "hidden");
            prefix[i] = for_tensor("language_00", "hidden");
            expert[i] = for_tensor("action_00", "hidden");
            actions[i] = for_tensor("action_input", "actions");
        }
        features = for_tensor("vision_tail", "output");
        const auto prefix_type = residual == "float32" ? DataType::kFLOAT : DataType::kHALF;
        require(images[0]->type == DataType::kHALF &&
                    images[0]->shape == std::vector<int64_t>({1, 3, 224, 224}) &&
                    prefix[0]->type == prefix_type && features->type == prefix_type,
                "image/prefix types do not match the bundle's language_residual");
        // Prompt rows only; the zero tail stays zero. Copied into the prefix before every pass.
        rows = make({1, prefix_tokens - 512, prefix[0]->shape.back()}, prefix_type);
        prompt_bytes = rows->bytes;
        require(2 * features->bytes + rows->bytes == prefix[0]->bytes,
                "prefix is not two 256-token cameras plus the prompt capacity");
        p_mask = for_tensor("language_00", "mask");
        p_cos = for_tensor("language_00", "cos");
        p_sin = for_tensor("language_00", "sin");
        s_mask = for_tensor("action_00", "mask");
        s_cos = for_tensor("action_00", "cos");
        s_sin = for_tensor("action_00", "sin");
        for (int i = 0; i < language_layers; ++i)
            kv.emplace_back(for_tensor("language_00", "key"), for_tensor("language_00", "value"));
        velocity = for_tensor("action_output", "velocity");
        require(actions[0]->type == DataType::kFLOAT && velocity->type == DataType::kFLOAT &&
                    velocity->bytes == actions[0]->bytes,
                "action/velocity buffers");
        require(conditioning.shape.size() == 3 && conditioning.shape[0] == steps &&
                    conditioning.shape[1] == 2 * action_layers + 1 && conditioning.dtype == "<f2",
                "conditioning contract");
        style_sites = conditioning.shape[1];
        style_row_bytes = conditioning.shape[2] * 2;
        styles = make(conditioning.shape, DataType::kHALF);
        upload(styles, conditioning.data, conditioning.bytes);
        // Euler op check: 0 + c * 1 must give exactly the bundle's -1/num_steps.
        std::vector<float> zero(noise_count(), 0.f), one(noise_count(), 1.f);
        require(zero.size() * 4 == actions[0]->bytes, "action buffer size");
        upload(actions[0], zero.data(), actions[0]->bytes);
        upload(velocity, one.data(), velocity->bytes);
        run(euler, {{"a", actions[0]->ptr}, {"b", velocity->ptr}, {"out", actions[1]->ptr}});
        ck(cudaMemcpyAsync(actions[1]->staging(), actions[1]->ptr, actions[1]->bytes,
                           cudaMemcpyDeviceToHost, stream),
           "Euler check download");
        ck(cudaStreamSynchronize(stream), "initial uploads");
        auto check = static_cast<const float *>(actions[1]->host);
        for (int i = 0; i < noise_count(); ++i)
            require(check[i] == euler_dt, "Euler op constant is not -1/num_steps");
        if (row_time_) {
            // The t = 0 style table, [sites][hidden] FP16, as the exporter computed it.
            const Npy clean_table(bundle / rtc_meta["style_clean"].as<std::string>());
            require(clean_table.dtype == "<f2" &&
                        clean_table.shape == std::vector<int64_t>({sites, hidden}),
                    "kaivuri.rtc.style_clean is not [sites][hidden] float16");
            clean = make({sites, 1, hidden}, DataType::kHALF);
            upload(clean, clean_table.data, clean_table.bytes);
            rtc_mask = make({1, horizon, 1}, DataType::kFLOAT);
            anchor = for_tensor("action_input", "actions");
            stepped = for_tensor("action_input", "actions");
            mods = make({sites, horizon, hidden}, DataType::kHALF);
            mod_bytes = mods->bytes / size_t(sites);
            check_rtc_ops(clean_table);
            std::cerr << "[load] Mode C (training-time RTC): up to " << max_rows
                      << " prefix rows, per-row modulation " << mods->bytes / 1048576.0 << " MiB\n";
        }
        std::cerr << "[load] shared scratch MiB=" << scratch_bytes / 1048576.0 << '\n';
    }
    ~Impl() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (graph)
            cudaGraphExecDestroy(graph);
        for (auto event : events)
            if (event)
                cudaEventDestroy(event);
        storage.clear();
        engines.clear();
        scratch.reset();
        runtime.reset();
        if (stream)
            cudaStreamDestroy(stream);
    }
    // Called every step for pi05_openpi_state. Uploads the prompt rows (~hidden*4 bytes per
    // token), masks/RoPE only when the valid token count changes. The CUDA graph reads these
    // buffers by address, so it is captured once and never rebuilt for a new prompt. Host data
    // is written straight into the pinned staging buffers: every earlier upload from them has
    // completed, since each inference (and the first prime) synchronizes the stream.
    void set_prompt(const std::vector<int64_t> &tokens) {
        if (tokens == current_tokens)
            return;
        require(!tokens.empty() && 512 + tokens.size() <= size_t(prefix_tokens),
                "prompt exceeds compact prefix; re-export without truncation");
        require(embedding.shape.size() == 2 && embedding.shape[1] == prefix[0]->shape.back() &&
                    embedding.dtype == "<f2",
                "embedding contract");
        for (auto token : tokens)
            require(token >= 0 && token < embedding.shape[0], "token outside embedding");
        const int hidden = embedding.shape[1], valid = 512 + tokens.size();
        const float scale = std::sqrt(float(hidden));
        const size_t count = size_t(prefix_tokens - 512) * hidden;
        // Scaled embedding rows rounded to FP16 as the export does; the tail stays zero.
        auto fill = [&](auto *prompt) {
            for (size_t k = tokens.size() * hidden; k < count; ++k)
                store(prompt[k], __float2half(0));
            for (size_t i = 0; i < tokens.size(); ++i) {
                auto source = reinterpret_cast<const __half *>(embedding.data) + tokens[i] * hidden;
                for (int j = 0; j < hidden; ++j)
                    store(prompt[i * hidden + j], __float2half_rn(__half2float(source[j]) * scale));
            }
        };
        if (rows->type == DataType::kHALF)
            fill(staged<__half>(rows, count));
        else
            fill(staged<float>(rows, count));
        push(rows);
        if (valid != valid_tokens) {
            constexpr float negative = -2.3819763e38f;
            float *mask = staged<float>(p_mask, size_t(prefix_tokens) * prefix_tokens);
            for (int i = 0; i < prefix_tokens; ++i)
                for (int j = 0; j < prefix_tokens; ++j)
                    mask[size_t(i) * prefix_tokens + j] = (i < valid && j < valid) ? 0.f : negative;
            push(p_mask);
            const int keys = prefix_tokens + horizon;
            mask = staged<float>(s_mask, size_t(horizon) * keys);
            for (int i = 0; i < horizon; ++i)
                for (int j = 0; j < keys; ++j)
                    mask[size_t(i) * keys + j] = (j < valid || j >= prefix_tokens) ? 0.f : negative;
            push(s_mask);
            require(rope_cos.dtype == "<f2" && rope_sin.dtype == "<f2" &&
                        rope_cos.shape == rope_sin.shape && rope_cos.shape.size() == 2,
                    "RoPE contract");
            auto rope = [&](Buffer *target, const Npy &table, bool suffix) {
                const size_t row_bytes = size_t(table.shape[1]) * 2;
                const int n = suffix ? horizon : prefix_tokens;
                auto bytes = staged<uint8_t>(target, n * row_bytes);
                for (int i = 0; i < n; ++i) {
                    int row = suffix ? valid + i + 1 : std::min(i + 1, valid);
                    require(row < table.shape[0], "RoPE position outside table");
                    std::memcpy(bytes + size_t(i) * row_bytes, table.data + size_t(row) * row_bytes,
                                row_bytes);
                }
                push(target);
            };
            rope(p_cos, rope_cos, false);
            rope(p_sin, rope_sin, false);
            rope(s_cos, rope_cos, true);
            rope(s_sin, rope_sin, true);
            valid_tokens = valid;
        }
        current_tokens = tokens;
        if (primed)
            return; // Uploads are stream-ordered before the next launch.
        ck(cudaStreamSynchronize(stream), "prompt uploads");
        enqueue();
        ck(cudaStreamSynchronize(stream), "prime chain");
        if (graph_enabled) {
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph");
            cudaGraph_t captured = nullptr;
            capturing = true;
            try {
                enqueue();
            } catch (...) {
                capturing = false;
                cudaStreamEndCapture(stream, &captured);
                if (captured)
                    cudaGraphDestroy(captured);
                throw;
            }
            capturing = false;
            ck(cudaStreamEndCapture(stream, &captured), "end graph");
            auto result = cudaGraphInstantiate(&graph, captured, 0);
            cudaGraphDestroy(captured);
            ck(result, "instantiate graph");
        }
        primed = true;
    }
    const YAML::Node &bundle_info() const { return info; }
    const std::string &policy_name() const { return policy; }
    bool native_cache() const { return native; }
    int prefix_size() const { return prefix_tokens; }
    std::vector<int64_t> task_tokens(const std::string &task) const {
        auto entry = info["tasks"][task];
        require(bool(entry), "task missing from bundle token table");
        auto ids = entry["tokens"], mask = entry["mask"];
        require(ids.size() == mask.size(), "task token mask");
        std::vector<int64_t> tokens;
        for (size_t i = 0; i < ids.size(); ++i)
            if (mask[i].as<bool>())
                tokens.push_back(ids[i].as<int64_t>());
        return tokens;
    }
    int noise_count() const { return horizon * width; }
    int action_horizon() const { return horizon; }
    int action_dim() const { return width; }
    int denoising_steps() const { return steps; }
    bool row_time() const { return row_time_; }
    int max_prefix() const { return max_rows; }
    std::vector<float> infer(const void *image0, const void *image1, const float *noise,
                             Timing &timing, int prefix_rows, const float *prefix,
                             int prefix_dims) {
        require(!current_tokens.empty(), "set prompt before inference");
        require(prefix_rows >= 0, "negative RTC prefix");
        if (prefix_rows) {
            require(row_time_, "RTC prefix given, but this bundle has no Mode C (row-time) "
                               "action templates");
            require(prefix_rows <= max_rows, "RTC prefix of " + std::to_string(prefix_rows) +
                                                 " rows exceeds the trained maximum " +
                                                 std::to_string(max_rows));
            require(prefix != nullptr && prefix_dims > 0 && prefix_dims <= width,
                    "RTC prefix shape");
            for (int i = 0; i < prefix_rows * prefix_dims; ++i)
                require(std::isfinite(prefix[i]), "non-finite RTC prefix");
        }
        uint64_t start = monotonic_ns();
        upload(images[0], image0, images[0]->bytes);
        upload(images[1], image1, images[1]->bytes);
        const int n = noise_count();
        float *initial = staged<float>(actions[0], size_t(n));
        std::memcpy(initial, noise, actions[0]->bytes);
        if (row_time_) {
            // Every call: rows < d are [prefix, zeros] (a clean training row), the mask marks
            // them; with d = 0 both are all zero and the graph runs the plain sampler.
            float *mask = staged<float>(rtc_mask, size_t(horizon));
            float *clean_rows = staged<float>(anchor, size_t(n));
            std::fill(mask, mask + horizon, 0.f);
            std::fill(clean_rows, clean_rows + n, 0.f);
            for (int r = 0; r < prefix_rows; ++r) {
                mask[r] = 1.f;
                for (int c = 0; c < width; ++c)
                    initial[r * width + c] = clean_rows[r * width + c] =
                        c < prefix_dims ? prefix[r * prefix_dims + c] : 0.f;
            }
            push(rtc_mask);
            push(anchor);
        }
        push(actions[0]);
        if (graph)
            ck(cudaGraphLaunch(graph, stream), "launch graph");
        else
            enqueue();
        Buffer *result = actions[steps % 2];
        ck(cudaMemcpyAsync(result->staging(), result->ptr, result->bytes, cudaMemcpyDeviceToHost,
                           stream),
           "download actions");
        ck(cudaStreamSynchronize(stream), "inference synchronize");
        auto elapsed = [&](int stage, const char *what) {
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, events[stage], events[stage + 1]), what);
            return double(ms);
        };
        timing.vision_ms = elapsed(0, "vision timing");
        timing.language_ms = elapsed(1, "language timing");
        timing.denoise_ms = elapsed(2, "denoise timing");
        timing.total_ms = (monotonic_ns() - start) / 1e6;
        auto begin = static_cast<float *>(result->host);
        return {begin, begin + noise_count()};
    }
};

Runtime::Runtime(const fs::path &bundle, const fs::path &cache, bool cuda_graph)
    : impl_(std::make_unique<Impl>(bundle, cache, cuda_graph)) {}
Runtime::~Runtime() = default;
const YAML::Node &Runtime::bundle_info() const { return impl_->bundle_info(); }
const std::string &Runtime::policy_name() const { return impl_->policy_name(); }
bool Runtime::native_cache() const { return impl_->native_cache(); }
int Runtime::prefix_size() const { return impl_->prefix_size(); }
int Runtime::noise_count() const { return impl_->noise_count(); }
int Runtime::action_horizon() const { return impl_->action_horizon(); }
int Runtime::action_dim() const { return impl_->action_dim(); }
int Runtime::denoising_steps() const { return impl_->denoising_steps(); }
std::vector<int64_t> Runtime::task_tokens(const std::string &task) const {
    return impl_->task_tokens(task);
}
void Runtime::set_prompt(const std::vector<int64_t> &tokens) { impl_->set_prompt(tokens); }
std::vector<float> Runtime::infer(const void *image0, const void *image1, const float *noise,
                                  Timing &timing) {
    return impl_->infer(image0, image1, noise, timing, 0, nullptr, 0);
}
std::vector<float> Runtime::infer(const void *image0, const void *image1, const float *noise,
                                  Timing &timing, int prefix_rows, const float *prefix,
                                  int prefix_dims) {
    return impl_->infer(image0, image1, noise, timing, prefix_rows, prefix, prefix_dims);
}
bool Runtime::row_time() const { return impl_->row_time(); }
int Runtime::max_prefix() const { return impl_->max_prefix(); }
} // namespace kaivuri::pi05

namespace {
struct Tap {
    int index;
    float weight;
};
std::vector<std::vector<Tap>> taps(int input, int output) {
    double scale = double(output) / input, support = std::max(1., 1. / scale);
    int count = int(std::ceil(2 * support)) + 2;
    std::vector<std::vector<Tap>> result(output);
    for (int i = 0; i < output; ++i) {
        double center = (i + .5) / scale, sum = 0;
        int lo = int(std::floor(center - support));
        std::vector<double> weights(count);
        for (int k = 0; k < count; ++k) {
            int n = lo + k;
            weights[k] =
                (n < 0 || n >= input) ? 0. : std::max(0., 1. - std::abs(n + .5 - center) / support);
            sum += weights[k];
        }
        for (int k = 0; k < count; ++k)
            result[i].push_back(
                {std::clamp(lo + k, 0, input - 1), float(weights[k] / std::max(sum, 1e-12))});
    }
    return result;
}
constexpr int side = 224, plane = side * side;
// Aspect-preserving resize target inside the 224x224 model image, centered with padding.
struct Fit {
    int width, height, top, left;
};
Fit fit(int width, int height, int stride, int channels) {
    require(width > 0 && height > 0 && (channels == 1 || channels == 3) &&
                stride >= width * channels,
            "invalid image");
    double ratio = std::max(width / double(side), height / double(side));
    int rh = int(height / ratio), rw = int(width / ratio);
    require(rh > 0 && rw > 0, "image aspect ratio");
    return {rw, rh, (side - rh) / 2, (side - rw) / 2};
}
// Writes channel c of one padded-image pixel; gray input (one channel) fills all three.
void put(std::vector<__half> &image, int channels, int c, int pixel, __half value) {
    if (channels == 1)
        for (int ch = 0; ch < 3; ++ch)
            image[ch * plane + pixel] = value;
    else
        image[c * plane + pixel] = value;
}
// OpenPI's separable antialiased linear resize, centered black padding, uint8
// round-to-even, [-1,1] normalization. Gray IR is replicated into three channels.
std::vector<__half> preprocess(const uint8_t *input, int width, int height, int stride,
                               int channels) {
    const Fit f = fit(width, height, stride, channels);
    auto yt = taps(height, f.height), xt = taps(width, f.width);
    std::vector<float> vertical(size_t(f.height) * width * channels);
    for (int y = 0; y < f.height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c) {
                float value = 0;
                for (auto tap : yt[y])
                    value +=
                        float(input[size_t(tap.index) * stride + x * channels + c]) * tap.weight;
                vertical[(size_t(y) * width + x) * channels + c] = value;
            }
    std::vector<__half> result(3 * plane, __float2half(-1));
    for (int y = 0; y < f.height; ++y)
        for (int x = 0; x < f.width; ++x)
            for (int c = 0; c < channels; ++c) {
                float value = 0;
                for (auto tap : xt[x])
                    value += vertical[(size_t(y) * width + tap.index) * channels + c] * tap.weight;
                float pixel = std::clamp(std::nearbyint(value), 0.f, 255.f);
                put(result, channels, c, (y + f.top) * side + x + f.left,
                    __float2half_rn((pixel / 255.f) * 2.f - 1.f));
            }
    return result;
}
// pi05_openpi_state: OpenPI's ResizeImages data transform (openpi_client image_tools: Pillow
// BILINEAR, black padding), then the model's x/255*2-1. Gray IR is replicated.
std::vector<__half> preprocess_pil(const uint8_t *input, int width, int height, int stride,
                                   int channels) {
    const Fit f = fit(width, height, stride, channels);
    const bool same = width == side && height == side; // Pillow returns an unresampled copy
    std::vector<__half> result(3 * plane, __float2half(-1));
    for (int c = 0; c < channels; ++c) {
        std::vector<uint8_t> resized;
        if (!same)
            resized =
                kp::pil_bilinear(input + c, width, height, stride, channels, f.width, f.height);
        for (int y = 0; y < f.height; ++y)
            for (int x = 0; x < f.width; ++x) {
                const uint8_t value = same ? input[size_t(y) * stride + x * channels + c]
                                           : resized[size_t(y) * f.width + x];
                put(result, channels, c, (y + f.top) * side + x + f.left,
                    __float2half_rn((float(value) / 255.f) * 2.f - 1.f));
            }
    }
    return result;
}
struct Comparison {
    double cosine = 0, max_abs = 0, mean_abs = 0;
    bool finite = true;
};
Comparison compare(const std::vector<float> &values, const Npy &reference) {
    require(reference.dtype == "<f4" && reference.count == values.size(), "reference shape/type");
    auto ref = reinterpret_cast<const float *>(reference.data);
    double dot = 0, a = 0, b = 0;
    Comparison result;
    for (size_t i = 0; i < values.size(); ++i) {
        result.finite &= std::isfinite(values[i]) && std::isfinite(ref[i]);
        double error = std::abs(double(values[i]) - ref[i]);
        result.max_abs = std::max(result.max_abs, error);
        result.mean_abs += error;
        dot += double(values[i]) * ref[i];
        a += double(values[i]) * values[i];
        b += double(ref[i]) * ref[i];
    }
    result.cosine = dot / std::max(std::sqrt(a * b), 1e-30);
    result.mean_abs /= values.size();
    return result;
}
std::string comparison_json(const Comparison &c) {
    std::ostringstream out;
    out << std::setprecision(12) << "{\"cosine\":" << c.cosine << ",\"max_abs\":" << c.max_abs
        << ",\"mean_abs\":" << c.mean_abs << '}';
    return out.str();
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
std::vector<float> floats(const Npy &array) {
    require(array.dtype == "<f4", "float32 array required");
    auto begin = reinterpret_cast<const float *>(array.data);
    return {begin, begin + array.count};
}
} // namespace

namespace kaivuri::pi05 {
namespace {
Image bits(const std::vector<__half> &image) {
    Image out(image.size());
    std::memcpy(out.data(), image.data(), image.size() * sizeof(__half));
    return out;
}
} // namespace
Image preprocess_libero(const uint8_t *input, int width, int height, int stride, int channels) {
    return bits(preprocess(input, width, height, stride, channels));
}
Image preprocess_openpi(const uint8_t *input, int width, int height, int stride, int channels) {
    return bits(preprocess_pil(input, width, height, stride, channels));
}
// OpenPI Unnormalize of a normalized [horizon][width] chunk: the first robot_dim columns.
std::vector<double> robot_chunk(const std::vector<float> &normalized, int horizon, int width,
                                const Contract &contract) {
    const int r = contract.robot_dim;
    std::vector<double> robot(size_t(horizon) * r);
    for (int i = 0; i < horizon; ++i)
        for (int j = 0; j < r; ++j)
            robot[size_t(i) * r + j] =
                unnormalize(normalized[size_t(i) * width + j], contract.actions.q01[j],
                                contract.actions.q99[j]);
    return robot;
}
std::string validate(Runtime &model, const fs::path &bundle, const fs::path &prepared,
                     const Contract *contract, const ParityGate &gate) {
    auto manifest = YAML::LoadFile((prepared / "manifest.json").string());
    require(fs::canonical(manifest["bundle"].as<std::string>()) == fs::canonical(bundle) &&
                kaivuri::sha256_file(bundle / "bundle.json") ==
                    manifest["bundle_sha256"].as<std::string>() &&
                kaivuri::sha256_file(bundle / "MANIFEST.sha256") ==
                    manifest["source_manifest_sha256"].as<std::string>(),
            "prepared fixtures belong to another bundle");
    require(value_or(manifest["policy"], std::string("pi05_libero")) == model.policy_name(),
            "prepared fixtures are for another policy");
    const bool state = contract != nullptr;
    if (state)
        require(manifest["fixtures"].size() >= 1, "pi05_openpi_state fixtures required");
    else
        require(manifest["fixtures"].size() == 5, "all five OpenPI fixtures required");
    auto prep = state ? preprocess_pil : preprocess;
    std::ostringstream report;
    const char *separator = "";
    report << "[";
    for (auto entry : manifest["fixtures"]) {
        std::string name = entry["directory"].as<std::string>();
        require(kaivuri::sha256_file(bundle / "fixtures" / (name + ".npz")) ==
                    entry["source_sha256"].as<std::string>(),
                "original fixture changed");
        std::map<std::string, std::unique_ptr<Npy>> arrays;
        for (auto file : entry["files"])
            arrays[file.first.as<std::string>()] = std::make_unique<Npy>(
                prepared / name / file.first.as<std::string>(), file.second.as<std::string>());
        auto &tokens = *arrays.at("tokens.npy"), &mask = *arrays.at("mask.npy");
        require(tokens.dtype == "<i8" && mask.dtype == "|b1" && tokens.count == mask.count,
                "fixture token contract");
        std::vector<int64_t> valid;
        for (size_t i = 0; i < tokens.count; ++i)
            if (mask.data[i])
                valid.push_back(reinterpret_cast<const int64_t *>(tokens.data)[i]);
        std::ostringstream extra;
        if (state) {
            // The board composes the prompt from the raw state; it must equal the tokenizer's,
            // and its float64 normalization must equal OpenPI's exactly.
            auto values = floats(*arrays.at("state.npy"));
            auto &normalized = *arrays.at("state_normalized.npy");
            require(normalized.dtype == "<f8" && normalized.count == values.size(),
                    "fixture normalized state");
            auto mine = normalize(values, contract->state);
            require(std::memcmp(mine.data(), normalized.data, normalized.bytes) == 0,
                    "state normalization differs from OpenPI: " + name);
            const auto task = entry["task"].as<std::string>();
            require(contract->tokens(task, values) == valid,
                    "board-composed state prompt differs from the tokenizer's: " + name);
            extra << ",\"task\":" << kaivuri::json_quote(task) << ",\"bins\":\""
                  << join(contract->bins(values), ' ') << "\",\"prompt_tokens\":" << valid.size()
                  << ",\"prompt_matches_tokenizer\":true";
        }
        const uint64_t prompt_start = monotonic_ns();
        model.set_prompt(valid); // first fixture also primes and captures the CUDA graph
        extra << ",\"set_prompt_ms\":" << (monotonic_ns() - prompt_start) / 1e6;
        auto &noise = *arrays.at("noise.npy");
        require(noise.dtype == "<f4" && noise.count == size_t(model.noise_count()),
                "fixture noise contract");
        auto &image0 = *arrays.at("image0.npy"), &image1 = *arrays.at("image1.npy");
        require(image0.dtype == "<f2" && image1.dtype == "<f2" && image0.count == 3 * plane &&
                    image1.count == 3 * plane,
                "fixture image contract");
        Timing timing;
        auto actions = model.infer(image0.data, image1.data,
                                   reinterpret_cast<const float *>(noise.data), timing);
        auto &reference = *arrays.at("reference.npy"),
             &mixed_reference = *arrays.at("mixed_reference.npy");
        auto fp16 = compare(actions, reference);
        auto mixed = compare(actions, mixed_reference);
        require(fp16.finite && mixed.finite && fp16.cosine >= gate.cosine &&
                    fp16.max_abs <= gate.max_abs,
                "C++ action parity failed: " + name + " " + comparison_json(fp16));
        if (state) {
            auto &robot = *arrays.at("robot_reference.npy");
            const int h = model.action_horizon();
            require(robot.dtype == "<f8" && robot.count == size_t(h) * contract->robot_dim,
                    "robot reference shape");
            auto mine = robot_chunk(floats(reference), h, model.action_dim(), *contract);
            auto expected = reinterpret_cast<const double *>(robot.data);
            double worst = 0;
            for (size_t k = 0; k < mine.size(); ++k)
                worst = std::max(worst, std::abs(mine[k] - expected[k]));
            require(worst <= 1e-12, "action unnormalization differs from OpenPI: " + name);
            extra << ",\"unnormalize_max_abs\":" << worst;
        }
        size_t different = 0;
        bool gray_identical = true;
        for (int i = 0; i < 2; ++i) {
            auto &raw = *arrays.at("raw" + std::to_string(i) + ".npy");
            auto &expected = *arrays.at("image" + std::to_string(i) + ".npy");
            const int channels = raw.shape.size() == 3 ? int(raw.shape[2]) : 1;
            require(raw.dtype == "|u1" && (raw.shape.size() == 2 || raw.shape.size() == 3) &&
                        (channels == 1 || channels == 3),
                    "fixture raw image");
            const int h = raw.shape[0], w = raw.shape[1];
            auto image = prep(raw.data, w, h, w * channels, channels);
            auto ref = reinterpret_cast<const uint16_t *>(expected.data);
            for (size_t j = 0; j < image.size(); ++j) {
                uint16_t bits;
                std::memcpy(&bits, &image[j], 2);
                different += bits != ref[j];
            }
            if (state && channels == 3) {
                // Live IR is one Y8 channel; with equal channels the gray path must match.
                std::vector<uint8_t> gray(size_t(h) * w);
                bool equal = true;
                for (size_t p = 0; p < gray.size(); ++p) {
                    gray[p] = raw.data[p * 3];
                    equal &= raw.data[p * 3 + 1] == gray[p] && raw.data[p * 3 + 2] == gray[p];
                }
                require(equal, "fixture raw image is not replicated gray IR");
                auto from_gray = prep(gray.data(), w, h, w, 1);
                gray_identical &=
                    std::memcmp(from_gray.data(), image.data(), image.size() * 2) == 0;
            }
        }
        require(different == 0, "C++ OpenPI image preprocessing differs: " + name + " (" +
                                    std::to_string(different) + " pixels)");
        require(gray_identical, "gray Y8 preprocessing differs from its 3-channel copy: " + name);
        report << separator << "{\"fixture\":" << kaivuri::json_quote(name)
               << ",\"fp16\":" << comparison_json(fp16) << ",\"mixed\":" << comparison_json(mixed)
               << ",\"preprocess_different_elements\":" << different << extra.str() << '}';
        separator = ",";
        std::cerr << "[parity] " << name << " PASS " << comparison_json(fp16) << '\n';
    }
    // Mode C: the exporter's clean-prefix references (rtc/rtc_fixture_<i>.npz, unpacked by
    // prepare_pi05.py), each on its source fixture's images and prompt. The same noise runs
    // with RTC off (the per-row engines with every row on the step's style) and with the
    // prefix; the prefix rows must come back bit-exact as [anchor, zeros].
    if (model.row_time()) {
        const YAML::Node &prepared_manifest = manifest;
        auto rtc = prepared_manifest["rtc"];
        require(rtc && rtc.size() >= 1, "Mode C bundle without prepared RTC fixtures: re-run "
                                         "prepare_pi05.py with this checkout");
        auto load = [&](const std::string &directory, const YAML::Node &files) {
            std::map<std::string, std::unique_ptr<Npy>> arrays;
            for (auto file : files)
                arrays[file.first.as<std::string>()] = std::make_unique<Npy>(
                    prepared / directory / file.first.as<std::string>(),
                    file.second.as<std::string>());
            return arrays;
        };
        for (auto entry : rtc) {
            const auto name = entry["directory"].as<std::string>(),
                       source = entry["source"].as<std::string>();
            require(kaivuri::sha256_file(bundle / entry["file"].as<std::string>()) ==
                        entry["source_sha256"].as<std::string>(),
                    "original RTC fixture changed: " + name);
            YAML::Node source_entry;
            for (auto f : prepared_manifest["fixtures"])
                if (f["directory"].as<std::string>() == source)
                    source_entry.reset(static_cast<const YAML::Node &>(f));
            require(bool(source_entry), "RTC fixture " + name + " names a missing source " + source);
            auto src = load(source, source_entry["files"]);
            auto arrays = load(name, entry["files"]);
            const int rows = entry["rows"].as<int>(), dims = entry["dims"].as<int>();
            const int h = model.action_horizon(), w = model.action_dim();
            auto &noise = *arrays.at("noise.npy"), &prefix = *arrays.at("anchor.npy");
            require(noise.dtype == "<f4" && noise.count == size_t(h) * w,
                    "RTC fixture noise contract: " + name);
            require(prefix.dtype == "<f4" && prefix.shape == std::vector<int64_t>({rows, dims}),
                    "RTC fixture anchor contract: " + name);
            auto &tokens = *src.at("tokens.npy"), &mask = *src.at("mask.npy");
            std::vector<int64_t> valid;
            for (size_t i = 0; i < tokens.count; ++i)
                if (mask.data[i])
                    valid.push_back(reinterpret_cast<const int64_t *>(tokens.data)[i]);
            model.set_prompt(valid);
            auto &image0 = *src.at("image0.npy"), &image1 = *src.at("image1.npy");
            auto z = reinterpret_cast<const float *>(noise.data);
            Timing timing;
            auto off = model.infer(image0.data, image1.data, z, timing, 0, nullptr, 0);
            auto clean = model.infer(image0.data, image1.data, z, timing, rows,
                                     reinterpret_cast<const float *>(prefix.data), dims);
            auto c_off = compare(off, *arrays.at("off.npy"));
            auto c_clean = compare(clean, *arrays.at("clean.npy"));
            require(c_off.finite && c_off.cosine >= gate.cosine && c_off.max_abs <= gate.max_abs,
                    "C++ Mode C parity (RTC off) failed: " + name + " " + comparison_json(c_off));
            require(c_clean.finite && c_clean.cosine >= gate.cosine &&
                        c_clean.max_abs <= gate.max_abs,
                    "C++ Mode C parity (prefix) failed: " + name + " " + comparison_json(c_clean));
            auto anchor_values = reinterpret_cast<const float *>(prefix.data);
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < w; ++c) {
                    const float want = c < dims ? anchor_values[r * dims + c] : 0.f;
                    require(std::memcmp(&clean[size_t(r) * w + c], &want, 4) == 0,
                            "Mode C prefix row " + std::to_string(r) + " is not the committed "
                            "action [anchor, zeros]: " + name);
                }
            report << separator << "{\"fixture\":" << kaivuri::json_quote(name)
                   << ",\"mode\":\"C\",\"source\":" << kaivuri::json_quote(source)
                   << ",\"rows\":" << rows << ",\"fp16\":" << comparison_json(c_clean)
                   << ",\"off\":" << comparison_json(c_off) << ",\"prefix_exact\":true}";
            separator = ",";
            std::cerr << "[parity] " << name << " Mode C d=" << rows << " PASS "
                      << comparison_json(c_clean) << " off " << comparison_json(c_off) << '\n';
        }
    }
    report << ']';
    return report.str();
}
// norm_stats.json must be the bundle's own (listed in MANIFEST.sha256).
YAML::Node bundle_norm_stats(const fs::path &bundle) {
    std::ifstream manifest(bundle / "MANIFEST.sha256");
    require(bool(manifest), "bundle lacks MANIFEST.sha256");
    std::string digest, name;
    while (manifest >> digest >> name)
        if (name == "norm_stats.json" || name == "*norm_stats.json") {
            require(kaivuri::sha256_file(bundle / "norm_stats.json") == digest,
                    "norm_stats.json checksum");
            return YAML::LoadFile((bundle / "norm_stats.json").string());
        }
    throw std::runtime_error("norm_stats.json missing from MANIFEST.sha256");
}
} // namespace kaivuri::pi05
