// OpenPI pi0.5 TensorRT runtime: the pi05_libero bench bundle and the fine-tuned
// pi05_openpi_state excavator bundle (joint angles in a per-step prompt). The device chain
// follows the validated jetson-orin-nano-vla/bench/vendor/pi05_trt.py implementation.
// Without --live, model actions are recorded only. With --live (pi05_openpi_state only),
// chunks go to kaivuri-control as drive plans through pi05::Producer; no I2C path here.
#include "kaivuri/frame_store.hpp"
#include "kaivuri/pi05_policy.hpp"
#include "kaivuri/util.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
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
using namespace nvinfer1;
using kaivuri::monotonic_ns;
using kaivuri::require;
using kaivuri::stop_requested;

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
struct Sample {
    double vision_ms = 0, language_ms = 0, denoise_ms = 0, total_ms = 0;
};
// Prompt row element: FP16 as rounded, or widened for an FP32 residual stream.
void store(__half &target, __half value) {
    target = value;
}
void store(float &target, __half value) {
    target = __half2float(value);
}
class Pi05 {
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
            run("action_input", {{"actions", actions[s % 2]->ptr}, {"hidden", expert[0]->ptr}});
            for (int i = 0; i < action_layers; ++i)
                run(layer("action", i), {{"hidden", expert[i % 2]->ptr},
                                         {"output", expert[1 - i % 2]->ptr},
                                         {"mask", s_mask->ptr},
                                         {"cos", s_cos->ptr},
                                         {"sin", s_sin->ptr},
                                         {"prefix_k", kv[i].first->ptr},
                                         {"prefix_v", kv[i].second->ptr},
                                         {"mod_in", style(2 * i)},
                                         {"mod_post", style(2 * i + 1)}});
            run("action_output", {{"hidden", expert[action_layers % 2]->ptr},
                                  {"mod_final", style(style_sites - 1)},
                                  {"velocity", velocity->ptr}});
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

  public:
    explicit Pi05(const fs::path &bundle, const fs::path &cache, bool use_graph)
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
        std::cerr << "[load] shared scratch MiB=" << scratch_bytes / 1048576.0 << '\n';
    }
    ~Pi05() {
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
    std::vector<float> infer(const void *image0, const void *image1, const float *noise,
                             Sample &timing) {
        require(!current_tokens.empty(), "set prompt before inference");
        uint64_t start = monotonic_ns();
        upload(images[0], image0, images[0]->bytes);
        upload(images[1], image1, images[1]->bytes);
        upload(actions[0], noise, actions[0]->bytes);
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
// OpenPI Unnormalize of a normalized [horizon][width] chunk: the first robot_dim columns.
std::vector<double> robot_chunk(const std::vector<float> &normalized, int horizon, int width,
                                const kp::Contract &contract) {
    const int r = contract.robot_dim;
    std::vector<double> robot(size_t(horizon) * r);
    for (int i = 0; i < horizon; ++i)
        for (int j = 0; j < r; ++j)
            robot[size_t(i) * r + j] =
                kp::unnormalize(normalized[size_t(i) * width + j], contract.actions.q01[j],
                                contract.actions.q99[j]);
    return robot;
}
// Full normalized chunk vs the exporter's FP16 split reference. pi05_libero (AOT kernel in
// the exported order) keeps 0.99999 / 0.005. pi05_openpi_state on native TensorRT attention
// defaults to 0.9998 / 0.01: the exporter's own compact-vs-padded FP16 paths already differ
// by 0.99994 / 0.004, and GB10 TRT 10.14 measured 0.99990 / 0.0070 worst on smoke9.
struct StateGate {
    double cosine = .99999, max_abs = .005;
    bool overridden = false;
};
std::string validate(Pi05 &model, const fs::path &bundle, const fs::path &prepared,
                     const kp::Contract *contract, const StateGate &gate) {
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
            auto mine = kp::normalize(values, contract->state);
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
        Sample timing;
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
    report << ']';
    return report.str();
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
        StateGate gate;
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
        Pi05 model(bundle, cache, graph);
        double load_ms = (monotonic_ns() - load_start) / 1e6;
        std::unique_ptr<kp::Contract> contract;
        if (model.policy_name() == "pi05_openpi_state") {
            contract = std::make_unique<kp::Contract>(
                kp::load_contract(model.bundle_info(), bundle_norm_stats(bundle)));
            require(contract->prompt.longest(contract->state_dim) + 512 <=
                        size_t(model.prefix_size()),
                    "longest state prompt exceeds the compact prefix");
        }
        require(contract || !gate.overridden, "parity overrides apply to pi05_openpi_state only");
        if (contract && !gate.overridden)
            gate = {.9998, .01, false};
        require(!drive || contract, "--live requires a pi05_openpi_state bundle");
        require(!drive || contract->fps > 0, "--live requires kaivuri.fps in bundle.json");
        std::string parity = validate(model, bundle, fixtures, contract.get(), gate);
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
            auto prep = contract ? preprocess_pil : preprocess;
            auto image0 = prep(first.pixels.data(), a.width, a.height, a.stride, 1);
            auto image1 = prep(second.pixels.data(), b.width, b.height, b.stride, 1);
            double preprocess_ms = (monotonic_ns() - observed) / 1e6;
            for (auto &value : noise)
                value = gaussian(random);
            Sample timing;
            last_action = model.infer(image0.data(), image1.data(), noise.data(), timing);
            for (auto value : last_action)
                require(std::isfinite(value), "nonfinite model action");
            kp::Plan plan;
            int clamped = 0;
            bool sent = false;
            if (contract) {
                const int h = model.action_horizon(), r = contract->robot_dim;
                last_robot = robot_chunk(last_action, h, model.action_dim(), *contract);
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
