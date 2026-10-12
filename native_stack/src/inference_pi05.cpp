// OpenPI pi05_libero bench runtime. The device chain follows the validated
// jetson-orin-nano-vla/bench/vendor/pi05_trt.py implementation.
// This executable has no actuator IPC or I2C path: model actions are recorded only.
#include "kaivuri/frame_store.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <openssl/evp.h>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <yaml-cpp/yaml.h>
namespace fs = std::filesystem;
using namespace nvinfer1;
static volatile sig_atomic_t stopped = 0;
static void stop_signal(int) {
    stopped = 1;
}
static uint64_t mono_ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
static void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}
static void ck(cudaError_t code, const char *operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
static std::string hash_bytes(const void *data, size_t bytes) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned length = 0;
    require(EVP_Digest(data, bytes, digest, &length, EVP_sha256(), nullptr) == 1, "SHA256 failed");
    std::ostringstream out;
    for (unsigned i = 0; i < length; ++i)
        out << std::hex << std::setw(2) << std::setfill('0') << unsigned(digest[i]);
    return out.str();
}
static std::vector<char> read_file(const fs::path &path) {
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
static std::string hash_file(const fs::path &path) {
    auto bytes = read_file(path);
    return hash_bytes(bytes.data(), bytes.size());
}
struct Npy {
    std::shared_ptr<void> mapping;
    size_t file_bytes = 0, bytes = 0, count = 1;
    const uint8_t *data = nullptr;
    std::string dtype;
    std::vector<int64_t> shape;
    explicit Npy(const fs::path &path, const std::string &expected_sha = "") {
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        require(fd >= 0, "cannot open array " + path.string());
        struct stat st{};
        if (fstat(fd, &st) || st.st_size < 10) {
            close(fd);
            throw std::runtime_error("invalid array " + path.string());
        }
        file_bytes = size_t(st.st_size);
        void *address = mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
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
                      : dtype == "<i8"                     ? 8
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
            require(hash_bytes(p, file_bytes) == expected_sha,
                    "array checksum mismatch: " + path.string());
    }
};
static size_t item_size(DataType type) {
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
    int vision_layers, language_layers, action_layers, horizon, width, steps, prefix_tokens;
    int style_sites;
    size_t style_row_bytes = 0, prompt_bytes = 0;
    std::vector<int64_t> current_tokens;
    bool graph_enabled, capturing = false;
    std::string euler;
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
    void upload(Buffer *buffer, const void *data, size_t size) {
        require(size == buffer->bytes, "upload size mismatch");
        std::memcpy(buffer->staging(), data, size);
        ck(cudaMemcpyAsync(buffer->ptr, buffer->host, size, cudaMemcpyHostToDevice, stream),
           "upload");
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
        std::ostringstream out;
        out << prefix << '_' << std::setfill('0') << std::setw(2) << n;
        return out.str();
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
        for (int i = 0; i < language_layers; ++i)
            run(layer("language", i), {{"hidden", prefix[i % 2]->ptr},
                                       {"output", prefix[1 - i % 2]->ptr},
                                       {"mask", p_mask->ptr},
                                       {"cos", p_cos->ptr},
                                       {"sin", p_sin->ptr},
                                       {"key", kv[i].first->ptr},
                                       {"value", kv[i].second->ptr}});
        record(2);
        for (int s = 0; s < steps; ++s) {
            run("action_input", {{"actions", actions[s % 2]->ptr}, {"hidden", expert[0]->ptr}});
            for (int i = 0; i < action_layers; ++i)
                run(layer("action", i),
                    {{"hidden", expert[i % 2]->ptr},
                     {"output", expert[1 - i % 2]->ptr},
                     {"mask", s_mask->ptr},
                     {"cos", s_cos->ptr},
                     {"sin", s_sin->ptr},
                     {"prefix_k", kv[i].first->ptr},
                     {"prefix_v", kv[i].second->ptr},
                     {"mod_in", offset(styles, (s * style_sites + 2 * i) * style_row_bytes)},
                     {"mod_post",
                      offset(styles, (s * style_sites + 2 * i + 1) * style_row_bytes)}});
            run("action_output", {{"hidden", expert[action_layers % 2]->ptr},
                                  {"mod_final", offset(styles, (s * style_sites + style_sites - 1) *
                                                                   style_row_bytes)},
                                  {"velocity", velocity->ptr}});
            run(euler, {{"a", actions[s % 2]->ptr},
                        {"b", velocity->ptr},
                        {"out", actions[(s + 1) % 2]->ptr}});
        }
        record(3);
    }

  public:
    explicit Pi05(const fs::path &bundle, const fs::path &cache, bool use_graph)
        : info(YAML::LoadFile((bundle / "bundle.json").string())),
          embedding(bundle / "embedding_fp16.npy", info["embedding_sha256"].as<std::string>()),
          rope_cos(bundle / "rope_cos_fp16.npy"), rope_sin(bundle / "rope_sin_fp16.npy"),
          conditioning(bundle / "conditioning_10step_fp16.npy"), graph_enabled(use_graph) {
        require(info["complete"].as<bool>() && info["policy"].as<std::string>() == "pi05_libero" &&
                    info["valid_camera_compute_only"].as<bool>(),
                "requires compact OpenPI pi05_libero bundle");
        vision_layers = info["vision_layers"].as<int>();
        language_layers = info["language_layers"].as<int>();
        action_layers = info["action_layers"].as<int>();
        horizon = info["action_horizon"].as<int>();
        width = info["action_dim"].as<int>();
        steps = info["num_steps"].as<int>();
        prefix_tokens = info["prefix_tokens"].as<int>();
        require(info["valid_cameras"].as<int>() == 2 && prefix_tokens >= 512 &&
                    action_layers == language_layers && steps > 0 && horizon > 0 && width > 0,
                "unsupported bundle dimensions");
        auto candidate = YAML::LoadFile((cache / "candidate.json").string());
        require(fs::canonical(candidate["bundle"].as<std::string>()) == fs::canonical(bundle),
                "engine cache belongs to another bundle");
        for (auto entry : candidate["template_sha256"])
            require(hash_file(bundle / "templates" / entry.first.as<std::string>()) ==
                        entry.second.as<std::string>(),
                    "template checksum mismatch");
        require(initLibNvInferPlugins(&logger, ""), "plugin registration failed");
        runtime.reset(createInferRuntime(logger));
        require(bool(runtime), "TensorRT runtime");
        runtime->setGpuAllocator(&allocator);
        std::map<std::string, std::vector<char>> plans;
        for (auto entry : candidate["engine_sha256"]) {
            auto name = entry.first.as<std::string>();
            auto bytes = read_file(cache / name);
            require(hash_bytes(bytes.data(), bytes.size()) == entry.second.as<std::string>(),
                    "engine checksum mismatch");
            plans[fs::path(name).stem().string()] = std::move(bytes);
        }
        size_t scratch_bytes = 0;
        for (auto component : info["components"]) {
            auto name = component["label"].as<std::string>(),
                 kind = component["kind"].as<std::string>();
            auto &plan = plans.at(kind);
            auto &item = engines[name];
            item.engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
            require(bool(item.engine), "cannot deserialize " + name);
            fs::path directory = bundle / "weights" / name;
            auto index = YAML::LoadFile((directory / "index.json").string());
            std::map<std::string, YAML::Node> by_name;
            for (size_t i = 0; i < index["weights"].size(); ++i) {
                YAML::Node wt = index["weights"][i];
                by_name[wt["name"].as<std::string>()] = wt;
            }
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
        for (auto &path : fs::directory_iterator(cache))
            if (path.path().filename().string().find("op_axpy_") == 0 &&
                path.path().extension() == ".engine") {
                require(euler.empty(), "ambiguous Euler op cache");
                euler = path.path().stem().string();
                auto bytes = read_file(path.path());
                engines[euler].engine.reset(
                    runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
                require(bool(engines[euler].engine), "cannot deserialize Euler op");
                scratch_bytes =
                    std::max(scratch_bytes, size_t(engines[euler].engine->getDeviceMemorySizeV2()));
            }
        require(!euler.empty(), "missing cached Euler op");
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
        require(images[0]->type == DataType::kHALF &&
                    images[0]->shape == std::vector<int64_t>({1, 3, 224, 224}) &&
                    prefix[0]->type == DataType::kHALF,
                "unsupported image/prefix types");
        features = for_tensor("vision_tail", "output");
        rows = make({1, prefix_tokens, prefix[0]->shape.back()}, DataType::kHALF);
        p_mask = for_tensor("language_00", "mask");
        p_cos = for_tensor("language_00", "cos");
        p_sin = for_tensor("language_00", "sin");
        s_mask = for_tensor("action_00", "mask");
        s_cos = for_tensor("action_00", "cos");
        s_sin = for_tensor("action_00", "sin");
        for (int i = 0; i < language_layers; ++i)
            kv.emplace_back(for_tensor("language_00", "key"), for_tensor("language_00", "value"));
        velocity = for_tensor("action_output", "velocity");
        require(conditioning.shape.size() == 3 && conditioning.shape[0] == steps &&
                    conditioning.shape[1] == 2 * action_layers + 1 && conditioning.dtype == "<f2",
                "conditioning contract");
        style_sites = conditioning.shape[1];
        style_row_bytes = conditioning.shape[2] * 2;
        styles = make(conditioning.shape, DataType::kHALF);
        upload(styles, conditioning.data, conditioning.bytes);
        ck(cudaStreamSynchronize(stream), "initial uploads");
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
    void set_prompt(const std::vector<int64_t> &tokens) {
        if (tokens == current_tokens)
            return;
        require(!tokens.empty() && 512 + tokens.size() <= size_t(prefix_tokens),
                "prompt exceeds compact prefix; re-export without truncation");
        require(embedding.shape.size() == 2 && embedding.shape[1] == prefix[0]->shape.back() &&
                    embedding.dtype == "<f2",
                "embedding contract");
        if (graph) {
            ck(cudaGraphExecDestroy(graph), "destroy old graph");
            graph = nullptr;
        }
        int hidden = embedding.shape[1], valid = 512 + tokens.size();
        std::vector<__half> prompt(size_t(prefix_tokens) * hidden, __float2half(0));
        for (size_t i = 0; i < tokens.size(); ++i) {
            require(tokens[i] >= 0 && tokens[i] < embedding.shape[0], "token outside embedding");
            auto source = reinterpret_cast<const __half *>(embedding.data) + tokens[i] * hidden;
            float scale = std::sqrt(float(hidden));
            for (int j = 0; j < hidden; ++j)
                prompt[i * hidden + j] = __float2half_rn(__half2float(source[j]) * scale);
        }
        upload(rows, prompt.data(), rows->bytes);
        prompt_bytes = size_t(prefix_tokens - 512) * hidden * 2;
        std::vector<float> mask(size_t(prefix_tokens) * prefix_tokens);
        constexpr float negative = -2.3819763e38f;
        for (int i = 0; i < prefix_tokens; ++i)
            for (int j = 0; j < prefix_tokens; ++j)
                mask[size_t(i) * prefix_tokens + j] = (i < valid && j < valid) ? 0.f : negative;
        upload(p_mask, mask.data(), mask.size() * 4);
        mask.resize(size_t(horizon) * (prefix_tokens + horizon));
        for (int i = 0; i < horizon; ++i)
            for (int j = 0; j < prefix_tokens + horizon; ++j)
                mask[size_t(i) * (prefix_tokens + horizon) + j] =
                    (j < valid || j >= prefix_tokens) ? 0.f : negative;
        upload(s_mask, mask.data(), mask.size() * 4);
        require(rope_cos.dtype == "<f2" && rope_sin.dtype == "<f2" &&
                    rope_cos.shape == rope_sin.shape && rope_cos.shape.size() == 2,
                "RoPE contract");
        auto rope = [&](Buffer *target, const Npy &table, bool suffix) {
            size_t row_bytes = size_t(table.shape[1]) * 2;
            int n = suffix ? horizon : prefix_tokens;
            std::vector<uint8_t> bytes(size_t(n) * row_bytes);
            for (int i = 0; i < n; ++i) {
                int row = suffix ? valid + i + 1 : std::min(i + 1, valid);
                require(row < table.shape[0], "RoPE position outside table");
                std::memcpy(bytes.data() + size_t(i) * row_bytes,
                            table.data + size_t(row) * row_bytes, row_bytes);
            }
            upload(target, bytes.data(), bytes.size());
        };
        rope(p_cos, rope_cos, false);
        rope(p_sin, rope_sin, false);
        rope(s_cos, rope_cos, true);
        rope(s_sin, rope_sin, true);
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
        current_tokens = tokens;
    }
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
        uint64_t start = mono_ns();
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
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, events[0], events[1]), "vision timing");
        timing.vision_ms = ms;
        ck(cudaEventElapsedTime(&ms, events[1], events[2]), "language timing");
        timing.language_ms = ms;
        ck(cudaEventElapsedTime(&ms, events[2], events[3]), "denoise timing");
        timing.denoise_ms = ms;
        timing.total_ms = (mono_ns() - start) / 1e6;
        auto begin = static_cast<float *>(result->host);
        return {begin, begin + noise_count()};
    }
};
struct Tap {
    int index;
    float weight;
};
static std::vector<std::vector<Tap>> taps(int input, int output) {
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
// OpenPI's separable antialiased linear resize, centered black padding, uint8
// round-to-even, [-1,1] normalization. Gray IR is replicated into three channels.
static std::vector<__half> preprocess(const uint8_t *input, int width, int height, int stride,
                                      int channels) {
    require(width > 0 && height > 0 && (channels == 1 || channels == 3) &&
                stride >= width * channels,
            "invalid image");
    double ratio = std::max(width / 224., height / 224.);
    int rh = int(height / ratio), rw = int(width / ratio);
    require(rh > 0 && rw > 0, "image aspect ratio");
    auto yt = taps(height, rh), xt = taps(width, rw);
    std::vector<float> vertical(size_t(rh) * width * channels);
    for (int y = 0; y < rh; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c) {
                float value = 0;
                for (auto tap : yt[y])
                    value +=
                        float(input[size_t(tap.index) * stride + x * channels + c]) * tap.weight;
                vertical[(size_t(y) * width + x) * channels + c] = value;
            }
    std::vector<__half> result(3 * 224 * 224, __float2half(-1));
    int top = (224 - rh) / 2, left = (224 - rw) / 2;
    for (int y = 0; y < rh; ++y)
        for (int x = 0; x < rw; ++x)
            for (int c = 0; c < channels; ++c) {
                float value = 0;
                for (auto tap : xt[x])
                    value += vertical[(size_t(y) * width + tap.index) * channels + c] * tap.weight;
                float pixel = std::clamp(std::nearbyint(value), 0.f, 255.f);
                __half normalized = __float2half_rn((pixel / 255.f) * 2.f - 1.f);
                if (channels == 1)
                    for (int ch = 0; ch < 3; ++ch)
                        result[ch * 224 * 224 + (y + top) * 224 + x + left] = normalized;
                else
                    result[c * 224 * 224 + (y + top) * 224 + x + left] = normalized;
            }
    return result;
}
struct Comparison {
    double cosine = 0, max_abs = 0, mean_abs = 0;
    bool finite = true;
};
static Comparison compare(const std::vector<float> &values, const Npy &reference) {
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
static std::string comparison_json(const Comparison &c) {
    std::ostringstream out;
    out << std::setprecision(12) << "{\"cosine\":" << c.cosine << ",\"max_abs\":" << c.max_abs
        << ",\"mean_abs\":" << c.mean_abs << '}';
    return out.str();
}
static std::string validate(Pi05 &model, const fs::path &bundle, const fs::path &prepared) {
    auto manifest = YAML::LoadFile((prepared / "manifest.json").string());
    require(fs::canonical(manifest["bundle"].as<std::string>()) == fs::canonical(bundle) &&
                hash_file(bundle / "bundle.json") == manifest["bundle_sha256"].as<std::string>() &&
                hash_file(bundle / "MANIFEST.sha256") ==
                    manifest["source_manifest_sha256"].as<std::string>(),
            "prepared fixtures belong to another bundle");
    std::ostringstream report;
    report << "[";
    bool first = true;
    require(manifest["fixtures"].size() == 5, "all five OpenPI fixtures required");
    for (auto entry : manifest["fixtures"]) {
        std::string name = entry["directory"].as<std::string>();
        require(hash_file(bundle / "fixtures" / (name + ".npz")) ==
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
        model.set_prompt(valid);
        auto &noise = *arrays.at("noise.npy");
        require(noise.dtype == "<f4" && noise.count == size_t(model.noise_count()),
                "fixture noise contract");
        auto &image0 = *arrays.at("image0.npy"), &image1 = *arrays.at("image1.npy");
        require(image0.dtype == "<f2" && image1.dtype == "<f2" && image0.count == 3 * 224 * 224 &&
                    image1.count == 3 * 224 * 224,
                "fixture image contract");
        Sample timing;
        auto actions = model.infer(image0.data, image1.data,
                                   reinterpret_cast<const float *>(noise.data), timing);
        auto fp16 = compare(actions, *arrays.at("reference.npy"));
        auto mixed = compare(actions, *arrays.at("mixed_reference.npy"));
        require(fp16.finite && mixed.finite && fp16.cosine >= .99999 && fp16.max_abs <= .005,
                "C++ action parity failed: " + name + " " + comparison_json(fp16));
        size_t different = 0;
        for (int i = 0; i < 2; ++i) {
            auto &raw = *arrays.at("raw" + std::to_string(i) + ".npy");
            auto &expected = *arrays.at("image" + std::to_string(i) + ".npy");
            require(raw.dtype == "|u1" && raw.shape.size() == 3 && raw.shape[2] == 3,
                    "fixture raw image");
            auto image = preprocess(raw.data, raw.shape[1], raw.shape[0], raw.shape[1] * 3, 3);
            auto ref = reinterpret_cast<const uint16_t *>(expected.data);
            for (size_t j = 0; j < image.size(); ++j) {
                uint16_t bits;
                std::memcpy(&bits, &image[j], 2);
                different += bits != ref[j];
            }
        }
        require(different == 0, "C++ OpenPI image preprocessing differs: " + name + " (" +
                                    std::to_string(different) + " pixels)");
        if (!first)
            report << ',';
        first = false;
        report << "{\"fixture\":" << std::quoted(name) << ",\"fp16\":" << comparison_json(fp16)
               << ",\"mixed\":" << comparison_json(mixed)
               << ",\"preprocess_different_elements\":" << different << '}';
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
        uint64_t deadline = mono_ns() + 1000000000ULL;
        while (!stopped && mono_ns() < deadline) {
            if (kaivuri_frame_read(handle, &metadata, pixels.data(), pixels.size()) == 1) {
                auto now = mono_ns();
                if (metadata.host_monotonic_ns && metadata.host_monotonic_ns <= now &&
                    now - metadata.host_monotonic_ns <= 200000000ULL)
                    return metadata;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error("camera source missing/stale for one second");
    }
};
int main(int argc, char **argv) {
    try {
        fs::path bundle, cache, fixtures, out;
        std::vector<fs::path> frames;
        std::string task = "pick up the red block";
        double duration = 60, rate_hz = 0;
        int warmup = 5;
        bool only = false, graph = true;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() {
                require(++i < argc, "missing argument " + arg);
                return std::string(argv[i]);
            };
            if (arg == "--bundle")
                bundle = value();
            else if (arg == "--cache")
                cache = value();
            else if (arg == "--fixtures")
                fixtures = value();
            else if (arg == "--out")
                out = value();
            else if (arg == "--camera-frame")
                frames.push_back(value());
            else if (arg == "--task")
                task = value();
            else if (arg == "--duration-s")
                duration = std::stod(value());
            else if (arg == "--rate-hz")
                rate_hz = std::stod(value());
            else if (arg == "--warmup")
                warmup = std::stoi(value());
            else if (arg == "--validate-only")
                only = true;
            else if (arg == "--no-graph")
                graph = false;
            else if (arg == "--help") {
                std::cout
                    << "kaivuri-inference --bundle OPENPI_BUNDLE --cache "
                       "VALIDATED_OPTIMIZED_CACHE\n"
                       "  --fixtures PREPARED_FIXTURES --out NEW_REPORT.json [--validate-only]\n"
                       "  [--camera-frame FIRST.frame --camera-frame SECOND.frame]\n"
                       "  [--task EXPORTED_TASK --duration-s 60 --warmup 5 --no-graph]\n"
                       "  [--rate-hz 2] (0 = unrestricted; wait before reading fresh frames)\n"
                       "Bench only: records normalized LIBERO actions, never drives robot "
                       "hardware.\n";
                return 0;
            } else
                throw std::runtime_error("unknown option " + arg);
        }
        require(!bundle.empty() && !cache.empty() && !fixtures.empty() && !out.empty(),
                "bundle, cache, fixtures and out required");
        require(duration > 0 && std::isfinite(duration) && warmup >= 0 &&
                    (only || frames.size() == 2),
                "positive duration and two camera frames required");
        require(std::isfinite(rate_hz) && rate_hz >= 0 && rate_hz <= 500 &&
                    (rate_hz == 0 || rate_hz >= .001),
                "rate must be 0 (unrestricted) or 0.001..500 Hz");
        require(!fs::exists(out), "report already exists; choose a new path");
        signal(SIGINT, stop_signal);
        signal(SIGTERM, stop_signal);
        uint64_t load_start = mono_ns();
        Pi05 model(bundle, cache, graph);
        double load_ms = (mono_ns() - load_start) / 1e6;
        std::string parity = validate(model, bundle, fixtures);
        if (!out.parent_path().empty())
            fs::create_directories(out.parent_path());
        std::ofstream summary(out);
        require(bool(summary), "cannot create report");
        summary << std::setprecision(12)
                << "{\"model\":\"pi05_libero\",\"runtime\":\"C++ TensorRT + AOT\","
                   "\"actuator_output\":false,\"cuda_graph\":"
                << (graph ? "true" : "false") << ",\"load_ms\":" << load_ms
                << ",\"action_horizon\":" << model.action_horizon()
                << ",\"action_dim\":" << model.action_dim()
                << ",\"denoising_steps\":" << model.denoising_steps() << ",\"parity\":" << parity;
        if (only) {
            summary << ",\"status\":\"PASS\"}\n";
            return 0;
        }
        model.set_prompt(model.task_tokens(task));
        FrameReader first(frames[0]), second(frames[1]);
        std::mt19937 random(42);
        std::normal_distribution<float> gaussian(0, 1);
        std::vector<float> noise(model.noise_count());
        fs::path csv_path = out;
        csv_path.replace_extension(".csv");
        std::ofstream csv(csv_path);
        require(bool(csv), "cannot create timing CSV");
        csv << "sample,start_ns,finish_ns,frame0_sequence,frame1_sequence,frame0_arrival_ns,frame1_"
               "arrival_ns,"
               "frame0_sensor_us,frame1_sensor_us,frame0_exposure_us,frame1_exposure_us,frame0_"
               "valid_bits,frame1_valid_bits,"
               "preprocess_ms,vision_ms,language_ms,denoise_ms,inference_ms,iteration_ms,"
               "observation_age_start_ms,observation_to_result_ms,camera_arrival_skew_ms,"
               "rate_wait_ms,rate_lateness_ms\n";
        std::vector<double> latencies, delays;
        int count = 0;
        uint64_t begin = 0;
        const uint64_t period_ns = rate_hz ? static_cast<uint64_t>(std::llround(1e9 / rate_hz)) : 0;
        uint64_t next_start = 0, rate_overruns = 0;
        std::vector<double> waits;
        std::vector<float> last_action;
        while (!stopped && (!begin || (mono_ns() - begin) / 1e9 < duration)) {
            const uint64_t wait_begin = mono_ns();
            if (next_start && wait_begin > next_start)
                ++rate_overruns;
            while (!stopped && next_start && mono_ns() < next_start) {
                // Bounded absolute sleeps remain responsive to shutdown, even at low rates.
                const uint64_t wake_ns = std::min<uint64_t>(next_start, mono_ns() + 50000000ULL);
                timespec wake{static_cast<time_t>(wake_ns / 1000000000ULL),
                              static_cast<long>(wake_ns % 1000000000ULL)};
                const int result = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr);
                require(result == 0 || result == EINTR, "inference pacing sleep failed");
            }
            if (stopped || (begin && (mono_ns() - begin) / 1e9 >= duration))
                break;
            uint64_t start = mono_ns();
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
            uint64_t oldest = std::min(a.host_monotonic_ns, b.host_monotonic_ns),
                     observed = mono_ns();
            require(observed >= oldest && observed - oldest <= 200000000ULL,
                    "oldest observation exceeds 200ms freshness limit");
            auto image0 = preprocess(first.pixels.data(), a.width, a.height, a.stride, 1);
            auto image1 = preprocess(second.pixels.data(), b.width, b.height, b.stride, 1);
            double preprocess_ms = (mono_ns() - observed) / 1e6;
            for (auto &value : noise)
                value = gaussian(random);
            Sample timing;
            last_action = model.infer(image0.data(), image1.data(), noise.data(), timing);
            for (auto value : last_action)
                require(std::isfinite(value), "nonfinite model action");
            uint64_t finish = mono_ns();
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
            double skew = std::abs(double(a.host_monotonic_ns) - double(b.host_monotonic_ns)) / 1e6;
            csv << std::setprecision(12) << latencies.size() << ',' << start << ',' << finish << ','
                << a.sequence << ',' << b.sequence << ',' << a.host_monotonic_ns << ','
                << b.host_monotonic_ns << ',' << a.sensor_timestamp_us << ','
                << b.sensor_timestamp_us << ',' << a.actual_exposure_us << ','
                << b.actual_exposure_us << ',' << a.valid_metadata << ',' << b.valid_metadata << ','
                << preprocess_ms << ',' << timing.vision_ms << ',' << timing.language_ms << ','
                << timing.denoise_ms << ',' << timing.total_ms << ',' << iteration << ','
                << (observed - oldest) / 1e6 << ',' << delay << ',' << skew << ',' << rate_wait_ms
                << ',' << rate_lateness_ms << '\n';
            require(bool(csv), "timing write failed");
        }
        require(!latencies.empty(), "no inference samples");
        auto percentile = [](std::vector<double> values, double p) {
            std::sort(values.begin(), values.end());
            double pos = p * (values.size() - 1);
            size_t lo = size_t(pos), hi = std::min(lo + 1, values.size() - 1);
            return values[lo] + (values[hi] - values[lo]) * (pos - lo);
        };
        summary << ",\"task\":" << std::quoted(task)
                << ",\"views\":2,\"input\":\"IR Y8 replicated to RGB; unfinetuned LIBERO bench\","
                   "\"samples\":"
                << latencies.size() << ",\"elapsed_s\":" << (mono_ns() - begin) / 1e9
                << ",\"rate_hz\":" << rate_hz << ",\"rate_overruns\":" << rate_overruns
                << ",\"rate_wait_p50_ms\":" << percentile(waits, .5)
                << ",\"inference_p50_ms\":" << percentile(latencies, .5)
                << ",\"inference_p95_ms\":" << percentile(latencies, .95)
                << ",\"inference_p99_ms\":" << percentile(latencies, .99)
                << ",\"inference_max_ms\":" << percentile(latencies, 1)
                << ",\"arrival_to_result_p50_ms\":" << percentile(delays, .5)
                << ",\"arrival_to_result_p99_ms\":" << percentile(delays, .99)
                << ",\"last_normalized_chunk\":[";
        for (size_t i = 0; i < last_action.size(); ++i) {
            if (i)
                summary << ',';
            summary << last_action[i];
        }
        summary << "],\"status\":\"PASS\"}\n";
        require(bool(summary), "summary write failed");
        std::cout << "PASS samples=" << latencies.size() << " p50_ms=" << percentile(latencies, .5)
                  << " p99_ms=" << percentile(latencies, .99) << " report=" << out << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "kaivuri-inference: " << e.what() << '\n';
        return 1;
    }
}
