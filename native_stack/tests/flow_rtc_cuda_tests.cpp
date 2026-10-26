#include "kaivuri/flow_rtc.hpp"
#include "kaivuri/flow_rtc_cuda.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace kaivuri::flow;
static void ck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
int main() {
    float *device = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    try {
        constexpr size_t n = 40 * 132, bytes = n * sizeof(float);
        ck(cudaStreamCreate(&stream));
        ck(cudaMalloc(reinterpret_cast<void **>(&device), 4 * bytes));
        float *x = device, *next = device + n, *strength = device + 2 * n, *out = device + 3 * n;
        Tensor host_x(n), host_v(n), host_strength(n);
        for (size_t i = 0; i < n; ++i) {
            host_x[i] = float(int(i % 31) - 15) / 13;
            host_v[i] = float(int(i % 19) - 9) / 17;
            host_strength[i] = float(i % 9) / 8;
        }
        auto host_next = euler(host_x, host_v, .25f);
        auto expected = groot_post_step(host_x, host_next, host_strength);
        auto direct = euler(host_x, host_v, .25f, &host_strength);
        ck(cudaMemcpyAsync(x, host_x.data(), bytes, cudaMemcpyHostToDevice, stream));
        ck(cudaMemcpyAsync(next, host_next.data(), bytes, cudaMemcpyHostToDevice, stream));
        ck(cudaMemcpyAsync(strength, host_strength.data(), bytes, cudaMemcpyHostToDevice, stream));
        ck(groot_post_step_cuda(x, next, strength, out, n, stream));
        ck(cudaStreamSynchronize(stream));
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        ck(groot_post_step_cuda(x, next, strength, out, n, stream));
        ck(cudaStreamEndCapture(stream, &graph));
        ck(cudaGraphInstantiate(&executable, graph, 0));
        ck(cudaGraphLaunch(executable, stream));
        Tensor result(n);
        ck(cudaMemcpyAsync(result.data(), out, bytes, cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));
        float maximum = 0;
        for (size_t i = 0; i < n; ++i) {
            if (result[i] != expected[i]) throw std::runtime_error("CUDA/CPU post-step mismatch");
            maximum = std::max(maximum, std::abs(result[i] - direct[i]));
        }
        if (maximum > 2e-7f) throw std::runtime_error("post-step correction drift");
        // Alias and no-RTC path must retain exactly the unmodified engine output.
        std::fill(host_strength.begin(), host_strength.end(), 1.f);
        ck(cudaMemcpyAsync(strength, host_strength.data(), bytes, cudaMemcpyHostToDevice, stream));
        ck(groot_post_step_cuda(x, next, strength, next, n, stream));
        ck(cudaMemcpyAsync(result.data(), next, bytes, cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));
        if (result != host_next) throw std::runtime_error("no-RTC engine output changed");
        std::cout << n << " CUDA elements, graph capture, alias, no-RTC parity passed; "
                  << "max post-step vs direct Euler error " << maximum << '\n';
        cudaGraphExecDestroy(executable); cudaGraphDestroy(graph);
        cudaFree(device); cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (device) cudaFree(device);
        if (stream) cudaStreamDestroy(stream);
        return 1;
    }
}
