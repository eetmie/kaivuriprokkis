#include "kaivuri/flow_rtc_cuda.hpp"
#include <limits>

namespace kaivuri::flow {
namespace {
__global__ void correct(const float *actions, const float *next, const float *strength,
                        float *out, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float s = strength[i], x = actions[i], y = next[i];
        // Explicit endpoint selections preserve frozen values and no-RTC parity exactly.
        out[i] = s == 0.f ? x : s == 1.f ? y : x + s * (y - x);
    }
}
} // namespace
cudaError_t groot_post_step_cuda(const float *actions, const float *next, const float *strength,
                                float *out, size_t count, cudaStream_t stream) {
    if (!actions || !next || !strength || !out || !count ||
        count > size_t(std::numeric_limits<int>::max())) return cudaErrorInvalidValue;
    correct<<<unsigned((count + 255) / 256), 256, 0, stream>>>(actions, next, strength, out, count);
    return cudaGetLastError();
}
} // namespace kaivuri::flow
