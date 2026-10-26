#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>

namespace kaivuri::flow {
// All arrays are FP32 DEVICE pointers of count elements. Strength is in [0,1].
// out may alias actions or unmasked_next (pointwise); no allocations or synchronization.
// In a captured graph keep addresses stable, upload the seed/strength before launch.
// The caller validates shapes, dtypes, values, task epoch and timing on the host.
cudaError_t groot_post_step_cuda(const float *actions, const float *unmasked_next,
                                const float *strength, float *out, size_t count,
                                cudaStream_t stream);
} // namespace kaivuri::flow
