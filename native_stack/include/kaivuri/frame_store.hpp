#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
// ABI v1: SDK timestamp in milliseconds; raw hardware metadata in microseconds.
struct FrameMetadata {
    uint64_t sequence, host_monotonic_ns, sensor_timestamp_us, frame_timestamp_us;
    uint64_t frame_counter, actual_exposure_us, gain_level, valid_metadata;
    double sdk_timestamp_ms;
    uint32_t timestamp_domain, width, height, stride, ir_index, pixel_bytes;
};
static_assert(sizeof(FrameMetadata) == 96, "Frame metadata ABI v1 requires 96 bytes");
// Metadata bits: 0 sensor timestamp, 1 frame timestamp, 2 counter, 3 exposure, 4 gain.
extern "C" {
void *kaivuri_frame_open(const char *path);
// 1=snapshot, 0=no frame/busy, -1=bad arguments/capacity. Null pixels reads metadata only.
int kaivuri_frame_read(void *handle, FrameMetadata *metadata, void *pixels, size_t capacity);
void kaivuri_frame_close(void *handle);
}
namespace kaivuri {
class FrameWriter {
    void *mapping_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1, lock_fd_ = -1;

  public:
    FrameWriter(const std::string &path, uint32_t width, uint32_t height, uint32_t ir_index);
    ~FrameWriter();
    FrameWriter(const FrameWriter &) = delete;
    FrameWriter &operator=(const FrameWriter &) = delete;
    bool publish(const FrameMetadata &metadata, const void *pixels);
};
} // namespace kaivuri
