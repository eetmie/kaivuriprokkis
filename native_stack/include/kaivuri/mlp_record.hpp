#pragma once
#include "kaivuri/core.hpp"
#include "kaivuri/mlp_excitation.hpp"
#include "kaivuri/util.hpp"
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace kaivuri::mlp {
enum class DriveSource { Manual, Controller, Idle };
const char *drive_source_name(DriveSource);
struct DriveSample {
    ImuSample imu{};
    Commands manual{}, excitation{}, command{};
    // Optional controller contribution and correction applied after combination.
    // Neither contains controller logic; both default to zero for manual takes.
    Commands assist{}, constraint{};
    std::array<double, 4> angles{};
    std::array<double, 16> pulses{};
    uint64_t tick = 0, scheduled_ns = 0, start_ns = 0, post_write_ns = 0;
    uint64_t missed_periods = 0, checksum_errors = 0, invalid_frames = 0, resets = 0;
    uint64_t block_id = 0, noise_tick = 0;
    uint32_t seed = 0;
    unsigned target = 14;
    Waveform waveform = Waveform::Sine;
    Stage stage = Stage::Off;
    double command_age_s = 0, state_age_s = 0, excitation_elapsed_s = 0;
    double gyro_range_dps = 0, accel_range_g = 0;
    bool input_valid = false, imu_valid = false, write_ok = false, pump_on = false, enabled = false;
    bool controller_valid = true;
    DriveSource source = DriveSource::Manual;
};
enum class RecordKind { Start, Raw, Drive, Block, Finish };
struct Record {
    RecordKind kind = RecordKind::Raw;
    DriveSample sample{};
    ExcitationBlock block{};
    bool complete = true;
    uint64_t transition = 0;
    uint64_t drops_at_start = 0;
};
enum class WriterState { Idle, Opening, Recording, Closing, Failed };
struct RecordingMetadata {
    Config config{};
    bool dry = false, slew = false, tracks = false;
    std::string suffix, source_revision, executable_sha256, options_json, collection_id;
    std::string collector = "kaivuri-mlp-collect";
    std::vector<std::pair<std::string, std::string>> files;
};

// Optional sidecar owner. Callbacks run only on the writer thread; paths use
// .partial suffixes. The shared writer hashes/syncs/commits every attachment
// before the final session manifest. The base training trio stays unchanged.
class RecordingAttachment {
  public:
    virtual ~RecordingAttachment() = default;
    virtual void begin(const std::filesystem::path &directory, const std::string &stem) = 0;
    virtual void drive(const DriveSample &) = 0;
    virtual void close() = 0;
    virtual void abort() noexcept = 0;
    virtual std::vector<std::filesystem::path> paths() const = 0;
};

// One producer, one writer. No mutex, allocation or disk operation in push().
class MlpWriter {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    MlpWriter(const std::filesystem::path &out, RecordingMetadata metadata, size_t capacity = 4096,
              std::shared_ptr<RecordingAttachment> attachment = {});
    ~MlpWriter();
    MlpWriter(const MlpWriter &) = delete;
    MlpWriter &operator=(const MlpWriter &) = delete;
    bool start(); // Also accepts a start queued behind a closing take.
    bool rotate();
    bool push(const Record &);
    void finish(bool complete = true);
    void shutdown();
    WriterState state() const;
    uint64_t dropped() const;
    size_t high_water() const;
};

// Shared raw-packet capture/epoch rotation. Application operator/PID lifecycles
// remain in their own executables.
struct RawCapture {
    MlpWriter *writer;
    ImuParser *parser;
    bool recording = false, reset = false, failed = false;
    uint64_t epoch = 0, last_raw = 0;
    void packet(const ImuSample &);
    static void sink(void *self, const ImuSample &sample) {
        static_cast<RawCapture *>(self)->packet(sample);
    }
};
} // namespace kaivuri::mlp
