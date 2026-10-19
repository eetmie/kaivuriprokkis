#pragma once
#include "kaivuri/core.hpp"
#include "kaivuri/mlp_excitation.hpp"
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace kaivuri::mlp {
struct DriveSample {
    ImuSample imu{};
    Commands manual{}, excitation{}, command{};
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
    std::vector<std::pair<std::string, std::string>> files;
};
std::string sha256_file(const std::filesystem::path &);
std::string json_quote(const std::string &);

// One producer, one writer. No mutex, allocation or disk operation in push().
class MlpWriter {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    MlpWriter(const std::filesystem::path &out, RecordingMetadata metadata, size_t capacity = 4096);
    ~MlpWriter();
    MlpWriter(const MlpWriter &) = delete;
    MlpWriter &operator=(const MlpWriter &) = delete;
    bool start();
    bool rotate();
    bool push(const Record &);
    void finish(bool complete = true);
    void shutdown();
    WriterState state() const;
    uint64_t dropped() const;
    size_t high_water() const;
};
} // namespace kaivuri::mlp
