// Native teleoperation and bounded, lossless episode spool. No encoder or Python in this process.
#include "kaivuri/core.hpp"
#include "kaivuri/frame_store.hpp"
#include "kaivuri/gamepad.hpp"
#include "kaivuri/util.hpp"
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <linux/input.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <zstd.h>
using namespace kaivuri;
namespace fs = std::filesystem;

namespace {
struct Control {
    Fd fd;
    uint64_t sequence = 0;
    explicit Control(const std::string &path)
        : fd(socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0)) {
        require(bool(fd), "control socket failed");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        require(path.size() < sizeof(address.sun_path), "socket path too long");
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        require(connect(fd.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0,
                "cannot connect to controller");
    }
    void send(Operation op, const Commands &action = {}, uint32_t flags = 0) {
        std::array<uint8_t, 76> data{};
        std::memcpy(data.data(), "KVR1", 4);
        auto put = [&](size_t offset, auto value) {
            std::memcpy(data.data() + offset, &value, sizeof(value));
        };
        put(4, uint16_t(1));
        put(6, uint16_t(op));
        put(8, flags);
        put(12, ++sequence);
        put(20, monotonic_ns());
        const bool drive = op == Operation::Drive;
        put(28, uint32_t(drive));
        put(32, float(100));
        if (drive)
            for (size_t i = 0; i < 6; ++i)
                put(52 + i * 4, float(action[i]));
        const size_t size = drive ? 76 : 52;
        require(::send(fd.get(), data.data(), size, MSG_DONTWAIT | MSG_NOSIGNAL) == ssize_t(size),
                "control send failed");
    }
    std::string status() {
        send(Operation::Status);
        pollfd p{fd.get(), POLLIN, 0};
        int ready;
        do {
            ready = poll(&p, 1, 100);
        } while (ready < 0 && errno == EINTR);
        require(ready == 1, "controller status timeout");
        char bytes[8192];
        ssize_t count;
        do {
            count = recv(fd.get(), bytes, sizeof(bytes), MSG_TRUNC);
        } while (count < 0 && errno == EINTR);
        require(count > 0 && count <= ssize_t(sizeof(bytes)), "invalid/truncated status");
        return std::string(bytes, count);
    }
    ~Control() {
        try {
            send(Operation::Disarm);
            status(); // Drain acknowledgement before closing the owning connection.
        } catch (...) {
        }
    }
};

// Latest frame of one camera store; pixels are re-read only when the sequence advances.
struct Image {
    void *handle;
    FrameMetadata m{};
    std::vector<uint8_t> pixels;
    explicit Image(const std::string &path) : handle(kaivuri_frame_open(path.c_str())) {
        require(handle, "cannot open camera store: " + path);
    }
    Image(const Image &) = delete;
    Image &operator=(const Image &) = delete;
    ~Image() { kaivuri_frame_close(handle); }
    bool read() {
        FrameMetadata info{};
        if (kaivuri_frame_read(handle, &info, nullptr, 0) != 1)
            return false;
        require(info.pixel_bytes && info.pixel_bytes <= 16 * 1024 * 1024, "invalid camera size");
        require(pixels.empty() || info.pixel_bytes == pixels.size(), "camera size changed");
        if (info.sequence == m.sequence && !pixels.empty())
            return true;
        pixels.resize(info.pixel_bytes);
        const int r = kaivuri_frame_read(handle, &m, pixels.data(), pixels.size());
        require(r >= 0, "invalid camera frame");
        return r == 1;
    }
};

// Written straight into the sample row; the caller's stream already has precision 17.
void put_metadata(std::ostream &o, const FrameMetadata &m) {
    o << "{\"sequence\":" << m.sequence << ",\"host_monotonic_ns\":" << m.host_monotonic_ns
      << ",\"sensor_timestamp_us\":" << m.sensor_timestamp_us
      << ",\"frame_timestamp_us\":" << m.frame_timestamp_us
      << ",\"frame_counter\":" << m.frame_counter
      << ",\"actual_exposure_us\":" << m.actual_exposure_us << ",\"gain_level\":" << m.gain_level
      << ",\"valid_metadata\":" << m.valid_metadata
      << ",\"sdk_timestamp_ms\":" << m.sdk_timestamp_ms
      << ",\"timestamp_domain\":" << m.timestamp_domain << ",\"width\":" << m.width
      << ",\"height\":" << m.height << ",\"stride\":" << m.stride << ",\"ir_index\":" << m.ir_index
      << ",\"pixel_bytes\":" << m.pixel_bytes << "}";
}

std::string depth_file(int k) {
    return "cam" + std::to_string(k + 1) + ".depth.zst";
}

// JSON list of the cameras with depth, e.g. ["cam1","cam2"].
void put_depth_list(std::ostream &o, const std::array<bool, 2> &enabled) {
    o << '[';
    for (int k = 0, n = 0; k < 2; ++k)
        if (enabled[k])
            o << (n++ ? "," : "") << "\"cam" << k + 1 << "\"";
    o << ']';
}

void fsync_path(const fs::path &path, int flags, const std::string &what) {
    Fd fd(open(path.c_str(), O_RDONLY | O_CLOEXEC | flags));
    require(bool(fd), what + " sync open failed");
    require(fsync(fd.get()) == 0, what + " fsync failed");
}

struct Record {
    std::string json;
    std::array<std::vector<uint8_t>, 2> images;
    std::array<std::vector<uint8_t>, 2> depth; // packed Z16, compressed on the writer thread
};

class Writer {
    fs::path path_;
    uint64_t episode_number_;
    std::ofstream rows_;
    std::array<std::ofstream, 2> cams_;
    // camN.depth.zst: one independent zstd frame per recorded frame (self-delimiting).
    std::array<bool, 2> depth_enabled_;
    std::array<std::ofstream, 2> depth_;
    std::array<uint64_t, 2> depth_written_{};
    std::vector<uint8_t> packed_;
    ZSTD_CCtx *zstd_ = nullptr;
    std::deque<Record> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    bool done_ = false;
    std::atomic<bool> failed_{false};
    size_t limit_;
    uint64_t written_ = 0;

    bool ok() const {
        for (int k = 0; k < 2; ++k)
            if (!cams_[k] || (depth_enabled_[k] && !depth_[k]))
                return false;
        return bool(rows_);
    }
    template <class F> void each_open(F f) {
        f(rows_);
        for (auto &c : cams_)
            f(c);
        for (auto &d : depth_)
            if (d.is_open())
                f(d);
    }
    void close_files() {
        each_open([](std::ofstream &f) { f.close(); });
    }
    void write(std::ofstream &f, const uint8_t *data, size_t bytes) {
        f.write(reinterpret_cast<const char *>(data), std::streamsize(bytes));
    }
    void run() {
        try {
            for (;;) {
                Record r;
                {
                    std::unique_lock<std::mutex> l(mutex_);
                    cv_.wait(l, [&] { return done_ || !queue_.empty(); });
                    if (queue_.empty())
                        break;
                    r = std::move(queue_.front());
                    queue_.pop_front();
                }
                if (!r.images[0].empty()) {
                    for (int k = 0; k < 2; ++k)
                        write(cams_[k], r.images[k].data(), r.images[k].size());
                    ++written_;
                    for (int k = 0; k < 2; ++k) {
                        if (!depth_enabled_[k])
                            continue;
                        const auto &z = r.depth[k];
                        packed_.resize(ZSTD_compressBound(z.size()));
                        const size_t bytes = ZSTD_compressCCtx(
                            zstd_, packed_.data(), packed_.size(), z.data(), z.size(), 1);
                        require(!z.empty() && !ZSTD_isError(bytes), "depth compression failed");
                        write(depth_[k], packed_.data(), bytes);
                        ++depth_written_[k];
                    }
                }
                rows_ << r.json << '\n';
                require(ok(), "episode disk write failed");
            }
            each_open([](std::ofstream &f) { f.flush(); });
            require(ok(), "episode flush failed");
        } catch (...) {
            failed_ = true;
        }
    }

  public:
    Writer(const fs::path &root, size_t limit, std::array<bool, 2> depth, uint64_t episode_number)
        : path_(root / ("episode-" + std::to_string(monotonic_ns()) + ".partial")),
          episode_number_(episode_number), depth_enabled_(depth), limit_(limit) {
        require(fs::create_directory(path_), "episode directory exists");
        rows_.open(path_ / "samples.jsonl");
        cams_[0].open(path_ / "cam1.y8", std::ios::binary);
        cams_[1].open(path_ / "cam2.y8", std::ios::binary);
        for (int k = 0; k < 2; ++k)
            if (depth_enabled_[k])
                depth_[k].open(path_ / depth_file(k), std::ios::binary);
        require(ok(), "cannot create episode files");
        if (depth_enabled_[0] || depth_enabled_[1]) {
            zstd_ = ZSTD_createCCtx();
            require(zstd_, "zstd context allocation failed");
        }
        thread_ = std::thread([this] { run(); });
    }
    std::string name() const { return std::to_string(episode_number_); }
    ~Writer() {
        join();
        ZSTD_freeCCtx(zstd_);
    }
    // Stop writing without committing: the take stays as .partial, which export skips.
    void abandon(const std::string &reason) {
        join();
        close_files();
        std::cerr << "Stopped ep " << name() << " (" << reason
                  << "); kept incomplete at " << path_ << std::endl;
    }
    bool push(Record r) {
        std::lock_guard<std::mutex> l(mutex_);
        if (failed_ || queue_.size() >= limit_)
            return false;
        queue_.push_back(std::move(r));
        cv_.notify_one();
        return true;
    }
    void join() {
        {
            std::lock_guard<std::mutex> l(mutex_);
            done_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable())
            thread_.join();
    }
    bool finish(bool save, uint64_t frames, uint64_t start, uint64_t end,
                const std::string &reason) {
        std::cout << "Stopping ep " << name() << " (" << reason << "): " << frames
                  << " frames; " << (save && frames ? "saving..." : "discarding...") << std::endl;
        join();
        close_files();
        for (int k = 0; k < 2; ++k)
            require(!depth_enabled_[k] || depth_written_[k] == frames,
                    "incomplete depth retained at " + path_.string());
        require(!failed_ && written_ == frames, "incomplete episode retained at " + path_.string());
        if (!save || !frames) {
            fs::remove_all(path_);
            std::cout << "Discarded ep " << name() << ": " << frames << " frames" << std::endl;
            return false;
        }
        std::ofstream summary(path_ / "episode.json");
        summary << "{\"episode_number\":" << episode_number_ << ",\"frames\":" << frames
                << ",\"start_ns\":" << start << ",\"end_ns\":" << end
                << ",\"reason\":" << json_quote(reason) << ",\"depth\":";
        put_depth_list(summary, depth_enabled_);
        summary << "}\n";
        summary.close();
        require(bool(summary), "episode summary write failed");
        std::vector<std::string> files{"samples.jsonl", "cam1.y8", "cam2.y8", "episode.json"};
        for (int k = 0; k < 2; ++k)
            if (depth_enabled_[k])
                files.push_back(depth_file(k));
        for (const auto &file : files)
            fsync_path(path_ / file, 0, "episode");
        fsync_path(path_, O_DIRECTORY, "episode directory");
        fs::path ready = path_;
        ready.replace_extension(".ready");
        fs::rename(path_, ready);
        fsync_path(ready.parent_path(), O_DIRECTORY, "session");
        std::cout << "Saved ep " << name() << ": " << frames << " frames -> " << ready << std::endl;
        return true;
    }
};

// Linux evdev uses X/Y labels BTN_NORTH/BTN_WEST, matching reference/modules/gamepad.py.
std::array<bool, 4> buttons(const Pad &pad) {
    return {pad.keys[BTN_SOUTH], pad.keys[BTN_EAST], pad.keys[BTN_NORTH], pad.keys[BTN_WEST]};
}

const char *usage =
    "kaivuri-record --root NEW_SESSION --task TEXT --gamepad /dev/input/eventN\n"
    "  --camera-file .../SERIAL1-ir1.frame --camera-file .../SERIAL2-ir2.frame\n"
    "  [--cam1-depth-file .../SERIAL1-depth.frame] [--cam2-depth-file ...]\n"
    "  (depth is paired to its IR frame by sensor timestamp; camN.depth.zst)\n"
    "  [--socket PATH --fps 30 --enable-slew --enable-tracks]\n"
    "  [--episode-offset N] (number of previously saved episodes; display numbering only)\n"
    "  [--max-episode-s 180 --queue-frames 64 --max-pair-skew-ms 40 --max-age-ms 200]\n"
    "Arms automatically and re-arms after controller faults; sticks drive at all times.\n"
    "A=start/save B=discard X=pump Y=re-arm now; Ctrl+C=save and exit.\n"
    "No automatic config reload: restart services after configuration changes.\n"
    "--scripted [--auto-record-s SEC [--auto-episodes N] [--auto-sine AMP]] "
    "reads stdin: arm/start/save/discard/pump/quit.\n"
    "--auto-sine drives each enabled joint with an AMP-amplitude sine while recording, so test "
    "datasets have non-constant actions.\n";
} // namespace

int main(int argc, char **argv) {
    try {
        std::string socket = (runtime_dir() / "control.sock").string(), gamepad, task, root,
                    revision;
        std::vector<std::string> camera_paths;
        std::array<std::string, 2> depth_paths;
        int fps = 30, queue = 64, auto_episodes = 1;
        uint64_t saved_episodes = 0;
        double max_episode = 180, auto_seconds = 0, max_skew_ms = 40, max_age_ms = 200;
        double auto_sine = 0;
        bool scripted = false, tracks = false, slew = false;
        for (Args args(argc, argv); args.next();) {
            if (args.is("--socket"))
                socket = args.value();
            else if (args.is("--root"))
                root = args.value();
            else if (args.is("--task"))
                task = args.value();
            else if (args.is("--gamepad"))
                gamepad = args.value();
            else if (args.is("--camera-file"))
                camera_paths.push_back(args.value());
            else if (args.is("--cam1-depth-file"))
                depth_paths[0] = args.value();
            else if (args.is("--cam2-depth-file"))
                depth_paths[1] = args.value();
            else if (args.is("--fps"))
                fps = args.integer();
            else if (args.is("--queue-frames"))
                queue = args.integer();
            else if (args.is("--episode-offset"))
                saved_episodes = args.natural();
            else if (args.is("--max-episode-s"))
                max_episode = args.number();
            else if (args.is("--auto-record-s"))
                auto_seconds = args.number();
            else if (args.is("--auto-episodes"))
                auto_episodes = args.integer();
            else if (args.is("--auto-sine"))
                auto_sine = args.number();
            else if (args.is("--max-pair-skew-ms"))
                max_skew_ms = args.number();
            else if (args.is("--max-age-ms"))
                max_age_ms = args.number();
            else if (args.is("--revision"))
                revision = args.value();
            else if (args.is("--scripted"))
                scripted = true;
            else if (args.is("--enable-tracks"))
                tracks = true;
            else if (args.is("--enable-slew"))
                slew = true;
            else if (args.is("--help")) {
                std::cout << usage;
                return 0;
            } else
                args.unknown();
        }
        require(!root.empty() && !task.empty() && camera_paths.size() == 2 &&
                    camera_paths[0] != camera_paths[1] && fps >= 1 && fps <= 90 && queue >= 2 &&
                    queue <= 1024 && std::isfinite(max_episode) && max_episode > 0 &&
                    std::isfinite(auto_seconds) && auto_seconds >= 0 &&
                    (scripted || auto_seconds <= 0) && std::isfinite(max_age_ms) &&
                    max_age_ms > 0 && std::isfinite(max_skew_ms) && max_skew_ms >= 0 &&
                    (scripted || !gamepad.empty()) && auto_episodes >= 1 &&
                    (auto_episodes <= 1 || auto_seconds > 0) && std::isfinite(auto_sine) &&
                    auto_sine >= 0 && auto_sine <= 1 && (auto_sine <= 0 || auto_seconds > 0),
                "invalid recorder arguments");
        Control control(socket);
        auto initial = YAML::Load(control.status());
        require(bool(initial["applied_action"]),
                "rebuild controller: applied_action metadata missing");
        require(initial["enable_slew"].as<bool>() == slew &&
                    initial["enable_tracks"].as<bool>() == tracks,
                "recorder/controller slew/track flags differ");
        std::unique_ptr<Pad> pad;
        if (!scripted)
            pad = std::make_unique<Pad>(gamepad);
        Image cam[2]{Image(camera_paths[0]), Image(camera_paths[1])};
        std::array<std::unique_ptr<Image>, 2> depth;
        const std::array<bool, 2> depth_enabled{!depth_paths[0].empty(), !depth_paths[1].empty()};
        for (int k = 0; k < 2; ++k)
            if (depth_enabled[k])
                depth[k] = std::make_unique<Image>(depth_paths[k]);
        require(fs::create_directory(root), "session root must be new");
        fs::permissions(root, fs::perms::owner_all, fs::perm_options::replace);
        // live_pwm and simulated_imu are streamed as 1/0, which readers already expect.
        std::ofstream manifest(fs::path(root) / "session.json");
        manifest << "{\"format\":\"kaivuri-native-episode-v1\",\"task\":" << json_quote(task)
                 << ",\"fps\":" << fps << ",\"revision\":" << json_quote(revision)
                 << ",\"enable_slew\":" << (slew ? "true" : "false")
                 << ",\"enable_tracks\":" << (tracks ? "true" : "false")
                 << ",\"live_pwm\":" << initial["live_pwm"].as<bool>()
                 << ",\"simulated_imu\":" << initial["simulated_imu"].as<bool>()
                 << ",\"virtual_gamepad\":" << (pad && pad->virtual_pad ? "true" : "false")
                 << ",\"camera_files\":[" << json_quote(camera_paths[0]) << ","
                 << json_quote(camera_paths[1]) << "],\"depth\":";
        put_depth_list(manifest, depth_enabled);
        manifest << ",\"depth_files\":[" << json_quote(depth_paths[0]) << ","
                 << json_quote(depth_paths[1]) << "],\"initial_status\":" << control.status()
                 << "}\n";
        manifest.close();
        require(bool(manifest), "manifest write failed");
        // Calibration written by kaivuri-camera next to its depth store (SERIAL-depth.json).
        for (int k = 0; k < 2; ++k)
            if (depth_enabled[k])
                fs::copy_file(fs::path(depth_paths[k]).replace_extension(".json"),
                              fs::path(root) / ("cam" + std::to_string(k + 1) + ".depth.json"));
        control.send(Operation::Disarm);
        install_stop_handlers();
        std::unique_ptr<Writer> writer;
        uint64_t begin = 0, frames = 0, last_frame = 0, last_master_ns = 0, next_progress = 0;
        std::array<uint64_t, 2> last_ir{}, last_depth{};
        // No operator arming: the controller's centered arming hold is sent automatically
        // (zero commands with the arm flag) whenever it is not armed, then sticks take over.
        bool pump = false, pending_start = auto_seconds > 0;
        int auto_done = 0;
        bool imu_ready = initial["imu_healthy"].as<bool>(), was_armed = false;
        std::string last_fault;
        uint64_t last_recovery = 0;
        std::array<bool, 4> previous{};
        if (pad)
            previous = buttons(*pad);
        uint64_t next = monotonic_ns(), session_start = next;
        const uint64_t period = 10000000;
        if (scripted)
            fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK);
        std::string input;
        bool quit = false;
        // Ends the take only; the controller stays armed and the pump keeps its setting.
        auto finish = [&](bool save, const std::string &reason) {
            pending_start = false;
            if (writer) {
                if (writer->finish(save, frames, begin, monotonic_ns(), reason))
                    ++saved_episodes;
                writer.reset();
                std::cout << "Saved episodes in session: " << saved_episodes << std::endl;
            }
        };
        auto abandon = [&](const std::string &reason) {
            require(auto_seconds <= 0, reason);
            pending_start = false;
            if (writer) {
                writer->abandon(reason);
                writer.reset();
            }
        };
        std::cout << "A=start/save B=discard X=pump Y=re-arm; arms automatically; native spool "
                     "ready\n"
                  << std::flush;
        try {
            while (!stop_requested && !quit) {
                const uint64_t now = monotonic_ns();
                const double late_ms = double(now - next) / 1e6;
                Commands action{};
                bool start = false, save = false, discard = false, toggle = false, rearm = false;
                if (pad) {
                    pad->poll_events();
                    action = pad->action(slew, tracks);
                    const auto pressed = buttons(*pad);
                    start = pressed[0] && !previous[0];
                    discard = pressed[1] && !previous[1];
                    toggle = pressed[2] && !previous[2];
                    rearm = pressed[3] && !previous[3];
                    previous = pressed;
                } else {
                    char b[256];
                    ssize_t n;
                    while ((n = read(STDIN_FILENO, b, sizeof(b))) > 0)
                        input.append(b, n);
                    for (size_t pos; (pos = input.find('\n')) != std::string::npos;) {
                        const auto cmd = input.substr(0, pos);
                        input.erase(0, pos + 1);
                        if (cmd == "arm")
                            rearm = true;
                        else if (cmd == "start")
                            start = true;
                        else if (cmd == "save")
                            save = true;
                        else if (cmd == "discard")
                            discard = true;
                        else if (cmd == "pump")
                            toggle = true;
                        else if (cmd == "quit")
                            quit = true;
                        else
                            throw std::runtime_error("unknown scripted command");
                    }
                    if (auto_sine > 0 && writer) {
                        // Distinct per-joint frequencies; sin(0) = 0 keeps arming centered.
                        const double t = double(now - begin) / 1e9;
                        const double hz[4] = {0.2, 0.3, 0.45, 0.7};
                        for (int j = slew ? 0 : 1; j < 4; ++j)
                            action[j] = auto_sine * std::sin(2 * M_PI * hz[j] * t);
                    }
                }
                if (save || (start && writer)) {
                    finish(true, "save");
                    start = false;
                }
                if (discard) {
                    finish(false, "discard");
                    start = false;
                }
                if (quit)
                    break;
                if (rearm && was_armed) {
                    std::cout << "Already armed; driving/recording continues\n" << std::flush;
                    rearm = false;
                }
                if (rearm)
                    control.send(Operation::Disarm);
                if (start)
                    pending_start = true;
                if (toggle)
                    pump = !pump;
                // Until armed, send the centered hold the controller needs instead of the sticks.
                if (imu_ready)
                    control.send(Operation::Drive, was_armed ? action : Commands{},
                                 deadman_flag | arm_flag | (pump ? pump_flag : 0));
                const auto raw = control.status();
                const auto status = YAML::Load(raw);
                const auto state = status["state"].as<std::string>();
                if (state == "fault") {
                    const auto reason = status["reason"].as<std::string>();
                    if (writer)
                        abandon("controller fault: " + reason);
                    else
                        require(auto_seconds <= 0, "controller fault: " + reason);
                    if (reason != last_fault)
                        std::cerr << "Controller fault: " << reason
                                  << "; re-arming automatically\n";
                    last_fault = reason;
                    // Clear the latched fault at most once per second; arming resumes after it.
                    if (monotonic_ns() - last_recovery >= 1000000000) {
                        control.send(Operation::Disarm);
                        last_recovery = monotonic_ns();
                    }
                } else
                    last_fault.clear();
                imu_ready = status["imu_healthy"].as<bool>();
                const bool healthy = status["fk_valid"].as<bool>();
                const bool armed = state == "armed";
                if (armed && !was_armed)
                    std::cout << "Armed\n" << std::flush;
                was_armed = armed;
                if (pending_start && armed && healthy) {
                    writer = std::make_unique<Writer>(root, queue, depth_enabled, saved_episodes + 1);
                    frames = 0;
                    begin = monotonic_ns();
                    last_frame = begin;
                    next_progress = begin + 2000000000;
                    // Start with new exposures, rather than a cached frame from before A.
                    for (int k = 0; k < 2; ++k) {
                        last_ir[k] = cam[k].read() ? cam[k].m.sequence : 0;
                        last_depth[k] = depth[k] && depth[k]->read() ? depth[k]->m.sequence : 0;
                    }
                    pending_start = false;
                    std::cout << "Recording ep " << writer->name() << ": 0 frames" << std::endl;
                }
                if (writer && (!healthy || !armed))
                    abandon("recording lost healthy armed controller");
                if (writer) {
                    Record record;
                    bool fresh = cam[0].read() && cam[1].read();
                    const uint64_t capture = monotonic_ns();
                    auto valid = [&](const FrameMetadata &m) {
                        return m.host_monotonic_ns <= capture &&
                               double(capture - m.host_monotonic_ns) / 1e6 <= max_age_ms;
                    };
                    const double skew = std::abs(double(cam[0].m.host_monotonic_ns) -
                                                 double(cam[1].m.host_monotonic_ns)) /
                                        1e6;
                    fresh = fresh && valid(cam[0].m) && valid(cam[1].m) &&
                            cam[0].m.sequence != last_ir[0] && cam[1].m.sequence != last_ir[1] &&
                            skew <= max_skew_ms;
                    // Depth arrives within about a millisecond of its IR frame, either side: hold
                    // the sample for a later tick until the same exposure's depth is present.
                    for (int k = 0; k < 2 && fresh; ++k)
                        if (depth[k])
                            fresh = depth[k]->read() && depth[k]->m.sequence != last_depth[k] &&
                                    (cam[k].m.valid_metadata & 1) &&
                                    (depth[k]->m.valid_metadata & 1) &&
                                    depth[k]->m.sensor_timestamp_us == cam[k].m.sensor_timestamp_us;
                    // Do not anchor a new take to an old master exposure while waiting
                    // for camera 2; that can create a skipped frame at the next pair.
                    if (frames == 0)
                        fresh = fresh && capture - cam[0].m.host_monotonic_ns <= uint64_t(.7e9 / fps);
                    // FPS is an upper bound; sample each new master frame without a second
                    // capture clock.
                    if (fresh && (frames == 0 || cam[0].m.host_monotonic_ns - last_master_ns >=
                                                     uint64_t(.7e9 / fps))) {
                        for (const Image &im : cam)
                            require(im.m.width && im.m.height && im.m.stride == im.m.width &&
                                        im.m.pixel_bytes == uint64_t(im.m.width) * im.m.height,
                                    "only packed Y8 images supported");
                        for (int k = 0; k < 2; ++k)
                            record.images[k].assign(cam[k].pixels.begin(),
                                                    cam[k].pixels.begin() + cam[k].m.pixel_bytes);
                        for (int k = 0; k < 2; ++k)
                            if (depth[k]) {
                                const auto &d = depth[k]->m;
                                require(d.width == cam[k].m.width && d.height == cam[k].m.height &&
                                            d.stride == d.width * 2 &&
                                            d.pixel_bytes == uint64_t(d.stride) * d.height,
                                        "depth must be packed Z16 of IR size");
                                record.depth[k].assign(depth[k]->pixels.begin(),
                                                       depth[k]->pixels.begin() + d.pixel_bytes);
                                last_depth[k] = d.sequence;
                            }
                        last_ir = {cam[0].m.sequence, cam[1].m.sequence};
                        last_frame = capture;
                        last_master_ns = cam[0].m.host_monotonic_ns;
                    }
                    require(capture - last_frame <= uint64_t(max_age_ms * 1e6),
                            "camera stalled, stale, duplicated, pair skew exceeds limit or depth "
                            "unpaired");
                    std::ostringstream row;
                    row.precision(17);
                    row << "{\"sample_ns\":" << capture << ",\"episode_ns\":" << capture - begin
                        << ",\"teleop_lateness_ms\":" << late_ms << ",\"requested_action\":[";
                    for (size_t i = 0; i < action.size(); ++i)
                        row << (i ? "," : "") << action[i];
                    row << "],\"requested_sequence\":" << control.sequence - 1
                        << ",\"status\":" << raw;
                    if (!record.images[0].empty()) {
                        row << ",\"frame_index\":" << frames++;
                        for (int k = 0; k < 2; ++k) {
                            row << ",\"cam" << k + 1 << "\":";
                            put_metadata(row, cam[k].m);
                        }
                        for (int k = 0; k < 2; ++k)
                            if (depth[k]) {
                                row << ",\"cam" << k + 1 << "_depth\":";
                                put_metadata(row, depth[k]->m);
                            }
                    }
                    row << "}";
                    record.json = row.str();
                    require(writer->push(std::move(record)),
                            "recording disk queue full/write failed; episode retained incomplete");
                    const double duration = double(capture - begin) / 1e9;
                    if (capture >= next_progress) {
                        std::cout << "Recording ep " << writer->name() << ": " << frames
                                  << " frames, " << uint64_t(duration) << " s" << std::endl;
                        next_progress = capture + 2000000000;
                    }
                    if (duration >= max_episode || (auto_seconds > 0 && duration >= auto_seconds)) {
                        finish(true, "duration");
                        if (auto_seconds > 0 && ++auto_done < auto_episodes) {
                            pending_start = true;
                            session_start = monotonic_ns();
                        } else if (auto_seconds > 0)
                            quit = true;
                    }
                }
                require(auto_seconds <= 0 || writer || !pending_start ||
                            double(monotonic_ns() - session_start) / 1e9 <= 20,
                        "automatic arming timeout");
                next += period;
                if (next < monotonic_ns())
                    next = monotonic_ns() + period;
                const timespec wake{time_t(next / 1000000000), long(next % 1000000000)};
                while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr) == EINTR &&
                       !stop_requested) {
                }
            }
            std::cout << "Stopping recorder..." << std::endl;
            finish(true, "session_end");
        } catch (...) {
            try {
                control.send(Operation::Disarm);
            } catch (...) {
            }
            writer.reset();
            throw;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Recorder failed: " << e.what() << std::endl;
        return 1;
    }
}
