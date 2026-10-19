// Native teleoperation and bounded, lossless episode spool. No encoder or Python in this process.
#include "kaivuri/core.hpp"
#include "kaivuri/gamepad.hpp"
#include "kaivuri/frame_store.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <linux/input.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <zstd.h>
using namespace kaivuri;
namespace fs = std::filesystem;
static volatile sig_atomic_t stopped = 0;
static void stop_signal(int) {
    stopped = 1;
}
static uint64_t ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
static std::string quote(const std::string &s) {
    std::ostringstream o;
    o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            o << '\\' << c;
        else if (c < 32)
            o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else
            o << c;
    }
    o << '"';
    return o.str();
}
struct Control {
    int fd = -1;
    uint64_t sequence = 0;
    explicit Control(const std::string &path) {
        fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        if (fd < 0)
            throw std::runtime_error("control socket failed");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (path.size() >= sizeof(address.sun_path)) {
            close(fd);
            throw std::runtime_error("socket path too long");
        }
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
            close(fd);
            throw std::runtime_error("cannot connect to controller");
        }
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
        put(20, ns());
        bool drive = op == Operation::Drive;
        put(28, uint32_t(drive));
        put(32, float(100));
        if (drive)
            for (size_t i = 0; i < 6; ++i)
                put(52 + i * 4, float(action[i]));
        const size_t size = drive ? 76 : 52;
        if (::send(fd, data.data(), size, MSG_DONTWAIT | MSG_NOSIGNAL) != ssize_t(size))
            throw std::runtime_error("control send failed");
    }
    std::string status() {
        send(Operation::Status);
        pollfd p{fd, POLLIN, 0};
        int ready;
        do {
            ready = poll(&p, 1, 100);
        } while (ready < 0 && errno == EINTR);
        if (ready != 1)
            throw std::runtime_error("controller status timeout");
        char bytes[8192];
        ssize_t count;
        do {
            count = recv(fd, bytes, sizeof(bytes), MSG_TRUNC);
        } while (count < 0 && errno == EINTR);
        if (count <= 0 || count > ssize_t(sizeof(bytes)))
            throw std::runtime_error("invalid/truncated status");
        return std::string(bytes, count);
    }
    ~Control() {
        if (fd >= 0) {
            try {
                send(Operation::Disarm);
                status(); // Drain acknowledgement before closing the owning connection.
            } catch (...) {
            }
            close(fd);
        }
    }
};
// Linux evdev uses X/Y labels BTN_NORTH/BTN_WEST, matching modules/gamepad.py.
struct Image {
    void *handle = nullptr;
    FrameMetadata m{};
    std::vector<uint8_t> pixels;
    explicit Image(const std::string &path) {
        handle = kaivuri_frame_open(path.c_str());
        if (!handle)
            throw std::runtime_error("cannot open camera store: " + path);
    }
    ~Image() { kaivuri_frame_close(handle); }
    bool read() {
        FrameMetadata info{};
        if (kaivuri_frame_read(handle, &info, nullptr, 0) != 1)
            return false;
        if (!info.pixel_bytes || info.pixel_bytes > 16 * 1024 * 1024)
            throw std::runtime_error("invalid camera size");
        if (!pixels.empty() && info.pixel_bytes != pixels.size())
            throw std::runtime_error("camera size changed");
        if (info.sequence == m.sequence && !pixels.empty())
            return true;
        pixels.resize(info.pixel_bytes);
        int r = kaivuri_frame_read(handle, &m, pixels.data(), pixels.size());
        if (r < 0)
            throw std::runtime_error("invalid camera frame");
        return r == 1;
    }
};
static std::string metadata(const FrameMetadata &m) {
    std::ostringstream o;
    o << "{\"sequence\":" << m.sequence << ",\"host_monotonic_ns\":" << m.host_monotonic_ns
      << ",\"sensor_timestamp_us\":" << m.sensor_timestamp_us
      << ",\"frame_timestamp_us\":" << m.frame_timestamp_us
      << ",\"frame_counter\":" << m.frame_counter
      << ",\"actual_exposure_us\":" << m.actual_exposure_us << ",\"gain_level\":" << m.gain_level
      << ",\"valid_metadata\":" << m.valid_metadata
      << ",\"sdk_timestamp_ms\":" << std::setprecision(17) << m.sdk_timestamp_ms
      << ",\"timestamp_domain\":" << m.timestamp_domain << ",\"width\":" << m.width
      << ",\"height\":" << m.height << ",\"stride\":" << m.stride << ",\"ir_index\":" << m.ir_index
      << ",\"pixel_bytes\":" << m.pixel_bytes << "}";
    return o.str();
}
struct Record {
    std::string json;
    std::array<std::vector<uint8_t>, 2> images;
    std::array<std::vector<uint8_t>, 2> depth; // packed Z16, compressed on the writer thread
};
class Writer {
    fs::path path_;
    std::ofstream rows_, cam1_, cam2_;
    // camN.depth.zst: one independent zstd frame per recorded frame (self-delimiting).
    std::array<bool, 2> depth_enabled_;
    std::array<std::ofstream, 2> depth_;
    std::array<uint64_t, 2> depth_written_{};
    std::vector<uint8_t> packed_;
    ZSTD_CCtx *zstd_ = nullptr;
    bool depth_ok() const {
        for (int k = 0; k < 2; ++k)
            if (depth_enabled_[k] && !depth_[k])
                return false;
        return true;
    }
    void close_files() {
        rows_.close();
        cam1_.close();
        cam2_.close();
        for (auto &d : depth_)
            if (d.is_open())
                d.close();
    }
    std::deque<Record> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    bool done_ = false;
    std::atomic<bool> failed_{false};
    size_t limit_;
    uint64_t written_ = 0;
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
                    cam1_.write(reinterpret_cast<const char *>(r.images[0].data()),
                                r.images[0].size());
                    cam2_.write(reinterpret_cast<const char *>(r.images[1].data()),
                                r.images[1].size());
                    ++written_;
                    for (int k = 0; k < 2; ++k) {
                        if (!depth_enabled_[k])
                            continue;
                        const auto &z = r.depth[k];
                        packed_.resize(ZSTD_compressBound(z.size()));
                        const size_t bytes = ZSTD_compressCCtx(zstd_, packed_.data(),
                                                               packed_.size(), z.data(), z.size(), 1);
                        if (z.empty() || ZSTD_isError(bytes))
                            throw std::runtime_error("depth compression failed");
                        depth_[k].write(reinterpret_cast<const char *>(packed_.data()), bytes);
                        ++depth_written_[k];
                    }
                }
                rows_ << r.json << '\n';
                if (!rows_ || !cam1_ || !cam2_ || !depth_ok())
                    throw std::runtime_error("episode disk write failed");
            }
            rows_.flush();
            cam1_.flush();
            cam2_.flush();
            for (auto &d : depth_)
                if (d.is_open())
                    d.flush();
            if (!rows_ || !cam1_ || !cam2_ || !depth_ok())
                throw std::runtime_error("episode flush failed");
        } catch (...) {
            failed_ = true;
        }
    }

  public:
    Writer(const fs::path &root, size_t limit, std::array<bool, 2> depth)
        : depth_enabled_(depth), limit_(limit) {
        path_ = root / ("episode-" + std::to_string(ns()) + ".partial");
        if (!fs::create_directory(path_))
            throw std::runtime_error("episode directory exists");
        rows_.open(path_ / "samples.jsonl");
        cam1_.open(path_ / "cam1.y8", std::ios::binary);
        cam2_.open(path_ / "cam2.y8", std::ios::binary);
        for (int k = 0; k < 2; ++k)
            if (depth_enabled_[k])
                depth_[k].open(path_ / ("cam" + std::to_string(k + 1) + ".depth.zst"),
                               std::ios::binary);
        if (!rows_ || !cam1_ || !cam2_ || !depth_ok())
            throw std::runtime_error("cannot create episode files");
        if (depth_enabled_[0] || depth_enabled_[1]) {
            zstd_ = ZSTD_createCCtx();
            if (!zstd_)
                throw std::runtime_error("zstd context allocation failed");
        }
        thread_ = std::thread([this] { run(); });
    }
    // Stop writing without committing: the take stays as .partial, which export skips.
    void abandon(const std::string &reason) {
        join();
        close_files();
        std::cerr << "Take stopped (" << reason << "); kept incomplete at " << path_ << std::endl;
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
    ~Writer() {
        join();
        ZSTD_freeCCtx(zstd_);
    }
    void finish(bool save, uint64_t frames, uint64_t start, uint64_t end,
                const std::string &reason) {
        join();
        close_files();
        for (int k = 0; k < 2; ++k)
            if (depth_enabled_[k] && depth_written_[k] != frames)
                throw std::runtime_error("incomplete depth retained at " + path_.string());
        if (failed_ || written_ != frames)
            throw std::runtime_error("incomplete episode retained at " + path_.string());
        if (!save || !frames) {
            fs::remove_all(path_);
            std::cout << "Episode discarded\n";
            return;
        }
        std::ofstream summary(path_ / "episode.json");
        summary << "{\"frames\":" << frames << ",\"start_ns\":" << start << ",\"end_ns\":" << end
                << ",\"reason\":" << quote(reason) << ",\"depth\":[";
        for (int k = 0, n = 0; k < 2; ++k)
            if (depth_enabled_[k])
                summary << (n++ ? "," : "") << "\"cam" << k + 1 << "\"";
        summary << "]}\n";
        summary.close();
        if (!summary)
            throw std::runtime_error("episode summary write failed");
        std::vector<std::string> files{"samples.jsonl", "cam1.y8", "cam2.y8", "episode.json"};
        for (int k = 0; k < 2; ++k)
            if (depth_enabled_[k])
                files.push_back("cam" + std::to_string(k + 1) + ".depth.zst");
        for (const auto &file : files) {
            int fd = open((path_ / file).c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0)
                throw std::runtime_error("episode sync open failed");
            int result = fsync(fd);
            close(fd);
            if (result)
                throw std::runtime_error("episode fsync failed");
        }
        int dir = open(path_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dir < 0)
            throw std::runtime_error("episode directory sync open failed");
        int result = fsync(dir);
        close(dir);
        if (result)
            throw std::runtime_error("episode directory fsync failed");
        fs::path ready = path_;
        ready.replace_extension(".ready");
        fs::rename(path_, ready);
        dir = open(ready.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dir < 0)
            throw std::runtime_error("session sync open failed");
        result = fsync(dir);
        close(dir);
        if (result)
            throw std::runtime_error("session fsync failed");
        std::cout << "Saved " << frames << " frames: " << ready << std::endl;
    }
};
int main(int argc, char **argv) {
    try {
        std::string runtime = "/tmp/kaivuri-native-" + std::to_string(geteuid());
        std::string socket = runtime + "/control.sock", gamepad, task, root, revision;
        std::vector<std::string> camera_paths;
        std::array<std::string, 2> depth_paths;
        int fps = 30, queue = 64;
        double max_episode = 180, auto_seconds = 0, max_skew_ms = 40, max_age_ms = 200;
        double auto_sine = 0;
        int auto_episodes = 1;
        bool scripted = false, tracks = false, slew = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto val = [&]() {
                if (++i == argc)
                    throw std::runtime_error("missing argument");
                return std::string(argv[i]);
            };
            if (a == "--socket")
                socket = val();
            else if (a == "--root")
                root = val();
            else if (a == "--task")
                task = val();
            else if (a == "--gamepad")
                gamepad = val();
            else if (a == "--camera-file")
                camera_paths.push_back(val());
            else if (a == "--cam1-depth-file")
                depth_paths[0] = val();
            else if (a == "--cam2-depth-file")
                depth_paths[1] = val();
            else if (a == "--fps")
                fps = std::stoi(val());
            else if (a == "--queue-frames")
                queue = std::stoi(val());
            else if (a == "--max-episode-s")
                max_episode = std::stod(val());
            else if (a == "--auto-record-s")
                auto_seconds = std::stod(val());
            else if (a == "--auto-episodes")
                auto_episodes = std::stoi(val());
            else if (a == "--auto-sine")
                auto_sine = std::stod(val());
            else if (a == "--max-pair-skew-ms")
                max_skew_ms = std::stod(val());
            else if (a == "--max-age-ms")
                max_age_ms = std::stod(val());
            else if (a == "--revision")
                revision = val();
            else if (a == "--scripted")
                scripted = true;
            else if (a == "--enable-tracks")
                tracks = true;
            else if (a == "--enable-slew")
                slew = true;
            else if (a == "--help") {
                std::cout
                    << "kaivuri-record --root NEW_SESSION --task TEXT --gamepad /dev/input/eventN\n"
                       "  --camera-file .../SERIAL1-ir1.frame --camera-file .../SERIAL2-ir2.frame\n"
                       "  [--cam1-depth-file .../SERIAL1-depth.frame] [--cam2-depth-file ...]\n"
                       "  (depth is paired to its IR frame by sensor timestamp; camN.depth.zst)\n"
                       "  [--socket PATH --fps 30 --enable-slew --enable-tracks]\n"
                       "  [--max-episode-s 180 --queue-frames 64 --max-pair-skew-ms 40 "
                       "--max-age-ms 200]\n"
                       "Arms automatically and re-arms after controller faults; sticks drive at "
                       "all times.\n"
                       "A=start/save B=discard X=pump Y=re-arm now; Ctrl+C=save and exit.\n"
                       "No automatic config reload: restart services after configuration "
                       "changes.\n"
                       "--scripted [--auto-record-s SEC [--auto-episodes N] [--auto-sine AMP]] "
                       "reads stdin: arm/start/save/discard/pump/quit.\n"
                       "--auto-sine drives each enabled joint with an AMP-amplitude sine while "
                       "recording, so test datasets have non-constant actions.\n";
                return 0;
            } else
                throw std::runtime_error("unknown option: " + a);
        }
        if (root.empty() || task.empty() || camera_paths.size() != 2 ||
            camera_paths[0] == camera_paths[1] || fps < 1 || fps > 90 || queue < 2 ||
            queue > 1024 || !std::isfinite(max_episode) || max_episode <= 0 ||
            !std::isfinite(auto_seconds) || auto_seconds < 0 || (!scripted && auto_seconds > 0) ||
            !std::isfinite(max_age_ms) || max_age_ms <= 0 || !std::isfinite(max_skew_ms) ||
            max_skew_ms < 0 || (!scripted && gamepad.empty()) || auto_episodes < 1 ||
            (auto_episodes > 1 && auto_seconds <= 0) || !std::isfinite(auto_sine) ||
            auto_sine < 0 || auto_sine > 1 || (auto_sine > 0 && auto_seconds <= 0))
            throw std::runtime_error("invalid recorder arguments");
        Control control(socket);
        auto initial = YAML::Load(control.status());
        if (!initial["applied_action"])
            throw std::runtime_error("rebuild controller: applied_action metadata missing");
        if (initial["enable_slew"].as<bool>() != slew ||
            initial["enable_tracks"].as<bool>() != tracks)
            throw std::runtime_error("recorder/controller slew/track flags differ");
        std::unique_ptr<Pad> pad;
        if (!scripted)
            pad = std::make_unique<Pad>(gamepad);
        Image cam1(camera_paths[0]), cam2(camera_paths[1]);
        std::array<std::unique_ptr<Image>, 2> depth;
        const std::array<bool, 2> depth_enabled{!depth_paths[0].empty(), !depth_paths[1].empty()};
        for (int k = 0; k < 2; ++k)
            if (depth_enabled[k])
                depth[k] = std::make_unique<Image>(depth_paths[k]);
        if (!fs::create_directory(root))
            throw std::runtime_error("session root must be new");
        fs::permissions(root, fs::perms::owner_all, fs::perm_options::replace);
        std::ofstream manifest(fs::path(root) / "session.json");
        manifest << "{\"format\":\"kaivuri-native-episode-v1\",\"task\":" << quote(task)
                 << ",\"fps\":" << fps << ",\"revision\":" << quote(revision)
                 << ",\"enable_slew\":" << (slew ? "true" : "false")
                 << ",\"enable_tracks\":" << (tracks ? "true" : "false")
                 << ",\"live_pwm\":" << initial["live_pwm"].as<bool>()
                 << ",\"simulated_imu\":" << initial["simulated_imu"].as<bool>()
                 << ",\"virtual_gamepad\":" << (pad && pad->virtual_pad ? "true" : "false")
                 << ",\"camera_files\":[" << quote(camera_paths[0]) << "," << quote(camera_paths[1])
                 << "],\"depth\":[";
        for (int k = 0, n = 0; k < 2; ++k)
            if (depth_enabled[k])
                manifest << (n++ ? "," : "") << "\"cam" << k + 1 << "\"";
        manifest << "],\"depth_files\":[" << quote(depth_paths[0]) << "," << quote(depth_paths[1])
                 << "],\"initial_status\":" << control.status() << "}\n";
        manifest.close();
        if (!manifest)
            throw std::runtime_error("manifest write failed");
        // Calibration written by kaivuri-camera next to its depth store (SERIAL-depth.json).
        for (int k = 0; k < 2; ++k)
            if (depth_enabled[k])
                fs::copy_file(fs::path(depth_paths[k]).replace_extension(".json"),
                              fs::path(root) / ("cam" + std::to_string(k + 1) + ".depth.json"));
        control.send(Operation::Disarm);
        signal(SIGINT, stop_signal);
        signal(SIGTERM, stop_signal);
        std::unique_ptr<Writer> writer;
        uint64_t begin = 0, frames = 0, last1 = 0, last2 = 0, last_frame = 0, last_master_ns = 0;
        std::array<uint64_t, 2> last_depth{};
        // No operator arming: the controller's centered arming hold is sent automatically
        // (zero commands with the arm flag) whenever it is not armed, then sticks take over.
        bool pump = false, pending_start = auto_seconds > 0;
        int auto_done = 0;
        bool imu_ready = initial["imu_healthy"].as<bool>(), was_armed = false;
        std::string last_fault;
        uint64_t last_recovery = 0;
        std::array<bool, 4> previous{};
        if (pad)
            previous = {pad->keys[BTN_SOUTH], pad->keys[BTN_EAST], pad->keys[BTN_NORTH],
                        pad->keys[BTN_WEST]};
        uint64_t next = ns(), session_start = next;
        const uint64_t period = 10000000;
        if (scripted)
            fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK);
        std::string input;
        bool quit = false;
        // Ends the take only; the controller stays armed and the pump keeps its setting.
        auto finish = [&](bool save, const std::string &reason) {
            pending_start = false;
            if (writer) {
                writer->finish(save, frames, begin, ns(), reason);
                writer.reset();
            }
        };
        auto abandon = [&](const std::string &reason) {
            if (auto_seconds > 0)
                throw std::runtime_error(reason);
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
            while (!stopped && !quit) {
                uint64_t now = ns();
                double late_ms = double(now - next) / 1e6;
                Commands action{};
                bool start = false, save = false, discard = false, toggle = false, rearm = false;
                if (pad) {
                    pad->poll_events();
                    action = pad->action(slew, tracks);
                    std::array<bool, 4> buttons{pad->keys[BTN_SOUTH], pad->keys[BTN_EAST],
                                                pad->keys[BTN_NORTH], pad->keys[BTN_WEST]};
                    start = buttons[0] && !previous[0];
                    discard = buttons[1] && !previous[1];
                    toggle = buttons[2] && !previous[2];
                    rearm = buttons[3] && !previous[3];
                    previous = buttons;
                } else {
                    char b[256];
                    ssize_t n;
                    while ((n = read(0, b, sizeof(b))) > 0)
                        input.append(b, n);
                    size_t pos;
                    while ((pos = input.find('\n')) != std::string::npos) {
                        auto cmd = input.substr(0, pos);
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
                auto raw = control.status();
                auto status = YAML::Load(raw);
                if (status["state"].as<std::string>() == "fault") {
                    const auto reason = status["reason"].as<std::string>();
                    if (writer)
                        abandon("controller fault: " + reason);
                    else if (auto_seconds > 0)
                        throw std::runtime_error("controller fault: " + reason);
                    if (reason != last_fault)
                        std::cerr << "Controller fault: " << reason << "; re-arming automatically\n";
                    last_fault = reason;
                    // Clear the latched fault at most once per second; arming resumes after it.
                    if (ns() - last_recovery >= 1000000000) {
                        control.send(Operation::Disarm);
                        last_recovery = ns();
                    }
                } else
                    last_fault.clear();
                imu_ready = status["imu_healthy"].as<bool>();
                bool healthy = status["fk_valid"].as<bool>();
                bool armed = status["state"].as<std::string>() == "armed";
                if (armed && !was_armed)
                    std::cout << "Armed\n" << std::flush;
                was_armed = armed;
                if (pending_start && armed && healthy) {
                    writer = std::make_unique<Writer>(root, queue, depth_enabled);
                    frames = 0;
                    begin = ns();
                    last_frame = begin;
                    last1 = last2 = 0;
                    last_depth = {};
                    pending_start = false;
                    std::cout << "Recording\n" << std::flush;
                }
                if (writer && (!healthy || !armed))
                    abandon("recording lost healthy armed controller");
                if (writer) {
                    Record record;
                    bool fresh = cam1.read() && cam2.read();
                    uint64_t capture = ns();
                    auto valid = [&](const FrameMetadata &m) {
                        return m.host_monotonic_ns <= capture &&
                               double(capture - m.host_monotonic_ns) / 1e6 <= max_age_ms;
                    };
                    double skew = std::abs(double(cam1.m.host_monotonic_ns) -
                                           double(cam2.m.host_monotonic_ns)) /
                                  1e6;
                    fresh = fresh && valid(cam1.m) && valid(cam2.m) && cam1.m.sequence != last1 &&
                            cam2.m.sequence != last2 && skew <= max_skew_ms;
                    // Depth arrives within about a millisecond of its IR frame, either side: hold
                    // the sample for a later tick until the same exposure's depth is present.
                    const Image *ir[2]{&cam1, &cam2};
                    for (int k = 0; k < 2 && fresh; ++k)
                        if (depth[k])
                            fresh = depth[k]->read() && depth[k]->m.sequence != last_depth[k] &&
                                    (ir[k]->m.valid_metadata & 1) &&
                                    (depth[k]->m.valid_metadata & 1) &&
                                    depth[k]->m.sensor_timestamp_us == ir[k]->m.sensor_timestamp_us;
                    // FPS is an upper bound; sample each new master frame without a second capture
                    // clock.
                    if (fresh && (frames == 0 || cam1.m.host_monotonic_ns - last_master_ns >=
                                                     uint64_t(.7e9 / fps))) {
                        for (Image *im : {&cam1, &cam2})
                            if (!im->m.width || !im->m.height || im->m.stride != im->m.width ||
                                im->m.pixel_bytes != uint64_t(im->m.width) * im->m.height)
                                throw std::runtime_error("only packed Y8 images supported");
                        record.images[0].assign(cam1.pixels.begin(),
                                                cam1.pixels.begin() + cam1.m.pixel_bytes);
                        record.images[1].assign(cam2.pixels.begin(),
                                                cam2.pixels.begin() + cam2.m.pixel_bytes);
                        for (int k = 0; k < 2; ++k)
                            if (depth[k]) {
                                const auto &d = depth[k]->m;
                                if (d.width != ir[k]->m.width || d.height != ir[k]->m.height ||
                                    d.stride != d.width * 2 ||
                                    d.pixel_bytes != uint64_t(d.stride) * d.height)
                                    throw std::runtime_error("depth must be packed Z16 of IR size");
                                record.depth[k].assign(depth[k]->pixels.begin(),
                                                       depth[k]->pixels.begin() + d.pixel_bytes);
                                last_depth[k] = d.sequence;
                            }
                        last1 = cam1.m.sequence;
                        last2 = cam2.m.sequence;
                        last_frame = capture;
                        last_master_ns = cam1.m.host_monotonic_ns;
                    }
                    if (capture - last_frame > uint64_t(max_age_ms * 1e6))
                        throw std::runtime_error("camera stalled, stale, duplicated, pair skew "
                                                 "exceeds limit or depth unpaired");
                    std::ostringstream row;
                    row << std::setprecision(17) << "{\"sample_ns\":" << capture
                        << ",\"episode_ns\":" << capture - begin
                        << ",\"teleop_lateness_ms\":" << late_ms << ",\"requested_action\":[";
                    for (size_t i = 0; i < action.size(); ++i)
                        row << (i ? "," : "") << action[i];
                    row << "],\"requested_sequence\":" << control.sequence - 1
                        << ",\"status\":" << raw;
                    if (!record.images[0].empty()) {
                        row << ",\"frame_index\":" << frames++ << ",\"cam1\":" << metadata(cam1.m)
                            << ",\"cam2\":" << metadata(cam2.m);
                        for (int k = 0; k < 2; ++k)
                            if (depth[k])
                                row << ",\"cam" << k + 1 << "_depth\":" << metadata(depth[k]->m);
                    }
                    row << "}";
                    record.json = row.str();
                    if (!writer->push(std::move(record)))
                        throw std::runtime_error(
                            "recording disk queue full/write failed; episode retained incomplete");
                    double duration = double(capture - begin) / 1e9;
                    if (duration >= max_episode || (auto_seconds > 0 && duration >= auto_seconds)) {
                        finish(true, "duration");
                        if (auto_seconds > 0 && ++auto_done < auto_episodes) {
                            pending_start = true;
                            session_start = ns();
                        } else if (auto_seconds > 0)
                            quit = true;
                    }
                }
                if (auto_seconds > 0 && !writer && double(ns() - session_start) / 1e9 > 20 &&
                    pending_start)
                    throw std::runtime_error("automatic arming timeout");
                next += period;
                if (next < ns())
                    next = ns() + period;
                timespec wake{time_t(next / 1000000000), long(next % 1000000000)};
                while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr) == EINTR &&
                       !stopped) {
                }
            }
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
