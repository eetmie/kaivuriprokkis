#include "kaivuri/util.hpp"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <openssl/evp.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

namespace kaivuri {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}
void require_errno(bool condition, const char *what) {
    if (!condition)
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}
void require_errno(bool condition, const std::string &what) {
    require_errno(condition, what.c_str());
}
uint64_t monotonic_ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}
void Fd::reset(int fd) {
    if (fd_ >= 0)
        ::close(fd_);
    fd_ = fd;
}
Fd open_serial(const std::filesystem::path &device, int baud) {
    Fd fd(::open(device.c_str(), O_RDONLY | O_NONBLOCK | O_NOCTTY | O_CLOEXEC));
    require_errno(bool(fd), "open serial " + device.string());
    if (!isatty(fd.get()))
        return fd;
    require_errno(ioctl(fd.get(), TIOCEXCL) == 0, "exclusive serial " + device.string());
    termios t{};
    require_errno(tcgetattr(fd.get(), &t) == 0, "get serial settings");
    cfmakeraw(&t);
    if (baud == 115200) {
        cfsetispeed(&t, B115200);
        cfsetospeed(&t, B115200);
    } else
        require(baud == 0, "unsupported serial baud rate");
    t.c_cflag |= CLOCAL | CREAD;
    require_errno(tcsetattr(fd.get(), TCSANOW, &t) == 0, "set serial settings");
    return fd;
}
volatile std::sig_atomic_t stop_requested = 0;
void install_stop_handlers() {
    std::signal(SIGINT, [](int) { stop_requested = 1; });
    std::signal(SIGTERM, [](int) { stop_requested = 1; });
}
std::filesystem::path runtime_dir() {
    return "/tmp/kaivuri-native-" + std::to_string(geteuid());
}
bool Args::next() {
    if (++index_ >= argc_)
        return false;
    flag_ = argv_[index_];
    return true;
}
std::string Args::value() {
    require(++index_ < argc_, "missing argument: " + flag_);
    return argv_[index_];
}
double Args::number() {
    auto text = value();
    size_t used = 0;
    double v = 0;
    try {
        v = std::stod(text, &used);
    } catch (const std::exception &) {
    }
    require(used == text.size() && used > 0, "invalid value for " + flag_ + ": " + text);
    return v;
}
double Args::finite() {
    return finite(flag_, value());
}
double Args::finite(const std::string &flag, const std::string &text) {
    size_t used = 0;
    double v = 0;
    try {
        v = std::stod(text, &used);
    } catch (const std::exception &) {
    }
    require(used > 0 && used == text.size() && std::isfinite(v),
            "invalid value for " + flag + ": " + text);
    return v;
}
uint64_t Args::natural() {
    auto text = value();
    size_t used = 0;
    uint64_t v = 0;
    try {
        if (text[0] != '-')
            v = std::stoull(text, &used);
    } catch (const std::exception &) {
    }
    require(used > 0 && used == text.size(), "invalid value for " + flag_ + ": " + text);
    return v;
}
int Args::integer() {
    auto text = value();
    size_t used = 0;
    int v = 0;
    try {
        v = std::stoi(text, &used);
    } catch (const std::exception &) {
    }
    require(used == text.size() && used > 0, "invalid value for " + flag_ + ": " + text);
    return v;
}
void Args::unknown() const {
    throw std::runtime_error("unknown option: " + flag_);
}
static std::string hex(const unsigned char *digest, unsigned length) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out(size_t(length) * 2, '0');
    for (unsigned i = 0; i < length; ++i) {
        out[2 * i] = digits[digest[i] >> 4];
        out[2 * i + 1] = digits[digest[i] & 15];
    }
    return out;
}
std::string sha256_hex(const void *data, size_t bytes) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned length = 0;
    require(EVP_Digest(data, bytes, digest, &length, EVP_sha256(), nullptr) == 1, "SHA256 failed");
    return hex(digest, length);
}
std::string sha256_file(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot hash: " + path.string());
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "SHA256 init failed");
    char bytes[65536];
    while (in.read(bytes, sizeof(bytes)) || in.gcount())
        EVP_DigestUpdate(ctx.get(), bytes, size_t(in.gcount()));
    require(in.eof(), "SHA256 read failed: " + path.string());
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned length = 0;
    EVP_DigestFinal_ex(ctx.get(), digest, &length);
    return hex(digest, length);
}
std::string json_quote(const std::string &text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\')
            out += {'\\', char(c)};
        else if (c == '\n')
            out += "\\n";
        else if (c == '\r')
            out += "\\r";
        else if (c == '\t')
            out += "\\t";
        else if (c < 32) {
            char escaped[7];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            out += escaped;
        } else
            out += char(c);
    }
    return out + '"';
}
} // namespace kaivuri
