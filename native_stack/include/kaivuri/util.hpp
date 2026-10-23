#pragma once
// Small helpers shared by every native executable: errors, clocks, file descriptors,
// shutdown signals, command-line options, hashing and JSON strings.
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

namespace kaivuri {
// Throws std::runtime_error(message) when the condition is false. The const char * overloads
// keep a passing check allocation-free (realtime loops); a std::string message is built even
// when the check passes.
void require(bool condition, const char *message);
void require(bool condition, const std::string &message);
// Throws std::runtime_error("what: strerror(errno)") when the condition is false.
void require_errno(bool condition, const char *what);
void require_errno(bool condition, const std::string &what);

uint64_t monotonic_ns();

// Owning file descriptor; closes on destruction. Move-only.
class Fd {
  public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    ~Fd() { reset(); }
    Fd(Fd &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    Fd &operator=(Fd &&other) noexcept {
        if (this != &other) {
            reset();
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }
    Fd(const Fd &) = delete;
    Fd &operator=(const Fd &) = delete;
    int get() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }
    void reset(int fd = -1);

  private:
    int fd_ = -1;
};

// Opens a serial device read-only, non-blocking and exclusive; a tty is set raw with
// CLOCAL|CREAD (baud > 0 also sets that speed). Non-tty paths (FIFOs, files) are left as is.
Fd open_serial(const std::filesystem::path &device, int baud = 0);

// Set by SIGINT/SIGTERM once install_stop_handlers() has run.
extern volatile std::sig_atomic_t stop_requested;
void install_stop_handlers();

// Per-user runtime directory for sockets and locks: /tmp/kaivuri-native-<uid>.
std::filesystem::path runtime_dir();

// Command-line walker: while (args.next()) { if (args.is("--x")) x = args.number(); ... }
// Errors read "missing argument: --x", "invalid value for --x: abc", "unknown option: --y".
// Numbers must use the whole argument ("1junk" is rejected).
class Args {
  public:
    Args(int argc, char **argv) : argc_(argc), argv_(argv) {}
    bool next();
    bool is(const char *flag) const { return flag_ == flag; }
    const std::string &flag() const { return flag_; }
    std::string value();
    double number(); // accepts inf/nan (e.g. --action-tolerance inf)
    double finite();
    int integer();
    uint64_t natural(); // non-negative, full 64-bit range; a leading '-' is rejected
    // finite() for a value that is not the next argument (e.g. one item of a list).
    static double finite(const std::string &flag, const std::string &text);
    [[noreturn]] void unknown() const;

  private:
    int argc_, index_ = 0;
    char **argv_;
    std::string flag_;
};

std::string sha256_hex(const void *data, size_t bytes);
std::string sha256_file(const std::filesystem::path &path);
// JSON string literal with quotes and escapes.
std::string json_quote(const std::string &text);
} // namespace kaivuri
