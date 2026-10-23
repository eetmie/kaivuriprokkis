#include "kaivuri/pwm.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <linux/i2c-dev.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kaivuri {
namespace {
// require_errno builds its message string on every call; writes happen each control tick.
void check_write(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}
} // namespace
void Pwm::reg(uint8_t address, uint8_t value) {
    const uint8_t bytes[]{address, value};
    check_write(::write(fd_.get(), bytes, 2) == 2, "PCA9685 register write");
}
void Pwm::center() {
    for (const auto &v : cfg_.valves)
        pulses[v.channel] = v.center;
    pulses[cfg_.pump_channel] = cfg_.pump_min;
}
Pwm::Pwm(const Config &config, bool live) : cfg_(config), live_(live) {
    for (const auto &v : cfg_.valves) {
        lo_ = std::min(lo_, v.channel);
        hi_ = std::max(hi_, v.channel);
    }
    lo_ = std::min(lo_, cfg_.pump_channel);
    hi_ = std::max(hi_, cfg_.pump_channel);
    center();
    if (!live_)
        return;
    fd_ = Fd(::open(("/dev/i2c-" + std::to_string(cfg_.i2c_bus)).c_str(), O_RDWR | O_CLOEXEC));
    require_errno(bool(fd_), "open I2C");
    require_errno(ioctl(fd_.get(), I2C_SLAVE, cfg_.address) >= 0, "select PCA9685");
    reg(0x00, 0x30); // MODE1: sleep + auto-increment; the prescaler is writable only asleep
    reg(0x01, 0x04); // MODE2: totem-pole outputs
    reg(0xfe,
        static_cast<uint8_t>(std::clamp(int(25000000. / (4096 * cfg_.frequency) - 1), 3, 255)));
    flush();
    reg(0x00, 0x20); // wake, then restart once the oscillator has settled
    usleep(5000);
    reg(0x00, 0xa0);
    flush();
}
void Pwm::flush() {
    if (!live_)
        return;
    std::array<uint8_t, 65> bytes{};
    bytes[0] = static_cast<uint8_t>(0x06 + 4 * lo_); // LEDn_ON_L of the first channel
    for (int i = lo_; i <= hi_; ++i) {
        const auto count = pulse_count(pulses[i], cfg_.frequency);
        uint8_t *off = &bytes[1 + 4 * (i - lo_) + 2]; // ON stays 0, OFF = count
        off[0] = count & 255;
        off[1] = count ? count >> 8 : 0x10; // count 0 sets the full-off bit
    }
    const size_t size = 1 + 4 * (hi_ - lo_ + 1);
    check_write(::write(fd_.get(), bytes.data(), size) == static_cast<ssize_t>(size),
                "PCA9685 burst write");
}
void Pwm::neutral() {
    center();
    flush();
}
Pwm::~Pwm() {
    if (live_ && fd_) {
        try {
            neutral();
        } catch (...) {
            std::fputs("Neutral write failed.\n", stderr);
        }
    }
}
ControlLock::ControlLock(const std::string &runtime) {
    std::filesystem::create_directories(runtime);
    struct stat st{};
    require_errno(lstat(runtime.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && st.st_uid == geteuid(),
                  "unsafe runtime directory");
    require_errno(chmod(runtime.c_str(), 0700) == 0, "runtime permissions");
    fd_ = Fd(::open((runtime + "/control.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW,
                    0600));
    require(fd_ && flock(fd_.get(), LOCK_EX | LOCK_NB) == 0,
            "another native controller is running");
}
} // namespace kaivuri
