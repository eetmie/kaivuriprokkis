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
void check(bool value, const char *what) {
    if (!value)
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}
} // namespace
void Pwm::reg(uint8_t address, uint8_t value) {
    const uint8_t bytes[]{address, value};
    check(::write(fd_, bytes, 2) == 2, "PCA9685 register write");
}
Pwm::Pwm(const Config &config, bool live) : cfg_(config), live_(live) {
    for (const auto &v : cfg_.valves)
        pulses[v.channel] = v.center;
    pulses[cfg_.pump_channel] = cfg_.pump_min;
    if (!live_)
        return;
    try {
        fd_ = ::open(("/dev/i2c-" + std::to_string(cfg_.i2c_bus)).c_str(), O_RDWR | O_CLOEXEC);
        check(fd_ >= 0, "open I2C");
        check(ioctl(fd_, I2C_SLAVE, cfg_.address) >= 0, "select PCA9685");
        reg(0x00, 0x30);
        reg(0x01, 0x04);
        reg(0xfe,
            static_cast<uint8_t>(std::clamp(int(25000000. / (4096 * cfg_.frequency) - 1), 3, 255)));
        flush();
        reg(0x00, 0x20);
        usleep(5000);
        reg(0x00, 0xa0);
        flush();
    } catch (...) {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        throw;
    }
}
void Pwm::flush() {
    if (!live_)
        return;
    int lo = 15, hi = 0;
    for (const auto &v : cfg_.valves) {
        lo = std::min(lo, v.channel);
        hi = std::max(hi, v.channel);
    }
    lo = std::min(lo, cfg_.pump_channel);
    hi = std::max(hi, cfg_.pump_channel);
    std::array<uint8_t, 65> bytes{};
    bytes[0] = static_cast<uint8_t>(0x06 + 4 * lo);
    for (int i = lo; i <= hi; ++i) {
        const auto count = pulse_count(pulses[i], cfg_.frequency);
        const size_t j = 1 + 4 * (i - lo);
        bytes[j + 2] = count & 255;
        bytes[j + 3] = count >> 8;
        if (count == 0)
            bytes[j + 3] = 0x10;
    }
    const size_t size = 1 + 4 * (hi - lo + 1);
    check(::write(fd_, bytes.data(), size) == static_cast<ssize_t>(size), "PCA9685 burst write");
}
void Pwm::neutral() {
    for (const auto &v : cfg_.valves)
        pulses[v.channel] = v.center;
    pulses[cfg_.pump_channel] = cfg_.pump_min;
    flush();
}
Pwm::~Pwm() {
    if (live_ && fd_ >= 0) {
        try {
            neutral();
        } catch (...) {
            std::fputs("Neutral write failed.\n", stderr);
        }
        ::close(fd_);
    }
}
ControlLock::ControlLock(const std::string &runtime) {
    std::filesystem::create_directories(runtime);
    struct stat st{};
    check(lstat(runtime.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && st.st_uid == geteuid(),
          "unsafe runtime directory");
    check(chmod(runtime.c_str(), 0700) == 0, "runtime permissions");
    fd_ = ::open((runtime + "/control.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW,
                 0600);
    if (fd_ < 0 || flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("another native controller is running");
    }
}
ControlLock::~ControlLock() {
    if (fd_ >= 0)
        ::close(fd_);
}
} // namespace kaivuri
