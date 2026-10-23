#pragma once
#include "kaivuri/core.hpp"
#include "kaivuri/util.hpp"

namespace kaivuri {
// Shared PCA9685 register writer; the caller retains its existing live/dry policy. A dry Pwm
// only tracks pulses; a live one writes neutral on destruction.
class Pwm {
    Config cfg_;
    Fd fd_;
    bool live_;
    int lo_ = 15, hi_ = 0; // channel span written by each burst
    void reg(uint8_t address, uint8_t value);
    void center();

  public:
    std::array<double, 16> pulses{};
    Pwm(const Config &config, bool live);
    ~Pwm();
    Pwm(const Pwm &) = delete;
    Pwm &operator=(const Pwm &) = delete;
    void flush();
    void neutral();
};

// Exclusive per-user hardware lock (runtime/control.lock): two native daemons cannot own PWM.
// Creates the runtime directory and requires it to be a directory owned by this user (0700).
class ControlLock {
    Fd fd_;

  public:
    explicit ControlLock(const std::string &runtime);
};
} // namespace kaivuri
