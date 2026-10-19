#pragma once
#include "kaivuri/core.hpp"

namespace kaivuri {
// Shared register writer; the caller retains its existing live/dry policy.
class Pwm {
    Config cfg_;
    int fd_ = -1;
    bool live_;
    void reg(uint8_t address, uint8_t value);

  public:
    std::array<double, 16> pulses{};
    Pwm(const Config &config, bool live);
    ~Pwm();
    Pwm(const Pwm &) = delete;
    Pwm &operator=(const Pwm &) = delete;
    void flush();
    void neutral();
};

class ControlLock {
    int fd_ = -1;

  public:
    explicit ControlLock(const std::string &runtime);
    ~ControlLock();
    ControlLock(const ControlLock &) = delete;
    ControlLock &operator=(const ControlLock &) = delete;
};
} // namespace kaivuri
