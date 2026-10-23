#pragma once
#include "kaivuri/core.hpp"
#include "kaivuri/util.hpp"
#include <linux/input.h>

namespace kaivuri {
// Xbox-layout evdev gamepad, or a virtual pad FIFO (tools/virtual_gamepad.py).
class Pad {
  public:
    Fd fd;
    bool virtual_pad = false;
    std::array<input_absinfo, ABS_CNT> axes{};
    std::array<bool, KEY_CNT> keys{};
    explicit Pad(const std::string &path);
    void poll_events();
    double axis(int code, bool trigger = false) const;
    Commands action(bool slew, bool tracks) const;
};
} // namespace kaivuri
