#pragma once
#include "kaivuri/core.hpp"
#include <linux/input.h>

namespace kaivuri {
class Pad {
  public:
    int fd = -1;
    bool virtual_pad = false;
    std::array<input_absinfo, ABS_CNT> axes{};
    std::array<bool, KEY_CNT> keys{};
    explicit Pad(const std::string &path);
    ~Pad();
    Pad(const Pad &) = delete;
    Pad &operator=(const Pad &) = delete;
    void poll_events();
    double axis(int code, bool trigger = false) const;
    Commands action(bool slew, bool tracks) const;
};
} // namespace kaivuri
