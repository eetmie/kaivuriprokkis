#include "kaivuri/gamepad.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kaivuri {
Pad::Pad(const std::string &path) : fd(::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC)) {
    require(bool(fd), "cannot open gamepad: " + path);
    struct stat st{};
    if (fstat(fd.get(), &st) == 0 && S_ISFIFO(st.st_mode)) {
        // Virtual pad: raw input_event structs through a FIFO, with the xpad Xbox ranges,
        // because this kernel has no uinput. Everything after the event read is the real path.
        virtual_pad = true;
        for (int code : {ABS_X, ABS_Y, ABS_RX, ABS_RY})
            axes[code] = input_absinfo{0, -32768, 32767, 16, 128, 0};
        for (int code : {ABS_Z, ABS_RZ})
            axes[code] = input_absinfo{0, 0, 1023, 0, 0, 0};
        return;
    }
    for (int code : {ABS_X, ABS_Y, ABS_RX, ABS_RY, ABS_Z, ABS_RZ})
        require(ioctl(fd.get(), EVIOCGABS(code), &axes[code]) >= 0 &&
                    axes[code].maximum > axes[code].minimum,
                "gamepad must expose Xbox X/Y/RX/RY/Z/RZ axes");
    unsigned char bits[(KEY_CNT + 7) / 8]{};
    require(ioctl(fd.get(), EVIOCGKEY(sizeof(bits)), bits) >= 0, "cannot read gamepad buttons");
    for (int i = 0; i < KEY_CNT; ++i)
        keys[i] = (bits[i / 8] >> (i % 8)) & 1;
}
void Pad::poll_events() {
    input_event e{};
    for (;;) {
        const ssize_t n = ::read(fd.get(), &e, sizeof(e));
        if (n < 0 && (errno == EAGAIN || errno == EINTR))
            return;
        if (n != sizeof(e)) // explicit throws: require() would build a string per event
            throw std::runtime_error("gamepad disconnected");
        if (e.type == EV_SYN && e.code == SYN_DROPPED)
            throw std::runtime_error("gamepad event queue overrun");
        if (e.type == EV_KEY && e.code < KEY_CNT)
            keys[e.code] = e.value != 0;
        if (e.type != EV_ABS)
            continue;
        if (e.code < ABS_CNT)
            axes[e.code].value = e.value;
        if (e.code == ABS_HAT0Y) { // the d-pad reports as a hat; mirror it into the buttons
            keys[BTN_DPAD_UP] = e.value < 0;
            keys[BTN_DPAD_DOWN] = e.value > 0;
        } else if (e.code == ABS_HAT0X) {
            keys[BTN_DPAD_LEFT] = e.value < 0;
            keys[BTN_DPAD_RIGHT] = e.value > 0;
        }
    }
}
double Pad::axis(int code, bool trigger) const {
    const auto &a = axes[code];
    double x; // triggers map to 0..1, sticks to -1..1
    if (trigger)
        x = std::clamp(double(a.value - a.minimum) / (a.maximum - a.minimum), 0., 1.);
    else
        x = std::clamp(double(a.value) / std::max(std::abs(a.minimum), std::abs(a.maximum) + 1),
                       -1., 1.);
    return std::abs(x) < .25 ? 0 : std::copysign((std::abs(x) - .25) / .75, x);
}
Commands Pad::action(bool slew, bool tracks) const {
    // Same signs and deadzone as the existing LocalGamepadInput/manual_action_from_axes.
    return {slew ? -axis(ABS_X) : 0,
            -axis(ABS_RY),
            axis(ABS_Y),
            -axis(ABS_RX),
            tracks ? axis(ABS_Z, true) * (keys[BTN_TL] ? -1 : 1) : 0,
            tracks ? axis(ABS_RZ, true) * (keys[BTN_TR] ? -1 : 1) : 0};
}
} // namespace kaivuri
