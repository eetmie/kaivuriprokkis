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
Pad::Pad(const std::string &path) {
    fd = open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0)
        throw std::runtime_error("cannot open gamepad: " + path);
    struct stat st{};
    if (fstat(fd, &st) == 0 && S_ISFIFO(st.st_mode)) {
        // Virtual pad (tools/virtual_gamepad.py): raw input_event structs through a FIFO,
        // with the xpad Xbox ranges, because this kernel has no uinput. Everything after
        // the event read is the real path.
        virtual_pad = true;
        for (int code : {ABS_X, ABS_Y, ABS_RX, ABS_RY})
            axes[code] = input_absinfo{0, -32768, 32767, 16, 128, 0};
        for (int code : {ABS_Z, ABS_RZ})
            axes[code] = input_absinfo{0, 0, 1023, 0, 0, 0};
        return;
    }
    for (int code : {ABS_X, ABS_Y, ABS_RX, ABS_RY, ABS_Z, ABS_RZ})
        if (ioctl(fd, EVIOCGABS(code), &axes[code]) < 0 ||
            axes[code].maximum <= axes[code].minimum) {
            close(fd);
            throw std::runtime_error("gamepad must expose Xbox X/Y/RX/RY/Z/RZ axes");
        }
    unsigned char bits[(KEY_CNT + 7) / 8]{};
    if (ioctl(fd, EVIOCGKEY(sizeof(bits)), bits) < 0) {
        close(fd);
        throw std::runtime_error("cannot read gamepad buttons");
    }
    for (int i = 0; i < KEY_CNT; ++i)
        keys[i] = (bits[i / 8] >> (i % 8)) & 1;
}
Pad::~Pad() {
    if (fd >= 0)
        close(fd);
}
void Pad::poll_events() {
    input_event e{};
    for (;;) {
        ssize_t n = read(fd, &e, sizeof(e));
        if (n < 0 && (errno == EAGAIN || errno == EINTR))
            return;
        if (n != sizeof(e))
            throw std::runtime_error("gamepad disconnected");
        if (e.type == EV_SYN && e.code == SYN_DROPPED)
            throw std::runtime_error("gamepad event queue overrun");
        if (e.type == EV_ABS && e.code < ABS_CNT)
            axes[e.code].value = e.value;
        if (e.type == EV_ABS && e.code == ABS_HAT0Y) {
            keys[BTN_DPAD_UP] = e.value < 0;
            keys[BTN_DPAD_DOWN] = e.value > 0;
        }
        if (e.type == EV_ABS && e.code == ABS_HAT0X) {
            keys[BTN_DPAD_LEFT] = e.value < 0;
            keys[BTN_DPAD_RIGHT] = e.value > 0;
        }
        if (e.type == EV_KEY && e.code < KEY_CNT)
            keys[e.code] = e.value != 0;
    }
}
double Pad::axis(int code, bool trigger) const {
    auto a = axes[code];
    double x = trigger ? double(a.value - a.minimum) / (a.maximum - a.minimum)
                       : double(a.value) / std::max(std::abs(a.minimum), std::abs(a.maximum) + 1);
    x = std::clamp(x, trigger ? 0. : -1., 1.);
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
