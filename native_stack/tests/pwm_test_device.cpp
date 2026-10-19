// Test-only I2C replacement. The real shared Pwm writes its exact register bytes
// into KAIVURI_TEST_PWM_TRACE; no /dev/i2c-* file is ever opened by these tests.
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>

namespace {
int test_fd = -1;
}
extern "C" int open(const char *path, int flags, ...) {
    const auto real_open =
        reinterpret_cast<int (*)(const char *, int, ...)>(dlsym(RTLD_NEXT, "open"));
    const char *trace = std::getenv("KAIVURI_TEST_PWM_TRACE");
    if (trace && std::strncmp(path, "/dev/i2c-", 9) == 0) {
        test_fd = real_open(trace, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        return test_fd;
    }
    mode_t mode = 0;
    if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, mode_t);
        va_end(args);
    }
    return real_open(path, flags, mode);
}
extern "C" int ioctl(int fd, unsigned long request, ...) noexcept {
    if (fd == test_fd && request == I2C_SLAVE)
        return 0;
    const auto real_ioctl =
        reinterpret_cast<int (*)(int, unsigned long, ...)>(dlsym(RTLD_NEXT, "ioctl"));
    va_list args;
    va_start(args, request);
    void *arg = va_arg(args, void *);
    va_end(args);
    return real_ioctl(fd, request, arg);
}
