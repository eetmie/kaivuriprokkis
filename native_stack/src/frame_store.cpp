#include "kaivuri/frame_store.hpp"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>
#include <pthread.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace {
constexpr size_t header_bytes = 4096;
constexpr uint64_t magic = 0x31594b5652464d49ULL;
struct Header {
    uint64_t magic_value;
    uint32_t version, capacity;
    pthread_mutex_t mutex;
    FrameMetadata metadata;
};
static_assert(sizeof(Header) < header_bytes);
struct Reader {
    void *mapping;
    size_t size;
    int fd;
};
bool acquire(pthread_mutex_t *mutex) {
    int r = pthread_mutex_trylock(mutex);
    if (r == EOWNERDEAD) {
        pthread_mutex_consistent(mutex);
        return true;
    }
    return r == 0;
}
void verify(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}
} // namespace
kaivuri::FrameWriter::FrameWriter(const std::string &path, uint32_t width, uint32_t height,
                                  uint32_t ir_index, uint32_t bytes_per_pixel)
    : bytes_per_pixel_(bytes_per_pixel) {
    if (width < 1 || height < 1 || (bytes_per_pixel != 1 && bytes_per_pixel != 2) ||
        uint64_t(width) * height * bytes_per_pixel > 16 * 1024 * 1024)
        throw std::runtime_error("invalid frame dimensions");
    // Atomic rename preserves the previous mapping for already connected readers on restart.
    std::string temporary = path + ".new." + std::to_string(getpid());
    try {
        lock_fd_ =
            ::open((path + ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        verify(lock_fd_ >= 0 && flock(lock_fd_, LOCK_EX | LOCK_NB) == 0,
               "camera already has a writer");
        fd_ = ::open(temporary.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        verify(fd_ >= 0, "create frame store");
        size_ = header_bytes + size_t(width) * height * bytes_per_pixel;
        verify(ftruncate(fd_, size_) == 0, "size frame store");
        mapping_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            throw std::runtime_error("map frame store");
        }
        auto *h = new (mapping_) Header{};
        h->magic_value = magic;
        h->version = 1;
        h->capacity = width * height * bytes_per_pixel;
        pthread_mutexattr_t attr;
        verify(pthread_mutexattr_init(&attr) == 0, "mutex attributes");
        int result = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        if (!result)
            result = pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
        if (!result)
            result = pthread_mutex_init(&h->mutex, &attr);
        pthread_mutexattr_destroy(&attr);
        verify(result == 0, "shared robust mutex");
        h->metadata.width = width;
        h->metadata.height = height;
        h->metadata.stride = width * bytes_per_pixel;
        h->metadata.ir_index = ir_index;
        verify(rename(temporary.c_str(), path.c_str()) == 0, "publish frame store");
    } catch (...) {
        if (mapping_)
            munmap(mapping_, size_);
        if (fd_ >= 0)
            ::close(fd_);
        if (lock_fd_ >= 0)
            ::close(lock_fd_);
        unlink(temporary.c_str());
        throw;
    }
}
kaivuri::FrameWriter::~FrameWriter() {
    if (mapping_)
        munmap(mapping_, size_);
    if (fd_ >= 0)
        ::close(fd_);
    if (lock_fd_ >= 0)
        ::close(lock_fd_);
}
bool kaivuri::FrameWriter::publish(const FrameMetadata &m, const void *pixels) {
    auto *h = static_cast<Header *>(mapping_);
    if (!pixels || m.pixel_bytes > h->capacity || m.stride != m.width * bytes_per_pixel_ ||
        m.pixel_bytes != uint64_t(m.stride) * m.height)
        throw std::runtime_error("invalid packed frame");
    if (!acquire(&h->mutex))
        return false;
    // Copy pixels first: a recovered owner-dead lock must not expose partially copied pixels.
    h->metadata.sequence = 0;
    std::memcpy(static_cast<uint8_t *>(mapping_) + header_bytes, pixels, m.pixel_bytes);
    h->metadata = m;
    pthread_mutex_unlock(&h->mutex);
    return true;
}
extern "C" void *kaivuri_frame_open(const char *path) {
    if (!path)
        return nullptr;
    int fd = ::open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0)
        return nullptr;
    struct stat st{};
    if (fstat(fd, &st) < 0 || st.st_size < static_cast<off_t>(header_bytes) ||
        st.st_size > static_cast<off_t>(header_bytes + 16 * 1024 * 1024)) {
        ::close(fd);
        return nullptr;
    }
    void *mapping = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        ::close(fd);
        return nullptr;
    }
    auto *h = static_cast<Header *>(mapping);
    if (h->magic_value != magic || h->version != 1 || h->capacity != st.st_size - header_bytes) {
        munmap(mapping, st.st_size);
        ::close(fd);
        return nullptr;
    }
    return new Reader{mapping, static_cast<size_t>(st.st_size), fd};
}
extern "C" int kaivuri_frame_read(void *handle, FrameMetadata *m, void *pixels, size_t capacity) {
    if (!handle || !m)
        return -1;
    auto *r = static_cast<Reader *>(handle);
    auto *h = static_cast<Header *>(r->mapping);
    int result = pthread_mutex_trylock(&h->mutex);
    if (result == EOWNERDEAD) {
        h->metadata.sequence = 0;
        pthread_mutex_consistent(&h->mutex);
    } else if (result != 0)
        return 0;
    int ok = 0;
    if (h->metadata.sequence && h->metadata.pixel_bytes <= h->capacity) {
        if (pixels && h->metadata.pixel_bytes > capacity)
            ok = -1;
        else {
            *m = h->metadata;
            if (pixels)
                std::memcpy(pixels, static_cast<uint8_t *>(r->mapping) + header_bytes,
                            m->pixel_bytes);
            ok = 1;
        }
    }
    pthread_mutex_unlock(&h->mutex);
    return ok;
}
extern "C" void kaivuri_frame_close(void *handle) {
    if (!handle)
        return;
    auto *r = static_cast<Reader *>(handle);
    munmap(r->mapping, r->size);
    ::close(r->fd);
    delete r;
}
