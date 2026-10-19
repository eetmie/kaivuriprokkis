#include "kaivuri/frame_store.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>
using namespace kaivuri;
int main() {
    try {
        const std::string path = "/tmp/kaivuri-frame-test-" + std::to_string(getpid());
        struct Cleanup {
            std::string path;
            ~Cleanup() {
                unlink(path.c_str());
                unlink((path + ".lock").c_str());
            }
        } cleanup{path};
        FrameWriter writer(path, 128, 128, 1);
        void *reader = kaivuri_frame_open(path.c_str());
        if (!reader)
            throw std::runtime_error("open reader");
        struct Close {
            void *p;
            ~Close() { kaivuri_frame_close(p); }
        } close{reader};
        FrameMetadata m{};
        std::vector<uint8_t> pixels(128 * 128);
        if (kaivuri_frame_read(reader, &m, pixels.data(), pixels.size()) != 0)
            throw std::runtime_error("uninitialized frame should be unavailable");
        bool rejected = false;
        try {
            FrameWriter duplicate(path, 128, 128, 1);
        } catch (...) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("two writers acquired same store");
        std::atomic<bool> finished{false};
        std::thread producer([&] {
            std::vector<uint8_t> source(pixels.size());
            for (uint64_t i = 1; i <= 20000; ++i) {
                std::fill(source.begin(), source.end(), uint8_t(i % 256));
                FrameMetadata metadata{};
                metadata.sequence = i;
                metadata.width = 128;
                metadata.height = 128;
                metadata.stride = 128;
                metadata.pixel_bytes = source.size();
                writer.publish(metadata, source.data());
            }
            finished = true;
        });
        bool coherent = true;
        uint64_t snapshots = 0;
        while (!finished) {
            if (kaivuri_frame_read(reader, &m, pixels.data(), pixels.size()) == 1) {
                ++snapshots;
                coherent = coherent &&
                           std::all_of(pixels.begin(), pixels.end(),
                                       [&](uint8_t value) { return value == m.sequence % 256; });
            }
        }
        producer.join();
        if (!coherent || snapshots == 0)
            throw std::runtime_error("torn snapshot or no concurrent snapshots");
        if (kaivuri_frame_read(reader, &m, pixels.data(), 1) != -1)
            throw std::runtime_error("undersized destination accepted");
        if (kaivuri_frame_read(reader, &m, nullptr, 0) != 1)
            throw std::runtime_error("metadata-only read failed");
        {
            // Z16 depth store: two bytes per pixel, packed rows.
            const std::string depth_path = path + "-depth";
            Cleanup depth_cleanup{depth_path};
            FrameWriter depth(depth_path, 64, 48, 0, 2);
            void *depth_reader = kaivuri_frame_open(depth_path.c_str());
            if (!depth_reader)
                throw std::runtime_error("open depth reader");
            Close depth_close{depth_reader};
            std::vector<uint16_t> z(64 * 48);
            for (size_t i = 0; i < z.size(); ++i)
                z[i] = uint16_t(i * 7);
            FrameMetadata d{};
            d.sequence = 1;
            d.width = 64;
            d.height = 48;
            d.stride = 64;
            d.pixel_bytes = 64 * 48;
            bool y8_rejected = false;
            try {
                depth.publish(d, z.data());
            } catch (...) {
                y8_rejected = true;
            }
            d.stride = 128;
            d.pixel_bytes = 128 * 48;
            if (!y8_rejected || !depth.publish(d, z.data()))
                throw std::runtime_error("Z16 geometry not enforced");
            std::vector<uint16_t> back(z.size());
            if (kaivuri_frame_read(depth_reader, &m, back.data(), back.size() * 2) != 1 ||
                back != z || m.pixel_bytes != 128 * 48)
                throw std::runtime_error("Z16 round trip failed");
        }
        std::printf("Concurrent frame snapshot and Z16 checks passed (%llu snapshots)\n",
                    (unsigned long long)snapshots);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Frame test failed: %s\n", e.what());
        return 1;
    }
}
