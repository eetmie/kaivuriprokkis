// Binary stdin/stdout adapter for the pinned FastDEM core. See replay.py.
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <fastdem/fastdem.hpp>

namespace {
constexpr int size = 160;
constexpr double cell = 0.01;

template <typename T>
bool read(T* buffer, std::size_t count) {
  return bool(std::cin.read(reinterpret_cast<char*>(buffer), count * sizeof(T)));
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  fastdem::ElevationMap map(size * cell, size * cell, cell, "undercarriage");
  auto config = fastdem::loadConfig(argv[1]);
  fastdem::FastDEM mapper(map, config);
  mapper.setMappingMode(fastdem::MappingMode::GLOBAL);
  mapper.setHeightFilter(-0.30f, 0.30f).setRangeFilter(0.10f, 0.90f);
  std::array<int, size * size> last;
  last.fill(-1);
  int frame = 0;
  mapper.onScanRasterized([&](const fastdem::PointCloud& cloud) {
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      const auto p = cloud.point(i);
      const int row = int(std::floor((0.8 - p.x()) / cell));
      const int col = int(std::floor((0.8 - p.y()) / cell));
      if (row >= 0 && row < size && col >= 0 && col < size)
        last[row * size + col] = frame;
    }
  });

  std::uint32_t scans;
  while (read(&scans, 1)) {
    if (scans > 2) return 3;
    for (std::uint32_t scan = 0; scan < scans; ++scan) {
      std::uint32_t count;
      std::array<double, 12> transform;
      if (!read(&count, 1) || count > 307200 ||
          !read(transform.data(), transform.size())) return 3;
      std::vector<float> points(3 * count);
      if (!read(points.data(), points.size()) && count != 0) return 3;
      if (count == 0) continue;
      fastdem::PointCloud cloud;
      cloud.reserve(count);
      for (std::size_t i = 0; i < count; ++i)
        cloud.add(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
      Eigen::Isometry3d sensor = Eigen::Isometry3d::Identity();
      sensor.matrix().topRows<3>() =
          Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(transform.data());
      mapper.integrate(cloud, sensor, Eigen::Isometry3d::Identity());
    }
    std::array<float, size * size> height;
    std::array<std::uint16_t, size * size> age;
    for (int row = 0; row < size; ++row) {
      for (int col = 0; col < size; ++col) {
        const int i = row * size + col;
        const nanogrid::Position pos(0.8 - (row + .5) * cell, 0.8 - (col + .5) * cell);
        height[i] = map.elevationAt(pos);
        age[i] = std::isfinite(height[i]) && last[i] >= 0
                     ? std::min(frame - last[i], 65534) : 65535;
      }
    }
    std::cout.write(reinterpret_cast<const char*>(height.data()), sizeof(height));
    std::cout.write(reinterpret_cast<const char*>(age.data()), sizeof(age)).flush();
    if (!std::cout) return 4;
    ++frame;
  }
  return std::cin.eof() ? 0 : 3;
}
