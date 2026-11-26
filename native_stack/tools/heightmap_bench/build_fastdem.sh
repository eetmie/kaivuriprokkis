#!/bin/bash
# Build fastdem_replay against the FastDEM commit the robot runs, with its core fixes (fastdem-core.patch).
# usage: build_fastdem.sh [WORK_DIR]   needs cmake, g++, Eigen3, yaml-cpp, spdlog, fmt
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
w=${1:-$here/.fastdem}
if [ ! -d "$w/FastDEM" ]; then
  git clone -q https://github.com/Ikhyeon-Cho/FastDEM.git "$w/FastDEM"
  git -C "$w/FastDEM" checkout -q 5cd3e2552dba62be14cd8fe83d5c81261eafd12e
  git -C "$w/FastDEM" apply "$here/fastdem-core.patch"
fi
f=$w/FastDEM/fastdem b=$w/build
cmake -S "$f" -B "$b" -DCMAKE_BUILD_TYPE=Release -DFASTDEM_BUILD_EXAMPLES=OFF -DFASTDEM_BUILD_TOOLS=OFF >/dev/null
cmake --build "$b" -j"$(nproc)" >/dev/null
g++ -O3 -std=c++17 -I/usr/include/eigen3 -I"$f/include" -I"$f/lib/nanoPCL/include" \
  -I"$b/_deps/nanogrid-src/include" "$here/fastdem_replay.cpp" "$b/libfastdem.a" \
  "$b/_deps/nanogrid-build/libnanoGrid.a" -lyaml-cpp -lspdlog -lfmt -o "$here/fastdem_replay"
echo "built $here/fastdem_replay"
