#!/usr/bin/env bash
set -euo pipefail
stack_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_revision=34d6c778e1134d8505adcd56bb57acbe7598a459
sdk_source="$stack_dir/.deps/librealsense-2.58.4"
sdk_prefix="$stack_dir/.deps/realsense-2.58.4"
if [[ ! -d "$sdk_source/.git" ]]; then
    git clone --depth 1 --branch v2.58.4 https://github.com/realsenseai/librealsense.git "$sdk_source"
fi
[[ "$(git -C "$sdk_source" rev-parse HEAD)" == "$sdk_revision" ]] || { echo 'SDK revision mismatch' >&2; exit 1; }
cmake -S "$sdk_source" -B "$sdk_source/build-native" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$sdk_prefix" \
    -DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_DDS=OFF \
    -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENMP=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_BINDINGS=OFF
cmake --build "$sdk_source/build-native" --parallel "${BUILD_JOBS:-2}"
# Upstream installs unconditionally invoke ldconfig even for a private prefix.
# Omit that generated install step; loader paths come from the application's RPATH.
python3 - "$sdk_source/build-native/cmake_install.cmake" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
script = path.read_text()
assert script.count("execute_process(COMMAND ldconfig)") == 1
path.write_text(script.replace("execute_process(COMMAND ldconfig)", "# Private prefix: no system ldconfig"))
PY
cmake --install "$sdk_source/build-native"
printf 'Isolated SDK installed at %s\n' "$sdk_prefix"
