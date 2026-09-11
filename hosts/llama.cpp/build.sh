#!/usr/bin/env bash
# Build upstream llama.cpp with the FlashRT SM120 windows, from scratch.
#
#   hosts/llama.cpp/build.sh <work dir> [llama.cpp git url or local clone]
#
# Clones llama.cpp at the pinned commit (PIN), applies the patch series,
# mounts this overlay at ggml/src/ggml-cuda/flashrt, and builds
# llama-bench, llama-server and test-backend-ops. Set FLASHRT_OVERLAY to
# use a different overlay checkout (default: the one this script is in).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
overlay="${FLASHRT_OVERLAY:-$(cd "$here/../.." && pwd)}"
work="${1:?usage: build.sh <work dir> [llama.cpp url]}"
src="${2:-https://github.com/ggml-org/llama.cpp}"
pin="$(cat "$here/PIN")"
jobs="${JOBS:-8}"

if [ ! -d "$work/.git" ]; then
    git clone -q "$src" "$work"
fi
cd "$work"
git fetch -q origin "$pin" 2>/dev/null || true
git checkout -q --detach "$pin"
git am -q --3way "$here"/patches/*.patch
[ -d "$overlay/third_party/cutlass/include" ] || { echo "overlay CUTLASS submodule missing: run 'git submodule update --init --recursive' in $overlay" >&2; exit 1; }
rm -rf ggml/src/ggml-cuda/flashrt
ln -s "$overlay" ggml/src/ggml-cuda/flashrt

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DGGML_CUDA_FLASHRT_SM120=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench llama-server test-backend-ops -j "$jobs"
echo "built: $work/build/bin"
