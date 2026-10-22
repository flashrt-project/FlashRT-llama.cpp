#!/usr/bin/env bash
# Build llama.cpp with Flint support (upstream PR #30928) and the FlashRT SM120
# Flint windows, from scratch.
#
#   hosts/llama.cpp-flint/build.sh <work dir> [llama.cpp git url or local clone]
#
# Clones llama.cpp, fetches the Flint PR head, checks out the pinned commit
# (PIN), applies the patch series, mounts this overlay at
# ggml/src/ggml-cuda/flashrt and builds llama-server, llama-bench,
# llama-cli, llama-perplexity and llama-batched-bench. Set FLASHRT_OVERLAY to use a different
# overlay checkout (default: the one this script is in).
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
# the pin is the Flint PR head; fetch it by hash, falling back to the PR ref
git cat-file -e "$pin^{commit}" 2>/dev/null || git fetch -q origin "$pin" 2>/dev/null \
    || git fetch -q origin pull/30928/head
git checkout -q --detach "$pin"
git am -q --3way "$here"/patches/*.patch
rm -rf ggml/src/ggml-cuda/flashrt
ln -s "$overlay" ggml/src/ggml-cuda/flashrt

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DGGML_CUDA_FLASHRT_SM120=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-bench llama-cli llama-perplexity llama-batched-bench -j "$jobs"
echo "built: $work/build/bin"
