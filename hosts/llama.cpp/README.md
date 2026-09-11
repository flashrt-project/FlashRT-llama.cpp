# Host: upstream llama.cpp (RTX 5090, SM120, Qwen3.6-35B-A3B)

Upstream host: https://github.com/ggml-org/llama.cpp at the commit in
`PIN`. The host-side changes are the patch series in `patches/`:

1. `GGML_CUDA_FLASHRT_SM120` option, the fuse-hook call sites in
   `ggml-cuda.cu` (compiled out unless the option is on), the NVFP4
   quantization target for `llama-quantize`, per-shape graph slots in
   `llama-context` (`LLAMA_GRAPH_SLOTS=N`, default 1 = stock), and the
   Qwen3-8B M=1 GEMV worklist in `test-backend-ops`.
2. A path-only commit that predates the overlay (kept for history).
3. The CMake block that consumes this overlay from
   `ggml/src/ggml-cuda/flashrt` (submodule) or `GGML_CUDA_FLASHRT_DIR`.

`build.sh <work dir>` reproduces the build from scratch: clone at the
pin, `git am` the series, mount the overlay, configure with
`-DGGML_CUDA_FLASHRT_SM120=ON`. The sm_120a pass is scoped to the three
FlashRT sources; no global `CMAKE_CUDA_FLAGS` needed.

Moving the pin: check out the new upstream commit, `git am --3way` the
series, resolve, `git format-patch` back into `patches/`, update `PIN`,
re-run the qualification gates (`../../qualification/`).

Runtime: `GGML_CUDA_FLASHRT_DISABLE=1` reproduces stock; the per-window
switches are documented in `../../USAGE.md`.
