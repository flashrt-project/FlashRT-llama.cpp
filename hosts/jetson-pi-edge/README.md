# Host: Jetson-PI-Edge (Jetson AGX Thor, SM110, pi0 / pi0.5)

Upstream host: https://github.com/PKU-SEC-Lab/Jetson-PI-Edge. The FlashRT
integration lives on the `feat/flashrt-thor-kernels` branch of
https://github.com/LiangSu8899/Jetson-PI-Edge (commit in `PIN`): the
fuse-hook call sites in `ggml-cuda.cu`, the CMake block, the NVFP4
quantization type, and the GR00T N1.7 / mtmd host changes. Those belong
to the host and are not carried here.

That branch still mounts the whole FlashRT repository as a submodule at
`ggml/src/ggml-cuda/flashrt/flashrt-public` and lists the adapter from
`flash_rt/structures/adapters/ggml/`, a path that no longer exists. The
patch here switches the CMake block to this overlay.

## Migrating the host branch (run on the host checkout)

```bash
git checkout feat/flashrt-thor-kernels
git submodule deinit -f ggml/src/ggml-cuda/flashrt/flashrt-public
git rm -f ggml/src/ggml-cuda/flashrt/flashrt-public ggml/src/ggml-cuda/flashrt/README.md
git submodule add https://github.com/flashrt-project/FlashRT-llama.cpp ggml/src/ggml-cuda/flashrt
git submodule update --init --recursive
git apply <overlay>/hosts/jetson-pi-edge/0001-ggml-cuda-consume-the-FlashRT-llama.cpp-overlay.patch
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FLASHRT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

Options stay as before: `GGML_CUDA_FLASHRT` (OFF by default),
`GGML_CUDA_FLASHRT_CUTLASS_DIR` (override CUTLASS). The old
`GGML_CUDA_FLASHRT_PUBLIC_DIR` becomes `GGML_CUDA_FLASHRT_DIR`, pointing
at an overlay checkout instead of a FlashRT checkout.

Status: the patch is a path-only rewrite of the block that was building
on 2026-08-25 and has not been rebuilt on a Thor since the move; the
first build on device is the acceptance run (qualification gates in
`../../qualification/`, [testing protocol](../../docs/testing.md)).
