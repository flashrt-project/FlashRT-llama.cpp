# FlashRT-llama.cpp

**Native CUDA acceleration for llama.cpp, powered by FlashRT.**

[FlashRT](https://github.com/flashrt-project/FlashRT) ·
[llama.cpp](https://github.com/ggml-org/llama.cpp) ·
[Build & run](docs/usage.md) ·
[Benchmarks](hosts/llama.cpp-flint/README.md) ·
[Paper](https://arxiv.org/abs/2610.24349)

FlashRT-llama.cpp brings model-specific fused kernels and graph optimizations
to llama.cpp's ggml CUDA backend. llama.cpp continues to handle GGUF loading,
KV caches, tokenization, sampling and serving. This repository supplies the
native integration, selected FlashRT kernels and reproducible host patches.

| Project | Role |
|---|---|
| [FlashRT](https://github.com/flashrt-project/FlashRT) | The inference engine and upstream source of the reused kernels |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | The model runtime, ggml backend, CLI and server |
| **FlashRT-llama.cpp** | CUDA fusion windows and build-time integration between them |

## Supported integrations

| Host | Model | Hardware | Guide |
|---|---|---|---|
| llama.cpp | Recursive Flint NVFP4 | RTX 5090 / SM120 | [Flint](hosts/llama.cpp-flint/README.md) |
| llama.cpp | Qwen3.6-35B-A3B | RTX 5090 / SM120 | [Qwen](hosts/llama.cpp/README.md) |
| Jetson-PI-Edge | π0 / π0.5 | Jetson AGX Thor / SM110 | [Thor integration](hosts/jetson-pi-edge/README.md) |

Each host has a pinned revision and patch series. The Thor integration's
on-device revalidation status is recorded in its guide. Optimized windows
are enabled by the host's build option; unmatched shapes use ggml kernels.

## Quick start: Flint on RTX 5090

Requires Linux, a CUDA toolchain, CMake and Git. The measured build uses
CUDA 13.0. Set `MODEL` to the official Flint NVFP4 GGUF file.

```bash
git clone --recursive https://github.com/flashrt-project/FlashRT-llama.cpp
cd FlashRT-llama.cpp
bash hosts/llama.cpp-flint/build.sh ../llama.cpp-flint
../llama.cpp-flint/build/bin/llama-cli -m "$MODEL" -ngl 99 -fa on
```

The script builds a separate pinned llama.cpp checkout with the integration
compiled into ggml-cuda. Inference runs in C++/CUDA; optional development
and artifact-generation tools use Python. Serving flags and runtime switches
are in [Usage](docs/usage.md).

## Performance

Matched stock and FlashRT runs on RTX 5090, CUDA 13.0: 1K prompt + 1K decode
tokens per sequence, f16 KV, `-ub 512`, 8 CPU threads, median of two fresh
processes. Prefill and decode are timed separately.

**Decode**

| Parallel sequences | Stock tok/s | FlashRT tok/s | Speedup |
|---:|---:|---:|---:|
| 1 | 230.5 | 517.4 | 2.24× |
| 64 | 1625.8 | 2482.7 | 1.53× |

**Prefill**

| Parallel sequences | Stock tok/s | FlashRT tok/s | Speedup |
|---:|---:|---:|---:|
| 1 | 4981.5 | 5450.7 | 1.09× |
| 64 | 6319.0 | 7124.0 | 1.13× |

[Full results, baseline definition and reproduction commands](hosts/llama.cpp-flint/README.md) ·
[Correctness protocol](docs/testing.md)

## Repository layout

```text
src/             CUDA implementations and ggml fusion windows
include/         Native launch interface
hosts/           Host pins, patches and build scripts
docs/            Usage, validation, development and upstream credits
vendor/          Pinned FlashRT kernel sources and sync tools
third_party/     CUTLASS submodule
fa4_aot/         Generated FlashAttention-4 modules
qualification/   Model pins and correctness gates
tools/           Development utilities
```

Start with [Usage](docs/usage.md), [Testing](docs/testing.md) or
[Development](docs/development.md).

## Citation and credits

Please cite the [FlashRT paper](https://arxiv.org/abs/2610.24349) and
acknowledge [llama.cpp](https://github.com/ggml-org/llama.cpp) when using this
integration. Machine-readable metadata is in [CITATION.cff](CITATION.cff).

This integration uses [Apache-2.0](LICENSE). Upstream components retain
their licenses and credits; see [Upstream projects](docs/credits.md) for
FlashRT, llama.cpp, CUTLASS and FlashAttention.
