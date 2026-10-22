# Host: llama.cpp + Flint (RTX 5090, SM120, official NVFP4 GGUF)

Host: https://github.com/ggml-org/llama.cpp, Flint model support from
upstream PR #30928 at the commit in `PIN`. The host-side changes are the
patch series in `patches/`:

1. Jinja hexadecimal escapes (backport of upstream PR #30929, needed by
   the Flint chat template).
2. `GGML_CUDA_FLASHRT_SM120`: the CMake block that compiles
   `src/fr_win_flint_sm120.cu` from the overlay (submodule at
   `ggml/src/ggml-cuda/flashrt` or `GGML_CUDA_FLASHRT_DIR`), the fuse-hook
   call at the top of `ggml_cuda_try_fuse` and an eager pre-capture hook in
   graph compute. Compiled out unless the option is on;
   `GGML_CUDA_FLASHRT_DISABLE=1` is stock at runtime.
3. Small host-to-device tensor uploads without a blocking stream sync
   (`GGML_CUDA_SYNC_UPLOADS=1` restores it).
4. A multi-warp `mm_ids_helper` for large token counts
   (`GGML_CUDA_MMID_SINGLE_WARP=1` restores the stock kernel).
5. The tensor-core FA kernel skips fully masked KV tiles below and inside
   the sliding window (`GGML_CUDA_FA_KV_MIN=0` disables).
6. The vector FA kernel skips fully masked 32-row warp groups (build with
   `GGML_CUDA_FA_VEC_NO_MASKED_SKIP` to disable): decode speed no longer
   depends on `-ub`, which sizes the sliding-window ring.
7. The reused rows of the KQ mask are filled in parallel
   (`LLAMA_KQ_MASK_SERIAL=1` restores the serial fill).
8. A scheduler re-reserve that keeps the node budget reuses the scheduler
   and its compute buffers. The server sets a backend sampler per request
   (`--backend-sampling`), which used to reallocate every compute buffer
   before each prompt (`LLAMA_SCHED_RESERVE_NEW=1` restores that).
9. Keep the scheduler while its node budget fits, growing with 25% headroom
   (`LLAMA_SCHED_NODE_HEADROOM`). This covers per-request sampler-count
   changes that patch 0008's exact budget match still reallocated for.

Patches 3–9 change no values: every patch keeps logits bit-identical.

`build.sh <work dir> [llama.cpp url or local clone]` reproduces the build
from scratch: clone, fetch the PR head, check out the pin, `git am` the
series, mount the overlay, configure with `-DGGML_CUDA_FLASHRT_SM120=ON`.
Recommended serving flags and the runtime switches are in `../../docs/usage.md`.

## Numerics

The original M = 1 path and stock fallbacks are exact: with
`GGML_FLASHRT_NO_BATCHED=1`, decode and prefill logits are bit-identical to
stock llama.cpp at the same `-ub`, checked against the unmodified stock
build with an FNV-1a hash of every position's logits: 1024 decode
positions; 99,673 decode positions on a code corpus; 4096 decode positions
with a 4096-ubatch context; 8192 prefill tokens at `-ub` 512, 2048, 4096,
8192 and 3000 (a partial last ubatch); 512 prefill tokens at ubatches of
2, 4, 8 and 32. Real-prompt greedy outputs are token-identical to stock. The opt-in decode attention
(`FRT_ATTNDEC=1`) is non-bitwise; its quality is gated against stock's own
path variation (see `../../docs/testing.md`). The default batched windows also change fp32
summation order and use the sampled double-precision host-reference gate
in `../../docs/testing.md`.

## Decode throughput

| Parallel sequences | Stock decode tok/s | FlashRT decode tok/s | Speedup | Improvement |
|---:|---:|---:|---:|---:|
| 1 | 230.5 | 517.4 | 2.24x | +124.5% |
| 8 | 1018.9 | 1459.2 | 1.43x | +43.2% |
| 16 | 924.0 | 1654.3 | 1.79x | +79.0% |
| 32 | 1301.4 | 2103.3 | 1.62x | +61.6% |
| 64 | 1625.8 | 2482.7 | 1.53x | +52.7% |

Metric: aggregate decode throughput = `parallel_sequences * 1000 / t_tg`, the `speed_tg` (`S_TG t/s`) field from `llama-batched-bench`. Prompt processing is outside this timer.

## Prefill throughput

| Parallel sequences | Stock prefill tok/s | FlashRT prefill tok/s | Speedup | Improvement |
|---:|---:|---:|---:|---:|
| 1 | 4981.5 | 5450.7 | 1.09x | +9.4% |
| 8 | 6438.3 | 7261.8 | 1.13x | +12.8% |
| 16 | 6365.6 | 7184.3 | 1.13x | +12.9% |
| 32 | 6339.7 | 7148.3 | 1.13x | +12.8% |
| 64 | 6319.0 | 7124.0 | 1.13x | +12.7% |

Aggregate prefill throughput = `parallel_sequences * 1000 / t_pp`, the `speed_pp` (`S_PP t/s`) field from the same runs. Prefill and decode use their own measured times; all parameters are identical between the two builds.

RTX 5090, CUDA 13.0, Release, SM120, the same official Flint NVFP4 GGUF, f16 KV, Flash Attention on, CUDA graphs at their defaults, 8 CPU threads. Fixed batches with random decode token IDs: each sequence has 1,000 prompt tokens followed by 1,000 decode steps; KV depth grows from 1,000 to 2,000 tokens during decode. Each cell is the median of two fresh processes, run in stock / FlashRT / FlashRT / stock order.

Baseline: llama.cpp at `1b44e30775dcb2f93952116b05f89e5bd797fb00` with only patch 0001 (Jinja compatibility); CUDA and scheduler are stock. FlashRT: the same pin with patches 0001–0009 and this overlay.

The 64-sequence sweep needs 128,000 KV cells, so both builds use
`-c 131072`. All results use the same workload, with the prefill and decode
phases timed separately.

Set `MODEL` to the GGUF, `STOCK_BIN` to the stock `llama-batched-bench`, and
`FLASHRT_BIN` to the binary produced by this host's `build.sh`, then run:

```bash
for bench in "$STOCK_BIN" "$FLASHRT_BIN" "$FLASHRT_BIN" "$STOCK_BIN"; do
  "$bench" -m "$MODEL" -ngl 99 -fa on -ctk f16 -ctv f16 -t 8 \
    -c 131072 -b 2048 -ub 512 -npp 1000 -ntg 1000 -npl 1,8,16,32,64 \
    --output-format jsonl
done
```

Build the baseline from the same pin with only
`patches/0001-common-jinja-hexadecimal-escapes-upstream-PR-30929-5.patch`.
Configure both builds with `-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`;
only the FlashRT build enables `GGML_CUDA_FLASHRT_SM120` and adds patches
0002–0009. The baseline's Jinja backport changes template parsing, not the
decode kernels.

## Validation

| Check | Result |
|---|---|
| Fresh host build, pinned base + patches 0001–0009 | Pass; server, CLI, perplexity, bench and batched-bench built |
| Host-reference checks, ubatches 8 / 24 / 64 | 168 window/partial reports, all `bad 0`; zero unexplained q8 or scale mismatches |
| Attention vs ggml f32 vector reference, 8 / 64 sequences | Max relative difference `2.42e-6` |
| Reference-helper boundary and NaN/Inf regression cases | Pass |
| Existing Qwen patches, whitespace and privacy scans | Pass |

The default windows handle 2–128 tokens with int8 tensor-core dot products,
expert grouping and multi-sequence decode attention (up to 32 sequences by
default). See `../../docs/usage.md` for caps, switches and the
one-active-context-per-GPU requirement. Batched output is not claimed
bit-exact; numerical tolerances, sampling coverage and correctness commands
are specified in `../../docs/testing.md`.
