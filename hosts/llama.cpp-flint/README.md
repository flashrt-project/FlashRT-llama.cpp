# Host: llama.cpp + Flint (RTX 5090, SM120, official NVFP4 GGUF)

Host: https://github.com/ggml-org/llama.cpp, Flint model support from
upstream PR #30928 at the commit in `PIN`. The host-side changes are the
patch series in `patches/`:

1. Jinja hexadecimal escapes (backport of upstream PR #30929, needed by
   the Flint chat template).
2. `GGML_CUDA_FLASHRT_SM120`: the CMake block that compiles
   `fr_win_flint_sm120.cu` from the overlay (submodule at
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

Patches 3–7 change no values: every patch keeps logits bit-identical.

`build.sh <work dir> [llama.cpp url or local clone]` reproduces the build
from scratch: clone, fetch the PR head, check out the pin, `git am` the
series, mount the overlay, configure with `-DGGML_CUDA_FLASHRT_SM120=ON`.
Recommended serving flags and the runtime switches are in `../../USAGE.md`.

## Numerics

The default path is exact: decode and prefill logits are bit-identical to
stock llama.cpp at the same `-ub` (FNV-1a hash of every position's logits:
1024 decode positions; ~100K decode positions on a code corpus; 4096 decode
positions with a 4096-ubatch context; 8192 prefill tokens at `-ub 2048`
and `-ub 4096`, the latter against the unmodified stock build). Real-prompt
greedy outputs are token-identical to stock. The opt-in decode attention
(`FRT_ATTNDEC=1`) is the only non-bitwise component; its quality is gated
against stock's own path variation (below).

## Measurements (RTX 5090, CUDA 13.0, f16 KV, `-fa on`, clocks not locked, GPU otherwise idle)

Stock = unmodified llama.cpp at the pin (default `-ub 512`); FlashRT =
this host + overlay. Prefill and decode are reported separately; "served"
numbers are llama-server, cache-cold (`cache_prompt: false`), CPU
sampling, one request at a time.

Decode, `llama-bench` tg128 at populated KV depth (tok/s, median of 5):

| depth | 0 | 1K | 4K | 8K | 16K | 32K | 64K | 100K |
|---|---|---|---|---|---|---|---|---|
| stock | 242.5 | 236.4 | 228.7 | 227.4 | 219.5 | 205.6 | 182.7 | 151.8 |
| FlashRT | 550.2 | 537.1 | 502.2 | 490.5 | 455.1 | 398.7 | 321.1 | 265.3 |
| speedup | 2.27x | 2.27x | 2.20x | 2.16x | 2.07x | 1.94x | 1.76x | 1.75x |

Prefill, `llama-bench` pp at depth 0 (tok/s):

| prompt | 1K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|
| stock `-ub 512` | 7637 | 7324 | 7236 | 7044 | 6661 | 5853 |
| stock `-ub 4096` | 10673 | 11876 | 12000 | 11705 | 10809 | 9144 |
| FlashRT `-ub 4096` | 12311 | 16641 | 16484 | 15737 | 14064 | 11271 |

Served, real-code agent prompts (exact token counts checked against the
server's `prompt_n`), 128 (≤32K) or 256 (64K/100K) decode tokens:

| prompt tokens | stock `-ub 512` TTFT | stock `-ub 4096` TTFT | FlashRT `-ub 4096` TTFT | FlashRT decode tok/s |
|---|---|---|---|---|
| 1,022 | 0.18 s | 0.13 s | 0.12 s | 482 |
| 8,188 | 1.25 s | 0.76 s | 0.56 s | 405 |
| 32,765 | 5.25 s | 3.12 s | 2.38 s | 343 |
| 65,523 | 11.9 s | 7.28 s | 5.83 s | 287 |
| 99,994 | 20.9 s | 13.1 s | 10.9 s | 241 |

Real-suite decode (8 coding/agent prompts, 512 greedy tokens, A/B/A):
220.3 → 452.9 tok/s geomean (2.06x), tokens identical to stock.

100K context, batch 1: decode model forward 3.51 ms excluding the LM head
(3.61 ms GPU step, 3.81 ms per token in `llama-bench`, 4.15 ms served
with CPU sampling; stock 5.95 / 6.05 / 6.20 ms). An 8,192-token prefill
on top of 92K cached tokens takes 1.34 s (stock 2.30 s at `-ub 512`,
1.55 s at `-ub 4096`); at that depth the full-attention layers dominate and
run near the tensor-core rate of the card. Peak memory with a 104K context
and `-ub 4096`: 27.3 GB.

Quality: perplexity is unchanged by construction (exact path); `-ub 4096`
vs `-ub 512` (different stock numerics) at 4K context: wikitext 17.098 vs
17.068, code 6.168 vs 6.291. HumanEval pass@1 (164, greedy): stock 72.6%,
FlashRT 72.6% (161/164 identical completions); a benign stock numeric
variant (32-token prompt ubatches) scores 67.7%, `FRT_ATTNDEC=1` 68.3% —
this model's greedy outputs diverge under any numeric perturbation, so
only stock's own path variation is a meaningful reference for non-exact
options. Tool calls in agent prompts are well-formed in all variants.

## Remaining headroom

- MoE grouped GEMMs in prefill (stock MMQ, W4A8) run at ~55–115 TOPS
  against ~237 TOPS for the dense projections; a MoE-specific tile
  configuration needs an MMQ change that keeps the dense stream-k split.
- Batched decode (2–8 concurrent requests) runs mostly stock kernels: the
  decode windows are M = 1 (batch 8: 1173 vs 948 tok/s aggregate).
- About 0.15 ms per decoded token is host-side (token round trip, graph
  launch); `--backend-sampling` removes part of it but re-reserves the
  scheduler per request in this llama.cpp version.
