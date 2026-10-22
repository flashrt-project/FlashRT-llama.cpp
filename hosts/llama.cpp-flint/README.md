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
8. A scheduler re-reserve that keeps the node budget reuses the scheduler
   and its compute buffers. The server sets a backend sampler per request
   (`--backend-sampling`), which used to reallocate every compute buffer
   before each prompt (`LLAMA_SCHED_RESERVE_NEW=1` restores that).

Patches 3–8 change no values: every patch keeps logits bit-identical.

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

Decode, `llama-bench` tg128 at populated KV depth (tok/s, median of 6;
FlashRT at `-ub 4096`, stock at its default `-ub 512`; FlashRT decode does
not depend on `-ub`, within 0.5%):

| depth | 0 | 1K | 4K | 8K | 16K | 32K | 64K | 100K |
|---|---|---|---|---|---|---|---|---|
| stock | 242.1 | 236.2 | 228.4 | 227.1 | 219.3 | 205.3 | 182.3 | 162.0 |
| FlashRT | 550.1 | 534.9 | 501.6 | 489.8 | 454.3 | 398.2 | 320.6 | 264.9 |
| speedup | 2.27x | 2.26x | 2.20x | 2.16x | 2.07x | 1.94x | 1.76x | 1.64x |
| `FRT_ATTNDEC=1` | 548.0 | 548.2 | 528.3 | 510.8 | 471.3 | 408.9 | 322.0 | 263.2 |

The 100K column is the median of three fresh processes (stock 161.8 /
162.0 / 162.1, FlashRT 264.6 / 264.9 / 265.1). Inside one process that
sweeps all depths in turn, stock measures 151.4 at 100K and FlashRT 265.3.

Prefill, `llama-bench` pp at depth 0 (tok/s):

| prompt | 1K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|
| stock `-ub 512` (default) | 7640 | 7320 | 7219 | 7045 | 6667 | 5852 |
| stock `-ub 4096` | 10680 | 11855 | 11990 | 11697 | 10808 | 9142 |
| FlashRT `-ub 4096` | 12336 | 16506 | 16419 | 15723 | 14055 | 11256 |
| FlashRT vs stock, both `-ub 4096` | 1.16x | 1.39x | 1.37x | 1.34x | 1.30x | 1.23x |

Most of the gain over stock's default comes from the larger ubatch, which
stock can use as well; the kernel gain at the same ubatch is the last row.
FlashRT at `-ub 512` runs 8483 / 7920 / 6283 tok/s at 1K / 8K / 64K.

Served, real-code agent prompts (token counts checked against the
server's `prompt_n`), cache-cold, CPU sampling, 128 (≤32K) or 256
(64K/100K) decode tokens; time to first token (TTFT) and the decode rate
after the prompt, median of 2 requests (≤32K after one warm-up request):

| prompt tokens | stock `-ub 512` TTFT | stock `-ub 4096` TTFT | FlashRT `-ub 4096` TTFT | stock `-ub 512` decode | FlashRT decode |
|---|---|---|---|---|---|
| 1,022 | 0.18 s | 0.13 s | 0.12 s | 221 tok/s | 467 tok/s |
| 4,089 | 0.63 s | 0.37 s | 0.28 s | 212 | 420 |
| 8,188 | 1.25 s | 0.76 s | 0.56 s | 209 | 409 |
| 16,371 | 2.51 s | 1.48 s | 1.10 s | 204 | 397 |
| 32,765 | 5.31 s | 3.12 s | 2.39 s | 194 | 348 |
| 65,523 | 11.9 s | 7.30 s | 5.91 s | 173 | 291 |
| 99,994 | 21.0 s | 13.1 s | 11.1 s | 157 | 249 |

With `--backend-sampling` (FlashRT `-ub 4096`, 1K–32K): TTFT within 6 ms
of CPU sampling (0.25 / 0.69 / 2.51 s at 1K / 8K / 32K without patch
0008), decode 501 / 460 / 451 / 418 / 371 tok/s, identical greedy outputs.

Real-suite decode (8 coding/agent prompts of 57–12,725 tokens, 512 greedy
tokens, stock / FlashRT / stock server processes, both at the default
`-ub 512`): 220.4 → 457.9 tok/s geomean (2.08x) with CPU sampling and
224.4 → 491.7 (2.19x) with `--backend-sampling` on both; tokens identical
to stock in every leg.

100K context, batch 1: 3.78 ms per decoded token in `llama-bench` (stock
6.17 ms) and 4.02 ms served with CPU sampling (stock 6.36 ms); in an nsys
profile of the same decode kernels the model forward without the LM head
takes 3.51 ms (stock 5.95 ms). An 8,192-token prefill on top of 92K cached
tokens takes 1.37 s (stock 2.32 s at `-ub 512`, 1.58 s at `-ub 4096`); at
that depth the full-attention layers dominate and run near the
tensor-core rate of the card. Device memory in use (nvml, sampled every
0.2 s) peaks at 27,349 MiB while serving the 100K prompt with a 104K
context at `-ub 4096` (stock at most 27,175 MiB; both include the ~0.5 GiB
in use before the server starts).

Quality: perplexity is unchanged by construction (exact path); `-ub 4096`
vs `-ub 512` (different stock numerics) at 4K context: wikitext 17.098 vs
17.068, code 6.168 vs 6.291. HumanEval pass@1 (164, greedy): stock at its
default `-ub 512` 72.6%, FlashRT at `-ub 4096` 72.6% (161/164 identical
completions; the two differ only by the ubatch); stock numerics with
32-token prompt ubatches (run on FlashRT, which is bit-identical to stock
at that ubatch) 67.7%; `FRT_ATTNDEC=1` 68.3%. This model's greedy outputs
diverge under any numeric perturbation, so only stock's own path variation
is a meaningful reference for non-exact options. Tool calls in agent
prompts are well-formed in all variants.

## Remaining headroom

- Short-context decode: at depth 0 the GPU spends 1.77 ms per token against
  ~0.85 ms to stream the active weights once at full DRAM bandwidth. Each
  MoE layer is a chain of 9 dependent kernels (~34 µs): the four GEMV
  windows reach 0.8–1.5 TB/s, and the links between them (head norm and
  RoPE, attention, combine, routing, GLU quantization) are latency-bound.
  Most of the gap is fixed per-kernel latency, so closing it needs fusion
  across dependent stages (a persistent layer kernel with cheap grid
  synchronization; a first attempt with a single-counter barrier was 7%
  slower).
- MoE GEMMs in prefill (stock MMQ, W4A8): at 4096-token ubatches the
  routed gate/up GEMMs run at ~89 TOPS and the down GEMM at ~49 TOPS,
  against ~240 TOPS for the dense projections. At dense efficiency they
  would remove about 30% of an 8K prefill. That needs a MoE-specific
  grouped GEMM for ~100 tokens per expert; picking the MMQ tile from the
  per-expert token count was slower.
- Batched decode: the decode windows are M = 1, so 2–8 concurrent
  requests run mostly stock kernels. Aggregate decode at 2,048-token
  prompts (`llama-batched-bench`, both `-ub 512`), batch 1 / 2 / 4 / 8:
  stock 226 / 381 / 608 / 963 tok/s, FlashRT 514 / 481 / 769 / 1191.
- Host time per decoded token (sampling, token round trip, graph launch)
  is not overlapped with the GPU: served decode with CPU sampling takes
  ~0.3 ms per token more than the GPU step. `--backend-sampling` removes
  most of the sampling share; the rest needs a GPU-resident decode loop.
