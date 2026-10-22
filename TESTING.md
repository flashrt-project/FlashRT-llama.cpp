# Testing

Three layers of validation, from operator level to release gate.

## 1. Operator tests (`test-backend-ops`)

The host's `test-backend-ops` exercises the NVFP4 mul_mat path against the
CPU reference:

```bash
GGML_CUDA_FLASHRT_NO_CACHE=1 ./build/bin/test-backend-ops test -o MUL_MAT
```

`GGML_CUDA_FLASHRT_NO_CACHE=1` is **required**: the weight repack cache is
keyed by tensor data pointer under the assumption that weights are
immortal, which holds for models but not for the test harness's rapidly
recycled tensors. NVFP4 mismatches up to ~2e-2 are inherent W4A4
activation-quantization noise (upstream applies the same tolerance to
native FP4 backends), not failures.

## 2. Qualification gates (`qualification/`)

`qualification/run_qualification.py` gates a build the way a release
would:

- **manifest** — the pipeline binding
  (`bindings/jetson_pi_edge_pi05.yaml`) must map the host's complete hot
  path onto catalog structures.
- **pins** — the structure versions the binding names must match the
  catalog (`qualification/pins.yaml`).
- **e2e golden** (`--e2e`, on-device, needs a running server) — drives the
  fixed synthetic-input protocol and compares the steady-state action
  chunk against `qualification/goldens/pi05_thor_action.json` **exactly**.
  The adapter is bitwise deterministic across processes after warmup, so
  any bit difference is a real change.

```bash
python qualification/run_qualification.py           # offline gates
python qualification/run_qualification.py --e2e     # + on-device golden
python qualification/run_qualification.py --e2e --update-golden
```

Take the golden only after at least two warm-up inferences (the first
inference after cold start differs from steady state) and only for
changes whose numerics were judged (below).

## 3. Benchmark and parity methodology

Every performance change must pass this protocol on device:

- **Hot-regime A/B/A sandwich** — run the candidate, the fallback (via its
  runtime switch), and the candidate again as three separate server
  processes, ≥15 warm-up + ~20 measured inferences each, comparing P50.
  Thor drifts ±1–3 ms across long sessions, so only same-session
  back-to-back numbers are comparable; single measurements and
  cross-session comparisons are not accepted.
- **Bitwise parity** — save the action chunk from each leg. A change that
  claims numeric neutrality must be bit-identical to the previous
  accepted state. Note the converse trap: bit-identical output *plus*
  zero performance delta usually means the window never fired — verify
  the window triggers (kernel census, `GGML_FLASHRT_DEBUG`) before
  interpreting the A/B.
- **Real-observation judge** — for changes that move numerics, run a set
  of real robot observations (gripper-active frames) through the NVFP4
  build and an f16-weights build of the same tree, and compare per-dim
  cosine of the action chunks against the f16 reference. The distance to
  the reference must not systematically regress. Any bit-level change in
  the action path amplifies to ~2e-2 absolute wobble on final actions
  through the 10 denoise steps, so raw action diffs are meaningless —
  only the distance-to-reference comparison judges accuracy.
- **Kernel-level accounting** — attribute wins with an nsys census
  (`GGML_CUDA_DISABLE_GRAPHS=1`, full-lifetime `-t cuda` trace with a
  graceful server exit so buffers flush). CUDA-graph replays hide kernels
  from the profiler, and profiling on Tegra inflates kernel times, so the
  census attributes *where* time went while the non-profiled A/B decides
  *whether* the change lands.

## 4. Flint NVFP4 (SM120) protocol

The original M = 1 windows and stock fallbacks claim bit-exactness against
stock llama.cpp. With `GGML_FLASHRT_NO_BATCHED=1`, their primary gate is
logit identity, checked with a teacher-forced driver that
records an FNV-1a hash of every position's f32 logits:

- **Decode path** (`n_batch = 1`, every position through the M = 1 graph):
  hashes equal to stock for 1024 positions of mixed text, for a
  ~100K-position code corpus (cache filled to ~100K), and for 4096
  positions with a 4096-ubatch context (large sliding-window ring).
- **Prefill path**: hashes equal to stock at the same ubatch for 8192
  tokens of code at `-ub` 512, 2048, 4096, 8192 and 3000 (a partial last
  ubatch), and for 512 tokens at ubatches of 2, 4, 8 and 32 tokens (the
  M = 2..8 paths that batched decode also takes). Different ubatch sizes
  are different stock numerics (tiling, stream-k splits) and are compared
  by perplexity, not by hash.
- **Served A/B/A** — stock, candidate, stock as three server processes on
  real coding/agent prompts (cache-cold, `cache_prompt: false`, streamed
  for time to first token, fixed decode budget with `ignore_eos`); greedy
  tokens must be identical in every leg, and decode/prefill/TTFT/request
  time are reported separately. Context-length sweeps use real-code
  prompts of 1K–100K tokens whose token counts are checked against the
  server's `prompt_n`.

Non-bitwise options (`FRT_ATTNDEC=1`) are gated by quality instead:
per-call error against ggml's kernel (`FRT_ATTNDEC_CHECK`), paired
`llama-perplexity --kl-divergence` against stock decode logits (2K
context; decode-path perplexity at 32K), compared with stock's own
prefill-vs-decode KLD (this model
shows KLD ~0.24 on code between two stock paths: any perturbation, even
1e-6 per call, saturates to that level), and generated-code pass rates.

Window triggering is verified with an nsys kernel census (the replaced
stock kernels must disappear); a window that never fires is bit-identical
and speed-neutral by construction. `FRT_STATS=1` also reports whether the
MoE window reuses the router window's quantized activation
(`moeffn ... act_q8=1`); a host change that breaks that handoff stays
exact but loses speed, so it is part of the check.

### Batched decode windows (2..128 tokens)

Batched outputs are not bit-identical to the M = 1 windows (fp32 summation
order), so they are gated against a double-precision host reference that
re-derives every window from the device's own q8_1 blocks (NVFP4 x q8_1 dot
products exactly as ggml defines them):

```
GGML_CUDA_DISABLE_GRAPHS=1 FRT_B_HREF=1 llama-perplexity -m <nvfp4.gguf> -ngl 99 -fa on -ctk f16 -ctv f16 \
    -c 2048 --chunks 1 -b 2048 -ub <T> -f <text>
```

prints `frt-href <window> call <k> ... bad <n>` lines for the first six
eligible calls per window (batches smaller than four tokens are skipped).
The reference samples token rows and output channels; it is a window check,
not an exhaustive end-to-end quality check. Gate: `bad 0` for every reported
window (a mismatch requires both absolute error > 1e-4 and relative error
> 2e-3), no routing-ID mismatches, zero scale mismatches, and zero
unexplained quantization flips. `qkv-q8` must have zero raw flips.
The routed and shared MoE down-projection partials (`moe-part`,
`moe-shpart`) report the same error gate as final window outputs. Any
non-finite reference or result is a failure.
`dense-q8`, `moe-q8` and `oproj-q8` requantize device fp32 rows on the host;
CPU division and CUDA fast division can straddle a rounding boundary.
The check retains raw flip counts and prints `quant-ties` only when the
two int8 values are adjacent integers straddling a half-integer within two
fp32 ULPs. Every raw flip must be accounted for by those diagnostics;
any `quant-bad` is a failure. These ties do not enter the dot-product
reference, which consumes the device's q8 blocks. Run it at `-ub 8`,
`24` and `64` (one, two and four token tiles). The batched attention has its
own check: `GGML_CUDA_DISABLE_GRAPHS=1 FRT_ATTNDEC_CHECK=<k>
FRT_ATTNDEC_CHECK_REF=vec FRT_B_ATTN_MAX_SEQ=64` on `llama-batched-bench`
forces coverage of 64 sequences, beyond the default cap of 32, and reports
the sampled per-call difference against
ggml's f32 vector kernel (review rerun max rel 2.42e-6 at 8 and 64 sequences).
Throughput: `llama-batched-bench -npp 1000 -ntg 1000 -npl 1,8,16,32,64`
(fixed batches, random decode tokens), two fresh processes, median; the
served number is a separate `llama-server` run and is reported separately.

Use `-c 131072 -b 2048 -ub 512 -npp 1000 -ntg 1000 -npl 1,8,16,32,64`
for that throughput sweep: 64 independent sequences need 128,000 KV cells
for 1,000 prompt plus 1,000 generated tokens each. `-c 90000` is too small.
