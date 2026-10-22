# Usage

## Building a host against the adapter

This repository is mounted by a host at `ggml/src/ggml-cuda/flashrt`
(as a git submodule) and consumed at build time: the host's CMake
compiles the translation units here, the vendored FlashRT kernels under
`vendor/csrc/`, and takes CUTLASS from `third_party/cutlass`. Two hosts
are wired today, each with its pin, patches and instructions under
`hosts/`:

- `hosts/llama.cpp/` — upstream llama.cpp on RTX 5090 (SM120), the
  Qwen3.6-35B-A3B decode window set. `hosts/llama.cpp/build.sh` builds
  it from scratch.
- `hosts/jetson-pi-edge/` — Jetson-PI-Edge on Jetson AGX Thor (SM110),
  pi0 / pi0.5.

```bash
git clone --recursive https://github.com/flashrt-project/FlashRT-llama.cpp
FlashRT-llama.cpp/hosts/llama.cpp/build.sh /path/to/work
```

Options:

- `GGML_CUDA_FLASHRT` (OFF by default) — enables the layer. Without it the
  build is stock llama.cpp; every integration point is compiled out.
- `GGML_CUDA_FLASHRT_DIR` — path to an overlay checkout, overriding the
  submodule location.
- `GGML_CUDA_FLASHRT_CUTLASS_DIR` — CUTLASS override; defaults to
  `third_party/cutlass` inside this repository.

The adapter is built as a separate CMake OBJECT library with
`-arch=sm_110a` and CUTLASS headers; the rest of ggml-cuda compiles
unchanged. The AOT FlashAttention-4 windows enable automatically when the
`fa4_aot/*.o` artifacts are present (they are checked in; see
`fa4_aot/README.md` to regenerate).

## Model preparation

- LLM weights: quantize with the host's `llama-quantize` to the `NVFP4`
  target (exposed by the FlashRT branch). Setting `GGML_NVFP4_MSE=1`
  during quantization selects per-block scales by reconstruction-MSE
  search instead of plain absmax (slower to quantize, more accurate).
- The mmproj (vision tower) is quantized to NVFP4 the same way.

## Running

```bash
PI_MODEL=pi05 ./build/bin/llama-server -m <llm.gguf> --mmproj <mmproj.gguf> \
    -ngl 99 --flash-attn on --port <port>
```

The server exposes the host's action-chunk HTTP protocol (reset → images →
state → infer). Warm-up matters on Thor: latency reaches its steady state
after roughly 15 inferences.

## Runtime switches

All switches are environment variables; unset means enabled/default.

| variable | effect |
|---|---|
| `GGML_CUDA_FLASHRT_DISABLE=1` | disable the whole layer at runtime (stock kernels) |
| `GGML_FLASHRT_NO_RMS_GEMMA=1` | disable the Gemma norm-chain window |
| `GGML_FLASHRT_NO_QKV_PREFILL=1` | disable the fused prefill QKV window |
| `GGML_FLASHRT_NO_DEC_ATTN=1` | disable the decomposed decode attention |
| `GGML_FLASHRT_NO_VIT_FA4=1` | disable the AOT FA4 vision attention |
| `GGML_FLASHRT_NO_PREFILL_FA4=1` | disable the AOT FA4 prefill attention |
| `GGML_FLASHRT_NO_KV_TAIL=1` | disable the batched persistent-KV tail copies |
| `GGML_FLASHRT_NO_VIS_F16=1` | disable the vision QKV window's direct f16 K/V outputs |
| `GGML_CUDA_FLASHRT_NO_CACHE=1` | disable the pointer-keyed weight repack cache (required for `test-backend-ops`, see TESTING.md) |
| `GGML_FLASHRT_DEBUG=1` | print window-match failure diagnostics |
| `GGML_FLASHRT_DUMP=<file>` / `GGML_FLASHRT_DUMP_MAX=<n>` | dump the first n evaluated graphs' node sequences |

Host-side switches on the FlashRT branch (outside this repository):
`GGML_PI05_MOD_PRECOMP=0` disables the denoise-schedule modulation
precompute, `LLAMA_GRAPH_REUSE_DISABLE=1` disables graph reuse,
`GGML_CUDA_DISABLE_GRAPHS=1` disables CUDA graphs (useful for profiling:
kernels are invisible to nsys while CUDA graphs replay).

Every window degrades gracefully: when its predicate does not match (or
its switch is set) the nodes run on stock ggml kernels, so the switches
bisect regressions window by window.

## SM120 / Qwen3.6 target

Build a llama.cpp tree with the overlay mounted at
`ggml/src/ggml-cuda/flashrt` (see `hosts/llama.cpp/`; `build.sh` does
all of this from scratch):

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_FLASHRT_SM120=ON
```

The sm_120a pass and the CUTLASS include path are scoped to the FlashRT
sources by the host's CMake block; no `CMAKE_CUDA_FLAGS` needed.

The safe tier is the zero-configuration default: a compiled-in build runs
every quality-neutral window (fused packs, GDN span, MoE span, out-proj,
draft-head serving, in-process repack) with no environment at all —
`./llama-server -m <stock gguf> [-md <draft gguf> -bs]` is the whole story.
Switch semantics, most specific wins:

| layer | variable |
|---|---|
| whole layer off (stock llama.cpp) | `GGML_CUDA_FLASHRT_DISABLE=1` |
| per-window disable | `GGML_FLASHRT_NO_{INPROJ,ATTNQKV,GDN,MOEGLUE,MOEFUSE,SHEXP_FOLD,OUTNATIVE,HEAD_DRAFT,ONLINE_REPACK}=1` |
| per-window A/B override | historic `FRT_<X>_SWAP=0/1` |

Opt-in extras: `FRT_HEAD_SWAP=1 FRT_HEAD_PACK=<pack>` (full tier — the FP4
lm-head trades a measured perplexity increment for speed, so it never
defaults on); `FRT_DRAFT_REGIONS=1` (also FP4-serve the draft model's own
qkv projections; judged flat, off by default); archive windows
(`FRT_SHEXP_SWAP`, `FRT_OUTPROJ_SWAP`, `FRT_MOE_SWAP`, `FRT_ATTNGATE_SWAP`,
`FRT_GDN_NORMFOLD`) stay opt-in. Recommended host-side flags for
speculative decode: `LLAMA_GRAPH_SLOTS=6` + `--backend-sampling`.

**Model artifacts** (sm120 target): the safe tier runs any stock GGUF
as-is. The speed tier is itself just a GGUF — the FlashRT edition splices
an NVFP4 lm-head (quantized from the BF16 checkpoint via
`llama-quantize --output-tensor-type NVFP4` on a bf16 conversion) into the
shipping body with `tools/splice_nvfp4_head.py`; no side-band packs, no
environment. Do not requantize the whole body from scratch for this: the
FP4 regions inherit the source tensors' quantization quality, so the best
shipping body stays the best base.

FP4 region weights repack **in-process by default** (set `FRT_REGIONS_PACK`
to use an offline pack instead): the pre-capture hook dequantizes the GGUF
members on device and rebuilds the wire format byte-identically to the
offline packer (validated by `FRT_REPACK_CHECK=1` with
`FRT_REGIONS_PACK_REF=<file>`). The
lm-head is the exception: the shipped head pack is quantized from the BF16
checkpoint (the GGUF only holds Q6_K), and the BF16-sourced pack drafts and
scores measurably better than an online Q6_K-sourced rebuild — keep
`FRT_HEAD_PACK` for the head (both tiers); the online head build is a
fallback only. Diagnostics: `FRT_STATS=1`, `FRT_MOEFUSE_DBG=<n>`,
`FRT_MOEFUSE_SELFTEST=1`, `FRT_DUMP_GRAPH=1` + `FRT_DUMP_M=<m>`
(+ `FRT_DUMP_PATH`).

## SM120 / Flint NVFP4 target

Recursive Flint (49 layers, sliding-window + full attention, sigmoid
top-6/256 MoE with an ungated shared expert) on the official NVFP4 GGUF.
The host is llama.cpp with the Flint model (upstream PR #30928, pinned in
`hosts/llama.cpp-flint/PIN`); `hosts/llama.cpp-flint/build.sh` clones it,
applies the patch series and builds with `-DGGML_CUDA_FLASHRT_SM120=ON`.
The window file is `fr_win_flint_sm120.cu`; it is compiled inside the
ggml-cuda target with ggml's own flags.

Recommended serving flags (official Flint flags plus the ubatch):

```bash
llama-server -m flint-nvfp4.gguf -ngl 99 -c <ctx> -np 1 -fa on -ctk f16 -ctv f16 --jinja \
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0 -b 4096 -ub 4096
```

`-ub 4096` (default 512) roughly doubles prompt throughput for this MoE;
decode speed does not depend on it with this host (patch 0006). `-ub 8192`
is slower.

Numerics. The default path is exact: decode and prefill logits are
bit-identical to stock llama.cpp at the same `-ub` (the windows reproduce
ggml's per-element arithmetic order; TESTING.md lists the coverage). The
only non-bitwise component is the opt-in decode attention `FRT_ATTNDEC=1`
(tensor-core GQA attention with f32-level operands), gated by
perplexity/KLD against stock's own prefill-vs-decode variation. Per call
it differs from ggml's f32 vector kernel by ~5e-7 relative (max 4e-6) at
every cache length. From 8K cached tokens on, stock decode runs its
full-attention layers on an f16 tensor-core kernel instead (GQA ratio >
4), which differs from both by ~6e-4.

| variable | effect |
|---|---|
| `GGML_CUDA_FLASHRT_DISABLE=1` | whole layer off (stock kernels) |
| `GGML_FLASHRT_NO_{AMAX,ATTNQKV,ROUTE,OPROJ,MOEFFN,DENSEFFN,LMHEAD,MOETAILPF,GUGLUPF}=1` | disable one window (decode: AMAX, ATTNQKV, ROUTE, OPROJ, MOEFFN, DENSEFFN, LMHEAD; prefill: AMAX, MOETAILPF, GUGLUPF) |
| `FRT_ATTNDEC=1` | opt-in decode attention (non-bitwise, see above); `FRT_ATTNDEC_QKN=0` keeps the separate head-norm kernel, `FRT_ATTNDEC_MAX_TILES=<n>` (default 1280 = ~80K rows) hands longer caches back to ggml |
| `FRT_NW_{QKV,OPROJ,GU,DOWN,ROUTE,DGU,DDOWN,LMHEAD}=4\|8\|16` | warps per CTA of the GEMV windows (defaults from a shape search) |
| `FRT_ATTNDEC_CHECK=<k>` | test aid: every k-th ATTNDEC call is compared against ggml's kernel (eager mode only, `GGML_CUDA_DISABLE_GRAPHS=1`); `FRT_ATTNDEC_CHECK_REF=vec` compares against ggml's f32 vector kernel at every cache length |
| `FRT_PF_CHECK=1` | test aid: the prefill MoE tail window compares against ggml's own execution (eager mode) |
| `FRT_STATS=1`, `FRT_DUMP_GRAPH=1` + `FRT_DUMP_M=<m>` (+ `FRT_DUMP_PATH`) | diagnostics |

### Batched decode (2 to 128 tokens per ubatch)

The decode windows also take ubatches of 2..128 tokens (several sequences
decoding together, or a short prompt): weight rows are read once per kernel
and applied to all tokens with `mma.sync.m16n8k16.s8` (one MMA per 16-value
NVFP4 sub-block: the integer dot products are exact, the UE4M3 and q8_1
scales are applied in fp32), the routed experts are grouped by the top-k
selection, and the decode attention runs over the sequences (per-sequence KV
slots, i.e. without `-kvu`). Each token is computed with the same per-element
arithmetic as the M = 1 windows; only the fp32 summation order differs, so
batched outputs are gated by the host reference in TESTING.md, not by bits.
Defaults were measured on an RTX 5090 with 1000-token prompts.

| variable | effect |
|---|---|
| `GGML_FLASHRT_NO_BATCHED=1` | stock kernels for every ubatch of more than one token |
| `FRT_B_MOE_MIN_T=<t>` (9), `FRT_B_MOE_MAX_T=<t>` (128) | token range of the batched routing / MoE windows; below it stock's per-slot MMVQ is faster, above it stock MMQ (prompt-sized ubatches) |
| `FRT_B_ATTN=0`, `FRT_B_ATTN_MAX_SEQ=<n>` (32) | batched decode attention off / its sequence cap (stock's f16 tensor-core kernel is faster at 64 x 1K rows); `FRT_B_ATTN_CTAS`, `FRT_B_ATTN_NS` tune chunking |
| `FRT_B_DP4A=1`, `FRT_S_OFF=1`, `FRT_S_DN=1`, `FRT_P_ON=1`, `FRT_T_ON=1` | kernel variants: CUDA-core dp4a; gate/up back to the register-resident kernel; down with the shared-memory kernel; persistent ring; TMA bulk ring (all slower than the defaults, kept for study) |
| `FRT_B_HREF=1` | test aid (with `GGML_CUDA_DISABLE_GRAPHS=1`): every batched window is checked against a double-precision host reference, see TESTING.md |
| `FRT_B_CHECK=1` | test aid: batched windows compute into private buffers and are compared against ggml's own execution |
| `FRT_B_DBG=<bits>` | timing aid: 1 skips the MMAs, 2 the token staging, 4 the weight loads (outputs are wrong) |

Server flags that matter for batched decode (measured, no output change):
per-sequence KV slots (no `-kvu`), `--backend-sampling` with
`--samplers "top_k;top_p;min_p;temperature"` (a chain every sampler of which
has a backend implementation), `--ctx-checkpoints 0` (the default
checkpoints copy the SWA cache rows to host at every request boundary),
`--threads-http 16`. `-ub` above 512 does not fit 64 slots: the SWA cache is
sized slots x (513 + n_ubatch).

The windows' scratch buffers belong to one GPU (the first SM120 device
that runs a graph; any other device runs stock) and to one active context:
run one Flint context per GPU at a time (one `llama-server` process with
any number of slots is one context).

Host-side switches added by the Flint patch series (all default on,
identical results either way):

| variable | effect |
|---|---|
| `GGML_CUDA_SYNC_UPLOADS=1` | blocking small host-to-device uploads (stock behaviour) |
| `GGML_CUDA_MMID_SINGLE_WARP=1` | single-warp MoE id helper (stock) |
| `GGML_CUDA_FA_KV_MIN=0` | no masked-tile skipping in the tensor-core FA kernel |
| `LLAMA_KQ_MASK_SERIAL=1` | serial KQ-mask fill |
| `LLAMA_SCHED_RESERVE_NEW=1` | recreate the scheduler (and its compute buffers) on every re-reserve |
| `LLAMA_SCHED_NODE_HEADROOM=<f>` | patch 0009: the scheduler is recreated only when the node budget grows, with this headroom (default 0.25); the budget moves with the number of per-sequence backend samplers |
| build with `GGML_CUDA_FA_VEC_NO_MASKED_SKIP` | no masked-warp skipping in the vector FA kernel |

`--backend-sampling` (stock option) samples on the GPU: served decode is
5–10% faster (1K–32K prompts) and greedy outputs are unchanged; sampled
outputs come from the GPU sampler's random stream instead of the CPU one.
Setting a backend sampler re-reserves the scheduler for every request;
with patch 0008 that no longer reallocates the compute buffers, so time to
first token is the same as with CPU sampling (without it, +~0.13 s per
request at `-ub 4096`).
