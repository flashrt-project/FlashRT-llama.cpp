// Batched (M > 1) decode windows for Flint on SM120. Included by
// fr_win_flint_sm120.cu (same translation unit: shares the exact per-element
// helpers, the matchers and the private buffers).
//
// The M = 1 windows own every node of a layer for one token; these kernels own
// the same spans for a ubatch of T tokens (2 <= T <= FRT_B_TMAX). Every token
// is computed with the same arithmetic as the M = 1 kernels (q8_1 per 32
// values, NVFP4 block dot products, fp32 per-block scaling, the stock
// epilogue order), so a token's result does not depend on the other tokens of
// its batch; only the cross-block fp32 summation order may differ from the
// M = 1 kernels where lanes cover a row differently.
//
// Weight rows are read once per kernel and applied to all T tokens: a warp
// holds RPW rows' NVFP4 fragments in registers, the tokens' q8_1 activations
// are staged through shared memory in tiles of TT tokens.

#include <map>
#include <string>

static constexpr int FRT_B_TMAX = 128;   // tokens per ubatch handled by the batched windows
static constexpr int FRT_B_NW   = 4;     // warps per GEMV CTA

// ---- private buffers --------------------------------------------------------
static block_q8_1 * g_b_q8    = nullptr;   // [T][64]  q8_1 of a 2048-wide normalized activation
static block_q8_1 * g_b_q8o   = nullptr;   // [T][128] q8_1 of the 4096-wide attention output
static block_q8_1 * g_b_q8d   = nullptr;   // [T][192] q8_1 of the 6144-wide dense GLU
static float      * g_b_qkv   = nullptr;   // [T][5120] scaled Q | K | V rows
static float      * g_b_glu   = nullptr;   // [T][6144] dense GLU / routed (6 x 512) + shared (1024) GLU
// MoE (T > 1)
static float        * g_b_logits = nullptr;  // [T][256]
static int32_t      * g_b_ids    = nullptr;  // [T][8] routing snapshot
static float        * g_b_wts    = nullptr;  // [T][8]
static float        * g_b_esc    = nullptr;  // [T][8] activation rescale per slot
static block_q8_1   * g_b_gluq8  = nullptr;  // [T][n_used*16 + 32] routed slots then shared
static int32_t      * g_b_eoff   = nullptr;  // [257] expert -> first slot in g_b_slots
static int32_t      * g_b_slots  = nullptr;  // [T*8] slot ids (t * 8 + k) grouped by expert
static int32_t      * g_b_act    = nullptr;  // [258] active experts (count at [256])
static unsigned int * g_b_done   = nullptr;  // top-k CTA completion counter
static float        * g_b_part   = nullptr;  // [T][n_used][2048] routed down outputs (unscaled)
static float        * g_b_shpart = nullptr;  // [T][2048] shared-expert down output (unscaled)

// ---- check mode (FRT_B_CHECK=1, eager test aid) -----------------------------------
// A batched window computes its final output into a private buffer and returns
// false; ggml then runs the span, and the next hook call after the span compares.
struct frt_b_chk_state { const ggml_tensor * out = nullptr; int end = -1; float * mine = nullptr; size_t cap = 0; const char * name = ""; int64_t n = 0; };
static frt_b_chk_state g_bchk;
static bool frt_b_check_on() {
    static const bool on = getenv("FRT_B_CHECK") && getenv("FRT_B_CHECK")[0] == '1';
    return on;
}
static float * frt_b_check_begin(const ggml_tensor * out, const int end, const char * name) {
    if (!frt_b_check_on() || g_bchk.out != nullptr) return nullptr;
    const int64_t n = ggml_nelements(out);
    if (g_bchk.cap < (size_t) n) {
        if (g_bchk.mine) CUDA_CHECK(cudaFree(g_bchk.mine));
        g_bchk.cap = (size_t) n;
        CUDA_CHECK(cudaMalloc(&g_bchk.mine, g_bchk.cap * sizeof(float)));
    }
    g_bchk.out = out; g_bchk.end = end; g_bchk.name = name; g_bchk.n = n;
    return g_bchk.mine;
}
static void frt_b_check_compare(ggml_backend_cuda_context & ctx, const int i) {
    if (g_bchk.out == nullptr || i <= g_bchk.end) return;
    const int64_t n = g_bchk.n;
    std::vector<float> a(n), b(n);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaMemcpy(a.data(), g_bchk.mine, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), g_bchk.out->data, n * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0, max_rel = 0, sum_abs = 0, ref_abs = 0;
    int64_t n_bad = 0, worst = 0;
    for (int64_t k = 0; k < n; ++k) {
        const double d = fabs((double) a[k] - (double) b[k]);
        const double rel = d / (fabs((double) b[k]) + 1e-6);
        sum_abs += d; ref_abs += fabs((double) b[k]);
        if (d > max_abs) { max_abs = d; worst = k; }
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-3 && d > 1e-4) ++n_bad;
    }
    struct st { long calls; double max_abs, max_rel; long bad; };
    static std::map<std::string, st> stats;
    st & S = stats[g_bchk.name];
    S.calls++; S.max_abs = std::max(S.max_abs, max_abs); S.max_rel = std::max(S.max_rel, max_rel); S.bad += n_bad;
    if (S.calls <= 4 || (S.calls & 255) == 0 || n_bad > 0) {
        fprintf(stderr, "frt-bcheck %-10s call %ld n=%lld max_abs %.3e (mine %.6g ref %.6g @%lld) max_rel %.3e mean_abs %.3e mean_ref %.3e bad %lld | run max_abs %.3e max_rel %.3e bad %ld\n",
                g_bchk.name, S.calls, (long long) n, max_abs, a[worst], b[worst], (long long) worst, max_rel, sum_abs / n, ref_abs / n, (long long) n_bad,
                S.max_abs, S.max_rel, S.bad);
    }
    g_bchk.out = nullptr;
}

static bool frt_b_on() {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_BATCHED");
    return on && g_b_q8 != nullptr;
}

static void frt_b_alloc() {
    const size_t T = FRT_B_TMAX;
    CUDA_CHECK(cudaMalloc(&g_b_q8,  T * 64  * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&g_b_q8o, T * 128 * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&g_b_q8d, T * 192 * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&g_b_qkv, T * 5120 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_b_glu, T * 6144 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_b_logits, T * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_b_ids, T * 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g_b_wts, T * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_b_esc, T * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_b_gluq8, T * (8 * 16 + 32) * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&g_b_eoff, 257 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g_b_slots, T * 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g_b_act, 258 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g_b_done, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(g_b_done, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&g_b_part, T * 8 * 2048 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_b_shpart, T * 2048 * sizeof(float)));
}

// ---- generic row dot over KB NVFP4 blocks (K = 64 KB) -------------------------
// Lane pair (2 p, 2 p + 1) covers block p + 16 w (w = 0..NBW-1); lanes whose
// block index exceeds KB (K = 512, 1024) contribute nothing.
template <int NBW>
static __device__ __forceinline__ void frt_b_ld_row(frt_nvb (&b)[NBW], const char * row, const int lane, const int KB) {
#pragma unroll
    for (int w = 0; w < NBW; ++w) {
        const int kb = lane / 2 + 16 * w;
        if (kb < KB) {
            b[w] = frt_ld_nvb(row, kb, frt_lane_iqs(lane));
        }
    }
}

template <int NBW>
static __device__ __forceinline__ float frt_b_rowdot(const frt_nvb (&b)[NBW], const block_q8_1 * y, const int lane, const int KB, const float fzero) {
    const int iqs = frt_lane_iqs(lane);
    float t = fzero;
#pragma unroll
    for (int w = 0; w < NBW; ++w) {
        const int kb = lane / 2 + 16 * w;
        if (kb < KB) {
            t += frt_vd(b[w], y + 2 * kb, iqs);
        }
    }
    return warp_reduce_sum(t);
}

// ---- prologue kernels ---------------------------------------------------------
// RMSNorm * w of T rows (one CTA per row) -> optional f32 copy and q8_1 blocks.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE) frt_b_norm_q8(
        const float * x, const int64_t ld_x, const float * norm_w, const float eps,
        float * xn_out, const int64_t ld_xn, block_q8_1 * q8_out, const int ncols) {
    ggml_cuda_pdl_lc();
    __shared__ float xn[FRT_K_EMBD];
    __shared__ float red[32];
    const int t = blockIdx.x;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    ggml_cuda_pdl_sync();
    frt_rms_norm_2048<NW>(x + (int64_t) t * ld_x, norm_w, eps, xn, red, ncols);
    __syncthreads();
    if (xn_out != nullptr) {
        float4 * o4 = (float4 *) (xn_out + (int64_t) t * ld_xn);
        for (int c = threadIdx.x; c < FRT_K_EMBD / 4; c += NW * WARP_SIZE) {
            o4[c] = ((const float4 *) xn)[c];
        }
    }
    block_q8_1 * q8 = q8_out + (int64_t) t * (FRT_K_EMBD / QK8_1);
    for (int c = warp; c < FRT_K_EMBD / QK8_1; c += NW) {
        frt_q8_1_chunk(xn[c * QK8_1 + lane], lane, &q8[c]);
    }
}

// q8_1 of T rows of K values (one CTA per row), stock quantize_q8_1 arithmetic.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE) frt_b_q8(
        const float * x, const int64_t ld_x, block_q8_1 * q8_out, const int K) {
    ggml_cuda_pdl_lc();
    const int t = blockIdx.x;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    ggml_cuda_pdl_sync();
    const float * xr = x + (int64_t) t * ld_x;
    block_q8_1 * q8 = q8_out + (int64_t) t * (K / QK8_1);
    for (int c = warp; c < K / QK8_1; c += NW) {
        frt_q8_1_chunk(xr[c * QK8_1 + lane], lane, &q8[c]);
    }
}

// ---- multi-token GEMV -----------------------------------------------------------
enum { FRT_B_EPI_SCALE = 0, FRT_B_EPI_RESID = 1, FRT_B_EPI_GLU = 2 };

struct frt_b_gemv_args {
    const block_q8_1 * yq;     // [T][2 KB] q8_1 blocks per token
    const int32_t * tok_map;   // output row t reads token tok_map[t] (nullptr: t)
    const char * w[3];         // row segments (Q | K | V), consecutive output rows
    const float * s[3];
    int seg_end[3];            // exclusive end row of each segment
    int n_seg;
    const char * w2;           // GLU: up weights (w[0] = gate)
    const float * s2;
    size_t row_stride;
    float * out;               // [T][ld_out]
    int64_t ld_out;
    const float * resid;       // RESID: [.][ld_resid], row tok_map[t]
    int64_t ld_resid;
    int n_rows;
    int T;
    float fzero;
};

template <int KB, int RPW, int NW, int TT, int EPI>
static __global__ void __launch_bounds__(NW * WARP_SIZE) frt_b_gemv(const frt_b_gemv_args a) {
    constexpr int NBW = (KB + 15) / 16;
    constexpr int Q8T = 2 * KB;              // q8_1 blocks per token
    constexpr int TOK_BYTES = Q8T * (int) sizeof(block_q8_1);
    static_assert(TOK_BYTES % 16 == 0, "token q8 rows copy in int4");
    ggml_cuda_pdl_lc();
    __shared__ __align__(16) block_q8_1 yq[TT][Q8T];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r0 = (blockIdx.x * NW + warp) * RPW;
    frt_nvb b[RPW][NBW];
    frt_nvb b2[EPI == FRT_B_EPI_GLU ? RPW : 1][NBW];
    float sc[RPW], sc2[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const int r = r0 + i;
        sc[i] = 0.0f;
        sc2[i] = 0.0f;
        if (r < a.n_rows) {   // weights and scales are immutable: fetch before the dependency wait
            int sg = 0, base = 0;
            while (sg + 1 < a.n_seg && r >= a.seg_end[sg]) { base = a.seg_end[sg]; ++sg; }
            frt_b_ld_row<NBW>(b[i], a.w[sg] + (size_t) (r - base) * a.row_stride, lane, KB);
            sc[i] = a.s[sg][0];
            if constexpr (EPI == FRT_B_EPI_GLU) {
                frt_b_ld_row<NBW>(b2[i], a.w2 + (size_t) r * a.row_stride, lane, KB);
                sc2[i] = a.s2[0];
            }
        }
    }
    ggml_cuda_pdl_sync();
    for (int t0 = 0; t0 < a.T; t0 += TT) {
        const int nt = min(TT, a.T - t0);
        // stage nt tokens' q8_1 rows (written by the previous kernel: read through L2)
        for (int c = threadIdx.x; c < nt * (TOK_BYTES / 16); c += NW * WARP_SIZE) {
            const int tt = c / (TOK_BYTES / 16), q = c % (TOK_BYTES / 16);
            const int tok = a.tok_map ? a.tok_map[t0 + tt] : t0 + tt;
            ((int4 *) yq[tt])[q] = __ldcg((const int4 *) (a.yq + (int64_t) tok * Q8T) + q);
        }
        __syncthreads();
        for (int tt = 0; tt < nt; ++tt) {
            const block_q8_1 * y = yq[tt];
            const int t = t0 + tt;
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const int r = r0 + i;
                if (r >= a.n_rows) continue;
                const float d = frt_b_rowdot<NBW>(b[i], y, lane, KB, a.fzero);
                float d2 = 0.0f;
                if constexpr (EPI == FRT_B_EPI_GLU) {
                    d2 = frt_b_rowdot<NBW>(b2[i], y, lane, KB, a.fzero);
                }
                if (lane == 0) {
                    float result = d;   // mmvq epilogue: scale, zero bias
                    result *= sc[i];
                    result += a.fzero;
                    if constexpr (EPI == FRT_B_EPI_RESID) {
                        const int tok = a.tok_map ? a.tok_map[t] : t;
                        float o = result;   // k_bin_bcast add
                        o = o + a.resid[(int64_t) tok * a.ld_resid + r];
                        result = o;
                    } else if constexpr (EPI == FRT_B_EPI_GLU) {
                        // fused mmvq epilogue: x = up, gate = gate, SwiGLU
                        float up = d2;
                        up *= sc2[i];
                        up += a.fzero;
                        up *= ggml_cuda_op_silu_single(result);
                        result = up;
                    }
                    a.out[(int64_t) t * a.ld_out + r] = result;
                }
            }
        }
        __syncthreads();
    }
}

struct frt_b_moe_args {
    const block_q8_1 * q8;      // routed: [T][ld_q8] (+ k * kq8 for slot k); shared: + sh_q8_off
    int ld_q8, kq8, sh_q8_off;  // blocks
    const int32_t * eoff, * slots, * act;
    const char * w, * w2;       // routed weights (w2: up for GLU), expert e at + e * e_stride
    size_t e_stride, row_stride, sh_row_stride;
    const float * sc, * sc2;    // per-expert scales
    const char * shw, * shw2;   // shared-expert weights
    const float * shsc, * shsc2; // per-tensor scales
    float * out;                // routed: out + t * ld_out + k * k_out + r
    int64_t ld_out, k_out;
    float * sh_out;             // shared: sh_out + t * sh_ld + r
    int64_t sh_ld;
    int n_rows, sh_rows;        // output rows per routed expert / of the shared expert
    int T, n_used;
    int tiles_r;                // row tiles per routed expert; shared tiles follow
    float fzero;
};

#include "fr_win_flint_bmma.cuh"

static bool frt_b_dp4a() {
    static const bool on = getenv("FRT_B_DP4A") && getenv("FRT_B_DP4A")[0] == '1';
    return on;
}
// sm_120 allows 99 KB of shared memory per block: token tiles of 32 / 16 / 8 for K = 2048 / 4096 / 6144
template <int KB, int EPI> struct frt_m_ntt { static constexpr int v = KB <= 64 ? 2 : 1; };   // 16-token tiles (8 for K = 6144)
template <int KB> struct frt_m_nw { static constexpr int v = KB <= 32 ? 8 : 16; };

// token-tile size per K: keep the static shared buffer under 48 KB
template <int KB> struct frt_b_tt { static constexpr int v = KB <= 16 ? 32 : KB <= 32 ? 16 : KB <= 64 ? 8 : 4; };

template <int KB, int EPI>
static void frt_b_gemv_launch(const frt_b_gemv_args & a, cudaStream_t stream) {
    constexpr int NW = FRT_B_NW, TT = frt_b_tt<KB>::v;
    if (!frt_b_dp4a()) {
        frt_m_gemv_launch<KB, frt_m_nw<KB>::v, frt_m_ntt<KB, EPI>::v, EPI>(a, stream);
        return;
    }
    // rows per warp: as many as keep >= ~256 CTAs
    const int rpw = a.n_rows >= 4096 * 1 ? 4 : a.n_rows >= 2048 ? 2 : 1;
    switch (rpw) {
        case 4: {
            const int cta = (a.n_rows + NW * 4 - 1) / (NW * 4);
            ggml_cuda_kernel_launch(frt_b_gemv<KB, 4, NW, TT, EPI>, ggml_cuda_kernel_launch_params((dim3) cta, (dim3) (NW * WARP_SIZE), 0, stream), a);
        } break;
        case 2: {
            const int cta = (a.n_rows + NW * 2 - 1) / (NW * 2);
            ggml_cuda_kernel_launch(frt_b_gemv<KB, 2, NW, TT, EPI>, ggml_cuda_kernel_launch_params((dim3) cta, (dim3) (NW * WARP_SIZE), 0, stream), a);
        } break;
        default: {
            const int cta = (a.n_rows + NW - 1) / NW;
            ggml_cuda_kernel_launch(frt_b_gemv<KB, 1, NW, TT, EPI>, ggml_cuda_kernel_launch_params((dim3) cta, (dim3) (NW * WARP_SIZE), 0, stream), a);
        } break;
    }
}

static frt_b_gemv_args frt_b_args1(const block_q8_1 * yq, const char * w, const float * s, const size_t row_stride,
        float * out, const int64_t ld_out, const int n_rows, const int T) {
    frt_b_gemv_args a = {};
    a.yq = yq;
    a.w[0] = w; a.s[0] = s; a.seg_end[0] = n_rows; a.n_seg = 1;
    a.row_stride = row_stride;
    a.out = out; a.ld_out = ld_out;
    a.n_rows = n_rows; a.T = T; a.fzero = 0.0f;
    return a;
}

// ---- attention front, T tokens -------------------------------------------------
// K_qkn for T tokens: one warp per (token, head).
template <bool rope>
static __global__ void frt_b_attn_qkn(const float * qkv, const int64_t ld_qkv, const frt_qkn_params p, const int T) {
    ggml_cuda_pdl_lc();
    const int g = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int nh = p.hq + 2 * p.hk;
    if (g >= T * nh) {
        return;
    }
    const int t = g / nh, h = g % nh;
    ggml_cuda_pdl_sync();
    frt_qkn_head<rope>(qkv + (int64_t) t * ld_qkv, h, lane, p, p.pos ? p.pos[t] : 0, p.k_idx[t], p.v_idx[t],
                       p.q_out + (int64_t) t * p.hq * 128);
}


// ---- host double-precision reference (check mode) -------------------------------------
// NVFP4 x q8_1 dot products as ggml defines them (q8_1 of the activation with the
// fp16-rounded block scale, 2*E2M1 integer weights, UE4M3/2 sub-block scales),
// accumulated in double. Used to verify the batched windows at any T without
// depending on stock's M > 8 kernels (which quantize more coarsely).
struct frt_h_q8 { std::vector<int8_t> q; std::vector<float> d; };

static void frt_h_quant(const float * x, const int K, frt_h_q8 & o) {
    o.q.resize(K); o.d.resize(K / 32);
    for (int c = 0; c < K / 32; ++c) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, fabsf(x[c * 32 + j]));
        const float d = amax / 127.0f;
        for (int j = 0; j < 32; ++j) o.q[c * 32 + j] = amax == 0.0f ? 0 : (int8_t) roundf(x[c * 32 + j] / d);
        o.d[c] = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(d));
    }
}

// q8_1 blocks produced by the device (ds half2 then 32 int8)
static void frt_h_q8_from_dev(const block_q8_1 * dev, const int K, frt_h_q8 & o) {
    std::vector<block_q8_1> b(K / 32);
    CUDA_CHECK(cudaMemcpy(b.data(), dev, b.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
    o.q.resize(K); o.d.resize(K / 32);
    for (int c = 0; c < K / 32; ++c) {
        memcpy(&o.q[c * 32], b[c].qs, 32);
        o.d[c] = __half2float(__low2half(b[c].ds));
    }
}
// device quantization vs host quantization of the same f32 row: flips (int8 differs by 1) and scale mismatches
static void frt_h_quant_cmp(const char * name, const float * x, const frt_h_q8 & dev, const int K, long & flips, long & dbad) {
    frt_h_q8 h; frt_h_quant(x, K, h);
    for (int j = 0; j < K; ++j) if (h.q[j] != dev.q[j]) ++flips;
    for (int c = 0; c < K / 32; ++c) if (h.d[c] != dev.d[c]) ++dbad;
    GGML_UNUSED(name);
}

static float frt_h_ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
    const int exp = (x >> 3) & 0xF, man = x & 7;
    const float raw = exp == 0 ? ldexpf((float) man, -9) : ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    return raw / 2;   // sign bit ignored, as the CUDA decode does
}

// row: NVFP4 blocks of 64 (4 scale bytes, 32 nibble bytes); K/64 blocks
static double frt_h_dot(const uint8_t * row, const frt_h_q8 & y, const int K) {
    static const int kv[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    double acc = 0.0;
    for (int b = 0; b < K / 64; ++b) {
        const uint8_t * blk = row + (size_t) b * 36;
        for (int sb = 0; sb < 4; ++sb) {
            const uint8_t * qs = blk + 4 + 8 * sb;
            int sumi = 0;
            for (int e = 0; e < 16; ++e) {   // element e of the sub-block: low nibbles of bytes 0..7 then high nibbles
                const int nib = e < 8 ? (qs[e] & 0xF) : (qs[e - 8] >> 4);
                sumi += kv[nib] * (int) y.q[b * 64 + sb * 16 + e];
            }
            acc += (double) frt_h_ue4m3(blk[sb]) * (double) y.d[(b * 64 + sb * 16) / 32] * (double) sumi;
        }
    }
    return acc;
}

static void frt_h_rmsnorm(const float * x, const float * w, const float eps, const int K, std::vector<float> & out) {
    double ss = 0.0;
    for (int j = 0; j < K; ++j) ss += (double) x[j] * x[j];
    const float scale = 1.0f / sqrtf((float) (ss / K) + eps);
    out.resize(K);
    for (int j = 0; j < K; ++j) out[j] = x[j] * scale * w[j];
}

static float frt_h_silu(float x) { return x / (1.0f + expf(-x)); }

struct frt_h_cmp { double max_abs = 0, max_rel = 0, sum_abs = 0, sum_ref = 0; long n = 0, bad = 0; };
static void frt_h_acc(frt_h_cmp & c, double mine, double ref) {
    const double d = fabs(mine - ref), rel = d / (fabs(ref) + 1e-6);
    c.max_abs = std::max(c.max_abs, d); c.max_rel = std::max(c.max_rel, rel);
    c.sum_abs += d; c.sum_ref += fabs(ref); c.n++;
    if (rel > 2e-3 && d > 1e-4) c.bad++;
}
static void frt_h_report(const char * name, long call, const frt_h_cmp & c) {
    fprintf(stderr, "frt-href %-10s call %ld n=%ld max_abs %.3e max_rel %.3e mean_abs %.3e mean_ref %.3e bad %ld\n",
            name, call, c.n, c.max_abs, c.max_rel, c.sum_abs / std::max(1L, c.n), c.sum_ref / std::max(1L, c.n), c.bad);
}
static std::vector<float> frt_h_d2h(const void * p, size_t n) {
    std::vector<float> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}
static std::vector<uint8_t> frt_h_d2h_u8(const void * p, size_t n) {
    std::vector<uint8_t> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), p, n, cudaMemcpyDeviceToHost));
    return v;
}
static std::vector<int32_t> frt_h_d2h_i32(const void * p, size_t n) {
    std::vector<int32_t> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    return v;
}
// tokens sampled per call: first, second, middle, last
static std::vector<int> frt_h_tokens(const int T) {
    std::vector<int> t = { 0 };
    if (T > 1) t.push_back(1);
    if (T > 3) t.push_back(T / 2);
    if (T > 2) t.push_back(T - 1);
    return t;
}
static bool frt_h_check_on() {
    static const bool on = getenv("FRT_B_HREF") && getenv("FRT_B_HREF")[0] == '1';
    return on;
}
static constexpr long FRT_H_CALLS = 6;   // calls per window verified against the host reference

// ---- per-window host reference checks (FRT_B_HREF=1; run with GGML_CUDA_DISABLE_GRAPHS=1) ----
// raw Q|K|V GEMV: g_b_qkv[t][r] vs s_seg * dot(w_seg row, q8(rmsnorm(x_t) * nw))
static void frt_h_check_qkv(cudaStream_t stream, const float * x, const float * nw, const float eps, const int T,
        const char * wq, const char * wk, const char * wv, const size_t row_stride, const float * sq, const float * sk, const float * sv,
        const int nq, const int nk) {
    static long calls = 0;
    if (T < 4 || ++calls > FRT_H_CALLS) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int nrows = nq + 2 * nk;
    const auto xs = frt_h_d2h(x, (size_t) T * FRT_K_EMBD);
    const auto nws = frt_h_d2h(nw, FRT_K_EMBD);
    const auto mine = frt_h_d2h(g_b_qkv, (size_t) T * nrows);
    const auto wqs = frt_h_d2h_u8(wq, (size_t) nq * row_stride);
    const auto wks = frt_h_d2h_u8(wk, (size_t) nk * row_stride);
    const auto wvs = frt_h_d2h_u8(wv, (size_t) nk * row_stride);
    const auto s = frt_h_d2h(sq, 1);
    const auto s2 = frt_h_d2h(sk, 1);
    const auto s3 = frt_h_d2h(sv, 1);
    frt_h_cmp c;
    long flips = 0, dbad = 0;
    for (const int t : frt_h_tokens(T)) {
        std::vector<float> xn;
        frt_h_rmsnorm(xs.data() + (size_t) t * FRT_K_EMBD, nws.data(), eps, FRT_K_EMBD, xn);
        frt_h_q8 y; frt_h_q8_from_dev(g_b_q8 + (size_t) t * (FRT_K_EMBD / QK8_1), FRT_K_EMBD, y);
        frt_h_quant_cmp("qkv", xn.data(), y, FRT_K_EMBD, flips, dbad);
        for (int r = 0; r < nrows; r += 7) {   // every 7th row
            const uint8_t * row; float sc;
            if (r < nq)           { row = wqs.data() + (size_t) r * row_stride; sc = s[0]; }
            else if (r < nq + nk) { row = wks.data() + (size_t) (r - nq) * row_stride; sc = s2[0]; }
            else                  { row = wvs.data() + (size_t) (r - nq - nk) * row_stride; sc = s3[0]; }
            frt_h_acc(c, mine[(size_t) t * nrows + r], frt_h_dot(row, y, FRT_K_EMBD) * sc);
        }
    }
    frt_h_report("qkv-raw", calls, c);
    fprintf(stderr, "frt-href qkv-q8     call %ld: %ld int8 flips, %ld scale mismatches (host quant of host norm vs device)\n", calls, flips, dbad);
}

// out[t][r] (T_out rows, token map) vs s * dot(w row, q8(a[tok])) + resid[tok][r]
static void frt_h_check_oproj(cudaStream_t stream, const float * a, const int T, const int T_out, const int32_t * tok_map,
        const char * w, const size_t row_stride, const float * s, const float * resid, const float * out, const int n_out) {
    static long calls = 0;
    if (T < 4 || ++calls > FRT_H_CALLS) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto as = frt_h_d2h(a, (size_t) T * 4096);
    const auto rs = frt_h_d2h(resid, (size_t) T * n_out);
    const auto os = frt_h_d2h(out, (size_t) T_out * n_out);
    const auto ws = frt_h_d2h_u8(w, (size_t) n_out * row_stride);
    const auto sc = frt_h_d2h(s, 1);
    std::vector<int32_t> map;
    if (tok_map) map = frt_h_d2h_i32(tok_map, T_out);
    frt_h_cmp c;
    long flips = 0, dbad = 0;
    for (const int t : frt_h_tokens(T_out)) {
        const int tok = tok_map ? map[t] : t;
        frt_h_q8 y; frt_h_q8_from_dev(g_b_q8o + (size_t) tok * (4096 / QK8_1), 4096, y);
        frt_h_quant_cmp("oproj", as.data() + (size_t) tok * 4096, y, 4096, flips, dbad);
        for (int r = 0; r < n_out; r += 5) {
            frt_h_acc(c, os[(size_t) t * n_out + r], frt_h_dot(ws.data() + (size_t) r * row_stride, y, 4096) * sc[0] + rs[(size_t) tok * n_out + r]);
        }
    }
    frt_h_report("oproj", calls, c);
    fprintf(stderr, "frt-href oproj-q8   call %ld: %ld int8 flips, %ld scale mismatches\n", calls, flips, dbad);
}

static void frt_h_check_dense(cudaStream_t stream, const float * x, const float * nw, const float eps, const int T,
        const char * gw, const char * uw, const char * dw, const size_t row_stride, const size_t drow_stride,
        const float * gs, const float * us, const float * ds, const float * resid, const float * out) {
    static long calls = 0;
    if (T < 4 || ++calls > FRT_H_CALLS) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto xs = frt_h_d2h(x, (size_t) T * FRT_K_EMBD);
    const auto nws = frt_h_d2h(nw, FRT_K_EMBD);
    const auto rs = frt_h_d2h(resid, (size_t) T * FRT_K_EMBD);
    const auto os = frt_h_d2h(out, (size_t) T * FRT_K_EMBD);
    const auto gws = frt_h_d2h_u8(gw, (size_t) 6144 * row_stride);
    const auto uws = frt_h_d2h_u8(uw, (size_t) 6144 * row_stride);
    const auto dws = frt_h_d2h_u8(dw, (size_t) FRT_K_EMBD * drow_stride);
    const auto sg = frt_h_d2h(gs, 1);
    const auto su = frt_h_d2h(us, 1);
    const auto sd = frt_h_d2h(ds, 1);
    const auto dglu = frt_h_d2h(g_b_glu, (size_t) T * 6144);
    frt_h_cmp c, cg;
    long flips = 0, dbad = 0;
    for (const int t : frt_h_tokens(T)) {
        std::vector<float> xn, glu(6144);
        frt_h_rmsnorm(xs.data() + (size_t) t * FRT_K_EMBD, nws.data(), eps, FRT_K_EMBD, xn);
        frt_h_q8 y; frt_h_q8_from_dev(g_b_q8 + (size_t) t * (FRT_K_EMBD / QK8_1), FRT_K_EMBD, y);
        for (int r = 0; r < 6144; ++r) {
            const float g = (float) frt_h_dot(gws.data() + (size_t) r * row_stride, y, FRT_K_EMBD) * sg[0];
            const float u = (float) frt_h_dot(uws.data() + (size_t) r * row_stride, y, FRT_K_EMBD) * su[0];
            glu[r] = u * frt_h_silu(g);
            frt_h_acc(cg, dglu[(size_t) t * 6144 + r], glu[r]);
        }
        frt_h_q8 yg; frt_h_q8_from_dev(g_b_q8d + (size_t) t * (6144 / QK8_1), 6144, yg);
        frt_h_quant_cmp("dense", dglu.data() + (size_t) t * 6144, yg, 6144, flips, dbad);
        for (int r = 0; r < FRT_K_EMBD; r += 3) {
            frt_h_acc(c, os[(size_t) t * FRT_K_EMBD + r], frt_h_dot(dws.data() + (size_t) r * drow_stride, yg, 6144) * sd[0] + rs[(size_t) t * FRT_K_EMBD + r]);
        }
    }
    frt_h_report("dense-glu", calls, cg);
    frt_h_report("denseffn", calls, c);
    fprintf(stderr, "frt-href dense-q8   call %ld: %ld int8 flips, %ld scale mismatches (device glu requantized on host)\n", calls, flips, dbad);
}

static void frt_h_check_lmhead(cudaStream_t stream, const float * xn_rows, const int T,
        const char * w, const size_t row_stride, const float * s, const float * out, const int n_rows) {
    static long calls = 0;
    if (T < 4 || ++calls > FRT_H_CALLS) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int nr = std::min(n_rows, 4096);
    const auto xs = frt_h_d2h(xn_rows, (size_t) T * FRT_K_EMBD);
    const auto os = frt_h_d2h(out, (size_t) T * n_rows);
    const auto ws = frt_h_d2h_u8(w, (size_t) nr * row_stride);
    const auto sc = frt_h_d2h(s, 1);
    frt_h_cmp c;
    for (const int t : frt_h_tokens(T)) {
        frt_h_q8 y; frt_h_q8_from_dev(g_b_q8 + (size_t) t * (FRT_K_EMBD / QK8_1), FRT_K_EMBD, y);
        for (int r = 0; r < nr; r += 3) {
            frt_h_acc(c, os[(size_t) t * n_rows + r], frt_h_dot(ws.data() + (size_t) r * row_stride, y, FRT_K_EMBD) * sc[0]);
        }
    }
    frt_h_report("lmhead", calls, c);
}

// routing: fn (our normalized rows) -> bf16 router logits in double -> sigmoid -> top-k (ties: lower id) -> clamp-normalized weights
static void frt_h_check_route(cudaStream_t stream, const float * fn, const int T, const nv_bfloat16 * wr, const int64_t wr_stride,
        const int n_used, const float clamp_val, const int32_t * ids, const int64_t ids_stride, const float * wts) {
    static long calls = 0;
    if (T < 4 || ++calls > FRT_H_CALLS) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto fs = frt_h_d2h(fn, (size_t) T * FRT_K_EMBD);
    std::vector<uint16_t> w16((size_t) 256 * wr_stride);
    CUDA_CHECK(cudaMemcpy(w16.data(), wr, w16.size() * 2, cudaMemcpyDeviceToHost));
    const auto ws = frt_h_d2h(wts, (size_t) T * n_used);
    const auto is = frt_h_d2h_i32(ids, (size_t) (T - 1) * ids_stride + 256);
    frt_h_cmp c;
    long id_bad = 0;
    for (int t = 0; t < T; ++t) {
        double p[256];
        for (int e = 0; e < 256; ++e) {
            double acc = 0.0;
            const uint16_t * row = w16.data() + (size_t) e * wr_stride;
            for (int j = 0; j < FRT_K_EMBD; ++j) {
                uint32_t b = (uint32_t) row[j] << 16; float wf; memcpy(&wf, &b, 4);
                acc += (double) wf * (double) fs[(size_t) t * FRT_K_EMBD + j];
            }
            p[e] = 1.0 / (1.0 + exp(-acc));
        }
        int sel[8]; double sw[8], sum = 0.0;
        bool used[256] = {};
        for (int k = 0; k < n_used; ++k) {
            int best = -1;
            for (int e = 0; e < 256; ++e) if (!used[e] && (best < 0 || p[e] > p[best])) best = e;
            used[best] = true; sel[k] = best; sw[k] = p[best]; sum += p[best];
        }
        sum = std::max(sum, (double) clamp_val);
        for (int k = 0; k < n_used; ++k) {
            frt_h_acc(c, ws[(size_t) t * n_used + k], sw[k] / sum);
            if (is[(size_t) t * ids_stride + k] != sel[k]) ++id_bad;
        }
    }
    frt_h_report("route-w", calls, c);
    if (id_bad) fprintf(stderr, "frt-href route-ids call %ld: %ld of %d slot ids differ from the double reference\n", calls, id_bad, T * n_used);
}

