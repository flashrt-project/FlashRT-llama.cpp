// SPDX-License-Identifier: Apache-2.0
//
// FlashRT ggml adapter — RTX 5090 (SM120) + Recursive Flint (NVFP4 GGUF)
// decode window set. Pattern-matched subgraph windows over ggml-cuda's fuse
// hook; every window pins the graph shape it replaces and falls back to the
// stock kernels when its predicate does not match.
//
// Runtime switches (unset = enabled):
//   GGML_CUDA_FLASHRT_DISABLE=1   whole layer off (stock llama.cpp)
//   GGML_FLASHRT_NO_<X>=1         per-window disable
// Diagnostics: FRT_DUMP_GRAPH=1 (+ FRT_DUMP_M, FRT_DUMP_PATH), FRT_STATS=1.

#include "common.cuh"
#include "ggml-impl.h"
#include "vecdotq.cuh"
#include "unary.cuh"

#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#include <cstring>

// ---- switch semantics -----------------------------------------------------
static bool frt_layer_enabled(void) {
    static int on = -1;
    if (on < 0) { const char * s = getenv("GGML_CUDA_FLASHRT_DISABLE"); on = (s && s[0] == '1') ? 0 : 1; }
    return on == 1;
}

static bool frt_window_on(const char * no_env) {
    if (!frt_layer_enabled()) return false;
    const char * n = getenv(no_env);
    return !(n && n[0] == '1');
}

// ---- graph dump (region-boundary mapping) ---------------------------------
static void frt_dump_graph(const ggml_cgraph * cgraph) {
    static int dump = -1;
    if (dump < 0) { const char * d = getenv("FRT_DUMP_GRAPH"); dump = (d && d[0] == '1') ? 1 : 0; }
    if (dump != 1 || cgraph->n_nodes < 200) return;
    static int dump_m = -1;
    if (dump_m < 0) { const char * d = getenv("FRT_DUMP_M"); dump_m = d ? atoi(d) : 1; }
    bool match = false;
    for (int i = 0; i < cgraph->n_nodes && i < 200; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (strncmp(n->name, "attn_norm", 9) == 0) { match = (n->ne[1] == dump_m); break; }
    }
    if (!match) return;
    dump = 2;
    const char * path = getenv("FRT_DUMP_PATH");
    FILE * f = fopen(path ? path : "frt_graph_dump.txt", "w");
    if (!f) return;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        fprintf(f, "%4d %-14s %-30s %-6s ne=[%lld,%lld,%lld,%lld]", i, ggml_op_name(n->op), n->name,
                ggml_type_name(n->type), (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3]);
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
            const ggml_tensor * t = n->src[s];
            fprintf(f, " | s%d=%s:%s[%lld,%lld,%lld]", s, t->name, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
        }
        fprintf(f, " flags=%d\n", (int) n->flags);
    }
    fclose(f);
    fprintf(stderr, "frt: graph dumped (%d nodes)\n", cgraph->n_nodes);
}

// warps per CTA of a GEMV window (tile search: FRT_NW_<KERNEL>=4|8|16)
static int frt_cfg_nw(const char * env, const int def) {
    const char * e = getenv(env);
    const int v = e ? atoi(e) : def;
    return (v == 4 || v == 8 || v == 16 || v == 32) ? v : def;
}

#define FRT_NW_DISPATCH(nw, ...) do { switch (nw) { \
    case 4:  { constexpr int NWT = 4;  __VA_ARGS__; } break; \
    case 16: { constexpr int NWT = 16; __VA_ARGS__; } break; \
    default: { constexpr int NWT = 8;  __VA_ARGS__; } break; } } while (0)

#define FRT_NW_DISPATCH32(nw, ...) do { switch (nw) { \
    case 4:  { constexpr int NWT = 4;  __VA_ARGS__; } break; \
    case 8:  { constexpr int NWT = 8;  __VA_ARGS__; } break; \
    case 32: { constexpr int NWT = 32; __VA_ARGS__; } break; \
    default: { constexpr int NWT = 16; __VA_ARGS__; } break; } } while (0)

static bool frt_stats_on(void) {
    static int on = -1;
    if (on < 0) { const char * s = getenv("FRT_STATS"); on = (s && s[0] == '1') ? 1 : 0; }
    return on == 1;
}

// node j exists, is computed, and (when elided) feeds only the next node
static bool frt_node_ok(const ggml_cgraph * cgraph, int j, ggml_op op, int n_uses) {
    if (j >= cgraph->n_nodes) return false;
    const ggml_tensor * n = cgraph->nodes[j];
    if (n->op != op || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) return false;
    if (n_uses > 0 && !ggml_node_has_n_uses(cgraph, j, n_uses)) return false;
    return true;
}

static bool frt_unary_is(const ggml_tensor * n, ggml_unary_op op) {
    return n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == op;
}

// ---- MoE expert-activation rescale (AMAX) ---------------------------------
// Flint's graph protects the q8_1 activation of the routed-expert down
// projection against f16 overflow with a per-(expert, token) rescale:
//   e = exp(scale(clamp(ceil(scale(log(scale(max|glu|)))))));  x' = glu / e
// and multiplies e back after the down GEMV. The graph runs it as ten
// launches, one of them a single-block serial pool. This window computes the
// same chain in one launch, one block per row, with the identical f32
// expression sequence (same translation-unit math flags as ggml-cuda), so x'
// and e are bit-identical to stock. The max reduction is order-independent.
template <int NT>
static __global__ void frt_amax_rescale(
        const float * x, float * out, float * e_out, const int n,
        const int64_t x_srow, const int64_t o_srow,
        const float s1, const float b1, const float s2, const float b2,
        const float cmin, const float cmax, const float s3, const float b3) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int r = blockIdx.x;
    const float * xr = x + (int64_t) r * x_srow;
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < n; j += NT) {
        m = fmaxf(m, fabsf(xr[j]));
    }
    m = warp_reduce_max(m);
    __shared__ float red[NT / WARP_SIZE];
    __shared__ float e_sh;
    if (threadIdx.x % WARP_SIZE == 0) red[threadIdx.x / WARP_SIZE] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = red[0];
#pragma unroll
        for (int w = 1; w < NT / WARP_SIZE; ++w) a = fmaxf(a, red[w]);
        float e = s1 * a + b1;      // SCALE
        e = logf(e);                // LOG
        e = s2 * e + b2;            // SCALE
        e = ceilf(e);               // CEIL
        e = fminf(fmaxf(e, cmin), cmax);   // CLAMP
        e = s3 * e + b3;            // SCALE
        e = expf(e);                // EXP
        e_sh = e;
        e_out[r] = e;
    }
    __syncthreads();
    const float e = e_sh;
    float * o = out + (int64_t) r * o_srow;
    for (int j = threadIdx.x; j < n; j += NT) {
        o[j] = xr[j] / e;           // DIV
    }
}

// anchor: UNARY(ABS) over the routed-expert GLU output.
static bool frt_amax_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_AMAX");
    if (!on) return false;
    const ggml_tensor * n0 = cgraph->nodes[i];
    if (!frt_unary_is(n0, GGML_UNARY_OP_ABS)) return false;
    const ggml_tensor * g = n0->src[0];
    if (g->type != GGML_TYPE_F32 || g->op != GGML_OP_GLU || !ggml_is_contiguous(g)) return false;
    // exact op sequence, consecutive, intermediates single-use
    static const ggml_op ops[10] = { GGML_OP_UNARY, GGML_OP_POOL_1D, GGML_OP_SCALE, GGML_OP_LOG, GGML_OP_SCALE,
                                     GGML_OP_UNARY, GGML_OP_CLAMP, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_DIV };
    for (int k = 0; k < 10; ++k) {
        if (!frt_node_ok(cgraph, i + k, ops[k], k < 8 ? 1 : 0)) return false;
        if (k > 0 && k < 9 && cgraph->nodes[i + k]->src[0] != cgraph->nodes[i + k - 1]) return false;
    }
    const ggml_tensor * pool = cgraph->nodes[i + 1];
    const ggml_tensor * sc1  = cgraph->nodes[i + 2];
    const ggml_tensor * sc2  = cgraph->nodes[i + 4];
    const ggml_tensor * ceil = cgraph->nodes[i + 5];
    const ggml_tensor * clmp = cgraph->nodes[i + 6];
    const ggml_tensor * sc3  = cgraph->nodes[i + 7];
    ggml_tensor * ex   = cgraph->nodes[i + 8];
    ggml_tensor * dv   = cgraph->nodes[i + 9];
    if (!frt_unary_is(ceil, GGML_UNARY_OP_CEIL) || !frt_unary_is(ex, GGML_UNARY_OP_EXP)) return false;
    if (dv->src[0] != g || dv->src[1] != ex) return false;
    // pool: MAX over the whole row
    const int32_t * pp = (const int32_t *) pool->op_params;
    const int64_t n = g->ne[0];
    if (pp[0] != GGML_OP_POOL_MAX || pp[1] != n || pp[2] != n || pp[3] != 0 || pool->ne[0] != 1) return false;
    const int64_t rows = ggml_nrows(g);
    if (ggml_nrows(ex) != rows || ex->ne[0] != 1 || !ggml_is_contiguous(ex)) return false;
    if (!ggml_are_same_shape(dv, g) || dv->type != GGML_TYPE_F32 || !ggml_is_contiguous(dv)) return false;
    for (int k = 0; k < 9; ++k) {
        if (cgraph->nodes[i + k]->type != GGML_TYPE_F32) return false;
    }
    if (rows > INT_MAX || n > INT_MAX) return false;

    const ggml_cuda_kernel_launch_params lp((dim3) (unsigned) rows, (dim3) 128, 0, ctx.stream());
    ggml_cuda_kernel_launch(frt_amax_rescale<128>, lp,
        (const float *) g->data, (float *) dv->data, (float *) ex->data, (int) n,
        (int64_t) (g->nb[1] / sizeof(float)), (int64_t) (dv->nb[1] / sizeof(float)),
        ggml_get_op_params_f32(sc1, 0), ggml_get_op_params_f32(sc1, 1),
        ggml_get_op_params_f32(sc2, 0), ggml_get_op_params_f32(sc2, 1),
        ggml_get_op_params_f32(clmp, 0), ggml_get_op_params_f32(clmp, 1),
        ggml_get_op_params_f32(sc3, 0), ggml_get_op_params_f32(sc3, 1));
    if (frt_stats_on()) {
        static long fires = 0;
        if ((++fires & 0xFFF) == 1) fprintf(stderr, "frt-stats amax fires=%ld rows=%lld\n", fires, (long long) rows);
    }
    *skip_count = 10;
    return true;
}

// ---- graph-matching helpers -------------------------------------------------

static constexpr int FRT_MOE_MAX_USED = 8;
static constexpr int FRT_K_EMBD = 2048;


static bool frt_aligned16(const void * p) { return ((uintptr_t) p & 15) == 0; }

// The graph allocator may place a window's output on the buffer of a tensor
// that dies inside the window. Kernels whose CTAs read an input after other
// CTAs may already have written an output must reject such placements.
static bool frt_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_backend_buffer_get_alloc_size(a->buffer, a);
    const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_backend_buffer_get_alloc_size(b->buffer, b);
    return a0 < b1 && b0 < a1;
}

// scale tensor behind the GET_ROWS(REPEAT(RESHAPE(scale)), ids) chain of a
// scaled MUL_MAT_ID, or nullptr when the chain does not match.
static const ggml_tensor * frt_id_scale_src(const ggml_tensor * mul, const ggml_tensor * mm, const ggml_tensor * ids) {
    if (mul->op != GGML_OP_MUL || mul->src[0] != mm) return nullptr;
    const ggml_tensor * gr = mul->src[1];
    if (gr->op != GGML_OP_GET_ROWS || gr->src[1] != ids) return nullptr;
    const ggml_tensor * rp = gr->src[0];
    if (rp->op != GGML_OP_REPEAT || rp->ne[1] != mm->src[0]->ne[2]) return nullptr;
    const ggml_tensor * rs = rp->src[0];
    if (rs->op != GGML_OP_RESHAPE) return nullptr;
    const ggml_tensor * s = rs->src[0];
    if (s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) || s->ne[0] != mm->src[0]->ne[2]) return nullptr;
    return s;
}

// per-tensor scale of MUL(MUL_MAT(w, x), s): returns s or nullptr
static const ggml_tensor * frt_mm_scale_src(const ggml_tensor * mul, const ggml_tensor * mm) {
    if (mul->op != GGML_OP_MUL || mul->src[0] != mm) return nullptr;
    const ggml_tensor * s = mul->src[1];
    if (s->type != GGML_TYPE_F32 || ggml_nelements(s) != 1) return nullptr;
    return s;
}

// every node of [a, b) except `keep` must be consumed only inside [a, b]
static bool frt_span_closed(const ggml_cgraph * cgraph, int a, int b, std::initializer_list<const ggml_tensor *> keep) {
    for (int j = a; j < b; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (std::find(keep.begin(), keep.end(), n) != keep.end()) continue;
        if (n->flags & GGML_TENSOR_FLAG_OUTPUT) return false;
        int refs = 0;
        for (int k = a; k <= b; ++k) {
            const ggml_tensor * m = cgraph->nodes[k];
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (m->src[s] == n) ++refs;
            }
        }
        if (ggml_node_get_use_count(cgraph, j) != refs) return false;
    }
    return true;
}

// ---- exact replicas of the stock decode arithmetic ------------------------
// The fused kernels below reproduce, value for value, what ggml-cuda computes
// for the nodes they replace at M = 1:
//   - quantize_q8_1: one warp per 32-element chunk, lane = element;
//   - mul_mat_vec_q (NVFP4 x q8_1): the per-lane block assignment and the
//     cross-warp / butterfly reduction order of the kernel variant ggml picks
//     for the row length (K = 512, 1024, 2048), including the additions of the
//     idle warps' zero partials;
//   - the fused mmvq epilogue (scale, zero bias, SwiGLU), the MoE weighted
//     reduction and the chained residual adds.
// `fzero` is a runtime 0.0f so that x + 0 is never folded away (it maps -0 to
// +0 exactly as the stock kernels' zero partials do).

// stock quantize_q8_1 for one chunk: lane holds element `lane` of the chunk
static __device__ __forceinline__ void frt_q8_1_chunk(const float xi, const int lane, block_q8_1 * dst) {
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    dst->qs[lane] = q;
    if (lane == 0) {
        dst->ds = make_half2(d, sum);
    }
}

// One lane's share of an NVFP4 block under the mmvq lane mapping (iqs = 0 or
// 4): the four qs ints and the two sub-block scale bytes vec_dot_nvfp4_q8_1
// reads. Weights are immutable, so kernels fetch these before the PDL
// dependency wait and before their activation prologue.
struct frt_nvb { int q0, q1, q2, q3; uint32_t d; };

static __device__ __forceinline__ frt_nvb frt_ld_nvb(const char * row, const int kb, const int iqs) {
    const char * blk = row + (size_t) kb * sizeof(block_nvfp4);
    const int * qs = (const int *) (blk + offsetof(block_nvfp4, qs)) + iqs;
    frt_nvb r;
    r.q0 = qs[0];
    r.q1 = qs[1];
    r.q2 = qs[2];
    r.q3 = qs[3];
    r.d  = *(const uint16_t *) (blk + offsetof(block_nvfp4, d) + iqs / 2);
    return r;
}

// vec_dot_nvfp4_q8_1 over a preloaded block share; y = q8_1 blocks of this
// NVFP4 block (2 x 32 values). Same operations in the same order.
static __device__ __forceinline__ float frt_vd(const frt_nvb & b, const block_q8_1 * y, const int iqs) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        const int32_t iqs0 = iqs + 2*i;
        const int32_t is = iqs0 >> 1;
        const int2 v0 = get_int_from_table_16(i == 0 ? b.q0 : b.q2, kvalues_mxfp4);
        const int2 v1 = get_int_from_table_16(i == 0 ? b.q1 : b.q3, kvalues_mxfp4);
        const block_q8_1 * bq8 = y + (is >> 1);
        const int32_t i8 = ((is & 1) << 2);

        int sumi = ggml_cuda_dp4a(v0.x, get_int_b4(bq8->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(v0.y, get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(v1.x, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(v1.y, get_int_b4(bq8->qs, i8 + 3), sumi);

        const float d = ggml_cuda_ue4m3_to_fp32((uint8_t) (b.d >> (8*i))) * __low2float(bq8->ds);
        sum += d * float(sumi);
    }
    return sum;
}

static __device__ __forceinline__ int frt_lane_iqs(const int lane) { return 4 * (lane % 2); }

// q8_1 of NCH chunks of `src` per warp (chunks c0, c0 + cstride, ...): all
// loads are issued first, then the chunks are quantized from registers.
template <int NCH>
static __device__ __forceinline__ void frt_q8_1_chunks(const float * src, block_q8_1 * dst, const int c0, const int cstride, const int lane) {
    float v[NCH];
#pragma unroll
    for (int k = 0; k < NCH; ++k) {
        v[k] = src[(c0 + k * cstride) * QK8_1 + lane];
    }
#pragma unroll
    for (int k = 0; k < NCH; ++k) {
        frt_q8_1_chunk(v[k], lane, &dst[c0 + k * cstride]);
    }
}

// mmvq row dot, NVFP4, K = 2048 (32 blocks): ggml launches 4 warps of which
// warps 0/1 cover blocks [0,16) / [16,32) (one block per lane pair), then sums
// warp partials 0+1+2+3 in order and reduces across the warp.
// b[w] = this lane's share of block lane/2 + 16 w.
static __device__ __forceinline__ float frt_rowdot_k2048(const frt_nvb (&b)[2], const block_q8_1 * y, const int lane, const float fzero) {
    const int kb = lane / 2, iqs = frt_lane_iqs(lane);
    const float p0 = frt_vd(b[0], y + 2 * kb,        iqs);
    const float p1 = frt_vd(b[1], y + 2 * (kb + 16), iqs);
    float t = fzero + p0;
    t += fzero + p1;
    t += fzero;
    t += fzero;
    return warp_reduce_sum(t);
}

// K = 1024 (16 blocks): only warp 0 active, one block per lane pair.
static __device__ __forceinline__ float frt_rowdot_k1024(const frt_nvb & b, const block_q8_1 * y, const int lane, const float fzero) {
    const int kb = lane / 2, iqs = frt_lane_iqs(lane);
    float t = fzero + frt_vd(b, y + 2 * kb, iqs);
    t += fzero;
    t += fzero;
    t += fzero;
    return warp_reduce_sum(t);
}

// K = 512 (8 blocks): only lanes 0..15 of warp 0 active. Two independent rows
// per warp: half h = lane/16 handles its own (row, y); the stock butterfly's
// first step (xor 16) adds the idle half's +0, the rest stays within 16 lanes.
// b = this lane's share of block (lane % 16) / 2 of its half's row.
static __device__ __forceinline__ float frt_rowdot_k512_pair(const frt_nvb & b, const block_q8_1 * y, const int lane, const float fzero) {
    const int kb = (lane % 16) / 2, iqs = frt_lane_iqs(lane);
    float t = fzero + frt_vd(b, y + 2 * kb, iqs);
    t += fzero;
    t += fzero;
    t += fzero;
    t += fzero;   // xor-16 partner (idle lane) contributes +0
    return warp_reduce_sum<16>(t);
}

// K = 4096 (64 blocks): 4 warps, one block per lane pair each; the warp
// partials are summed in warp order before the butterfly.
static __device__ __forceinline__ float frt_rowdot_k4096(const frt_nvb (&b)[4], const block_q8_1 * y, const int lane, const float fzero) {
    const int kb = lane / 2, iqs = frt_lane_iqs(lane);
    float p[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        p[w] = fzero + frt_vd(b[w], y + 2 * (kb + 16 * w), iqs);
    }
    float t = p[0];
    t += p[1];
    t += p[2];
    t += p[3];
    return warp_reduce_sum(t);
}

// warp-cooperative L2 prefetch of [p, p + nbytes): no registers held
static __device__ __forceinline__ void frt_l2_prefetch(const char * p, const size_t nbytes, const int lane) {
    const uintptr_t lo = (uintptr_t) p & ~(uintptr_t) 127;
    const uintptr_t hi = (uintptr_t) p + nbytes;
    for (uintptr_t a = lo + (uintptr_t) lane * 128; a < hi; a += 32 * 128) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
    }
}

template <int NB>
static __device__ __forceinline__ void frt_ld_row(frt_nvb (&b)[NB], const char * row, const int lane) {
#pragma unroll
    for (int w = 0; w < NB; ++w) {
        b[w] = frt_ld_nvb(row, lane / 2 + 16 * w, frt_lane_iqs(lane));
    }
}

// Stock rms_norm_f32<512, 1, vec, mul> over one 2048-wide row, reproduced by
// NW warps of one CTA: virtual thread v (0..511) owns float4 v; virtual warp
// partial sums are reduced exactly as block_reduce does. Writes the
// normalized, weighted row to `out` (shared memory).
template <int NW>
static __device__ __forceinline__ void frt_rms_norm_2048(const float * x, const float * w, const float eps,
        float * out, float * red, const int ncols) {
    constexpr int NT = NW * WARP_SIZE;
    static_assert(NT >= 512 ? NT % 512 == 0 : 512 % NT == 0, "virtual block of 512 threads");
    constexpr int VPT = NT >= 512 ? 1 : 512 / NT;   // virtual threads per real thread
    const bool part = NT <= 512 || threadIdx.x < 512;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const float4 * x4 = (const float4 *) x;
    float4 xs[VPT];
#pragma unroll
    for (int k = 0; k < VPT && part; ++k) {
        const int v = threadIdx.x + k * NW * WARP_SIZE;   // virtual thread id
        float4 xv[1];
        float tmp = 0.0f;
        xv[0] = x4[v];
        tmp += xv[0].x*xv[0].x + xv[0].y*xv[0].y + xv[0].z*xv[0].z + xv[0].w*xv[0].w;
        tmp = warp_reduce_sum(tmp);
        if (lane == 0) red[v / WARP_SIZE] = tmp;
        xs[k] = xv[0];
    }
    __syncthreads();
    float tot = lane < 512 / WARP_SIZE ? red[lane] : 0.0f;
    tot = warp_reduce_sum(tot);
    const float scale = rsqrtf(tot / ncols + eps);
    const float4 * w4 = (const float4 *) w;
    float4 * o4 = (float4 *) out;
#pragma unroll
    for (int k = 0; k < VPT && part; ++k) {
        const int v = threadIdx.x + k * NW * WARP_SIZE;
        float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
        const float4 m = w4[v];
        r.x *= m.x;
        r.y *= m.y;
        r.z *= m.z;
        r.w *= m.w;
        o4[v] = r;
    }
    GGML_UNUSED(warp);
}

// ---- attention front (FRT window ATTNQKV) -----------------------------------
// attn RMSNorm -> q8_1 -> Q/K/V GEMV (+ per-tensor scales) -> per-head
// RMSNorm * q/k_norm (+ NEOX RoPE on rope layers) -> Q (f32) for the
// attention, K/V rows (f16) into the cache: 11 stock launches -> 2.

struct frt_rope_params {
    int   n_dims;
    float freq_scale, ext_factor, attn_factor, theta_scale;
    float corr0, corr1;
};

static constexpr int FRT_QKV_ROWS_MAX = 4096 + 2 * 512;

struct frt_qkn_params {
    const float * qn_w; const float * kn_w; float eps;
    float * q_out; half * k_cache; half * v_cache; const int64_t * k_idx; const int64_t * v_idx;
    size_t k_row_bytes, v_row_bytes;
    int hq, hk;
    const int32_t * pos;
    frt_rope_params rp;
    int ncols;
};

// ATTNQKV -> ATTNDEC handoff: when the FLASH_ATTN_EXT node that follows an
// ATTNQKV span (only views in between) is taken by ATTNDEC, the per-head
// norm / rope / cache-write epilogue runs inside the attention kernel instead
// of its own launch. Set by ATTNQKV, consumed by ATTNDEC on the same node.
struct frt_fa_pending {
    const ggml_tensor * fa = nullptr;
    frt_qkn_params qp;
    bool rope;
};
static frt_fa_pending g_fa_pend;
static bool frt_attndec_eligible(const ggml_cgraph * cgraph, int i);

// One head of the attention-front epilogue (K_qkn), one warp. Q heads [0,hq), K heads
// [hq,hq+hk), V heads after. rope=false: stock rms_norm_f32<32,1,vec,mul>
// (one warp per 128-wide row); rope=true: stock rms_norm_mul_rope_f32<256>
// (8 warps, 4 of them loaded), NEOX pairs (i, i + n_dims/2), f32 out; K and V
// rows go to the cache as f16 (set_rows).
// Q or K head: normalized (+ roped) values out[k] for elements oix[k] of the head.
template <bool rope>
static __device__ __forceinline__ void frt_qkn_vals(const float * xc, const float * mulc, const int lane,
        const frt_qkn_params & p, float (&out)[4], int (&oix)[4]) {
    if constexpr (!rope) {
        const float4 * x4 = (const float4 *) xc;
        float4 xs[1];
        float tmp = 0.0f;
        xs[0] = __ldcg(x4 + lane);
        tmp += xs[0].x*xs[0].x + xs[0].y*xs[0].y + xs[0].z*xs[0].z + xs[0].w*xs[0].w;
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(tmp / p.ncols + p.eps);
        float4 r = make_float4(scale*xs[0].x, scale*xs[0].y, scale*xs[0].z, scale*xs[0].w);
        const float4 m = ((const float4 *) mulc)[lane];
        r.x *= m.x;
        r.y *= m.y;
        r.z *= m.z;
        r.w *= m.w;
        out[0] = r.x; out[1] = r.y; out[2] = r.z; out[3] = r.w;
#pragma unroll
        for (int k = 0; k < 4; ++k) oix[k] = 4 * lane + k;
    } else {
        // 256 virtual threads: v < 128 holds element v, virtual warps 4..7 hold
        // 0; block_reduce then sums the 8 warp partials (+ zero sentinels).
        float s4[4];
#pragma unroll
        for (int wv = 0; wv < 4; ++wv) {
            const float xi = __ldcg(xc + wv * WARP_SIZE + lane);
            float tmp = 0.0f;
            tmp += xi * xi;
            s4[wv] = warp_reduce_sum(tmp);
        }
        float tot = 0.0f;
        if (lane == 0) tot = s4[0];
        else if (lane == 1) tot = s4[1];
        else if (lane == 2) tot = s4[2];
        else if (lane == 3) tot = s4[3];
        tot = warp_reduce_sum(tot);
        const float scale = rsqrtf(tot/p.ncols + p.eps);
        // pairs i0 = 2*v for virtual v < 64: lane covers v = lane and v = lane + 32
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int v  = lane + k * WARP_SIZE;
            const int i0 = 2 * v;
            const int ix0 = i0 / 2;
            const int ix1 = i0 / 2 + p.rp.n_dims / 2;
            const float x0 = scale * __ldcg(xc + ix0) * mulc[ix0];
            const float x1 = scale * __ldcg(xc + ix1) * mulc[ix1];
            const float theta_base  = p.pos[0]*powf(p.rp.theta_scale, i0/2.0f);
            const float freq_factor = 1.0f;
            float cos_theta;
            float sin_theta;
            {   // rope_yarn<true>
                const float theta_extrap = theta_base/freq_factor;
                float mscale = p.rp.attn_factor;
                float theta_interp = p.rp.freq_scale * theta_extrap;
                float theta = theta_interp;
                if (p.rp.ext_factor != 0.0f) {
                    const float y = (i0 / 2 - p.rp.corr0) / max(0.001f, p.rp.corr1 - p.rp.corr0);
                    const float ramp_mix = (1.0f - min(1.0f, max(0.0f, y))) * p.rp.ext_factor;
                    theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
                    mscale *= 1.0f + 0.1f * logf(1.0f / p.rp.freq_scale);
                }
                cos_theta = cosf(theta) * mscale;
                sin_theta = sinf(theta) * mscale;
            }
            out[2 * k]     = x0*cos_theta - x1*sin_theta;
            out[2 * k + 1] = x0*sin_theta + x1*cos_theta;
            oix[2 * k]     = ix0;
            oix[2 * k + 1] = ix1;
        }
    }
}

template <bool rope>
static __device__ __forceinline__ void frt_qkn_head(const float * qkv, const int h, const int lane, const frt_qkn_params & p) {
    constexpr int D = 128;
    const bool is_v = h >= p.hq + p.hk;
    const bool is_k = !is_v && h >= p.hq;
    const float * xc = qkv + (size_t) h * D;   // written by other CTAs of this or the previous kernel
    if (is_v) {   // set_rows f32 -> f16 of the scaled V head
        half * dst = (half *) ((char *) p.v_cache + (size_t) p.v_idx[0] * p.v_row_bytes) + (size_t) (h - p.hq - p.hk) * D;
#pragma unroll
        for (int k = 0; k < D / WARP_SIZE; ++k) {
            dst[lane + k * WARP_SIZE] = __float2half(__ldcg(xc + lane + k * WARP_SIZE));
        }
        return;
    }
    float out[4];
    int   oix[4];
    frt_qkn_vals<rope>(xc, is_k ? p.kn_w : p.qn_w, lane, p, out, oix);
    if (is_k) {
        half * dst = (half *) ((char *) p.k_cache + (size_t) p.k_idx[0] * p.k_row_bytes) + (size_t) (h - p.hq) * D;
#pragma unroll
        for (int k = 0; k < 4; ++k) dst[oix[k]] = __float2half(out[k]);
    } else {
        float * dst = p.q_out + (size_t) h * D;
#pragma unroll
        for (int k = 0; k < 4; ++k) dst[oix[k]] = out[k];
    }
}

// K_qkv: one warp per output row; rows [0,nq) Q, [nq,nq+nk) K, then V.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_attn_qkv(
        const float * x, const float * norm_w, const float eps,
        const char * wq, const char * wk, const char * wv, const size_t row_stride,
        const float * sq, const float * sk, const float * sv,
        float * out, const int nq, const int nk, const float fzero, const int ncols) {
    ggml_cuda_pdl_lc();
    __shared__ float xn[FRT_K_EMBD];
    __shared__ float red[32];
    __shared__ block_q8_1 yq[FRT_K_EMBD / QK8_1];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r = blockIdx.x * NW + warp;
    const bool active = r < nq + 2 * nk;
    frt_nvb b[2];
    float s = 0.0f;
    if (active) {   // weights and scales are immutable: fetch before the dependency wait
        const char * w;
        if (r < nq)           { w = wq + (size_t) r * row_stride;               s = sq[0]; }
        else if (r < nq + nk) { w = wk + (size_t) (r - nq) * row_stride;        s = sk[0]; }
        else                  { w = wv + (size_t) (r - nq - nk) * row_stride;   s = sv[0]; }
        frt_ld_row<2>(b, w, lane);
    }
    ggml_cuda_pdl_sync();
    frt_rms_norm_2048<NW>(x, norm_w, eps, xn, red, ncols);
    __syncthreads();
    for (int c = warp; c < FRT_K_EMBD / QK8_1; c += NW) {
        frt_q8_1_chunk(xn[c * QK8_1 + lane], lane, &yq[c]);
    }
    __syncthreads();
    if (!active) {
        return;
    }
    const float t = frt_rowdot_k2048(b, yq, lane, fzero);
    if (lane == 0) {
        float result = t;
        result *= s;
        result += fzero;
        out[r] = result;
    }
}

// K_qkn: one warp per head. Q heads [0,hq), K heads [hq,hq+hk), V heads after.
// rope=false: stock rms_norm_f32<32,1,vec,mul> (one warp per 128-wide row);
// rope=true:  stock rms_norm_mul_rope_f32<256> (8 warps, 4 of them loaded),
// NEOX pairs (i, i + n_dims/2), f32 out; K and V rows go to the cache as f16.
template <bool rope>
static __global__ void frt_attn_qkn(const float * qkv, const frt_qkn_params p) {
    ggml_cuda_pdl_lc();
    const int h = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (h >= p.hq + 2 * p.hk) {
        return;
    }
    ggml_cuda_pdl_sync();
    frt_qkn_head<rope>(qkv, h, lane, p);
}

static float * g_qkv = nullptr;

// matches MUL_MAT(w, x) -> MUL(s) -> RESHAPE [128, h] -> RMS_NORM -> MUL(norm_w) [-> ROPE]
struct frt_head_chain {
    const ggml_tensor * mm = nullptr, * smul = nullptr, * norm = nullptr, * nmul = nullptr, * rope = nullptr;
    const ggml_tensor * last() const { return rope ? rope : nmul; }
};

static bool frt_rope_params_of(const ggml_tensor * rope, frt_rope_params & rp) {
    const int32_t * op = (const int32_t *) rope->op_params;
    const int n_dims = op[1], mode = op[2], n_ctx_orig = op[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   op +  5, sizeof(float));
    memcpy(&freq_scale,  op +  6, sizeof(float));
    memcpy(&ext_factor,  op +  7, sizeof(float));
    memcpy(&attn_factor, op +  8, sizeof(float));
    memcpy(&beta_fast,   op +  9, sizeof(float));
    memcpy(&beta_slow,   op + 10, sizeof(float));
    if (!(mode & GGML_ROPE_TYPE_NEOX) || n_dims != 128 || rope->src[2] != nullptr) return false;
    float corr[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr);
    rp = { n_dims, freq_scale, ext_factor, attn_factor, powf(freq_base, -2.0f/n_dims), corr[0], corr[1] };
    return true;
}

// anchor: RMS_NORM over the residual stream followed by MUL(attn_norm.weight).
static bool frt_attnqkv_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_ATTNQKV");
    if (!on || g_qkv == nullptr) return false;
    const ggml_tensor * rn = cgraph->nodes[i];
    if (rn->op != GGML_OP_RMS_NORM || i + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * an = cgraph->nodes[i + 1];
    if (an->op != GGML_OP_MUL || an->src[0] != rn || !strstr(an->src[1]->name, ".attn_norm.weight")) return false;
    const ggml_tensor * x = rn->src[0];
    if (x->type != GGML_TYPE_F32 || x->ne[0] != FRT_K_EMBD || ggml_nrows(x) != 1 || !ggml_is_contiguous(x)) return false;
    const ggml_tensor * nw = an->src[1];
    if (nw->type != GGML_TYPE_F32 || nw->ne[0] != FRT_K_EMBD || !ggml_is_contiguous(nw)) return false;
    if (!frt_aligned16(x->data) || !frt_aligned16(nw->data)) return false;
    float eps; memcpy(&eps, rn->op_params, sizeof(float));

    frt_head_chain q, k;
    const ggml_tensor * vmm = nullptr, * vmul = nullptr;
    ggml_tensor * k_set = nullptr, * v_set = nullptr;
    int end = -1;
    frt_head_chain * cur = nullptr;
    const int LIM = std::min(cgraph->n_nodes, i + 32);
    for (int j = i + 2; j < LIM && end < 0; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        switch (n->op) {
            case GGML_OP_RESHAPE: case GGML_OP_VIEW:
                break;
            case GGML_OP_MUL_MAT:
                if (n->src[1] != an) return false;
                if (strstr(n->src[0]->name, ".attn_q.weight") && !q.mm) { q.mm = n; cur = &q; }
                else if (strstr(n->src[0]->name, ".attn_k.weight") && !k.mm) { k.mm = n; cur = &k; }
                else if (strstr(n->src[0]->name, ".attn_v.weight") && !vmm) { vmm = n; cur = nullptr; }
                else return false;
                break;
            case GGML_OP_MUL:
                if (vmm && !vmul && n->src[0] == vmm) { vmul = n; break; }
                if (!cur) return false;
                if (!cur->smul && n->src[0] == cur->mm) { cur->smul = n; break; }
                if (cur->norm && !cur->nmul && n->src[0] == cur->norm) { cur->nmul = n; break; }
                return false;
            case GGML_OP_RMS_NORM:
                if (!cur || !cur->smul || cur->norm || n->src[0]->op != GGML_OP_RESHAPE || n->src[0]->src[0] != cur->smul) return false;
                cur->norm = n;
                break;
            case GGML_OP_ROPE:
                if (!cur || !cur->nmul || cur->rope || n->src[0] != cur->nmul) return false;
                cur->rope = n;
                break;
            case GGML_OP_SET_ROWS:
                if (!k_set) k_set = n;
                else { v_set = n; end = j; }
                break;
            default:
                return false;
        }
    }
    if (end < 0 || !q.mm || !q.smul || !q.norm || !q.nmul || !k.mm || !k.smul || !k.norm || !k.nmul || !vmm || !vmul) return false;
    const bool rope = q.rope != nullptr;
    if ((k.rope != nullptr) != rope) return false;
    // weights / scales
    const ggml_tensor * wq = q.mm->src[0], * wk = k.mm->src[0], * wv = vmm->src[0];
    for (const ggml_tensor * w : { wq, wk, wv }) {
        if (w->type != GGML_TYPE_NVFP4 || w->ne[0] != FRT_K_EMBD || w->nb[1] != wq->nb[1] || !ggml_is_contiguous(w)) return false;
    }
    const int nq = (int) wq->ne[1], nk = (int) wk->ne[1];
    if (nq % 128 != 0 || nk % 128 != 0 || wv->ne[1] != nk || nq + 2 * nk > FRT_QKV_ROWS_MAX) return false;
    const ggml_tensor * sq = frt_mm_scale_src(q.smul, q.mm), * sk = frt_mm_scale_src(k.smul, k.mm), * sv = frt_mm_scale_src(vmul, vmm);
    if (!sq || !sk || !sv) return false;
    // head norms: [128, h] rows, eps shared, f32 weights [128]
    float qeps, keps;
    memcpy(&qeps, q.norm->op_params, sizeof(float));
    memcpy(&keps, k.norm->op_params, sizeof(float));
    if (qeps != keps || q.norm->ne[0] != 128 || k.norm->ne[0] != 128) return false;
    const ggml_tensor * qnw = q.nmul->src[1], * knw = k.nmul->src[1];
    if (qnw->ne[0] != 128 || knw->ne[0] != 128 || ggml_nelements(qnw) != 128 || ggml_nelements(knw) != 128) return false;
    if (!frt_aligned16(qnw->data) || !frt_aligned16(knw->data)) return false;
    frt_rope_params rp = {};
    const int32_t * pos = nullptr;
    if (rope) {
        frt_rope_params rk;
        if (!frt_rope_params_of(q.rope, rp) || !frt_rope_params_of(k.rope, rk)) return false;
        if (memcmp(&rp, &rk, sizeof(rp)) != 0 || q.rope->src[1] != k.rope->src[1]) return false;
        if (q.rope->src[1]->type != GGML_TYPE_I32 || q.rope->src[1]->ne[0] != 1) return false;
        pos = (const int32_t *) q.rope->src[1]->data;
    }
    const ggml_tensor * qfin = q.last();
    const ggml_tensor * kfin = k.last();
    if (qfin->type != GGML_TYPE_F32 || !ggml_is_contiguous(qfin) || qfin->ne[0] != 128 || qfin->ne[1] * 128 != nq) return false;
    // cache writes: SET_ROWS(VIEW(head block of this token), idx) into f16 rows of nk elements
    auto set_ok = [&](const ggml_tensor * s, const ggml_tensor * src) {
        const ggml_tensor * v = s->src[0];
        if (v->op != GGML_OP_VIEW || v->view_offs != 0 || v->ne[0] != nk || ggml_nrows(v) != 1) return false;
        const ggml_tensor * vs = v->src[0];
        if (!(vs == src || (vs->op == GGML_OP_RESHAPE && vs->src[0] == src))) return false;
        return s->type == GGML_TYPE_F16 && s->ne[0] == nk && s->nb[0] == sizeof(half) &&
               s->src[1]->type == GGML_TYPE_I64 && ggml_nelements(s->src[1]) == 1;
    };
    if (!set_ok(k_set, kfin) || !set_ok(v_set, vmul)) return false;
    // everything inside the span except Q's final tensor and the two cache writes is internal
    if (!frt_span_closed(cgraph, i, end, { qfin, k_set })) return false;

    cudaStream_t stream = ctx.stream();
    static const int nw_qkv = frt_cfg_nw("FRT_NW_QKV", 16);   // tile search: 16 > 8 > 4
    const int nrows = nq + 2 * nk;
    const int hq = nq / 128, hk = nk / 128, nh = hq + 2 * hk;
    const frt_qkn_params qp = { (const float *) qnw->data, (const float *) knw->data, qeps,
        (float *) qfin->data, (half *) k_set->data, (half *) v_set->data,
        (const int64_t *) k_set->src[1]->data, (const int64_t *) v_set->src[1]->data,
        (size_t) k_set->nb[1], (size_t) v_set->nb[1], hq, hk, pos, rp, 128 };
    FRT_NW_DISPATCH32(nw_qkv,
        ggml_cuda_kernel_launch(frt_attn_qkv<NWT>, ggml_cuda_kernel_launch_params((dim3) ((nrows + NWT - 1) / NWT), (dim3) (NWT * WARP_SIZE), 0, stream),
            (const float *) x->data, (const float *) nw->data, eps,
            (const char *) wq->data, (const char *) wk->data, (const char *) wv->data, (size_t) wq->nb[1],
            (const float *) sq->data, (const float *) sk->data, (const float *) sv->data,
            g_qkv, nq, nk, 0.0f, FRT_K_EMBD));
    // the attention node, reached through views only: if ATTNDEC takes it, the
    // epilogue below runs inside the attention kernel (frt_fa_pending)
    static const bool defer_ok = !(getenv("FRT_ATTNDEC_QKN") && getenv("FRT_ATTNDEC_QKN")[0] == '0');
    bool defer = false;
    if (defer_ok) {
        int j = end + 1;
        while (j < cgraph->n_nodes && (cgraph->nodes[j]->op == GGML_OP_VIEW || cgraph->nodes[j]->op == GGML_OP_PERMUTE ||
                                       cgraph->nodes[j]->op == GGML_OP_RESHAPE || cgraph->nodes[j]->op == GGML_OP_TRANSPOSE)) ++j;
        if (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_FLASH_ATTN_EXT) {
            const ggml_tensor * fa = cgraph->nodes[j];
            const ggml_tensor * fq = fa->src[0], * fk = fa->src[1], * fv = fa->src[2];
            const ggml_tensor * kc = k_set->view_src, * vc = v_set->view_src;
            defer = (fq == qfin || fq->view_src == qfin) && fq->data == qfin->data &&
                    kc && vc && fk->view_src == kc && fv->view_src == vc && fk->data == kc->data && fv->data == vc->data &&
                    k_set->data == kc->data && v_set->data == vc->data &&
                    (size_t) fk->nb[1] == (size_t) k_set->nb[1] && (size_t) fv->nb[1] == (size_t) v_set->nb[1] &&
                    fk->nb[2] == 128 * sizeof(half) && fv->nb[2] == 128 * sizeof(half) &&
                    fq->nb[2] == 128 * sizeof(float) && frt_attndec_eligible(cgraph, j);
            if (defer) {
                GGML_ASSERT(g_fa_pend.fa == nullptr);
                g_fa_pend.fa = fa;
                g_fa_pend.qp = qp;
                g_fa_pend.rope = rope;
            }
        }
    }
    if (!defer) {
        constexpr int HW = 4;   // heads (warps) per CTA
        const ggml_cuda_kernel_launch_params lp2((dim3) ((nh + HW - 1) / HW), (dim3) (HW * WARP_SIZE), 0, stream);
        ggml_cuda_kernel_launch(rope ? frt_attn_qkn<true> : frt_attn_qkn<false>, lp2, (const float *) g_qkv, qp);
    }
    if (frt_stats_on()) {
        static long fires = 0;
        if ((++fires & 0xFFF) == 1) fprintf(stderr, "frt-stats attnqkv fires=%ld span=%d rope=%d\n", fires, end - i + 1, (int) rope);
    }
    *skip_count = end - i + 1;
    return true;
}

// ---- attention output projection (FRT window OPROJ) -----------------------
// quantize(kqv_out) -> MUL_MAT(attn_output) -> MUL(scale) -> ADD(residual):
// 3 launches -> 1, one warp per output row.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_oproj(
        const float * a, const char * w, const size_t row_stride, const float * s,
        const float * resid, float * out, const int n_out, const float fzero) {
    ggml_cuda_pdl_lc();
    __shared__ block_q8_1 yq[4096 / QK8_1];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r = blockIdx.x * NW + warp;
    const bool active = r < n_out;
    frt_nvb b[4];
    float sc = 0.0f;
    if (active) {
        frt_ld_row<4>(b, w + (size_t) r * row_stride, lane);
        sc = s[0];
    }
    ggml_cuda_pdl_sync();
    static_assert((4096 / QK8_1) % NW == 0, "chunks per warp");
    frt_q8_1_chunks<4096 / QK8_1 / NW>(a, yq, warp, NW, lane);
    __syncthreads();
    if (!active) {
        return;
    }
    const float t = frt_rowdot_k4096(b, yq, lane, fzero);
    if (lane == 0) {
        float result = t;
        result *= sc;
        result += fzero;
        float o = result;   // k_bin_bcast add
        o = o + resid[r];
        out[r] = o;
    }
}

static bool frt_oproj_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_OPROJ");
    if (!on || i + 2 >= cgraph->n_nodes) return false;
    const ggml_tensor * mm = cgraph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !strstr(mm->src[0]->name, ".attn_output.weight")) return false;
    const ggml_tensor * w = mm->src[0], * a = mm->src[1];
    if (w->type != GGML_TYPE_NVFP4 || w->ne[0] != 4096 || !ggml_is_contiguous(w)) return false;
    if (a->type != GGML_TYPE_F32 || a->ne[0] != 4096 || ggml_nrows(a) != 1 || !ggml_is_contiguous(a)) return false;
    const ggml_tensor * smul = cgraph->nodes[i + 1];
    const ggml_tensor * s = frt_mm_scale_src(smul, mm);
    if (!s) return false;
    // last layer: GET_ROWS(proj, out_ids) and GET_ROWS(residual, out_ids) feed the add.
    // With one token both select row 0 of a 1-row tensor, i.e. plain copies.
    int nsp = 3;
    ggml_tensor * add = cgraph->nodes[i + 2];
    const ggml_tensor * resid = nullptr;
    if (add->op == GGML_OP_GET_ROWS) {
        if (i + 4 >= cgraph->n_nodes) return false;
        const ggml_tensor * g0 = cgraph->nodes[i + 2], * g1 = cgraph->nodes[i + 3];
        add = cgraph->nodes[i + 4];
        if (g0->src[0] != smul || g1->op != GGML_OP_GET_ROWS || g0->src[1] != g1->src[1] || add->op != GGML_OP_ADD ||
            add->src[0] != g0 || add->src[1] != g1) return false;
        const ggml_tensor * ids = g0->src[1];
        if (ids->type != GGML_TYPE_I32 || ggml_nelements(ids) != 1 || g0->type != GGML_TYPE_F32 || g1->type != GGML_TYPE_F32 ||
            ggml_nrows(g1->src[0]) != 1 || g1->src[0]->type != GGML_TYPE_F32 || ggml_nrows(g0) != 1 || ggml_nrows(g1) != 1) return false;
        if (!frt_node_ok(cgraph, i + 2, GGML_OP_GET_ROWS, 1) || !frt_node_ok(cgraph, i + 3, GGML_OP_GET_ROWS, 1)) return false;
        resid = g1->src[0];
        nsp = 5;
    } else {
        if (add->op != GGML_OP_ADD || add->src[0] != smul) return false;
        resid = add->src[1];
    }
    const int n_out = (int) w->ne[1];
    if (resid->type != GGML_TYPE_F32 || !ggml_is_contiguous(resid) || resid->ne[0] != n_out || ggml_nrows(resid) != 1) return false;
    if (add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || add->ne[0] != n_out || ggml_nrows(add) != 1) return false;
    if (!frt_node_ok(cgraph, i, GGML_OP_MUL_MAT, 1) || !frt_node_ok(cgraph, i + 1, GGML_OP_MUL, 1) || !frt_node_ok(cgraph, i + nsp - 1, GGML_OP_ADD, 0)) return false;
    // every CTA reads all of `a`; thread r reads resid[r] before writing out[r]
    if (frt_overlap(add, a) || (add->data != resid->data && frt_overlap(add, resid))) return false;
    static const int nw_op = frt_cfg_nw("FRT_NW_OPROJ", 16);   // tile search: 16 > 8 > 4
    FRT_NW_DISPATCH(nw_op,
        ggml_cuda_kernel_launch(frt_oproj<NWT>, ggml_cuda_kernel_launch_params((dim3) ((n_out + NWT - 1) / NWT), (dim3) (NWT * WARP_SIZE), 0, ctx.stream()),
            (const float *) a->data, (const char *) w->data, (size_t) w->nb[1], (const float *) s->data,
            (const float *) resid->data, (float *) add->data, n_out, 0.0f));
    *skip_count = nsp;
    return true;
}

// ---- dense FFN (FRT window DENSEFFN, the leading dense layers) ----------------
// ffn RMSNorm -> q8_1 -> gate / up GEMV (+ scales) -> SwiGLU -> q8_1 -> down
// GEMV (+ scale) -> residual add: 6 stock launches (rms_norm, quantize, the
// fused gate/up/GLU mmvq, quantize, mmvq, add) -> 2.

static constexpr int FRT_DFF = 6144;   // pinned: dense FFN width
static float * g_dense_glu = nullptr;

// mmvq row dot, K = 6144 (96 blocks): ggml's 4 warps start at blocks lane/2 +
// 16 w and step by 64, so warps 0/1 cover two blocks per lane pair and warps
// 2/3 one; the warp partials are summed in warp order before the butterfly.
// b[j] = this lane's share of block lane/2 + 16 j (j = 0..5).
static __device__ __forceinline__ float frt_rowdot_k6144(const frt_nvb (&b)[6], const block_q8_1 * y, const int lane, const float fzero) {
    const int kb = lane / 2, iqs = frt_lane_iqs(lane);
    float t0 = fzero + frt_vd(b[0], y + 2 * kb, iqs);
    t0 += frt_vd(b[4], y + 2 * (kb + 64), iqs);
    float t1 = fzero + frt_vd(b[1], y + 2 * (kb + 16), iqs);
    t1 += frt_vd(b[5], y + 2 * (kb + 80), iqs);
    const float t2 = fzero + frt_vd(b[2], y + 2 * (kb + 32), iqs);
    const float t3 = fzero + frt_vd(b[3], y + 2 * (kb + 48), iqs);
    float t = t0;
    t += t1;
    t += t2;
    t += t3;
    return warp_reduce_sum(t);
}

// K1: one warp per GLU output row; every CTA recomputes the (exact) RMSNorm
// and q8_1 of the activation, as the attention front does.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_dense_gu(
        const float * x, const float * norm_w, const float eps,
        const char * gw, const char * uw, const size_t row_stride, const float * gs, const float * us,
        float * glu, const int n_ff, const float fzero) {
    ggml_cuda_pdl_lc();
    __shared__ float xn[FRT_K_EMBD];
    __shared__ float red[32];
    __shared__ block_q8_1 yq[FRT_K_EMBD / QK8_1];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r = blockIdx.x * NW + warp;
    const bool active = r < n_ff;
    frt_nvb bg[2], bu[2];
    float sg = 0.0f, su = 0.0f;
    if (active) {   // weights and scales are immutable: fetch before the dependency wait
        frt_ld_row<2>(bg, gw + (size_t) r * row_stride, lane);
        frt_ld_row<2>(bu, uw + (size_t) r * row_stride, lane);
        sg = gs[0];
        su = us[0];
    }
    ggml_cuda_pdl_sync();
    frt_rms_norm_2048<NW>(x, norm_w, eps, xn, red, FRT_K_EMBD);
    __syncthreads();
    for (int c = warp; c < FRT_K_EMBD / QK8_1; c += NW) {
        frt_q8_1_chunk(xn[c * QK8_1 + lane], lane, &yq[c]);
    }
    __syncthreads();
    if (!active) {
        return;
    }
    const float tg = frt_rowdot_k2048(bg, yq, lane, fzero);
    const float tu = frt_rowdot_k2048(bu, yq, lane, fzero);
    if (lane == 0) {   // mmvq fused epilogue: x = up, gate = gate, SwiGLU
        float result = tu;
        result *= su;
        result += fzero;
        float gate_value = tg;
        gate_value *= sg;
        gate_value += fzero;
        result *= ggml_cuda_op_silu_single(gate_value);
        glu[r] = result;
    }
}

// K2: one warp per output row; every CTA quantizes the GLU row to q8_1 (as
// the output projection does for its input), then down GEMV, scale, residual.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_dense_down(
        const float * glu, const char * dw, const size_t row_stride, const float * ds,
        const float * resid, float * out, const int n_out, const float fzero) {
    ggml_cuda_pdl_lc();
    __shared__ block_q8_1 yq[FRT_DFF / QK8_1];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r = blockIdx.x * NW + warp;
    const bool active = r < n_out;
    frt_nvb b[6];
    float sc = 0.0f;
    if (active) {
        frt_ld_row<6>(b, dw + (size_t) r * row_stride, lane);
        sc = ds[0];
    }
    ggml_cuda_pdl_sync();
    static_assert((FRT_DFF / QK8_1) % NW == 0, "chunks per warp");
    frt_q8_1_chunks<FRT_DFF / QK8_1 / NW>(glu, yq, warp, NW, lane);
    __syncthreads();
    if (!active) {
        return;
    }
    const float t = frt_rowdot_k6144(b, yq, lane, fzero);
    if (lane == 0) {
        float result = t;   // mmvq epilogue: scale, zero bias
        result *= sc;
        result += fzero;
        float o = result;   // k_bin_bcast add
        o = o + resid[r];
        out[r] = o;
    }
}

// anchor: RMS_NORM -> MUL(ffn_norm) -> MUL_MAT(ffn_gate) -> MUL(s) -> MUL_MAT(ffn_up)
// -> MUL(s) -> GLU(swiglu) -> MUL_MAT(ffn_down) -> MUL(s) -> ADD(residual), M = 1.
static bool frt_denseffn_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_DENSEFFN");
    if (!on || g_dense_glu == nullptr || i + 9 >= cgraph->n_nodes) return false;
    ggml_tensor * const * n = cgraph->nodes + i;
    const ggml_tensor * rn = n[0], * nm = n[1], * gm = n[2], * gsm = n[3], * um = n[4], * usm = n[5];
    const ggml_tensor * glu = n[6], * dm = n[7], * dsm = n[8];
    ggml_tensor * add = n[9];
    if (rn->op != GGML_OP_RMS_NORM || nm->op != GGML_OP_MUL || nm->src[0] != rn || !strstr(nm->src[1]->name, ".ffn_norm.weight")) return false;
    if (gm->op != GGML_OP_MUL_MAT || gm->src[1] != nm || !strstr(gm->src[0]->name, ".ffn_gate.weight")) return false;
    if (um->op != GGML_OP_MUL_MAT || um->src[1] != nm || !strstr(um->src[0]->name, ".ffn_up.weight")) return false;
    const ggml_tensor * gs = frt_mm_scale_src(gsm, gm), * us = frt_mm_scale_src(usm, um);
    if (!gs || !us) return false;
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
        glu->src[0] != gsm || glu->src[1] != usm) return false;
    if (dm->op != GGML_OP_MUL_MAT || dm->src[1] != glu || !strstr(dm->src[0]->name, ".ffn_down.weight")) return false;
    const ggml_tensor * ds = frt_mm_scale_src(dsm, dm);
    if (!ds || add->op != GGML_OP_ADD || add->src[0] != dsm) return false;
    const ggml_tensor * x = rn->src[0], * nw = nm->src[1], * resid = add->src[1];
    const ggml_tensor * gw = gm->src[0], * uw = um->src[0], * dw = dm->src[0];
    if (x->type != GGML_TYPE_F32 || x->ne[0] != FRT_K_EMBD || ggml_nrows(x) != 1 || !ggml_is_contiguous(x) || !frt_aligned16(x->data)) return false;
    if (nw->type != GGML_TYPE_F32 || nw->ne[0] != FRT_K_EMBD || ggml_nrows(nw) != 1 || !ggml_is_contiguous(nw) || !frt_aligned16(nw->data)) return false;
    if (nm->ne[0] != FRT_K_EMBD || ggml_nrows(nm) != 1) return false;
    for (const ggml_tensor * w : { gw, uw }) {
        if (w->type != GGML_TYPE_NVFP4 || w->ne[0] != FRT_K_EMBD || w->ne[1] != FRT_DFF || !ggml_is_contiguous(w) || w->nb[1] != gw->nb[1]) return false;
    }
    if (dw->type != GGML_TYPE_NVFP4 || dw->ne[0] != FRT_DFF || dw->ne[1] != FRT_K_EMBD || !ggml_is_contiguous(dw)) return false;
    if (resid->type != GGML_TYPE_F32 || resid->ne[0] != FRT_K_EMBD || ggml_nrows(resid) != 1 || !ggml_is_contiguous(resid)) return false;
    if (add->type != GGML_TYPE_F32 || add->ne[0] != FRT_K_EMBD || ggml_nrows(add) != 1 || !ggml_is_contiguous(add)) return false;
    // K2 thread r reads resid[r] before writing out[r]: an exact alias is safe, a partial one is not
    if (add->data != resid->data && frt_overlap(add, resid)) return false;
    if (!frt_span_closed(cgraph, i, i + 9, {})) return false;
    float eps;
    memcpy(&eps, rn->op_params, sizeof(float));
    cudaStream_t stream = ctx.stream();
    static const int nw_gu = frt_cfg_nw("FRT_NW_DGU", 16);
    static const int nw_dn = frt_cfg_nw("FRT_NW_DDOWN", 16);
    FRT_NW_DISPATCH(nw_gu,
        ggml_cuda_kernel_launch(frt_dense_gu<NWT>, ggml_cuda_kernel_launch_params((dim3) ((FRT_DFF + NWT - 1) / NWT), (dim3) (NWT * WARP_SIZE), 0, stream),
            (const float *) x->data, (const float *) nw->data, eps, (const char *) gw->data, (const char *) uw->data, (size_t) gw->nb[1],
            (const float *) gs->data, (const float *) us->data, g_dense_glu, FRT_DFF, 0.0f));
    FRT_NW_DISPATCH(nw_dn,
        ggml_cuda_kernel_launch(frt_dense_down<NWT>, ggml_cuda_kernel_launch_params((dim3) ((FRT_K_EMBD + NWT - 1) / NWT), (dim3) (NWT * WARP_SIZE), 0, stream),
            (const float *) g_dense_glu, (const char *) dw->data, (size_t) dw->nb[1], (const float *) ds->data,
            (const float *) resid->data, (float *) add->data, FRT_K_EMBD, 0.0f));
    *skip_count = 10;
    return true;
}

// ---- output head (FRT window LMHEAD) ------------------------------------------
// final RMSNorm * output_norm -> q8_1 -> output GEMV (+ scale), M = 1:
// rms_norm, quantize, mmvq -> 2 launches (the normalized row is a graph
// output and is still written).

static block_q8_1 * g_lm_q8 = nullptr;

static __global__ void __launch_bounds__(512, 1) frt_lm_norm(
        const float * x, const float * norm_w, const float eps, float * xn_out, block_q8_1 * q8_out) {
    ggml_cuda_pdl_lc();
    constexpr int NW = 16;
    __shared__ float xn[FRT_K_EMBD];
    __shared__ float red[32];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    ggml_cuda_pdl_sync();
    frt_rms_norm_2048<NW>(x, norm_w, eps, xn, red, FRT_K_EMBD);
    __syncthreads();
    ((float4 *) xn_out)[threadIdx.x] = ((const float4 *) xn)[threadIdx.x];
    for (int c = warp; c < FRT_K_EMBD / QK8_1; c += NW) {
        frt_q8_1_chunk(xn[c * QK8_1 + lane], lane, &q8_out[c]);
    }
}

// one warp per vocabulary row (stock small-K mmvq: same per-row block
// assignment and partial order as frt_rowdot_k2048), scale epilogue
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_lm_head(
        const block_q8_1 * q8, const char * w, const size_t row_stride, const float * s,
        float * out, const int n_rows, const float fzero) {
    ggml_cuda_pdl_lc();
    __shared__ block_q8_1 yq[FRT_K_EMBD / QK8_1];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r = blockIdx.x * NW + warp;
    const bool active = r < n_rows;
    frt_nvb b[2];
    float sc = 0.0f;
    if (active) {
        frt_ld_row<2>(b, w + (size_t) r * row_stride, lane);
        sc = s[0];
    }
    ggml_cuda_pdl_sync();
    static_assert(sizeof(yq) % 16 == 0, "16-byte staging");
    for (int k = threadIdx.x; k < (int) (sizeof(yq) / 16); k += NW * WARP_SIZE) {
        ((int4 *) yq)[k] = __ldcg((const int4 *) q8 + k);
    }
    __syncthreads();
    if (!active) {
        return;
    }
    const float t = frt_rowdot_k2048(b, yq, lane, fzero);
    if (lane == 0) {
        float result = t;   // mmvq epilogue: scale, zero bias
        result *= sc;
        result += fzero;
        out[r] = result;
    }
}

// anchor: RMS_NORM -> MUL(output_norm) -> MUL_MAT(output) -> MUL(output.scale)
static bool frt_lmhead_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_LMHEAD");
    if (!on || g_lm_q8 == nullptr || i + 3 >= cgraph->n_nodes) return false;
    const ggml_tensor * rn = cgraph->nodes[i], * nm = cgraph->nodes[i + 1], * mm = cgraph->nodes[i + 2];
    ggml_tensor * smul = cgraph->nodes[i + 3];
    if (rn->op != GGML_OP_RMS_NORM || nm->op != GGML_OP_MUL || nm->src[0] != rn || strcmp(nm->src[1]->name, "output_norm.weight") != 0) return false;
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != nm || strcmp(mm->src[0]->name, "output.weight") != 0) return false;
    const ggml_tensor * s = frt_mm_scale_src(smul, mm);
    if (!s) return false;
    const ggml_tensor * x = rn->src[0], * nw = nm->src[1], * w = mm->src[0];
    if (x->type != GGML_TYPE_F32 || x->ne[0] != FRT_K_EMBD || ggml_nrows(x) != 1 || !ggml_is_contiguous(x) || !frt_aligned16(x->data)) return false;
    if (nw->type != GGML_TYPE_F32 || ggml_nelements(nw) != FRT_K_EMBD || !ggml_is_contiguous(nw) || !frt_aligned16(nw->data)) return false;
    if (nm->type != GGML_TYPE_F32 || ggml_nrows(nm) != 1 || !ggml_is_contiguous(nm) || !frt_aligned16(nm->data)) return false;
    if (w->type != GGML_TYPE_NVFP4 || w->ne[0] != FRT_K_EMBD || !ggml_is_contiguous(w)) return false;
    if (smul->type != GGML_TYPE_F32 || ggml_nrows(smul) != 1 || !ggml_is_contiguous(smul) || smul->ne[0] != w->ne[1]) return false;
    // the normalized row is a graph output (kept), the raw norm and the unscaled product are internal
    if (!frt_span_closed(cgraph, i, i + 3, { nm })) return false;
    // nm may alias x: the single norm CTA reads all of x (into registers) before its first store
    float eps;
    memcpy(&eps, rn->op_params, sizeof(float));
    cudaStream_t stream = ctx.stream();
    ggml_cuda_kernel_launch(frt_lm_norm, ggml_cuda_kernel_launch_params((dim3) 1, (dim3) 512, 0, stream),
        (const float *) x->data, (const float *) nw->data, eps, (float *) nm->data, g_lm_q8);
    const int n_rows = (int) w->ne[1];
    static const int nw_lm = frt_cfg_nw("FRT_NW_LMHEAD", 16);
    FRT_NW_DISPATCH(nw_lm,
        ggml_cuda_kernel_launch(frt_lm_head<NWT>, ggml_cuda_kernel_launch_params((dim3) ((n_rows + NWT - 1) / NWT), (dim3) (NWT * WARP_SIZE), 0, stream),
            (const block_q8_1 *) g_lm_q8, (const char *) w->data, (size_t) w->nb[1], (const float *) s->data,
            (float *) smul->data, n_rows, 0.0f));
    *skip_count = 4;
    return true;
}

// ---- MoE routing (FRT window ROUTE) -----------------------------------------
// ffn RMSNorm -> router GEMV (BF16) -> sigmoid top-k with weight
// normalization: 3 stock launches -> 1. One CTA per router row as in
// mul_mat_vec_f<bf16, 256>, each recomputing the (exact) RMSNorm; the last CTA
// to finish runs the topk_moe selection on the staged logits. CTA 0 also
// writes the normalized activation that the expert GEMVs consume.
// Per-layer MoE state shared by the expert kernels (private buffers, never
// aliased by the graph allocator).
struct frt_moe_tail {
    block_q8_1 * glu_q8;     // n_used * n_ff/32 routed blocks, then n_ff_sh/32 shared
    float      * esc;        // activation rescale e per routed slot
    int32_t    * ids;        // snapshot of the routing ids / weights: the layer output written
    float      * wts;        // by K2 may be allocated over the (dead after K2) graph tensors
};
static frt_moe_tail g_moe_tail = {};
static float * g_route_logits = nullptr;
static unsigned int * g_route_count = nullptr;
static block_q8_1 * g_act_q8 = nullptr;          // q8_1 of the ffn-normed activation (64 blocks)
static const void * g_act_q8_src = nullptr;      // tensor data the last matched ROUTE published for

// topk_moe_cuda<256, false> for one row, one warp: sigmoid, iterative top-k
// (lower index wins ties), clamped-sum weight normalization. Outputs go to
// two destinations (graph tensor and private snapshot); logits are read
// through L2 (written by other CTAs).
static __device__ __forceinline__ void frt_topk256_warp(const float * logits, const int n_expert_used,
        const float clamp_val, const float scale_val, const int lane,
        int32_t * ids_out, float * w_out, int32_t * ids_snap, float * w_snap) {
    constexpr int n_experts = 256;
    constexpr int experts_per_thread = n_experts / WARP_SIZE;
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        wt[i / WARP_SIZE] = __ldcg(logits + i + lane);
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = 1.f / (1.f + expf(-wt[i]));
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }
    float wt_sum = 0.f;
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if (wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids_out[k] = max_expert;
            ids_snap[k] = max_expert;
            wt_sum += max_val;
        }
    }
    wt_sum              = warp_reduce_sum(wt_sum);
    wt_sum              = max(wt_sum, clamp_val);
    const float inv_sum = 1.0f / wt_sum;
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] *= inv_sum;
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            w_out[idx] = output_weights[i] * scale_val;
            w_snap[idx] = output_weights[i] * scale_val;
        }
    }
}

template <int NW>   // NW * 32 = 256 x (router rows per CTA)
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_route(
        const float * x, const float * norm_w, const float eps, const int ncols, float * xn_out,
        const nv_bfloat16 * wr, const int64_t wr_stride, float * logits, unsigned int * count,
        int32_t * ids_out, float * w_out, const int n_expert_used, const float clamp_val, const float scale_val,
        block_q8_1 * xq_out, int32_t * ids_snap, float * w_snap) {
    constexpr int RPC = NW * WARP_SIZE / 256;   // router rows per CTA, one 256-thread group each
    static_assert(RPC * 256 == NW * WARP_SIZE, "mul_mat_vec_f block size");
    constexpr int n_experts = 256;
    ggml_cuda_pdl_lc();
    __shared__ float xn[FRT_K_EMBD];
    __shared__ float red[32];
    __shared__ float buf_iw[RPC][WARP_SIZE];
    __shared__ bool last;
    const int tid = threadIdx.x, warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const int grp = tid / 256, t256 = tid % 256;
    const int row = blockIdx.x * RPC + grp;
    // router row (immutable): fetch before the dependency wait
    const nv_bfloat162 * x2 = (const nv_bfloat162 *) (wr + (int64_t) row * wr_stride);
    nv_bfloat162 wv[FRT_K_EMBD / 2 / 256];
#pragma unroll
    for (int k = 0; k < FRT_K_EMBD / 2 / 256; ++k) {
        wv[k] = x2[t256 + 256 * k];
    }
    if (t256 < WARP_SIZE) {
        buf_iw[grp][t256] = 0.0f;
    }
    ggml_cuda_pdl_sync();
    frt_rms_norm_2048<NW>(x, norm_w, eps, xn, red, ncols);
    __syncthreads();
    if (blockIdx.x == 0) {
        for (int c = tid; c < FRT_K_EMBD / 4; c += NW * WARP_SIZE) {
            ((float4 *) xn_out)[c] = ((const float4 *) xn)[c];
        }
    }
    if (blockIdx.x < FRT_K_EMBD / QK8_1 / NW) {   // q8_1 of the activation for the expert GEMVs, one chunk per warp
        const int c = blockIdx.x * NW + warp;
        frt_q8_1_chunk(xn[c * QK8_1 + lane], lane, &xq_out[c]);
    }
    // mul_mat_vec_f<nv_bfloat16, float, 1, 256>, one group per row
    const float2 * y2 = (const float2 *) xn;
    float sumf = 0.0f;
#pragma unroll
    for (int k = 0; k < FRT_K_EMBD / 2 / 256; ++k) {
        const int col2 = t256 + 256 * k;
        const nv_bfloat162 tmpx = wv[k];
        const float2 tmpy = y2[col2];
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    buf_iw[grp][t256 / WARP_SIZE] = sumf;
    __syncthreads();
    if (t256 < WARP_SIZE) {
        sumf = buf_iw[grp][t256];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
        if (t256 == 0) {
            logits[row] = sumf;
            __threadfence();
        }
    }
    __syncthreads();
    if (tid == 0) {
        last = atomicAdd(count, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || warp != 0) {
        return;
    }
    __threadfence();
    frt_topk256_warp(logits, n_expert_used, clamp_val, scale_val, lane, ids_out, w_out, ids_snap, w_snap);
    if (lane == 0) {
        *count = 0;   // ready for the next layer
    }
}

struct frt_route_m {
    const ggml_tensor * x, * nw, * wr;
    ggml_tensor * fn, * as, * dv;
    float eps, clamp_val;
    int n_used;
};

// RMS_NORM + MUL(ffn_norm.weight) + router span (12 nodes from i).
static bool frt_route_match(ggml_cgraph * cgraph, int i, frt_route_m & m) {
    if (g_route_logits == nullptr || i + 11 >= cgraph->n_nodes) return false;
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * rn = nd[0];
    ggml_tensor * fn = nd[1];
    if (rn->op != GGML_OP_RMS_NORM || fn->op != GGML_OP_MUL || fn->src[0] != rn || !strstr(fn->src[1]->name, ".ffn_norm.weight")) return false;
    const ggml_tensor * mm = nd[2];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != fn || !strstr(mm->src[0]->name, ".ffn_gate_inp.weight")) return false;
    const ggml_tensor * x = rn->src[0], * nw = fn->src[1], * wr = mm->src[0];
    if (x->type != GGML_TYPE_F32 || x->ne[0] != FRT_K_EMBD || ggml_nrows(x) != 1 || !ggml_is_contiguous(x) || !frt_aligned16(x->data)) return false;
    if (nw->type != GGML_TYPE_F32 || ggml_nelements(nw) != FRT_K_EMBD || !frt_aligned16(nw->data)) return false;
    if (fn->type != GGML_TYPE_F32 || !ggml_is_contiguous(fn) || !frt_aligned16(fn->data)) return false;
    if (wr->type != GGML_TYPE_BF16 || wr->ne[0] != FRT_K_EMBD || wr->ne[1] != 256 || !ggml_is_contiguous(wr)) return false;
    // sigmoid -> reshape -> argsort(desc) -> view top-k -> get_rows -> reshape -> sum_rows -> clamp -> div
    const ggml_tensor * sg = nd[3], * prs = nd[4], * tk = nd[6], * gr = nd[7], * wrs = nd[8], * sr = nd[9], * cl = nd[10];
    ggml_tensor * as = nd[5];
    ggml_tensor * dv = nd[11];
    if (!frt_unary_is(sg, GGML_UNARY_OP_SIGMOID) || sg->src[0] != mm) return false;
    if (prs->op != GGML_OP_RESHAPE || prs->src[0] != sg) return false;
    if (as->op != GGML_OP_ARGSORT || as->src[0] != sg || ggml_get_op_params_i32(as, 0) != GGML_SORT_ORDER_DESC || as->type != GGML_TYPE_I32) return false;
    if (tk->op != GGML_OP_VIEW || tk->src[0] != as || tk->view_offs != 0) return false;
    const int n_used = (int) tk->ne[0];
    if (gr->op != GGML_OP_GET_ROWS || gr->src[0] != prs || gr->src[1] != tk) return false;
    if (wrs->op != GGML_OP_RESHAPE || wrs->src[0] != gr) return false;
    if (sr->op != GGML_OP_SUM_ROWS || sr->src[0] != wrs) return false;
    if (cl->op != GGML_OP_CLAMP || cl->src[0] != sr) return false;
    if (dv->op != GGML_OP_DIV || dv->src[0] != wrs || dv->src[1] != cl || dv->type != GGML_TYPE_F32 || !ggml_is_contiguous(dv)) return false;
    if (n_used < 1 || n_used > FRT_MOE_MAX_USED || tk->ne[1] != 1 || ggml_nrows(as) != 1 || as->ne[0] != 256) return false;
    if (!frt_span_closed(cgraph, i, i + 11, { fn, as, tk })) return false;
    if (frt_overlap(fn, x) || frt_overlap(as, x) || frt_overlap(dv, x)) return false;
    float eps; memcpy(&eps, rn->op_params, sizeof(float));
    m = { x, nw, wr, fn, as, dv, eps, ggml_get_op_params_f32(cl, 0), n_used };
    return true;
}

static void frt_route_launch(ggml_backend_cuda_context & ctx, const frt_route_m & m) {
    static const int nw = frt_cfg_nw("FRT_NW_ROUTE", 16) == 8 ? 8 : 16;   // 512 (2 rows / CTA) >= 256 threads
    auto launch = [&](auto kernel, const int nwt) {
        const ggml_cuda_kernel_launch_params lp((dim3) (256 / (nwt / 8)), (dim3) (nwt * WARP_SIZE), 0, ctx.stream());
        ggml_cuda_kernel_launch(kernel, lp,
            (const float *) m.x->data, (const float *) m.nw->data, m.eps, FRT_K_EMBD, (float *) m.fn->data,
            (const nv_bfloat16 *) m.wr->data, (int64_t) (m.wr->nb[1] / sizeof(nv_bfloat16)), g_route_logits, g_route_count,
            (int32_t *) m.as->data, (float *) m.dv->data, m.n_used, m.clamp_val, 1.0f, g_act_q8, g_moe_tail.ids, g_moe_tail.wts);
    };
    if (nw == 16) {
        launch(frt_route<16>, 16);
    } else {
        launch(frt_route<8>, 8);
    }
}

static bool frt_route_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_ROUTE");
    frt_route_m m;
    if (!on || !frt_route_match(cgraph, i, m)) return false;
    frt_route_launch(ctx, m);
    g_act_q8_src = m.fn->data;
    *skip_count = 12;
    return true;
}

// ---- MoE FFN span (FRT window MOEFFN) --------------------------------------
// Replaces, for one decode token, everything after the router/top-k up to the
// layer output: routed gate/up MUL_MAT_ID + scales + SwiGLU, the activation
// rescale chain, routed down MUL_MAT_ID + scale, weighted expert sum, the
// shared expert (gate, up, SwiGLU, down, scales) and both residual adds:
// 14 stock launches -> 2.

struct frt_moe_chain { float s1, b1, s2, b2, cmin, cmax, s3, b3; };

// K1: one warp per GLU output; outputs [0, n_used*n_ff) routed, then n_ff_sh
// shared. CTA 0 snapshots the routing ids / weights for K2.

template <int NW, bool ACT_Q8>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_moe_gu(
        const float * act, const block_q8_1 * act_q8, const int32_t * ids, const float * wts,
        const char * gate_w, const char * up_w, const size_t e_stride, const size_t row_stride,
        const float * gate_sc, const float * up_sc,
        const char * shg_w, const char * shu_w, const float * shg_sc, const float * shu_sc,
        float * glu_out, const frt_moe_tail tl,
        const int n_used, const int n_ff, const int n_ff_sh, const float fzero) {
    ggml_cuda_pdl_lc();
    __shared__ block_q8_1 yq_sh[ACT_Q8 ? 1 : FRT_K_EMBD / QK8_1];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int o = blockIdx.x * NW + warp;
    const int n_routed = n_used * n_ff;
    const bool active = o < n_routed + n_ff_sh;
    const bool routed = o < n_routed;
    if (active && !routed) {   // shared expert rows do not depend on the router: warm L2
        const int r = o - n_routed;
        frt_l2_prefetch(shg_w + (size_t) r * row_stride, row_stride, lane);
        frt_l2_prefetch(shu_w + (size_t) r * row_stride, row_stride, lane);
    }
    ggml_cuda_pdl_sync();
    if (blockIdx.x == 0 && threadIdx.x < n_used) {
        tl.ids[threadIdx.x] = ids[threadIdx.x];
        tl.wts[threadIdx.x] = wts[threadIdx.x];
    }
    const block_q8_1 * yq;
    if constexpr (ACT_Q8) {
        yq = act_q8;
    } else {
        static_assert((FRT_K_EMBD / QK8_1) % NW == 0, "chunks per warp");
        frt_q8_1_chunks<FRT_K_EMBD / QK8_1 / NW>(act, yq_sh, warp, NW, lane);
        __syncthreads();
        yq = yq_sh;
    }
    if (active) {
        float res;
        frt_nvb bg[2], bu[2];
        float gs, us;
        if (routed) {
            const int e = ids[o / n_ff];
            const int r = o % n_ff;
            frt_ld_row<2>(bg, gate_w + (size_t) e * e_stride + (size_t) r * row_stride, lane);
            frt_ld_row<2>(bu, up_w   + (size_t) e * e_stride + (size_t) r * row_stride, lane);
            gs = gate_sc[e];
            us = up_sc[e];
        } else {
            const int r = o - n_routed;
            frt_ld_row<2>(bg, shg_w + (size_t) r * row_stride, lane);
            frt_ld_row<2>(bu, shu_w + (size_t) r * row_stride, lane);
            gs = shg_sc[0];
            us = shu_sc[0];
        }
        const float tg = frt_rowdot_k2048(bg, yq, lane, fzero);
        const float tu = frt_rowdot_k2048(bu, yq, lane, fzero);
        if (routed) {   // mmvq fused epilogue: x = up, gate = gate, SwiGLU
            float result = tu;
            result *= us;
            result += fzero;
            float gate_value = tg;
            gate_value *= gs;
            gate_value += fzero;
            result *= ggml_cuda_op_silu_single(gate_value);
            res = result;
        } else {        // two scaled GEMVs, then the GLU kernel
            float g = tg;
            g *= gs;
            g += fzero;
            float u = tu;
            u *= us;
            u += fzero;
            res = ggml_cuda_op_silu_single(g) * u;
        }
        if (lane == 0) {
            glu_out[o] = res;
        }
    }
}

// K1b: one CTA per routed slot (+1 for the shared expert). Routed: thread t holds glu[slot][t] (n_ff = NW * 32), block max ->
// e (stock chain) -> warp w quantizes chunk w of glu / e. Shared: plain q8_1.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_moe_glu_q8(
        const float * glu, const frt_moe_tail tl, const frt_moe_chain ch,
        const int n_used, const int n_ff, const int n_ff_sh) {
    ggml_cuda_pdl_lc();
    __shared__ float red[NW];
    __shared__ float e_sh;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int s = blockIdx.x;
    ggml_cuda_pdl_sync();
    if (s < n_used) {
        const float v = glu[s * n_ff + threadIdx.x];
        float m = fmaxf(-FLT_MAX, fabsf(v));
        m = warp_reduce_max(m);
        if (lane == 0) {
            red[warp] = m;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float a = red[0];
            for (int w = 1; w < NW; ++w) {
                a = fmaxf(a, red[w]);
            }
            float e = ch.s1 * a + ch.b1;
            e = logf(e);
            e = ch.s2 * e + ch.b2;
            e = ceilf(e);
            e = fminf(fmaxf(e, ch.cmin), ch.cmax);
            e = ch.s3 * e + ch.b3;
            e = expf(e);
            e_sh = e;
            tl.esc[s] = e;
        }
        __syncthreads();
        frt_q8_1_chunk(v / e_sh, lane, &tl.glu_q8[s * (n_ff / QK8_1) + warp]);
    } else {
        const int base = n_used * (n_ff / QK8_1);
        for (int k = warp; k < n_ff_sh / QK8_1; k += NW) {
            frt_q8_1_chunk(glu[n_used * n_ff + k * QK8_1 + lane], lane, &tl.glu_q8[base + k]);
        }
    }
}

// K2: one warp per output row over the q8_1 GLU blocks and rescale factors
// published by K1's tail (no per-CTA prologue).
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_moe_down(
        const block_q8_1 * glu_q8, const float * esc, const int32_t * ids, const float * wts,
        const char * down_w, const size_t de_stride, const size_t drow_stride, const float * down_sc,
        const char * shd_w, const size_t shd_row_stride, const float * shd_sc,
        const float * resid, float * out,
        const int n_used, const int n_ff, const int n_embd, const float fzero) {
    ggml_cuda_pdl_lc();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * NW + warp;
    if (row >= n_embd) {
        return;
    }
    // shared-expert down row does not depend on the router: warm L2
    frt_l2_prefetch(shd_w + (size_t) row * shd_row_stride, shd_row_stride, lane);
    const float shs = shd_sc[0];
    ggml_cuda_pdl_sync();
    const int cpe = n_ff / QK8_1;
    // per-slot scalars, lane s holds slot s: loaded in parallel instead of a serial chain in the epilogue
    int sid = 0;
    float sw = 0.0f, se = 0.0f, sds = 0.0f;
    if (lane < n_used) {
        sid = ids[lane];
        sw  = wts[lane];
        se  = esc[lane];
        sds = down_sc[sid];
    }
    frt_nvb be[FRT_MOE_MAX_USED / 2];
#pragma unroll
    for (int p = 0; p < FRT_MOE_MAX_USED / 2; ++p) {
        const int slot = 2 * p + lane / 16;   // n_used is even (checked by the predicate)
        const int id = __shfl_sync(0xffffffff, sid, slot);
        if (2 * p < n_used) {
            be[p] = frt_ld_nvb(down_w + (size_t) id * de_stride + (size_t) row * drow_stride,
                               (lane % 16) / 2, frt_lane_iqs(lane));
        }
    }
    const frt_nvb bsh = frt_ld_nvb(shd_w + (size_t) row * shd_row_stride, lane / 2, frt_lane_iqs(lane));
    float ds[FRT_MOE_MAX_USED], ws[FRT_MOE_MAX_USED], es[FRT_MOE_MAX_USED];
#pragma unroll
    for (int k = 0; k < FRT_MOE_MAX_USED; ++k) {
        ds[k] = __shfl_sync(0xffffffff, sds, k);
        ws[k] = __shfl_sync(0xffffffff, sw, k);
        es[k] = __shfl_sync(0xffffffff, se, k);
    }
    float de[FRT_MOE_MAX_USED];
#pragma unroll
    for (int p = 0; p < FRT_MOE_MAX_USED / 2; ++p) {
        if (2 * p < n_used) {
            const int slot = 2 * p + lane / 16;
            const float t = frt_rowdot_k512_pair(be[p], glu_q8 + slot * cpe, lane, fzero);
            de[2 * p]     = __shfl_sync(0xffffffff, t, 0);
            de[2 * p + 1] = __shfl_sync(0xffffffff, t, 16);
        }
    }
    const float tsh = frt_rowdot_k1024(bsh, glu_q8 + n_used * cpe, lane, fzero);
    if (lane == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < FRT_MOE_MAX_USED; ++s) {
            if (s < n_used) {
                float r = de[s];          // mmvq epilogue: per-expert scale, zero bias
                r *= ds[s];
                r += fzero;
                if (s == 0) {
                    sum = (r * es[0]) * ws[0];
                } else {
                    sum += (r * es[s]) * ws[s];
                }
            }
        }
        float sh = tsh;
        sh *= shs;
        sh += fzero;
        float result = sum;           // k_bin_bcast with two fused addends
        result = result + sh;
        result = result + resid[row];
        out[row] = result;
    }
}

static float * g_moe_glu = nullptr;   // n_used*n_ff + n_ff_sh f32, allocated eagerly
static constexpr int FRT_MOE_GLU_CAP = FRT_MOE_MAX_USED * 512 + 1024;

// anchor: MUL_MAT_ID on blk.N.ffn_gate_exps.weight at M = 1.
struct frt_moeffn_m {
    const ggml_tensor * act, * ids, * wts, * resid;
    const ggml_tensor * gw, * uw, * dw, * gsc, * usc, * dsc;
    const ggml_tensor * shgw, * shuw, * shdw, * shgs, * shus, * shds;
    ggml_tensor * l_out;
    frt_moe_chain ch;
    int n_used, n_ff, n_ff_sh, K, end;
};

static bool frt_moeffn_match(ggml_cgraph * cgraph, int i, frt_moeffn_m & m) {
    if (g_moe_glu == nullptr) return false;
    ggml_tensor * gate_id = cgraph->nodes[i];
    if (gate_id->op != GGML_OP_MUL_MAT_ID || !strstr(gate_id->src[0]->name, ".ffn_gate_exps.weight")) return false;
    const ggml_tensor * gw  = gate_id->src[0];
    const ggml_tensor * act = gate_id->src[1];
    const ggml_tensor * ids = gate_id->src[2];
    const int64_t K = gw->ne[0], n_ff = gw->ne[1], n_exp = gw->ne[2];
    const int n_used = (int) ids->ne[0];
    // pinned shapes: K = n_embd = 2048, n_ff = 512, M = 1, even n_used <= 8
    if (gw->type != GGML_TYPE_NVFP4 || K != FRT_K_EMBD || n_ff != 512) return false;
    if (act->type != GGML_TYPE_F32 || act->ne[0] != K || act->ne[1] != 1 || act->ne[2] != 1 || !ggml_is_contiguous(act)) return false;
    // n_used < 8: the down kernel's prologue gives each routed slot a warp and needs one spare for the shared expert
    if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || n_used < 2 || n_used >= FRT_MOE_MAX_USED || (n_used & 1)) return false;
    if (gate_id->ne[1] != n_used || gate_id->ne[2] != 1) return false;

    ggml_tensor * up_id = nullptr, * glu = nullptr, * down_id = nullptr, * ex = nullptr, * dv = nullptr;
    const ggml_tensor * gate_mul = nullptr, * up_mul = nullptr, * down_mul = nullptr, * e_mul = nullptr, * w_mul = nullptr;
    const ggml_tensor * moe_out = nullptr, * shg = nullptr, * shu = nullptr, * shd = nullptr;
    const ggml_tensor * shg_mul = nullptr, * shu_mul = nullptr, * shd_mul = nullptr, * sh_glu = nullptr, * ffn_out = nullptr;
    ggml_tensor * l_out = nullptr;
    const ggml_tensor * pool = nullptr, * sc[3] = {}, * clmp = nullptr;
    int n_sc = 0, end = -1;
    const int LIM = std::min(cgraph->n_nodes, i + 64);
    for (int j = i + 1; j < LIM && end < 0; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        switch (n->op) {
            case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_REPEAT: case GGML_OP_GET_ROWS:
                break;   // scale-gather plumbing and expert views, wiring checked below
            case GGML_OP_MUL_MAT_ID:
                if (!up_id && strstr(n->src[0]->name, ".ffn_up_exps.weight")) up_id = n;
                else if (!down_id && strstr(n->src[0]->name, ".ffn_down_exps.weight")) down_id = n;
                else return false;
                break;
            case GGML_OP_MUL:
                if (n->src[0] == gate_id) gate_mul = n;
                else if (up_id && n->src[0] == up_id) up_mul = n;
                else if (down_id && n->src[0] == down_id) down_mul = n;
                else if (down_mul && n->src[0] == down_mul && n->src[1] == ex) e_mul = n;
                else if (e_mul && n->src[0] == e_mul) w_mul = n;
                else if (shg && n->src[0] == shg) shg_mul = n;
                else if (shu && n->src[0] == shu) shu_mul = n;
                else if (shd && n->src[0] == shd) shd_mul = n;
                else return false;
                break;
            case GGML_OP_GLU:
                if (ggml_get_glu_op(n) != GGML_GLU_OP_SWIGLU) return false;
                if (!glu && gate_mul && up_mul && n->src[0] == gate_mul && n->src[1] == up_mul) glu = n;
                else if (shg_mul && shu_mul && n->src[0] == shg_mul && n->src[1] == shu_mul) sh_glu = n;
                else return false;
                break;
            case GGML_OP_UNARY:
                if (frt_unary_is(n, GGML_UNARY_OP_ABS) && n->src[0] == glu) break;
                if (frt_unary_is(n, GGML_UNARY_OP_CEIL)) break;
                if (frt_unary_is(n, GGML_UNARY_OP_EXP)) { ex = n; break; }
                return false;
            case GGML_OP_POOL_1D: pool = n; break;
            case GGML_OP_SCALE: if (n_sc == 3) return false; sc[n_sc++] = n; break;
            case GGML_OP_LOG: break;
            case GGML_OP_CLAMP: clmp = n; break;
            case GGML_OP_DIV:
                if (dv || n->src[0] != glu || n->src[1] != ex) return false;
                dv = n;
                break;
            case GGML_OP_MUL_MAT:
                if (!moe_out) return false;
                if (strstr(n->src[0]->name, ".ffn_gate_shexp.weight")) shg = n;
                else if (strstr(n->src[0]->name, ".ffn_up_shexp.weight")) shu = n;
                else if (strstr(n->src[0]->name, ".ffn_down_shexp.weight")) shd = n;
                else return false;
                break;
            case GGML_OP_ADD:
                if (strncmp(n->name, "ffn_moe_out", 11) == 0) moe_out = n;
                else if (moe_out && shd_mul && !ffn_out && strncmp(n->name, "ffn_out", 7) == 0) ffn_out = n;
                else if (ffn_out && strncmp(n->name, "l_out", 5) == 0) { l_out = n; end = j; }
                else if (moe_out) return false;
                break;   // expert-sum chain adds before ffn_moe_out: verified via the reduction match below
            default:
                return false;
        }
    }
    if (end < 0 || !up_id || !glu || !down_id || !ex || !dv || !gate_mul || !up_mul || !down_mul || !e_mul || !w_mul ||
        !moe_out || !shg || !shu || !shd || !shg_mul || !shu_mul || !shd_mul || !sh_glu || !ffn_out || !pool || !clmp || n_sc != 3) {
        return false;
    }
    // routed wiring
    const ggml_tensor * uw = up_id->src[0], * dw = down_id->src[0];
    if (uw->type != GGML_TYPE_NVFP4 || dw->type != GGML_TYPE_NVFP4 || !ggml_are_same_shape(uw, gw) || uw->nb[1] != gw->nb[1] || uw->nb[2] != gw->nb[2]) return false;
    if (dw->ne[0] != n_ff || dw->ne[1] != K || dw->ne[2] != n_exp) return false;
    if (up_id->src[1] != act || up_id->src[2] != ids || down_id->src[1] != dv || down_id->src[2] != ids) return false;
    const ggml_tensor * gsc = frt_id_scale_src(gate_mul, gate_id, ids);
    const ggml_tensor * usc = frt_id_scale_src(up_mul, up_id, ids);
    const ggml_tensor * dsc = frt_id_scale_src(down_mul, down_id, ids);
    if (!gsc || !usc || !dsc) return false;
    // activation-rescale chain (same pins as the AMAX window)
    const int32_t * pp = (const int32_t *) pool->op_params;
    if (pp[0] != GGML_OP_POOL_MAX || pp[1] != n_ff || pp[2] != n_ff || pp[3] != 0 || pool->src[0]->src[0] != glu) return false;
    if (sc[0]->src[0] != pool || ex->src[0] != sc[2] || clmp->src[0]->src[0] != sc[1]) return false;
    // weighted sum: e_mul = down_mul * ex, w_mul = e_mul * weights[1, n_used, 1]
    const ggml_tensor * wts = w_mul->src[1];
    if (wts->type != GGML_TYPE_F32 || !ggml_is_contiguous(wts) || wts->ne[0] != 1 || wts->ne[1] != n_used) return false;
    if (moe_out->ne[0] != K || moe_out->ne[1] != 1) return false;
    {   // expert sum in slot order: ((v0 + v1) + v2) + ... over views of w_mul
        auto slot_of = [&](const ggml_tensor * v) -> int {
            if (v->op != GGML_OP_VIEW || v->view_src != w_mul || v->ne[0] != K || v->nb[0] != w_mul->nb[0]) return -1;
            if (v->view_offs % w_mul->nb[1] != 0) return -1;
            return (int) (v->view_offs / w_mul->nb[1]);
        };
        const ggml_tensor * a = moe_out;
        for (int k = n_used - 1; k >= 1; --k) {
            if (a->op != GGML_OP_ADD || slot_of(a->src[1]) != k) return false;
            a = a->src[0];
        }
        if (slot_of(a) != 0) return false;
    }
    // shared expert wiring
    const ggml_tensor * shgw = shg->src[0], * shuw = shu->src[0], * shdw = shd->src[0];
    const int64_t n_ff_sh = shgw->ne[1];
    if (shgw->type != GGML_TYPE_NVFP4 || shuw->type != GGML_TYPE_NVFP4 || shdw->type != GGML_TYPE_NVFP4) return false;
    if (n_ff_sh != 1024 || shgw->ne[0] != K || !ggml_are_same_shape(shgw, shuw) || shdw->ne[0] != n_ff_sh || shdw->ne[1] != K) return false;
    if (shg->src[1]->data != act->data || shu->src[1]->data != act->data || shd->src[1] != sh_glu) return false;
    if (shgw->nb[1] != gw->nb[1] || shuw->nb[1] != gw->nb[1]) return false;
    const ggml_tensor * shgs = frt_mm_scale_src(shg_mul, shg), * shus = frt_mm_scale_src(shu_mul, shu), * shds = frt_mm_scale_src(shd_mul, shd);
    if (!shgs || !shus || !shds) return false;
    if (!((ffn_out->src[0] == moe_out && ffn_out->src[1] == shd_mul))) return false;
    const ggml_tensor * resid = l_out->src[0] == ffn_out ? l_out->src[1] : (l_out->src[1] == ffn_out ? l_out->src[0] : nullptr);
    if (!resid || l_out->src[0] != ffn_out || resid->type != GGML_TYPE_F32 || !ggml_is_contiguous(resid) || resid->ne[0] != K || resid->ne[1] != 1) return false;
    if (l_out->type != GGML_TYPE_F32 || !ggml_is_contiguous(l_out) || l_out->ne[0] != K || l_out->ne[1] != 1) return false;
    if (n_used * n_ff + n_ff_sh > FRT_MOE_GLU_CAP) return false;
    // K2 thread r reads resid[r] before writing out[r]: an exact alias is safe, a partial one is not
    if (l_out->data != resid->data && frt_overlap(l_out, resid)) return false;
    if (!frt_span_closed(cgraph, i, end, {})) return false;

    const frt_moe_chain ch = {
        ggml_get_op_params_f32(sc[0], 0), ggml_get_op_params_f32(sc[0], 1),
        ggml_get_op_params_f32(sc[1], 0), ggml_get_op_params_f32(sc[1], 1),
        ggml_get_op_params_f32(clmp, 0),  ggml_get_op_params_f32(clmp, 1),
        ggml_get_op_params_f32(sc[2], 0), ggml_get_op_params_f32(sc[2], 1) };

    if (n_ff % 8 != 0 || n_ff_sh % (8 * QK8_1) != 0) return false;
    m = { act, ids, wts, resid, gw, uw, dw, gsc, usc, dsc, shgw, shuw, shdw, shgs, shus, shds, l_out, ch,
          n_used, (int) n_ff, (int) n_ff_sh, (int) K, end };
    return true;
}

static void frt_moeffn_launch(ggml_backend_cuda_context & ctx, const frt_moeffn_m & m, const bool act_q8) {
    cudaStream_t stream = ctx.stream();
    static const int nw_gu = frt_cfg_nw("FRT_NW_GU", 4);      // tile search: 4 >= 8 > 16
    const int n_out1 = m.n_used * m.n_ff + m.n_ff_sh;
    FRT_NW_DISPATCH(nw_gu,
        const ggml_cuda_kernel_launch_params lp1((dim3) ((n_out1 + NWT - 1) / NWT), (dim3) (NWT * WARP_SIZE), 0, stream);
        ggml_cuda_kernel_launch(act_q8 ? frt_moe_gu<NWT, true> : frt_moe_gu<NWT, false>, lp1,
            (const float *) m.act->data, (const block_q8_1 *) g_act_q8, (const int32_t *) m.ids->data, (const float *) m.wts->data,
            (const char *) m.gw->data, (const char *) m.uw->data, (size_t) m.gw->nb[2], (size_t) m.gw->nb[1],
            (const float *) m.gsc->data, (const float *) m.usc->data,
            (const char *) m.shgw->data, (const char *) m.shuw->data, (const float *) m.shgs->data, (const float *) m.shus->data,
            g_moe_glu, g_moe_tail, m.n_used, m.n_ff, m.n_ff_sh, 0.0f));
    constexpr int NWQ = 16;   // n_ff = 512 = NWQ * 32 (pinned by the predicate)
    ggml_cuda_kernel_launch(frt_moe_glu_q8<NWQ>, ggml_cuda_kernel_launch_params((dim3) (m.n_used + 1), (dim3) (NWQ * WARP_SIZE), 0, stream),
        (const float *) g_moe_glu, g_moe_tail, m.ch, m.n_used, m.n_ff, m.n_ff_sh);
    static const int nw_dn = frt_cfg_nw("FRT_NW_DOWN", 8);
    FRT_NW_DISPATCH(nw_dn,
        ggml_cuda_kernel_launch(frt_moe_down<NWT>, ggml_cuda_kernel_launch_params((dim3) ((m.K + NWT - 1) / NWT), (dim3) (NWT * WARP_SIZE), 0, stream),
            (const block_q8_1 *) g_moe_tail.glu_q8, (const float *) g_moe_tail.esc, (const int32_t *) g_moe_tail.ids, (const float *) g_moe_tail.wts,
            (const char *) m.dw->data, (size_t) m.dw->nb[2], (size_t) m.dw->nb[1], (const float *) m.dsc->data,
            (const char *) m.shdw->data, (size_t) m.shdw->nb[1], (const float *) m.shds->data,
            (const float *) m.resid->data, (float *) m.l_out->data,
            m.n_used, m.n_ff, m.K, 0.0f));
    if (frt_stats_on()) {
        static long fires = 0;
        if ((++fires & 0xFFF) == 1) fprintf(stderr, "frt-stats moeffn fires=%ld act_q8=%d\n", fires, (int) act_q8);
    }
}

static bool frt_moeffn_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_MOEFFN");
    frt_moeffn_m m;
    if (!on || !frt_moeffn_match(cgraph, i, m)) return false;
    const bool act_q8 = g_act_q8_src == m.act->data;   // ROUTE already published q8_1(act) for this layer
    g_act_q8_src = nullptr;
    frt_moeffn_launch(ctx, m, act_q8);
    *skip_count = m.end - i + 1;
    return true;
}

// ---- decode attention (FRT window ATTNDEC) --------------------------------
// M = 1 attention over the f16 KV cache, GQA-shared: one CTA per (KV head,
// KV chunk) computes all query heads of that KV head with m16n8k16 tensor-core
// MMAs (the query heads are the M rows), so K and V are read once per KV head
// instead of once per query head. Fully masked chunks (most of a sliding-
// window cache) exit after reading the mask. A combine kernel merges the
// chunk partials and the learned attention sinks.
// Numerics: ggml's vector kernel (NVIDIA path) works in f32 on the f16
// cache: f32 Q, f32 scores, f32 P and f32 PV accumulation. The MMAs here
// take f16 operands, so Q and P are split into f16 hi + lo parts. The M
// rows 8..15 of the m16n8k16 A fragment (padding with 8 query heads) carry
// the lo parts, and C rows h and h+8 are summed. This gives ~f32-level
// operand precision at no extra MMA cost. The summation order differs from
// ggml's, so the window is gated by the decode-path PPL/KLD protocol, not
// by bitwise parity.

static constexpr int FRT_FA_D   = 128;   // head size (pinned)
static constexpr int FRT_FA_G   = 8;     // query heads per KV head (pinned)
static constexpr int FRT_FA_NW  = 4;     // warps per CTA, 16 tokens each per step
static constexpr int FRT_FA_STEP = FRT_FA_NW * 16;   // KV rows per CTA step
static constexpr int FRT_FA_MAX_SPLITS = 512;

static __device__ __forceinline__ void frt_mma_16816(float (&c)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ void frt_ldsm_x4(uint32_t (&r)[4], const void * p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

static __device__ __forceinline__ void frt_ldsm_x4_t(uint32_t (&r)[4], const void * p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

static __device__ __forceinline__ void frt_cp_async16(void * dst, const void * src) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(src));
}

// f32 pair -> f16 hi + f16 lo (hi + lo == x up to ~2^-22 relative)
static __device__ __forceinline__ void frt_h2_split(const float a, const float b, uint32_t & hi, uint32_t & lo) {
    const half2 h = __floats2half2_rn(a, b);
    const half2 l = __floats2half2_rn(a - __low2float(h), b - __high2float(h));
    hi = *(const uint32_t *) &h;
    lo = *(const uint32_t *) &l;
}

// smem row pitch (halves) for a 128-wide f16 row: +8 halves breaks ldmatrix bank conflicts
static constexpr int FRT_FA_PITCH = FRT_FA_D + 8;

// partial for (query head, chunk): running max, sum and unnormalized output
struct frt_fa_part { float m, l; };

// QKN: 0 = Q from global (written by K_qkn); 1 / 2 = the per-head epilogue
// (norm [+ rope when 2], set_rows of the new K / V row) runs here, see
// frt_fa_pending: each CTA normalizes its 8 query heads from g_qkv, and the
// CTA whose chunk holds the new cache cell writes that row (its KV head)
// and patches it into its staged tile.
template <bool PF, int QKN>
static __global__ void __launch_bounds__(FRT_FA_NW * WARP_SIZE, 1) frt_fa_decode(
        const float * Q, const int64_t q_head_stride,           // f32 [D] per query head
        const char * K, const char * V, const int64_t kv_row, const int64_t kv_head,   // f16 rows
        const half * mask, const float scale, const int n_kv,
        frt_fa_part * part_ml, float * part_o, const int n_split,
        const float * qkv, const frt_qkn_params qp, const int steps, const bool prestage_ok) {
    constexpr int D = FRT_FA_D;
    const int CH = FRT_FA_STEP * steps;   // KV rows per CTA
    const bool prestage = QKN != 0 && prestage_ok;
    __shared__ __align__(16) half sK[FRT_FA_STEP][FRT_FA_PITCH];
    __shared__ __align__(16) half sV[FRT_FA_STEP][FRT_FA_PITCH];
    __shared__ float sm_m[FRT_FA_NW][FRT_FA_G], sm_l[FRT_FA_NW][FRT_FA_G];
    __shared__ float sQ[QKN ? FRT_FA_G : 1][D];
    __shared__ __align__(16) half sNK[QKN ? D : 8], sNV[QKN ? D : 8];
    __shared__ int any_live;
    ggml_cuda_pdl_lc();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int split = blockIdx.x, g = blockIdx.y;
    const int t0 = split * CH;
    const char * Kg = K + (int64_t) g * kv_head;
    const char * Vg = V + (int64_t) g * kv_head;
    // The mask is a graph input (final before the graph starts), so the
    // chunk's live test runs before the dependency wait.
    if (threadIdx.x == 0) {
        any_live = 0;
    }
    __syncthreads();
    {
        bool live = false;
        for (int t = threadIdx.x; t < CH && t0 + t < n_kv; t += FRT_FA_NW * WARP_SIZE) {
            live |= __half2float(mask[t0 + t]) != -INFINITY;
        }
        if (live) {
            any_live = 1;
        }
    }
    __syncthreads();
    const bool chunk_live = any_live != 0;
    // Older KV rows are final before the graph starts; the only row written
    // during this graph is the new cell. With QKN that write happens in this
    // kernel and the staged copy of the cell is patched below, so the first
    // step is staged before the dependency wait. Otherwise the new row comes
    // from the previous kernel and only L2 prefetches (coherent with that
    // write) are issued early. Masked rows are not fetched.
    auto stage = [&](const int tb) {   // K and V rows [tb, tb + STEP): 16 B per cp.async, 16 per row
        for (int c = threadIdx.x; c < FRT_FA_STEP * (D / 8); c += FRT_FA_NW * WARP_SIZE) {
            const int r = c / (D / 8), cc = (c % (D / 8)) * 8;
            const int64_t off = (int64_t) (tb + r) * kv_row + cc * (int64_t) sizeof(half);
            frt_cp_async16(&sK[r][cc], Kg + off);
            frt_cp_async16(&sV[r][cc], Vg + off);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    if (chunk_live) {
        for (int t = prestage ? FRT_FA_STEP + threadIdx.x : threadIdx.x; PF && t < CH && t0 + t < n_kv; t += FRT_FA_NW * WARP_SIZE) {
            if (__half2float(mask[t0 + t]) == -INFINITY) {
                continue;
            }
            asm volatile("prefetch.global.L2 [%0];" :: "l"(Kg + (int64_t) (t0 + t) * kv_row));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(Kg + (int64_t) (t0 + t) * kv_row + 128));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(Vg + (int64_t) (t0 + t) * kv_row));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(Vg + (int64_t) (t0 + t) * kv_row + 128));
        }
        if (prestage) {
            stage(t0);
        }
    }
    ggml_cuda_pdl_sync();
    int ck = -1, cv = -1;   // new cache cells (QKN)
    if constexpr (QKN != 0) {
        constexpr bool rope = QKN == 2;
        ck = (int) qp.k_idx[0];
        cv = (int) qp.v_idx[0];
        for (int hl = warp; hl < FRT_FA_G; hl += FRT_FA_NW) {
            const int h = g * FRT_FA_G + hl;
            float out[4];
            int   oix[4];
            frt_qkn_vals<rope>(qkv + (size_t) h * D, qp.qn_w, lane, qp, out, oix);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                sQ[hl][oix[k]] = out[k];
            }
            if (split == 0) {   // the Q node's own output, for completeness
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    qp.q_out[(size_t) h * D + oix[k]] = out[k];
                }
            }
        }
        // the owner of a cell is the chunk holding it (the last chunk if it lies past the view)
        const bool own_k = (ck >= t0 && ck < t0 + CH) || (split == n_split - 1 && ck >= n_kv);
        const bool own_v = (cv >= t0 && cv < t0 + CH) || (split == n_split - 1 && cv >= n_kv);
        if (own_k && warp == 0) {
            float out[4];
            int   oix[4];
            frt_qkn_vals<rope>(qkv + (size_t) (qp.hq + g) * D, qp.kn_w, lane, qp, out, oix);
            half * dst = (half *) ((char *) qp.k_cache + (size_t) ck * qp.k_row_bytes) + (size_t) g * D;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const half hv = __float2half(out[k]);
                dst[oix[k]] = hv;
                sNK[oix[k]] = hv;
            }
        }
        if (own_v && warp == 1) {
            const float * xc = qkv + (size_t) (qp.hq + qp.hk + g) * D;
            half * dst = (half *) ((char *) qp.v_cache + (size_t) cv * qp.v_row_bytes) + (size_t) g * D;
#pragma unroll
            for (int k = 0; k < D / WARP_SIZE; ++k) {
                const half hv = __float2half(__ldcg(xc + lane + k * WARP_SIZE));
                dst[lane + k * WARP_SIZE] = hv;
                sNV[lane + k * WARP_SIZE] = hv;
            }
        }
    }
    __syncthreads();   // sQ / sNK / sNV
    frt_fa_part * pml = part_ml + (int64_t) (g * FRT_FA_G) * n_split + split;
    if (!chunk_live) {   // fully masked chunk
        if (threadIdx.x < FRT_FA_G) {
            pml[(int64_t) threadIdx.x * n_split] = { -INFINITY, 0.0f };
        }
        return;
    }
    // Q fragments (A operand): rows 0..7 = f16 hi of query heads 0..7, rows 8..15 = f16 lo
    uint32_t qa[D / 16][4];   // {a0a1 (hi, k lo), a2a3 (lo, k lo), a4a5 (hi, k hi), a6a7 (lo, k hi)}
    {
        const float * q = QKN ? &sQ[lane / 4][0] : Q + (int64_t) (g * FRT_FA_G + lane / 4) * q_head_stride;
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            const int c = kk * 16 + (lane % 4) * 2;
            frt_h2_split(q[c],     q[c + 1], qa[kk][0], qa[kk][1]);
            frt_h2_split(q[c + 8], q[c + 9], qa[kk][2], qa[kk][3]);
        }
    }
    float m_run = -INFINITY, l_run = 0.0f;   // for query head lane/4 (row of the C fragments)
    float o[D / 8][2];                        // O rows 0..7: dims tile j, cols (lane%4)*2 + {0,1}
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        o[j][0] = o[j][1] = 0.0f;
    }
    for (int it = 0; it < steps; ++it) {
        const int tb = t0 + it * FRT_FA_STEP;
        if (tb >= n_kv) {
            break;
        }
        if (!prestage || it > 0) {
            __syncthreads();   // previous step's smem reads done
            stage(tb);
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if constexpr (QKN != 0) {   // the new row: use this CTA's own values
            const bool pk = ck >= tb && ck < tb + FRT_FA_STEP, pv = cv >= tb && cv < tb + FRT_FA_STEP;
            if (pk || pv) {
                if (pk && threadIdx.x < D / 8) {
                    *(uint4 *) &sK[ck - tb][threadIdx.x * 8] = *(const uint4 *) &sNK[threadIdx.x * 8];
                }
                if (pv && threadIdx.x >= 32 && threadIdx.x < 32 + D / 8) {
                    *(uint4 *) &sV[cv - tb][(threadIdx.x - 32) * 8] = *(const uint4 *) &sNV[(threadIdx.x - 32) * 8];
                }
                __syncthreads();
            }
        }
        // S = Q K^T for this warp's 16 rows: two n8 tiles x 8 k-steps
        const int wr = warp * 16;   // first KV row of the warp (in the step)
        float s[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        }
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            // x4: matrices (rows wr..+7, k lo), (rows wr..+7, k hi), (rows wr+8..+15, k lo), (rows wr+8.., k hi)
            uint32_t kb[4];
            const int mat = lane / 8, r8 = lane % 8;
            frt_ldsm_x4(kb, &sK[wr + (mat / 2) * 8 + r8][kk * 16 + (mat % 2) * 8]);
            frt_mma_16816(s[0], qa[kk], kb[0], kb[1]);
            frt_mma_16816(s[1], qa[kk], kb[2], kb[3]);
        }
        // Q.K = hi row + lo row
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            s[n][0] += s[n][2];
            s[n][1] += s[n][3];
        }
        // scale + mask; columns of the C fragment are KV rows, rows are query heads
        float mx = -INFINITY;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int t = tb + wr + n * 8 + (lane % 4) * 2 + e;
                const float mv = __half2float(mask[t]);
                s[n][e] = s[n][e] * scale + mv;
                mx = fmaxf(mx, s[n][e]);
            }
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        const float m_new = fmaxf(m_run, mx);
        const float corr = m_new == -INFINITY ? 1.0f : expf(m_run - m_new);
        float ps = 0.0f;
        float p[2][2];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                p[n][e] = m_new == -INFINITY ? 0.0f : expf(s[n][e] - m_new);
                ps += p[n][e];
            }
        }
        ps += __shfl_xor_sync(0xffffffff, ps, 1);
        ps += __shfl_xor_sync(0xffffffff, ps, 2);
        l_run = l_run * corr + ps;
        m_run = m_new;
#pragma unroll
        for (int j = 0; j < D / 8; ++j) {
            o[j][0] *= corr;
            o[j][1] *= corr;
        }
        // O += P V: A = P (rows 0..7 = heads hi, rows 8..15 = lo; k = the warp's 16 KV rows),
        // B = V tile via ldmatrix.trans
        uint32_t pa[4];
        frt_h2_split(p[0][0], p[0][1], pa[0], pa[1]);
        frt_h2_split(p[1][0], p[1][1], pa[2], pa[3]);
#pragma unroll
        for (int j2 = 0; j2 < D / 16; ++j2) {
            // x4.trans: (k rows wr..+7, dims j lo), (k rows wr+8..+15, dims j lo), (wr.., dims hi), (wr+8.., dims hi)
            uint32_t vb[4];
            const int mat = lane / 8, r8 = lane % 8;
            frt_ldsm_x4_t(vb, &sV[wr + (mat % 2) * 8 + r8][j2 * 16 + (mat / 2) * 8]);
            float c0[4] = { o[2 * j2][0], o[2 * j2][1], 0.0f, 0.0f };
            float c1[4] = { o[2 * j2 + 1][0], o[2 * j2 + 1][1], 0.0f, 0.0f };
            frt_mma_16816(c0, pa, vb[0], vb[1]);
            frt_mma_16816(c1, pa, vb[2], vb[3]);
            o[2 * j2][0] = c0[0] + c0[2]; o[2 * j2][1] = c0[1] + c0[3];
            o[2 * j2 + 1][0] = c1[0] + c1[2]; o[2 * j2 + 1][1] = c1[1] + c1[3];
        }
    }
    // merge the 4 warps' (m, l, O) for each query head
    if (lane % 4 == 0) {
        sm_m[warp][lane / 4] = m_run;
        sm_l[warp][lane / 4] = l_run;
    }
    __syncthreads();
    float * so = (float *) &sK[0][0];   // reuse: [NW][G][D] f32 = 16 KB
    const int h = lane / 4;
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < FRT_FA_NW; ++w) {
        M = fmaxf(M, sm_m[w][h]);
    }
    const float f = M == -INFINITY ? 0.0f : expf(m_run - M);
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        const int d = j * 8 + (lane % 4) * 2;
        so[(warp * FRT_FA_G + h) * D + d]     = o[j][0] * f;
        so[(warp * FRT_FA_G + h) * D + d + 1] = o[j][1] * f;
    }
    __syncthreads();
    // thread -> (head, 4 dims): 8 heads x 128 dims over 128 threads
    for (int idx = threadIdx.x; idx < FRT_FA_G * D; idx += FRT_FA_NW * WARP_SIZE) {
        const int hh = idx / D, d = idx % D;
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < FRT_FA_NW; ++w) {
            acc += so[(w * FRT_FA_G + hh) * D + d];
        }
        part_o[((int64_t) (g * FRT_FA_G + hh) * n_split + split) * D + d] = acc;
    }
    if (threadIdx.x < FRT_FA_G) {
        const int hh = threadIdx.x;
        float MM = -INFINITY, L = 0.0f;
#pragma unroll
        for (int w = 0; w < FRT_FA_NW; ++w) {
            MM = fmaxf(MM, sm_m[w][hh]);
        }
#pragma unroll
        for (int w = 0; w < FRT_FA_NW; ++w) {
            L += MM == -INFINITY ? 0.0f : sm_l[w][hh] * expf(sm_m[w][hh] - MM);
        }
        pml[(int64_t) hh * n_split] = { MM, L };
    }
}

// one CTA per query head, one thread per dim: merge chunk partials + sink.
// No block-level reduction: every thread reads the (m, l) pairs itself
// (broadcast loads) and its own O column, CH partials per batch of
// independent loads, merged with an online max rescale.
static __global__ void frt_fa_combine(const frt_fa_part * part_ml, const float * part_o, const float * sinks,
        float * dst, const int64_t dst_head_stride, const int n_split) {
    ggml_cuda_pdl_lc();
    constexpr int D = FRT_FA_D;
    constexpr int CH = 32;
    const int h = blockIdx.x, d = threadIdx.x;
    const float sink = sinks ? sinks[h] : -INFINITY;   // immutable
    ggml_cuda_pdl_sync();
    const frt_fa_part * pm = part_ml + (int64_t) h * n_split;
    const float * po = part_o + (int64_t) h * n_split * D + d;
    float M = sink, L = sinks ? 1.0f : 0.0f, acc = 0.0f;   // the sink adds exp(sink - M) to the denominator only
    for (int j0 = 0; j0 < n_split; j0 += CH) {
        frt_fa_part q[CH];
        float o[CH];
#pragma unroll
        for (int k = 0; k < CH; ++k) {
            if (j0 + k < n_split) {
                q[k] = pm[j0 + k];
                o[k] = po[(int64_t) (j0 + k) * D];
            } else {
                q[k] = { -INFINITY, 0.0f };
                o[k] = 0.0f;
            }
        }
        float mc = M;
#pragma unroll
        for (int k = 0; k < CH; ++k) {
            mc = fmaxf(mc, q[k].m);
        }
        if (mc == -INFINITY) {
            continue;   // nothing live yet
        }
        const float f = expf(M - mc);   // M == -inf -> 0 (L and acc are 0 then)
        L *= f;
        acc *= f;
#pragma unroll
        for (int k = 0; k < CH; ++k) {
            const float w = expf(q[k].m - mc);   // masked chunk: m = -inf -> 0
            L   += q[k].l * w;
            acc += o[k] * w;
        }
        M = mc;
    }
    dst[(int64_t) h * dst_head_stride + d] = acc / L;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static frt_fa_part * g_fa_ml = nullptr;
static float * g_fa_o = nullptr;
static constexpr int FRT_FA_HEADS_MAX = 64;

static bool frt_attndec_on() {
    static const bool on = frt_layer_enabled() && getenv("FRT_ATTNDEC") && getenv("FRT_ATTNDEC")[0] == '1';
    return on && g_fa_ml != nullptr;
}

struct frt_fa_plan {
    const ggml_tensor * Q, * K, * V, * mask, * sinks;
    float scale;
    int n_kv, steps, n_split;
};

static bool frt_attndec_plan(const ggml_cgraph * cgraph, const int i, frt_fa_plan & pl) {
    const ggml_tensor * fa = cgraph->nodes[i];
    if (fa->op != GGML_OP_FLASH_ATTN_EXT) return false;
    // the fuse hook treats a 1-node window as "not fused" and would run the
    // stock kernel over our output: consume the pure-view reshape that follows
    if (i + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * nxt = cgraph->nodes[i + 1];
    if ((nxt->op != GGML_OP_RESHAPE && nxt->op != GGML_OP_VIEW) || nxt->src[0] != fa) return false;
    const ggml_tensor * Q = fa->src[0], * K = fa->src[1], * V = fa->src[2], * mask = fa->src[3], * sinks = fa->src[4];
    // Q [D, 1 token, H], K/V [D, n_kv, Hkv] f16 views of the cache, mask [n_kv, >=1] f16
    if (Q->type != GGML_TYPE_F32 || Q->ne[0] != FRT_FA_D || Q->ne[1] != 1 || Q->ne[3] != 1 || Q->nb[0] != sizeof(float)) return false;
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || K->ne[0] != FRT_FA_D || V->ne[0] != FRT_FA_D) return false;
    if (K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || K->nb[1] != V->nb[1] || K->nb[2] != V->nb[2]) return false;
    if (K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] || K->ne[3] != 1 || Q->ne[2] != K->ne[2] * FRT_FA_G) return false;
    if (Q->ne[2] > FRT_FA_HEADS_MAX) return false;
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->nb[0] != sizeof(half)) return false;
    if (sinks && (sinks->type != GGML_TYPE_F32 || ggml_nelements(sinks) != Q->ne[2])) return false;
    if (fa->type != GGML_TYPE_F32 || fa->ne[0] != FRT_FA_D || !ggml_is_contiguous(fa)) return false;
    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) fa->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) return false;
    const int n_kv = (int) K->ne[1];
    if (n_kv % FRT_FA_STEP != 0) return false;   // the cache view is padded to 256 rows
    if (((uintptr_t) K->data | (uintptr_t) V->data | (size_t) K->nb[1] | (size_t) K->nb[2]) & 15) return false;
    // chunking: up to two CTAs per SM over (chunk, KV head); each CTA walks
    // its chunk 64 rows at a time, so a second resident CTA hides the
    // stage -> compute latency (measured: 1x SM count is 4% slower at 16k
    // context, equal below 4k)
    static const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    static const int target_ctas = getenv("FRT_ATTNDEC_CTAS") ? atoi(getenv("FRT_ATTNDEC_CTAS")) : 2 * nsm;
    const int tiles = n_kv / FRT_FA_STEP;
    const int max_split = std::max(1, std::min(FRT_FA_MAX_SPLITS, target_ctas / (int) K->ne[2]));
    const int steps = (tiles + max_split - 1) / max_split;
    const int n_split = (tiles + steps - 1) / steps;
    pl = { Q, K, V, mask, sinks, scale, n_kv, steps, n_split };
    return true;
}

static bool frt_attndec_eligible(const ggml_cgraph * cgraph, int i) {
    frt_fa_plan pl;
    return frt_attndec_on() && frt_attndec_plan(cgraph, i, pl);
}

static bool frt_attndec_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    ggml_tensor * fa = cgraph->nodes[i];
    const bool fused = g_fa_pend.fa != nullptr && g_fa_pend.fa == fa;
    frt_fa_plan pl;
    if (!frt_attndec_on() || !frt_attndec_plan(cgraph, i, pl)) {
        GGML_ASSERT(!fused && "ATTNQKV deferred its epilogue to an attention node ATTNDEC cannot take");
        return false;
    }
    const frt_qkn_params qp = fused ? g_fa_pend.qp : frt_qkn_params{};
    const int qkn = fused ? (g_fa_pend.rope ? 2 : 1) : 0;
    g_fa_pend.fa = nullptr;
    const ggml_tensor * Q = pl.Q, * K = pl.K, * V = pl.V;
    const int n_kv = pl.n_kv, steps = pl.steps, n_split = pl.n_split;
    cudaStream_t stream = ctx.stream();
    const dim3 grid((unsigned) n_split, (unsigned) K->ne[2]);
    const ggml_cuda_kernel_launch_params lp(grid, (dim3) (FRT_FA_NW * WARP_SIZE), 0, stream);
    // first-step staging before the dependency wait pays off while the grid
    // fits one CTA per SM (measured: slower beyond that)
    static const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const bool prestage = n_split * (int) K->ne[2] <= nsm;
    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, lp, (const float *) Q->data, (int64_t) (Q->nb[2] / sizeof(float)),
            (const char *) K->data, (const char *) V->data, (int64_t) K->nb[1], (int64_t) K->nb[2],
            (const half *) pl.mask->data, pl.scale, n_kv, g_fa_ml, g_fa_o, n_split, (const float *) g_qkv, qp, steps, prestage);
    };
    static const bool pf = !(getenv("FRT_ATTNDEC_PF") && getenv("FRT_ATTNDEC_PF")[0] == '0');   // measured: prefetch on is faster
    switch (qkn) {
        case 0:  pf ? launch(frt_fa_decode<true, 0>) : launch(frt_fa_decode<false, 0>); break;
        case 1:  pf ? launch(frt_fa_decode<true, 1>) : launch(frt_fa_decode<false, 1>); break;
        default: pf ? launch(frt_fa_decode<true, 2>) : launch(frt_fa_decode<false, 2>); break;
    }
    // FRT_ATTNDEC_CHECK=k (test aid, eager mode only): every k-th call also runs
    // ggml's kernel on the same inputs and prints the max abs / rel difference
    static int check = getenv("FRT_ATTNDEC_CHECK") ? atoi(getenv("FRT_ATTNDEC_CHECK")) : 0;
    static int64_t ncall = 0;
    float * chk = nullptr;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (check > 0 && cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap == cudaStreamCaptureStatusNone &&
        (ncall++ % check) == 0) {
        CUDA_CHECK(cudaMalloc(&chk, ggml_nbytes(fa)));
    }
    ggml_cuda_kernel_launch(frt_fa_combine, ggml_cuda_kernel_launch_params((dim3) (unsigned) Q->ne[2], (dim3) FRT_FA_D, 0, stream),
        (const frt_fa_part *) g_fa_ml, (const float *) g_fa_o, pl.sinks ? (const float *) pl.sinks->data : (const float *) nullptr,
        chk ? chk : (float *) fa->data, (int64_t) (fa->nb[1] / sizeof(float)), n_split);
    if (chk) {
        ggml_cuda_flash_attn_ext(ctx, fa);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> a(ggml_nelements(fa)), b(ggml_nelements(fa));
        CUDA_CHECK(cudaMemcpy(a.data(), chk, ggml_nbytes(fa), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), fa->data, ggml_nbytes(fa), cudaMemcpyDeviceToHost));
        double md = 0, mr = 0, nb = 0;
        for (size_t k = 0; k < a.size(); ++k) { md = std::max(md, (double) fabsf(a[k] - b[k])); nb = std::max(nb, (double) fabsf(b[k])); }
        mr = md / std::max(nb, 1e-20);
        fprintf(stderr, "frt-attndec-check %s n_kv=%d splits=%d steps=%d qkn=%d max|d|=%.3g max|ref|=%.3g rel=%.3g\n", fa->name, n_kv, n_split, steps, qkn, md, nb, mr);
        CUDA_CHECK(cudaFree(chk));
    }
    *skip_count = 2;
    return true;
}

// ---- entry points called from the host's ggml-cuda.cu ---------------------

bool ggml_cuda_frt_flint_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    if (!frt_layer_enabled()) return false;
    static const bool cc_ok = ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL;
    if (!cc_ok) return false;
    if (frt_attnqkv_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_route_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_denseffn_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_lmhead_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_oproj_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_attndec_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_moeffn_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_amax_try(ctx, cgraph, i, skip_count)) return true;
    return false;
}

void ggml_cuda_frt_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    frt_dump_graph(cgraph);
    if (!frt_layer_enabled()) return;
    if (g_moe_glu == nullptr) {   // eager: this hook always runs before any capture begins
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc(&g_moe_glu, FRT_MOE_GLU_CAP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_qkv, FRT_QKV_ROWS_MAX * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_dense_glu, FRT_DFF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_lm_q8, (FRT_K_EMBD / QK8_1) * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&g_route_logits, 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_route_count, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(g_route_count, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&g_act_q8, (FRT_K_EMBD / QK8_1) * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.glu_q8, (FRT_MOE_GLU_CAP / QK8_1) * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.esc, FRT_MOE_MAX_USED * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.ids, FRT_MOE_MAX_USED * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.wts, FRT_MOE_MAX_USED * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_fa_ml, (size_t) FRT_FA_HEADS_MAX * FRT_FA_MAX_SPLITS * sizeof(frt_fa_part)));
        CUDA_CHECK(cudaMalloc(&g_fa_o, (size_t) FRT_FA_HEADS_MAX * FRT_FA_MAX_SPLITS * FRT_FA_D * sizeof(float)));
    }
}
