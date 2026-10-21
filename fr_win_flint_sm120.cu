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
    static_assert(512 % (NW * WARP_SIZE) == 0, "virtual block of 512 threads");
    constexpr int VPT = 512 / (NW * WARP_SIZE);   // virtual threads per real thread
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const float4 * x4 = (const float4 *) x;
    float4 xs[VPT];
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
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
    for (int k = 0; k < VPT; ++k) {
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
static __global__ void frt_attn_qkn(
        const float * qkv, const float * qn_w, const float * kn_w, const float eps,
        float * q_out, half * k_cache, half * v_cache, const int64_t * k_idx, const int64_t * v_idx,
        const size_t k_row_bytes, const size_t v_row_bytes,
        const int hq, const int hk, const int32_t * pos, const frt_rope_params rp, const int ncols) {
    ggml_cuda_pdl_lc();
    constexpr int D = 128;
    const int h = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (h >= hq + 2 * hk) {
        return;
    }
    ggml_cuda_pdl_sync();
    const bool is_v = h >= hq + hk;
    const bool is_k = !is_v && h >= hq;
    const float * xc = qkv + (size_t) h * D;
    if (is_v) {   // set_rows f32 -> f16 of the scaled V head
        half * dst = (half *) ((char *) v_cache + (size_t) v_idx[0] * v_row_bytes) + (size_t) (h - hq - hk) * D;
#pragma unroll
        for (int k = 0; k < D / WARP_SIZE; ++k) {
            dst[lane + k * WARP_SIZE] = __float2half(xc[lane + k * WARP_SIZE]);
        }
        return;
    }
    const float * mulc = is_k ? kn_w : qn_w;
    float out[4];
    int   oix[4];
    if constexpr (!rope) {
        const float4 * x4 = (const float4 *) xc;
        float4 xs[1];
        float tmp = 0.0f;
        xs[0] = x4[lane];
        tmp += xs[0].x*xs[0].x + xs[0].y*xs[0].y + xs[0].z*xs[0].z + xs[0].w*xs[0].w;
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(tmp / ncols + eps);
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
            const float xi = xc[wv * WARP_SIZE + lane];
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
        const float scale = rsqrtf(tot/ncols + eps);
        // pairs i0 = 2*v for virtual v < 64: lane covers v = lane and v = lane + 32
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int v  = lane + k * WARP_SIZE;
            const int i0 = 2 * v;
            const int ix0 = i0 / 2;
            const int ix1 = i0 / 2 + rp.n_dims / 2;
            const float x0 = scale * xc[ix0] * mulc[ix0];
            const float x1 = scale * xc[ix1] * mulc[ix1];
            const float theta_base  = pos[0]*powf(rp.theta_scale, i0/2.0f);
            const float freq_factor = 1.0f;
            float cos_theta;
            float sin_theta;
            {   // rope_yarn<true>
                const float theta_extrap = theta_base/freq_factor;
                float mscale = rp.attn_factor;
                float theta_interp = rp.freq_scale * theta_extrap;
                float theta = theta_interp;
                if (rp.ext_factor != 0.0f) {
                    const float y = (i0 / 2 - rp.corr0) / max(0.001f, rp.corr1 - rp.corr0);
                    const float ramp_mix = (1.0f - min(1.0f, max(0.0f, y))) * rp.ext_factor;
                    theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
                    mscale *= 1.0f + 0.1f * logf(1.0f / rp.freq_scale);
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
    if (is_k) {
        half * dst = (half *) ((char *) k_cache + (size_t) k_idx[0] * k_row_bytes) + (size_t) (h - hq) * D;
#pragma unroll
        for (int k = 0; k < 4; ++k) dst[oix[k]] = __float2half(out[k]);
    } else {
        float * dst = q_out + (size_t) h * D;
#pragma unroll
        for (int k = 0; k < 4; ++k) dst[oix[k]] = out[k];
    }
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
    constexpr int NW = 8;
    const int nrows = nq + 2 * nk;
    ggml_cuda_kernel_launch(frt_attn_qkv<NW>, ggml_cuda_kernel_launch_params((dim3) ((nrows + NW - 1) / NW), (dim3) (NW * WARP_SIZE), 0, stream),
        (const float *) x->data, (const float *) nw->data, eps,
        (const char *) wq->data, (const char *) wk->data, (const char *) wv->data, (size_t) wq->nb[1],
        (const float *) sq->data, (const float *) sk->data, (const float *) sv->data,
        g_qkv, nq, nk, 0.0f, FRT_K_EMBD);
    const int hq = nq / 128, hk = nk / 128, nh = hq + 2 * hk;
    constexpr int HW = 4;   // heads (warps) per CTA
    const ggml_cuda_kernel_launch_params lp2((dim3) ((nh + HW - 1) / HW), (dim3) (HW * WARP_SIZE), 0, stream);
    if (rope) {
        ggml_cuda_kernel_launch(frt_attn_qkn<true>, lp2,
            (const float *) g_qkv, (const float *) qnw->data, (const float *) knw->data, qeps,
            (float *) qfin->data, (half *) k_set->data, (half *) v_set->data,
            (const int64_t *) k_set->src[1]->data, (const int64_t *) v_set->src[1]->data,
            (size_t) k_set->nb[1], (size_t) v_set->nb[1], hq, hk, pos, rp, 128);
    } else {
        ggml_cuda_kernel_launch(frt_attn_qkn<false>, lp2,
            (const float *) g_qkv, (const float *) qnw->data, (const float *) knw->data, qeps,
            (float *) qfin->data, (half *) k_set->data, (half *) v_set->data,
            (const int64_t *) k_set->src[1]->data, (const int64_t *) v_set->src[1]->data,
            (size_t) k_set->nb[1], (size_t) v_set->nb[1], hq, hk, (const int32_t *) nullptr, rp, 128);
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
    ggml_tensor * add = cgraph->nodes[i + 2];
    const ggml_tensor * s = frt_mm_scale_src(smul, mm);
    if (!s || add->op != GGML_OP_ADD || add->src[0] != smul) return false;
    const ggml_tensor * resid = add->src[1];
    const int n_out = (int) w->ne[1];
    if (resid->type != GGML_TYPE_F32 || !ggml_is_contiguous(resid) || resid->ne[0] != n_out || ggml_nrows(resid) != 1) return false;
    if (add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || add->ne[0] != n_out || ggml_nrows(add) != 1) return false;
    if (!frt_node_ok(cgraph, i, GGML_OP_MUL_MAT, 1) || !frt_node_ok(cgraph, i + 1, GGML_OP_MUL, 1) || !frt_node_ok(cgraph, i + 2, GGML_OP_ADD, 0)) return false;
    // every CTA reads all of `a`; thread r reads resid[r] before writing out[r]
    if (frt_overlap(add, a) || (add->data != resid->data && frt_overlap(add, resid))) return false;
    constexpr int NW = 8;
    ggml_cuda_kernel_launch(frt_oproj<NW>, ggml_cuda_kernel_launch_params((dim3) ((n_out + NW - 1) / NW), (dim3) (NW * WARP_SIZE), 0, ctx.stream()),
        (const float *) a->data, (const char *) w->data, (size_t) w->nb[1], (const float *) s->data,
        (const float *) resid->data, (float *) add->data, n_out, 0.0f);
    *skip_count = 3;
    return true;
}

// ---- MoE routing (FRT window ROUTE) -----------------------------------------
// ffn RMSNorm -> router GEMV (BF16) -> sigmoid top-k with weight
// normalization: 3 stock launches -> 1. One CTA per router row as in
// mul_mat_vec_f<bf16, 256>, each recomputing the (exact) RMSNorm; the last CTA
// to finish runs the topk_moe selection on the staged logits. CTA 0 also
// writes the normalized activation that the expert GEMVs consume.
struct frt_moe_tail {
    block_q8_1 * glu_q8;     // n_used * n_ff/32 routed blocks, then n_ff_sh/32 shared
    float      * esc;        // e per routed slot
    int        * amax;       // per slot, f32 bits (values >= 0 or -FLT_MAX), reset by the last CTA
    unsigned   * cnt;        // per slot + 1 shared, reset by the last CTA
    int32_t    * ids;        // snapshot of the routing ids / weights: the layer output written
    float      * wts;        // by K2 may be allocated over the (dead after K2) graph tensors
};
static frt_moe_tail g_moe_tail = {};
static float * g_route_logits = nullptr;
static unsigned int * g_route_count = nullptr;
static block_q8_1 * g_act_q8 = nullptr;          // q8_1 of the ffn-normed activation (64 blocks)
static const void * g_act_q8_src = nullptr;      // tensor data the last matched ROUTE published for

template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_route(
        const float * x, const float * norm_w, const float eps, const int ncols, float * xn_out,
        const nv_bfloat16 * wr, const int64_t wr_stride, float * logits, unsigned int * count,
        int32_t * ids_out, float * w_out, const int n_expert_used, const float clamp_val, const float scale_val,
        block_q8_1 * xq_out, int32_t * ids_snap, float * w_snap) {
    static_assert(NW * WARP_SIZE == 256, "mul_mat_vec_f block size");
    constexpr int n_experts = 256;
    ggml_cuda_pdl_lc();
    __shared__ float xn[FRT_K_EMBD];
    __shared__ float red[32];
    __shared__ float buf_iw[WARP_SIZE];
    __shared__ bool last;
    const int tid = threadIdx.x, warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const int row = blockIdx.x;
    // router row (immutable): fetch before the dependency wait
    const nv_bfloat162 * x2 = (const nv_bfloat162 *) (wr + (int64_t) row * wr_stride);
    nv_bfloat162 wv[FRT_K_EMBD / 2 / 256];
#pragma unroll
    for (int k = 0; k < FRT_K_EMBD / 2 / 256; ++k) {
        wv[k] = x2[tid + 256 * k];
    }
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 0.0f;
    }
    ggml_cuda_pdl_sync();
    frt_rms_norm_2048<NW>(x, norm_w, eps, xn, red, ncols);
    __syncthreads();
    if (row == 0) {
        for (int c = tid; c < FRT_K_EMBD / 4; c += 256) {
            ((float4 *) xn_out)[c] = ((const float4 *) xn)[c];
        }
    }
    if (row < FRT_K_EMBD / QK8_1 / NW) {   // q8_1 of the activation for the expert GEMVs, one chunk per warp
        const int c = row * NW + warp;
        frt_q8_1_chunk(xn[c * QK8_1 + lane], lane, &xq_out[c]);
    }
    // mul_mat_vec_f<nv_bfloat16, float, 1, 256>
    const float2 * y2 = (const float2 *) xn;
    float sumf = 0.0f;
#pragma unroll
    for (int k = 0; k < FRT_K_EMBD / 2 / 256; ++k) {
        const int col2 = tid + 256 * k;
        const nv_bfloat162 tmpx = wv[k];
        const float2 tmpy = y2[col2];
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }
    sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    buf_iw[tid / WARP_SIZE] = sumf;
    __syncthreads();
    if (tid < WARP_SIZE) {
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }
    if (tid == 0) {
        logits[row] = sumf;
        __threadfence();
        last = atomicAdd(count, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || warp != 0) {
        return;
    }
    __threadfence();
    // topk_moe_cuda<256, false>: sigmoid, top-k, normalized weights
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
    constexpr int NW = 8;
    const ggml_cuda_kernel_launch_params lp((dim3) 256, (dim3) (NW * WARP_SIZE), 0, ctx.stream());
    ggml_cuda_kernel_launch(frt_route<NW>, lp,
        (const float *) m.x->data, (const float *) m.nw->data, m.eps, FRT_K_EMBD, (float *) m.fn->data,
        (const nv_bfloat16 *) m.wr->data, (int64_t) (m.wr->nb[1] / sizeof(nv_bfloat16)), g_route_logits, g_route_count,
        (int32_t *) m.as->data, (float *) m.dv->data, m.n_used, m.clamp_val, 1.0f, g_act_q8, g_moe_tail.ids, g_moe_tail.wts);
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
// shared; each CTA's NW outputs lie in one routed slot or in the shared expert.
// Tail: per routed slot, CTAs fold max|glu| into amax[slot]; the last CTA of
// the slot derives the rescale e (stock expression chain) and writes the q8_1
// chunks of glu / e; the last shared CTA writes the shared q8_1 chunks. Every
// down-projection CTA then reads these blocks instead of rebuilding them.

template <int NW, bool ACT_Q8>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_moe_gu(
        const float * act, const block_q8_1 * act_q8, const int32_t * ids, const float * wts,
        const char * gate_w, const char * up_w, const size_t e_stride, const size_t row_stride,
        const float * gate_sc, const float * up_sc,
        const char * shg_w, const char * shu_w, const float * shg_sc, const float * shu_sc,
        float * glu_out, const frt_moe_tail tl, const frt_moe_chain ch,
        const int n_used, const int n_ff, const int n_ff_sh, const float fzero) {
    ggml_cuda_pdl_lc();
    __shared__ block_q8_1 yq_sh[ACT_Q8 ? 1 : FRT_K_EMBD / QK8_1];
    __shared__ float cmax[NW];
    __shared__ bool last;
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
    float res = -FLT_MAX;
    if (active) {
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
            __threadfence();   // visible to the slot's last CTA
        }
    }
    // ---- tail: slot-wide max |glu| and last-CTA quantization
    const int o0 = blockIdx.x * NW;
    const bool cta_routed = o0 < n_routed;
    const int slot = cta_routed ? o0 / n_ff : n_used;
    if (lane == 0) {
        cmax[warp] = active ? fabsf(res) : -FLT_MAX;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (cta_routed) {
            float m = -FLT_MAX;
            for (int w = 0; w < NW; ++w) {
                m = fmaxf(m, cmax[w]);
            }
            atomicMax(&tl.amax[slot], __float_as_int(m));
        }
        __threadfence();
        const unsigned n_cta = cta_routed ? (unsigned) (n_ff / NW) : (unsigned) ((n_ff_sh + NW - 1) / NW);
        last = atomicAdd(&tl.cnt[slot], 1u) == n_cta - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    if (cta_routed) {
        const float m = __int_as_float(__ldcg(&tl.amax[slot]));
        float e = ch.s1 * m + ch.b1;
        e = logf(e);
        e = ch.s2 * e + ch.b2;
        e = ceilf(e);
        e = fminf(fmaxf(e, ch.cmin), ch.cmax);
        e = ch.s3 * e + ch.b3;
        e = expf(e);
        const int cpe = n_ff / QK8_1;
        for (int k = warp; k < cpe; k += NW) {
            const float x = __ldcg(glu_out + slot * n_ff + k * QK8_1 + lane) / e;
            frt_q8_1_chunk(x, lane, &tl.glu_q8[slot * cpe + k]);
        }
        if (threadIdx.x == 0) {
            tl.esc[slot] = e;
            tl.amax[slot] = __float_as_int(-FLT_MAX);
            tl.cnt[slot] = 0;
        }
    } else {
        const int base = n_used * (n_ff / QK8_1);
        for (int k = warp; k < n_ff_sh / QK8_1; k += NW) {
            frt_q8_1_chunk(__ldcg(glu_out + n_routed + k * QK8_1 + lane), lane, &tl.glu_q8[base + k]);
        }
        if (threadIdx.x == 0) {
            tl.cnt[slot] = 0;
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
    ggml_cuda_pdl_sync();
    const int cpe = n_ff / QK8_1;
    frt_nvb be[FRT_MOE_MAX_USED / 2];
#pragma unroll
    for (int p = 0; p < FRT_MOE_MAX_USED / 2; ++p) {
        if (2 * p < n_used) {
            const int slot = 2 * p + lane / 16;   // n_used is even (checked by the predicate)
            be[p] = frt_ld_nvb(down_w + (size_t) ids[slot] * de_stride + (size_t) row * drow_stride,
                               (lane % 16) / 2, frt_lane_iqs(lane));
        }
    }
    const frt_nvb bsh = frt_ld_nvb(shd_w + (size_t) row * shd_row_stride, lane / 2, frt_lane_iqs(lane));
    const float shs = shd_sc[0];
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
        for (int s = 0; s < n_used; ++s) {
            float r = de[s];          // mmvq epilogue: per-expert scale, zero bias
            r *= down_sc[ids[s]];
            r += fzero;
            if (s == 0) {
                sum = (r * esc[0]) * wts[0];
            } else {
                sum += (r * esc[s]) * wts[s];
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
    constexpr int NW1 = 8;
    const int n_out1 = m.n_used * m.n_ff + m.n_ff_sh;
    const ggml_cuda_kernel_launch_params lp1((dim3) ((n_out1 + NW1 - 1) / NW1), (dim3) (NW1 * WARP_SIZE), 0, stream);
    auto gu = act_q8 ? frt_moe_gu<NW1, true> : frt_moe_gu<NW1, false>;
    ggml_cuda_kernel_launch(gu, lp1,
        (const float *) m.act->data, (const block_q8_1 *) g_act_q8, (const int32_t *) m.ids->data, (const float *) m.wts->data,
        (const char *) m.gw->data, (const char *) m.uw->data, (size_t) m.gw->nb[2], (size_t) m.gw->nb[1],
        (const float *) m.gsc->data, (const float *) m.usc->data,
        (const char *) m.shgw->data, (const char *) m.shuw->data, (const float *) m.shgs->data, (const float *) m.shus->data,
        g_moe_glu, g_moe_tail, m.ch, m.n_used, m.n_ff, m.n_ff_sh, 0.0f);
    constexpr int NW2 = 8;
    ggml_cuda_kernel_launch(frt_moe_down<NW2>, ggml_cuda_kernel_launch_params((dim3) ((m.K + NW2 - 1) / NW2), (dim3) (NW2 * WARP_SIZE), 0, stream),
        (const block_q8_1 *) g_moe_tail.glu_q8, (const float *) g_moe_tail.esc, (const int32_t *) g_moe_tail.ids, (const float *) g_moe_tail.wts,
        (const char *) m.dw->data, (size_t) m.dw->nb[2], (size_t) m.dw->nb[1], (const float *) m.dsc->data,
        (const char *) m.shdw->data, (size_t) m.shdw->nb[1], (const float *) m.shds->data,
        (const float *) m.resid->data, (float *) m.l_out->data,
        m.n_used, m.n_ff, m.K, 0.0f);
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

// ---- entry points called from the host's ggml-cuda.cu ---------------------

bool ggml_cuda_frt_flint_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    if (!frt_layer_enabled()) return false;
    static const bool cc_ok = ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL;
    if (!cc_ok) return false;
    if (frt_attnqkv_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_route_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_oproj_try(ctx, cgraph, i, skip_count)) return true;
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
        CUDA_CHECK(cudaMalloc(&g_route_logits, 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_route_count, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(g_route_count, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&g_act_q8, (FRT_K_EMBD / QK8_1) * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.glu_q8, (FRT_MOE_GLU_CAP / QK8_1) * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.esc, FRT_MOE_MAX_USED * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.amax, FRT_MOE_MAX_USED * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.cnt, (FRT_MOE_MAX_USED + 1) * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(g_moe_tail.cnt, 0, (FRT_MOE_MAX_USED + 1) * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.ids, FRT_MOE_MAX_USED * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&g_moe_tail.wts, FRT_MOE_MAX_USED * sizeof(float)));
        {
            int init[FRT_MOE_MAX_USED];
            for (int k = 0; k < FRT_MOE_MAX_USED; ++k) { const float v = -FLT_MAX; memcpy(&init[k], &v, sizeof(v)); }
            CUDA_CHECK(cudaMemcpy(g_moe_tail.amax, init, sizeof(init), cudaMemcpyHostToDevice));
        }
    }
}
