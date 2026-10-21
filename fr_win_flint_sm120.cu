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

#include <cstdint>
#include <cstdio>
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

// mmvq row dot, NVFP4, K = 2048 (32 blocks): ggml launches 4 warps of which
// warps 0/1 cover blocks [0,16) / [16,32) (one block per lane pair), then sums
// warp partials 0+1+2+3 in order and reduces across the warp.
static __device__ __forceinline__ float frt_rowdot_k2048(const void * row, const block_q8_1 * y, const int lane, const float fzero) {
    const int kb = lane / 2, iqs = 4 * (lane % 2);
    const float p0 = vec_dot_nvfp4_q8_1(row, y + 2 * kb,        kb,      iqs);
    const float p1 = vec_dot_nvfp4_q8_1(row, y + 2 * (kb + 16), kb + 16, iqs);
    float t = fzero + p0;
    t += fzero + p1;
    t += fzero;
    t += fzero;
    return warp_reduce_sum(t);
}

// K = 1024 (16 blocks): only warp 0 active, one block per lane pair.
static __device__ __forceinline__ float frt_rowdot_k1024(const void * row, const block_q8_1 * y, const int lane, const float fzero) {
    const int kb = lane / 2, iqs = 4 * (lane % 2);
    float t = fzero + vec_dot_nvfp4_q8_1(row, y + 2 * kb, kb, iqs);
    t += fzero;
    t += fzero;
    t += fzero;
    return warp_reduce_sum(t);
}

// K = 512 (8 blocks): only lanes 0..15 of warp 0 active. Two independent rows
// per warp: half h = lane/16 handles (row_h, y_h); the stock butterfly's first
// step (xor 16) adds the idle half's +0, the rest stays within 16 lanes.
static __device__ __forceinline__ float frt_rowdot_k512_pair(const void * row, const block_q8_1 * y, const int lane, const float fzero) {
    const int l = lane % 16;
    const int kb = l / 2, iqs = 4 * (l % 2);
    float t = fzero + vec_dot_nvfp4_q8_1(row, y + 2 * kb, kb, iqs);
    t += fzero;
    t += fzero;
    t += fzero;
    t += fzero;   // xor-16 partner (idle lane) contributes +0
    return warp_reduce_sum<16>(t);
}

// ---- MoE FFN span (FRT window MOEFFN) --------------------------------------
// Replaces, for one decode token, everything after the router/top-k up to the
// layer output: routed gate/up MUL_MAT_ID + scales + SwiGLU, the activation
// rescale chain, routed down MUL_MAT_ID + scale, weighted expert sum, the
// shared expert (gate, up, SwiGLU, down, scales) and both residual adds:
// 14 stock launches -> 2.

struct frt_moe_chain { float s1, b1, s2, b2, cmin, cmax, s3, b3; };

static constexpr int FRT_MOE_MAX_USED = 8;
static constexpr int FRT_K_EMBD = 2048;

// K1: one warp per GLU output; outputs [0, n_used*n_ff) routed, then n_ff_sh shared.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_moe_gu(
        const float * act, const int32_t * ids,
        const char * gate_w, const char * up_w, const size_t e_stride, const size_t row_stride,
        const float * gate_sc, const float * up_sc,
        const char * shg_w, const char * shu_w, const float * shg_sc, const float * shu_sc,
        float * glu_out, const int n_used, const int n_ff, const int n_ff_sh, const float fzero) {
    ggml_cuda_pdl_lc();
    __shared__ block_q8_1 yq[FRT_K_EMBD / QK8_1];
    __shared__ int sids[FRT_MOE_MAX_USED];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    ggml_cuda_pdl_sync();
    for (int c = warp; c < FRT_K_EMBD / QK8_1; c += NW) {
        frt_q8_1_chunk(act[c * QK8_1 + lane], lane, &yq[c]);
    }
    if (threadIdx.x < n_used) {
        sids[threadIdx.x] = ids[threadIdx.x];
    }
    __syncthreads();
    const int o = blockIdx.x * NW + warp;
    const int n_routed = n_used * n_ff;
    if (o >= n_routed + n_ff_sh) {
        return;
    }
    const char * gw; const char * uw; float gs, us;
    const bool routed = o < n_routed;
    if (routed) {
        const int e = sids[o / n_ff];
        const int r = o % n_ff;
        gw = gate_w + (size_t) e * e_stride + (size_t) r * row_stride;
        uw = up_w   + (size_t) e * e_stride + (size_t) r * row_stride;
        gs = gate_sc[e];
        us = up_sc[e];
    } else {
        const int r = o - n_routed;
        gw = shg_w + (size_t) r * row_stride;
        uw = shu_w + (size_t) r * row_stride;
        gs = shg_sc[0];
        us = shu_sc[0];
    }
    const float tg = frt_rowdot_k2048(gw, yq, lane, fzero);
    const float tu = frt_rowdot_k2048(uw, yq, lane, fzero);
    if (lane == 0) {
        float res;
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
        glu_out[o] = res;
    }
}

// K2: per-CTA prologue (activation rescale + q8_1 of all routed / shared GLU
// rows into shared memory), then one warp per output row.
template <int NW>
static __global__ void __launch_bounds__(NW * WARP_SIZE, 1) frt_moe_down(
        const float * glu, const int32_t * ids, const float * wts,
        const char * down_w, const size_t de_stride, const size_t drow_stride, const float * down_sc,
        const char * shd_w, const size_t shd_row_stride, const float * shd_sc,
        const float * resid, float * out, const frt_moe_chain ch,
        const int n_used, const int n_ff, const int n_ff_sh, const int n_embd, const float fzero) {
    ggml_cuda_pdl_lc();
    __shared__ block_q8_1 yq_e[FRT_MOE_MAX_USED * 512 / QK8_1];
    __shared__ block_q8_1 yq_s[1024 / QK8_1];
    __shared__ float esc[FRT_MOE_MAX_USED];
    __shared__ float sw[FRT_MOE_MAX_USED];
    __shared__ int   sids[FRT_MOE_MAX_USED];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    ggml_cuda_pdl_sync();
    for (int s = warp; s < n_used; s += NW) {
        float m = -FLT_MAX;
        for (int j = lane; j < n_ff; j += WARP_SIZE) {
            m = fmaxf(m, fabsf(glu[s * n_ff + j]));
        }
        m = warp_reduce_max(m);
        if (lane == 0) {
            float e = ch.s1 * m + ch.b1;
            e = logf(e);
            e = ch.s2 * e + ch.b2;
            e = ceilf(e);
            e = fminf(fmaxf(e, ch.cmin), ch.cmax);
            e = ch.s3 * e + ch.b3;
            e = expf(e);
            esc[s] = e;
            sids[s] = ids[s];
            sw[s] = wts[s];
        }
    }
    __syncthreads();
    const int cpe = n_ff / QK8_1;
    const int n_ce = n_used * cpe;
    for (int c = warp; c < n_ce + n_ff_sh / QK8_1; c += NW) {
        if (c < n_ce) {
            const float x = glu[c * QK8_1 + lane] / esc[c / cpe];
            frt_q8_1_chunk(x, lane, &yq_e[c]);
        } else {
            const int cs = c - n_ce;
            frt_q8_1_chunk(glu[n_used * n_ff + cs * QK8_1 + lane], lane, &yq_s[cs]);
        }
    }
    __syncthreads();
    const int row = blockIdx.x * NW + warp;
    if (row >= n_embd) {
        return;
    }
    float de[FRT_MOE_MAX_USED];
#pragma unroll
    for (int p = 0; p < FRT_MOE_MAX_USED / 2; ++p) {
        if (2 * p < n_used) {
            const int slot = 2 * p + lane / 16;   // n_used is even (checked by the predicate)
            const char * rp = down_w + (size_t) sids[slot] * de_stride + (size_t) row * drow_stride;
            const float t = frt_rowdot_k512_pair(rp, yq_e + slot * cpe, lane, fzero);
            de[2 * p]     = __shfl_sync(0xffffffff, t, 0);
            de[2 * p + 1] = __shfl_sync(0xffffffff, t, 16);
        }
    }
    const float tsh = frt_rowdot_k1024(shd_w + (size_t) row * shd_row_stride, yq_s, lane, fzero);
    if (lane == 0) {
        float sum = 0.0f;
        for (int s = 0; s < n_used; ++s) {
            float r = de[s];          // mmvq epilogue: per-expert scale, zero bias
            r *= down_sc[sids[s]];
            r += fzero;
            if (s == 0) {
                sum = (r * esc[0]) * sw[0];
            } else {
                sum += (r * esc[s]) * sw[s];
            }
        }
        float sh = tsh;
        sh *= shd_sc[0];
        sh += fzero;
        float result = sum;           // k_bin_bcast with two fused addends
        result = result + sh;
        result = result + resid[row];
        out[row] = result;
    }
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
static bool frt_span_closed(const ggml_cgraph * cgraph, int a, int b, const ggml_tensor * keep0, const ggml_tensor * keep1) {
    for (int j = a; j < b; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n == keep0 || n == keep1) continue;
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

static float * g_moe_glu = nullptr;   // n_used*n_ff + n_ff_sh f32, allocated eagerly
static constexpr int FRT_MOE_GLU_CAP = FRT_MOE_MAX_USED * 512 + 1024;

// anchor: MUL_MAT_ID on blk.N.ffn_gate_exps.weight at M = 1.
static bool frt_moeffn_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static const bool on = frt_window_on("GGML_FLASHRT_NO_MOEFFN");
    if (!on || g_moe_glu == nullptr) return false;
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
    if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || n_used < 2 || n_used > FRT_MOE_MAX_USED || (n_used & 1)) return false;
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
    if (!frt_span_closed(cgraph, i, end, nullptr, nullptr)) return false;

    const frt_moe_chain ch = {
        ggml_get_op_params_f32(sc[0], 0), ggml_get_op_params_f32(sc[0], 1),
        ggml_get_op_params_f32(sc[1], 0), ggml_get_op_params_f32(sc[1], 1),
        ggml_get_op_params_f32(clmp, 0),  ggml_get_op_params_f32(clmp, 1),
        ggml_get_op_params_f32(sc[2], 0), ggml_get_op_params_f32(sc[2], 1) };

    cudaStream_t stream = ctx.stream();
    constexpr int NW1 = 8;
    const int n_out1 = n_used * (int) n_ff + (int) n_ff_sh;
    ggml_cuda_kernel_launch(frt_moe_gu<NW1>, ggml_cuda_kernel_launch_params((dim3) ((n_out1 + NW1 - 1) / NW1), (dim3) (NW1 * WARP_SIZE), 0, stream),
        (const float *) act->data, (const int32_t *) ids->data,
        (const char *) gw->data, (const char *) uw->data, (size_t) gw->nb[2], (size_t) gw->nb[1],
        (const float *) gsc->data, (const float *) usc->data,
        (const char *) shgw->data, (const char *) shuw->data, (const float *) shgs->data, (const float *) shus->data,
        g_moe_glu, n_used, (int) n_ff, (int) n_ff_sh, 0.0f);
    constexpr int NW2 = 8;
    ggml_cuda_kernel_launch(frt_moe_down<NW2>, ggml_cuda_kernel_launch_params((dim3) ((K + NW2 - 1) / NW2), (dim3) (NW2 * WARP_SIZE), 0, stream),
        (const float *) g_moe_glu, (const int32_t *) ids->data, (const float *) wts->data,
        (const char *) dw->data, (size_t) dw->nb[2], (size_t) dw->nb[1], (const float *) dsc->data,
        (const char *) shdw->data, (size_t) shdw->nb[1], (const float *) shds->data,
        (const float *) resid->data, (float *) l_out->data, ch,
        n_used, (int) n_ff, (int) n_ff_sh, (int) K, 0.0f);
    if (frt_stats_on()) {
        static long fires = 0;
        if ((++fires & 0xFFF) == 1) fprintf(stderr, "frt-stats moeffn fires=%ld span=%d\n", fires, end - i + 1);
    }
    *skip_count = end - i + 1;
    return true;
}

// ---- entry points called from the host's ggml-cuda.cu ---------------------

bool ggml_cuda_frt_flint_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    if (!frt_layer_enabled()) return false;
    static const bool cc_ok = ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL;
    if (!cc_ok) return false;
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
    }
}
