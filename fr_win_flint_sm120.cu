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

// ---- entry points called from the host's ggml-cuda.cu ---------------------

bool ggml_cuda_frt_flint_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    if (!frt_layer_enabled()) return false;
    static const bool cc_ok = ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL;
    if (!cc_ok) return false;
    if (frt_amax_try(ctx, cgraph, i, skip_count)) return true;
    return false;
}

void ggml_cuda_frt_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    frt_dump_graph(cgraph);
}
