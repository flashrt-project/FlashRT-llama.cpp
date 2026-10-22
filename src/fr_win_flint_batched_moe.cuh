// Batched MoE windows (T > 1): included after the ROUTE helpers of fr_win_flint_sm120.cu.

// ---- MoE routing, T tokens ---------------------------------------------------------
// Router logits: CTA of 256 threads for RPC router rows, all T tokens (bf16 rows
// held in registers, the normalized activation read through L2).
template <int RPC>
static __global__ void __launch_bounds__(256) frt_b_router(const float * xn, const nv_bfloat16 * wr, const int64_t wr_stride,
        float * logits, const int T) {
    ggml_cuda_pdl_lc();
    constexpr int KP = FRT_K_EMBD / 2 / 256;   // bf16 pairs per thread per row
    constexpr int TU = 4;                      // tokens per iteration (loads hoisted)
    const int tid = threadIdx.x, warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const int row0 = blockIdx.x * RPC;
    nv_bfloat162 wv[RPC][KP];
#pragma unroll
    for (int r = 0; r < RPC; ++r) {
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) (wr + (int64_t) (row0 + r) * wr_stride);
#pragma unroll
        for (int k = 0; k < KP; ++k) {
            wv[r][k] = x2[tid + 256 * k];
        }
    }
    __shared__ float red[TU][8][RPC];
    ggml_cuda_pdl_sync();
    for (int t0 = 0; t0 < T; t0 += TU) {
        float2 yv[TU][KP];
#pragma unroll
        for (int u = 0; u < TU; ++u) {
            const int t = min(t0 + u, T - 1);
            const float2 * y2 = (const float2 *) (xn + (int64_t) t * FRT_K_EMBD);
#pragma unroll
            for (int k = 0; k < KP; ++k) {
                yv[u][k] = __ldcg(y2 + tid + 256 * k);
            }
        }
#pragma unroll
        for (int u = 0; u < TU; ++u) {
            float sum[RPC];
#pragma unroll
            for (int r = 0; r < RPC; ++r) {
                sum[r] = 0.0f;
#pragma unroll
                for (int k = 0; k < KP; ++k) {
                    ggml_cuda_mad(sum[r], wv[r][k].x, yv[u][k].x);
                    ggml_cuda_mad(sum[r], wv[r][k].y, yv[u][k].y);
                }
                sum[r] = warp_reduce_sum<WARP_SIZE>(sum[r]);
            }
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < RPC; ++r) {
                    red[u][warp][r] = sum[r];
                }
            }
        }
        __syncthreads();
        if (tid < TU * RPC) {
            const int u = tid / RPC, r = tid % RPC;
            const int t = t0 + u;
            if (t < T) {
                float v = red[u][0][r];
#pragma unroll
                for (int w = 1; w < 8; ++w) {
                    v += red[u][w][r];
                }
                logits[(int64_t) t * 256 + row0 + r] = v;
            }
        }
        __syncthreads();
    }
}

// top-k of every token (one warp each, ceil(T/8) CTAs); the last CTA to finish
// groups the T * n_used slots by expert: eoff[e] .. eoff[e+1] index slots[]
// (slot = t * 8 + k), act[0 .. n_active) lists the experts with tokens.
static __global__ void __launch_bounds__(256) frt_b_topk_group(const float * logits, const int T, const int n_used, const float clamp_val,
        int32_t * ids_out, const int64_t ids_stride, float * w_out, const int64_t w_stride,
        int32_t * ids_snap, float * w_snap, int32_t * eoff, int32_t * slots, int32_t * act, unsigned int * done,
        frt_t_item * items, int32_t * tbl_n) {
    ggml_cuda_pdl_lc();
    __shared__ int cnt[256], cur[256], wsum[8];
    __shared__ int n_act;
    __shared__ bool last;
    const int tid = threadIdx.x, warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    ggml_cuda_pdl_sync();
    const int t = blockIdx.x * 8 + warp;
    if (t < T) {
        frt_topk256_warp(logits + (int64_t) t * 256, n_used, clamp_val, 1.0f, lane,
                         ids_out + (int64_t) t * ids_stride, w_out + (int64_t) t * w_stride, ids_snap + t * 8, w_snap + t * 8);
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        last = atomicAdd(done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    cnt[tid] = 0;
    if (tid == 0) n_act = 0;
    __syncthreads();
    const int n_slots = T * n_used;
    for (int s = tid; s < n_slots; s += 256) {
        const int tt = s / n_used, k = s % n_used;
        atomicAdd(&cnt[__ldcg(ids_snap + tt * 8 + k)], 1);
    }
    __syncthreads();
    const int c = cnt[tid];
    int incl = c;
#pragma unroll
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
        const int v = __shfl_up_sync(0xffffffff, incl, d);
        if (lane >= d) incl += v;
    }
    if (lane == WARP_SIZE - 1) wsum[warp] = incl;
    __syncthreads();
    int base = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        if (w < warp) base += wsum[w];
    }
    const int excl = base + incl - c;
    eoff[tid] = excl;
    cur[tid] = excl;
    if (tid == 255) eoff[256] = excl + c;
    if (c > 0) {
        const int a = atomicAdd(&n_act, 1);
        act[a] = tid;
    }
    __syncthreads();
    if (tid == 0) { act[256] = n_act; *done = 0; }
    for (int s = tid; s < n_slots; s += 256) {
        const int tt = s / n_used, k = s % n_used;
        const int pos = atomicAdd(&cur[__ldcg(ids_snap + tt * 8 + k)], 1);
        slots[pos] = tt * 8 + k;
    }
    // work table for the bulk MoE kernel: shared-expert token tiles, then (active expert, token tile) entries
    {
        constexpr int TT = FRT_T_TT;
        const int n_sh = (T + TT - 1) / TT;
        if (tid < n_sh) items[tid] = { -1, tid * TT, min(TT, T - tid * TT) };
        // per active expert: ceil(cnt / TT) entries; prefix over the active list (<= 256) by one warp
        __shared__ int ent[256];
        if (tid < 256) ent[tid] = tid < n_act ? (cnt[act[tid]] + TT - 1) / TT : 0;   // cnt[] still holds the counts
        __syncthreads();
        int v = ent[tid];
        int inc = v;
#pragma unroll
        for (int d = 1; d < WARP_SIZE; d <<= 1) { const int o = __shfl_up_sync(0xffffffff, inc, d); if (lane >= d) inc += o; }
        __shared__ int wsum2[8];
        if (lane == WARP_SIZE - 1) wsum2[warp] = inc;
        __syncthreads();
        int base2 = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) if (w < warp) base2 += wsum2[w];
        const int excl2 = base2 + inc - v;
        if (tid < n_act) {
            const int e = act[tid];
            const int c0 = eoff[e], ccnt = cnt[e];
            for (int j = 0; j < v; ++j) items[n_sh + excl2 + j] = { e, c0 + j * TT, min(TT, ccnt - j * TT) };
        }
        if (tid == 255) { tbl_n[0] = n_sh; tbl_n[1] = excl2 + v; }
    }
}

// ---- grouped expert GEMVs, T tokens -------------------------------------------------
// CTA = (active expert, row tile) for the routed experts, or (row tile) for the
// shared expert (expert index -1: all T tokens in order). Rows' NVFP4 fragments
// stay in registers while the expert's tokens pass through shared memory.

template <int KB, int KBS, int RPW, int NW, int TT, bool GLU>
static __global__ void __launch_bounds__(NW * WARP_SIZE) frt_b_moe_gemv(const frt_b_moe_args a) {
    constexpr int KBM = KB > KBS ? KB : KBS;
    constexpr int NBW = (KBM + 15) / 16;
    constexpr int Q8T = 2 * KBM;
    static_assert((2 * KB * (int) sizeof(block_q8_1)) % 16 == 0 && (2 * KBS * (int) sizeof(block_q8_1)) % 16 == 0, "int4 staging");
    ggml_cuda_pdl_lc();
    __shared__ __align__(16) block_q8_1 yq[TT][Q8T];
    __shared__ int tok_s[TT], slot_s[TT];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int tiles_r = a.tiles_r;
    const int n_act_max = (gridDim.x - (a.sh_rows + NW * RPW - 1) / (NW * RPW)) / tiles_r;
    const bool shared = (int) blockIdx.x >= n_act_max * tiles_r;
    int e = -1, tile, kb;
    const char * w; const char * w2; const float * sc; const float * sc2;
    int s0 = 0, s1 = 0, n_rows;
    if (shared) {
        tile = blockIdx.x - n_act_max * tiles_r;
        kb = KBS;
        w = a.shw; w2 = a.shw2; sc = a.shsc; sc2 = a.shsc2;
        s0 = 0; s1 = a.T;
        n_rows = a.sh_rows;
    } else {
        tile = blockIdx.x % tiles_r;
        kb = KB;
        n_rows = a.n_rows;
    }
    const int r0 = tile * NW * RPW + warp * RPW;
    frt_nvb b[RPW][NBW];
    frt_nvb b2[GLU ? RPW : 1][NBW];
    float scv = 0.0f, scv2 = 0.0f;
    if (shared) {   // immutable, independent of the router: fetch before the dependency wait
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            if (r0 + i < n_rows) {
                frt_b_ld_row<NBW>(b[i], w + (size_t) (r0 + i) * a.sh_row_stride, lane, kb);
                if constexpr (GLU) frt_b_ld_row<NBW>(b2[i], w2 + (size_t) (r0 + i) * a.sh_row_stride, lane, kb);
            }
        }
        scv = sc[0];
        if constexpr (GLU) scv2 = sc2[0];
    }
    ggml_cuda_pdl_sync();
    if (!shared) {
        const int ai = blockIdx.x / tiles_r;
        if (ai >= a.act[256]) {
            return;
        }
        e = a.act[ai];
        s0 = a.eoff[e];
        s1 = a.eoff[e + 1];
        w = a.w + (size_t) e * a.e_stride;
        w2 = a.w2 + (size_t) e * a.e_stride;
        scv = a.sc[e];
        if constexpr (GLU) scv2 = a.sc2[e];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            if (r0 + i < n_rows) {
                frt_b_ld_row<NBW>(b[i], w + (size_t) (r0 + i) * a.row_stride, lane, kb);
                if constexpr (GLU) frt_b_ld_row<NBW>(b2[i], w2 + (size_t) (r0 + i) * a.row_stride, lane, kb);
            }
        }
    }
    const int tok_bytes = 2 * kb * (int) sizeof(block_q8_1);
    for (int j0 = s0; j0 < s1; j0 += TT) {
        const int nt = min(TT, s1 - j0);
        if (threadIdx.x < nt) {
            if (shared) {
                tok_s[threadIdx.x] = j0 + threadIdx.x;
                slot_s[threadIdx.x] = 0;
            } else {
                const int sl = a.slots[j0 + threadIdx.x];
                tok_s[threadIdx.x] = sl >> 3;
                slot_s[threadIdx.x] = sl & 7;
            }
        }
        __syncthreads();
        for (int c = threadIdx.x; c < nt * (tok_bytes / 16); c += NW * WARP_SIZE) {
            const int tt = c / (tok_bytes / 16), q = c % (tok_bytes / 16);
            const block_q8_1 * src = a.q8 + (int64_t) tok_s[tt] * a.ld_q8 + (shared ? a.sh_q8_off : slot_s[tt] * a.kq8);
            ((int4 *) yq[tt])[q] = __ldcg((const int4 *) src + q);
        }
        __syncthreads();
        for (int tt = 0; tt < nt; ++tt) {
            const block_q8_1 * y = yq[tt];
            const int t = tok_s[tt], k = slot_s[tt];
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const int r = r0 + i;
                if (r >= n_rows) continue;
                const float d = frt_b_rowdot<NBW>(b[i], y, lane, kb, a.fzero);
                float d2 = 0.0f;
                if constexpr (GLU) d2 = frt_b_rowdot<NBW>(b2[i], y, lane, kb, a.fzero);
                if (lane == 0) {
                    float res;
                    if constexpr (GLU) {
                        if (shared) {   // two scaled GEMVs, then the GLU kernel
                            float g = d;  g *= scv;  g += a.fzero;
                            float u = d2; u *= scv2; u += a.fzero;
                            res = ggml_cuda_op_silu_single(g) * u;
                        } else {        // fused mmvq epilogue: x = up, gate = gate, SwiGLU
                            float result = d2;
                            result *= scv2;
                            result += a.fzero;
                            float gate_value = d;
                            gate_value *= scv;
                            gate_value += a.fzero;
                            result *= ggml_cuda_op_silu_single(gate_value);
                            res = result;
                        }
                    } else {
                        res = d;   // unscaled: the combine applies the per-expert scale in the stock order
                    }
                    if (shared) {
                        a.sh_out[(int64_t) t * a.sh_ld + r] = res;
                    } else {
                        a.out[(int64_t) t * a.ld_out + (int64_t) k * a.k_out + r] = res;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Activation rescale (stock chain) + q8_1 of the routed GLU rows, q8_1 of the
// shared GLU row: one CTA per (token, slot) then one per token (shared).
// Routed rows are n_ff = 512 = 16 warps x 32 values.
static __global__ void __launch_bounds__(512) frt_b_moe_glu_q8(const float * glu, const int64_t ld_glu, const int n_used,
        const int n_ff, const int n_ff_sh, const frt_moe_chain ch, float * esc, block_q8_1 * q8_out, const int ld_q8) {
    ggml_cuda_pdl_lc();
    __shared__ float red[16];
    __shared__ float e_sh;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int T_used = gridDim.x / (n_used + 1);   // grid = T * (n_used + 1)
    const int t = blockIdx.x % T_used, s = blockIdx.x / T_used;
    ggml_cuda_pdl_sync();
    const float * row = glu + (int64_t) t * ld_glu;
    block_q8_1 * q8 = q8_out + (int64_t) t * ld_q8;
    if (s < n_used) {
        const float v = row[s * n_ff + threadIdx.x];
        float m = fmaxf(-FLT_MAX, fabsf(v));
        m = warp_reduce_max(m);
        if (lane == 0) red[warp] = m;
        __syncthreads();
        if (threadIdx.x == 0) {
            float a = red[0];
            for (int w = 1; w < 16; ++w) a = fmaxf(a, red[w]);
            float e = ch.s1 * a + ch.b1;
            e = logf(e);
            e = ch.s2 * e + ch.b2;
            e = ceilf(e);
            e = fminf(fmaxf(e, ch.cmin), ch.cmax);
            e = ch.s3 * e + ch.b3;
            e = expf(e);
            e_sh = e;
            esc[t * 8 + s] = e;
        }
        __syncthreads();
        frt_q8_1_chunk(v / e_sh, lane, &q8[s * (n_ff / QK8_1) + warp]);
    } else {
        const int base = n_used * (n_ff / QK8_1);
        for (int c = warp; c < n_ff_sh / QK8_1; c += 16) {
            frt_q8_1_chunk(row[n_used * n_ff + c * QK8_1 + lane], lane, &q8[base + c]);
        }
    }
}

// out[t][r] = sum_k ((part[t][k][r] * dsc[e]) * esc) * w  (slot order) + shared * shsc + resid
static __global__ void __launch_bounds__(256) frt_b_moe_combine(const float * part, const int64_t ld_part, const float * shpart,
        const int32_t * ids, const float * wts, const float * esc, const float * dsc, const float * shsc,
        const float * resid, float * out, const int n_used, const int n_embd, const float fzero) {
    ggml_cuda_pdl_lc();
    const int t = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float ds[8], ws[8], es[8];
    ggml_cuda_pdl_sync();
    if (threadIdx.x < n_used) {
        const int e = ids[t * 8 + threadIdx.x];
        ds[threadIdx.x] = dsc[e];
        ws[threadIdx.x] = wts[t * 8 + threadIdx.x];
        es[threadIdx.x] = esc[t * 8 + threadIdx.x];
    }
    __syncthreads();
    if (r >= n_embd) {
        return;
    }
    const float shs = shsc[0];
    float sum = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        float v = part[(int64_t) t * ld_part + (int64_t) s * n_embd + r];   // mmvq epilogue: per-expert scale, zero bias
        v *= ds[s];
        v += fzero;
        if (s == 0) {
            sum = (v * es[0]) * ws[0];
        } else {
            sum += (v * es[s]) * ws[s];
        }
    }
    float sh = shpart[(int64_t) t * n_embd + r];
    sh *= shs;
    sh += fzero;
    float result = sum;   // k_bin_bcast with two fused addends
    result = result + sh;
    result = result + resid[(int64_t) t * n_embd + r];
    out[(int64_t) t * n_embd + r] = result;
}

// MoE span: act (f32 rows) with OUR routing snapshot (ids/wts) -> l_out
static void frt_h_check_moe(cudaStream_t stream, const float * act, const int T, const int n_used, const int n_ff, const int n_ff_sh,
        const char * gw, const char * uw, const char * dw, const size_t e_stride, const size_t row_stride, const size_t de_stride, const size_t drow_stride,
        const float * gsc, const float * usc, const float * dsc,
        const char * shgw, const char * shuw, const char * shdw, const size_t sh_row_stride, const size_t shd_row_stride,
        const float * shgs, const float * shus, const float * shds, const frt_moe_chain & ch,
        const float * resid, const float * out) {
    static long calls = 0;
    if (T < 4 || ++calls > FRT_H_CALLS) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int K = FRT_K_EMBD;
    const auto as = frt_h_d2h(act, (size_t) T * K);
    const auto rs = frt_h_d2h(resid, (size_t) T * K);
    const auto os = frt_h_d2h(out, (size_t) T * K);
    const auto ids = frt_h_d2h_i32(g_b_ids, (size_t) T * 8);
    const auto wts = frt_h_d2h(g_b_wts, (size_t) T * 8);
    const auto gs = frt_h_d2h(gsc, 256);
    const auto us = frt_h_d2h(usc, 256);
    const auto ds = frt_h_d2h(dsc, 256);
    const auto sgs = frt_h_d2h(shgs, 1);
    const auto sus = frt_h_d2h(shus, 1);
    const auto sds = frt_h_d2h(shds, 1);
    const auto shg = frt_h_d2h_u8(shgw, (size_t) n_ff_sh * sh_row_stride);
    const auto shu = frt_h_d2h_u8(shuw, (size_t) n_ff_sh * sh_row_stride);
    const auto shd = frt_h_d2h_u8(shdw, (size_t) K * shd_row_stride);
    const int ld_glu = n_used * n_ff + n_ff_sh, ld_q8 = n_used * (n_ff / 32) + n_ff_sh / 32;
    const auto dglu = frt_h_d2h(g_b_glu, (size_t) T * ld_glu);
    const auto desc = frt_h_d2h(g_b_esc, (size_t) T * 8);
    frt_h_cmp c, cg, ce, cp, csp;
    long flips = 0, dbad = 0;
    GGML_UNUSED(as);
    for (const int t : frt_h_tokens(T)) {
        frt_h_q8 y; frt_h_q8_from_dev(g_b_q8 + (size_t) t * (K / 32), K, y);
        std::vector<double> sum(K, 0.0);
        for (int k = 0; k < n_used; ++k) {
            const int e = ids[t * 8 + k];
            const auto eg = frt_h_d2h_u8(gw + (size_t) e * e_stride, (size_t) n_ff * row_stride);
            const auto eu = frt_h_d2h_u8(uw + (size_t) e * e_stride, (size_t) n_ff * row_stride);
            const auto ed = frt_h_d2h_u8(dw + (size_t) e * de_stride, (size_t) K * drow_stride);
            std::vector<float> glu(n_ff);
            float amax = 0.0f;
            for (int r = 0; r < n_ff; ++r) {
                const float g = (float) frt_h_dot(eg.data() + (size_t) r * row_stride, y, K) * gs[e];
                const float u = (float) frt_h_dot(eu.data() + (size_t) r * row_stride, y, K) * us[e];
                glu[r] = u * frt_h_silu(g);
                amax = std::max(amax, fabsf(glu[r]));
                frt_h_acc(cg, dglu[(size_t) t * ld_glu + k * n_ff + r], glu[r]);
            }
            float esc = ch.s1 * amax + ch.b1; esc = logf(esc); esc = ch.s2 * esc + ch.b2; esc = ceilf(esc);
            esc = fminf(fmaxf(esc, ch.cmin), ch.cmax); esc = ch.s3 * esc + ch.b3; esc = expf(esc);
            frt_h_acc(ce, desc[t * 8 + k], esc);
            esc = desc[t * 8 + k];   // continue with the device rescale (a ceil boundary would otherwise fork the chain)
            std::vector<float> gd(n_ff);
            for (int r = 0; r < n_ff; ++r) gd[r] = dglu[(size_t) t * ld_glu + k * n_ff + r] / esc;
            frt_h_q8 yg; frt_h_q8_from_dev(g_b_gluq8 + (size_t) t * ld_q8 + k * (n_ff / 32), n_ff, yg);
            frt_h_quant_cmp("moe", gd.data(), yg, n_ff, flips, dbad);
            {   // raw per-slot down output (g_b_part) vs host
                const auto dp = frt_h_d2h(g_b_part + ((size_t) t * n_used + k) * K, K);
                for (int r = 0; r < K; ++r) {
                    const double ref = frt_h_dot(ed.data() + (size_t) r * drow_stride, yg, n_ff);
                    frt_h_acc(cp, dp[r], ref);
                }
            }
            for (int r = 0; r < K; ++r) {
                sum[r] += frt_h_dot(ed.data() + (size_t) r * drow_stride, yg, n_ff) * ds[e] * esc * wts[t * 8 + k];
            }
        }
        std::vector<float> sglu(n_ff_sh);
        for (int r = 0; r < n_ff_sh; ++r) {
            const float g = (float) frt_h_dot(shg.data() + (size_t) r * sh_row_stride, y, K) * sgs[0];
            const float u = (float) frt_h_dot(shu.data() + (size_t) r * sh_row_stride, y, K) * sus[0];
            sglu[r] = frt_h_silu(g) * u;
            frt_h_acc(cg, dglu[(size_t) t * ld_glu + n_used * n_ff + r], sglu[r]);
        }
        frt_h_q8 ys; frt_h_q8_from_dev(g_b_gluq8 + (size_t) t * ld_q8 + n_used * (n_ff / 32), n_ff_sh, ys);
        frt_h_quant_cmp("moe-sh", dglu.data() + (size_t) t * ld_glu + n_used * n_ff, ys, n_ff_sh, flips, dbad);
        {   // raw shared down output (g_b_shpart) vs host
            const auto dp = frt_h_d2h(g_b_shpart + (size_t) t * K, K);
            for (int r = 0; r < K; ++r) {
                const double ref = frt_h_dot(shd.data() + (size_t) r * shd_row_stride, ys, n_ff_sh);
                frt_h_acc(csp, dp[r], ref);
            }
        }
        for (int r = 0; r < K; ++r) {
            const double ref = sum[r] + frt_h_dot(shd.data() + (size_t) r * shd_row_stride, ys, n_ff_sh) * sds[0] + rs[(size_t) t * K + r];
            frt_h_acc(c, os[(size_t) t * K + r], ref);
        }
    }
    frt_h_report("moe-glu", calls, cg);
    frt_h_report("moe-esc", calls, ce);
    frt_h_report("moe-part", calls, cp);
    frt_h_report("moe-shpart", calls, csp);
    frt_h_report("moeffn", calls, c);
    fprintf(stderr, "frt-href moe-q8     call %ld: %ld int8 flips, %ld scale mismatches (device glu/esc requantized on host)\n", calls, flips, dbad);
}
