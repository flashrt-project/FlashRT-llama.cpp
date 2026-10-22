// Tensor-core core for the batched windows: NVFP4 weights x q8_1 activations
// with mma.sync.m16n8k16.s8 (one MMA per 16-value NVFP4 sub-block, so the
// integer dot products are exact; the UE4M3 sub-block scale and the fp16 q8_1
// block scale are applied to the int32 tile in fp32). A CTA owns 16 output
// rows and all T tokens of a tile; its NW warps split the K blocks and reduce
// through shared memory. Token q8_1 rows are staged in shared memory with a
// 16-byte pad so the B-fragment loads of the 8 token groups hit distinct banks.

static constexpr int FRT_M_ROWS = 16;

static __device__ __forceinline__ void frt_m_mma_s8(int (&s)[4], const uint32_t a0, const uint32_t a1, const uint32_t b) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %7, %7, %7};"
        : "=r"(s[0]), "=r"(s[1]), "=r"(s[2]), "=r"(s[3])
        : "r"(a0), "r"(a1), "r"(b), "r"(0));
}

static __device__ __forceinline__ void frt_m_cp16(void * smem_dst, const void * gmem_src) {
    const uint32_t d = (uint32_t) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(gmem_src));
}
static __device__ __forceinline__ void frt_m_cp_wait() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

// byte stride of a staged token row (q8_1 blocks of a KB-block row, + pad)
template <int KB> struct frt_m_tok { static constexpr int bytes = 2 * KB * (int) sizeof(block_q8_1); static constexpr int stride = bytes + 16; };

// The warp's K range of two weight rows (g, g+8) in registers: KBW blocks x
// (4 sub-block ints + 1 scale int) per row. Loaded once per CTA (immutable
// weights), reused for every token tile.
template <int KBW, bool GLU>
struct frt_m_wreg {
    uint32_t q[2][KBW][4], sc[2][KBW];
    uint32_t q2[GLU ? 2 : 1][GLU ? KBW : 1][4], sc2[GLU ? 2 : 1][GLU ? KBW : 1];
};

template <int KB, int KBW, int NW, bool GLU>
static __device__ __forceinline__ void frt_m_load(frt_m_wreg<KBW, GLU> & w, const char * wa, const char * wb,
        const char * wa2, const char * wb2, const int warp, const int lane) {
    constexpr int KBL = KB / NW;   // blocks of this warp's K range
    static_assert(KBL <= KBW, "register set holds the K range");
    const int c = lane % 4;
    const int bsel = (c & 1) * 4;
    const char * rows[2] = { wa, wb };
    const char * rows2[2] = { wa2, wb2 };
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int k = 0; k < KBL; ++k) {
            const char * b = rows[i] ? rows[i] + (size_t) (warp * KBL + k) * sizeof(block_nvfp4) : nullptr;
            w.sc[i][k] = b ? *(const uint32_t *) b : 0u;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                w.q[i][k][sb] = b ? *(const uint32_t *) (b + 4 + 8 * sb + bsel) : 0u;
            }
            if constexpr (GLU) {
                const char * b2 = rows2[i] ? rows2[i] + (size_t) (warp * KBL + k) * sizeof(block_nvfp4) : nullptr;
                w.sc2[i][k] = b2 ? *(const uint32_t *) b2 : 0u;
#pragma unroll
                for (int sb = 0; sb < 4; ++sb) {
                    w.q2[i][k][sb] = b2 ? *(const uint32_t *) (b2 + 4 + 8 * sb + bsel) : 0u;
                }
            }
        }
    }
}

// Accumulate the warp's K range over NTT n-tiles of 8 staged tokens.
template <int KB, int KBWS, int TS, int NW, int NTT, bool GLU>
static __device__ __forceinline__ void frt_m_rows(const frt_m_wreg<KBWS, GLU> & w,
        const uint8_t * tok_smem, const int nt_tiles, float (&acc)[NTT][4], float (&acc2)[GLU ? NTT : 1][4], const int warp, const int lane) {
    constexpr int KBW = KB / NW;
    static_assert(KB % NW == 0 && KBW <= KBWS, "warps split the K blocks");
    const int g = lane / 4, c = lane % 4;
    const bool hi = c >= 2;         // high nibbles (k 8..15 of the sub-block)
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
#pragma unroll
        for (int q = 0; q < 4; ++q) { acc[nt][q] = 0.0f; if constexpr (GLU) acc2[nt][q] = 0.0f; }
    }
#pragma unroll
    for (int k = 0; k < KBW; ++k) {
        const int kb = warp * KBW + k;
        // token block scales of this lane's two output columns (tokens 2c, 2c+1 of each n-tile), per 32-block
        float dt[NTT][2][2];
#pragma unroll
        for (int nt = 0; nt < NTT; ++nt) {
            if (nt < nt_tiles) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint8_t * b0 = tok_smem + (size_t) (nt * 8 + 2 * c) * TS + (2 * kb + h) * sizeof(block_q8_1);
                    dt[nt][h][0] = __half2float(*(const half *) b0);
                    dt[nt][h][1] = __half2float(*(const half *) (b0 + TS));
                }
            }
        }
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const int2 la = get_int_from_table_16((int) w.q[0][k][sb], kvalues_mxfp4);
            const int2 lb = get_int_from_table_16((int) w.q[1][k][sb], kvalues_mxfp4);
            const uint32_t a0 = (uint32_t) (hi ? la.y : la.x), a1 = (uint32_t) (hi ? lb.y : lb.x);
            const float dra = ggml_cuda_ue4m3_to_fp32((uint8_t) (w.sc[0][k] >> (8 * sb)));
            const float drb = ggml_cuda_ue4m3_to_fp32((uint8_t) (w.sc[1][k] >> (8 * sb)));
            uint32_t a02 = 0, a12 = 0; float dra2 = 0.0f, drb2 = 0.0f;
            if constexpr (GLU) {
                const int2 la2 = get_int_from_table_16((int) w.q2[0][k][sb], kvalues_mxfp4);
                const int2 lb2 = get_int_from_table_16((int) w.q2[1][k][sb], kvalues_mxfp4);
                a02 = (uint32_t) (hi ? la2.y : la2.x); a12 = (uint32_t) (hi ? lb2.y : lb2.x);
                dra2 = ggml_cuda_ue4m3_to_fp32((uint8_t) (w.sc2[0][k] >> (8 * sb)));
                drb2 = ggml_cuda_ue4m3_to_fp32((uint8_t) (w.sc2[1][k] >> (8 * sb)));
            }
            const int h = sb >> 1;
#pragma unroll
            for (int nt = 0; nt < NTT; ++nt) {
                if (nt >= nt_tiles) break;
                const uint8_t * tb = tok_smem + (size_t) (nt * 8 + g) * TS + (2 * kb + h) * sizeof(block_q8_1) + 4 + (sb & 1) * 16 + 4 * c;
                const uint32_t b = *(const uint32_t *) tb;
                int sacc[4];
                frt_m_mma_s8(sacc, a0, a1, b);
                const float t0 = dt[nt][h][0], t1 = dt[nt][h][1];
                acc[nt][0] += (float) sacc[0] * (dra * t0);
                acc[nt][1] += (float) sacc[1] * (dra * t1);
                acc[nt][2] += (float) sacc[2] * (drb * t0);
                acc[nt][3] += (float) sacc[3] * (drb * t1);
                if constexpr (GLU) {
                    int s2[4];
                    frt_m_mma_s8(s2, a02, a12, b);
                    acc2[nt][0] += (float) s2[0] * (dra2 * t0);
                    acc2[nt][1] += (float) s2[1] * (dra2 * t1);
                    acc2[nt][2] += (float) s2[2] * (drb2 * t0);
                    acc2[nt][3] += (float) s2[3] * (drb2 * t1);
                }
            }
        }
    }
}

// stage nt token rows (q8_1, TOK bytes each) into padded shared rows; tok_of(i) gives the source token index
template <int KB, int TS, int NW>
static __device__ __forceinline__ void frt_m_stage(uint8_t * tok_smem, const block_q8_1 * q8, const int ld_q8_blocks,
        const int * tok_idx /* smem, nt entries */, const int nt) {
    constexpr int TB = frt_m_tok<KB>::bytes;
    static_assert(TB % 16 == 0, "int4 rows");
    for (int cidx = threadIdx.x; cidx < nt * (TB / 16); cidx += NW * WARP_SIZE) {
        const int i = cidx / (TB / 16), q = cidx % (TB / 16);
        frt_m_cp16(tok_smem + (size_t) i * TS + q * 16, (const int4 *) (q8 + (int64_t) tok_idx[i] * ld_q8_blocks) + q);
    }
    frt_m_cp_wait();
}

// write the warp accumulators to red[warp][token][16] and reduce; returns per output element via callback
template <int NW, int NTT, bool GLU>
static __device__ __forceinline__ void frt_m_reduce_store(float * red, const float (&acc)[NTT][4], const float (&acc2)[GLU ? NTT : 1][4],
        const int nt_tiles, const int warp, const int lane) {
    const int g = lane / 4, c = lane % 4;
    constexpr int TT = NTT * 8;
    float * r = red + (size_t) warp * (GLU ? 2 : 1) * TT * FRT_M_ROWS;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
        if (nt >= nt_tiles) break;
        const int t0 = nt * 8 + 2 * c;
        r[(t0) * FRT_M_ROWS + g]         = acc[nt][0];
        r[(t0 + 1) * FRT_M_ROWS + g]     = acc[nt][1];
        r[(t0) * FRT_M_ROWS + g + 8]     = acc[nt][2];
        r[(t0 + 1) * FRT_M_ROWS + g + 8] = acc[nt][3];
        if constexpr (GLU) {
            float * r2 = r + TT * FRT_M_ROWS;
            r2[(t0) * FRT_M_ROWS + g]         = acc2[nt][0];
            r2[(t0 + 1) * FRT_M_ROWS + g]     = acc2[nt][1];
            r2[(t0) * FRT_M_ROWS + g + 8]     = acc2[nt][2];
            r2[(t0 + 1) * FRT_M_ROWS + g + 8] = acc2[nt][3];
        }
    }
}

template <int NW, int NTT, bool GLU>
static __device__ __forceinline__ float frt_m_sum(const float * red, const int tok, const int row, const int which) {
    constexpr int TT = NTT * 8;
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        s += red[((size_t) w * (GLU ? 2 : 1) + which) * TT * FRT_M_ROWS + tok * FRT_M_ROWS + row];
    }
    return s;
}

// ---- dense window kernel (segments, epilogues as frt_b_gemv) ---------------------------
template <int KB, int NW, int NTT, int EPI>
static __global__ void __launch_bounds__(NW * WARP_SIZE) frt_m_gemv(const frt_b_gemv_args a) {
    constexpr bool GLU = EPI == FRT_B_EPI_GLU;
    constexpr int TT = NTT * 8;
    constexpr int TS = frt_m_tok<KB>::stride;
    extern __shared__ __align__(16) uint8_t frt_m_smem[];
    uint8_t * tok_smem = frt_m_smem;
    float * red = (float *) (frt_m_smem + (size_t) TT * TS);
    __shared__ int tok_idx[TT];
    ggml_cuda_pdl_lc();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int g = lane / 4;
    const int r0 = blockIdx.x * FRT_M_ROWS;
    // row pointers / scales for rows g and g + 8 (immutable: resolved before the dependency wait)
    const char * wr[2] = { nullptr, nullptr }; const char * wr2[2] = { nullptr, nullptr };
    float sc[2] = { 0.0f, 0.0f }, sc2[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = r0 + g + 8 * i;
        if (r < a.n_rows) {
            int sg = 0, base = 0;
            while (sg + 1 < a.n_seg && r >= a.seg_end[sg]) { base = a.seg_end[sg]; ++sg; }
            wr[i] = a.w[sg] + (size_t) (r - base) * a.row_stride;
            sc[i] = a.s[sg][0];
            if constexpr (GLU) { wr2[i] = a.w2 + (size_t) r * a.row_stride; sc2[i] = a.s2[0]; }
        }
    }
    frt_m_wreg<KB / NW, GLU> wreg;
    frt_m_load<KB, KB / NW, NW, GLU>(wreg, wr[0], wr[1], wr2[0], wr2[1], warp, lane);
    ggml_cuda_pdl_sync();
    for (int t0 = 0; t0 < a.T; t0 += TT) {
        const int nt = min(TT, a.T - t0);
        const int nt_tiles = (nt + 7) / 8;
        if (threadIdx.x < nt) tok_idx[threadIdx.x] = a.tok_map ? a.tok_map[t0 + threadIdx.x] : t0 + threadIdx.x;
        __syncthreads();
        frt_m_stage<KB, TS, NW>(tok_smem, a.yq, 2 * KB, tok_idx, nt);
        __syncthreads();
        float acc[NTT][4], acc2[GLU ? NTT : 1][4];
        frt_m_rows<KB, KB / NW, TS, NW, NTT, GLU>(wreg, tok_smem, nt_tiles, acc, acc2, warp, lane);
        frt_m_reduce_store<NW, NTT, GLU>(red, acc, acc2, nt_tiles, warp, lane);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nt * FRT_M_ROWS; idx += NW * WARP_SIZE) {
            const int tok = idx / FRT_M_ROWS, row = idx % FRT_M_ROWS;
            const int r = r0 + row;
            if (r >= a.n_rows) continue;
            const int t = t0 + tok;
            float result = frt_m_sum<NW, NTT, GLU>(red, tok, row, 0);   // mmvq epilogue: scale, zero bias
            result *= sc[row / 8];
            result += a.fzero;
            if constexpr (EPI == FRT_B_EPI_RESID) {
                const int tk = tok_idx[tok];
                float o = result;
                o = o + a.resid[(int64_t) tk * a.ld_resid + r];
                result = o;
            } else if constexpr (GLU) {
                float up = frt_m_sum<NW, NTT, GLU>(red, tok, row, 1);
                up *= sc2[row / 8];
                up += a.fzero;
                up *= ggml_cuda_op_silu_single(result);
                result = up;
            }
            a.out[(int64_t) t * a.ld_out + r] = result;
        }
        __syncthreads();
    }
}

template <int KB, int NW, int NTT, int EPI>
static void frt_m_gemv_launch(const frt_b_gemv_args & a, cudaStream_t stream) {
    constexpr bool GLU = EPI == FRT_B_EPI_GLU;
    constexpr int TT = NTT * 8;
    constexpr size_t smem = (size_t) TT * frt_m_tok<KB>::stride + (size_t) NW * (GLU ? 2 : 1) * TT * FRT_M_ROWS * sizeof(float);
    static_assert(smem <= 99 * 1024, "sm_120: 99 KB of shared memory per block");
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(frt_m_gemv<KB, NW, NTT, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        attr = true;
    }
    const int cta = (a.n_rows + FRT_M_ROWS - 1) / FRT_M_ROWS;
    ggml_cuda_kernel_launch(frt_m_gemv<KB, NW, NTT, EPI>, ggml_cuda_kernel_launch_params((dim3) cta, (dim3) (NW * WARP_SIZE), smem, stream), a);
}

// ---- grouped MoE kernel (frt_b_moe_args) ------------------------------------------------
template <int KB, int KBS, int NW, int NTT, bool GLU>
static __global__ void __launch_bounds__(NW * WARP_SIZE) frt_m_moe(const frt_b_moe_args a) {
    constexpr int KBM = KB > KBS ? KB : KBS;
    constexpr int TT = NTT * 8;
    constexpr int TS = frt_m_tok<KBM>::stride;
    extern __shared__ __align__(16) uint8_t frt_m_smem[];
    uint8_t * tok_smem = frt_m_smem;
    float * red = (float *) (frt_m_smem + (size_t) TT * TS);
    __shared__ int tok_idx[TT], slot_k[TT];
    ggml_cuda_pdl_lc();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int g = lane / 4;
    const int tiles_r = a.tiles_r;
    const int n_act_max = (gridDim.x - (a.sh_rows + FRT_M_ROWS - 1) / FRT_M_ROWS) / tiles_r;
    const bool shared = (int) blockIdx.x >= n_act_max * tiles_r;
    const char * wr[2] = { nullptr, nullptr }; const char * wr2[2] = { nullptr, nullptr };
    float scv = 0.0f, scv2 = 0.0f;
    int s0 = 0, s1 = 0, n_rows, r0;
    const block_q8_1 * q8;
    int ld_q8;
    if (shared) {
        const int tile = blockIdx.x - n_act_max * tiles_r;
        r0 = tile * FRT_M_ROWS;
        n_rows = a.sh_rows;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = r0 + g + 8 * i;
            if (r < n_rows) {
                wr[i] = a.shw + (size_t) r * a.sh_row_stride;
                if constexpr (GLU) wr2[i] = a.shw2 + (size_t) r * a.sh_row_stride;
            }
        }
        scv = a.shsc[0];
        if constexpr (GLU) scv2 = a.shsc2[0];
        s0 = 0; s1 = a.T;
        q8 = a.q8 + a.sh_q8_off; ld_q8 = a.ld_q8;
    } else {
        r0 = (blockIdx.x % tiles_r) * FRT_M_ROWS;
        n_rows = a.n_rows;
        q8 = a.q8; ld_q8 = a.ld_q8;
    }
    ggml_cuda_pdl_sync();
    if (!shared) {
        const int ai = blockIdx.x / tiles_r;
        if (ai >= a.act[256]) {
            return;
        }
        const int e = a.act[ai];
        s0 = a.eoff[e];
        s1 = a.eoff[e + 1];
        const char * w = a.w + (size_t) e * a.e_stride;
        const char * w2 = a.w2 + (size_t) e * a.e_stride;
        scv = a.sc[e];
        if constexpr (GLU) scv2 = a.sc2[e];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = r0 + g + 8 * i;
            if (r < n_rows) {
                wr[i] = w + (size_t) r * a.row_stride;
                if constexpr (GLU) wr2[i] = w2 + (size_t) r * a.row_stride;
            }
        }
    }
    frt_m_wreg<KBM / NW, GLU> wreg;
    if (shared) frt_m_load<KBS, KBM / NW, NW, GLU>(wreg, wr[0], wr[1], wr2[0], wr2[1], warp, lane);
    else        frt_m_load<KB, KBM / NW, NW, GLU>(wreg, wr[0], wr[1], wr2[0], wr2[1], warp, lane);
    for (int j0 = s0; j0 < s1; j0 += TT) {
        const int nt = min(TT, s1 - j0);
        const int nt_tiles = (nt + 7) / 8;
        if (threadIdx.x < nt) {
            if (shared) { tok_idx[threadIdx.x] = j0 + threadIdx.x; slot_k[threadIdx.x] = 0; }
            else { const int sl = a.slots[j0 + threadIdx.x]; tok_idx[threadIdx.x] = sl >> 3; slot_k[threadIdx.x] = sl & 7; }
        }
        __syncthreads();
        // routed slot rows live at q8 + tok * ld_q8 + k * kq8: stage through a per-token base index (kq8 is a multiple of the row)
        if (shared) {
            frt_m_stage<KBS, TS, NW>(tok_smem, q8, ld_q8, tok_idx, nt);
        } else {
            constexpr int TB = frt_m_tok<KB>::bytes;
            for (int cidx = threadIdx.x; cidx < nt * (TB / 16); cidx += NW * WARP_SIZE) {
                const int i = cidx / (TB / 16), qq = cidx % (TB / 16);
                frt_m_cp16(tok_smem + (size_t) i * TS + qq * 16, (const int4 *) (q8 + (int64_t) tok_idx[i] * ld_q8 + (int64_t) slot_k[i] * a.kq8) + qq);
            }
            frt_m_cp_wait();
        }
        __syncthreads();
        float acc[NTT][4], acc2[GLU ? NTT : 1][4];
        if (shared) {
            frt_m_rows<KBS, KBM / NW, TS, NW, NTT, GLU>(wreg, tok_smem, nt_tiles, acc, acc2, warp, lane);
        } else {
            frt_m_rows<KB, KBM / NW, TS, NW, NTT, GLU>(wreg, tok_smem, nt_tiles, acc, acc2, warp, lane);
        }
        frt_m_reduce_store<NW, NTT, GLU>(red, acc, acc2, nt_tiles, warp, lane);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nt * FRT_M_ROWS; idx += NW * WARP_SIZE) {
            const int ti = idx / FRT_M_ROWS, row = idx % FRT_M_ROWS;
            const int r = r0 + row;
            if (r >= n_rows) continue;
            const int t = tok_idx[ti], k = slot_k[ti];
            const float d = frt_m_sum<NW, NTT, GLU>(red, ti, row, 0);
            float res;
            if constexpr (GLU) {
                const float d2 = frt_m_sum<NW, NTT, GLU>(red, ti, row, 1);
                if (shared) {   // two scaled GEMVs, then the GLU kernel
                    float gg = d;  gg *= scv;  gg += a.fzero;
                    float u = d2;  u *= scv2;  u += a.fzero;
                    res = ggml_cuda_op_silu_single(gg) * u;
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
                res = d;
            }
            if (shared) a.sh_out[(int64_t) t * a.sh_ld + r] = res;
            else        a.out[(int64_t) t * a.ld_out + (int64_t) k * a.k_out + r] = res;
        }
        __syncthreads();
    }
}

template <int KB, int KBS, int NW, int NTT, bool GLU>
static void frt_m_moe_launch(const frt_b_moe_args & a, const int n_act_max, cudaStream_t stream) {
    constexpr int KBM = KB > KBS ? KB : KBS;
    constexpr int TT = NTT * 8;
    constexpr size_t smem = (size_t) TT * frt_m_tok<KBM>::stride + (size_t) NW * (GLU ? 2 : 1) * TT * FRT_M_ROWS * sizeof(float);
    static_assert(smem <= 99 * 1024, "sm_120: 99 KB of shared memory per block");
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(frt_m_moe<KB, KBS, NW, NTT, GLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        attr = true;
    }
    const int grid = n_act_max * a.tiles_r + (a.sh_rows + FRT_M_ROWS - 1) / FRT_M_ROWS;
    ggml_cuda_kernel_launch(frt_m_moe<KB, KBS, NW, NTT, GLU>, ggml_cuda_kernel_launch_params((dim3) grid, (dim3) (NW * WARP_SIZE), smem, stream), a);
}
