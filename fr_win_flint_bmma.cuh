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
static __device__ __forceinline__ void frt_p_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> static __device__ __forceinline__ void frt_p_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

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
// MT m-tiles (16 rows each) per CTA; the NW warps form MT groups of NW/MT warps that split K.
template <int KB, int NW, int MT, int NTT, int EPI>
static __global__ void __launch_bounds__(NW * WARP_SIZE) frt_m_gemv(const frt_b_gemv_args a) {
    constexpr bool GLU = EPI == FRT_B_EPI_GLU;
    constexpr int TT = NTT * 8;
    constexpr int TS = frt_m_tok<KB>::stride;
    constexpr int WPM = NW / MT;              // warps per m-tile
    constexpr int ROWS = MT * FRT_M_ROWS;
    static_assert(NW % MT == 0, "warps per m-tile");
    constexpr size_t RED = (size_t) NW * (GLU ? 2 : 1) * TT * FRT_M_ROWS * sizeof(float);
    constexpr int NB = ((size_t) 2 * TT * TS + RED <= 99 * 1024) ? 2 : 1;   // token-tile buffers (double-buffered when they fit)
    extern __shared__ __align__(16) uint8_t frt_m_smem[];
    uint8_t * tok_buf = frt_m_smem;                               // [NB][TT * TS]
    float * red = (float *) (frt_m_smem + (size_t) NB * TT * TS);
    __shared__ int tok_idx[2][TT];
    ggml_cuda_pdl_lc();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int mt = warp / WPM, kw = warp % WPM;
    const int g = lane / 4;
    const int r0 = blockIdx.x * ROWS + mt * FRT_M_ROWS;
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
    frt_m_wreg<KB / WPM, GLU> wreg;
    if (!(a.dbg & 4)) frt_m_load<KB, KB / WPM, WPM, GLU>(wreg, wr[0], wr[1], wr2[0], wr2[1], kw, lane);
    ggml_cuda_pdl_sync();
    // tile 0 -> buffer 0
    {
        const int nt = min(TT, a.T);
        if (threadIdx.x < nt) tok_idx[0][threadIdx.x] = a.tok_map ? a.tok_map[threadIdx.x] : threadIdx.x;
        __syncthreads();
        if (!(a.dbg & 2)) frt_m_stage<KB, TS, NW>(tok_buf, a.yq, 2 * KB, tok_idx[0], nt);
        frt_p_commit();
    }
    int p = 0;
    for (int t0 = 0; t0 < a.T; t0 += TT, p ^= (NB - 1)) {
        const int nt = min(TT, a.T - t0);
        const int nt_tiles = (nt + 7) / 8;
        uint8_t * tok_smem = tok_buf + (size_t) p * TT * TS;
        const bool more = t0 + TT < a.T;
        if (NB == 1 && t0 > 0) {   // single buffer: this tile was not prefetched
            if (threadIdx.x < nt) tok_idx[0][threadIdx.x] = a.tok_map ? a.tok_map[t0 + threadIdx.x] : t0 + threadIdx.x;
            __syncthreads();
            if (!(a.dbg & 2)) frt_m_stage<KB, TS, NW>(tok_buf, a.yq, 2 * KB, tok_idx[0], nt);
            frt_p_commit();
        }
        if (NB == 2 && more) {   // next tile -> the other buffer (free since the previous iteration's final barrier)
            const int nt1 = min(TT, a.T - (t0 + TT));
            if (threadIdx.x < nt1) tok_idx[p ^ 1][threadIdx.x] = a.tok_map ? a.tok_map[t0 + TT + threadIdx.x] : t0 + TT + threadIdx.x;
            __syncthreads();
            if (!(a.dbg & 2)) frt_m_stage<KB, TS, NW>(tok_buf + (size_t) (p ^ 1) * TT * TS, a.yq, 2 * KB, tok_idx[p ^ 1], nt1);
            frt_p_commit();
            frt_p_wait<1>();
        } else {
            frt_p_wait<0>();
        }
        __syncthreads();
        float acc[NTT][4], acc2[GLU ? NTT : 1][4];
        if (!(a.dbg & 1)) frt_m_rows<KB, KB / WPM, TS, WPM, NTT, GLU>(wreg, tok_smem, nt_tiles, acc, acc2, kw, lane);
        else {
#pragma unroll
            for (int nt2 = 0; nt2 < NTT; ++nt2) {
#pragma unroll
                for (int q = 0; q < 4; ++q) { acc[nt2][q] = 0.0f; if constexpr (GLU) acc2[nt2][q] = 0.0f; }
            }
        }
        frt_m_reduce_store<NW, NTT, GLU>(red, acc, acc2, nt_tiles, warp, lane);
        __syncthreads();
        for (int idx = threadIdx.x; idx < nt * ROWS; idx += NW * WARP_SIZE) {
            const int tok = idx / ROWS, rr = idx % ROWS;
            const int mti = rr / FRT_M_ROWS, row = rr % FRT_M_ROWS;
            const int r = blockIdx.x * ROWS + rr;
            if (r >= a.n_rows) continue;
            const int t = t0 + tok;
            // this row's scale: resolved per row (segments)
            int sg = 0;
            while (sg + 1 < a.n_seg && r >= a.seg_end[sg]) ++sg;
            float result = 0.0f;
#pragma unroll
            for (int w = 0; w < WPM; ++w) {
                result += red[((size_t) (mti * WPM + w) * (GLU ? 2 : 1)) * TT * FRT_M_ROWS + tok * FRT_M_ROWS + row];
            }
            result *= a.s[sg][0];   // mmvq epilogue: scale, zero bias
            result += a.fzero;
            if constexpr (EPI == FRT_B_EPI_RESID) {
                const int tk = tok_idx[p][tok];
                float o = result;
                o = o + a.resid[(int64_t) tk * a.ld_resid + r];
                result = o;
            } else if constexpr (GLU) {
                float up = 0.0f;
#pragma unroll
                for (int w = 0; w < WPM; ++w) {
                    up += red[((size_t) (mti * WPM + w) * 2 + 1) * TT * FRT_M_ROWS + tok * FRT_M_ROWS + row];
                }
                up *= a.s2[0];
                up += a.fzero;
                up *= ggml_cuda_op_silu_single(result);
                result = up;
            }
            a.out[(int64_t) t * a.ld_out + r] = result;
        }
        __syncthreads();
    }
}

template <int KB, int NW, int MT, int NTT, int EPI>
static void frt_m_gemv_launch(const frt_b_gemv_args & a, cudaStream_t stream) {
    constexpr bool GLU = EPI == FRT_B_EPI_GLU;
    constexpr int TT = NTT * 8;
    constexpr size_t red = (size_t) NW * (GLU ? 2 : 1) * TT * FRT_M_ROWS * sizeof(float);
    constexpr size_t smem1 = (size_t) TT * frt_m_tok<KB>::stride;
    constexpr size_t smem = (2 * smem1 + red <= 99 * 1024) ? 2 * smem1 + red : smem1 + red;
    static_assert(smem <= 99 * 1024, "sm_120: 99 KB of shared memory per block");
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(frt_m_gemv<KB, NW, MT, NTT, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        attr = true;
    }
    const int cta = (a.n_rows + MT * FRT_M_ROWS - 1) / (MT * FRT_M_ROWS);
    ggml_cuda_kernel_launch(frt_m_gemv<KB, NW, MT, NTT, EPI>, ggml_cuda_kernel_launch_params((dim3) cta, (dim3) (NW * WARP_SIZE), smem, stream), a);
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
    if (!(a.dbg & 4)) {
        if (shared) frt_m_load<KBS, KBM / NW, NW, GLU>(wreg, wr[0], wr[1], wr2[0], wr2[1], warp, lane);
        else        frt_m_load<KB, KBM / NW, NW, GLU>(wreg, wr[0], wr[1], wr2[0], wr2[1], warp, lane);
    }
    for (int j0 = s0; j0 < s1; j0 += TT) {
        const int nt = min(TT, s1 - j0);
        const int nt_tiles = (nt + 7) / 8;
        if (threadIdx.x < nt) {
            if (shared) { tok_idx[threadIdx.x] = j0 + threadIdx.x; slot_k[threadIdx.x] = 0; }
            else { const int sl = a.slots[j0 + threadIdx.x]; tok_idx[threadIdx.x] = sl >> 3; slot_k[threadIdx.x] = sl & 7; }
        }
        __syncthreads();
        // routed slot rows live at q8 + tok * ld_q8 + k * kq8: stage through a per-token base index (kq8 is a multiple of the row)
        if (a.dbg & 2) {
        } else if (shared) {
            frt_m_stage<KBS, TS, NW>(tok_smem, q8, ld_q8, tok_idx, nt);
            frt_m_cp_wait();
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
        if (a.dbg & 1) {
#pragma unroll
            for (int nt2 = 0; nt2 < NTT; ++nt2) {
#pragma unroll
                for (int q = 0; q < 4; ++q) { acc[nt2][q] = 0.0f; if constexpr (GLU) acc2[nt2][q] = 0.0f; }
            }
        } else if (shared) {
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

// ---- persistent grouped MoE kernel ---------------------------------------------------
// Work items = (active expert, group of RT m-tiles) + shared-expert items. A
// CTA loops over items; the weights of the next item stream into shared
// memory with cp.async (two matrices x two items in flight) while the current
// item computes, so the small per-expert tiles neither pay CTA launch cost nor
// expose DRAM latency. Weight rows live in shared memory (padded rows), the
// A fragments are read from there.
//   gate/up: KB = 32 (K = 2048), RT = 1 (16 rows), 8 warps split K (KBW = 4)
//   down:    routed KB = 8, RT = 4 (64 rows, 2 warps per m-tile); shared KB = 16, RT = 2 (4 warps per m-tile)
static constexpr int FRT_P_NW  = 8;
static constexpr int FRT_P_WB  = 64 * 304;          // weight buffer bytes: max(16 x 1168, 64 x 304, 32 x 592)


struct frt_p_geom { int kb, rt, rs; };   // blocks per row, m-tiles per item, padded smem row stride

// copy rows [0, rt*16) of an item's weights (row bytes kb*36, global row stride) into a buffer
static __device__ __forceinline__ void frt_p_issue_rows(uint8_t * buf, const char * w, const size_t row_stride, const frt_p_geom g, const int n_rows_valid) {
    const int chunks = g.kb * 36 / 16;   // 16-byte chunks per row (kb*36 is a multiple of 16 for kb % 4 == 0)
    const int total = g.rt * 16 * chunks;
    for (int cidx = threadIdx.x; cidx < total; cidx += FRT_P_NW * WARP_SIZE) {
        const int r = cidx / chunks, q = cidx % chunks;
        if (r < n_rows_valid) {
            frt_m_cp16(buf + (size_t) r * g.rs + q * 16, w + (size_t) r * row_stride + q * 16);
        }
    }
}

static __device__ __forceinline__ void frt_p_issue_tokens(uint8_t * tok, const block_q8_1 * q8, const int ld_q8, const int kq8,
        const int32_t * slots, const int s0, const int nt, const bool shared, const int kb, const int ts) {
    const int chunks = kb * 72 / 16;   // 2*kb q8 blocks x 36 B
    for (int cidx = threadIdx.x; cidx < nt * chunks; cidx += FRT_P_NW * WARP_SIZE) {
        const int i = cidx / chunks, q = cidx % chunks;
        int tokn, k;
        if (shared) { tokn = s0 + i; k = 0; } else { const int sl = slots[s0 + i]; tokn = sl >> 3; k = sl & 7; }
        frt_m_cp16(tok + (size_t) i * ts + q * 16, (const char *) (q8 + (int64_t) tokn * ld_q8 + (int64_t) k * kq8) + q * 16);
    }
}

// MMA over the warp's K range of one m-tile with A fragments from a shared-memory weight buffer
template <int KBW, int TT, bool GLU>
static __device__ __forceinline__ void frt_p_rows(const uint8_t * wbuf, const uint8_t * wbuf2, const int rs, const int kb0,
        const uint8_t * tok, const int ts, const int nt_tiles, float (&acc)[4], float (&acc2)[GLU ? 1 : 1][4], const int mrow0, const int lane) {
    const int g0 = lane / 4, c = lane % 4;
    const int g = g0 < TT ? g0 : 0;                 // token rows beyond the tile read row 0 (results unused)
    const int ct0 = 2 * c < TT ? 2 * c : 0, ct1 = 2 * c + 1 < TT ? 2 * c + 1 : 0;
    const int bsel = (c & 1) * 4;
    const bool hi = c >= 2;
    const uint8_t * ra  = wbuf  + (size_t) (mrow0 + g0) * rs;
    const uint8_t * rb  = wbuf  + (size_t) (mrow0 + g0 + 8) * rs;
    const uint8_t * ra2 = wbuf2 + (size_t) (mrow0 + g0) * rs;
    const uint8_t * rb2 = wbuf2 + (size_t) (mrow0 + g0 + 8) * rs;
#pragma unroll
    for (int q = 0; q < 4; ++q) { acc[q] = 0.0f; if constexpr (GLU) acc2[0][q] = 0.0f; }
    if (nt_tiles < 1) return;
#pragma unroll
    for (int k = 0; k < KBW; ++k) {
        const int kb = kb0 + k;
        const uint32_t sca = *(const uint32_t *) (ra + kb * 36), scb = *(const uint32_t *) (rb + kb * 36);
        uint32_t sca2 = 0, scb2 = 0;
        if constexpr (GLU) { sca2 = *(const uint32_t *) (ra2 + kb * 36); scb2 = *(const uint32_t *) (rb2 + kb * 36); }
        float dt[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            dt[h][0] = __half2float(*(const half *) (tok + (size_t) ct0 * ts + (2 * kb + h) * sizeof(block_q8_1)));
            dt[h][1] = __half2float(*(const half *) (tok + (size_t) ct1 * ts + (2 * kb + h) * sizeof(block_q8_1)));
        }
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const int2 la = get_int_from_table_16(*(const int *) (ra + kb * 36 + 4 + 8 * sb + bsel), kvalues_mxfp4);
            const int2 lb = get_int_from_table_16(*(const int *) (rb + kb * 36 + 4 + 8 * sb + bsel), kvalues_mxfp4);
            const uint32_t a0 = (uint32_t) (hi ? la.y : la.x), a1 = (uint32_t) (hi ? lb.y : lb.x);
            const float dra = ggml_cuda_ue4m3_to_fp32((uint8_t) (sca >> (8 * sb)));
            const float drb = ggml_cuda_ue4m3_to_fp32((uint8_t) (scb >> (8 * sb)));
            const int h = sb >> 1;
            const uint8_t * tb = tok + (size_t) g * ts + (2 * kb + h) * sizeof(block_q8_1) + 4 + (sb & 1) * 16 + 4 * c;
            const uint32_t b = *(const uint32_t *) tb;
            int s[4];
            frt_m_mma_s8(s, a0, a1, b);
            acc[0] += (float) s[0] * (dra * dt[h][0]);
            acc[1] += (float) s[1] * (dra * dt[h][1]);
            acc[2] += (float) s[2] * (drb * dt[h][0]);
            acc[3] += (float) s[3] * (drb * dt[h][1]);
            if constexpr (GLU) {
                const int2 la2 = get_int_from_table_16(*(const int *) (ra2 + kb * 36 + 4 + 8 * sb + bsel), kvalues_mxfp4);
                const int2 lb2 = get_int_from_table_16(*(const int *) (rb2 + kb * 36 + 4 + 8 * sb + bsel), kvalues_mxfp4);
                const uint32_t a02 = (uint32_t) (hi ? la2.y : la2.x), a12 = (uint32_t) (hi ? lb2.y : lb2.x);
                const float dra2 = ggml_cuda_ue4m3_to_fp32((uint8_t) (sca2 >> (8 * sb)));
                const float drb2 = ggml_cuda_ue4m3_to_fp32((uint8_t) (scb2 >> (8 * sb)));
                int s2[4];
                frt_m_mma_s8(s2, a02, a12, b);
                acc2[0][0] += (float) s2[0] * (dra2 * dt[h][0]);
                acc2[0][1] += (float) s2[1] * (dra2 * dt[h][1]);
                acc2[0][2] += (float) s2[2] * (drb2 * dt[h][0]);
                acc2[0][3] += (float) s2[3] * (drb2 * dt[h][1]);
            }
        }
    }
}

// item geometry: routed (kb, rt) / shared (kbs, rts); warps per m-tile = NW / rt; KBW = kb / wpm == 4 in all uses
template <bool GLU>
static __global__ void __launch_bounds__(FRT_P_NW * WARP_SIZE, 1) frt_p_moe(const frt_b_moe_args a, const frt_p_geom gr, const frt_p_geom gs) {
    constexpr int NW = FRT_P_NW;
    constexpr int NMAT = GLU ? 2 : 1;
    constexpr int TT = GLU ? 4 : 8;                                   // tokens per tile
    constexpr int TS = GLU ? 2 * 32 * 36 + 16 : 2 * 16 * 36 + 16;     // token row stride (K <= 2048 / 1024)
    constexpr int TOKB = TT * TS;
    extern __shared__ __align__(16) uint8_t frt_p_smem[];
    uint8_t * wbuf = frt_p_smem;                            // [2 items][NMAT][FRT_P_WB]
    uint8_t * tbuf = frt_p_smem + (size_t) 2 * NMAT * FRT_P_WB; // [2 items][TOKB]; the current item's also holds the reduction
    __shared__ int tok_idx[2][TT], slot_k[2][TT];
    __shared__ int act_s[256], eoff_s[257];   // routing tables (one global read each; resolve() is then shared-memory only)
    ggml_cuda_pdl_lc();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int tiles_r = a.tiles_r;                           // items per routed expert
    const int sh_tiles = (a.sh_rows + gs.rt * 16 - 1) / (gs.rt * 16);
    ggml_cuda_pdl_sync();
    const int n_act = a.act[256];
    for (int i = threadIdx.x; i < 257; i += NW * WARP_SIZE) {
        eoff_s[i] = a.eoff[i];
        if (i < n_act) act_s[i] = a.act[i];
    }
    __syncthreads();
    const int n_items = n_act * tiles_r + sh_tiles;
    // item -> (shared, expert, row0, rows, s0, s1); the long shared-expert items come first
    auto resolve = [&](const int it, bool & shared, int & e, int & row0, int & nrows, int & s0, int & s1) {
        if (it < sh_tiles) {
            shared = true; e = -1;
            row0 = it * gs.rt * 16; nrows = min(gs.rt * 16, a.sh_rows - row0);
            s0 = 0; s1 = a.T;
        } else {
            const int ir = it - sh_tiles;
            shared = false; e = act_s[ir / tiles_r];
            row0 = (ir % tiles_r) * gr.rt * 16; nrows = min(gr.rt * 16, a.n_rows - row0);
            s0 = eoff_s[e]; s1 = eoff_s[e + 1];
        }
    };
    // issue the weight rows of item `it` into buffer slot p
    auto issue_w = [&](const int it, const int p) {
        if (a.dbg & 4) { frt_p_commit(); if constexpr (GLU) frt_p_commit(); return; }
        bool shared; int e, row0, nrows, s0, s1;
        resolve(it, shared, e, row0, nrows, s0, s1);
        const frt_p_geom g = shared ? gs : gr;
        uint8_t * b0 = wbuf + (size_t) p * NMAT * FRT_P_WB;
        if (shared) {
            frt_p_issue_rows(b0, a.shw + (size_t) row0 * a.sh_row_stride, a.sh_row_stride, g, nrows);
            frt_p_commit();
            if constexpr (GLU) { frt_p_issue_rows(b0 + FRT_P_WB, a.shw2 + (size_t) row0 * a.sh_row_stride, a.sh_row_stride, g, nrows); frt_p_commit(); }
        } else {
            frt_p_issue_rows(b0, a.w + (size_t) e * a.e_stride + (size_t) row0 * a.row_stride, a.row_stride, g, nrows);
            frt_p_commit();
            if constexpr (GLU) { frt_p_issue_rows(b0 + FRT_P_WB, a.w2 + (size_t) e * a.e_stride + (size_t) row0 * a.row_stride, a.row_stride, g, nrows); frt_p_commit(); }
        }
    };
    // issue the first token tile of item `it` into token slot p
    auto issue_t = [&](const int it, const int p) {
        bool shared; int e, row0, nrows, s0, s1;
        resolve(it, shared, e, row0, nrows, s0, s1);
        const frt_p_geom g = shared ? gs : gr;
        const int nt = min(TT, s1 - s0);
        if (a.dbg & 2) { frt_p_commit(); return; }
        if (threadIdx.x < nt) {
            if (shared) { tok_idx[p][threadIdx.x] = s0 + threadIdx.x; slot_k[p][threadIdx.x] = 0; }
            else { const int sl = a.slots[s0 + threadIdx.x]; tok_idx[p][threadIdx.x] = sl >> 3; slot_k[p][threadIdx.x] = sl & 7; }
        }
        frt_p_issue_tokens(tbuf + (size_t) p * TOKB, a.q8 + (shared ? a.sh_q8_off : 0), a.ld_q8, a.kq8, a.slots, s0, nt, shared, g.kb, TS);
        frt_p_commit();
    };
    int it = blockIdx.x;
    if (it < n_items) { issue_w(it, 0); issue_t(it, 0); }
    if (it + gridDim.x < n_items) { issue_w(it + gridDim.x, 1); issue_t(it + gridDim.x, 1); }
    int p = 0;
    for (; it < n_items; it += gridDim.x, p ^= 1) {
        bool shared; int e, row0, nrows, s0, s1;
        resolve(it, shared, e, row0, nrows, s0, s1);
        const frt_p_geom g = shared ? gs : gr;
        const int wpm = NW / g.rt;                  // warps per m-tile
        const int mt = warp / wpm, kw = warp % wpm; // this warp's m-tile and K slice
        const int kbw = g.kb / wpm;                 // == 4
        const float scv  = shared ? a.shsc[0] : a.sc[e];
        const float scv2 = GLU ? (shared ? a.shsc2[0] : a.sc2[e]) : 0.0f;
        const uint8_t * wb0 = wbuf + (size_t) p * NMAT * FRT_P_WB;
        const uint8_t * wb1 = wb0 + (GLU ? FRT_P_WB : 0);
        uint8_t * tok = tbuf + (size_t) p * TOKB;
        float * red = (float *) tok;                // reused after the MMAs
        const bool next_pending = it + gridDim.x < n_items;
        // groups pending: this item's (NMAT + 1) [+ next item's (NMAT + 1)]
        if (next_pending) frt_p_wait<NMAT + 1>(); else frt_p_wait<0>();
        __syncthreads();
        for (int j0 = s0; j0 < s1; j0 += TT) {
            const int nt = min(TT, s1 - j0);
            if (j0 > s0) {   // further token tiles of a long item (shared expert): staged synchronously
                if (threadIdx.x < nt) {
                    if (shared) { tok_idx[p][threadIdx.x] = j0 + threadIdx.x; slot_k[p][threadIdx.x] = 0; }
                    else { const int sl = a.slots[j0 + threadIdx.x]; tok_idx[p][threadIdx.x] = sl >> 3; slot_k[p][threadIdx.x] = sl & 7; }
                }
                frt_p_issue_tokens(tok, a.q8 + (shared ? a.sh_q8_off : 0), a.ld_q8, a.kq8, a.slots, j0, nt, shared, g.kb, TS);
                frt_p_commit();
                frt_p_wait<0>();   // the newest group: wait for everything (long items only)
                __syncthreads();
            }
            float acc[4], acc2[1][4];
            if (kbw == 4 && !(a.dbg & 1)) {
                frt_p_rows<4, TT, GLU>(wb0, wb1, g.rs, kw * 4, tok, TS, (nt + 7) / 8, acc, acc2, mt * 16, lane);
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) { acc[q] = 0.0f; acc2[0][q] = 0.0f; }
            }
            __syncthreads();   // all warps done reading the token tile: reuse it for the reduction
            {
                const int gg = lane / 4, c = lane % 4;
                float * r = red + (size_t) warp * (NMAT * TT * 16);
                const int t0 = 2 * c;
                if (t0 < TT) {   // MMA columns beyond the token tile hold nothing
                    r[t0 * 16 + gg] = acc[0]; r[(t0 + 1) * 16 + gg] = acc[1]; r[t0 * 16 + gg + 8] = acc[2]; r[(t0 + 1) * 16 + gg + 8] = acc[3];
                    if constexpr (GLU) {
                        float * r2 = r + TT * 16;
                        r2[t0 * 16 + gg] = acc2[0][0]; r2[(t0 + 1) * 16 + gg] = acc2[0][1]; r2[t0 * 16 + gg + 8] = acc2[0][2]; r2[(t0 + 1) * 16 + gg + 8] = acc2[0][3];
                    }
                }
            }
            __syncthreads();
            for (int idx = threadIdx.x; idx < nt * g.rt * 16; idx += NW * WARP_SIZE) {
                const int ti = idx / (g.rt * 16), rr = idx % (g.rt * 16);
                const int mtile = rr / 16, row = rr % 16;
                if (rr >= nrows) continue;
                const int r = row0 + rr;
                const int t = tok_idx[p][ti], k = slot_k[p][ti];
                float d = 0.0f, d2 = 0.0f;
                for (int w = mtile * wpm; w < (mtile + 1) * wpm; ++w) {
                    d += red[(size_t) w * (NMAT * TT * 16) + ti * 16 + row];
                    if constexpr (GLU) d2 += red[(size_t) w * (NMAT * TT * 16) + TT * 16 + ti * 16 + row];
                }
                float res;
                if constexpr (GLU) {
                    if (shared) { float gg = d; gg *= scv; gg += a.fzero; float u = d2; u *= scv2; u += a.fzero; res = ggml_cuda_op_silu_single(gg) * u; }
                    else { float result = d2; result *= scv2; result += a.fzero; float gate_value = d; gate_value *= scv; gate_value += a.fzero;
                           result *= ggml_cuda_op_silu_single(gate_value); res = result; }
                } else {
                    res = d;
                }
                if (shared) a.sh_out[(int64_t) t * a.sh_ld + r] = res;
                else        a.out[(int64_t) t * a.ld_out + (int64_t) k * a.k_out + r] = res;
            }
            __syncthreads();
        }
        // this item's buffers are free: prefetch item it + 2 * gridDim.x into slot p
        if (it + 2 * (int) gridDim.x < n_items) { issue_w(it + 2 * gridDim.x, p); issue_t(it + 2 * gridDim.x, p); }
    }
}

template <bool GLU>
static void frt_p_moe_launch(const frt_b_moe_args & a, const frt_p_geom gr, const frt_p_geom gs, cudaStream_t stream) {
    constexpr int NMAT = GLU ? 2 : 1;
    constexpr int TT = GLU ? 4 : 8, TS = GLU ? 2 * 32 * 36 + 16 : 2 * 16 * 36 + 16;
    constexpr size_t smem = (size_t) 2 * NMAT * FRT_P_WB + (size_t) 2 * TT * TS;
    static_assert(smem <= 99 * 1024, "sm_120 shared memory");
    static bool attr = false;
    static int n_sm = 0;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(frt_p_moe<GLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        int dev; CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
        attr = true;
    }
    static const int grid_mult = getenv("FRT_P_GRID") ? atoi(getenv("FRT_P_GRID")) : 1;
    ggml_cuda_kernel_launch(frt_p_moe<GLU>, ggml_cuda_kernel_launch_params((dim3) (n_sm * grid_mult), (dim3) (FRT_P_NW * WARP_SIZE), smem, stream), a, gr, gs);
}
