// SM90 backward kernel: reuses FA3 Hopper TMA+WGMMA backward kernel implementation.
// This TU only includes FA3 headers (flash.h) -- NOT PaddleBox's fmha.h
// to avoid Qkv_params naming conflict.

// Clean interface (no dependency on either flash.h or fmha.h)
#include "fmha_sm90_interface.h"

// FA3 Hopper headers (resolved via include path to flash-attention/hopper/)
#include "named_barrier.hpp"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_postprocess_kernel.h"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_bwd.hpp"
#include "flash_bwd_kernel_sm90.h"

// CUTLASS 3.x headers
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/kernel_launch.h"

using namespace cute;

// ---------------------------------------------------------------------------
// LSE rearrangement kernel: PaddleBox [b, h, msq] -> FA3 varlen padded layout
// Write raw LSE at cu_seqlens_q[bidb] offset (non-padded) to match preprocess reads.
// ---------------------------------------------------------------------------
__global__ void rearrange_lse_padded_to_varlen_bwd(
    float* __restrict__ dst,              // FA3 varlen padded [h, seqlen_q_rounded]
    const float* __restrict__ src,        // PaddleBox [b, h, msq]
    const int* __restrict__ cu_seqlens_q, // [b+1]
    int h, int msq, int batch,
    int kBlockM,                          // tile size for padding computation
    int64_t dst_head_stride) {            // = seqlen_q_rounded
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = batch * h * msq;
    if (idx >= total_elems) return;

    int bidb = idx / (h * msq);
    int rem = idx % (h * msq);
    int bidh = rem / msq;
    int row = rem % msq;

    int seqlen_cur = cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb];
    if (row >= seqlen_cur) return;

    // The preprocess kernel reads raw LSE using seqlen_info.offset (= cu_seqlens_q[bidb]),
    // NOT offset_padded. So we must write at cu_seqlens_q[bidb] + row.
    int offset_q = cu_seqlens_q[bidb];

    dst[bidh * dst_head_stride + offset_q + row] =
        src[bidb * h * msq + bidh * msq + row];
}

// ---------------------------------------------------------------------------
// Simplified FA3 backward launch for SM90 + hdim128 + Varlen
// ---------------------------------------------------------------------------
template <typename Element, bool Is_causal>
bool run_fmha_bwd_sm90_hdim128(const SM90_fwd_params &p,
                                cudaStream_t stream,
                                void* workspace_ptr) {
    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;
    static constexpr int kHeadDim = 128;

    // Tile sizes from flash_bwd_launch_template.h for SM90 hdim128
    static constexpr int kBlockM = Is_causal ? 64 : 80;
    static constexpr int kBlockN = 128;
    static constexpr bool Is_local = false;
    static constexpr bool Has_softcap = false;
    static constexpr bool Varlen = true;
    static constexpr bool Deterministic = false;
    static constexpr int Stages = 2;
    static constexpr int Stages_dO = 2;
    static constexpr int Stages_dS = 2;
    static constexpr bool SdP_swapAB = true;
    static constexpr bool dKV_swapAB = false;
    static constexpr bool dQ_swapAB = Is_causal ? false : true;
    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int AtomLayoutMSdP = 1;
    static constexpr int AtomLayoutNdKV = 2;
    static constexpr int AtomLayoutMdQ = 1;
    static constexpr bool V_in_regs = false;

    int const total_q = p.total_q;
    int const total_k = p.total_k;
    int const h = p.h;
    int const h_k = h;  // MHA: no GQA in PaddleBox
    int const d = p.d;
    int const d_rounded = 128;  // hdim128
    int const dv = d;
    int const batch = p.b;
    int const msq = p.seqlen_q;  // max_seqlen_q
    int const msk = p.seqlen_k;  // max_seqlen_k

    // Compute padded sizes for varlen (matches SeqlenInfo padding formula)
    // Upper bound: total_q + batch * kBlockM, then round up to kBlockM
    int64_t const seqlen_q_rounded = int64_t((total_q + batch * kBlockM + kBlockM - 1) / kBlockM) * kBlockM;

    // --- Workspace layout: [dQaccum | LSE_log2 | dPsum | raw_LSE | semaphores] ---
    char* ws = static_cast<char*>(workspace_ptr);
    uint64_t dqaccum_size = uint64_t(seqlen_q_rounded) * h * d_rounded * sizeof(float);
    uint64_t lse_size     = uint64_t(seqlen_q_rounded) * h * sizeof(float);
    uint64_t dpsum_size   = lse_size;
    uint64_t raw_lse_size = lse_size;

    ElementAccum* dq_accum_ptr = reinterpret_cast<ElementAccum*>(ws);
    float* lse_log2_ptr = reinterpret_cast<float*>(ws + dqaccum_size);
    float* dpsum_ptr    = reinterpret_cast<float*>(ws + dqaccum_size + lse_size);
    float* raw_lse_ptr  = reinterpret_cast<float*>(ws + dqaccum_size + lse_size + dpsum_size);
    // semaphores after raw_lse (only needed for deterministic, but allocate anyway)

    // --- Step 0: Rearrange LSE from PaddleBox [b,h,msq] to FA3 varlen padded ---
    {
        // Zero padding regions first
        cudaMemsetAsync(raw_lse_ptr, 0, raw_lse_size, stream);
        int total_elems = batch * h * msq;
        int threads = 256;
        int blocks = (total_elems + threads - 1) / threads;
        rearrange_lse_padded_to_varlen_bwd<<<blocks, threads, 0, stream>>>(
            raw_lse_ptr,
            static_cast<float*>(p.softmax_lse_ptr),
            p.cu_seqlens_q,
            h, msq, batch, kBlockM, seqlen_q_rounded
        );
    }

    // --- Step 1: Preprocess kernel ---
    // Computes dPsum = rowsum(dO * O), converts LSE -> LSE_log2, clears dQaccum
    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, ArchTag,
        /*Clear_dQaccum=*/true, Varlen>;

    int num_m_block = cute::ceil_div(msq, kBlockM);

    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(p.o_ptr),                         // ptr_O
        {total_q, dv, h, 1},                                          // shape_O
        {p.o_row_stride, _1{}, p.o_head_stride, int64_t(0)},          // stride_O
        static_cast<Element const*>(p.do_ptr),                        // ptr_dO
        {p.do_row_stride, _1{}, p.do_head_stride, int64_t(0)},        // stride_dO
        dpsum_ptr,                                                     // ptr_dPsum
        {int32_t(seqlen_q_rounded), h, 1},                            // shape_dPsum
        {_1{}, int64_t(seqlen_q_rounded), int64_t(0)},                // stride_dPsum
        raw_lse_ptr,                                                   // ptr_LSE (raw, rearranged)
        {_1{}, int64_t(seqlen_q_rounded), int64_t(0)},                // stride_LSE
        lse_log2_ptr,                                                  // ptr_LSE_log2
        {_1{}, int64_t(seqlen_q_rounded), int64_t(0)},                // stride_LSE_log2
        dq_accum_ptr,                                                  // ptr_dQaccum
        {int32_t(seqlen_q_rounded * d_rounded), h, 1},                // shape_dQaccum
        {_1{}, int64_t(seqlen_q_rounded) * d_rounded, int64_t(0)},    // stride_dQaccum
        batch,                                                         // num_batch
        static_cast<int*>(nullptr),                                    // dq_semaphore
        p.cu_seqlens_q,                                                // cu_seqlens
        static_cast<int*>(nullptr)                                     // seqused
    };

    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    dim3 grid_preprocess(num_m_block, h, batch);
    {
        auto kernel = cutlass::device_kernel<PreprocessKernel>;
        int smem = PreprocessKernel::SharedStorageSize;
        if (smem >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        kernel<<<grid_preprocess, PreprocessKernel::MaxThreadsPerBlock, smem, stream>>>(preprocess_params);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[SM90 bwd] preprocess launch failed: %s\n", cudaGetErrorString(err));
            cudaGetLastError();
            return false;
        }
    }

    // --- Step 2: Main backward kernel ---
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, _1, _1>;

    using CollectiveMainloop = flash::CollectiveMainloopBwdSm90<
        Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK,
        Element, ElementAccum, cutlass::arch::Sm90,
        Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
        SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups,
        AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>;

    using CollectiveEpilogue = flash::CollectiveEpilogueBwd<
        TileShape_MNK, Element, ArchTag,
        CollectiveMainloop::NumMmaThreads, Varlen, dKV_swapAB,
        NumMmaWarpGroups / AtomLayoutNdKV>;

    using Scheduler = std::conditional_t<Is_causal,
        flash::SingleTileBwdLPTScheduler<Varlen, kBlockN, false /*SPT*/>,
        flash::SingleTileScheduler<Varlen, false /*Split*/, false /*PackGQA*/, kBlockN>>;

    using AttnKernel = flash::enable_sm90<
        flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(p.q_ptr),
        {total_q, d, h, 1},                                        // shape_Q
        {p.q_row_stride, _1{}, p.q_head_stride, int64_t(0)},       // stride_Q
        static_cast<Element const*>(p.k_ptr),
        {total_k, d, h_k, 1},                                      // shape_K
        {p.k_row_stride, _1{}, p.k_head_stride, int64_t(0)},       // stride_K
        static_cast<Element const*>(p.v_ptr),
        {total_k, dv, h_k, 1},                                     // shape_V
        {p.v_row_stride, _1{}, p.v_head_stride, int64_t(0)},       // stride_V
        static_cast<Element const*>(p.do_ptr),
        {total_q, dv, h, 1},                                       // shape_dO
        {p.do_row_stride, _1{}, p.do_head_stride, int64_t(0)},     // stride_dO
        dq_accum_ptr,                                                // ptr_dQaccum
        {int32_t(seqlen_q_rounded * d_rounded), h, 1},              // shape_dQaccum
        {_1{}, int64_t(seqlen_q_rounded) * d_rounded, int64_t(0)},  // stride_dQaccum
        lse_log2_ptr,                                                // ptr_LSE_log2
        {int32_t(seqlen_q_rounded), h, 1},                          // shape_LSE
        {_1{}, int64_t(seqlen_q_rounded), int64_t(0)},              // stride_LSE_log2
        dpsum_ptr,                                                   // ptr_dPsum
        {_1{}, int64_t(seqlen_q_rounded), int64_t(0)},              // stride_dPsum
        p.scale,                                                     // softmax_scale
        -1,                                                          // window_size_left
        Is_causal ? 0 : -1,                                         // window_size_right
        0,                                                           // attention_chunk
        0.f,                                                         // softcap_val
        batch,                                                       // num_batch
        static_cast<int*>(nullptr),                                  // dq_semaphore
        p.cu_seqlens_q, p.cu_seqlens_k,                             // cu_seqlens
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),      // seqused
        // PaddleBox: bias, mask, dbias
        p.attn_bias_ptr,
        p.bias_seq_offsets,
        p.attn_mask_ptr,
        p.mask_head_mod_size,
        p.mask_seq_mod_size,
        p.mask_row_stride,
        p.attn_ds_ptr                                                // dbias output
    };

    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<Element*>(p.dk_ptr),
        {total_k, d, h, 1},                                        // shape_dK
        {p.dk_row_stride, _1{}, p.dk_head_stride, int64_t(0)},     // stride_dK
        static_cast<Element*>(p.dv_ptr),
        {total_k, dv, h, 1},                                       // shape_dV
        {p.dv_row_stride, _1{}, p.dv_head_stride, int64_t(0)},     // stride_dV
        batch, h,                                                    // num_batch, num_heads_q
        static_cast<int*>(nullptr),                                  // dk_semaphore
        static_cast<int*>(nullptr),                                  // dv_semaphore
        p.cu_seqlens_k,                                              // cu_seqlens
        static_cast<int*>(nullptr)                                   // seqused
    };

    // Allocate a dummy semaphore at end of workspace (SingleTileBwdLPTScheduler asserts non-null)
    uint64_t sema_offset = dqaccum_size + lse_size + dpsum_size + raw_lse_size;
    int* bwd_tile_count_semaphore = reinterpret_cast<int*>(ws + sema_offset);
    if constexpr (Is_causal) {
        cudaMemsetAsync(bwd_tile_count_semaphore, 0, sizeof(int), stream);
    }

    int num_blocks_n = cutlass::ceil_div(msk, kBlockN);
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_n, h, batch, 1 /*num_splits*/,
        h / h_k,                                                    // qhead_per_khead
        msk,                                                        // seqlen (outer loop = K dim)
        msk, d, dv, (int)sizeof(Element),                           // seqlen_k, headdim, headdim_v, element_size
        bwd_tile_count_semaphore,                                   // tile_count_semaphore
        p.cu_seqlens_k, static_cast<int*>(nullptr)                  // cu_seqlens, seqused
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, p.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;

    {
        auto kernel = cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            cudaError_t attr_err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            if (attr_err != cudaSuccess) {
                fprintf(stderr, "[SM90 bwd] cudaFuncSetAttribute failed: %s (smem=%d)\n",
                        cudaGetErrorString(attr_err), smem_size);
                return false;
            }
        }
        kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
        cudaError_t launch_err = cudaPeekAtLastError();
        if (launch_err != cudaSuccess) {
            fprintf(stderr, "[SM90 bwd] main kernel launch failed: %s "
                    "(grid=%dx%dx%d, block=%dx%dx%d, smem=%d)\n",
                    cudaGetErrorString(launch_err),
                    grid_dims.x, grid_dims.y, grid_dims.z,
                    block_dims.x, block_dims.y, block_dims.z, smem_size);
            cudaGetLastError();
            return false;
        }
    }

    // --- Step 3: Postprocess - convert dQaccum (fp32) to dQ (fp16/bf16) ---
    using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<
        TileShape_MK, Element, ElementAccum, ArchTag,
        AttnKernel::CollectiveMainloop::NumMmaThreads,
        typename AttnKernel::CollectiveMainloop::TiledMmadQ,
        AttnKernel::CollectiveMainloop::dQ_swapAB>;

    typename PostprocessKernel::Arguments postprocess_args {
        static_cast<ElementAccum const*>(dq_accum_ptr),                // ptr_dQaccum
        {int32_t(seqlen_q_rounded * d_rounded), h, 1},                // shape_dQaccum
        {_1{}, int64_t(seqlen_q_rounded) * d_rounded, int64_t(0)},    // stride_dQaccum
        static_cast<Element*>(p.dq_ptr),                               // ptr_dQ
        {total_q, d, h, 1},                                            // shape_dQ
        {p.dq_row_stride, _1{}, p.dq_head_stride, int64_t(0)},        // stride_dQ
        p.scale,                                                        // softmax_scale
        p.cu_seqlens_q,                                                 // cu_seqlens
        static_cast<int*>(nullptr)                                      // seqused
    };

    typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments(postprocess_args);
    dim3 grid_postprocess(num_m_block, h, batch);
    int smem_post = PostprocessKernel::SharedStorageSize;
    {
        auto kernel = cutlass::device_kernel<PostprocessKernel>;
        if (smem_post >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_post);
        }
        kernel<<<grid_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_post, stream>>>(postprocess_params);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[SM90 bwd] postprocess launch failed: %s\n", cudaGetErrorString(err));
            cudaGetLastError();
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Entry point called from flash_attn_with_bias_mask.cu via interface
// ---------------------------------------------------------------------------
bool run_fmha_bwd_with_mask_bias_sm90(
    const SM90_fwd_params& p,
    cudaStream_t stream,
    void* workspace_ptr) {

    if (p.d != 128) return false;  // Only hdim128 supported
    if (workspace_ptr == nullptr) return false;

    if (p.is_bf16) {
        if (p.is_causal) {
            return run_fmha_bwd_sm90_hdim128<cutlass::bfloat16_t, true>(p, stream, workspace_ptr);
        } else {
            return run_fmha_bwd_sm90_hdim128<cutlass::bfloat16_t, false>(p, stream, workspace_ptr);
        }
    } else {
        if (p.is_causal) {
            return run_fmha_bwd_sm90_hdim128<cutlass::half_t, true>(p, stream, workspace_ptr);
        } else {
            return run_fmha_bwd_sm90_hdim128<cutlass::half_t, false>(p, stream, workspace_ptr);
        }
    }
}
