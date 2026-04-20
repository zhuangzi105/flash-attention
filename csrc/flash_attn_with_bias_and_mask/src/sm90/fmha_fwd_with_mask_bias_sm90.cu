// SM90 forward kernel: reuses FA3 Hopper TMA+WGMMA kernel implementation.
// This TU only includes FA3 headers (flash.h) -- NOT PaddleBox's fmha.h
// to avoid Qkv_params naming conflict.

// Clean interface (no dependency on either flash.h or fmha.h)
#include "fmha_sm90_interface.h"

// FA3 Hopper headers (resolved via include path to flash-attention/hopper/)
#include "named_barrier.hpp"    // FwdNamedBarriers
#include "flash.h"              // Flash_fwd_params, Qkv_params (FA3 version)
#include "tile_scheduler.hpp"   // TileSchedulerArguments, SingleTileScheduler
#include "tile_size.h"          // tile_size_fwd_sm90
#include "flash_fwd_kernel_sm90.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd.hpp"

// CUTLASS 3.x headers
#include <type_traits>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/kernel_launch.h"

using namespace cute;

// ---------------------------------------------------------------------------
// LSE rearrangement kernel: FA3 varlen [h, total_q] -> PaddleBox [b, h, msq]
// ---------------------------------------------------------------------------
__global__ void rearrange_lse_varlen_to_padded(
    float* __restrict__ dst,              // [b, h, msq] - PaddleBox layout
    const float* __restrict__ src,        // [h, total_q] - FA3 varlen layout
    const int* __restrict__ cu_seqlens_q, // [b+1]
    int h, int msq, int total_q, int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = b * h * msq;
    if (idx >= total_elems) return;

    int bidb = idx / (h * msq);
    int rem = idx % (h * msq);
    int bidh = rem / msq;
    int row = rem % msq;

    int seq_start = cu_seqlens_q[bidb];
    int seq_len = cu_seqlens_q[bidb + 1] - seq_start;
    if (row < seq_len) {
        dst[idx] = src[bidh * total_q + seq_start + row];
    }
    // else: leave existing value (-inf, pre-filled by framework)
}

// ---------------------------------------------------------------------------
// Simplified FA3 forward launch for SM90 + hdim128 + Varlen
// ---------------------------------------------------------------------------
template <typename Element, bool Is_causal>
bool run_fmha_fwd_sm90_hdim128(const SM90_fwd_params &p,
                                cudaStream_t stream,
                                void* workspace_ptr) {
    using ElementOut = Element;
    static constexpr int Arch = 90;
    static constexpr int kHeadDim = 128;
    static constexpr int kHeadDimV = 128;
    static constexpr int ClusterM = 1;
    static constexpr bool Is_local = false;
    static constexpr bool Has_softcap = false;
    static constexpr bool Varlen = true;
    static constexpr bool PagedKVNonTMA = false;
    static constexpr bool AppendKV = false;
    static constexpr bool HasQv = false;
    static constexpr bool PackGQA = false;
    static constexpr bool Split = false;
    static constexpr bool V_colmajor = false;
    static constexpr bool FP8_TransposeV = false;

    // Tile sizes from FA3
    static constexpr auto kBlockMN_RS_IntraWGOverlap =
        tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local,
                           sizeof(Element), V_colmajor, PagedKVNonTMA, Has_softcap);
    static constexpr int kBlockM = std::get<0>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kBlockN = std::get<1>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool MmaPV_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kStages = 2;

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
    using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;

    using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<
        kStages, ClusterShape, TileShape_MNK, kHeadDimV, Element, float,
        cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap,
        Varlen, PagedKVNonTMA, AppendKV, HasQv, MmaPV_is_RS, IntraWGOverlap,
        PackGQA, Split, V_colmajor>;

    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV, ClusterShape, ElementOut, cutlass::arch::Sm90,
        CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, Split, FP8_TransposeV>;

    // Use SingleTileScheduler for forward (one CTA per tile, grid = [num_blocks_m, h, b])
    // Persistent schedulers may exceed smem/register limits for forward with bias loading.
    using Scheduler = flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;

    using AttnKernel = flash::enable_sm90<
        flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    // --- Map SM90_fwd_params to FA3 argument structures ---
    int total_q = p.total_q;
    int total_k = p.total_k;
    int h = p.h;
    int h_k = p.h;  // MHA: h_k == h (no GQA in PaddleBox)
    int d = p.d;

    // Temp LSE buffer: FA3 writes [h, total_q], we rearrange to [b, h, msq] after
    float* lse_tmp = static_cast<float*>(workspace_ptr);

    // SingleTileScheduler doesn't need semaphore
    int* tile_count_semaphore = nullptr;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(p.q_ptr),
        {total_q, d, h, 1},  // shape_Q: varlen batch=1
        {p.q_row_stride, _1{}, p.q_head_stride, int64_t(0)},  // stride_Q
        static_cast<Element*>(p.k_ptr),
        {total_k, d, h_k, 1},  // shape_K
        {p.k_row_stride, _1{}, p.k_head_stride, int64_t(0)},  // stride_K
        static_cast<Element*>(p.v_ptr),
        d,  // headdim_v
        make_stride(p.v_row_stride, _1{}, p.v_head_stride, int64_t(0)),  // stride_V
        // K_new (not used)
        static_cast<Element const*>(nullptr),
        {0, d, h_k, 1},
        {int64_t(0), _1{}, int64_t(0), int64_t(0)},
        // V_new (not used)
        static_cast<Element const*>(nullptr),
        {int64_t(0), _1{}, int64_t(0), int64_t(0)},
        // Qv (not used)
        static_cast<Element const*>(nullptr),
        {int64_t(0), _1{}, int64_t(0), int64_t(0)},
        // Rotary (not used)
        static_cast<Element const*>(nullptr),
        {0, 0},  // shape_rotary
        {0, _1{}},  // stride_rotary_cos
        static_cast<Element const*>(nullptr),
        {0, _1{}},  // stride_rotary_sin
        false,  // is_rotary_interleaved
        // Page table (not used)
        static_cast<int*>(nullptr),
        {1, 0},  // shape_page_table
        {0, _1{}},  // stride_page_table
        // Softmax scale
        p.scale,
        // FP8 descale (not used)
        static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        {int64_t(0), int64_t(0)}, {int64_t(0), int64_t(0)}, {int64_t(0), int64_t(0)},
        // Window sizes
        -1,  // window_size_left
        Is_causal ? 0 : -1,  // window_size_right
        0,   // attention_chunk
        0.f, // softcap
        1,   // num_splits
        static_cast<int*>(nullptr),  // kv_batch_idx
        // cu_seqlens
        p.cu_seqlens_q, p.cu_seqlens_k, static_cast<int*>(nullptr),  // cu_seqlens_knew
        // seqused
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        // leftpad_k, seqlens_rotary
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        // PaddleBox compact varlen bias (layout=2)
        p.attn_bias_ptr,
        p.bias_seq_offsets,
        // PaddleBox explicit attention mask (padded rectangular layout)
        p.attn_mask_ptr,
        p.mask_head_mod_size,
        p.mask_seq_mod_size,
        p.mask_row_stride
    };

    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(p.o_ptr),
        {total_q, d, h, 1, 1},  // shape_O: (seqlen, headdim, nhead, batch, num_splits)
        {p.o_row_stride, _1{}, p.o_head_stride, int64_t(0), int64_t(0)},  // stride_O
        static_cast<float*>(nullptr),  // oaccum_ptr (no split)
        {int64_t(0), _1{}, int64_t(0), int64_t(0), int64_t(0)},  // stride_O_partial
        lse_tmp,  // LSE writes to temp buffer
        {_1{}, int64_t(total_q), int64_t(0), int64_t(0)},  // stride_LSE: [h, total_q] layout
        static_cast<float*>(nullptr),  // softmax_lseaccum_ptr
        {_1{}, int64_t(0), int64_t(0), int64_t(0)},  // stride_LSE_partial
        h_k,
        p.cu_seqlens_q, static_cast<int*>(nullptr)  // seqused_q
    };

    // Scheduler arguments
    int num_blocks_m = cutlass::ceil_div(p.seqlen_q, kBlockM);
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, h, p.b, 1 /*num_splits*/,
        1 /*qhead_per_khead*/,
        p.seqlen_q,  // max seqlen (fallback when cu_seqlens == nullptr)
        p.seqlen_k, d, d, (int)sizeof(Element),
        tile_count_semaphore,  // for DynamicPersistentTileScheduler (causal)
        p.cu_seqlens_q, static_cast<int*>(nullptr),  // seqused
        static_cast<int*>(nullptr),  // num_splits_dynamic_ptr
        static_cast<int*>(nullptr),  // num_m_blocks_ptr
        static_cast<int*>(nullptr),  // varlen_batch_idx_ptr
        static_cast<int*>(nullptr)   // num_nheads_in_l2_ptr
    };

    // --- Launch the FA3 SM90 kernel ---
    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, p.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;

    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
        cudaError_t attr_err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (attr_err != cudaSuccess) {
            return false;
        }
    }
    kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
    {
        cudaError_t launch_err = cudaPeekAtLastError();
        if (launch_err != cudaSuccess) {
            // Clear the sticky error so SM80 fallback can proceed cleanly
            cudaGetLastError();
            return false;
        }
    }

    // --- Rearrange LSE: FA3 [h, total_q] -> PaddleBox [b, h, max_seqlen_q] ---
    {
        int msq = p.seqlen_q;  // max_seqlen_q (already rounded)
        int total_elems = p.b * h * msq;
        int threads = 256;
        int blocks = (total_elems + threads - 1) / threads;
        rearrange_lse_varlen_to_padded<<<blocks, threads, 0, stream>>>(
            static_cast<float*>(p.softmax_lse_ptr),
            lse_tmp,
            p.cu_seqlens_q,
            h, msq, total_q, p.b
        );
    }

    return true;
}

// ---------------------------------------------------------------------------
// Entry point called from flash_attn_with_bias_mask.cu via interface
// ---------------------------------------------------------------------------
bool run_fmha_fwd_with_mask_bias_sm90(
    const SM90_fwd_params& p,
    cudaStream_t stream,
    void* workspace_ptr) {

    if (p.d != 128) return false;  // Only hdim128 supported

    // Task 5: both bias and mask are now supported on SM90.

    // Need workspace for LSE temp buffer
    if (workspace_ptr == nullptr) { return false; }

    if (p.is_bf16) {
        if (p.is_causal) {
            return run_fmha_fwd_sm90_hdim128<cutlass::bfloat16_t, true>(p, stream, workspace_ptr);
        } else {
            return run_fmha_fwd_sm90_hdim128<cutlass::bfloat16_t, false>(p, stream, workspace_ptr);
        }
    } else {
        if (p.is_causal) {
            return run_fmha_fwd_sm90_hdim128<cutlass::half_t, true>(p, stream, workspace_ptr);
        } else {
            return run_fmha_fwd_sm90_hdim128<cutlass::half_t, false>(p, stream, workspace_ptr);
        }
    }
}
