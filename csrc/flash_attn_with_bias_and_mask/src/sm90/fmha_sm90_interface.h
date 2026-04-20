// Clean interface between PaddleBox (fmha.h) and FA3 (flash.h) worlds.
// This header does NOT include either fmha.h or flash.h to avoid Qkv_params conflict.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct SM90_fwd_params {
    void* q_ptr;
    void* k_ptr;
    void* v_ptr;
    void* o_ptr;
    void* softmax_lse_ptr;

    int64_t q_row_stride, k_row_stride, v_row_stride, o_row_stride;
    int64_t q_head_stride, k_head_stride, v_head_stride, o_head_stride;

    int* cu_seqlens_q;
    int* cu_seqlens_k;

    int b, seqlen_q, seqlen_k, d;
    int h;
    int total_q, total_k;

    float scale;

    bool is_bf16;
    bool is_causal;

    void* attn_bias_ptr;
    void* attn_mask_ptr;

    // Bias layout=2 (compact varlen) fields
    int* bias_seq_offsets;   // prefix sum array [b+1], in elements (not per-head)

    // Mask addressing fields (padded rectangular layout from SM80)
    int mask_head_mod_size;  // number of heads in mask dim (head broadcasting)
    int mask_seq_mod_size;   // number of query rows in mask dim (row broadcasting)
    int mask_row_stride;     // physical last dim of mask tensor (= max_seqlen_k)

    int num_sm;

    // ---- Backward-specific fields ----
    void* do_ptr;           // dO input gradient
    void* dq_ptr;           // dQ output gradient
    void* dk_ptr;           // dK output gradient
    void* dv_ptr;           // dV output gradient

    int64_t do_row_stride, dq_row_stride, dk_row_stride, dv_row_stride;
    int64_t do_head_stride, dq_head_stride, dk_head_stride, dv_head_stride;

    void* dsoftmax_sum;     // softmax backward sum: dot(dO, O), [b, h, seqlen_q]
    void* attn_ds_ptr;      // dbias output (same dtype as Element), or nullptr

    int seqlen_q_rounded;   // seqlen_q rounded up to kBlockM multiple (for dQaccum sizing)
    int seqlen_k_rounded;   // seqlen_k rounded up to kBlockN multiple

    // dQ accumulator (fp32) workspace for backward
    void* dq_accum_ptr;     // fp32 dQ accumulator [h * total_q * d_rounded] or nullptr
    void* dk_accum_ptr;     // fp32 dK accumulator (GQA only), nullptr for MHA
    void* dv_accum_ptr;     // fp32 dV accumulator (GQA only), nullptr for MHA

    // Semaphores for deterministic backward
    int* dq_semaphore;
    int* dk_semaphore;
    int* dv_semaphore;

    int d_rounded;          // head dim rounded up (typically same as d for hdim128)
    int dv_rounded;         // value head dim rounded (same as d for hdim128)
};

// Returns true if SM90 kernel handled the request, false to fall back to SM80.
bool run_fmha_fwd_with_mask_bias_sm90(const SM90_fwd_params& params,
                                       cudaStream_t stream,
                                       void* workspace_ptr);

// Backward: returns true if SM90 kernel handled the request, false to fall back to SM80.
bool run_fmha_bwd_with_mask_bias_sm90(const SM90_fwd_params& params,
                                       cudaStream_t stream,
                                       void* workspace_ptr);
