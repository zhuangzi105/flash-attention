// Copyright (c) 2022, Tri Dao.

#include "fmha_bwd_with_mask_bias_launch_template.h"

bool run_fmha_bwd_with_mask_bias_hdim128(FMHA_dgrad_params &params, cudaStream_t stream) {
    bool status = true;
    auto dprops = GetDeviceProperties(-1);
    FP16_SWITCH(params.is_bf16, ([&] {
        if (dprops->major >= 9) {
            // H100 (sm_90) / B100 (sm_100): larger tile (Cq=256) for better compute intensity.
            // WARPS_N=8 + shmem 232KB comfortably fits blocksize_c=256; V kept in smem (0x100u)
            // to avoid register spill on the 3-output (dQ/dK/dV) backward kernel.
            using Kernel_traits = FMHA_kernel_traits<256, 128, 16, 1, 8, 0x100u, elem_type>;
            status = run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        } else {
            // A100 and older: existing fixed path
            using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 8, 0x100u, elem_type>;
            status = run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        }
    }));
    return status;
}
