cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flash-attention LANGUAGES CXX CUDA)

find_package(Git QUIET REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)

add_definitions("-DFLASH_ATTN_WITH_TORCH=0")

set(CUTLASS_3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)

set(FA2_SOURCES_CU
     flash_attn/src/cuda_utils.cu
     flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu
  )

add_library(flashattn SHARED
    capi/flash_attn.cu
    ${FA2_SOURCES_CU}
  )
target_include_directories(flashattn PRIVATE
    flash_attn
    ${CUTLASS_3_DIR}/include)

set(FA1_SOURCES_CU
    flash_attn_with_bias_and_mask/flash_attn_with_bias_mask.cu
    flash_attn_with_bias_and_mask/src/cuda_utils.cu
    flash_attn_with_bias_and_mask/src/fmha_fwd_with_mask_bias_hdim32.cu
    flash_attn_with_bias_and_mask/src/fmha_fwd_with_mask_bias_hdim64.cu
    flash_attn_with_bias_and_mask/src/fmha_fwd_with_mask_bias_hdim128.cu
    flash_attn_with_bias_and_mask/src/fmha_bwd_with_mask_bias_hdim32.cu
    flash_attn_with_bias_and_mask/src/fmha_bwd_with_mask_bias_hdim64.cu
    flash_attn_with_bias_and_mask/src/fmha_bwd_with_mask_bias_hdim128.cu
    flash_attn_with_bias_and_mask/src/utils.cu)

add_library(flashattn_with_bias_mask STATIC
    flash_attn_with_bias_and_mask/
    ${FA1_SOURCES_CU}
  )

target_include_directories(flashattn_with_bias_mask PRIVATE
    flash_attn_with_bias_and_mask/src
    flash_attn_with_bias_and_mask/cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(flashattn_with_bias_mask INTERFACE
    flash_attn_with_bias_and_mask)

target_link_libraries(flashattn flashattn_with_bias_mask)
  
add_dependencies(flashattn flashattn_with_bias_mask)

# ---- SM90 TMA+WGMMA kernels for flash_attn_with_bias_and_mask (H100/B100) ----
# Uses PaddleBox's CUTLASS v4.3.1 (extern_cutlass) which has full SM90 TMA/WGMMA support
set(CUTLASS_V4_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cutlass/src/extern_cutlass)
# FA3 Hopper headers (local copy with bias/mask/dbias modifications for SM90 kernels)
set(FA3_HOPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn_with_bias_and_mask/src/sm90/hopper)

set(FA1_SM90_SOURCES_CU
    flash_attn_with_bias_and_mask/src/sm90/fmha_fwd_with_mask_bias_sm90.cu
    flash_attn_with_bias_and_mask/src/sm90/fmha_bwd_with_mask_bias_sm90.cu)

add_library(flashattn_with_bias_mask_sm90 STATIC ${FA1_SM90_SOURCES_CU})

target_include_directories(flashattn_with_bias_mask_sm90 PRIVATE
    flash_attn_with_bias_and_mask/src
    flash_attn_with_bias_and_mask/src/sm90
    ${FA3_HOPPER_DIR}
    ${CUTLASS_V4_DIR}/include
    ${CUTLASS_V4_DIR}/tools/util/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(flashattn_with_bias_mask_sm90 INTERFACE
    flash_attn_with_bias_and_mask)

target_link_libraries(flashattn flashattn_with_bias_mask_sm90)
add_dependencies(flashattn flashattn_with_bias_mask_sm90)


if (NOT DEFINED NVCC_ARCH_BIN)
  message(FATAL_ERROR "NVCC_ARCH_BIN is not defined.")
endif()

if (NVCC_ARCH_BIN STREQUAL "")
  message(FATAL_ERROR "NVCC_ARCH_BIN is not set.")
endif()

STRING(REPLACE "-" ";" FA_NVCC_ARCH_BIN ${NVCC_ARCH_BIN})

set(FA_GENCODE_OPTION "SHELL:")
foreach(arch ${FA_NVCC_ARCH_BIN})
   if(${arch} GREATER_EQUAL 80)
     set(FA_GENCODE_OPTION "${FA_GENCODE_OPTION} -gencode arch=compute_${arch},code=sm_${arch}")
   endif()
endforeach()
# Ensure sm_90 is always present for H100 fallback compatibility (backward pass,
# or SM90 kernel returning false). CUDA 12.8+ only puts "80" in NVCC_ARCH_BIN.
string(FIND "${FA_GENCODE_OPTION}" "sm_90" _has_sm90)
if(_has_sm90 EQUAL -1)
  set(FA_GENCODE_OPTION "${FA_GENCODE_OPTION} -gencode arch=compute_90,code=sm_90")
  message(STATUS "flash-attn: added sm_90 gencode for H100 fallback compat")
endif()
# sm_90a/sm_100a native gencode (H100/B100) -- disabled for now, no P3 kernel yet.
# Uncomment when wgmma/tcgen05 kernel is ready (P3 optimization).
# string(REGEX MATCHALL "[0-9]+" _cuda_ver_parts ${CMAKE_CUDA_COMPILER_VERSION})
# list(GET _cuda_ver_parts 0 _cuda_major)
# if (_cuda_major GREATER 11)
#   if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 12.8)
#     set(FA_GENCODE_OPTION "${FA_GENCODE_OPTION} -gencode arch=compute_90a,code=sm_90a")
#     message(STATUS "flash-attn-with-bias-mask: adding sm_90a (H100 native)")
#   else()
#     set(FA_GENCODE_OPTION "${FA_GENCODE_OPTION} -gencode arch=compute_100a,code=sm_100a")
#     message(STATUS "flash-attn-with-bias-mask: adding sm_100a (B100 native)")
#   endif()
# endif()

target_compile_options(flashattn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -w
  -Xcompiler="-fPIC"
  -Xcompiler="-O3"
  -std=c++17
  -ccbin
  ${CMAKE_CXX_COMPILER}
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  --expt-relaxed-constexpr
  --expt-extended-lambda
  --use_fast_math
  "${FA_GENCODE_OPTION}"
  >)

target_compile_options(flashattn_with_bias_mask PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -w
  -Xcompiler="-fPIC"
  -Xcompiler="-O3"
  -std=c++17
  -ccbin
  ${CMAKE_CXX_COMPILER}
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  --expt-relaxed-constexpr
  --expt-extended-lambda
  --use_fast_math
  "${FA_GENCODE_OPTION}"
  >)

# SM90 target: compile with sm_90a native gencode (required for WGMMA/TMA instructions)
# Also add sm_100a for B200 (Blackwell SM100). SM100 supports WGMMA/TMA (forward compatible
# with SM90 ISA), so the same FA3 Hopper kernels run correctly on B200 with sm_100a CUBIN.
# Without sm_100a, B200 hits cudaErrorNoKernelImageForDevice because sm_90a CUBIN is
# SM90-specific and cannot execute on SM100.
# compute_100a requires CUDA 12.8+; skip it on older toolchains to avoid nvcc fatal error.
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "12.8")
  set(FA_SM90_GENCODE_OPTION "SHELL: -gencode arch=compute_90a,code=sm_90a \
  -gencode arch=compute_100a,code=sm_100a")
else()
  set(FA_SM90_GENCODE_OPTION "SHELL: -gencode arch=compute_90a,code=sm_90a")
endif()
target_compile_options(flashattn_with_bias_mask_sm90 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -w
  -Xcompiler="-fPIC"
  -Xcompiler="-O3"
  -std=c++17
  -ccbin
  ${CMAKE_CXX_COMPILER}
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  --expt-relaxed-constexpr
  --expt-extended-lambda
  --use_fast_math
  "${FA_SM90_GENCODE_OPTION}"
  >)

INSTALL(TARGETS flashattn
        LIBRARY DESTINATION "lib")

INSTALL(FILES capi/flash_attn.h DESTINATION "include")
