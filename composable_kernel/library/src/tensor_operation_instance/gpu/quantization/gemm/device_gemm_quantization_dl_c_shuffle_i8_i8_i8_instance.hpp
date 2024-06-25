// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_quantization_common.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_dl.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename OutElementOp>
using device_gemm_quantization_dl_c_shuffle_i8_i8_i8_km_kn_mn_instances = std::tuple<
    // clang-format off
        //####################|      A|      B|          Ds|      E|  AData|  BData| AccData|      DsData|   EData|           A|           B|           CDE|           GEMM| Block|  MPer|  NPer|  KPer| K1|   M1Per|  N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
        //####################| Layout| Layout|      Layout| Layout|   Type|   Type|    Type|        Type|    Type| Elementwise| Elementwise|   Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM| Thread| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
        //####################|       |       |            |       |       |       |        |            |        |   Operation|   Operation|     Operation|               |      |      |      |      |   |        |       |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
        //####################|       |       |            |       |       |       |        |            |        |            |            |              |               |      |      |      |      |   |        |       |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        DeviceGemmMultipleD_Dl<    Col,    Row, Empty_Tuple,    Row, int8_t, int8_t, int32_t, Empty_Tuple,  int8_t, PassThrough, PassThrough,  OutElementOp,     MNKPadding,   256,   128,   128,    16,  4,       4,      4,      1,       S<8, 2>,       S<8, 2>,      S<2, 1, 4, 4>,      S<8, 1,  32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,       S<1, 1, 4, 4>,      S<2, 1, 4, 4>,      S<8, 1,  32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,       S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,               5,                  4>
    // clang-format on
    >;

template <typename OutElementOp>
using device_gemm_quantization_dl_c_shuffle_i8_i8_i8_km_nk_mn_instances = std::tuple<
    // clang-format off
        //####################|      A|      B|          Ds|      E|  AData|  BData| AccData|      DsData|   EData|           A|           B|           CDE|           GEMM| Block|  MPer|  NPer|  KPer| K1|   M1Per|  N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
        //####################| Layout| Layout|      Layout| Layout|   Type|   Type|    Type|        Type|    Type| Elementwise| Elementwise|   Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM| Thread| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
        //####################|       |       |            |       |       |       |        |            |        |   Operation|   Operation|     Operation|               |      |      |      |      |   |        |       |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
        //####################|       |       |            |       |       |       |        |            |        |            |            |              |               |      |      |      |      |   |        |       |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        DeviceGemmMultipleD_Dl<    Col,    Col, Empty_Tuple,    Row, int8_t, int8_t, int32_t, Empty_Tuple,  int8_t, PassThrough, PassThrough,  OutElementOp,     MNKPadding,   256,   128,   128,    16,  4,      4,       4,      1,       S<8, 2>,       S<8, 2>,      S<2, 1, 4, 4>,      S<8, 1,  32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,       S<1, 1, 4, 4>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>, S<0, 1, 2, 3, 4, 5>,               5,                  4>
    // clang-format on
    >;

template <typename OutElementOp>
using device_gemm_quantization_dl_c_shuffle_i8_i8_i8_mk_kn_mn_instances = std::tuple<
    // clang-format off
        //####################|      A|      B|          Ds|      E|  AData|  BData| AccData|      DsData|   EData|           A|           B|           CDE|           GEMM| Block|  MPer|  NPer|  KPer| K1|   M1Per|  N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
        //####################| Layout| Layout|      Layout| Layout|   Type|   Type|    Type|        Type|    Type| Elementwise| Elementwise|   Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM| Thread| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
        //####################|       |       |            |       |       |       |        |            |        |   Operation|   Operation|     Operation|               |      |      |      |      |   |        |       |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
        //####################|       |       |            |       |       |       |        |            |        |            |            |              |               |      |      |      |      |   |        |       |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        DeviceGemmMultipleD_Dl<    Row,    Row, Empty_Tuple,    Row, int8_t, int8_t, int32_t, Empty_Tuple,  int8_t, PassThrough, PassThrough,  OutElementOp,     MNKPadding,   256,   128,   128,    16,  4,       4,      4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>,      S<2, 1, 4, 4>,      S<8, 1,  32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,       S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,               5,                  4>
    // clang-format on
    >;

template <typename OutElementOp>
using device_gemm_quantization_dl_c_shuffle_i8_i8_i8_mk_nk_mn_instances = std::tuple<
    // clang-format off
        //####################|      A|      B|          Ds|      E|  AData|  BData| AccData|      DsData|   EData|           A|           B|           CDE|           GEMM| Block|  MPer|  NPer|  KPer| K1|   M1Per|  N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
        //####################| Layout| Layout|      Layout| Layout|   Type|   Type|    Type|        Type|    Type| Elementwise| Elementwise|   Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM| Thread| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
        //####################|       |       |            |       |       |       |        |            |        |   Operation|   Operation|     Operation|               |      |      |      |      |   |        |       |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
        //####################|       |       |            |       |       |       |        |            |        |            |            |              |               |      |      |      |      |   |        |       |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        DeviceGemmMultipleD_Dl<    Row,    Col, Empty_Tuple,    Row, int8_t, int8_t, int32_t, Empty_Tuple,  int8_t, PassThrough, PassThrough,  OutElementOp,     MNKPadding,   256,   128,   128,    16,  4,       4,      4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>, S<0, 1, 2, 3, 4, 5>,               5,                  4>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
