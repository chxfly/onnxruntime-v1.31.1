// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, bool is_log_softmax>
Status FusedSoftMaxComputeHelper(cudaStream_t stream, const T *input,
                                 const bool *mask_data,
                                 const TensorShape &shape, T *Y,
                                 cudnnHandle_t handle, int64_t axis,
                                 int64_t first_dim);

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_fused_softmax_forward(cudaStream_t stream, output_t *dst,
                                    const input_t *src, const bool *mask_data,
                                    int softmax_elements,
                                    int softmax_elements_stride, int batches,
                                    int attn_heads, int first_dim);

template <typename T> class FusedSoftmax final : public CudaKernel {
public:
  FusedSoftmax(const OpKernelInfo &info) : CudaKernel{info} {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 1);
    log_softmax_ = info.GetAttrOrDefault<int64_t>("is_log_softmax", 0) == 1;

    // We need to cast away the const as PerThreadCublasHandle() is currently a
    // non-const method
    // TODO: Clean up the CUDAExecutionProvider interface to avoid this
    cuda_ep_ = const_cast<CUDAExecutionProvider *>(
        static_cast<const CUDAExecutionProvider *>(
            info.GetExecutionProvider()));
  }

  Status ComputeInternal(OpKernelContext *context) const override;

private:
  int64_t axis_;
  bool log_softmax_;

  // We need to access to the CUDA EP instance to get the cublas handle to use
  // for transposing(if applicable)
  CUDAExecutionProvider *cuda_ep_;
};

} // namespace cuda
} // namespace contrib
} // namespace onnxruntime
