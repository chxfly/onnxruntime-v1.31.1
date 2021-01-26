// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

class Yield final : public CudaKernel {
 public:
  Yield(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t push_input = info.GetAttrOrDefault<int64_t>("push_input", 0);
    push_input_ = (push_input == 1);
  }
  Status ComputeInternal(OpKernelContext* context) const override;

  private:
  bool push_input_;
};

}  // namespace cuda
}  // namespace onnxruntime
