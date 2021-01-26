// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class Yield final : public OpKernel {
 public:
  Yield(const OpKernelInfo& info) : OpKernel(info) {
    int64_t push_input = info.GetAttrOrDefault<int64_t>("push_input", 0);
    push_input_ = (push_input == 1);
  }
  Status Compute(OpKernelContext* context) const override;

  private:
  bool push_input_;
};

}  // namespace contrib
}  // namespace onnxruntime
