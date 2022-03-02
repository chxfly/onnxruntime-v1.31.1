// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

#include <xnnpack.h>


namespace onnxruntime {
namespace xnnpack {
struct XNNPackOperatorDeleter {
  void operator()(struct xnn_operator* p) {
    if (p != nullptr) {
      // Ignore returned value because it fails only when xnn pack wasn't initialized
      xnn_delete_operator(p);
    }
  }
};
using XNNPackOperator = std::unique_ptr<struct xnn_operator, XNNPackOperatorDeleter>;

class Convolution2d : public OpKernel {
 public:
  Convolution2d(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  XNNPackOperator op0 = nullptr;
  TensorShape output_shape;
};

class DepthWiseConvolution2d : public OpKernel {
 public:
  DepthWiseConvolution2d(const OpKernelInfo& info);
  Status Compute(OpKernelContext*) const override;

 private:
  XNNPackOperator op0 = nullptr;
  TensorShape output_shape;
  float* weight_ = nullptr;
};
}  // namespace xnnpack
}  // namespace onnxruntime
