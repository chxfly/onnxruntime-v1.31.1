// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/xnnpack/build_kernel_info.h"

namespace onnxruntime {
namespace xnnpack {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kXNNPackDomain, 1, XnnPackConvolution2d);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kXNNPackDomain, 1, XnnPackDepthwiseConvolution2d);

Status RegisterXNNPackKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kXNNPackDomain, 1, XnnPackConvolution2d)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kXNNPackDomain, 1, XnnPackDepthwiseConvolution2d)>,
  };
  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}
}  // namespace xnnpack
}  // namespace onnxruntime