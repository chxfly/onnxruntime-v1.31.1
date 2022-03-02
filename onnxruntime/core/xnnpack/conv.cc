/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* Modifications Copyright (c) Microsoft. */
//#define _CRT_SECURE_NO_WARNINGS

#include "core/xnnpack/conv.h"

#include "core/common/safeint.h"
#include "core/xnnpack/build_kernel_info.h"
#include "core/xnnpack/schema/xnnpack_onnx_defs.h"
#include "core/framework/tensorprotoutils.h"

#define XNNPACK_CPU_MS_DOMAIN_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kXNNPackDomain, ver, kCpuExecutionProvider, builder, __VA_ARGS__)

namespace onnxruntime {
namespace xnnpack {
#if 0
void CompareData(const Tensor& input, const char* varname) { 
  size_t len2 = input.Shape().Size();
  std::vector<float> v(len2);
  std::ostringstream oss;
  oss << "D:\\src\\onnxruntime\\a\\Debug\\" << varname << ".bin";
  FILE* f = fopen(oss.str().c_str(), "rb");
  assert(f != nullptr);
  size_t bytesToRead = len2 * sizeof(float);
  size_t readed = fread(v.data(), 1, bytesToRead, f);
  assert(bytesToRead == readed);
  const float* inputdata = input.Data<float>();
  for (size_t i = 0; i != len2; ++i) {
    float diff = std::abs(inputdata[i] - v[i]);
    assert(diff < 1e-5);
  }
}
#endif

Convolution2d::Convolution2d(const OpKernelInfo& info) : OpKernel(info) {
  const Tensor* weight = nullptr;
  const Tensor* B = nullptr;
  const ONNX_NAMESPACE::TypeProto* input_type_proto = info.GetInputType(0);
  const ONNX_NAMESPACE::TypeProto* output_type_proto = info.GetOutputType(0);

  ORT_ENFORCE(input_type_proto != nullptr);
  ORT_ENFORCE(output_type_proto != nullptr);

  output_shape = utils::GetTensorShapeFromTensorShapeProto(output_type_proto->tensor_type().shape());

  ORT_ENFORCE(info.TryGetConstantInput(1, &weight));
  ORT_ENFORCE(info.TryGetConstantInput(2, &B));

  int64_t input_channels = input_type_proto->tensor_type().shape().dim(3).dim_value();
  int64_t output_channels = output_type_proto->tensor_type().shape().dim(3).dim_value();
  const TensorShape& kernel_shape = weight->Shape();
  int64_t kernel_height = kernel_shape[1];
  int64_t kernel_width = kernel_shape[2];

  int64_t input_padding_top;
  int64_t input_padding_right;
  int64_t input_padding_bottom;
  int64_t input_padding_left;

  int64_t subsampling_height;
  int64_t subsampling_width;
  int64_t dilation_height;
  int64_t dilation_width;
  int64_t groups;
  float output_min;
  float output_max;
  int64_t padding_mode;
  ORT_ENFORCE(info.GetAttr("input_padding_top", &input_padding_top).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_right", &input_padding_right).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_bottom", &input_padding_bottom).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_left", &input_padding_left).IsOK());
  ORT_ENFORCE(info.GetAttr("subsampling_height", &subsampling_height).IsOK());
  ORT_ENFORCE(info.GetAttr("subsampling_width", &subsampling_width).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_height", &dilation_height).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_width", &dilation_width).IsOK());
  ORT_ENFORCE(info.GetAttr("groups", &groups).IsOK());
  // TODO: handle optional case
  ORT_ENFORCE(info.GetAttr("output_min", &output_min).IsOK());
  ORT_ENFORCE(info.GetAttr("output_max", &output_max).IsOK());
  ORT_ENFORCE(info.GetAttr("padding_mode", &padding_mode).IsOK());
  uint32_t flags = 0;
  if (padding_mode == 1) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  size_t group_input_channels = input_channels / groups;
  size_t group_output_channels = output_channels / groups;
  xnn_status status;
  struct xnn_operator* p;
  status = xnn_create_convolution2d_nhwc_f32(
      gsl::narrow<uint32_t>(input_padding_top),
      gsl::narrow<uint32_t>(input_padding_right),
      gsl::narrow<uint32_t>(input_padding_bottom),
      gsl::narrow<uint32_t>(input_padding_left),
      gsl::narrow<uint32_t>(kernel_height),
      gsl::narrow<uint32_t>(kernel_width),
      gsl::narrow<uint32_t>(subsampling_height),
      gsl::narrow<uint32_t>(subsampling_width),
      gsl::narrow<uint32_t>(dilation_height),
      gsl::narrow<uint32_t>(dilation_width),
      gsl::narrow<uint32_t>(groups),
      gsl::narrow<uint32_t>(group_input_channels),
      gsl::narrow<uint32_t>(group_output_channels),
      gsl::narrow<uint32_t>(input_channels),
      gsl::narrow<uint32_t>(output_channels),
      weight->Data<float>(),
      B->Data<float>(),
      output_min,
      output_max,
      flags,
      &p);
  ORT_ENFORCE(status == xnn_status_success);
  op0.reset(p);
}
Status Convolution2d::Compute(OpKernelContext* context) const {
  std::cout << "running " << context->GetNodeName() << std::endl;
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, output_shape);
  const TensorShape& input_shape = X->Shape();  
  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      op0.get(),
      input_shape[0] /* batch size */, input_shape[1] /* input height */, input_shape[2] /* input width */,
      X->Data<float>() /* input */, Y->MutableData<float>() /* output */,
      nullptr /* threadpool */);  
  ORT_ENFORCE(status == xnn_status_success);
  status = xnn_run_operator(op0.get(), nullptr);
  ORT_ENFORCE(status == xnn_status_success);  
  return Status::OK();
}

XNNPACK_CPU_MS_DOMAIN_OPERATOR_KERNEL(
    XnnPackConvolution2d,
    1,
    KernelDefBuilder(),
    Convolution2d);

XNNPACK_CPU_MS_DOMAIN_OPERATOR_KERNEL(
    XnnPackDepthwiseConvolution2d,
    1,
    KernelDefBuilder(),
    DepthWiseConvolution2d);

Status DepthWiseConvolution2d::Compute(OpKernelContext* context) const {
  //std::cout << "running " << context->GetNodeName() << std::endl;
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, output_shape);
  const TensorShape& input_shape = X->Shape();
  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      op0.get(),
      input_shape[0] /* batch size */, input_shape[1] /* input height */, input_shape[2] /* input width */,
      X->Data<float>() /* input */, Y->MutableData<float>() /* output */,
      nullptr /* threadpool */);
  ORT_ENFORCE(status == xnn_status_success);
  status = xnn_run_operator(op0.get(), nullptr);
  ORT_ENFORCE(status == xnn_status_success);

  return Status::OK();
}

static void hwc_to_chw(const float* input, size_t h, size_t w, size_t channels, float* output_data) {
  size_t stride = h * w;
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != channels; ++c) {
      output_data[c * stride + i] = input[i * channels + c];
    }
  }
}
DepthWiseConvolution2d::DepthWiseConvolution2d(const OpKernelInfo& info) : OpKernel(info) {
  const Tensor* weight = nullptr;
  const Tensor* B = nullptr;
  const ONNX_NAMESPACE::TypeProto* input_type_proto = info.GetInputType(0);
  const ONNX_NAMESPACE::TypeProto* output_type_proto = info.GetOutputType(0);
  ORT_ENFORCE(input_type_proto != nullptr);
  ORT_ENFORCE(output_type_proto != nullptr);
  output_shape = utils::GetTensorShapeFromTensorShapeProto(output_type_proto->tensor_type().shape());
  ORT_ENFORCE(info.TryGetConstantInput(1, &weight));
  ORT_ENFORCE(info.TryGetConstantInput(2, &B));
  const TensorShape& kernel_shape = weight->Shape();
  auto cpu_alloc = info.GetAllocator(0, OrtMemTypeDefault);
  weight_ = static_cast<float*>(cpu_alloc->AllocArray(kernel_shape.Size(), sizeof(float)));
  ORT_ENFORCE(weight_ != nullptr);
  auto weight_type = DataTypeImpl::GetType<float>();
  TensorShape new_weight_shape{kernel_shape[3], kernel_shape[1], kernel_shape[2], 1};
  hwc_to_chw(weight->Data<float>(), kernel_shape[1], kernel_shape[2], kernel_shape[3], weight_);
  Tensor new_weight(weight_type, new_weight_shape, weight_, cpu_alloc);  

  int64_t input_channels = input_type_proto->tensor_type().shape().dim(3).dim_value();
  // Weight shape : [ 1, kernel_height, kernel_width, input_channels * depth_multiplier ]
  ORT_ENFORCE(kernel_shape.NumDimensions() == 4);
  ORT_ENFORCE(kernel_shape[3] % input_channels == 0);
  int64_t kernel_height = kernel_shape[1];
  int64_t kernel_width = kernel_shape[2];

  int64_t input_padding_top;
  int64_t input_padding_right;
  int64_t input_padding_bottom;
  int64_t input_padding_left;

  int64_t subsampling_height;
  int64_t subsampling_width;
  int64_t dilation_height;
  int64_t dilation_width;
  float output_min;
  float output_max;
  int64_t padding_mode;
  ORT_ENFORCE(info.GetAttr("input_padding_top", &input_padding_top).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_right", &input_padding_right).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_bottom", &input_padding_bottom).IsOK());
  ORT_ENFORCE(info.GetAttr("input_padding_left", &input_padding_left).IsOK());
  ORT_ENFORCE(info.GetAttr("subsampling_height", &subsampling_height).IsOK());
  ORT_ENFORCE(info.GetAttr("subsampling_width", &subsampling_width).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_height", &dilation_height).IsOK());
  ORT_ENFORCE(info.GetAttr("dilation_width", &dilation_width).IsOK());
  // TODO: handle optional case
  ORT_ENFORCE(info.GetAttr("output_min", &output_min).IsOK());
  ORT_ENFORCE(info.GetAttr("output_max", &output_max).IsOK());
  ORT_ENFORCE(info.GetAttr("padding_mode", &padding_mode).IsOK());
  uint32_t flags = 0;
  if (padding_mode == 1) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  int64_t depth_multiplier = kernel_shape[3] / input_channels;
  struct xnn_operator* p;
  xnn_status status = xnn_create_convolution2d_nhwc_f32(
      gsl::narrow<uint32_t>(input_padding_top),
      gsl::narrow<uint32_t>(input_padding_right),
      gsl::narrow<uint32_t>(input_padding_bottom),
      gsl::narrow<uint32_t>(input_padding_left),
      gsl::narrow<uint32_t>(kernel_height),
      gsl::narrow<uint32_t>(kernel_width),
      gsl::narrow<uint32_t>(subsampling_height),
      gsl::narrow<uint32_t>(subsampling_width),
      gsl::narrow<uint32_t>(dilation_height),
      gsl::narrow<uint32_t>(dilation_width),
      gsl::narrow<uint32_t>(input_channels) /* groups */,
      1 /* group_input_channels */,
      depth_multiplier /* group_output_channels */,
      input_channels /* input_channel_stride */,
      kernel_shape[3] /* output_channel_stride */,
      weight_,
      B->Data<float>(),
      output_min,
      output_max,
      flags,
      &p);
  ORT_ENFORCE(status == xnn_status_success);
  op0.reset(p);
}

}  // namespace xnnpack
}  // namespace onnxruntime
