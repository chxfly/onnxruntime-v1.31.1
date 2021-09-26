// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include "core/providers/common.h"
// #include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "fused_softmax.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, bool is_log_softmax>
Status FusedSoftMaxComputeHelper(cudaStream_t stream, const T *X,
                                 const bool *mask_data,
                                 const TensorShape &input_shape, T *Y,
                                 cudnnHandle_t /*handle*/, int64_t axis,
                                 int64_t first_dim) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  int64_t N = input_shape.SizeToDimension(axis);
  int64_t batches = input_shape.GetDims()[0];
  int64_t attn_heads = input_shape.GetDims()[1];
  int64_t D = input_shape.SizeFromDimension(axis);
  auto Y_data = reinterpret_cast<CudaT *>(Y);
  auto X_data = reinterpret_cast<const CudaT *>(X);

  // cudnnSoftmaxForward/Backward is not optimal implementation.
  if (D <= 1024 && D * sizeof(T) <= 4096) {
    ORT_ENFORCE(gsl::narrow_cast<int>(N) == gsl::narrow_cast<int>(first_dim) ||
                    gsl::narrow_cast<int>(first_dim) == 1,
                "paded_patches must be 1 or batch_size");
    dispatch_fused_softmax_forward<CudaT, CudaT, AccumulationType_t<CudaT>,
                                   is_log_softmax>(
        stream, Y_data, X_data, mask_data, gsl::narrow_cast<int>(D),
        gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(batches),
        gsl::narrow_cast<int>(attn_heads), gsl::narrow_cast<int>(first_dim));
    return Status::OK();
  }

  // std::vector<int64_t> dims(
  //     {N, 1, 1, D}); // cudnn expects 4D shape in NCHW format

  // const auto alpha = Consts<CudaT>::One;
  // const auto beta = Consts<CudaT>::Zero;
  // CudnnTensor input_tensor;
  // CudnnTensor output_tensor;
  // ORT_RETURN_IF_ERROR(
  //     input_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
  // ORT_RETURN_IF_ERROR(
  //     output_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
  // if (is_log_softmax) {
  //   CUDNN_RETURN_IF_ERROR(cudnnSoftmaxForward(
  //       handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
  //       input_tensor, X_data, &beta, output_tensor, Y_data));
  // } else {
  //   CUDNN_RETURN_IF_ERROR(cudnnSoftmaxForward(
  //       handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
  //       input_tensor, X_data, &beta, output_tensor, Y_data));
  // }

  return Status::OK();
}

#define SPECIALIZED_SOFTMAX_HELPER_IMPL(T)                                     \
  template Status FusedSoftMaxComputeHelper<T, false>(                         \
      cudaStream_t stream, const T *input, const bool *mask_data,              \
      const TensorShape &shape, T *Y, cudnnHandle_t handle, int64_t axis,      \
      int64_t first_dim);                                                      \
  template Status FusedSoftMaxComputeHelper<T, true>(                          \
      cudaStream_t stream, const T *input, const bool *mask_data,              \
      const TensorShape &shape, T *Y, cudnnHandle_t handle, int64_t axis,      \
      int64_t first_dim);

SPECIALIZED_SOFTMAX_HELPER_IMPL(float)
SPECIALIZED_SOFTMAX_HELPER_IMPL(double)
SPECIALIZED_SOFTMAX_HELPER_IMPL(MLFloat16)

template <typename T>
Status FusedSoftmax<T>::ComputeInternal(OpKernelContext *ctx) const {
  const Tensor *X = ctx->Input<Tensor>(0);
  const Tensor *condition = ctx->Input<Tensor>(1);
  const TensorShape &input_shape{X->Shape()}; // input is a 4d tensor with dimensions [batches, attn_heads, seq_len, seq_len]
  size_t rank = input_shape.NumDimensions();
  Tensor *Y = ctx->Output(0, input_shape);

  ORT_ENFORCE(condition->Shape().NumDimensions() == static_cast<size_t>(4));

  // check shape broadcastable.
  // ORT_ENFORCE(input_shape != condition->Shape(), "only hanle same shape of
  // input and condition"); special case when there is a dim value of 0 in the
  // shape.
  if (input_shape.Size() == 0)
    return Status::OK();

  // handle negative and enforce axis is valid
  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));

  const T *X_data = nullptr;
  T *Y_data = nullptr;
  const TensorShape *compute_input_shape = nullptr;

  X_data = X->template Data<T>();
  Y_data = Y->template MutableData<T>();
  compute_input_shape = &input_shape;

  const int64_t first_dim = condition->Shape().GetDims()[0];

  Status status;
  if (log_softmax_) {
    status = FusedSoftMaxComputeHelper<T, true>(
        Stream(), X_data,
        reinterpret_cast<const bool *>(condition->template Data<bool>()),
        *compute_input_shape, Y_data, CudnnHandle(), static_cast<int64_t>(axis),
        first_dim);
  } else {
    status = FusedSoftMaxComputeHelper<T, false>(
        Stream(), X_data,
        reinterpret_cast<const bool *>(condition->template Data<bool>()),
        *compute_input_shape, Y_data, CudnnHandle(), static_cast<int64_t>(axis),
        first_dim);
  }

  if (!status.IsOK())
    return status;

  return Status::OK();
}

#define REGISTER_KERNEL_TYPED(op_name, T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      op_name, kMSDomain, 1, T, kCudaExecutionProvider,                        \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      onnxruntime::contrib::cuda::FusedSoftmax<T>);

#define SPECIALIZED_COMPUTE(T)                                                 \
  REGISTER_KERNEL_TYPED(FusedSoftmax, T)                                       \
  template Status FusedSoftmax<T>::ComputeInternal(OpKernelContext *ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

} // namespace cuda
} // namespace contrib
} // namespace onnxruntime
