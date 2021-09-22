// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      1, 8,                                                       \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      9, 12,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      13,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_KERNEL_TYPED(BFloat16)
#endif

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb,
                                     int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  int64_t left_k = transa ? left_shape[left_num_dims - 2] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1] : right_shape[right_num_dims - 2];
  if (left_k != right_k) {
    return false;
  }

  int64_t n = transa ? left_shape[left_num_dims - 1] : left_shape[left_num_dims - 2];
  int64_t m = transb ? right_shape[right_num_dims - 2] : right_shape[right_num_dims - 1];
  stride_A = n * left_k;
  stride_B = right_num_dims == 2 ? 0 : right_k * m;
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  const Tensor *B = ctx->Input<Tensor>(2);
  // Ignore the transpose flag if rank of input being 1.
  // Be noted: numpy.transpose on vector does not change anything.
  bool transa = trans_A_;
  bool transb = trans_B_;
  if (left_X->Shape().NumDimensions() == 1) {
    transa = false;
  }
  if (right_X->Shape().NumDimensions() == 1) {
    transb = false;
  }

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape(), transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  const CudaT *A_data =
      reinterpret_cast<const CudaT *>(left_X->template Data<T>());
  const CudaT *B_data =
      reinterpret_cast<const CudaT *>(right_X->template Data<T>());
  CudaT *out_data = reinterpret_cast<CudaT *>(Y->template MutableData<T>());
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const CudaT alpha = ToCudaType<T>::FromFloat(alpha_);
  const CudaT beta = ToCudaType<T>::FromFloat(beta_);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);
  const CudaT one = ToCudaType<T>::FromFloat(1.0f);

  auto N = static_cast<int>(helper.N());
  auto M = static_cast<int>(helper.M());
  auto K = static_cast<int>(helper.K());

  cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int lda = transa ? M : K;
  const int ldb = transb ? K : N;
  const int ldc = N;
  int64_t stride_A, stride_B, stride_C, batch_count;
  auto& device_prop = GetDeviceProp();

  // broadcast bias if needed and is present
  if (beta_ != 0 && B != nullptr) {
    auto &b_shape = B->Shape();
    const CudaT *b_data =
        reinterpret_cast<const CudaT *>(B->template Data<T>());
    if (b_shape.Size() == 1) {
      // if B is (), (1,) or (1, 1), broadcast the scalar
      CUBLAS_RETURN_IF_ERROR(cublasCopyHelper(Stream(), CublasHandle(), M * N,
                                              b_data, 0, out_data, 1));
    } else if (b_shape.NumDimensions() == 1 || b_shape[0] == 1) {
      // B is (N,) or (1, N), broadcast using Y(N,M) = 1 * B(N,1) x ones(1,M) +
      // 0 * Y
      CUBLAS_RETURN_IF_ERROR(
          cublasGemmHelper(CublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1,
                           /*alpha*/ &one, b_data, N, GetConstOnes<CudaT>(M), 1,
                           /*beta*/ &zero, out_data, N, device_prop));
    } else if (b_shape.NumDimensions() == 2 && b_shape[1] == 1) {
      // B is (M, 1), broadcast using Y(N,M) = 1 * ones(N,1) x B(1,M) + 0 * Y
      CUBLAS_RETURN_IF_ERROR(
          cublasGemmHelper(CublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1,
                           /*alpha*/ &one, GetConstOnes<CudaT>(N), N, b_data, 1,
                           /*beta*/ &zero, out_data, N, device_prop));
    } else {
      // B is (M, N), no broadcast needed.
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(out_data, b_data, M * N * sizeof(T),
                                           cudaMemcpyDeviceToDevice, Stream()));
    }
  }

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        Base::CublasHandle(), transB, transA, N, M, K, &alpha, B_data, ldb,
        A_data, lda, B != nullptr ? &beta : &zero, out_data, ldc, device_prop));
    return Status::OK();
  } else if (CanUseStridedBatchedGemm(left_X->Shape(), right_X->Shape(),
                                      transa, transb, stride_A, stride_B, stride_C, batch_count)) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
        Base::CublasHandle(), transB, transA, N, M, K, &alpha, B_data, ldb,
        stride_B, A_data, lda, stride_A, B != nullptr ? &beta : &zero, out_data,
        ldc, stride_C, static_cast<int>(batch_count), device_prop));

    return Status::OK();
  }

  CudaAsyncBuffer<const CudaT*> left_arrays(this, helper.LeftOffsets().size());
  CudaAsyncBuffer<const CudaT*> right_arrays(this, helper.RightOffsets().size());
  CudaAsyncBuffer<CudaT*> output_arrays(this, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(A_data, helper.LeftOffsets(),
                                      left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(B_data, helper.RightOffsets(),
                                      right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(out_data, helper.OutputOffsets(),
                                      output_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu());

  // note that onnxruntime OrtValue is row major, while cublas is column major,
  // so swap left/right operands
  CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
      Base::CublasHandle(), transB, transA, N, M, K, &alpha,
      right_arrays.GpuPtr(), ldb, left_arrays.GpuPtr(), lda,
      B != nullptr ? &beta : &zero, output_arrays.GpuPtr(), ldc,
      static_cast<int>(helper.OutputOffsets().size()), device_prop));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
