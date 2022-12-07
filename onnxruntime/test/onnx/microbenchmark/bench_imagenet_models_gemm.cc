#include "gemm.h"
#include "common.h"
#include <benchmark/benchmark.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/ort_env.h>
#include <core/util/thread_utils.h>
#include <core/common/eigen_common_wrapper.h>
#include "mlas.h"

#include "core/util/math_cpuonly.h"
#include "core/util/math.h"

// TODO: get it from system info
static constexpr int thread_count = 16;

static void BM_Mlas_Single_Thread_ImageNetModels(benchmark::State& state, const char* /*net*/) {
  const size_t M = state.range(0);
  const size_t N = state.range(1);
  const size_t K = state.range(2);

  float* A = GenerateArrayWithRandomValue<float>(M * K, -1.0f, 1.0f);
  float* B = GenerateArrayWithRandomValue<float>(K * N, -1.0f, 1.0f);
  float* C = GenerateArrayWithZeroValue<float>(M * N);
  for (auto _ : state) {
    state.PauseTiming();
    memset(C, 0, M * N * sizeof(float));
    state.ResumeTiming();
    MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 1.0, C, N, nullptr);
  }

  state.SetItemsProcessed(state.iterations() * 2 * M * N * K);
}
BENCHMARK_GEMM(BM_Mlas_Single_Thread_ImageNetModels)

static void BM_Mlas_Multi_Thread_ImageNetModels(benchmark::State& state, const char* /*net*/) {
  const size_t M = state.range(0);
  const size_t N = state.range(1);
  const size_t K = state.range(2);

  onnxruntime::Env& env = onnxruntime::Env::Default();
  onnxruntime::ThreadOptions to;
  onnxruntime::concurrency::ThreadPool tp(&env, to, "main", thread_count, true);

  float* A = GenerateArrayWithRandomValue<float>(M * K, -1.0f, 1.0f);
  float* B = GenerateArrayWithRandomValue<float>(K * N, -1.0f, 1.0f);
  float* C = GenerateArrayWithZeroValue<float>(M * N);
  for (auto _ : state) {
    state.PauseTiming();
    memset(C, 0, M * N * sizeof(float));
    state.ResumeTiming();
    MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 1.0, C, N, &tp);
  }

  state.SetItemsProcessed(state.iterations() * 2 * M * N * K);
}

BENCHMARK_GEMM(BM_Mlas_Multi_Thread_ImageNetModels)

// This one is twice faster than mlas single thread 32-bit GEMM because it can use 16-bit instructions
static void BM_Eigen_FP16_Single_Thread_ImageNetModels(benchmark::State& state, const char* /*net*/) {
  const size_t M = state.range(0);
  const size_t N = state.range(1);
  const size_t K = state.range(2);

  Eigen::half* A = GenerateArrayWithRandomValue<Eigen::half>(M * K, static_cast<Eigen::half>(-1.0f), static_cast<Eigen::half>(1.0f));
  Eigen::half* B = GenerateArrayWithRandomValue<Eigen::half>(K * N, static_cast<Eigen::half>(-1.0f), static_cast<Eigen::half>(1.0f));
  Eigen::half* C = GenerateArrayWithZeroValue<Eigen::half>(M * N);
  static_assert(sizeof(Eigen::half) == 2);
  for (auto _ : state) {
    state.PauseTiming();
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
    memset(C, 0, M * N * sizeof(Eigen::half));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    state.ResumeTiming();
    // Right now this implementation doesn't support multithreading
    onnxruntime::math::MatMul<Eigen::half>(M, N, K, A, B, C, nullptr);
  }

  state.SetItemsProcessed(state.iterations() * 2 * M * N * K);
}
BENCHMARK_GEMM(BM_Eigen_FP16_Single_Thread_ImageNetModels)

// This one is not any faster than the 32-bit mlas GEMM
static void BM_Eigen_FP16_Multi_Thread_ImageNetModels(benchmark::State& state, const char* /*net*/) {
  const size_t M = state.range(0);
  const size_t N = state.range(1);
  const size_t K = state.range(2);

  Eigen::half* A = GenerateArrayWithRandomValue<Eigen::half>(M * K, static_cast<Eigen::half>(-1.0f), static_cast<Eigen::half>(1.0f));
  Eigen::half* B = GenerateArrayWithRandomValue<Eigen::half>(K * N, static_cast<Eigen::half>(-1.0f), static_cast<Eigen::half>(1.0f));
  Eigen::half* C = GenerateArrayWithZeroValue<Eigen::half>(M * N);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> tensor_A(A, M, K);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> tensor_B(B, K, N);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
  Eigen::ThreadPool tp(thread_count);
  Eigen::ThreadPoolDevice my_device(&tp, thread_count);
  for (auto _ : state) {
    state.PauseTiming();
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> tensor_C(C, M, N);
    state.ResumeTiming();
    tensor_C.device(my_device) = tensor_A.contract(tensor_B, product_dims);
  }
  state.SetItemsProcessed(state.iterations() * 2 * M * N * K);
}
BENCHMARK_GEMM(BM_Eigen_FP16_Multi_Thread_ImageNetModels)
