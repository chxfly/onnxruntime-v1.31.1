// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "greedy_search_topk.h"

#include <cub/cub.cuh>

#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
struct Top1 {
  int32_t key;
  T value;

  __device__ __host__ __forceinline__ Top1(int32_t key = -1, T value = NumericLimits<T>::Min()) : key(key), value(value) {

  }

  __device__ __forceinline__ void Reduce(int32_t k, T v) {
    if (value < v || key == -1) {
      key = k;
      value = v;
    }
  }

};

template <typename T>
__device__ __forceinline__ Top1<T> ReduceTop1Op(const Top1<T>& a, const Top1<T>& b) {
  if ((a.value > b.value) || (a.value == b.value && a.key > b.key)) {
    return a;
  }

  return b;
}

// kernel to compute the top 1 on last axis for tensor with shape[batch, parts_of_vocab, vacab_part_size],
// and produce a tensor with shape [batch, parts_of_vocab]
// Its grid is [batch, parts_of_vocab]
template <typename T, int thread_block_size>
__launch_bounds__(thread_block_size) __global__ void GreedySearchTop1Stage1Kernel(
    const T* input,
    int32_t vocab_size,
    int32_t vocab_part_size,
    T* output_values,
    int32_t* output_token) {
  Top1<T> top_1_thread;

  int batch = blockIdx.x;
  int voc_part_id = blockIdx.y;

  int token_id_base = voc_part_id * vocab_part_size;
  const T* input_block = input + batch * vocab_size;
  // voc_part_size
  for (int i = threadIdx.x + token_id_base; i < vocab_part_size + token_id_base; i += blockDim.x) {
    if (i < vocab_size) {
      top_1_thread.Reduce(i, input_block[i]);
    }
  }

  // reduce in thread block
  typedef cub::BlockReduce<Top1<T>, thread_block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  Top1<T> top_1_block = BlockReduce(temp_storage).Reduce(top_1_thread, ReduceTop1Op<T>);
  if (threadIdx.x == 0) {
    output_values += batch * gridDim.y + voc_part_id;
    output_token += batch * gridDim.y + voc_part_id;
    *output_values = top_1_block.value;
    *output_token = top_1_block.key;
  }
}

// kernel to compute the top 1 on last axis for tensor with shape[batch, parts_of_vocab],
// and produce a tensor with shape [batch]
// Its grid is [batch]
template <typename T, int thread_block_size>
__launch_bounds__(thread_block_size) __global__ void GreedySearchTop1Stage2Kernel(
    const T* input_values,
    const int32_t* input_tokens,
    int32_t vocab_size,
    int32_t vocab_parts,
    T* output_values,
    int32_t* output_tokens) {
  const int vector_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  input_values += vector_id * vocab_parts;
  input_tokens += vector_id * vocab_parts;

  Top1<T> thread_top1;
  for (int idx = thread_id; idx < vocab_parts; idx += thread_block_size) {
    thread_top1.Reduce(input_tokens[idx], input_values[idx]);
  }

  typedef cub::BlockReduce<Top1<T>, thread_block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  Top1<T> topk_block = BlockReduce(temp_storage).Reduce(thread_top1, ReduceTop1Op<T>);
  if (thread_id == 0) {
    output_values[vector_id] = topk_block.value;
    output_tokens[vector_id] = topk_block.key;
  }
}

template <typename T>
void GreedySearchTop1(
    const T* input,
    int32_t batch_size,
    int32_t vocab_size,
    T* tmp_values,
    int32_t* tmp_tokens,
    T* output_values,
    int32_t* output_tokens,
    cudaStream_t stream) {
  constexpr int kThreadBlockSize = GridDim::maxThreadsPerBlock;

  int voc_parts = 4;
  if (batch_size < 256) {
    voc_parts = (240 + batch_size - 1) / batch_size;
    voc_parts = std::min(128, voc_parts);  // we implement up to 128
  }

  dim3 stage1_grid(batch_size, voc_parts);

#ifndef USE_ROCM
  cudaFuncSetAttribute(GreedySearchTop1Stage1Kernel<T, kThreadBlockSize>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
#endif  // !USE_ROCM

  GreedySearchTop1Stage1Kernel<T, kThreadBlockSize><<<stage1_grid, kThreadBlockSize, 0, stream>>>(
      input,
      vocab_size,
      (vocab_size + voc_parts - 1) / voc_parts,
      tmp_values,
      tmp_tokens);

  constexpr int KThreadBlockSizeStage2 = 128;
  GreedySearchTop1Stage2Kernel<T, KThreadBlockSizeStage2><<<batch_size, KThreadBlockSizeStage2, 0, stream>>>(
      tmp_values,
      tmp_tokens,
      vocab_size,
      voc_parts,
      output_values,
      output_tokens);
}

template void GreedySearchTop1(
    const float* input,
    int32_t batch_size,
    int32_t vocab_size,
    float* tmp_values,
    int32_t* tmp_tokens,
    float* output_values,
    int32_t* output_tokens,
    cudaStream_t stream);

template void GreedySearchTop1(
    const half* input,
    int32_t batch_size,
    int32_t vocab_size,
    half* tmp_values,
    int32_t* tmp_tokens,
    half* output_values,
    int32_t* output_tokens,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
