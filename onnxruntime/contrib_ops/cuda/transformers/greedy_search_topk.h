// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

//#include "core/common/gsl.h"
//#include "core/framework/allocator.h"
//#include "core/framework/ort_value.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

//template<typename ElementType>
//inline void AllocateTempBufferForGetGreedySearchTop1(
//    int32_t batch_size,
//    AllocatorPtr allocator,
//    BufferUniquePtr& buffer,
//    gsl::span<ElementType>& stage_1_scores,  // shape (batch_size, parts_of_vocab)
//    gsl::span<int32_t>& stage_1_tokens,      // shape (batch_size, parts_of_vocab)
//    gsl::span<ElementType>& output_scores,   // shape (batch_size)
//    gsl::span<int32_t>& output_tokens        // shape (batch_size)
//) {
//  constexpr size_t kMaxPartsPerVocab = 128;
//  const size_t stage_1_element_size = kMaxPartsPerVocab * batch_size;
//  const size_t output_element_size = batch_size;
//
//  void* topk_data = allocator->Alloc((stage_1_element_size + output_element_size) * (sizeof(ElementType) + sizeof(int32_t)));
//  BufferUniquePtr temp_buffer(topk_data, BufferDeleter(allocator));
//  buffer = std::move(temp_buffer);
//
//  ElementType* stage_1_scores_data = reinterpret_cast<ElementType*>(topk_data);
//  stage_1_scores = gsl::make_span<ElementType>(stage_1_scores_data, stage_1_element_size);
//
//  int32_t* stage_1_token_data = reinterpret_cast<int32_t*>(stage_1_scores_data + stage_1_element_size);
//  stage_1_tokens = gsl::make_span<int32_t>(stage_1_token_data, stage_1_element_size);
//
//  ElementType* output_score_data = reinterpret_cast<ElementType*>(stage_1_token_data + stage_1_element_size);
//  output_scores = gsl::make_span<ElementType>(output_score_data, output_element_size);
//
//  int32_t* output_token_data = reinterpret_cast<ElementType*>(output_score_data + output_element_size);
//  output_tokens = gsl::make_span<int32_t>(output_token_data, output_element_size);
//}


template <typename T>
void GreedySearchTop1(
    const T* input,
    int32_t batch_size,
    int32_t vocab_size,
    T* tmp_values,
    int32_t* tmp_tokens,
    T* output_values,
    int32_t* output_tokens,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
