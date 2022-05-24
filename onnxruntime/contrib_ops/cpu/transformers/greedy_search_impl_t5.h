// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "greedy_search_impl_base.h"
#include "subgraph_t5_encoder.h"
#include "subgraph_t5_decoder.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

// Beam search implementation for T5 model.
template <typename T>
class GreedySearchT5 : public GreedySearchBase<T> {
 public:
  GreedySearchT5(OpKernelContextInternal& context,
               const SessionState& encoder_session_state,
               const SessionState& decoder_session_state,
               T5EncoderSubgraph& encoder_subgraph,
               T5DecoderSubgraph& decoder_subgraph,
               concurrency::ThreadPool* thread_pool,
               void* cuda_stream,
               IConsoleDumper* cuda_dumper,
               BeamSearchParameters& params,
               const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
               const BeamSearchDeviceHelper::TopkFunc& topk_func,
               const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
               const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
               const BeamSearchDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
               const BeamSearchDeviceHelper::InitDecoderFeedsFunc<T>& init_decoder_feeds_func,
               const BeamSearchDeviceHelper::UpdateGreedySearchDecoderFeedsFunc<T>& update_decoder_feeds_func)
      : BeamSearchBase<T>(context, decoder_session_state, thread_pool, cuda_stream, cuda_dumper, params, topk_func, process_logits_func, device_copy_func),
        encoder_session_state_(encoder_session_state),
        encoder_subgraph_(encoder_subgraph),
        decoder_subgraph_(decoder_subgraph),
        add_to_feeds_func_(add_to_feeds_func),
        create_encoder_inputs_func_(create_encoder_inputs_func),
        init_decoder_feeds_func_(init_decoder_feeds_func),
        update_decoder_feeds_func_(update_decoder_feeds_func) {
  }

  // Execute beam search in iterations util stopping criteria is reached.
  Status Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                 const FeedsFetchesManager& decoder_feeds_fetches_manager);

 private:
  const SessionState& encoder_session_state_;

  T5EncoderSubgraph& encoder_subgraph_;
  T5DecoderSubgraph& decoder_subgraph_;

  // Device specific functions
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;

  BeamSearchDeviceHelper::CreateEncoderInputsFunc create_encoder_inputs_func_;
  BeamSearchDeviceHelper::InitDecoderFeedsFunc<T> init_decoder_feeds_func_;
  BeamSearchDeviceHelper::UpdateGreedySearchDecoderFeedsFunc<T> update_decoder_feeds_func_;
};

template <typename T>
Status GreedySearchT5<T>::Execute(const FeedsFetchesManager& encoder_feeds_fetches_manager,
                                ` const FeedsFetchesManager& decoder_feeds_fetches_manager) {
  auto status = Status::OK();

  const GreedySearchParameters* parameters = this->parameters_;

  // Allocate output tensors.
  int64_t sequences_dims[] = {parameters->batch_size, parameters->max_length};
  TensorShape sequences_shape(&sequences_dims[0], sizeof(sequences_dims) / sizeof(sequences_dims[0]));
  Tensor* output_sequences = this->context_.Output(0, sequences_shape);

  // ------------------------------------
  // Call encoder subgraph.
  // ------------------------------------
  std::vector<OrtValue> encoder_feeds;
  std::vector<OrtValue> encoder_fetches;

  const OrtValue* encoder_input_ids_value = this->context_.GetInputOrtValue(0);
  const Tensor& encoder_input_ids = encoder_input_ids_value->Get<Tensor>();

  GreedySearchState<T> greedysearch_state;
  greedysearch_state.Init(this->cpu_allocator_,
                          this->temp_space_allocator_,
                          static_cast<size_t>(parameters->BatchBeamSize()),
                          static_cast<int>(1),          // In encoder-decoder model, the initial sequence_length is 1
                          parameters->max_length,
                          this->IsCuda());

  IAllocatorUniquePtr<char> buffer;
  ORT_RETURN_IF_ERROR(this->encoder_subgraph_.CreateInitialFeeds(
      encoder_input_ids,
      this->implicit_inputs_,
      parameters->num_beams,
      parameters->pad_token_id,
      parameters->decoder_start_token_id,
      greedysearch_state.sequence_lengths,
      encoder_feeds,
      this->create_encoder_inputs_func_,
      this->add_to_feeds_func_,
      buffer));

  ORT_RETURN_IF_ERROR(utils::ExecuteSubgraph(this->encoder_session_state_, encoder_feeds_fetches_manager, encoder_feeds, encoder_fetches, {},
                                             ExecutionMode::ORT_SEQUENTIAL, this->context_.GetTerminateFlag(), this->context_.Logger()));

  // ------------------------------------
  // Initialize resources with the decoder input_ids
  // ------------------------------------
  greedysearch_state.SetSequence(
    encoder_feeds[2].Get<Tensor>().DataAsSpan<int64_t>(),
    static_cast<size_t>(parameters->BatchBeamSize()),
    parameters->max_length,
    1);

  // ------------------------------------------------------------------------------
  // Generate next token from logits output from encoder, and initialize decoder inputs.
  // ------------------------------------------------------------------------------
  gsl::span<int32_t> next_tokens;

  int iteration_counter = 0;
  std::vector<OrtValue> decoder_feeds;
  int current_length = 1;
  if (current_length + 1 < parameters->max_length) {
    ++iteration_counter;
    ORT_RETURN_IF_ERROR(this->GenerateNextToken(encoder_fetches[0], next_tokens, greedysearch_state, iteration_counter));
    ++current_length;  // Increase sequence length after a new token is generated.
    ORT_RETURN_IF_ERROR(decoder_subgraph_.CreateInitialFeeds(next_tokens.as_span<const int32_t>(),
                                                             this->implicit_inputs_,
                                                             encoder_feeds,
                                                             encoder_fetches,
                                                             decoder_feeds));
  }

  // TODO: allocate fetches. use ping-pong buffers for past state.
  std::vector<OrtValue> decoder_fetches;
  while (current_length < parameters->max_length) {
    iteration_counter++;
#ifdef DEBUG_BEAM_SEARCH
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);
#endif

    status = utils::ExecuteSubgraph(this->decoder_session_state_, decoder_feeds_fetches_manager, decoder_feeds, decoder_fetches, {},
                                    ExecutionMode::ORT_SEQUENTIAL, this->context_.GetTerminateFlag(), this->context_.Logger());

    ORT_RETURN_IF_ERROR(status);

    const OrtValue& logits = decoder_fetches[0];
    ORT_RETURN_IF_ERROR(this->GenerateNextToken(logits, next_tokens, greedysearch_state, iteration_counter));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (greedysearch_state.all_done) {
      break;
    }

    // Increase sequence length after a new token is generated.
    ++current_length;

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters->max_length) {
      ORT_RETURN_IF_ERROR(this->update_decoder_feeds_func_(
          this->temp_space_allocator_,
          this->cuda_stream_,
          decoder_fetches,
          decoder_feeds,
          current_length,
          next_tokens.as_span<const int32_t>(),
          this->GetConsoleDumper()));
    }
    decoder_fetches.clear();
  }

  // copy sequences to output buffer
  // TODO

  return status;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
