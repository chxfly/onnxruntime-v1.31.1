// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_transfer.h"
#include "core/providers/rocm/rocm_pch.h"

namespace onnxruntime {

enum HIPStreamType : int {
  kHipStreamDefault = 0,
  kHipStreamCopyIn,
  kHipStreamCopyOut,
  kTotalHipStreams,
};

class GPUDataTransfer : public IDataTransfer {
 public:
  GPUDataTransfer() {};
  ~GPUDataTransfer() {};

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  //common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream* stream) const override;

  hipStream_t GetStream(int queue_id) const {
    ORT_ENFORCE(queue_id >= 0 && queue_id < kTotalHipStreams);
    return streams_[queue_id];
  }

 private:
  bool do_copy_in_default_stream_;
  hipStream_t streams_[kTotalHipStreams];
};

}  // namespace onnxruntime
