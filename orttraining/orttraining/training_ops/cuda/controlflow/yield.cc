// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/controlflow/yield.h"
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"
#include "orttraining/training_ops/cpu/controlflow/message_queue.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(Yield, kMSDomain, 1, kCudaExecutionProvider,
                        KernelDefBuilder()
                        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                        .Alias(0,0)
                        .VariadicAlias(1, 1),  // outputs and inputs are mapped one to one, with input offset by 1,
                         Yield);

Status Yield::ComputeInternal(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  for (int i_in = 1; i_in < ctx->InputCount(); ++i_in) {
    // std::cout<<"Pushing to queue:"<< i_in <<"\n";
    onnxruntime::contrib::OrtMessageQueue::GetInstance().Push(*ctx_internal->GetInputMLValue(i_in));
  }

  // single event for InferenceSession::RunInBackgroundAndWaitForYield() that FW graph is done
  const int64_t main_thread_event_id = 0;
  onnxruntime::contrib::OrtEventPool::GetInstance().SignalEvent(main_thread_event_id);

  // wait for event from InferenceSession::ContinueRunInBackground() to continue the BW graph
  const int64_t background_thread_event_id = 1;
  onnxruntime::contrib::OrtEventPool::GetInstance().ResetAndWaitEvent(background_thread_event_id);
  // std::cout<<"InYield, resumed...\n";

  const Tensor* control_input = ctx->Input<Tensor>(0);
  const TensorShape& shape = control_input->Shape();
  Tensor* control_output = ctx->Output(0, shape);
  auto control_input_type = control_input->DataType();

  const void* source = control_input->DataRaw(control_input_type);
  void* target = control_output->MutableDataRaw(control_input_type);
  //If source and target pointers are not equal, we need to copy the data.
  if (target != source) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, control_input->Shape().Size() * control_input->DataType()->Size(), cudaMemcpyDeviceToDevice));
  }

  // Get output grad from somewhere and prepare Op outputs.
  for (int i_out = 1; i_out < ctx->OutputCount(); ++i_out) {
    OrtValue value = onnxruntime::contrib::OrtMessageQueue::GetInstance().Pop();
    const Tensor& X = value.Get<Tensor>();
    const TensorShape& data_shape = X.Shape();
    Tensor* Y = ctx->Output(i_out, data_shape);
    CopyTensor(X, *Y);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
