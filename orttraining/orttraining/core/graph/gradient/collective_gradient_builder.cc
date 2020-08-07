// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient/gradient_builder.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

IMPLEMENT_GRADIENT_BUILDER(GetSendGradient) {
  // Send inputs: signal A, remote, data; outputs: signal B
  // Recv inputs: signal B, remote; outputs: signal A', data'

  std::vector<ArgDef> out_args;
  out_args.push_back(GI(0));  // Signal
  for (int i = 2; i < GetSrcNodeInputSize(); ++i) {
    out_args.push_back(GI(i));  // Data
  }

  return std::vector<NodeDef>{
      NodeDef(OpDef{"Recv", kMSDomain, 1},
              {O(0), I(1)},  // {Signal, Remote}
              out_args,
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetRecvGradient) {
  // Recv inputs: signal A, remote; outputs: signal B, data
  // Send inputs: signal B, remote, data'; outputs: signal A'

  std::vector<ArgDef> in_args;
  in_args.push_back(O(0));  // Signal
  in_args.push_back(I(1));  // Remote

  for (int i = 1; i < GetSrcNodeOutputSize(); ++i) {
    in_args.push_back(GO(i));  // Data
  }

  return std::vector<NodeDef>{
      NodeDef(OpDef{"Send", kMSDomain, 1},
              in_args,
              {GI(0)},  // Signal
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetMegatronFGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
              {GO(0)},
              {GI(0)},
              {MakeAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel))})};
}

IMPLEMENT_GRADIENT_BUILDER(GetMegatronGGradient) {
  return std::vector<NodeDef>{
      NodeDef("Identity",
              {GO(0)},
              {GI(0)})};
}

}  // namespace training
}  // namespace onnxruntime
