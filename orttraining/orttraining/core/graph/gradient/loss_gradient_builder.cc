// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient/gradient_builder.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxCrossEntropyGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"SoftmaxCrossEntropyGrad", kMSDomain, 1},
              {GO(0), O(1), I(1)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetSparseSoftmaxCrossEntropyGradient) {
  if (GetSrcNodeInputSize() == 2) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SparseSoftmaxCrossEntropyGrad"},
                {GO(0), O(1), I(1)},
                {GI(0)},
                SrcNodeAttributes())};
  } else if (GetSrcNodeInputSize() == 3) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SparseSoftmaxCrossEntropyGrad"},
                {GO(0), O(1), I(1), I(2)},
                {GI(0)},
                SrcNodeAttributes())};
  } else {
    ORT_ENFORCE(false, "the number of input arguments must be 2 or 3");
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxCrossEntropyLossGradient) {
  if (GetSrcNodeInputSize() == 2) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SoftmaxCrossEntropyLossGrad", kMSDomain, 1},
                {GO(0), O(1), I(1)},
                {GI(0)},
                SrcNodeAttributes())};
  } else if (GetSrcNodeInputSize() == 3) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SoftmaxCrossEntropyLossGrad", kMSDomain, 1},
                {GO(0), O(1), I(1), I(2)},
                {GI(0)},
                SrcNodeAttributes())};
  } else {
    ORT_THROW(false, "the number of input arguments must be 2 or 3");
  }
}

}  // namespace training
}  // namespace onnxruntime
