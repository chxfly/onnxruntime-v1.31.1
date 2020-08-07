// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient/gradient_builder.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

IMPLEMENT_GRADIENT_BUILDER(GetReluGradient) {
  return std::vector<NodeDef>{
      NodeDef("ReluGrad",
              {GO(0), O(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"SoftmaxGrad", kMSDomain, 1},
              {GO(0), O(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetLogSoftmaxGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"LogSoftmaxGrad", kMSDomain, 1},
              {GO(0), O(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetGeluGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"GeluGrad", kMSDomain, 1},
              {GO(0), I(0)},
              {GI(0)})};
}

namespace {
std::vector<NodeDef> GetBiasGeluGradNodes(
    bool use_approximation,
    const ArgDef& dY, const ArgDef& X, const ArgDef& B,                    // inputs
    const ArgDef& dX, const ArgDef& dB,                                    // outputs
    const ArgDef& b_axes, const ArgDef& b_shape, const ArgDef& x_shape) {  //intermediate args
  std::vector<Dimension> B_shape, X_shape;
  if (GetShape(B, B_shape).IsOK() && GetShape(X, X_shape).IsOK()) {
    ORT_ENFORCE(B_shape.size() == 1, "B must have exactly one dimension.");

    const std::vector<int64_t> B_axes = [&B_shape, &X_shape]() {
      std::vector<int64_t> result{};
      ComputeBroadcastBackwardAxes(B_shape, X_shape, &result, nullptr);
      return result;
    }();
    return std::vector<NodeDef>{
        NodeDef(OpDef{use_approximation ? "BiasFastGeluGrad_dX" : "BiasGeluGrad_dX", kMSDomain, 1},
                {dY, X, B},
                {dX}),
        NodeDef("ReduceSum",
                {dX},
                {dB},
                {{"keepdims", MakeAttribute("keepdims", int64_t{0})},
                 {"axes", MakeAttribute("axes", B_axes)}})};
  } else {
    std::vector<NodeDef> result;
    ComputeBroadcastBackwardAxesDynamic(B, X, b_shape, x_shape, &b_axes, nullptr, result);
    result.push_back(
        NodeDef(OpDef{use_approximation ? "BiasFastGeluGrad_dX" : "BiasGeluGrad_dX", kMSDomain, 1},
                {dY, X, B},
                {dX}));
    result.push_back(
        NodeDef(OpDef{"ReduceSumTraining", kMSDomain, 1},
                {dX,
                 b_axes},
                {dB},
                {{"keepdims", MakeAttribute("keepdims", int64_t{0})}}));
    return result;
  }
}
}  // namespace

IMPLEMENT_GRADIENT_BUILDER(GetBiasGeluGradient) {
  const auto dY = GO(0), X = I(0), B = I(1),
             dX = GI(0), dB = GI(1);
  ArgDef b_axes = IA("ReduceAxes_" + B.name);
  ArgDef b_shape = IA("Shape_" + B.name);
  ArgDef x_shape = IA("Shape_" + X.name);
  return GetBiasGeluGradNodes(false, dY, X, B, dX, dB, b_axes, b_shape, x_shape);
}

IMPLEMENT_GRADIENT_BUILDER(GetFastGeluGradient) {
  const auto dY = GO(0), X = I(0),
             dX = GI(0);
  const auto num_src_node_inputs = GetSrcNodeInputSize();
  if (num_src_node_inputs == 2) {  // with bias
    // FastGeluGrad doesn't support bias - it needs to be composed with other ops
    const auto B = I(1),
               dB = GI(1);
    ArgDef b_axes = IA("ReduceAxes_" + B.name);
    ArgDef b_shape = IA("Shape_" + B.name);
    ArgDef x_shape = IA("Shape_" + X.name);
    return GetBiasGeluGradNodes(true, dY, X, B, dX, dB, b_axes, b_shape, x_shape);
  }
  if (num_src_node_inputs == 1) {  // without bias
    return std::vector<NodeDef>{
        NodeDef(OpDef{"FastGeluGrad", kMSDomain, 1},
                {dY, X},
                {dX})};
  }
  ORT_THROW("Unexpected number of FastGelu inputs: ", num_src_node_inputs);
}

}  // namespace training
}  // namespace onnxruntime
