// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient/gradient_builder.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

IMPLEMENT_GRADIENT_BUILDER(GetConvGradient) {
  std::vector<ArgDef> outputs;
  for (int i = 0; i < 3; i++) {
    if (IsGradientRequiredForSrcNodeInput(i)) {
      outputs.push_back(GI(i));
    } else {
      outputs.push_back(ArgDef("", nullptr));
    }
  }

  return std::vector<NodeDef>{
      NodeDef("ConvGrad",
              {GO(0), I(0), I(1)},
              outputs,
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetLayerNormalizationGradient) {
  if (GetGradientGraphConfiguration().use_invertible_layernorm_grad) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"InvertibleLayerNormalizationGrad", kMSDomain, 1},
                {GO(0), O(0), I(1), I(2), O(2)},
                {GI(0), GI(1), GI(2)},
                {SrcNodeAttributes()})};
  } else {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"LayerNormalizationGrad", kMSDomain, 1},
                {GO(0), I(0), I(1), O(1), O(2)},
                {GI(0), GI(1), GI(2)},
                {SrcNodeAttributes()})};
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetBatchNormalizationGradient) {
  auto attributes = SrcNodeAttributes();
  if (attributes.find("epsilon") != attributes.end()) {
    float epsilon = attributes.at("epsilon").f();
    return std::vector<NodeDef>{
        NodeDef(OpDef{"BatchNormalizationGrad", kMSDomain, 1},
                {GO(0), I(0), I(1), O(3), O(4)},
                {GI(0), GI(1), GI(2)},
                {MakeAttribute("epsilon", epsilon)})};
  } else {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"BatchNormalizationGrad", kMSDomain, 1},
                {GO(0), I(0), I(1), O(3), O(4)},
                {GI(0), GI(1), GI(2)})};
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetDropoutGradient) {
  std::vector<ArgDef> inputs{GO(0), O(1)};
  for (int i = 1; i < GetSrcNodeInputSize(); i++) {
    inputs.push_back(I(i));
  }
  return std::vector<NodeDef>{
      NodeDef(OpDef{"DropoutGrad", kMSDomain, 1},
              inputs,
              {GI(0)},
              {SrcNodeAttributes()})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTrainableDropoutGradient) {
  std::vector<ArgDef> inputs{GO(0), O(1)};
  for (int i = 1; i < GetSrcNodeInputSize(); i++) {
    inputs.push_back(I(i));
  }
  return std::vector<NodeDef>{
      NodeDef(OpDef{"TrainableDropoutGrad", kMSDomain, 1},
              inputs,
              {GI(0)},
              {SrcNodeAttributes()})};
}

IMPLEMENT_GRADIENT_BUILDER(GetWhereGradient) {
  std::vector<NodeDef> result;
  const int64_t data_type = static_cast<int64_t>(I(1).type_proto->tensor_type().elem_type());
  if (IsGradientRequiredForSrcNodeInput(1)) {
    result.push_back(NodeDef("Cast", {I(0)}, {IA("Positive_Mask")}, {MakeAttribute("to", data_type)}));
    result.push_back(NodeDef("Mul", {GO(0), IA("Positive_Mask")}, {GI(1)}));
  }

  if (IsGradientRequiredForSrcNodeInput(2)) {
    result.push_back(NodeDef("Not", {I(0)}, {IA("Not_Condition", IType(0))}));
    result.push_back(NodeDef("Cast", {IA("Not_Condition")}, {IA("Negative_Mask")}, {MakeAttribute("to", data_type)}));
    result.push_back(NodeDef("Mul", {GO(0), IA("Negative_Mask")}, {GI(2)}));
  }
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetGlobalAveragePoolGradient) {
  const ArgDef X = I(0);

  // TODO: ONNX supports unknown shape for the input feed, e.g. [1, 3, -1, 28],
  // thus the shape of input might be missing at graph construction time.
  // However, in practice, we haven't seen a single model with unknown input shape.
  // We need to get the shape at runtime if this case need to be supported.
  // One way to do it is: scale = Size_Op(X, from=2); scaled_dY = Mul_Op(dY, scale)
  const auto& x_dims = X.type_proto->tensor_type().shape().dim();
  ORT_ENFORCE(x_dims.size() >= 3, "Input dimension cannot be less than 3.");
  int64_t scale = 1;
  for (auto dim = x_dims.begin() + 2; dim < x_dims.end(); dim++) {
    if (dim->has_dim_value()) {
      scale *= dim->dim_value();
    } else {
      ORT_ENFORCE(false, "Dimension missing");
    }
  }

  NodeDef scale_node = ConstantValueNode(1.0f / static_cast<float>(scale), Name("Scale"));
  ArgDef SCALE = scale_node.output_args[0];
  return std::vector<NodeDef>{
      scale_node,
      NodeDef("Mul",
              {GO(0), SCALE},
              {IA("scaled_dY")}),
      NodeDef("Shape",
              {X},
              {IA("x_shape")}),
      NodeDef("Expand",
              {IA("scaled_dY"), IA("x_shape")},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetAveragePoolGradient) {
  return std::vector<NodeDef>{
      NodeDef("AveragePoolGrad",
              {GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetMaxPoolGradient) {
  return std::vector<NodeDef>{
      NodeDef("MaxPoolGrad",
              {GO(0), O(1)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetPoolGradient) {
  return std::vector<NodeDef>{
      NodeDef(SrcNodeOpType() + "Grad",
              {GO(0), I(0), O(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

}  // namespace training
}  // namespace onnxruntime
