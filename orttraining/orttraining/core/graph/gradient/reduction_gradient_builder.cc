// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient/gradient_builder.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

IMPLEMENT_GRADIENT_BUILDER(GetReduceMeanGradient) {
  std::vector<NodeDef> result;
  auto attributes = SrcNodeAttributes();
  bool keepdims = true;
  if (attributes.find("keepdims") != attributes.end() &&
      attributes.at("keepdims").has_i()) {
    keepdims = static_cast<bool>(attributes.at("keepdims").i());
  }

  ArgDef grad = GO(0);
  if (!keepdims && attributes.find("axes") != attributes.end()) {
    std::vector<int64_t> axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    grad = IA("Unqueezed_Grad");
    result.push_back(NodeDef("Unsqueeze", {GO(0)}, {grad}, {MakeAttribute("axes", axes_values)}));
  }

  const int64_t type_float = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  result.push_back(NodeDef("Size", {I(0)}, {IA("Scale_Denominator")}));
  result.push_back(
      NodeDef("Cast",
              {IA("Scale_Denominator")},
              {IA("Casted_Scale_Denominator")},
              {MakeAttribute("to", type_float)}));
  result.push_back(NodeDef("Size", {GO(0)}, {IA("Scale_Numerator")}));
  result.push_back(
      NodeDef("Cast",
              {IA("Scale_Numerator")},
              {IA("Casted_Scale_Numerator")},
              {MakeAttribute("to", type_float)}));
  result.push_back(
      NodeDef("Div",
              {IA("Casted_Scale_Numerator"), IA("Casted_Scale_Denominator")},
              {IA("Scale")}));
  result.push_back(NodeDef("Mul", {grad, IA("Scale")}, {IA("Scaled_Grad")}));
  result.push_back(NodeDef("Shape", {I(0)}, {IA("Shaped_X")}));
  result.push_back(NodeDef("Expand", {IA("Scaled_Grad"), IA("Shaped_X")}, {GI(0)}));
  return result;
}

// Reference computation is pytorch's logsumexp_backward
// dx_i = exp(xi) / reduceSum(exp(xi))
// O(0) = log(reduceSum(exp(xi)))
// Self_Sub_Result = I(0) - O(0) = xi - log(sum(exp(xi))) = log( xi / reduceSum(exp(xi)))
// Gradient computation is re-using output and input from forward op, can be a recomputation candidate.
IMPLEMENT_GRADIENT_BUILDER(GetReduceLogSumExpGradient) {
  std::vector<NodeDef> result;
  auto attributes = SrcNodeAttributes();
  bool keepdims = true;
  if (attributes.find("keepdims") != attributes.end() &&
      attributes.at("keepdims").has_i()) {
    keepdims = static_cast<bool>(attributes.at("keepdims").i());
  }

  ArgDef grad = GO(0);
  if (!keepdims && attributes.find("axes") != attributes.end()) {
    std::vector<int64_t> axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    grad = IA("Unsqueezed_Grad");
    result.push_back(NodeDef("Unsqueeze", {GO(0)}, {grad}, {MakeAttribute("axes", axes_values)}));

    result.push_back(NodeDef("Unsqueeze", {O(0)}, {IA("Unsqueezed_Output")}, {MakeAttribute("axes", axes_values)}));
    result.push_back(NodeDef("Sub", {I(0), IA("Unsqueezed_Output")}, {IA("Self_Sub_Result")}));
  } else {
    result.push_back(NodeDef("Sub", {I(0), O(0)}, {IA("Self_Sub_Result")}));
  }

  result.push_back(NodeDef("Exp", {IA("Self_Sub_Result")}, {IA("Self_Sub_Result_Exp")}));

  result.push_back(NodeDef("Mul", {IA("Self_Sub_Result_Exp"), grad}, {GI(0)}));

  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetReduceSumGradient) {
  std::vector<NodeDef> result;
  auto attributes = SrcNodeAttributes();
  bool keepdims = true;
  if (attributes.find("keepdims") != attributes.end() &&
      attributes.at("keepdims").has_i()) {
    keepdims = static_cast<bool>(attributes.at("keepdims").i());
  }

  ArgDef grad = GO(0);
  if (!keepdims && attributes.find("axes") != attributes.end()) {
    std::vector<int64_t> axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    grad = IA("Unqueezed_Grad");
    result.push_back(NodeDef("Unsqueeze", {GO(0)}, {grad}, {MakeAttribute("axes", axes_values)}));
  }

  result.push_back(NodeDef("Shape", {I(0)}, {IA("Shaped_X")}));
  result.push_back(NodeDef("Expand", {grad, IA("Shaped_X")}, {GI(0)}));
  return result;
}

}  // namespace training
}  // namespace onnxruntime
