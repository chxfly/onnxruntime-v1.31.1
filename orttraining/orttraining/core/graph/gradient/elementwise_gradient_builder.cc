// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient/gradient_builder.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

IMPLEMENT_GRADIENT_BUILDER(GetCastGradient) {
  // TODO: handle invalid conversion cases
  const auto data_type = I(0).type_proto->tensor_type().elem_type();
  return std::vector<NodeDef>{
      NodeDef("Cast",
              {GO(0)},
              {GI(0)},
              {MakeAttribute("to", int64_t(data_type))})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSinGradient) {
  return std::vector<NodeDef>{
      NodeDef("SinGrad",
              {GO(0), I(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTanhGradient) {
  return std::vector<NodeDef>{
      NodeDef("TanhGrad",
              {O(0), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSqrtGradient) {
  return std::vector<NodeDef>{
      NodeDef("SqrtGrad",
              {O(0), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetErfGradient) {
  return std::vector<NodeDef>{
      NodeDef("ErfGrad",
              {I(0), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetAddSubGradient) {
  bool is_sub = (SrcNodeOpType() == "Sub");

  const ArgDef a = I(0), b = I(1);
  std::vector<NodeDef> output;
  std::vector<Dimension> a_shape, b_shape;
  if (GetShape(a, a_shape).IsOK() && GetShape(b, b_shape).IsOK()) {
    std::vector<int64_t> a_axes, b_axes;
    ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes);
    if (IsGradientRequiredForSrcNodeInput(0)) {
      if (a_axes.size() > 0) {
        HandleBroadcasting(GO(0), a, GI(0), a_axes, output);
      } else {
        output.push_back(
            NodeDef("Identity",
                    {GO(0)},
                    {GI(0)}));
      }
    }

    if (IsGradientRequiredForSrcNodeInput(1)) {
      if (b_axes.size() > 0) {
        ArgDef reshape_output = is_sub ? IA("ReshapeReduceSum_2", IType(1)) : GI(1);
        HandleBroadcasting(GO(0), b, reshape_output, b_axes, output);

        if (is_sub) {
          output.push_back(
              NodeDef("Neg",
                      {reshape_output},
                      {GI(1)}));
        }
      } else {
        if (is_sub) {
          output.push_back(
              NodeDef("Neg",
                      {GO(0)},
                      {GI(1)}));
        } else /*is_add*/ {
          output.push_back(
              NodeDef("Identity",
                      {GO(0)},
                      {GI(1)}));
        }
      }
    }
  } else {
    //GetShape failed, build shape-independent gradient graph
    ArgDef a_axes = IA("ReduceAxes_" + a.name);
    ArgDef b_axes = IA("ReduceAxes_" + b.name);
    ArgDef A_shape = IA("Shape_" + a.name);
    ArgDef B_shape = IA("Shape_" + b.name);
    ComputeBroadcastBackwardAxesDynamic(a, b, A_shape, B_shape, &a_axes, &b_axes, output);

    if (IsGradientRequiredForSrcNodeInput(0)) {
      HandleBroadcastingDynamic(GO(0), a, A_shape, GI(0), a_axes, output);
    }

    if (IsGradientRequiredForSrcNodeInput(1)) {
      ArgDef reshape_output = is_sub ? IA("ReshapeReduceSum_2", IType(1)) : GI(1);
      HandleBroadcastingDynamic(GO(0), b, B_shape, reshape_output, b_axes, output);

      if (is_sub) {
        output.push_back(
            NodeDef("Neg",
                    {reshape_output},
                    {GI(1)}));
      }
    }
  }
  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetMulGradient) {
  const ArgDef a = I(0), b = I(1);

  std::vector<NodeDef> output;
  std::vector<Dimension> a_shape, b_shape;
  if (GetShape(a, a_shape).IsOK() && GetShape(b, b_shape).IsOK()) {
    std::vector<int64_t> a_axes, b_axes;
    ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes);

    if (IsGradientRequiredForSrcNodeInput(0)) {
      output.push_back(
          NodeDef("Mul",
                  {GO(0), I(1)},
                  {IA("PreReduceGrad0", OType(0))}));

      if (a_axes.size() > 0) {
        HandleBroadcasting(IA("PreReduceGrad0", OType(0)), a, GI(0), a_axes, output);
      } else {
        output.push_back(
            NodeDef("Identity",
                    {IA("PreReduceGrad0", OType(0))},
                    {GI(0)}));
      }
    }

    if (IsGradientRequiredForSrcNodeInput(1)) {
      output.push_back(
          NodeDef("Mul",
                  {GO(0), I(0)},
                  {IA("PreReduceGrad1", OType(0))}));

      if (b_axes.size() > 0) {
        HandleBroadcasting(IA("PreReduceGrad1", OType(0)), b, GI(1), b_axes, output);
      } else {
        output.push_back(
            NodeDef("Identity",
                    {IA("PreReduceGrad1", OType(0))},
                    {GI(1)}));
      }
    }
  } else {
    //GetShape failed, build shape-independent gradient graph
    ArgDef a_axes = IA("ReduceAxes_" + a.name);
    ArgDef b_axes = IA("ReduceAxes_" + b.name);
    ArgDef A_shape = IA("Shape_" + a.name);
    ArgDef B_shape = IA("Shape_" + b.name);
    ComputeBroadcastBackwardAxesDynamic(a, b, A_shape, B_shape, &a_axes, &b_axes, output);

    if (IsGradientRequiredForSrcNodeInput(0)) {
      output.push_back(
          NodeDef("Mul",
                  {GO(0), I(1)},
                  {IA("PreReduceGrad0", OType(0))}));

      HandleBroadcastingDynamic(IA("PreReduceGrad0", OType(0)), a, A_shape, GI(0), a_axes, output);
    }

    if (IsGradientRequiredForSrcNodeInput(1)) {
      output.push_back(
          NodeDef("Mul",
                  {GO(0), I(0)},
                  {IA("PreReduceGrad1", OType(0))}));

      HandleBroadcastingDynamic(IA("PreReduceGrad1", OType(0)), b, B_shape, GI(1), b_axes, output);
    }
  }

  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetDivGradient) {
  if (IsGradientRequiredForSrcNodeInput(0) && IsGradientRequiredForSrcNodeInput(1)) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"DivGrad", kMSDomain, 1},
                {GO(0), I(0), I(1)},
                {GI(0), GI(1)})};
  } else if (IsGradientRequiredForSrcNodeInput(0)) {
    // Y = A / B, dA = dY / B
    const ArgDef a = I(0), b = I(1);
    std::vector<NodeDef> output;
    std::vector<Dimension> a_shape, b_shape;
    if (GetShape(a, a_shape).IsOK() && GetShape(b, b_shape).IsOK()) {
      std::vector<int64_t> a_axes, b_axes;
      ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes);

      ArgDef tmp_grad = IA("PreReduceGrad0", OType(0));
      output.push_back(NodeDef("Div", {GO(0), I(1)}, {tmp_grad}));
      if (a_axes.size() > 0) {
        HandleBroadcasting(tmp_grad, a, GI(0), a_axes, output);
      } else {
        output.push_back(NodeDef("Identity", {tmp_grad}, {GI(0)}));
      }
    } else {
      //GetShape failed, build shape-independent gradient graph
      ArgDef a_axes = IA("ReduceAxes_" + a.name);
      ArgDef A_shape = IA("Shape_" + a.name);
      ArgDef B_shape = IA("Shape_" + b.name);

      ComputeBroadcastBackwardAxesDynamic(a, b, A_shape, B_shape, &a_axes, nullptr, output);

      ArgDef tmp_grad = IA("PreReduceGrad0", OType(0));
      output.push_back(NodeDef("Div", {GO(0), I(1)}, {tmp_grad}));
      HandleBroadcastingDynamic(tmp_grad, a, A_shape, GI(0), a_axes, output);
    }

    return output;
  } else if (IsGradientRequiredForSrcNodeInput(1)) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"DivGrad", kMSDomain, 1},
                {GO(0), I(0), I(1)},
                // TODO: this IA("") does not cause kernel to know it is unneeded.
                // Gradient for the first input is still calculated.
                {IA(""), GI(1)})};
  } else {
    return std::vector<NodeDef>{};
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetPowGradient) {
  if (IsGradientRequiredForSrcNodeInput(1)) {
    ORT_THROW("GradientBuilder is not implemented for CUDA Pow's input exponent.");
  }
  return std::vector<NodeDef>{
      NodeDef("PowGrad",
              {GO(0), I(0), I(1)},
              {GI(0)})};
}


}  // namespace training
}  // namespace onnxruntime
