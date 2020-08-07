// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient/gradient_builder.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

IMPLEMENT_GRADIENT_BUILDER(GetMatMulGradient) {
  std::vector<NodeDef> result;

  ArgDef A = I(0), B = I(1), Y = O(0);
  std::vector<Dimension> A_shape, B_shape, Y_shape;
  if (GetShape(A, A_shape).IsOK() && GetShape(B, B_shape).IsOK() && GetShape(Y, Y_shape).IsOK()) {
    std::vector<AttributeProto> shared_attributes;
    shared_attributes.push_back(MakeAttribute("beta", float(0)));
    AttributeProto transpose_first_input = MakeAttribute("transA", int64_t(1));
    AttributeProto transpose_second_input = MakeAttribute("transB", int64_t(1));

    if (A_shape.size() == 2 && B_shape.size() == 2) {
      NodeDef zero_constant_node = ZeroConstantNode();
      ArgDef ZERO = zero_constant_node.output_args[0];
      result.push_back(zero_constant_node);

      // is GI(0) required
      if (IsGradientRequiredForSrcNodeInput(0)) {
        // dA = dY * B'
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_second_input);
        result.push_back(
            NodeDef("Gemm",
                    {GO(0), B, ZERO},
                    {GI(0)},
                    attrs));
      }

      // is GI(1) required
      if (IsGradientRequiredForSrcNodeInput(1)) {
        // dB = A' * dY
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        result.push_back(
            NodeDef("Gemm",
                    {A, GO(0), ZERO},
                    {GI(1)},
                    attrs));
      }
    } else if (A_shape.size() > 2 || B_shape.size() > 2) {
      if (IsGradientRequiredForSrcNodeInput(0)) {
        // If B_shape.size() == 2, dA is computed through 2 ops: transpose and matmul.
        // It can be replaced with Gemm(dY_reshape, B_transpose) and reshape.
        // However, there is a performance degradation.
        // Thus this implementation is not implemented.
        int64_t B_rank = B_shape.size();
        std::vector<int64_t> B_perm(B_rank);
        std::iota(B_perm.begin(), B_perm.end(), 0);
        std::swap(B_perm[B_rank - 1], B_perm[B_rank - 2]);

        std::vector<Dimension> output_shape;
        for (size_t i = 0; i < Y_shape.size() - 1; i++) {
          output_shape.push_back(Y_shape[i]);
        }
        output_shape.push_back(B_shape[B_shape.size() - 2]);

        std::vector<int64_t> A_axes;
        ComputeBroadcastBackwardAxes(A_shape, output_shape, &A_axes, nullptr);

        result.push_back(
            NodeDef("Transpose",
                    {B},
                    {IA("B_t")},
                    {MakeAttribute("perm", B_perm)}));

        ArgDef matmul_out = A_axes.size() > 0 ? IA("PreReduceGrad0") : GI(0);

        result.push_back(
            NodeDef("MatMul",
                    {GO(0), IA("B_t")},
                    {matmul_out}));

        if (A_axes.size() > 0) {
          result.push_back(
              NodeDef("ReduceSum",
                      {IA("PreReduceGrad0")},
                      {IA("ReduceGrad0")},
                      {{"keepdims", MakeAttribute("keepdims", int64_t(1))},
                       {"axes", MakeAttribute("axes", A_axes)}}));

          result.push_back(
              NodeDef("Shape",
                      {A},
                      {IA("A_shape")}));

          result.push_back(
              NodeDef("Reshape",
                      {IA("ReduceGrad0"), IA("A_shape")},
                      {GI(0)}));
        }
      }
      if (IsGradientRequiredForSrcNodeInput(1)) {
        if (B_shape.size() == 2 &&
            (B_shape[0].has_dim_value() || A_shape[A_shape.size() - 1].has_dim_value()) &&
            (B_shape[1].has_dim_value() || Y_shape[Y_shape.size() - 1].has_dim_value())) {
          // A[M, K], B[K, N], Y[M, N]
          int64_t K, N;
          if (B_shape[0].has_dim_value()) {
            K = B_shape[0].dim_value();
          } else {
            K = A_shape[A_shape.size() - 1].dim_value();
          }
          if (B_shape[1].has_dim_value()) {
            N = B_shape[1].dim_value();
          } else {
            N = Y_shape[Y_shape.size() - 1].dim_value();
          }

          std::vector<int64_t> A_shape_2d{-1, K};
          NodeDef A_shape_2d_node = ConstantValueNode(A_shape_2d, Name("A_shape_2d"));
          ArgDef A_shape_2d_arg = A_shape_2d_node.output_args[0];
          result.push_back(A_shape_2d_node);

          std::vector<int64_t> dY_shape_2d{-1, N};
          NodeDef dY_shape_2d_node = ConstantValueNode(dY_shape_2d, Name("dY_shape_2d"));
          ArgDef dY_shape_2d_arg = dY_shape_2d_node.output_args[0];
          result.push_back(dY_shape_2d_node);

          NodeDef zero_constant_node = ZeroConstantNode();
          ArgDef ZERO = zero_constant_node.output_args[0];
          result.push_back(zero_constant_node);

          result.push_back(
              NodeDef("Reshape",
                      {A, A_shape_2d_arg},
                      {IA("A_reshape_2d")}));
          result.push_back(
              NodeDef("Reshape",
                      {GO(0), dY_shape_2d_arg},
                      {IA("dY_reshape_2d")}));

          // dB = A' * dY
          std::vector<AttributeProto> attrs(shared_attributes);
          attrs.push_back(transpose_first_input);
          result.push_back(
              NodeDef("Gemm",
                      {IA("A_reshape_2d"), IA("dY_reshape_2d"), ZERO},
                      {GI(1)},
                      attrs));
        } else {
          int64_t A_rank = A_shape.size();
          std::vector<int64_t> A_perm(A_rank);
          std::iota(A_perm.begin(), A_perm.end(), 0);
          std::swap(A_perm[A_rank - 1], A_perm[A_rank - 2]);

          std::vector<Dimension> output_shape;
          for (size_t i = 0; i < Y_shape.size() - 2; i++) {
            output_shape.push_back(Y_shape[i]);
          }
          output_shape.push_back(A_shape[A_shape.size() - 1]);
          output_shape.push_back(Y_shape[Y_shape.size() - 1]);

          std::vector<int64_t> B_axes;
          ComputeBroadcastBackwardAxes(B_shape, output_shape, &B_axes, nullptr);

          result.push_back(
              NodeDef("Transpose",
                      {A},
                      {IA("A_t")},
                      {MakeAttribute("perm", A_perm)}));

          ArgDef matmul_out = B_axes.size() > 0 ? IA("PreReduceGrad1") : GI(1);

          result.push_back(
              NodeDef("MatMul",
                      {IA("A_t"), GO(0)},
                      {matmul_out}));

          if (B_axes.size() > 0) {
            result.push_back(
                NodeDef("ReduceSum",
                        {IA("PreReduceGrad1")},
                        {IA("ReduceGrad1")},
                        {{"keepdims", MakeAttribute("keepdims", int64_t(0))},
                         {"axes", MakeAttribute("axes", B_axes)}}));
            result.push_back(
                NodeDef("Shape",
                        {B},
                        {IA("B_shape")}));
            result.push_back(
                NodeDef("Reshape",
                        {IA("ReduceGrad1"), IA("B_shape")},
                        {GI(1)}));
          }
        }
      }
    }
  } else {
    //GetShape failed, build shape-independent gradient graph
    ArgDef a_axes, b_axes, a_shape, b_shape, ia_shape;
    a_shape = IA("Shape_" + A.name);
    b_shape = IA("Shape_" + B.name);

    if (IsGradientRequiredForSrcNodeInput(0)) {
      ArgDef pre_reduce_grad_0 = IA("PreReduceGrad0");
      result.push_back(
          NodeDef(OpDef{"TransposeScaleMatMul", kMSDomain, 1},
                  {GO(0), B},
                  {pre_reduce_grad_0},
                  {{"transB", MakeAttribute("transB", int64_t(1))}}));

      a_axes = IA("ReduceAxes_" + A.name + "_for_" + A.name);
      ia_shape = IA("Shape_" + pre_reduce_grad_0.name);
      ComputeBroadcastBackwardAxesDynamic(A, pre_reduce_grad_0, a_shape, ia_shape, &a_axes, nullptr, result);
      HandleBroadcastingDynamic(pre_reduce_grad_0, A, a_shape, GI(0), a_axes, result);
    }
    if (IsGradientRequiredForSrcNodeInput(1)) {
      ArgDef pre_reduce_grad_1 = IA("PreReduceGrad1");
      result.push_back(
          NodeDef(OpDef{"TransposeScaleMatMul", kMSDomain, 1},
                  {A, GO(0)},
                  {pre_reduce_grad_1},
                  {{"transA", MakeAttribute("transA", int64_t(1))}}));

      b_axes = IA("ReduceAxes_" + B.name + "_for_" + B.name);
      ia_shape = IA("Shape_" + pre_reduce_grad_1.name);
      ComputeBroadcastBackwardAxesDynamic(pre_reduce_grad_1, B, ia_shape, b_shape, nullptr, &b_axes, result);
      HandleBroadcastingDynamic(pre_reduce_grad_1, B, b_shape, GI(1), b_axes, result);
    }
  }

  return result;
};

IMPLEMENT_GRADIENT_BUILDER(GetGemmGradient) {
  auto attributes = SrcNodeAttributes();

  bool has_alpha = attributes.at("alpha").has_f();
  float alpha = attributes.at("alpha").f();
  bool transA = static_cast<bool>(attributes.at("transA").i());
  bool transB = static_cast<bool>(attributes.at("transB").i());

  ArgDef A = I(0), B = I(1), C = I(2), dY = GO(0),
         dA = GI(0), dB = GI(1), dC = GI(2);
  AttributeProto transpose_first_input = MakeAttribute("transA", int64_t(1));
  AttributeProto transpose_second_input = MakeAttribute("transB", int64_t(1));

  NodeDef zero_contant_node = ZeroConstantNode();
  ArgDef ZERO = zero_contant_node.output_args[0];

  std::vector<NodeDef> result;
  result.push_back(zero_contant_node);

  std::vector<AttributeProto> shared_attributes;
  shared_attributes.push_back(MakeAttribute("beta", float(0)));
  if (has_alpha && alpha != 1.0f) {
    ORT_ENFORCE(alpha != 0.0f);
    AttributeProto alpha_attr = MakeAttribute("alpha", alpha);
    shared_attributes.push_back(alpha_attr);
  }

  if (transA) {
    if (transB) {
      // Y = alpha * A' * B'
      // dA = alpha * B' * dY', dB = alpha *  dY' * A'
      if (IsGradientRequiredForSrcNodeInput(0)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {B, dY, ZERO}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {dY, A, ZERO}, {dB}, attrs));
      }
    } else {
      // Y = alpha * A' * B
      // dA = alpha * B * dY', dB = alpha * A * dY
      if (IsGradientRequiredForSrcNodeInput(0)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {B, dY, ZERO}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        result.push_back(NodeDef("Gemm", {A, dY, ZERO}, {dB}, shared_attributes));
      }
    }
  } else {
    if (transB) {
      // Y = alpha * A * B'
      // dA = alpha * dY * B, dB = alpha * dY' * A
      if (IsGradientRequiredForSrcNodeInput(0)) {
        result.push_back(NodeDef("Gemm", {dY, B, ZERO}, {dA}, shared_attributes));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        result.push_back(NodeDef("Gemm", {dY, A, ZERO}, {dB}, attrs));
      }
    } else {
      // Y = alpha * A * B
      // dA = alpha * dY * B', dB = alpha * A' * dY
      if (IsGradientRequiredForSrcNodeInput(0)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {dY, B, ZERO}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        result.push_back(NodeDef("Gemm", {A, dY, ZERO}, {dB}, attrs));
      }
    }
  }

  if (IsGradientRequiredForSrcNodeInput(2)) {
    // Y = beta * C
    // dC = beta * dY
    bool has_beta = attributes.at("beta").has_f();
    float beta = attributes.at("beta").f();
    ORT_ENFORCE(beta != 0.0f);
    std::vector<Dimension> C_shape, dY_shape;
    if (GetShape(C, C_shape).IsOK() && GetShape(dY, dY_shape).IsOK()) {
      std::vector<int64_t> C_axes, dY_axes;
      ComputeBroadcastBackwardAxes(C_shape, dY_shape, &C_axes, &dY_axes);

      if (C_axes.size() > 0) {
        HandleBroadcasting(dY, C, IA("dC_reduced"), C_axes, result);

        if (has_beta && beta != 1.0f) {
          NodeDef scale_node = ConstantValueNode(beta, Name("Scale"));
          ArgDef SCALE = scale_node.output_args[0];
          result.push_back(scale_node);
          result.push_back(
              NodeDef("Mul",
                      {IA("dC_reduced"), SCALE},
                      {dC}));
        } else {
          result.push_back(
              NodeDef("Identity", {IA("dC_reduced")}, {dC}));
        }
      } else {
        if (has_beta && beta != 1.0f) {
          NodeDef scale_node = ConstantValueNode(beta, Name("Scale"));
          ArgDef SCALE = scale_node.output_args[0];
          result.push_back(scale_node);
          result.push_back(
              NodeDef("Mul",
                      {dY, SCALE},
                      {dC}));
        } else {
          result.push_back(
              NodeDef("Identity",
                      {dY},
                      {dC}));
        }
      }
    } else {
      //GetShape failed, build shape-independent gradient graph
      ArgDef c_axes = IA("ReduceAxes_" + C.name);
      ArgDef c_shape = IA("Shape_" + C.name);
      ArgDef dy_shape = IA("Shape_" + dY.name);

      ComputeBroadcastBackwardAxesDynamic(C, dY, c_shape, dy_shape, &c_axes, nullptr, result);

      HandleBroadcastingDynamic(dY, C, c_shape, IA("dC_reduced"), c_axes, result);

      if (has_beta && beta != 1.0f) {
        NodeDef scale_node = ConstantValueNode(beta, Name("Scale"));
        ArgDef SCALE = scale_node.output_args[0];
        result.push_back(scale_node);
        result.push_back(
            NodeDef("Mul",
                    {IA("dC_reduced"), SCALE},
                    {dC}));
      } else {
        result.push_back(
            NodeDef("Identity", {IA("dC_reduced")}, {dC}));
      }
    }
  }
  return result;
}

}  // namespace training
}  // namespace onnxruntime
