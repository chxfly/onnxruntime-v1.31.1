// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient/gradient_builder.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

static bool SimplifyReshape(const std::vector<Dimension>& target_shape,  // the output shape of Reshape
                            const std::vector<Dimension>& source_shape,  // the input shape of Reshape
                            TensorProto& shape_tensor) {                 // the simplified shape tensor if succeeded
  std::vector<int64_t> shape_const;
  std::list<std::string> target_dim_params;
  std::list<std::string> source_dim_params;
  auto get_dim_params = [](const std::vector<Dimension>& shape, std::list<std::string>& dim_params) {
    for (const auto& dim : shape) {
      if (utils::HasDimParam(dim)) {
        dim_params.push_back(dim.dim_param());
      } else if (utils::HasDimValue(dim)) {
        dim_params.push_back("");
      } else {
        return false;
      }
    }
    //trim empty strings in the tail of list
    while (!dim_params.empty() && dim_params.back().empty()) {
      dim_params.pop_back();
    }
    return true;
  };

  if (get_dim_params(target_shape, target_dim_params) &&
      get_dim_params(source_shape, source_dim_params) &&
      target_dim_params == source_dim_params) {
    for (const auto& dim : target_shape) {
      if (utils::HasDimParam(dim)) {
        shape_const.push_back(0);
      } else {
        shape_const.push_back(dim.dim_value());
      }
    }
    auto t = ToTensor<int64_t>(shape_const);
    t.add_dims(shape_const.size());
    shape_tensor.CopyFrom(t);
    return true;
  }
  return false;
}

IMPLEMENT_GRADIENT_BUILDER(GetReshapeGradient) {
  std::vector<Dimension> target_shape;
  std::vector<Dimension> source_shape;
  if (GetShape(I(0), target_shape).IsOK() &&
      GetShape(GO(0), source_shape).IsOK()) {
    TensorProto shape_tensor;
    if (SimplifyReshape(target_shape,
                        source_shape,
                        shape_tensor)) {
      return std::vector<NodeDef>{
          NodeDef("Constant",
                  {},
                  {IA("x_shape")},
                  {MakeAttribute("value", shape_tensor)}),
          NodeDef("Reshape",
                  {GO(0), IA("x_shape")},
                  {GI(0)})};
    }
  }
  return std::vector<NodeDef>{
      NodeDef("ReshapeGrad",
              {I(0), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTransposeGradient) {
  std::vector<int64_t> bw_perm;
  auto attributes = SrcNodeAttributes();
  std::vector<AttributeProto> new_attributes;
  if (attributes.empty()) {
    const TensorShapeProto& input_shape = I(0).type_proto->tensor_type().shape();
    if (input_shape.dim_size() > 0) {  //input_shape is available
      int n = input_shape.dim_size() - 1;
      bw_perm.resize(n + 1);
      std::generate(bw_perm.begin(), bw_perm.end(), [&n] { return n--; });
      new_attributes.push_back(MakeAttribute("perm", bw_perm));
    }
  } else {
    auto fw_perm = RetrieveValues<int64_t>(attributes.at("perm"));
    auto size = fw_perm.size();
    bw_perm.resize(size);
    for (int i = 0; i < static_cast<int>(size); ++i) {
      bw_perm[fw_perm[i]] = i;
    }
    new_attributes.push_back(MakeAttribute("perm", bw_perm));
  }

  return std::vector<NodeDef>{
      NodeDef("Transpose",
              {GO(0)},
              {GI(0)},
              new_attributes)};
}

IMPLEMENT_GRADIENT_BUILDER(GetSplitGradient) {
  std::vector<NodeDef> result = {};
  std::vector<ArgDef> input_args;

  for (int i = 0; i < GetSrcNodeOutputSize(); i++) {
    if (IsGradientAvailableForSrcNodeOutput(i)) {
      input_args.push_back(GO(i));
    }
  }

  if (!input_args.empty()) {
    auto attributes = SrcNodeAttributes();
    ORT_ENFORCE(attributes.at("axis").has_i());
    auto axis = attributes.at("axis").i();
    result.push_back(
        NodeDef("Concat",
                input_args,
                {GI(0)},
                {MakeAttribute("axis", axis)}));
  }
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetConcatGradient) {
  auto attributes = SrcNodeAttributes();
  ORT_ENFORCE(attributes.at("axis").has_i());
  auto axis = attributes.at("axis").i();

  std::vector<int64_t> split_attribute(GetSrcNodeInputSize());
  std::vector<ArgDef> outputs;
  for (int i = 0; i < GetSrcNodeInputSize(); ++i) {
    std::vector<Dimension> data_shape;
    ORT_ENFORCE(GetShape(I(i), data_shape).IsOK());
    int64_t axis_index = axis < 0 ? static_cast<int64_t>(data_shape.size()) + axis : axis;
    if (axis_index >= 0 && axis_index < static_cast<int64_t>(data_shape.size()) && data_shape[axis_index].has_dim_value()) {
      split_attribute[i] = data_shape[axis_index].dim_value();
    } else {
      ORT_THROW("Error: can't infer split attribute value for ConcatGrad");
    }
    outputs.push_back(GI(i));
  }

  std::vector<AttributeProto> new_attributes;
  new_attributes.push_back(MakeAttribute("axis", axis));
  new_attributes.push_back(MakeAttribute("split", split_attribute));

  return std::vector<NodeDef>{
      NodeDef("Split",
              {GO(0)},
              outputs,
              new_attributes)};
}

IMPLEMENT_GRADIENT_BUILDER(GetConcatTrainingGradient) {
  auto attributes = SrcNodeAttributes();
  ORT_ENFORCE(utils::HasInt(attributes.at("axis")));
  auto axis = attributes.at("axis").i();

  std::vector<int64_t> split_attribute(GetSrcNodeInputSize());
  std::vector<ArgDef> outputs;
  bool known_shapes = true;
  for (int i = 0; i < GetSrcNodeInputSize(); ++i) {
    std::vector<Dimension> data_shape;
    if (GetShape(I(i), data_shape).IsOK()) {
      int64_t rank = static_cast<int64_t>(data_shape.size());
      int64_t axis_index = HandleNegativeAxis(axis, rank);
      if (data_shape[axis_index].has_dim_value()) {
        split_attribute[i] = data_shape[axis_index].dim_value();
      } else {
        known_shapes = false;
      }
    } else {
      known_shapes = false;
    }

    outputs.push_back(GI(i));
  }

  std::vector<AttributeProto> new_attributes;
  new_attributes.push_back(MakeAttribute("axis", axis));
  if (known_shapes) {
    new_attributes.push_back(MakeAttribute("split", split_attribute));
    return std::vector<NodeDef>{
        NodeDef("Split",
                {GO(0)},
                outputs,
                new_attributes)};
  } else {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SplitTraining", kMSDomain, 1},
                {GO(0), O(1)},
                outputs,
                new_attributes)};
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetGatherGradient) {
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("I0_shape")}),
      NodeDef(OpDef{"GatherGrad", kMSDomain, 1},
              {IA("I0_shape"), I(1), GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetGatherElementsGradient) {
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("x_shape")}),
      NodeDef(OpDef{"GatherElementsGrad", kMSDomain, 1},
              {GO(0), IA("x_shape"), I(1)},
              {GI(0)},
              SrcNodeAttributes())};
};

IMPLEMENT_GRADIENT_BUILDER(GetGatherNDGradient) {
  auto attributes = SrcNodeAttributes();
  ORT_ENFORCE(attributes.at("batch_dims").has_i());
  auto batch_dims = attributes.at("batch_dims").i();
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("x_shape")}),
      NodeDef(OpDef{"GatherNDGrad", kMSDomain, 1},
              {IA("x_shape"), I(1), GO(0)},
              {GI(0)},
              {MakeAttribute("batch_dims", batch_dims)})};
};

IMPLEMENT_GRADIENT_BUILDER(GetSqueezeGradient) {
  std::vector<NodeDef> result;
  auto attributes = SrcNodeAttributes();
  std::vector<int64_t> axes_values;
  if (attributes.find("axes") != attributes.end()) {
    axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    result.push_back(
        NodeDef("Unsqueeze",
                {GO(0)},
                {GI(0)},
                {MakeAttribute("axes", axes_values)}));
    // if axes attribute not provided for squeeze
  } else {
    result.push_back(
        NodeDef("Shape",
                {I(0)},
                {IA("I0_shape")}));
    result.push_back(
        NodeDef("Reshape",
                {GO(0), IA("I0_shape")},
                {GI(0)}));
  }
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetUnsqueezeGradient) {
  return std::vector<NodeDef>{
      NodeDef("Squeeze",
              {GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetSliceGradient) {
  std::vector<ArgDef> inputs{GO(0), IA("I0_shape")};
  for (int i = 1; i < GetSrcNodeInputSize(); i++) {
    inputs.push_back(I(i));
  }

  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("I0_shape")}),
      NodeDef(OpDef{"SliceGrad", kMSDomain, 1}, inputs, {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetExpandGradient) {
  ArgDef a = I(0), y = O(0);
  std::vector<NodeDef> output;

  std::vector<Dimension> a_shape, y_shape;
  if (GetShape(a, a_shape).IsOK() && GetShape(y, y_shape).IsOK()) {
    std::vector<int64_t> a_axes;
    ComputeBroadcastBackwardAxes(a_shape, y_shape, &a_axes, nullptr);

    if (a_axes.size() > 0) {
      HandleBroadcasting(GO(0), a, GI(0), a_axes, output);
    } else {
      output.push_back(
          NodeDef("Identity",
                  {GO(0)},
                  {GI(0)}));
    }
  } else {
    //GetShape failed, build shape-independent gradient graph
    ArgDef a_axes = IA("ReduceAxes_" + a.name);
    ArgDef A_shape = IA("Shape_" + a.name);
    ArgDef Y_shape = IA("Shape_" + y.name);

    ComputeBroadcastBackwardAxesDynamic(a, y, A_shape, Y_shape, &a_axes, nullptr, output);

    HandleBroadcastingDynamic(GO(0), a, A_shape, GI(0), a_axes, output);
  }

  return output;
}
}  // namespace training
}  // namespace onnxruntime
