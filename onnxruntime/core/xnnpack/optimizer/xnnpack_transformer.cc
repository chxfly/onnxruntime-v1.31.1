// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/xnnpack/optimizer/xnnpack_transformer.h"

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/optimizer/selectors_actions/helpers.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"
#include "core/optimizer/nhwc_transformer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

Status XnnPackTransformer::ApplyImpl(Graph& main_graph, bool& modified, int /* graph_level */, const logging::Logger&) const {
  GraphViewer gv(main_graph);
  std::vector<NodeIndex> conv_nodes;
  for (auto& nodeRef : gv.Nodes()) {
    if (!NhwcTransformer::IsConvSupportedByXNNPack(nodeRef, false)) continue;
    if (nodeRef.OpType() != "NhwcConv") continue;
    conv_nodes.push_back(nodeRef.Index());
  }
  // Any error below is fatal, because if XNNPack couldn't run the node, we shouldn't convert it Nhwc.
  for (NodeIndex ni : conv_nodes) {
    Node* node_p = main_graph.GetNode(ni);
    if (node_p == nullptr)
      continue;
    Node& nodeRef = *node_p;
    ProtoHelperNodeContext nc(nodeRef);
    int64_t group = 1;
    OpNodeProtoHelper info(&nc);
    auto X_input = info.GetInputType(0);
    auto weight_input = info.GetInputType(1);
    TensorShape weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_input->tensor_type().shape());
    TensorShape X_shape = utils::GetTensorShapeFromTensorShapeProto(X_input->tensor_type().shape());
    if (weight_shape.NumDimensions() != 4) continue;
    if (group != 1 && group != X_shape[3]) continue;
    for (size_t i = 0; i != weight_shape.NumDimensions(); ++i) {
      if (weight_shape[i] <= 0) continue;
    }
    modified = true;
    // const_cast
    Node* conv_node = main_graph.GetNode(nodeRef.Index());
    NodeArg* bias_node_arg = nullptr;
    const bool has_bias = conv_node->InputDefs().size() >= 3;
    if (!has_bias) {
      ::ONNX_NAMESPACE::TensorProto bias_tensor;
      int64_t bias_size = weight_shape[0];
      std::vector<float> bias_data(bias_size, 0.0f);
      bias_tensor.mutable_float_data()->Add(bias_data.begin(), bias_data.end());
      bias_tensor.mutable_dims()->Add(bias_size);
      std::string bias_tensor_name = main_graph.GenerateNodeArgName(conv_node->Name() + "_bias");
      bias_tensor.set_name(bias_tensor_name);
      bias_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      bias_node_arg = &graph_utils::AddInitializer(main_graph, bias_tensor);
    }
    std::string auto_pad_str;
    ORT_RETURN_IF_ERROR(info.GetAttr<std::string>("auto_pad", &auto_pad_str));
    ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("group", &group));
    // group == 1 || group  == input / output channel count
    // For now we assume input channel count isn't 1, so that group count != input/output channel count
    bool is_depthwise = X_shape[3] != 1 && group == X_shape[3];

    InOutDefSlot src_slot{ArgType::kInput, 1};
    // Append a single slot to dest. As the dest is empty, it will be the first one.
    ValueMoveInfo value_move_info(src_slot, ArgType::kInput, false, false);

    if (conv_node == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Couldn't find the node with index=", nodeRef.Index());
    }
    if (conv_node->InputDefs().size() < 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expect at least 2 inputs, got ", conv_node->InputDefs().size());
    }

    // bool is_pointwise = weight_shape[2] == 1 && weight_shape[3] == 1;
    // if (!is_pointwise) {
    std::vector<int64_t> input_perm = is_depthwise ? std::vector<int64_t>{1, 2, 3, 0} : std::vector<int64_t>{0, 2, 3, 1};
    std::string output_name = main_graph.GenerateNodeArgName("trans");
    NodeArg& transpose_output = main_graph.GetOrCreateNodeArg(output_name, nullptr);

    Node& dest_node = main_graph.AddNode("", "Transpose", "", {}, {&transpose_output}, nullptr, kOnnxDomain);
    dest_node.AddAttribute("perm", input_perm);
    ORT_RETURN_IF_ERROR(MoveInputOutput(main_graph, *conv_node, dest_node, value_move_info, false));
    ORT_RETURN_IF_ERROR(main_graph.UpdateShapeInference(dest_node));
    main_graph.AddEdge(dest_node.Index(), conv_node->Index(), 0, 1);

    // onnx_layout_transformation::TransposesNodeInputs(*api_graph, *nodeRef, 1, input_perm);
    std::vector<int64_t> strides, dilations, pads;
    Status st = info.GetAttrs<int64_t>("strides", strides);
    if (!st.IsOK()) {
      // ONNX spec says: "If not present, the stride defaults is 1 along each spatial axis."
      strides.assign(4, 1);
    }
    st = info.GetAttrs<int64_t>("dilations", dilations);
    if (!st.IsOK()) {
      // ONNX spec says: "If not present, the dilation defaults is 1 along each spatial axis."
      dilations.assign(4, 1);
    }
    st = info.GetAttrs<int64_t>("pads", pads);
    if (!st.IsOK()) {
      // ONNX spec says: "If not present, the padding defaults to 0 along start and end of each spatial axis."
      pads.resize(4);
    }

    // auto inputs = nodeRef->Inputs();
    // auto outputs = nodeRef->Outputs();
    std::string node_name = conv_node->Name();
    Node& new_node = main_graph.AddNode(node_name, is_depthwise ? "XnnPackDepthwiseConvolution2d" : "XnnPackConvolution2d", "", conv_node->MutableInputDefs(), {},
                                        nullptr, "com.microsoft.xnnpack");
    // TODO: I'm not quite sure which is top, which is left
    new_node.AddAttribute("input_padding_top", pads[0]);
    new_node.AddAttribute("input_padding_right", pads[3]);
    new_node.AddAttribute("input_padding_bottom", pads[2]);
    new_node.AddAttribute("input_padding_left", pads[1]);

    new_node.AddAttribute("subsampling_height", strides[0]);
    new_node.AddAttribute("subsampling_width", strides[1]);

    new_node.AddAttribute("dilation_height", dilations[0]);
    new_node.AddAttribute("dilation_width", dilations[1]);

    if (!is_depthwise) new_node.AddAttribute("groups", group);

    // TODO: what is NOTSET?
    if (auto_pad_str == "NOTSET") {
      new_node.AddAttribute("padding_mode", static_cast<int64_t>(0));
    } else if (auto_pad_str == "VALID") {
      new_node.AddAttribute("padding_mode", static_cast<int64_t>(0));
    } else if (auto_pad_str == "SAME") {
      new_node.AddAttribute("padding_mode", static_cast<int64_t>(1));
    } else {
      return Status(ONNXRUNTIME, NOT_IMPLEMENTED);
    }

    ValueMoveInfo value_move_info2(InOutDefSlot{ArgType::kOutput, 0}, ArgType::kOutput, false, false);
    ORT_RETURN_IF_ERROR(MoveInputOutput(main_graph, *conv_node, new_node, value_move_info2, false));
    if (!main_graph.RemoveNode(ni)) {
      return Status(ONNXRUNTIME, FAIL, "remove node failed");
    }
    if (bias_node_arg != nullptr) {
      new_node.MutableInputDefs().push_back(bias_node_arg);
      new_node.MutableInputArgsCount().push_back(1);
    }
    // bool fused = false;

    if (optimizer_utils::CheckOutputEdges(main_graph, new_node, 1)) {
      const auto& next_node = *(new_node.OutputNodesBegin());
      float output_min;
      float output_max;

      bool has_clip = optimizer_utils::GetClipConstantMinMax(main_graph, next_node, output_min, output_max).IsOK();
      if (has_clip) {
        new_node.AddAttribute("output_min", output_min);
        new_node.AddAttribute("output_max", output_max);
        ValueMoveInfo value_move_info3(InOutDefSlot{ArgType::kOutput, 0}, InOutDefSlot{ArgType::kOutput, 0});
        // const_cast
        Node* clip_node = main_graph.GetNode(next_node.Index());
        ORT_RETURN_IF_ERROR(MoveInputOutput(main_graph, *clip_node, new_node, value_move_info3, false));
        if (!main_graph.RemoveNode(next_node.Index())) {
          return Status(ONNXRUNTIME, FAIL, "remove node failed");
        }
      }
    }
  }

  if (modified) {
    ORT_RETURN_IF_ERROR(main_graph.Resolve());
    auto api_graph = MakeApiGraph(main_graph, cpu_allocator_, kCpuExecutionProvider);
    // Ignore the return value.
    onnx_layout_transformation::Optimize(*api_graph, /*allow_extended_ops*/ true);
  }

  return Status::OK();
}

}  // namespace onnxruntime
