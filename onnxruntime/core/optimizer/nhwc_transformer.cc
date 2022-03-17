// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"
#include "core/framework/op_node_proto_helper.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;
#define ORT_RETURN_FALSE_IF_ERROR(expr) \
  do {                                  \
    auto _status = (expr);              \
    if ((!_status.IsOK())) {            \
      return false;                     \
    }                                   \
  } while (0)

namespace onnxruntime {

// This function runs before and after NhwcTransformer
bool NhwcTransformer::IsConvSupportedByXNNPack(const Node& nodeRef, bool input_is_nchw) {
  if (nodeRef.OpType() != "Conv" && nodeRef.OpType() != "NhwcConv") return false;
  // Conv has either 2 or 3 inputs.
  auto input_defs = nodeRef.InputDefs();
  if (input_defs.size() != 2 && input_defs.size() != 3) return false;

  // The two or three inputs are: X, W, B
  const NodeArg* weight_node_arg = input_defs[1];
  if (weight_node_arg == nullptr) return false;
  // Weight must be a const and all dims are known
  bool is_weight_shape_known = optimizer_utils::IsShapeKnownOnAllDims(*weight_node_arg, 4);
  if (!is_weight_shape_known) return false;

  ProtoHelperNodeContext nc(nodeRef);
  OpNodeProtoHelper info(&nc);
  auto X_input = input_defs[0]->TypeAsProto();
  if (X_input == nullptr || !X_input->has_tensor_type() || !X_input->tensor_type().has_shape() ||
      X_input->tensor_type().elem_type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return false;
  auto& input_shape = X_input->tensor_type().shape();
  if (input_shape.dim_size() != 4) return false;
  for (int i = 1; i != 3; ++i) {
    if (!input_shape.dim(i).has_dim_value()) {
      return false;
    }
  }
  auto weight_input = weight_node_arg->TypeAsProto();
  TensorShape weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_input->tensor_type().shape());
  TensorShape X_shape = utils::GetTensorShapeFromTensorShapeProto(X_input->tensor_type().shape());
  if (X_shape.NumDimensions() != 4) return false;
  int64_t group = 1;
  ORT_RETURN_FALSE_IF_ERROR(info.GetAttr<int64_t>("group", &group));
  int64_t input_channels = input_is_nchw ? X_shape[1] : X_shape[3];
  if (group != 1 && group != input_channels) return false;
  std::string auto_pad_str;
  ORT_RETURN_FALSE_IF_ERROR(info.GetAttr<std::string>("auto_pad", &auto_pad_str));
  if (auto_pad_str != "NOTSET" && auto_pad_str != "VALID" && auto_pad_str != "SAME") return false;
  std::vector<int64_t> pads;
  Status st = info.GetAttrs<int64_t>("pads", pads);
  if (st.IsOK()) {
    if (pads.size() != 4) return false;
  }
  return true;
}
Status NhwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
#if defined(ORT_MINIMAL_BUILD)
  // update the producer/consumer info as previous optimizations may have invalidated it.
  // in a full build this will happen as part of Graph::Resolve.
  ORT_RETURN_IF_ERROR(graph.PopulateNodeArgToProducerConsumerLookupsFromNodes());
#endif

  GraphViewer graph_viewer(graph);
  // Run constant propagation for XNNPack EP
  std::unordered_set<const NodeArg*> graph_const_values;

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    if (!node.ContainsSubgraph() && node.OpType() != "DequantizeLinear" && node.OpType() != "QuantizeLinear" &&
        optimizer_utils::IsOperationDeterministic(node.Domain(), node.OpType())) {
      bool is_all_const = true;
      for (const NodeArg* in : node.InputDefs()) {
        if (!in->Exists()) continue;
        if (graph_const_values.find(in) != graph_const_values.end()) continue;
        if (graph.GetConstantInitializer(in->Name(), false) != nullptr) {
          graph_const_values.insert(in);
          continue;
        }
        // This input is not const
        is_all_const = false;
      }
      if (is_all_const) {
        for (const NodeArg* out : node.OutputDefs()) {
          graph_const_values.insert(out);
        }
      }
    }
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  auto api_graph = MakeApiGraph(graph, cpu_allocator_, kCpuExecutionProvider);

  modified = false;
  for (std::unique_ptr<api::NodeRef>& node : api_graph->Nodes()) {
    // Only QLinearConv needs to be handled explicitly. The rest will be transformed if needed during transpose
    // optimization.
    if (node->OpType() == "QLinearConv") {
      auto domain = node->Domain();

      // Skip if domain is incorrect
      if (domain != kOnnxDomain && domain != kOnnxDomainAlias && domain != kMSDomain) {
        continue;
      }

      // Skip if already transformed
      if (node->GetAttributeIntDefault("channels_last", 0) == 1) {
        continue;
      }

      // Skip if unknown rank
      auto shape = NodeFromApiNode(*node).InputDefs()[0]->Shape();
      if (shape == nullptr) {
        continue;
      }

      // Convert to channels last
      size_t rank = shape->dim_size();
      node->SetAttributeInt("channels_last", 1);

      std::vector<int64_t> input_perm = ChannelFirstToLastPerm(rank);
      std::vector<int64_t> output_perm = ChannelLastToFirstPerm(rank);
      WrapTransposesAroundNode(*api_graph, *node, {&input_perm}, {&output_perm});

      if (domain != kMSDomain) {
        SwapNodeOpTypeAndDomain(*api_graph, *node, "QLinearConv", kMSDomain);
      }

      modified = true;
    }
    // Currently mlas doesn't NHWC Conv. So we only do the conversion when xnnpack is enabled
#ifdef USE_XNNPACK
    else if (IsConvSupportedByXNNPack(NodeFromApiNode(*node), true)) {
      auto domain = node->Domain();
      // Skip if domain is incorrect
      if (domain != kOnnxDomain && domain != kOnnxDomainAlias) {
        continue;
      }
      auto inputdefs = NodeFromApiNode(*node).InputDefs();
      if (inputdefs.size() != 2 && inputdefs.size() != 3) continue;
      if (!inputdefs[1]->Exists()) continue;

      if (graph_const_values.find(inputdefs[1]) == graph_const_values.end()) {
        // Weight is not const, we can't run it.
        continue;
      }
      // Skip if unknown rank
      auto shape = inputdefs[0]->Shape();
      if (shape == nullptr) {
        continue;
      }

      // Convert to channels last
      size_t rank = shape->dim_size();
      std::vector<int64_t> input_perm = onnx_layout_transformation::ChannelFirstToLastPerm(rank);
      std::vector<int64_t> output_perm = onnx_layout_transformation::ChannelLastToFirstPerm(rank);
      onnx_layout_transformation::WrapTransposesAroundNode(*api_graph, *node, {&input_perm}, {&output_perm});

      if (domain != kMSDomain) {
        auto inputs = node->Inputs();
        auto outputs = node->Outputs();
        auto new_node = api_graph->AddNode("NhwcConv", inputs, outputs.size(), kMSDomain, node->Name());
        for (size_t j = 0; j < outputs.size(); ++j) {
          if (outputs[j] != "") {
            api_graph->MoveOutput(*node, j, *new_node, j);
          }
        }
        new_node->CopyAttributes(*node);
        api_graph->RemoveNode(*node);
      }

      modified = true;
    }
#endif
  }

  if (modified) {
    Optimize(*api_graph, /*allow_extended_ops*/ true);
  }

  return Status::OK();
}

}  // namespace onnxruntime
