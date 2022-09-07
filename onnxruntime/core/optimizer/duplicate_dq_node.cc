
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include "core/common/inlined_containers.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_arg.h"

#include "core/optimizer/duplicate_dq_node.h"

namespace onnxruntime {
Status DuplicateDQTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                         const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed, this should not happen since we are not removing nodes.
      continue;

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // select DQ nodes with multiple outputs only
    if (node.OpType() != "DequantizeLinear" || node.GetOutputEdgesCount() < 2 ||
        graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    InlinedHashMap<const Node*, NodeArg*> consumer_nodes_to_def_map;
    auto old_output_name = node.OutputDefs()[0]->Name();

    int ind = 0;
    for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
      // DQ always produce one output, so it's safe to retrieve the first output
      std::string suffix = "_dup_" + std::to_string(ind);
      std::array output_defs{&graph.GetOrCreateNodeArg(it->InputDefs()[0]->Name() + suffix,
                                                       it->InputDefs()[0]->TypeAsProto())};
      Node& dup_dq_node = graph.AddNode(
          node.Name() + suffix,
          "DequantizeLinear",
          "duplicated DequantizeLinear node",
          node.MutableInputDefs(),
          output_defs,
          nullptr,
          kMSDomain);

      // Assign provider to this new node.
      dup_dq_node.SetExecutionProviderType(node.GetExecutionProviderType());
      dup_dq_node.GetMutableAttributes().insert(node.GetAttributes().begin(), node.GetAttributes().end());

      consumer_nodes_to_def_map[&(*it)] = output_defs[0];
      ind++;
    }

    // keep the connection of the original Dq NODE, so we can safely remove dq node.
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());

    // after removed dq node, we can re-construct the connection with duplicated dq node.
    for (auto [node_cs, out_def] : consumer_nodes_to_def_map) {
      auto& i_defs = graph.GetNode(node_cs->Index())->MutableInputDefs();
      std::replace_if(
          i_defs.begin(), i_defs.end(),
          [&old_output_name](NodeArg* arg) { return arg->Name() == old_output_name; }, out_def);
    }

    modified = true;
  }
  if (modified) {
    graph.SetGraphResolveNeeded();
  }
  return Status::OK();
}

}  // namespace onnxruntime
