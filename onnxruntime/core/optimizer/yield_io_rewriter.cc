// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/yield_io_rewriter.h"

namespace onnxruntime {

/**
  Reset the input and output names of the Yield op
 */
Status YieldIORewriter::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  auto& yield_inputs = node.MutableInputDefs();
  size_t i=0;
  for (auto* yield_input : yield_inputs) {
    auto current_name = yield_input->Name();
    auto expected_name = input_names_[i];
    if (current_name != expected_name) {
      // reset name to user supplied.      
      ORT_RETURN_IF_NOT(graph.GetNodeArg(expected_name) == nullptr, "NodeArg with name", expected_name ," already exists.");
      NodeArg& new_arg = graph.GetOrCreateNodeArg(expected_name, yield_input->TypeAsProto());
      
      // update producer and consumer nodeargs
      Node* producer_node = graph.GetMutableProducerNode(current_name);
      std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(current_name);

      int arg_index = graph_utils::GetNodeOutputIndexFromOutputName(*producer_node, current_name);
      producer_node->MutableOutputDefs()[arg_index] = &new_arg;

      for (Node* consumer: consumer_nodes){
        arg_index = graph_utils::GetNodeInputIndexFromInputName(*consumer, current_name);
        graph_utils::ReplaceNodeInput(*consumer, arg_index, new_arg);
      }

      rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;
    }
    i++;
  } 
  return Status::OK();
}

bool YieldIORewriter::SatisfyCondition(const Graph&, const Node& node, const logging::Logger&) const {
  auto yield_inputs_size = node.InputDefs().size();
  auto yield_outputs_size = node.OutputDefs().size();
  ORT_ENFORCE(yield_inputs_size == input_names_.size(), "Input count mismatch. The number of outputs between " ,
                                                              "the user's graph: " , input_names_.size() ,
                                                              " is different from the number of outputs in the ORT graph: " ,
                                                              yield_inputs_size);
  ORT_ENFORCE(yield_outputs_size == output_names_.size(), "Gradient input count mismatch. The number of grad inputs for backward between " ,
                                                              "the user's graph: " , output_names_.size() ,
                                                              " is different from the number expected in the ORT graph: " ,
                                                              yield_outputs_size);
  return true;
}

}  // namespace onnxruntime
