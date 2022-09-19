// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/constant_sharing.h"
#include "core/optimizer/utils.h"
#include <random>
#include <chrono>
#include <thread>
#include <fstream>

namespace onnxruntime {

namespace {

// Supports limited data types.
InlinedVector<int32_t> supported_data_types{
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
    ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
    ONNX_NAMESPACE::TensorProto_DataType_INT32,
    ONNX_NAMESPACE::TensorProto_DataType_INT64,
};

bool IsSupportedDataType(int32_t data_type) {
  if (std::find(supported_data_types.begin(), supported_data_types.end(), data_type) == supported_data_types.end()) {
    return false;
  }

  return true;
}

bool IsSingleValueShape(const ONNX_NAMESPACE::TensorShapeProto* input_shape) {
  if (input_shape == nullptr) {
    return false;
  }

  size_t dim_size = static_cast<size_t>(input_shape->dim_size());
  if (dim_size == 0 ||
      (dim_size == 1 && utils::HasDimValue(input_shape->dim(0)) && input_shape->dim(0).dim_value() == 1)) {
    return true;
  }

  return false;
}

static constexpr char SHARED_INITIALIZER_PREFIX[] = "ortshared_";

std::string SharedInitializerNamePrefix() {
  return SHARED_INITIALIZER_PREFIX;
}

bool IsSharedInitializer(const NodeArg* arg) {
  return arg && (arg->Name().rfind(SHARED_INITIALIZER_PREFIX, 0) == 0);
}

// Replace all consumer nodes to use shared initializers.
void ReplaceInputsToUseSharedInitializer(Graph& graph,
                                         InlinedHashMap<const Node*, InlinedVector<int>>&
                                             consumer_node_to_input_index_map,
                                         const NodeArg* origin_initializer_node_arg,
                                         NodeArg* shared_initializer_node_arg,
                                         std::ofstream& myfile) {
  for (auto it = consumer_node_to_input_index_map.begin(); it != consumer_node_to_input_index_map.end(); ++it) {
    ORT_ENFORCE(it->first, "it->first should not be nullptr.");
    Node* node = const_cast<Node*>(it->first);
    ORT_ENFORCE(node, "Node should not be nullptr.");
    // Iterate all input defs to replace those that are equal to origin_initializer_node_arg,
    // Then it would be safe to remove the consumer node.
    for (int input_index : it->second) {
      myfile << "Replacing Node (name= " << node->Name() << ")'s " << input_index << " with " << shared_initializer_node_arg->Name() << std::endl;
      graph_utils::ReplaceNodeInput(*node, input_index, *shared_initializer_node_arg);
    }
    // graph.RemoveConsumerNode(origin_initializer_node_arg->Name(), node);

    // // Add consumer ref count for shared scalar initializer.
    // std::vector<const Node*> consumers = graph.GetConsumerNodes(shared_initializer_node_arg->Name());
    // if (std::find(consumers.begin(), consumers.end(), node) == consumers.end()) {
    //   graph.AddConsumerNode(shared_initializer_node_arg->Name(), node);
    // }
  }

  // // Remove the initializer if no other consumer nodes.
  // if (graph.GetConsumerNodes(origin_initializer_node_arg->Name()).size() == 0) {
  //   graph.RemoveInitializedTensor(origin_initializer_node_arg->Name());
  // }
}

int32_t GetNewValueKey() {
  static int32_t value_key_generator = 1;
  return value_key_generator++;
}

}  // namespace

Status ConstantSharing::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                  const logging::Logger& logger) const {
  // Accumulated map from type/value/rank to initializer:
  //   The key is a string representation of initializer's data type, value and rank.
  //   The value is newly created initializer NodeArg* to be shared.
  std::map<std::string, NodeArg*> type_value_plus_rank_to_shared_arg_map;

  // ORT_RETURN_IF_ERROR(graph.PopulateNodeArgToProducerConsumerLookupsFromNodes());

  const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  InlinedVector<std::string> original_initializer_names;
  original_initializer_names.reserve(initialized_tensor_set.size());
  std::transform(
      initialized_tensor_set.begin(), initialized_tensor_set.end(),
      std::back_inserter(original_initializer_names),
      [](const auto& v) { return v.first; });

  InlinedHashMap<int32_t, int32_t> int32_value_to_key_map;
  InlinedHashMap<int64_t, int32_t> int64_value_to_key_map;
  InlinedHashMap<float, int32_t> float_value_to_key_map;
  InlinedHashMap<float, int32_t> float16_value_to_key_map;
  InlinedHashMap<double, int32_t> double_value_to_key_map;
  for (const auto& initializer_name : original_initializer_names) {
    NodeArg* origin_initializer_node_arg = graph.GetNodeArg(initializer_name);

    // Already handled, skip it.
    if (origin_initializer_node_arg == nullptr || IsSharedInitializer(origin_initializer_node_arg)) {
      continue;
    }

    auto input_shape = origin_initializer_node_arg->Shape();
    if (input_shape == nullptr || !IsSingleValueShape(input_shape)) {
      continue;
    }

    // Ignore if not constant initializers.
    const ONNX_NAMESPACE::TensorProto* tensor_proto = graph.GetConstantInitializer(
        origin_initializer_node_arg->Name(), true);
    if (!tensor_proto ||
        excluded_initializers_.find(origin_initializer_node_arg->Name()) != excluded_initializers_.end()) {
      continue;
    }

    int32_t data_type = tensor_proto->data_type();
    if (!IsSupportedDataType(data_type)) {
      continue;
    }

    std::vector<const Node*> consumers = graph.GetConsumerNodes(origin_initializer_node_arg->Name());
    InlinedHashMap<const Node*, InlinedVector<int>> consumer_node_to_input_index_map;
    // If usage is from subgraph, skip it now, can be extended to support if there is a need.
    bool found_subgraph_usage = false;
    for (const Node* const_node : consumers) {
      for (int i = 0; i < static_cast<int>(const_node->ImplicitInputDefs().size()); ++i) {
        if (const_node->ImplicitInputDefs()[i] == origin_initializer_node_arg) {
          found_subgraph_usage = true;
          break;
        }
      }

      if (found_subgraph_usage) {
        break;
      }

      // Iterate all input defs to replace those that are equal to origin_initializer_node_arg,
      // Then it would be safe to remove the consumer node aferwards.
      for (int i = 0; i < static_cast<int>(const_node->InputDefs().size()); ++i) {
        if (const_node->InputDefs()[i] == origin_initializer_node_arg) {
          consumer_node_to_input_index_map[const_node].push_back(i);
        }
      }
    }

    if (found_subgraph_usage || consumer_node_to_input_index_map.size() == 0) {
      continue;
    }
    std::string dp = ToUTF8String(graph.ModelPath().ToPathString());
    LOGS(logger, WARNING) << "handling initializer_name:" << initializer_name << " for " << dp;
    onnxruntime::Initializer initializer{*tensor_proto, graph.ModelPath()};

    std::string value_key;
    // std::ostringstream pattern_key_oss;
    // pattern_key_oss << data_type << origin_initializer_node_arg->Shape()->dim_size();

    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float flt_val = *initializer.data<float>();
        if (float_value_to_key_map.find(flt_val) == float_value_to_key_map.end()) {
          float_value_to_key_map[flt_val] = GetNewValueKey();
        }
        value_key = float_value_to_key_map[flt_val];
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        float flt16_val = math::halfToFloat(initializer.data<MLFloat16>()->val);
        if (float16_value_to_key_map.find(flt16_val) == float16_value_to_key_map.end()) {
          float16_value_to_key_map[flt16_val] = GetNewValueKey();
        }
        value_key = float16_value_to_key_map[flt16_val];
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double dbl_val = *initializer.data<double>();
        if (double_value_to_key_map.find(dbl_val) == double_value_to_key_map.end()) {
          double_value_to_key_map[dbl_val] = GetNewValueKey();
        }
        value_key = double_value_to_key_map[dbl_val];
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        int32_t int32_val = *initializer.data<int32_t>();
        if (int32_value_to_key_map.find(int32_val) == int32_value_to_key_map.end()) {
          int32_value_to_key_map[int32_val] = GetNewValueKey();
        }
        value_key = int32_value_to_key_map[int32_val];
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t int64_val = *initializer.data<int64_t>();
        if (int64_value_to_key_map.find(int64_val) == int64_value_to_key_map.end()) {
          int64_value_to_key_map[int64_val] = GetNewValueKey();
        }
        value_key = int64_value_to_key_map[int64_val];
      } break;
      default:
        ORT_THROW("Should not go here");
    }

    LOGS(logger, WARNING) << "handling  1111111111111111 initializer_name:" << initializer_name << " for " << dp;
    std::ostringstream pattern_key_oss;
    pattern_key_oss << value_key << "_" << data_type << "_" << origin_initializer_node_arg->Shape()->dim_size();
    const std::string pattern_key = pattern_key_oss.str();
    // If there is no such existing scalar pattern, add a new one.
    if (type_value_plus_rank_to_shared_arg_map.find(pattern_key) ==
        type_value_plus_rank_to_shared_arg_map.end()) {
      // ONNX_NAMESPACE::TensorProto constant_tensor_proto_as_replacement;

      ONNX_NAMESPACE::TensorProto constant_tensor_proto_as_replacement(*tensor_proto);
      // initializer.ToProto(constant_tensor_proto_as_replacement);
      constant_tensor_proto_as_replacement.set_name(graph.GenerateNodeArgName(pattern_key));
      NodeArg& shared_scalar_initializer_node_arg =
          graph_utils::AddInitializer(graph, constant_tensor_proto_as_replacement);
      type_value_plus_rank_to_shared_arg_map[pattern_key] = &shared_scalar_initializer_node_arg;
    }

    ORT_ENFORCE(type_value_plus_rank_to_shared_arg_map[pattern_key], "type_value_plus_rank_to_shared_arg_map[pattern_key] should not be null.");

    LOGS(logger, WARNING) << "handling  222222222222222 initializer_name:" << initializer_name << " for " << dp;

    // std::random_device rd;                          // obtain a random number from hardware
    // std::mt19937 gen(rd());                         // seed the generator
    // std::uniform_int_distribution<> distr(50, 500);  // define the range

    // std::this_thread::sleep_for(std::chrono::milliseconds(distr(gen)));
    // Replace the scalar reference using existing shared one.
    static std::ofstream myfile("C:\\Users\\pengwa\\dev\\onnxruntime\\build\\Windows\\Debug\\2012pengwa.log");
    ReplaceInputsToUseSharedInitializer(graph, consumer_node_to_input_index_map, origin_initializer_node_arg,
                                        type_value_plus_rank_to_shared_arg_map[pattern_key], myfile);

    modified = true;
    LOGS(logger, WARNING) << "done handling initializer_name:" << initializer_name << " for " << dp;
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif
