// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "orttraining/core/framework/module_gradient_graph_builder.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"

namespace onnxruntime {
namespace training {

using namespace onnxruntime::common;

Status ModuleGradientGraphBuilder::Initialize(std::istream& model_istream,
                                              const ModuleGradientGraphBuilderConfiguration& config) {
  // Save the model and config.
  ONNX_NAMESPACE::ModelProto model_proto;
  ORT_RETURN_IF_ERROR(Model::Load(model_istream, &model_proto));
  ORT_RETURN_IF_ERROR(Model::Load(model_proto, model_, nullptr, *logger_));
  config_ = config;

  // Handle original model inputs, outputs and trainable initializers.
  // We need to move the trainable initializers to graph inputs and keep the order in config,
  // it's possible that the graph already has some trainable initializers in graph inputs,
  // so we need to NOT assign these trainable initializers to the user inputs list.
  Graph& graph = model_->MainGraph();
  std::unordered_set<std::string> initializer_names_to_train_set(config.initializer_names_to_train.begin(),
                                                                 config.initializer_names_to_train.end());
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  for (auto& node_arg : graph_inputs) {
    if (initializer_names_to_train_set.find(node_arg->Name()) == initializer_names_to_train_set.end()) {
      split_graphs_info_.user_input_names.emplace_back(node_arg->Name());
    }
  }

  const std::vector<const NodeArg*>& graph_outputs = graph.GetOutputs();
  for (auto& node_arg : graph_outputs) {
    split_graphs_info_.user_output_names.emplace_back(node_arg->Name());
  }

  split_graphs_info_.initializer_names_to_train.assign(config.initializer_names_to_train.begin(),
                                                       config.initializer_names_to_train.end());

  std::vector<const NodeArg*> input_args;
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    input_args.emplace_back(graph.GetNodeArg(input_name));
  }

  // Remove the training initializers from the graph and move them to graph inputs.
  for (const auto& initializer_name : split_graphs_info_.initializer_names_to_train) {
    input_args.emplace_back(graph.GetNodeArg(initializer_name));
    graph.RemoveInitializedTensor(initializer_name);
  }

  graph.SetInputs(input_args);
  return Status::OK();
}

// Build the gradient graphs from original graph.
// Since the input shapes may differ, and the graph optimizers (mainly constant folding) may fold this
// shape info to constants, the optimized graph (before gradient graph building) can not be shared.
// So each time we need to start from the beginning, i.e., 1) replace input shapes, 2) apply graph optimizers,
// 3) build gradient graph, and finally 4) adjust the graph inputs and outputs.
Status ModuleGradientGraphBuilder::Build(const std::vector<std::vector<int64_t>>* input_shapes_ptr) {
  // Make a copy of the original model.
  auto model_proto = model_->ToProto();
  ORT_RETURN_IF_ERROR(Model::Load(model_proto, gradient_model_, nullptr, *logger_));

  // Replace the user input shapes if input_shapes_ptr is not null_ptr.
  if (input_shapes_ptr) {
    SetConcreteInputShapes(*input_shapes_ptr);
  }

  // Build the gradient graph.
  ORT_RETURN_IF_ERROR(BuildGradientGraph());

  // Add Yield Op.
  AddYieldOp();

  // Reorder outputs.
  ReorderOutputs();

  return Status::OK();
}

std::string ModuleGradientGraphBuilder::GetGradientModel() const {
  std::string model_str;
  if (!gradient_model_->ToProto().SerializeToString(&model_str)) {
    ORT_THROW("Fail to serialize gradient model to string.");
  }

  return model_str;
}

void ModuleGradientGraphBuilder::SetConcreteInputShapes(const std::vector<std::vector<int64_t>> input_shapes) {
  ORT_ENFORCE(input_shapes.size() == split_graphs_info_.user_input_names.size(),
              "The size of concrete input shapes and the size of user inputs does not match.");
  Graph& gradient_graph = gradient_model_->MainGraph();
  std::vector<const NodeArg*> input_args;
  size_t input_index = 0;
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    NodeArg* input_node_arg = gradient_graph.GetNodeArg(input_name);
    ONNX_NAMESPACE::TensorShapeProto new_shape;
    for (size_t i = 0; i < input_shapes[input_index].size(); i++) {
      new_shape.add_dim()->set_dim_value(input_shapes[input_index][i]);
    }

    input_node_arg->SetShape(new_shape);
    input_args.emplace_back(input_node_arg);
    input_index++;
  }

  // Move over all training initializer inputs. They already have the concrete shapes.
  const std::vector<const NodeArg*>& graph_inputs = gradient_graph.GetInputsIncludingInitializers();
  for (; input_index < graph_inputs.size(); input_index++) {
    input_args.emplace_back(graph_inputs[input_index]);
  }

  gradient_graph.SetInputs(input_args);
}

Status ModuleGradientGraphBuilder::BuildGradientGraph() {
  // Resolve original graph, register and apply transformers for pre-training.
  Graph& gradient_graph = gradient_model_->MainGraph();
  ORT_RETURN_IF_ERROR(gradient_graph.Resolve());

  const TrainingSession::TrainingConfiguration::GraphTransformerConfiguration graph_transformer_config{};
  GraphTransformerManager graph_transformation_mgr{2};
  std::unique_ptr<CPUExecutionProvider> cpu_execution_provider =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  std::unordered_set<std::string> x_node_arg_names;
  std::set_union(config_.initializer_names_to_train.begin(), config_.initializer_names_to_train.end(),
                 config_.input_names_require_grad.begin(), config_.input_names_require_grad.end(),
                 std::inserter(x_node_arg_names, x_node_arg_names.begin()));
  auto add_transformers = [&](TransformerLevel level) {
    std::unordered_map<std::string, std::string> updated_weight_names{};
    auto transformers_to_register = transformer_utils::GeneratePreTrainingTransformers(
        level, x_node_arg_names, graph_transformer_config, *cpu_execution_provider, updated_weight_names, {});
    for (auto& entry : transformers_to_register) {
      graph_transformation_mgr.Register(std::move(entry), level);
    }
  };

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    if (TransformerLevel::MaxLevel >= level) {
      add_transformers(level);
    }
  }

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR(
        graph_transformation_mgr.ApplyTransformers(gradient_graph, static_cast<TransformerLevel>(i), *logger_));
  }

  // Build gradient graph.
  GradientGraphConfiguration gradient_graph_config{};
  gradient_graph_config.use_invertible_layernorm_grad = config_.use_invertible_layernorm_grad;
  gradient_graph_config.set_gradients_as_graph_outputs = false;
  std::unordered_set<std::string> y_node_arg_names(split_graphs_info_.user_output_names.begin(),
                                                   split_graphs_info_.user_output_names.end());
  GradientGraphBuilder grad_graph_builder(&gradient_graph, y_node_arg_names, x_node_arg_names, "",
                                          gradient_graph_config, *logger_);

  ORT_RETURN_IF_ERROR(grad_graph_builder.Build());
  return Status::OK();
}

void ModuleGradientGraphBuilder::AddYieldOp() {
  Graph& gradient_graph = gradient_model_->MainGraph();
  GraphViewer gradient_graph_viewer(gradient_graph);
  const auto& gradient_node_topology_list = gradient_graph_viewer.GetNodesInTopologicalOrder();
  std::unordered_set<std::string> user_output_grad_names_set;
  for (const auto& name : split_graphs_info_.user_output_names) {
    user_output_grad_names_set.insert(name + "_grad");
  }

  // If an NodeArg is output of one of nodes, it's not the user output gradient needed by backward graph.
  std::unordered_set<std::string> non_backward_user_output_grad_names;
  for (auto node_index : gradient_node_topology_list) {
    auto& node = *gradient_graph.GetNode(node_index);
    std::cout << "Node: " << node.Name() << "\n";
    for (const auto& node_arg : node.OutputDefs()) {
      if (user_output_grad_names_set.find(node_arg->Name()) != user_output_grad_names_set.end()) {
        non_backward_user_output_grad_names.insert(node_arg->Name());
        std::cout<<"Grad NodeArg:"<<node_arg->Name() <<"\n";
      }
    }
  }

  // Yield inputs include all user outputs, those require output gradients come first, so Yield Op can use their shapes
  // to infer Op output shapes.
  std::vector<std::string> user_output_names_require_grad;
  std::vector<std::string> user_output_names_no_grad;
  split_graphs_info_.backward_output_grad_names.clear();
  for (const auto& name : split_graphs_info_.user_output_names) {
    std::string grad_name = name + "_grad";
    if (non_backward_user_output_grad_names.find(grad_name) == non_backward_user_output_grad_names.end()) {
      user_output_names_require_grad.emplace_back(name);
      split_graphs_info_.backward_output_grad_names.emplace_back(grad_name);
    } else {
      user_output_names_no_grad.emplace_back(name);
    }
  }

  // Reorder the user outputs.
  split_graphs_info_.user_output_names.clear();
  for (const auto& name : user_output_names_require_grad) {
    split_graphs_info_.user_output_names.emplace_back(name);
  }

  for (const auto& name : user_output_names_no_grad) {
    split_graphs_info_.user_output_names.emplace_back(name);
  }

  std::vector<NodeArg*> yield_input_node_args;
  std::vector<NodeArg*> yield_output_node_args;
  NodeArg* control_input = gradient_graph.GetNodeArg(split_graphs_info_.user_output_names[0]);
  NodeArg* control_output = &gradient_graph.GetOrCreateNodeArg(gradient_graph.GenerateNodeArgName(control_input->Name() + "_yield"), control_input->TypeAsProto());
  std::cout<<"Added user output yield:"<<control_input->Name() <<" : " << control_output->Name()<<"\n";
  yield_input_node_args.emplace_back(control_input);
  yield_output_node_args.emplace_back(control_output);
  for (const auto& name : split_graphs_info_.user_output_names) {
    yield_input_node_args.emplace_back(gradient_graph.GetNodeArg(name));
  }

  for (const auto& name : split_graphs_info_.backward_output_grad_names) {
    yield_output_node_args.emplace_back(gradient_graph.GetNodeArg(name));
  }

  gradient_graph.AddNode("YieldOp_fw_op", "Yield", "Yield Op", yield_input_node_args, yield_output_node_args, {}, kMSDomain);

  // Add Yield ops after each grad output is ready
  std::cout<<"Adding Grad Yield ops\n";
  // Get initializer gradients
  std::unordered_map<std::string, std::string> grad_name_map{};
  split_graphs_info_.initializer_grad_names_to_train.clear();
  for (const auto& initializer_name : split_graphs_info_.initializer_names_to_train) {
    std::string initializer_gradient_name = initializer_name + "_grad";
    split_graphs_info_.initializer_grad_names_to_train.emplace_back(initializer_gradient_name);
    grad_name_map[initializer_gradient_name] = initializer_name;
  }

  auto& grad_names = split_graphs_info_.initializer_grad_names_to_train;
  std::string last_yield_node_arg;
  std::unordered_map<std::string, NodeArg*> control_output_mapping;
  for (auto it = std::begin(gradient_node_topology_list); it != std::end(gradient_node_topology_list); ++it) {
    auto node_index = *it;
    auto& node = *gradient_graph.GetNode(node_index);
    // std::cout << "Node: " << node.Name() << "\n";
    for (const auto& node_arg : node.OutputDefs()) {
      if (std::find(grad_names.begin(), grad_names.end(), node_arg->Name()) != grad_names.end()) {
        auto& next_node = *gradient_graph.GetNode(*std::next(it,1));
        std::cout<<"Next Node:" << next_node.Name()<<"\n";
        NodeArg* control_input = next_node.MutableInputDefs()[0];
        NodeArg* control_output = &gradient_graph.GetOrCreateNodeArg(gradient_graph.GenerateNodeArgName(control_input->Name() + "_yield"), control_input->TypeAsProto());
        
        std::cout<<"Added grad output yield:"<< node_arg->Name() << " : " <<control_input->Name() <<" : " << control_output->Name()<<"\n";
        control_output_mapping[control_input->Name()] = control_output;
        std::vector<NodeArg*> yield_input_node_arg;
        std::vector<NodeArg*> yield_output_node_arg;
        yield_input_node_arg.emplace_back(control_input);
        yield_input_node_arg.emplace_back(gradient_graph.GetNodeArg(node_arg->Name()));
        yield_output_node_arg.emplace_back(control_output);
        yield_output_node_arg.emplace_back(&gradient_graph.GetOrCreateNodeArg(gradient_graph.GenerateNodeArgName(node_arg->Name() + "_yield"), node_arg->TypeAsProto()));
        Node& yield_node = gradient_graph.AddNode("YieldOp_" + node_arg->Name(), "Yield", "Yield Op", yield_input_node_arg, yield_output_node_arg, {}, kMSDomain);
        yield_node.AddAttribute("push_input", static_cast<int64_t>(1));
        split_graphs_info_.ordered_initializer_names.emplace_back(grad_name_map[node_arg->Name()]);
        std::cout<<"Yield for Grad:"<< node_arg->Name() <<"\n";
        last_yield_node_arg = yield_output_node_arg[0]->Name();
        next_node.MutableInputDefs()[0] = control_output;
      }
    }
  }

  // If an NodeArg is output of one of nodes, it's not the user output gradient needed by backward graph.
  // for (auto node_index : gradient_node_topology_list) {
  //   auto& node = *gradient_graph.GetNode(node_index);
  //   for (auto& node_arg : node.MutableOutputDefs()) {
  //     if (control_output_mapping.find(node_arg->Name()) != control_output_mapping.end()) {
  //       // replace the nodearg
  //       node_arg = control_output_mapping[node_arg->Name()];
  //       std::cout<<"Grad NodeArg:"<<node_arg->Name() <<"\n";
  //     }
  //   }
  // }

  // Add group node
  std::vector<NodeArg*> group_input_node_arg;
  group_input_node_arg.emplace_back(gradient_graph.GetNodeArg(last_yield_node_arg));
  TypeProto* group_done_argdef = onnxruntime::make_unique<TypeProto>().get();
  group_done_argdef->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  split_graphs_info_.group_output_node_arg_ = &gradient_graph.GetOrCreateNodeArg("complete", group_done_argdef);
  gradient_graph.AddNode("GroupOp", "Group", "Group Op", group_input_node_arg,
                          {split_graphs_info_.group_output_node_arg_}, {}, kMSDomain);

  //reverse the order to correctly get forward order
  std::reverse(split_graphs_info_.ordered_initializer_names.begin(), split_graphs_info_.ordered_initializer_names.end());
}

void ModuleGradientGraphBuilder::ReorderOutputs() {
  // Adjust gradient graph outputs by the following order:
  // 1. user input grads if required, with same order of user inputs,
  // 2. trainable initailizer grads, with same order of trainable initializers.
  Graph& gradient_graph = gradient_model_->MainGraph();
  const std::vector<const NodeArg*>& gradient_graph_outputs = gradient_graph.GetOutputs();
  std::unordered_map<std::string, const NodeArg*> gradient_output_arg_map;
  for (auto& node_arg : gradient_graph_outputs) {
    gradient_output_arg_map[node_arg->Name()] = node_arg;
  }

  std::unordered_set<std::string> user_input_require_grad_set(config_.input_names_require_grad.begin(),
                                                              config_.input_names_require_grad.end());

  std::vector<const NodeArg*> new_output_args;
  split_graphs_info_.user_input_grad_names.clear();
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    if (user_input_require_grad_set.find(input_name) != user_input_require_grad_set.end()) {
      std::string input_gradient_name = input_name + "_grad";
      ORT_ENFORCE(gradient_output_arg_map.find(input_gradient_name) != gradient_output_arg_map.end(),
                  "Required user input grad is not found on gradient graph.");
      split_graphs_info_.user_input_grad_names[input_name] = input_gradient_name;
      new_output_args.emplace_back(gradient_output_arg_map[input_gradient_name]);
    }
  }

  new_output_args.emplace_back(split_graphs_info_.group_output_node_arg_);


  gradient_graph.SetOutputs(new_output_args);
}

}  // namespace training
}  // namespace onnxruntime
