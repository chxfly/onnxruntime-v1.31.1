# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import Tuple, Union

import numpy as np
from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)

class FusionAttentionUnet(Fusion):
    """
    Fuse Attention subgraph of UNet into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int
    ):
        super().__init__(model, "Attention", ["InstanceNormalization"])
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto, add_q: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q
            add_q (NodeProto): add node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape = self.model.get_initializer(reshape_q.input[1])
        if q_shape is None:
            logger.debug(f"{reshape_q.input[1]} is not initializer.")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        q_shape_value = NumpyHelper.to_array(q_shape)
        if len(q_shape_value) != 4 or q_shape_value[2] <= 0:
            logger.debug(f"q_shape_value={q_shape_value}. Expected value are like [0, 0, num_heads, -1].")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = q_shape_value[2]


        bias = self.model.get_initializer(add_q.input[0]) or self.model.get_initializer(q_add.input[1])
        if bias is None:
            logger.debug(f"{add_q.input[0]} is not initializer.")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        hidden_size = NumpyHelper.to_array(bias).shape[0]

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
                self.num_heads_warning = False  # Do not show the warning more than once

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def create_attention_node(
        self,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        q_add: NodeProto,
        k_add: NodeProto,
        v_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for  K
            v_matmul (NodeProto): MatMul node in fully connection for  V
            q_add (NodeProto): Add bias node in fully connection for Q
            k_add (NodeProto): Add bias node in fully connection for K
            v_add (NodeProto): Add bias node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        q_bias = self.model.get_initializer(q_add.input[1]) or self.model.get_initializer(q_add.input[0])
        k_bias = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
        v_bias = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])

        if q_weight is None:
            print(
                f"{q_matmul.input[1]} is not an initializer. "
                "Please set do_constant_folding=True in torch.onnx.export to unblock attention fusion"
            )
            return None
        if not (k_weight and v_weight and q_bias and k_bias):
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size ({hidden_size}) is not same as weight matrix dimension of q,k,v ({qw_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        is_qkv_diff_dims = False
        if qw.shape != vw.shape:
            is_qkv_diff_dims = True

        # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
        # For 2d weights, the shapes would be [in_size, out_size].
        # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
        qw_out_size = np.prod(qw.shape[1:])
        kw_out_size = np.prod(qw.shape[1:])
        vw_out_size = np.prod(vw.shape[1:])

        qkv_weight_dim = 0
        if is_qkv_diff_dims:
            qkv_weight = np.concatenate((qw, kw, vw), axis=1)
            qkv_weight_dim = qw_out_size + kw_out_size + vw_out_size
        else:
            qkv_weight = np.stack((qw, kw, vw), axis=1)
            qkv_weight_dim = 3 * qw_out_size

        qb = NumpyHelper.to_array(q_bias)
        kb = NumpyHelper.to_array(k_bias)
        vb = NumpyHelper.to_array(v_bias)

        q_bias_shape = np.prod(qb.shape)
        k_bias_shape = np.prod(kb.shape)
        v_bias_shape = np.prod(vb.shape)

        assert q_bias_shape == k_bias_shape == qw_out_size
        assert v_bias_shape == vw_out_size

        qkv_bias_dim = 0
        if is_qkv_diff_dims:
            qkv_bias = np.concatenate((qb, kb, vb), axis=0)
            qkv_bias_dim = q_bias_shape + k_bias_shape + v_bias_shape
        else:
            qkv_bias = np.stack((qb, kb, vb), axis=0)
            qkv_bias_dim = 3 * q_bias_shape

        attention_node_name = self.model.create_node_name("Attention")

        weight = helper.make_tensor(
            name=attention_node_name + "_qkv_weight",
            data_type=TensorProto.FLOAT,
            dims=[qw_in_size, qkv_weight_dim],
            vals=qkv_weight.flatten().tolist(),
        )

        # Sometimes weights and bias are stored in fp16
        if q_weight.data_type == 10:
            weight.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(weight).astype(np.float16), weight.name))
        self.model.add_initializer(weight, self.this_graph_name)

        bias = helper.make_tensor(
            name=attention_node_name + "_qkv_bias",
            data_type=TensorProto.FLOAT,
            dims=[qkv_bias_dim],
            vals=qkv_bias.flatten().tolist(),
        )
        if q_bias.data_type == 10:
            bias.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(bias).astype(np.float16), bias.name))
        self.model.add_initializer(bias, self.this_graph_name)

        attention_inputs = [
            input,
            attention_node_name + "_qkv_weight",
            attention_node_name + "_qkv_bias",
        ]
        attention_inputs.append("")

        attention_node = helper.make_node(
            "Attention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        if is_qkv_diff_dims:
            attention_node.attribute.extend(
                [helper.make_attribute("qkv_hidden_sizes", [qw_out_size, kw_out_size, vw_out_size])]
            )

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        assert normalize_node.op_type == "InstanceNormalization"

        reshape_before_instance_norm = self.model.match_parent(normalize_node, "Reshape", 0)
        if reshape_before_instance_norm is None:
            return
            
        root_input = reshape_before_instance_norm.input[0]

        children_nodes = input_name_to_nodes[root_input]
        skip_add = None
        for node in children_nodes:
            if node.op_type == 'Add':
                skip_add = node
                break
        if skip_add is None:
            return

        another_input = 1 if skip_add.input[0] == root_input else 0
        qkv_nodes = self.model.match_parent_path(
            skip_add,
            ["Reshape",     "Transpose", "Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [another_input, 0,           0,     None,     None,      0,           0],
        )

        if qkv_nodes is None:
            return

        (_, _, _, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "Add", "MatMul", "Transpose", "Reshape", "Add", "Mul", "Reshape", "InstanceNormalization"],
            [1,           0,         0,      None,     0,           0,         0,     None,  None,      0],
        )
        if v_nodes is None:
            return
        if v_nodes[-1] != normalize_node:
            return


        (_, _, add_v, matmul_v, transpose, _, _, _, _, _) = v_nodes

        qk_nodes = self.model.match_parent_path(
            matmul_qkv, 
            ["Softmax", "MatMul"],
            [0,         0]
            )
        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        (softmax_qk, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(matmul_qk,
                                               ["Mul", "Transpose", "Reshape", "Add", "MatMul", "Transpose"],
                                               [0,      0,           0,         0,     None,    None])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (_mul_q, _transpose_q, reshape_q, add_q, matmul_q, _transpose) = q_nodes
        if _transpose != transpose:
            return

        k_nodes = self.model.match_parent_path(matmul_qk,
                                               ["Mul", "Transpose", "Reshape", "Add", "MatMul", "Transpose"],
                                               [1,      0,           0,         0,     None,    None])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return

        (_mul_k, _transpose_k, reshape_k, add_k, matmul_k, _transpose) = k_nodes
        if _transpose != transpose:
            return

        attention_last_node = reshape_qkv

        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q, add_q)
        
        # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
        # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
        new_node = self.create_attention_node(
            matmul_q,
            matmul_k,
            matmul_v,
            add_q,
            add_k,
            add_v,
            q_num_heads,
            q_hidden_size,
            input = transpose.output[0],
            output = attention_last_node.output[0],
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
        self.nodes_to_remove.extend(qk_nodes)
        self.nodes_to_remove.extend(q_nodes[:5])
        self.nodes_to_remove.extend(k_nodes[:5])
        self.nodes_to_remove.extend(v_nodes[:4])

        # Remove Div node with constant value 1.0.
        div = self.model.find_first_child_by_type(skip_add, 'Div', input_name_to_nodes=input_name_to_nodes, recursive=False)
        if div and self.model.has_constant_input(div, 1.0, delta=1e-8):
            self.model.replace_output_of_all_nodes(skip_add.output[0], div.output[0])
            self.model.remove_node(div)

        # Use prune graph to remove mask nodes since they are shared by all attention nodes.
        # self.nodes_to_remove.extend(mask_nodes)
        self.prune_graph = True
