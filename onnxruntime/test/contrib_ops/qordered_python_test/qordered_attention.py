import math
import numpy
import os

numpy.random.seed(0)
scale = 0.007874015718698502
weight_array = numpy.random.standard_normal(size = (768, 2304))
bias_array = numpy.random.standard_normal(size = (2304))

def create_qordered_attention_graph():
    from onnx import helper, numpy_helper, TensorProto

    nodes = [
        helper.make_node('QuantizeWithOrder', inputs=['input', 'scale_input'], outputs=['input_s8'], name='QuantizeWithOrder_0', domain='com.microsoft', order_input=1, order_output=1),
        helper.make_node('QuantizeWithOrder', inputs=['weight_fp32', 'scale_weight'], outputs=['weight_col'], name='QuantizeWithOrder_1', domain='com.microsoft', order_input=1, order_output=0),
        helper.make_node(
            'QOrderedAttention',
            inputs=['input_s8', 'scale_input', 'weight_col', 'scale_weight', 'bias', 'scale_bias', 'scale_gemm', 'mask_index', 'scale_output'],
            outputs=['output_s8'],
            name='Attention_quantized',
            domain='com.microsoft',
            num_heads=12,
            order_bias=1,
            order_input=1,
            order_output=1,
            order_weight=0,
            unidirectional=0,
        ),
        helper.make_node('DequantizeWithOrder', inputs=['output_s8', 'scale_output'], outputs=['output'], name='DeQuantizeWithOrder_0', domain='com.microsoft', order_input=1, order_output=1),
    ]

    initializers = [
        numpy_helper.from_array(weight_array.astype('float32').reshape([768, 2304]), name='weight_fp32'),
        numpy_helper.from_array(bias_array.astype('float32').reshape([2304]), name='bias'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_input'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_weight'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_bias'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_gemm'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_output'),
    ]

    graph = helper.make_graph(nodes, "QOrderedAttention_Graph", [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 32, 768]),
        helper.make_tensor_value_info('mask_index', TensorProto.INT32, [1, 32]),
    ], [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 768]),
    ], initializers)

    model = helper.make_model(graph=graph)
    return model.SerializeToString()

def create_attention_graph():
    from onnx import helper, numpy_helper, TensorProto

    nodes = [
        helper.make_node(
            'Attention',
            inputs=['input', 'weight', 'bias', 'mask_index'],
            outputs=['output'],
            name='Attention_normal',
            domain='com.microsoft',
            num_heads=12,
            unidirectional=0,
        ),
    ]

    initializers = [
        numpy_helper.from_array(weight_array.astype('float32').reshape([768, 2304]), name='weight'),
        numpy_helper.from_array(bias_array.astype('float32').reshape([2304]), name='bias'),
    ]

    graph = helper.make_graph(nodes, "Attention_Graph", [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 32, 768]),
        helper.make_tensor_value_info('mask_index', TensorProto.INT32, [1, 32]),
    ], [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 768]),
    ], initializers)

    model = helper.make_model(graph=graph)
    return model.SerializeToString()

qattn = create_qordered_attention_graph()
attn = create_attention_graph()

from onnxruntime import SessionOptions, InferenceSession
sess_options = SessionOptions()
ort_session_qattn = InferenceSession(qattn, sess_options, providers=['CUDAExecutionProvider'])
ort_session_attn = InferenceSession(attn, sess_options, providers=['CUDAExecutionProvider'])

ort_inputs = {
    'input' : numpy.random.standard_normal(size = (1, 32, 768)).astype('float32'),
    'mask_index' : numpy.random.randint(1, 2, [1, 32], dtype=numpy.int32)
}

ort_output_q = ort_session_qattn.run(None, ort_inputs)
ort_output = ort_session_attn.run(None, ort_inputs)
print(ort_output_q[0] - ort_output[0])
