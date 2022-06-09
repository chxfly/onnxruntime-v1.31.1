import math
import numpy
import os
from onnxruntime import SessionOptions, InferenceSession

numpy.random.seed(0)
DATA_DIR = './qordered_attention'

hidden_size = 32
seqlen = 4

float_range = 0.5
float_range_2 = 0.2
float_range_3 = 0.2

# Generate well 'QATed' data
#weight_array = numpy.random.uniform(-1 * float_range_2, float_range_2, [hidden_size, 3 * hidden_size])
weight_array = (numpy.random.randint(-127, 128, size = (hidden_size, 3 * hidden_size)) / 127.5 * float_range_2).astype('float32')
bias_array = numpy.random.uniform(-1 * float_range_3, float_range_3, [3 * hidden_size])

#input_data = numpy.random.uniform(-1 * float_range, float_range, [1, seqlen, hidden_size]).astype('float32')
input_data = (numpy.random.randint(-127, 128, size = (1, seqlen, hidden_size)) / 127.5 * float_range).astype('float32')
mask_index_data = numpy.random.randint(1, 2, [1, seqlen], dtype=numpy.int32)

def create_attention_graph():
    from onnx import helper, numpy_helper, TensorProto

    nodes = [
        helper.make_node(
            'Attention',
            inputs=['input', 'weight', 'bias', 'mask_index'],
            outputs=['output'],
            name='Attention_normal',
            domain='com.microsoft',
            num_heads=2,
            unidirectional=0,
        ),
    ]

    initializers = [
        numpy_helper.from_array(weight_array.astype('float32').reshape([hidden_size, 3 * hidden_size]), name='weight'),
        numpy_helper.from_array(bias_array.astype('float32').reshape([3 * hidden_size]), name='bias')
    ]

    graph = helper.make_graph(nodes, "Attention_Graph", [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, seqlen, hidden_size]),
        helper.make_tensor_value_info('mask_index', TensorProto.INT32, [1, seqlen]),
    ], [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, seqlen, hidden_size]),
    ], initializers)

    model = helper.make_model(graph=graph)
    return model.SerializeToString()

def create_qordered_attention_graph(scale_input, scale_weight, scale_gemm, scale_output):
    from onnx import helper, numpy_helper, TensorProto

    nodes = [
        helper.make_node('QuantizeWithOrder', inputs=['input', 'scale_input'], outputs=['input_s8'], name='QuantizeWithOrder_0', domain='com.microsoft', order_input=1, order_output=1),
        helper.make_node('QuantizeWithOrder', inputs=['weight', 'scale_weight'], outputs=['weight_col'], name='QuantizeWithOrder_1', domain='com.microsoft', order_input=1, order_output=0),
        helper.make_node(
            'QOrderedAttention',
            inputs=['input_s8', 'scale_input', 'weight_col', 'scale_weight', 'bias', 'scale_bias', 'scale_gemm', 'mask_index', 'scale_output'],
            outputs=['output_s8'],
            name='Attention_quantized',
            domain='com.microsoft',
            num_heads=2,
            order_bias=1,
            order_input=1,
            order_output=1,
            order_weight=0,
            unidirectional=0,
        ),
        helper.make_node('DequantizeWithOrder', inputs=['output_s8', 'scale_output'], outputs=['output'], name='DeQuantizeWithOrder_0', domain='com.microsoft', order_input=1, order_output=1),
    ]

    initializers = [
        numpy_helper.from_array(weight_array.astype('float32').reshape([hidden_size, 3 * hidden_size]), name='weight'),
        numpy_helper.from_array(bias_array.astype('float32').reshape([3 * hidden_size])/scale_gemm, name='bias'),
        numpy_helper.from_array(numpy.array(scale_input, dtype='float32'), name='scale_input'),
        numpy_helper.from_array(numpy.array(scale_weight, dtype='float32'), name='scale_weight'),
        numpy_helper.from_array(numpy.array(1, dtype='float32'), name='scale_bias'),
        numpy_helper.from_array(numpy.array(scale_gemm, dtype='float32'), name='scale_gemm'),
        numpy_helper.from_array(numpy.array(scale_output, dtype='float32'), name='scale_output'),
    ]

    graph = helper.make_graph(nodes, "QOrderedAttention_Graph", [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, seqlen, hidden_size]),
        helper.make_tensor_value_info('mask_index', TensorProto.INT32, [1, seqlen]),
    ], [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, seqlen, hidden_size]),
    ], initializers)

    model = helper.make_model(graph=graph)
    return model.SerializeToString()

# Run fp32 Attn
attn = create_attention_graph()
sess_options = SessionOptions()
ort_session_attn = InferenceSession(attn, sess_options, providers=['CUDAExecutionProvider'])
ort_inputs = {
    'input' : input_data,
    'mask_index' : mask_index_data
}
ort_output = ort_session_attn.run(None, ort_inputs)

# Calculate scale
scale = 127.5
scale_input = (numpy.abs(input_data).max()/scale).astype('float32')
scale_weight = (numpy.abs(weight_array).max()/scale).astype('float32')
scale_gemm = (numpy.abs(numpy.matmul(input_data, weight_array)).max()/scale).astype('float32')
scale_output = (numpy.abs(ort_output[0]).max()/scale).astype('float32')
print('scale_input', scale_input)
print('scale_weight', scale_weight)
print('scale_gemm', scale_gemm)
print('scale_output', scale_output)

qattn = create_qordered_attention_graph(scale_input, scale_weight, scale_gemm, scale_output)
ort_session_qattn = InferenceSession(qattn, sess_options, providers=['CUDAExecutionProvider'])
ort_output_q = ort_session_qattn.run(None, ort_inputs)
print('fp32 attn output:')
print(ort_output[0][0][0])
print('qordered_attn output')
print(ort_output_q[0][0][0])

tol_l = 1e-5
tol_r = 1e2
while (tol_r - tol_l > 1e-5):
    tol = tol_l + (tol_r - tol_l)/2
    if_close = numpy.allclose(ort_output_q[0], ort_output[0], rtol = tol, atol = tol)
    if if_close is True:
        tol_r = tol
    else:
        tol_l = tol

print("atol/rtol threshold:", tol_r)
#print(ort_output[0] - ort_output_q[0])
