import math
import numpy
import os

numpy.random.seed(0)
DATA_DIR = './qordered_attention'

def create_qordered_attention_graph():
    from onnx import helper, numpy_helper, TensorProto

    nodes = [
        helper.make_node('QuantizeWithOrder', inputs=['input', 'scale_input'], outputs=['input_s8'], name='QuantizeWithOrder_0', domain='com.microsoft', order_input=1, order_output=1),
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
        numpy_helper.from_array(numpy.load(os.path.join(DATA_DIR, 'const59_1572_s8_COL.npy')).astype('int8').reshape([768, 2304]), name='weight_col'),
        numpy_helper.from_array(numpy.load(os.path.join(DATA_DIR, 'const60_sentence_wise_att.layer.0.attention.self.merged_qkv_bias_fp32_0.0658777505159378.npy')).astype('float32').reshape([2304]), name='bias'),
        numpy_helper.from_array(numpy.array(0.0013535089092329144, dtype='float32'), name='scale_weight'),
        numpy_helper.from_array(numpy.array(0.0013535089092329144, dtype='float32'), name='scale_bias'),
        numpy_helper.from_array(numpy.array(0.0658777505159378, dtype='float32'), name='scale_gemm'),
        numpy_helper.from_array(numpy.array(0.020903365686535835, dtype='float32'), name='scale_output'),
    ]

    graph = helper.make_graph(nodes, "QOrderedAttention_Graph", [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 32, 768]),
        helper.make_tensor_value_info('scale_input', TensorProto.FLOAT, []),
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
        numpy_helper.from_array(numpy.load(os.path.join(DATA_DIR, 'const66_1572.npy')).astype('float32').reshape([768, 2304]), name='weight'),
        numpy_helper.from_array(numpy.load(os.path.join(DATA_DIR, 'const3_sentence_wise_att.layer.0.attention.self.merged_qkv_bias.npy')).astype('float32').reshape([2304]), name='bias')
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

input_data = numpy.random.standard_normal(size = (1, 32, 768)).astype('float32')
mask_index_data = numpy.random.randint(1, 2, [1, 32], dtype=numpy.int32)
input_scale_data = numpy.array(numpy.abs(input_data).max()/127).astype('float32')

ort_inputs = {
    'input' : input_data,
    'mask_index' : mask_index_data
}

ort_output = ort_session_attn.run(None, ort_inputs)

ort_inputs['scale_input'] = input_scale_data

ort_output_q = ort_session_qattn.run(None, ort_inputs)

tol_l = 1e-5
tol_r = 1e-1
delta = 1e-5
while (tol_r - tol_l > delta):
    tol = tol_l + (tol_r - tol_l)/2
    if_close = numpy.allclose(ort_output_q[0], ort_output[0], rtol = tol, atol = tol)
    if if_close is True:
        tol_r = tol
    else:
        tol_l = tol

print("atol/rtol range:", tol_r, "±", delta)
