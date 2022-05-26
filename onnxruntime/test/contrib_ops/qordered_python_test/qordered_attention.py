import math
import numpy
import os

DATA_DIR = './qordered_attention'

def create_qordered_attention_graph():
    from onnx import helper, numpy_helper, TensorProto

    nodes = [
        #helper.make_node('QuantizeWithOrder', inputs=['weight', 'scale_weight_fp16'], outputs=['weight_s8_COL32_2R_4R4'], name='764_QuantizeWithOrder', domain='com.microsoft', order_input=1, order_output=1),
        #helper.make_node('QuantizeWithOrder', inputs=['bias', 'scale_bias_fp16'], outputs=['bias_s8_COL32'], name='769_QuantizeWithOrder', domain='com.microsoft', order_input=1, order_output=),
        helper.make_node(
            'QOrderedAttention',
            inputs=['input', 'scale_input', 'weight', 'scale_weight', 'bias', 'scale_bias', 'scale_gemm', 'mask_index', 'scale_output'],
            outputs=['output'],
            name='Attention_quantized',
            domain='com.microsoft',
            num_heads=12,
            order_bias=1,
            order_input=1,
            order_output=1,
            order_weight=0,
            unidirectional=0,
        ),
    ]

    scale = 0.007874015718698502
    weight_array = numpy.random.randint(-127, 128, [768, 2304], dtype=numpy.int8)
    bias_array = numpy.random.rand(2304)

    initializers = [
        numpy_helper.from_array(weight_array.astype('int8').reshape([768, 2304]), name='weight'),
        numpy_helper.from_array(bias_array.astype('float32').reshape([2304]), name='bias'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_input'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_weight'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_bias'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_gemm'),
        numpy_helper.from_array(numpy.array(scale, dtype='float32'), name='scale_output'),
        #numpy_helper.from_array(numpy.array(1, dtype='float16'), name='scale_weight_fp16'),
        #numpy_helper.from_array(numpy.array(1, dtype='float16'), name='scale_bias_fp16'),
    ]

    graph = helper.make_graph(nodes, "QOrderedAttention_Graph", [
        helper.make_tensor_value_info('input', TensorProto.INT8, [1, 32, 768]),
        helper.make_tensor_value_info('mask_index', TensorProto.INT32, [1, 32]),
    ], [
        helper.make_tensor_value_info('output', TensorProto.INT8, [1, 32, 768]),
    ], initializers)

    model = helper.make_model(graph=graph)
    return model.SerializeToString()


onnx_model_str = create_qordered_attention_graph()

from onnxruntime import SessionOptions, InferenceSession
sess_options = SessionOptions()
ort_session = InferenceSession(onnx_model_str, sess_options, providers=['CUDAExecutionProvider'])

ort_inputs = {
    'input' : numpy.random.randint(-127, 128, [1, 32, 768], dtype=numpy.int8),
    'mask_index' : numpy.random.randint(1, 2, [1, 32], dtype=numpy.int32)
}

ort_output = ort_session.run(None, ort_inputs)
print(ort_output)
