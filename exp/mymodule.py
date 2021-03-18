import os
from torch.utils.dlpack import from_dlpack
import onnxruntime
import numpy as np
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        print("saved a tensor in ctx in forward pass, the tensor is ", input)
        output = input * 2 #input.clamp(min=0)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        print("get a tensor in ctx in backward pass, the tensor is ", input)
        return grad_input


class CustomFnWrapperModule(torch.nn.Module):
    def __init__(self, A,B,C):
        super(CustomFnWrapperModule, self).__init__()
        self.a,self.b,self.c = A,B,C
        self.x_t = None
        self.rets = None

    def compute(self, x):
        try:
            self.x_t = from_dlpack(x)
            # what if custom function modify x, and in ORT is using an unexpected value at the same time.
            self.x_t.requires_grad = True
            print("Current process id is ", os.getpid())
            self.rets = self.forward(self.x_t)

            # we need hold the self.rets because we passed the underlying data storage pointer to ORT PyOP
            # PyOP will reuse that buffer.
            outputs_ortvalue = []
            for o in self.rets:
                print("device: ", o.device)
                #onnxruntime.OrtValue.ortvalue_from_data_ptr(list(o.size()), dtype_torch_to_numpy(o.dtype), o.device.type, _get_device_index(o.device), o.data_ptr())]
                v = o.data_ptr()
                print("address: ", v)
                outputs_ortvalue.append(v)
            return tuple(outputs_ortvalue)
        except Exception as e:
            print(e)
            return []

    def forward(self, x):
        return MyReLU.apply(x)



# class CustomFnModule(torch.nn.Module):
#     def __init__(self, custom_fn):
#         super(CustomFnModule, self).__init__()
#         self.custom_fn = custom_fn
        
#     def forward(self):
#         cuda0 = torch.device('cuda:0')
#         input=torch.ones([2, 2], dtype=torch.float64, device=cuda0)
#         print("CustomFnModule called in forward pass")
#         print("Current process id is ", os.getpid())
#         print("Context info after running pythong script: ", context_info_2)
#         return self.custom_fn.apply(input)

# m=CustomFnModule(MyReLU)
# # todo, need give the input as param
# def simple_model_forward_func():
#     return m.forward()


# def forward_wrapper(x):
#     x.requires_grad = True
#     return MyReLU.apply(x)

# def simple_model_forward_func_address():
#     print("get adress of forward_wrapper address")
#     return forward_wrapper

# def backward_wrapper(ctx, grad_outputs):
#     return MyReLU.backward(ctx, grad_outputs)

# def simple_model_backward_func_address():
#     rint("get adress of backward_wrapper address")
#     return backward_wrapper


# # need give forward result, grad_out, grad_input as params.
# def simple_model_backward_func(forward_output):
#     cuda0 = torch.device('cuda:0')
#     grad_output=torch.ones([2, 2], dtype=torch.float64, device=cuda0)
#     forward_output.backward(grad_output)
#     output=torch.tensor(np.array([[22, 33], [55, 66]]), dtype=torch.float64, device=cuda0)
#     return output

