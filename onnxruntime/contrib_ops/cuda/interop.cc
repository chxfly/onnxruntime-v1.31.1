// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "interop.h"
#include "core/framework/op_kernel_context_internal.h"
#include <pybind11/pybind11.h>
#include <Python.h>
#include <iostream>
#include "core/util/dlpack_convertor.h"
namespace py = pybind11;
namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(Interop, kMSDomain, 1, kCudaExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), Interop);

void dlpack_capsule_destructor2(PyObject* data) {
  DLManagedTensor* dlmanged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  if (dlmanged_tensor) {
    // the dlmanged_tensor has not been consumed, call deleter ourselves.
    dlmanged_tensor->deleter(const_cast<DLManagedTensor*>(dlmanged_tensor));
  } else {
    // the dlmanged_tensor has been consumed,
    // PyCapsule_GetPointer has set an error indicator.
    PyErr_Clear();
  }
}

Status Interop::ComputeInternal(OpKernelContext* ctx) const {
  const OpKernelInfo& info = OpKernel::Info();
  int64_t external_fn_id;
  ORT_ENFORCE(info.GetAttr<int64_t>("external_fn", &external_fn_id).IsOK());
  int64_t is_backward;
  ORT_ENFORCE(info.GetAttr<int64_t>("is_backward", &is_backward).IsOK());

  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  std::cout << "fail 1" << std::endl;
  // Py_SetPythonHome(L"/opt/conda/lib/python3.7");
  std::cout << "fail 1.5" << std::endl;
  Py_Initialize();
  auto path_list = PySys_GetObject("path");  //do not release it
  std::cout << "fail 2" << std::endl;
  if (nullptr == path_list || !PyList_Check(path_list) ||
      PyList_Append(path_list, PyUnicode_FromString("/bert_ort/pengwa/dev4/onnxruntime/exp")) != 0) {
    std::cout << "fail 3" << std::endl;
    return Status::OK();
  }
  std::cout << "fail 4" << std::endl;
  // Pass data ORT->Python
  py::tuple py_inputs(ctx->InputCount());
  for (int i = 0; i < ctx->InputCount(); ++i) {
    auto ort_value = *ctx_internal->GetInputMLValue(i);
    DLManagedTensor* dlmanaged_tensor = onnxruntime::python::ort_value_to_dlpack(ort_value);
    py_inputs[i] = py::reinterpret_steal<py::object>(
        PyCapsule_New(dlmanaged_tensor, "dltensor", dlpack_capsule_destructor2));
  }

  std::cout << "#################### I AM IN Interop::ComputeInternal #######################" << std::endl;
  py::object onnx = py::module::import("mymodule");
  std::cout << "#################### Interop::ComputeInternal::Forward ##################" << std::endl;
  py::object fprward_address = onnx.attr("simple_model_forward_func_address")();
  py::object variable = fprward_address(py_inputs);
  py::object context_obj = variable.attr("grad_fn");
  std::cout << "#################### Interop::ComputeInternal::Backward ###############" << std::endl;
  py::object backward_address = onnx.attr("simple_model_backward_func_address")();
  py::object grad_inputs = backward_address(context_obj, py_inputs);
  std::cout << "#################### END Interop::ComputeInternal #######################" << std::endl;

  // Pass data Python->ORT

  // Pass data ORT->Python
  //   auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  //   for (int i_in = 0; i_in < ctx->InputCount(); ++i_in) {
  //     OrtMessageQueue::GetInstance().Push(*ctx_internal->GetInputMLValue(i_in));
  //   }

  //   // Signal that a portion of the graph is complete
  //   const int64_t main_thread_event_id = 0;
  //   OrtEventPool::GetInstance().SignalEvent(main_thread_event_id,
  //                                           is_backward ? (OrtEventPool::TOKEN_HOLE_BACKWARD + external_fn_id)
  //                                                       : (OrtEventPool::TOKEN_HOLE_FORWARD + external_fn_id));

  //   // Wait for resumption from Python
  //   const int64_t background_thread_event_id = 1;
  //   onnxruntime::contrib::OrtEventPool::GetInstance().ResetAndWaitEvent(background_thread_event_id);

  //   // Pass data Python->ORT
  //   for (int i_out = 0; i_out < ctx->OutputCount(); ++i_out) {
  //     ctx_internal->SetOutputMLValue(i_out, onnxruntime::contrib::OrtMessageQueue::GetInstance().Pop());
  //   }

  return Status::OK();
}

}  // namespace cuda
}
}  // namespace onnxruntime
