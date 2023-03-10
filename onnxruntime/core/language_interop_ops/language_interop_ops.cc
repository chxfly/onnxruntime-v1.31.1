// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "language_interop_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "pyop/pyop.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace onnxruntime {

void InterOpDomainDeleter(OrtCustomOpDomain* domain) {
  if (nullptr != domain) {
    for (auto op : domain->custom_ops_) {
      delete op;
    }
    delete domain;
  }
}

void LoadInterOp(const std::basic_string<ORTCHAR_T>& model_uri, InterOpDomains& domains, const InterOpLogFunc& log_func) {
  int fd;
  ORT_ENFORCE(Env::Default().FileOpenRd(model_uri, fd).IsOK(), "Failed to read model file");
  google::protobuf::io::FileInputStream f(fd);
  f.SetCloseOnDelete(true);
  ONNX_NAMESPACE::ModelProto model_proto;
  ORT_ENFORCE(model_proto.ParseFromZeroCopyStream(&f), "Failed to parse model proto");
  LoadInterOp(model_proto, domains, log_func);
}

void LoadInterOp(const ONNX_NAMESPACE::ModelProto& model_proto, InterOpDomains& domains, const InterOpLogFunc& log_func) {
  LoadInterOp(model_proto.graph(), domains, log_func);
}

void LoadInterOp(const ONNX_NAMESPACE::GraphProto& graph_proto, InterOpDomains& domains, const InterOpLogFunc& log_func) {
  for (int i = 0; i < graph_proto.node_size(); ++i) {
    const auto& node_proto = graph_proto.node(i);
    if (node_proto.op_type() == "PyOp") {
      OrtCustomOpDomain* pyop_domain = nullptr;
      const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
      Ort::ThrowOnError(ort, ort->CreateCustomOpDomain(node_proto.domain().c_str(), &pyop_domain));
      Ort::ThrowOnError(ort, ort->CustomOpDomain_Add(pyop_domain, LoadPyOp(node_proto, log_func)));
      auto ort_domain = std::unique_ptr<OrtCustomOpDomain, decltype(&InterOpDomainDeleter)>(pyop_domain, &InterOpDomainDeleter);
      domains.push_back(std::move(ort_domain));
    } else {
      for (int j = 0; j < node_proto.attribute_size(); ++j) {
        const auto& attr = node_proto.attribute(j);
        if (utils::HasGraph(attr)) {
          LoadInterOp(attr.g(), domains, log_func);  //load pyop in subgraph
        }
      }  //for
    }    //else
  }      //for
}
}  // namespace onnxruntime
