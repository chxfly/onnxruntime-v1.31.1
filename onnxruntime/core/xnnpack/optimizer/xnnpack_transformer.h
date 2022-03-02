// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class XnnPackTransformer

Transformer a normal graph to XnnPack nodes
*/
class XnnPackTransformer : public GraphTransformer {         
 public:
  explicit XnnPackTransformer(AllocatorPtr cpu_allocator) noexcept 
    : GraphTransformer("XnnPackTransformer"), cpu_allocator_(std::move(cpu_allocator)){};  

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  AllocatorPtr cpu_allocator_;
};

}  // namespace onnxruntime
