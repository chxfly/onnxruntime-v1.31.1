// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/gsl.h"
#include "core/framework/execution_provider.h"  // for IExecutionProvider::IKernelLookup
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"

namespace onnxruntime {

/**
 * Utility class for performing kernel lookup.
 * Primary usage pattern is to be created during graph partitioning and passed to IExecutionProvider::GetCapability().
 */
class KernelLookup final : public IExecutionProvider::IKernelLookup {
 public:
  KernelLookup(ProviderType provider_type,
               gsl::span<const gsl::not_null<const KernelRegistry*>> kernel_registries,
               const IKernelTypeStrResolver& kernel_type_str_resolver)
      : provider_type_{provider_type},
        kernel_registries_{kernel_registries},
        kernel_type_str_resolver_{kernel_type_str_resolver} {
    ORT_ENFORCE(!provider_type_.empty(), "provider_type must be specified.");
  }

  const KernelCreateInfo* LookUpKernel(const Node& node) const override {
    const KernelCreateInfo* kernel_create_info{};
#ifndef NDEBUG
    printf(" LookUpKernel() calling on node: [%s][%s][%s], provider type=%s\n", node.Domain().c_str(), node.OpType().c_str(), node.Name().c_str(), provider_type_.c_str());
#endif
    for (const auto& registry : kernel_registries_) {
      const auto lookup_status = registry->TryFindKernel(node, provider_type_, kernel_type_str_resolver_,
                                                         &kernel_create_info);
      if (lookup_status.IsOK() && kernel_create_info != nullptr) {
#ifndef NDEBUG
    printf(" - found\n");
#endif
        return kernel_create_info;
      }
    }

#ifndef NDEBUG
    printf(" - not found\n");
#endif
    return nullptr;
  }

 private:
  ProviderType provider_type_;
  const gsl::span<const gsl::not_null<const KernelRegistry*>> kernel_registries_;
  const IKernelTypeStrResolver& kernel_type_str_resolver_;
};

}  // namespace onnxruntime
