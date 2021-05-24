// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class YieldIORewriter

Rewrite rule that resets input and output names of the yield node.

It is attempted to be triggered only on nodes with op type "Yield".
*/
class YieldIORewriter : public RewriteRule {
 public:
  YieldIORewriter(const std::vector<std::string> input_names,
                  const std::vector<std::string> output_names) noexcept : RewriteRule("YieldIORewriter"),
                                                                          input_names_(input_names),
                                                                          output_names_(output_names) {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"YieldOp"};
  }

 private:
  std::vector<std::string> input_names_, output_names_;
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};  // namespace onnxruntime

}  // namespace onnxruntime
