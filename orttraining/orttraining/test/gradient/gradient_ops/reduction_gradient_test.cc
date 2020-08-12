// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef NDEBUG  // disable for debug builds because some of these tests are slow

#include <algorithm>
#include <bitset>
#include <cmath>
#include <random>
#include <thread>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/test_random_seed.h"
#include "orttraining/test/gradient/gradient_checker.h"
#include "orttraining/test/gradient/gradient_op_test_utils.h"

#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace test {

using ONNX_NAMESPACE::MakeAttribute;
using training::OpDef;
static void RunReductionTests(const OpDef& op_def) {
  TestDataVector test_data(
      // Input X
      {
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
      },
      // Input Y
      {
          {{1, 1, 1}},
          {{}},
          {{1, 3, 1}},
          {{2}},
          {{4, 1, 2}},
          {{4, 3}},
          {{4, 1, 2}},
          {{4}}},
      // Attributes
      {
          // default
          {},
          // axes = [0, 1, 2], keepdims = 0
          {MakeAttribute("axes", std::vector<int64_t>{0, 1, 2}),
           MakeAttribute("keepdims", int64_t(0))},
          // axes = [0, 2], keepdims = 1
          {MakeAttribute("axes", std::vector<int64_t>{0, 2})},
          // axes = [0, 1], keepdims = 0
          {MakeAttribute("axes", std::vector<int64_t>{0, 1}),
           MakeAttribute("keepdims", int64_t(0))},
          // axes = [1], keepdims = 1
          {MakeAttribute("axes", std::vector<int64_t>{1}),
           MakeAttribute("keepdims", int64_t(1))},
          // axes = [2], keepdims = 0
          {MakeAttribute("axes", std::vector<int64_t>{2}),
           MakeAttribute("keepdims", int64_t(0))},
          // axes = [-2], keepdims = 1
          {MakeAttribute("axes", std::vector<int64_t>{-2}),
           MakeAttribute("keepdims", int64_t(1))},
          // axes = [-2, -1], keepdims = 0
          {MakeAttribute("axes", std::vector<int64_t>{-2, -1}),
           MakeAttribute("keepdims", int64_t(0))}});

  GradientChecker<float, float, float> gradient_checker;

  float max_error;

  for (size_t i = 0; i < std::get<0>(test_data).size(); i++) {
    max_error = 0;
    gradient_checker.ComputeGradientError(op_def, std::get<0>(test_data)[i],
                                          std::get<1>(test_data)[i], &max_error,
                                          std::get<2>(test_data)[i]);
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ReduceMeanGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def{"ReduceMean", kOnnxDomain, 11};

  RunReductionTests(op_def);
}

TEST(GradientCheckerTest, ReduceSumGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def{"ReduceSum", kOnnxDomain, 11};

  RunReductionTests(op_def);
}

TEST(GradientCheckerTest, ReduceLogSumExpGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def{"ReduceLogSumExp", kOnnxDomain, 11};

  RunReductionTests(op_def);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // NDEBUG
