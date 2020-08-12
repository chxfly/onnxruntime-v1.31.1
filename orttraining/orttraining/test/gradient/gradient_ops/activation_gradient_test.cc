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

TEST(GradientCheckerTest, ReluGrad) {
  UnaryOpGradientTest("Relu");
}

void GradientCheckerSoftmaxGradHelper(bool is_log_softmax) {
  TensorShape shape({3, 4, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;

  const std::string op = is_log_softmax ? "LogSoftmax" : "Softmax";
  OpDef op_def{op};

  // default_axis
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // axis=0
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error, {MakeAttribute("axis", int64_t(0))});
    EXPECT_IS_TINY(max_error);
  }

  // axis=2
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error, {MakeAttribute("axis", int64_t(2))});
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, SoftMaxGrad) {
  GradientCheckerSoftmaxGradHelper(false);
}

TEST(GradientCheckerTest, LogSoftMaxGrad) {
  GradientCheckerSoftmaxGradHelper(true);
}

TEST(GradientCheckerTest, GeluGrad) {
  UnaryOpGradientTest("Gelu", kMSDomain, 1);
}

TEST(GradientCheckerTest, FastGeluGrad) {
  UnaryOpGradientTest("FastGelu", kMSDomain, 1);
}

// used for BiasGelu and FastGelu
void TestBiasGeluGrad(const std::string& op_type, const std::string& domain, int opset_version) {
  const TensorShape input_shape({2, 3, 4});
  const TensorShape bias_shape({4});

  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op_type, domain, opset_version};
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  float max_error;
  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
      op_def, {input_shape, bias_shape}, {input_shape}, &max_error,
      attributes, true, true));

  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, FastGeluGrad_Bias) {
  TestBiasGeluGrad("FastGelu", kMSDomain, 1);
}

TEST(GradientCheckerTest, BiasGeluGrad) {
  TestBiasGeluGrad("BiasGelu", kMSDomain, 1);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // NDEBUG
