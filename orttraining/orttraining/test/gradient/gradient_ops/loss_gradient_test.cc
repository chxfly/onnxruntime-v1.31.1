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

template <typename T>
void GenerateRandomDataWithOneHot(
    std::vector<std::vector<float>>& x_datas,
    std::vector<TensorShape> input_shapes,
    const std::unordered_set<int>& one_hot_input_indices) {
  for (int i = 0; i < 2; i++) {
    // TODO: Consider varying mean and variance
    float scale = 5.f;
    float mean = 0.f;
    const uint32_t seed = GetTestRandomSeed();

    std::default_random_engine generator{gsl::narrow_cast<decltype(generator)::result_type>(seed)};
    std::normal_distribution<T> distribution{mean, scale};

    auto x_data_length = input_shapes[i].Size();
    x_datas[i].resize(x_data_length);

    if (one_hot_input_indices.count(i) > 0 && input_shapes[i].NumDimensions() > 1) {
      int64_t N = input_shapes[i].SizeToDimension(input_shapes[i].NumDimensions() - 1);
      int64_t D = input_shapes[i][input_shapes[i].NumDimensions() - 1];

      std::fill(x_datas[i].begin(), x_datas[i].end(), (T)0);
      for (int64_t k = 0; k < N; k++)
        x_datas[i][k * D + (seed % D)] = (T)1;
    } else {
      std::generate(x_datas[i].begin(), x_datas[i].end(), [&] { return distribution(generator); });
    }
  }
}

void TestSoftmaxCrossEntropyGrad(const TensorShape& input_shape, const std::string& reduction) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropy", kMSDomain, 1};

  std::vector<std::vector<float>> x_datas(2);
  GenerateRandomDataWithOneHot<float>(x_datas, {input_shape, input_shape}, {1});

  gradient_checker.ComputeGradientError(op_def, {input_shape, {input_shape, false}},
                                        {{1}, {input_shape, false}}, &max_error, x_datas,
                                        {MakeAttribute("reduction", reduction)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, SoftmaxCrossEntropyGrad) {
  TestSoftmaxCrossEntropyGrad({5, 11}, "mean");
  TestSoftmaxCrossEntropyGrad({5, 11}, "sum");
  TestSoftmaxCrossEntropyGrad({2, 3, 2, 11}, "mean");
  TestSoftmaxCrossEntropyGrad({2, 3, 2, 11}, "sum");
}

void TestSparseSoftmaxCrossEntropyGrad(const TensorShape& index_shape, const std::string& reduction) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SparseSoftmaxCrossEntropy"};

  const int64_t D = 7;
  std::function<float(float)> transformer_index = [](float x) { return std::fmod(std::fabs(x) * 5.0f, 7.0f); };
  std::function<float(float)> transformer_weight = [](float x) { return std::fmod(std::fabs(x), 2.0f); };

  // without weight
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    logit_shape.emplace_back(D);

    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info},
                                          {{1}, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }

  // with weight
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    logit_shape.emplace_back(D);

    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo weight_info(index_shape, false, &transformer_weight);

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info, weight_info},
                                          {{1}, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, SparseSoftmaxCrossEntropyGrad) {
  TestSparseSoftmaxCrossEntropyGrad({5}, "mean");
  TestSparseSoftmaxCrossEntropyGrad({5}, "sum");
  TestSparseSoftmaxCrossEntropyGrad({2, 3, 2}, "mean");
  TestSparseSoftmaxCrossEntropyGrad({2, 3, 2}, "sum");
}

void TestSoftmaxCrossEntropyLossGrad(const TensorShape& index_shape,  //label_shape
                                     const std::string& reduction,
                                     int64_t ignore_index = 0,
                                     int64_t D = 2 /* num_class*/) {
  float max_error;
  bool include_ignore_index = false;
  bool insert_ignore_index = false;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropyLoss", kOnnxDomain, 12};
  std::function<float(float)> transformer_index = [D, &include_ignore_index, &insert_ignore_index, ignore_index](float x) {
    if (include_ignore_index) {
      if (insert_ignore_index) {
        insert_ignore_index = false;
        return static_cast<float>(ignore_index);
      } else {
        insert_ignore_index = true;
        return std::fmod(std::fabs(x) * 5.0f, D * 1.0f);
      }
    } else {
      return std::fmod(std::fabs(x) * 5.0f, D * 1.0f);
    }
  };

  std::function<float(float)> transformer_weight = [](float x) { return std::fmod(std::fabs(x), 2.0f); };

  // without weight and ignore_index
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = true;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info},
                                          {loss_info, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }

  // with weight and no ignore_index
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = false;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo weight_info({logit_shape[1]}, false, &transformer_weight);

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info, weight_info},
                                          {loss_info, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }

  // without weight and ignore index
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = true;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info},
                                          {loss_info, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction), MakeAttribute("ignore_index", ignore_index)});
    EXPECT_IS_TINY(max_error);
  }

  // with weight and ignore_index
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = true;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo weight_info({logit_shape[1]}, false, &transformer_weight);

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info, weight_info},
                                          {loss_info, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction), MakeAttribute("ignore_index", ignore_index)});
    EXPECT_IS_TINY(max_error);
  }
}

// TODO fix flaky test
// failing random seed: 1
TEST(GradientCheckerTest, DISABLED_SoftmaxCrossEntropyLossGrad) {
  TestSoftmaxCrossEntropyLossGrad({5}, "mean");
  TestSoftmaxCrossEntropyLossGrad({5}, "sum");
  TestSoftmaxCrossEntropyLossGrad({2}, "none");
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "mean");
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "sum");
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "none");
  TestSoftmaxCrossEntropyLossGrad({5}, "mean", -1);
  TestSoftmaxCrossEntropyLossGrad({5}, "sum", -1);
  TestSoftmaxCrossEntropyLossGrad({2}, "none", -1);
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "mean", -1);
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "sum", -1);
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "none", -1);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // NDEBUG
