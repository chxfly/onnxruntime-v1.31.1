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
#ifndef USE_CUDA

TEST(GradientCheckerTest, SplitGrad) {
  TensorShape shape({9, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Split"};

  gradient_checker.ComputeGradientError(op_def, {shape}, {{3, 5}, {3, 5}, {3, 5}}, &max_error,
                                        {MakeAttribute("axis", int64_t(0))});
  EXPECT_IS_TINY(max_error);
}

static void TestConcatOpGrad(const std::string& op_type,
                             const std::string& domain = kOnnxDomain,
                             int opset_version = 9,
                             bool check_not_have_shape_inferencing = false) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  const bool extra_input = op_type == "ConcatTraining";
  OpDef op_def{op_type, domain, opset_version};

  //concat_1d
  {
    TensorShape x_shape({2});
    TensorShape y_shape({6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(0))}, true,
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //concat_2d
  {
    TensorShape x_shape({2, 2});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(1))}, true,
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //concat_3d
  {
    TensorShape x_shape({1, 2, 3});
    TensorShape y_shape({1, 2, 9});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(2))}, true,
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //concat_different_shape
  {
    TensorShape x1_shape({2, 2});
    TensorShape x2_shape({2, 4});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x1_shape, x2_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(1))}, true,
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //concat_different_shape_and_negative_axis
  {
    TensorShape x1_shape({2, 2});
    TensorShape x2_shape({2, 4});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x1_shape, x2_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(-1))}, true,
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ConcatGrad) {
  TestConcatOpGrad("Concat");
}

TEST(GradientCheckerTest, ConcatTrainingGrad) { /*also test w/o shape inferencing */
  TestConcatOpGrad("ConcatTraining", kMSDomain, 1, true);
}

#endif

static bool SimplifyReshape(const std::vector<Dimension>& target_shape,  // the output shape of Reshape
                            const std::vector<Dimension>& source_shape,  // the input shape of Reshape
                            TensorProto& shape_tensor) {                 // the simplified shape tensor if succeeded
  std::vector<int64_t> shape_const;
  std::list<std::string> target_dim_params;
  std::list<std::string> source_dim_params;
  auto get_dim_params = [](const std::vector<Dimension>& shape, std::list<std::string>& dim_params) {
    for (const auto& dim : shape) {
      if (utils::HasDimParam(dim)) {
        dim_params.push_back(dim.dim_param());
      } else if (utils::HasDimValue(dim)) {
        dim_params.push_back("");
      } else {
        return false;
      }
    }
    //trim empty strings in the tail of list
    while (!dim_params.empty() && dim_params.back().empty()) {
      dim_params.pop_back();
    }
    return true;
  };

  if (get_dim_params(target_shape, target_dim_params) &&
      get_dim_params(source_shape, source_dim_params) &&
      target_dim_params == source_dim_params) {
    for (const auto& dim : target_shape) {
      if (utils::HasDimParam(dim)) {
        shape_const.push_back(0);
      } else {
        shape_const.push_back(dim.dim_value());
      }
    }
    auto t = ToTensor<int64_t>(shape_const);
    t.add_dims(shape_const.size());
    shape_tensor.CopyFrom(t);
    return true;
  }
  return false;
}

IMPLEMENT_GRADIENT_BUILDER(GetReshapeGradient) {
  std::vector<Dimension> target_shape;
  std::vector<Dimension> source_shape;
  if (GetShape(I(0), target_shape).IsOK() &&
      GetShape(GO(0), source_shape).IsOK()) {
    TensorProto shape_tensor;
    if (SimplifyReshape(target_shape,
                        source_shape,
                        shape_tensor)) {
      return std::vector<NodeDef>{
          NodeDef("Constant",
                  {},
                  {IA("x_shape")},
                  {MakeAttribute("value", shape_tensor)}),
          NodeDef("Reshape",
                  {GO(0), IA("x_shape")},
                  {GI(0)})};
    }
  }
  return std::vector<NodeDef>{
      NodeDef("ReshapeGrad",
              {I(0), GO(0)},
              {GI(0)})};
}

TEST(GradientCheckerTest, TransposeGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Transpose"};
  float error_tolerance = 1e-3f;

  // default
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({4, 3, 2});
    const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          attributes, true, true /*also test w/o shape inferencing */);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 012
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({2, 3, 4});
    std::vector<int64_t> perm{0, 1, 2};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 021
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({2, 4, 3});
    std::vector<int64_t> perm{0, 2, 1};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 102
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({3, 2, 4});
    std::vector<int64_t> perm{1, 0, 2};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 120
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({3, 4, 2});
    std::vector<int64_t> perm{1, 2, 0};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 201
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({4, 2, 3});
    std::vector<int64_t> perm{2, 0, 1};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 210
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({4, 3, 2});
    std::vector<int64_t> perm{2, 1, 0};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, GatherGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Gather"};

  TensorInfo x_info({5, 4, 3, 2});
  std::function<float(float)> transformer = [](float x) { return std::fmod(7 * std::fabs(x), 5.0f); };

  // gather_0 without duplicated indices
  {
    int num_indices = 2;
    TensorInfo indices_info({num_indices}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 0;
    y_shape[axis] = num_indices;

    gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", axis)});
    EXPECT_IS_TINY(max_error);
  }

  // gather_0 with duplicated indices
  {
    int num_indices = 10;
    TensorInfo indices_info({num_indices}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 0;
    y_shape[axis] = num_indices;

    gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", axis)});
    EXPECT_IS_TINY(max_error);
  }

  // gather_1
  {
    int num_indices = 8;
    std::function<float(float)> transformer2 = [](float x) { return std::fmod(7 * std::fabs(x), 4.0f); };
    TensorInfo indices_info({num_indices}, false, &transformer2, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 1;
    y_shape[axis] = num_indices;

    gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", axis)});
    EXPECT_IS_TINY(max_error);
  }

  // 2D Indices
  {
    TensorInfo indices_info({2, 3}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{2, 3, 4, 3, 2};

    gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", int64_t(0))});
    EXPECT_IS_TINY(max_error);
  }
}

#ifdef USE_CUDA
TEST(GradientCheckerTest, GatherNDGrad_repeat_float_data) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherND", kOnnxDomain, 12};

  TensorInfo x_info({2, 2}, true);
  TensorInfo indice_info({2, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3}, {1, 1, 1, 1}};

  TensorInfo y_info({2}, true);
  int64_t batch_dims = 0;

  gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("batch_dims", batch_dims)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherNDGrad_unique_float_data) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherND", kOnnxDomain, 12};

  {
    TensorInfo x_info({2, 2}, true);
    TensorInfo indice_info({2, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3}, {0, 1, 1, 0}};

    TensorInfo y_info({2}, true);
    int64_t batch_dims = 0;

    gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("batch_dims", batch_dims)});
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 2, 3}, true);
    TensorInfo indice_info({2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 0}};

    TensorInfo y_info({2, 3}, true);
    int64_t batch_dims = 1;

    gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("batch_dims", batch_dims)});
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 2, 3}, true);
    TensorInfo indice_info({2, 2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 0, 2, 1}};

    TensorInfo y_info({2, 2}, true);
    int64_t batch_dims = 2;

    gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("batch_dims", batch_dims)});
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, GatherElementsGradWithDuplicateUpdate) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherElements", kOnnxDomain, 11};

  TensorInfo data_info({3, 3}, true);
  TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 0, 2, 0, 0}};

  TensorInfo y_info({2, 3}, true);
  int64_t axis = 0;

  gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error, x_datas,
                                        {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherElementsGradWithoutDuplicateUpdate) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherElements", kOnnxDomain, 11};

  TensorInfo data_info({3, 3}, true);
  TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 2, 2, 2}};

  TensorInfo y_info({2, 3}, true);
  int64_t axis = 0;

  gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error, x_datas,
                                        {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherElementsGradAxisWithDuplicateUpdate) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherElements", kOnnxDomain, 11};

  TensorInfo data_info({3, 3}, true);
  TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 1}};

  TensorInfo y_info({2, 3}, true);
  int64_t axis = 1;

  gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error, x_datas,
                                        {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}
#endif

TEST(GradientCheckerTest, UnsqueezeGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Unsqueeze"};
  float error_tolerance = 1e-3f;

  {
    TensorShape x_shape({2, 3});
    TensorShape y_shape({1, 2, 3, 1});
    std::vector<int64_t> axes{0, 3};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({2, 3});
    TensorShape y_shape({1, 1, 2, 3});
    std::vector<int64_t> axes{0, 1};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({2, 3});
    TensorShape y_shape({1, 2, 1, 3, 1});
    std::vector<int64_t> axes{0, 2, 4};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, SqueezeGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Squeeze"};
  float error_tolerance = 1e-3f;

  {
    TensorShape x_shape({1, 2, 3, 1});
    TensorShape y_shape({2, 3});
    std::vector<int64_t> axes{0, 3};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({1, 1, 2, 3, 4});
    TensorShape y_shape({2, 3, 4});
    std::vector<int64_t> axes{0, 1};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({1, 2, 1, 3, 1});
    TensorShape y_shape({2, 3});
    std::vector<int64_t> axes{0, 2, 4};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({1, 2, 1, 3, 1});
    TensorShape y_shape({1, 2, 3, 1});
    std::vector<int64_t> axes{2};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  /* TODO: enable test with no axis when squeeze kernel is fixed (separate bug filed)
  {
    TensorShape x_shape({1, 2, 1, 3, 1});
    TensorShape y_shape({2, 3});
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
  */
}

TEST(GradientCheckerTest, SliceGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Slice", kOnnxDomain, 10};

  // default values for optional tensors like axes and steps.
  {
    TensorInfo x_info({2, 4}, true);
    TensorInfo start_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo end_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 0}, {2, 3}};

    TensorInfo y_info({1, 3}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, start_info, end_info}, {y_info}, &max_error, x_datas);
    EXPECT_IS_TINY(max_error);
  }

  // all input tensors have some value and slice end out of bound.
  {
    TensorInfo x_info({2, 4}, true);
    TensorInfo start_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo end_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo axes_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo steps_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 0}, {2, 3}, {0, 1}, {1, 2}};

    TensorInfo y_info({1, 2}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, start_info, end_info, axes_info, steps_info},
                                          {y_info}, &max_error, x_datas);

    EXPECT_IS_TINY(max_error);
  }

  // 3-D tensor
  {
    TensorInfo x_info({2, 4, 2}, true);
    TensorInfo start_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo end_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo axes_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo steps_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8}, {1, 0}, {2, 3}, {0, 1}, {1, 2}};

    TensorInfo y_info({1, 2, 2}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, start_info, end_info, axes_info, steps_info}, {y_info},
                                          &max_error, x_datas);

    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ExpandGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Expand"};
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  //input_shape = (2, 3, 1), target_shape = (2, 3, 4) ==> shape(result) = (2, 3, 4)
  {
    TensorInfo x_info({2, 3, 1}, true);
    TensorInfo shape_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {2, 3, 4}};

    TensorInfo y_info({2, 3, 4}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (2, 3, 1), target_shape = (1, 1, 4) ==> shape(result) = (2, 3, 4)
  {
    TensorInfo x_info({2, 3, 1}, true);
    TensorInfo shape_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {1, 1, 4}};

    TensorInfo y_info({2, 3, 4}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (2, 3, 1), target_shape = (4) ==> shape(result) = (2, 3, 4)
  {
    TensorInfo x_info({2, 3, 1}, true);
    TensorInfo shape_info({1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {4}};

    TensorInfo y_info({2, 3, 4}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (2, 3, 1), target_shape = (1, 1) ==> shape(result) = (2, 3, 1)
  {
    TensorInfo x_info({2, 3, 1}, true);
    TensorInfo shape_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {1, 1}};

    TensorInfo y_info({2, 3, 1}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (2, 3), target_shape = (4, 5, 2, 3) ==> shape(result) = (4, 5, 2, 3)
  {
    TensorInfo x_info({2, 3}, true);
    TensorInfo shape_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {4, 5, 2, 3}};

    TensorInfo y_info({4, 5, 2, 3}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (1, 2, 3), target_shape = (4, 5, 1, 1) ==> shape(result) = (4, 5, 2, 3)
  {
    TensorInfo x_info({1, 2, 3}, true);
    TensorInfo shape_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {4, 5, 1, 1}};

    TensorInfo y_info({4, 5, 2, 3}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // NDEBUG
