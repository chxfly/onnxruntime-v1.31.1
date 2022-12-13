// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/upsample.h"
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif
#include "core/providers/cpu/tensor/upsamplebase.h"
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {

struct FilterParamsBaseAA {
  std::vector<int64_t> bound;
  std::vector<float> original;
  int64_t window_size = 2;
  BufferUniquePtr weight_coefficients;
};

struct FilterParamsAA {
  float support_size = 2.0f;
  float cubic_coeff_a = -0.75f;

  /* Handles values form -640 to 639. */
  uint8_t* clip8_lookups_table;

  FilterParamsBaseAA dim_x;
  FilterParamsBaseAA dim_y;
  FilterParamsBaseAA dim_z;

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L20-L29

  static int32_t round_up(float f) {
    return ((int32_t)((f) >= 0.0 ? (f) + 0.5F : (f)-0.5F));
  }

  void init_clip_lookup() {
    if (clip8_lookups_table[1279] == 255) {
      return;
    }
    for (int i = 0; i < 1280; ++i) {
      clip8_lookups_table[i] = static_cast<uint8_t>(std::min(std::max(i - 640, 0), 255));
    }
  }
  virtual float filter(float x) const = 0;
};

struct BilinearParamsAA : FilterParamsAA {
  float filter(float x) const override {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

struct BiCubicParamsAA : FilterParamsAA {
  BiCubicParamsAA() {
    support_size = (4.0f);
  }
  float filter(float x) const override {
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
     */
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return ((cubic_coeff_a + 2.0f) * x - (cubic_coeff_a + 3.0f)) * x * x + 1;
    }
    if (x < 2.0f) {
      return (((x - 5.0f) * x + 8.f) * x - 4.f) * cubic_coeff_a;
    }
    return 0.0f;
  }
};

struct TriLinearParamsAA : FilterParamsAA {
  float filter(float x) const override {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

template <typename T>
struct AccumulateType {
  using type = int32_t;
  using Dtype = T;
};

template <>
struct AccumulateType<float> {
  using type = float;
};

template <>
struct AccumulateType<double> {
  using type = double;
};

void SetupUpsampleFilterAA(FilterParamsAA& p,
                           const gsl::span<int64_t> input_h_w_c,
                           const gsl::span<int64_t> output_h_w_c,
                           const gsl::span<float> scale_h_w_c,
                           const std::vector<float>& roi,
                           AllocatorPtr& alloc,
                           const GetOriginalCoordinateFunc& get_original_coordinate,
                           const int32_t dtype, bool exclude_outside, const bool is_nchw);

template <typename T>
void UpsampleBaseAA(FilterParamsAA& p,
                    const int64_t batch_size,
                    const int64_t num_channels,
                    const int64_t input_height,
                    const int64_t input_width,
                    const int64_t output_height,
                    const int64_t output_width,
                    const bool use_extrapolation,
                    const float extrapolation_value,
                    const T* const XdataBase,
                    T* const YdataBase,
                    AllocatorPtr& alloc,
                    concurrency::ThreadPool* tp) {
  const uint8_t* clip8_lookups = &p.clip8_lookups_table[640];

  constexpr bool is_8bit_data =
      (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value);

  auto image_temp_buffer = BufferUniquePtr(alloc->Alloc(input_height *
                                                        output_width * num_channels * sizeof(T)),
                                           BufferDeleter(alloc));

  int32_t mag_factor = 1 << (22 - 1);
  using ACtype = typename AccumulateType<T>::type;

  for (int64_t n = 0; n < batch_size; ++n) {
    auto* temp_buffer = static_cast<T*>(image_temp_buffer.get());
    // horizon interpolate
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_channels,
        [&](std::ptrdiff_t c) {
          const T* const Xdata =
              XdataBase + (n * num_channels + (c)) *
                              (input_height * input_width);
          T* const Ydata = temp_buffer + (n * num_channels + (c)) *
                                             (input_height * output_width);
          for (int64_t y = 0; y < input_height; ++y) {
            for (int64_t x = 0; x < output_width; ++x) {
              const int64_t output_offset = output_width * y + x;
              // when use_extrapolation is set and original index of x or y is out of the dim range
              // then use extrapolation_value as the output value.
              if (use_extrapolation &&
                  ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                   (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
                Ydata[output_offset] = static_cast<T>(extrapolation_value);
                continue;
              }
              ACtype output = 0;
              if constexpr (is_8bit_data) {
                output = mag_factor;
              }
              const auto* weight_coeff =
                  reinterpret_cast<const ACtype*>(p.dim_x.weight_coefficients.get()) +
                  p.dim_x.window_size * x;
              int64_t xmin = p.dim_x.bound[x * 2];
              int64_t xmax = p.dim_x.bound[x * 2 + 1];
              for (; xmin < xmax; ++xmin) {
                output += Xdata[y * input_width + xmin] * (*weight_coeff++);
              }

              if constexpr (is_8bit_data) {
                Ydata[output_offset] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata[output_offset] = p.round_up(output);
              } else {
                Ydata[output_offset] = static_cast<T>(output);
              }
            }
          }
        });
    // vertical interpolate
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_channels,
        [&](std::ptrdiff_t c) {
          const T* const Xdata =
              temp_buffer + (n * num_channels + (c)) *
                                (input_height * output_width);
          T* const Ydata =
              YdataBase + (n * num_channels + (c)) *
                              (output_height * output_width);
          for (int64_t y = 0; y < output_height; ++y) {
            for (int64_t x = 0; x < output_width; ++x) {
              const int64_t output_offset = output_width * y + x;
              if (use_extrapolation &&
                  ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                   (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
                Ydata[output_offset] = static_cast<T>(extrapolation_value);
                continue;
              }

              ACtype output = 0;
              if constexpr (is_8bit_data) {
                output = mag_factor;
              }
              const auto* weight_coeff =
                  reinterpret_cast<const ACtype*>(p.dim_y.weight_coefficients.get()) +
                  p.dim_y.window_size * y;
              int64_t ymin = p.dim_y.bound[y * 2];
              int64_t ymax = p.dim_y.bound[y * 2 + 1];
              for (; ymin < ymax; ++ymin) {
                output +=
                    Xdata[ymin * output_width + x] * (*weight_coeff++);
              }
              if constexpr (is_8bit_data) {
                Ydata[output_offset] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata[output_offset] = p.round_up(output);
              } else {  // float double
                Ydata[output_offset] = static_cast<T>(output);
              }
            }
          }
        });
  }
}

template <typename T>
void UpsampleBilinearAA(const int64_t batch_size,
                        const int64_t num_channels,
                        const int64_t input_height,
                        const int64_t input_width,
                        const int64_t output_height,
                        const int64_t output_width,
                        const float height_scale,
                        const float width_scale,
                        const std::vector<float>& roi,
                        const bool use_extrapolation,
                        const float extrapolation_value,
                        bool exclude_outside,
                        const Tensor* const X,
                        T* const YdataBase,
                        AllocatorPtr& alloc,
                        const GetOriginalCoordinateFunc& get_original_coordinate,
                        concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();

  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BilinearParamsAA p;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), exclude_outside, true);
  return UpsampleBaseAA<T>(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                           use_extrapolation, extrapolation_value,
                           XdataBase, YdataBase, alloc, tp);
}

template <typename T>
void NhwcUpsampleBilinearAA(const int64_t batch_size,
                            const int64_t num_channels,
                            const int64_t input_height,
                            const int64_t input_width,
                            const int64_t output_height,
                            const int64_t output_width,
                            const float height_scale,
                            const float width_scale,
                            const std::vector<float>& roi,
                            const bool use_extrapolation,
                            const float extrapolation_value,
                            bool exclude_outside,
                            const Tensor* const X,
                            T* const YdataBase,
                            AllocatorPtr& alloc,
                            const GetOriginalCoordinateFunc& get_original_coordinate,
                            concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();

  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BilinearParamsAA p;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), exclude_outside, false);
  return NhwcUpsampleBasicAA(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                             use_extrapolation, extrapolation_value,
                             XdataBase, YdataBase, alloc, tp);
}

template <typename T>
void NhwcResizeBiCubicAA(const int64_t batch_size,
                         const int64_t num_channels,
                         const int64_t input_height,
                         const int64_t input_width,
                         const int64_t output_height,
                         const int64_t output_width,
                         const float height_scale,
                         const float width_scale,
                         float cubic_coeff_a,
                         bool use_extrapolation,
                         float extrapolation_value,
                         bool exclude_outside,
                         const std::vector<float>& roi,
                         const Tensor* const X,
                         T* const YdataBase,
                         AllocatorPtr& alloc,
                         const GetOriginalCoordinateFunc& get_original_coordinate,
                         concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();

  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BiCubicParamsAA p;
  p.cubic_coeff_a = cubic_coeff_a;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), exclude_outside, false);
  return NhwcUpsampleBasicAA(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                             use_extrapolation, extrapolation_value,
                             XdataBase, YdataBase, alloc, tp);
}

template <typename T>
void NhwcUpsampleBasicAA(FilterParamsAA& p,
                         const int64_t batch_size,
                         const int64_t num_channels,
                         const int64_t input_height,
                         const int64_t input_width,
                         const int64_t output_height,
                         const int64_t output_width,
                         const bool use_extrapolation,
                         const float extrapolation_value,
                         const T* const XdataBase,
                         T* const YdataBase,
                         AllocatorPtr& alloc,
                         concurrency::ThreadPool* tp) {
  const uint8_t* clip8_lookups = &p.clip8_lookups_table[640];

  constexpr bool is_8bit_data =
      (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value);

  auto image_temp_buffer = BufferUniquePtr(alloc->Alloc(input_height *
                                                        output_width * num_channels * sizeof(T)),
                                           BufferDeleter(alloc));

  using ACtype = typename AccumulateType<T>::type;
  int32_t mag_factor = 1 << (22 - 1);

  for (int64_t n = 0; n < batch_size; ++n) {
    auto* temp_buffer = static_cast<T*>(image_temp_buffer.get());

    // horizon interpolate
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(input_height) * output_width,
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          const T* const Xdata =
              XdataBase +
              n * (input_height * input_width) * num_channels;
          T* const Ydata =
              temp_buffer + n * (input_height * output_width) * num_channels;
          for (std::ptrdiff_t i = first; i < last; ++i) {
            const auto x = static_cast<int32_t>(i % output_width);
            const auto y = static_cast<int32_t>(i / output_width);
            T* const Ydata_with_offset = Ydata + (output_width * y + x) * num_channels;
            if (use_extrapolation && ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                                      (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
              for (int64_t c = 0; c < num_channels; ++c) {
                Ydata_with_offset[c] = static_cast<T>(extrapolation_value);
              }
              continue;
            }

            const auto* weight_coeff =
                reinterpret_cast<const ACtype*>(p.dim_x.weight_coefficients.get()) +
                p.dim_x.window_size * x;
            int64_t xmin = p.dim_x.bound[x * 2];
            int64_t xmax = p.dim_x.bound[x * 2 + 1];
            for (int64_t c = 0; c < num_channels; ++c) {
              const auto* weight_coeff_start = weight_coeff;
              ACtype output = 0;
              if constexpr (is_8bit_data) {
                output = mag_factor;
              }
              for (int idx = xmin; idx < xmax; ++idx) {
                // printf("%f*%f + ", *weight_coeff_start, Xdata[(y * input_width + idx) * num_channels + c]);

                output += Xdata[(y * input_width + idx) * num_channels + c] *
                          (*weight_coeff_start++);
              }
              // printf("=%f\n", output);
              if constexpr (is_8bit_data) {
                Ydata_with_offset[c] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata_with_offset[c] = p.round_up(output);
              } else {  // float double
                Ydata_with_offset[c] = output;
              }
            }
          }
        });

    // vertical interpolate
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(output_height) * output_width,
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          const T* const Xdata = temp_buffer + n * (input_height * output_width) * num_channels;
          T* const Ydata = YdataBase + n * (output_height * output_width) * num_channels;

          for (std::ptrdiff_t i = first; i < last; ++i) {
            const auto x = static_cast<int32_t>(i % output_width);
            const auto y = static_cast<int32_t>(i / output_width);
            T* const Ydata_with_offset = Ydata + (output_width * y + x) * num_channels;

            if (use_extrapolation && ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                                      (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
              for (int64_t c = 0; c < num_channels; ++c) {
                Ydata_with_offset[c] = static_cast<T>(extrapolation_value);
              }
              continue;
            }

            const auto* weight_coeff =
                reinterpret_cast<const ACtype*>(p.dim_y.weight_coefficients.get()) + p.dim_y.window_size * y;
            int64_t ymin = p.dim_y.bound[y * 2];
            int64_t ymax = p.dim_y.bound[y * 2 + 1];

            for (int64_t c = 0; c < num_channels; ++c) {
              const auto* weight_coeff_start = weight_coeff;
              ACtype output = 0;
              if constexpr (is_8bit_data) {
                output = mag_factor;
              }
              for (int idy = ymin; idy < ymax; ++idy) {
                output += Xdata[(idy * output_width + x) * num_channels + c] *
                          (*weight_coeff_start++);
              }
              if constexpr (is_8bit_data) {
                Ydata_with_offset[c] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata_with_offset[c] = p.round_up(output);
              } else {  // float double
                Ydata_with_offset[c] = output;
              }
            }
          }
        });
  }
}

template <typename T>
inline void InterpolateCompute(const T* Xdata, const int64_t stride,
                               const float* weight_coeff,
                               const int64_t* idx_bound, T* Ydata) {
  float output = 0;

  for (int64_t idx = idx_bound[0]; idx < idx_bound[1]; ++idx) {
    output += Xdata[idx * stride] * (*weight_coeff++);
  }
  *Ydata = static_cast<T>(output);
}

template <typename T>
inline void InterpolateLoopForSingleDim(
    const T* Xdata, int64_t dim_size, int64_t section_idx,
    const int64_t y_stride, const int64_t x_stride, const float* weight_coeff,
    const std::vector<int64_t>& idx_bound, int64_t window_size, T* Ydata) {
  for (int64_t step = 0; step < dim_size; ++step) {
    InterpolateCompute(
        Xdata, y_stride,
        &weight_coeff[(x_stride == 0) ? window_size * step
                                      : window_size * section_idx],
        &idx_bound[(x_stride == 0) ? step * 2 : section_idx * 2], Ydata);
    Ydata++;
    Xdata += x_stride;
  }
}

template <typename T>
inline void
LoopInDimN(const T* Xdata_base, int64_t dim_idx, int64_t pre_level_idx, int64_t section_idx,
           const InlinedVector<int64_t>& input_stride,
           const InlinedVector<int64_t>& output_stride, const int64_t compute_dim,
           const float* weight_coeff, const std::vector<int64_t>& idx_bound,
           int64_t window_size, T* Ydata_base) {
  const int64_t x_ofs =
      pre_level_idx * ((compute_dim == dim_idx) ? 0 : input_stride[dim_idx]);

  const T* const Xdata = Xdata_base + x_ofs;
  T* const Ydata = Ydata_base + pre_level_idx * output_stride[dim_idx];
  if (dim_idx < int64_t(input_stride.size()) - 2) {
    for (int64_t sub_sec_idx = 0;
         sub_sec_idx < output_stride[dim_idx] / output_stride[dim_idx + 1];
         ++sub_sec_idx) {
      section_idx = (compute_dim == dim_idx + 1) ? sub_sec_idx : section_idx;
      LoopInDimN(Xdata, dim_idx + 1, sub_sec_idx, section_idx, input_stride,
                 output_stride, compute_dim, weight_coeff, idx_bound,
                 window_size, Ydata);
    }
    return;
  }
  int64_t x_stride = compute_dim == int64_t(output_stride.size()) - 1 ? 0 : 1;
  InterpolateLoopForSingleDim(
      Xdata, output_stride[dim_idx] / output_stride[dim_idx + 1], section_idx,
      output_stride[compute_dim], x_stride, weight_coeff, idx_bound,
      window_size, Ydata);
}

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 6001)
#endif
template <typename T>
void ResizeBiCubicAA(int64_t batch_size,
                     int64_t num_channels,
                     int64_t input_height,
                     int64_t input_width,
                     int64_t output_height,
                     int64_t output_width,
                     float height_scale,
                     float width_scale,
                     float cubic_coeff_a,
                     bool use_extrapolation,
                     float extrapolation_value,
                     bool exclude_outside,
                     const std::vector<float>& roi,
                     const Tensor* X,
                     T* YdataBase,
                     AllocatorPtr& alloc,
                     const GetOriginalCoordinateFunc& get_original_coordinate,
                     concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();
  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BiCubicParamsAA p;
  p.cubic_coeff_a = cubic_coeff_a;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), exclude_outside, false);

  return UpsampleBaseAA<T>(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                           use_extrapolation, extrapolation_value,
                           XdataBase, YdataBase, alloc, tp);
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

template <typename T>
void UpsampleTrilinearAA(int64_t batch_size,
                          int64_t num_channels,
                          int64_t input_depth,
                          int64_t input_height,
                          int64_t input_width,
                          int64_t output_depth,
                          int64_t output_height,
                          int64_t output_width,
                          float depth_scale,
                          float height_scale,
                          float width_scale,
                          const std::vector<float>& roi,
                          bool use_extrapolation,
                          float extrapolation_value,
                          const Tensor* X,
                          T* YdataBase,
                          AllocatorPtr& alloc,
                          const GetOriginalCoordinateFunc& get_original_coordinate,
                          concurrency::ThreadPool* tp) {

  //using ACtype = typename AccumulateType<T>::type;
  if constexpr (std::is_same<T, float>::value != true){
    return;
  }
  const auto* XdataBase = X->Data<T>();
  int64_t input_paras[] = {input_height, input_width, input_depth};
  int64_t output_paras[] = {output_height, output_width, output_depth};
  float scale_paras[] = {height_scale, width_scale, depth_scale};

  TriLinearParamsAA p;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), true, false);

  auto* buffer = alloc->Alloc(sizeof(T) * input_height * output_width *
                              input_depth * num_channels);
  auto temp1 = BufferUniquePtr(buffer, BufferDeleter(alloc));

  buffer = alloc->Alloc(sizeof(T) * output_height * output_width *
                        input_depth * num_channels);
  auto temp2 = BufferUniquePtr(buffer, BufferDeleter(alloc));

  InlinedVector<int64_t> in_shape = {batch_size, input_height, input_width,
                                     num_channels};
  InlinedVector<int64_t> tmp_shape = {batch_size, input_height, output_width,
                                      num_channels};
  InlinedVector<int64_t> out_shape = {batch_size, output_height, output_width,
                                      num_channels};
  InlinedVector<int64_t> in_stride = {1, 1, 1, 1, 1, 1};
  InlinedVector<int64_t> tmp_stride = in_stride;
  InlinedVector<int64_t> out_stride = in_stride;
  for (int64_t i = int64_t(in_shape.size() - 1); i >= 0; i--) {
    in_stride[i] = in_stride[i + 1] * in_shape[i];
    tmp_stride[i] = tmp_stride[i + 1] * tmp_shape[i];
    out_stride[i] = out_stride[i + 1] * out_shape[i];
  }

  if(use_extrapolation){
    YdataBase[0] = extrapolation_value;
  }
  for (int64_t n = 0; n < batch_size; ++n) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, static_cast<std::ptrdiff_t>(num_channels),
        [&](std::ptrdiff_t c) {
          const T* Xdata = XdataBase + (n * num_channels + c) * (input_depth * input_height * input_width);
          const auto* weight = static_cast<const float*>(p.dim_y.weight_coefficients.get());
          T* Ydata = static_cast<T*>(temp1.get());

          LoopInDimN<T>(Xdata, 2, c, 0, in_stride, tmp_stride, 5,
                        weight, p.dim_x.bound, p.dim_x.window_size,
                        Ydata);
        });
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, static_cast<std::ptrdiff_t>(num_channels),
        [&](std::ptrdiff_t c) {
          auto Xdatabase_temp = static_cast<T*>(temp1.get());
          const T* Xdata = Xdatabase_temp + (c) * (input_depth * input_height * output_width);
          const auto* weight = static_cast<const float*>(p.dim_y.weight_coefficients.get());
          T* Ydata = static_cast<T*>(temp2.get());
          LoopInDimN<T>(Xdata, 2, c, 0, tmp_stride, out_stride, 4,
                        weight, p.dim_y.bound, p.dim_y.window_size,
                        Ydata);
        });
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, static_cast<std::ptrdiff_t>(num_channels),
        [&](std::ptrdiff_t c) {
          auto Xdatabase_temp = static_cast<T*>(temp2.get());
          const auto* weight = static_cast<const float*>(p.dim_y.weight_coefficients.get());

          const T* Xdata = Xdatabase_temp + (c) * (input_depth * output_height * output_width);
          T* Ydata = YdataBase + (n * num_channels + c) * (output_depth * output_height * output_width);
          LoopInDimN<T>(Xdata, 2, c, 0, tmp_stride, out_stride, 3,
                        weight, p.dim_z.bound, p.dim_z.window_size,
                        Ydata);
        });
  }
}

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
