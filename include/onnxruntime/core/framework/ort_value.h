// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <chrono>
#include <iostream>
#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {
#if !defined(DISABLE_SPARSE_TENSORS)
class SparseTensor;
#endif
}  // namespace onnxruntime

#endif

/**
   Represents both tensors and non-tensors.
*/
struct OrtValue {
 public:
  OrtValue() : data_(nullptr) {}

  ~OrtValue() {
    // if (data_) {
    //   unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    //   const onnxruntime::Tensor& t = *static_cast<onnxruntime::Tensor*>(data_.get());
    //   std::cout << milliseconds_since_epoch << ", OrtValue Destroy 1," << data_.get() << "," << data_.use_count() << "," << t.SizeInBytes() 
    //           << ", " << t.GetElementType() << "," << t.Shape() << "," << t.Location().ToString() << std::endl;
    // }
  };

  OrtValue(void* pData, onnxruntime::MLDataType type, onnxruntime::DeleteFunc deleter) {
    // if (data_) {
    //   unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    //   const onnxruntime::Tensor& t = *static_cast<onnxruntime::Tensor*>(data_.get());
    //   std::cout << milliseconds_since_epoch << ", OrtValue Destroy 2," << data_.get() << "," << data_.use_count() << "," << t.SizeInBytes() 
    //           << ", " << t.GetElementType() << "," << t.Shape() << "," << t.Location().ToString() << std::endl;
    // }
    Init(pData, type, deleter);
    // unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    // const onnxruntime::Tensor& t = *static_cast<onnxruntime::Tensor*>(data_.get());
    // std::cout << milliseconds_since_epoch << ", OrtValue Init 1," << data_.get() << "," << data_.use_count() << "," << t.SizeInBytes() 
    //           << ", " << t.GetElementType() << "," << t.Shape() << "," << t.Location().ToString() << std::endl;
  }

  void Init(void* pData, onnxruntime::MLDataType type, onnxruntime::DeleteFunc deleter) {
    // if (data_) {
    //   unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    //   const onnxruntime::Tensor& t = *static_cast<onnxruntime::Tensor*>(data_.get());
    //   std::cout << milliseconds_since_epoch << ", OrtValue Destroy 3," << data_.get() << "," << data_.use_count() << "," << t.SizeInBytes() 
    //           << ", " << t.GetElementType() << "," << t.Shape() << "," << t.Location().ToString() << std::endl;
    // }
    data_.reset(pData, deleter);
    type_ = type;
    // unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    // const onnxruntime::Tensor& t = *static_cast<onnxruntime::Tensor*>(data_.get());
    // std::cout << milliseconds_since_epoch << ", OrtValue Init 2," << data_.get() << "," << data_.use_count() << "," << t.SizeInBytes() 
    //           << ", " << t.GetElementType() << "," << t.Shape() << "," << t.Location().ToString() << std::endl;
  }

  void Init(void* pData, onnxruntime::MLDataType type, const std::function<void(void*)>& deleter) {
    // if (data_) {
    //   unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    //   const onnxruntime::Tensor& t = *static_cast<onnxruntime::Tensor*>(data_.get());
    //   std::cout << milliseconds_since_epoch << ", OrtValue Destroy 4," << data_.get() << "," << data_.use_count() << "," << t.SizeInBytes() 
    //           << ", " << t.GetElementType() << "," << t.Shape() << "," << t.Location().ToString() << std::endl;
    // }
    data_.reset(pData, deleter);
    type_ = type;
    // unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    // const onnxruntime::Tensor& t = *static_cast<onnxruntime::Tensor*>(data_.get());
    // std::cout << milliseconds_since_epoch << ", OrtValue Init 3," << data_.get() << "," << data_.use_count() << "," << t.SizeInBytes() 
    //           << ", " << t.GetElementType() << "," << t.Shape() << "," << t.Location().ToString() << std::endl;
  }

  bool IsAllocated() const {
    return data_ && type_;
  }

  template <typename T>
  const T& Get() const {
    ORT_ENFORCE(onnxruntime::DataTypeImpl::GetType<T>() == type_, onnxruntime::DataTypeImpl::GetType<T>(), " != ", type_);
    return *static_cast<T*>(data_.get());
  }

  // template <typename T>
  // OrtValue(const OrtValue& a) {
  //   const onnxruntime::Tensor& b = static_cast<const onnxruntime::Tensor&>(a.Get<T>());
  //   unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
  //   std::cout << milliseconds_since_epoch << ", OrtValue Init 4," << &b << ",-1," << b.SizeInBytes() 
  //             << ", " << b.GetElementType() << "," << b.Shape() << "," << b.Location().ToString() << std::endl;
  // }

  // template <typename T>
  // OrtValue& operator = (const OrtValue &a)
  // {
  //   const onnxruntime::Tensor& b = static_cast<const onnxruntime::Tensor&>(a.Get<T>());
  //   unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
  //   std::cout << milliseconds_since_epoch << ", OrtValue Init 5," << &b << ",-1," << b.SizeInBytes() 
  //             << ", " << b.GetElementType() << "," << b.Shape() << "," << b.Location().ToString() << std::endl;
  //   return *this;
  // } 

  // template <typename T>
  // OrtValue& operator=(OrtValue a)
  // {
  //   const onnxruntime::Tensor& b = static_cast<const onnxruntime::Tensor&>(a.Get<T>());
  //   unsigned long milliseconds_since_epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
  //   std::cout << milliseconds_since_epoch << ", OrtValue Init 6," << &b << ",-1," << b.SizeInBytes() 
  //             << ", " << b.GetElementType() << "," << b.Shape() << "," << b.Location().ToString() << std::endl;
  //   return *this;
  // } 


  template <typename T>
  T* GetMutable() {
    ORT_ENFORCE(onnxruntime::DataTypeImpl::GetType<T>() == type_, onnxruntime::DataTypeImpl::GetType<T>(), " != ", type_);
    return static_cast<T*>(data_.get());
  }

  bool IsTensor() const noexcept {
    return (type_ != nullptr && type_->IsTensorType());
  }

  bool IsTensorSequence() const noexcept {
    return (type_ != nullptr && type_->IsTensorSequenceType());
  }

  bool IsSparseTensor() const {
#if !defined(DISABLE_SPARSE_TENSORS)
    return (type_ != nullptr && type_->IsSparseTensorType());
#else
    ORT_THROW("Sparse tensor is not supported in this build.");
#endif
  }

  onnxruntime::MLDataType Type() const {
    return type_;
  }

  onnxruntime::Fence_t Fence() const {
    return fence_.get();
  }

  void SetFence(onnxruntime::FencePtr fence) {
    fence_ = fence;
  }

  void ShareFenceWith(OrtValue& v) {
    fence_ = v.fence_;
  }

 private:
  std::shared_ptr<void> data_;
  onnxruntime::MLDataType type_{nullptr};
  onnxruntime::FencePtr fence_;
};

template <>
inline const onnxruntime::Tensor& OrtValue::Get<onnxruntime::Tensor>() const {
  ORT_ENFORCE(IsTensor(), "Trying to get a Tensor, but got: ", onnxruntime::DataTypeImpl::ToString(type_));
  return *static_cast<onnxruntime::Tensor*>(data_.get());
}

template <>
inline onnxruntime::Tensor* OrtValue::GetMutable<onnxruntime::Tensor>() {
  ORT_ENFORCE(IsTensor(), "Trying to get a Tensor, but got: ", onnxruntime::DataTypeImpl::ToString(type_));
  return static_cast<onnxruntime::Tensor*>(data_.get());
}

template <>
inline const onnxruntime::TensorSeq& OrtValue::Get<onnxruntime::TensorSeq>() const {
  ORT_ENFORCE(IsTensorSequence(), "Trying to get a TensorSeq, but got: ", onnxruntime::DataTypeImpl::ToString(type_));
  return *static_cast<onnxruntime::TensorSeq*>(data_.get());
}

template <>
inline onnxruntime::TensorSeq* OrtValue::GetMutable<onnxruntime::TensorSeq>() {
  ORT_ENFORCE(IsTensorSequence(), "Trying to get a TensorSeq, but got: ", onnxruntime::DataTypeImpl::ToString(type_));
  return static_cast<onnxruntime::TensorSeq*>(data_.get());
}

#if !defined(DISABLE_SPARSE_TENSORS)
template <>
inline const onnxruntime::SparseTensor& OrtValue::Get<onnxruntime::SparseTensor>() const {
  ORT_ENFORCE(IsSparseTensor(), "Trying to get a SparseTensor, but got: ", onnxruntime::DataTypeImpl::ToString(type_));
  return *static_cast<onnxruntime::SparseTensor*>(data_.get());
}

template <>
inline onnxruntime::SparseTensor* OrtValue::GetMutable<onnxruntime::SparseTensor>() {
  ORT_ENFORCE(IsSparseTensor(), "Trying to get a SparseTensor, but got: ", onnxruntime::DataTypeImpl::ToString(type_));
  return static_cast<onnxruntime::SparseTensor*>(data_.get());
}
#endif
