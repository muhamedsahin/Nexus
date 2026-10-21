#pragma once

/**
 * @file tensor_batch.hpp
 * @brief NexusData NDArray → nexus_model::Tensor görünümü.
 *
 * Float32 ve host ise depolama ödünç alınır (kopya yok). NDArray referans sayılır;
 * dönen Tensor sahip değildir, `Batch` yaşadığı sürece geçerlidir. Linear::forward
 * zaten girdiyi kendi tamponuna alır, bu yüzden ödünç ömür bir batch gövdesi kadardır.
 *
 * Düzen, MatrixFlash satır-major float32 ile aynıdır. NexusModel CPU sıcak yolu bu
 * depoda yürür; ayrı bir MatrixFlash tensörüne kopyalamak orkestrasyon kazancını yer.
 */

#include "nexus_model/core/tensor.hpp"
#include "nexusdata/loading/batch.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexus_train {

[[noreturn]] inline void fail(const std::string& message) {
  throw std::runtime_error("NexusTrain: " + message);
}

inline nexus_model::Tensor borrow_shape(float* data, const std::vector<std::size_t>& shape) {
  if (shape.empty() || shape.size() > static_cast<std::size_t>(nexus_model::kMaxRank)) {
    fail("tensor rank must be in [1, 8]");
  }
  switch (shape.size()) {
    case 1:
      return nexus_model::Tensor::borrow(data, {shape[0]});
    case 2:
      return nexus_model::Tensor::borrow(data, {shape[0], shape[1]});
    case 3:
      return nexus_model::Tensor::borrow(data, {shape[0], shape[1], shape[2]});
    case 4:
      return nexus_model::Tensor::borrow(data, {shape[0], shape[1], shape[2], shape[3]});
    case 5:
      return nexus_model::Tensor::borrow(data, {shape[0], shape[1], shape[2], shape[3], shape[4]});
    case 6:
      return nexus_model::Tensor::borrow(data, {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]});
    case 7:
      return nexus_model::Tensor::borrow(data, {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]});
    default:
      return nexus_model::Tensor::borrow(
          data, {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]});
  }
}

inline nexus_model::Tensor as_float_tensor(nexusdata::NDArray& array) {
  if (array.empty() || array.shape().empty()) {
    fail("batch tensor is empty");
  }
  if (!array.device().is_host()) {
    fail("training loop expects host batches (call NDArray::cpu() before fit)");
  }
  if (array.dtype() == nexusdata::DType::Float32) {
    return borrow_shape(array.data<float>(), array.shape());
  }

  nexus_model::Tensor out(array.shape());
  float* dst = out.data();
  const std::size_t n = array.numel();
  switch (array.dtype()) {
    case nexusdata::DType::Float64: {
      const double* src = array.data<double>();
      for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
      break;
    }
    case nexusdata::DType::Int32: {
      const std::int32_t* src = array.data<std::int32_t>();
      for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
      break;
    }
    case nexusdata::DType::Int64: {
      const std::int64_t* src = array.data<std::int64_t>();
      for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
      break;
    }
    case nexusdata::DType::UInt8: {
      const std::uint8_t* src = array.data<std::uint8_t>();
      for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
      break;
    }
    default:
      fail("unsupported batch dtype (want float32 host for the zero-copy path)");
  }
  return out;
}

/// Girdiler float32 host ise views, aksi halde sahip olunan kopyalar.
/// Kopya durumunda Tensor, NDArray'den bağımsız yaşar.
struct BatchTensors {
  nexus_model::Tensor inputs;
  nexus_model::Tensor labels;
};

/// Satır dilimi. Depolama ödünçtür; `tensor` yaşadığı sürece geçerlidir.
inline nexus_model::Tensor slice_rows(const nexus_model::Tensor& tensor, std::size_t row, std::size_t count) {
  if (tensor.rank() < 1 || tensor.size(0) < row + count) {
    fail("row slice is out of range");
  }
  std::vector<std::size_t> shape = tensor.shape_vec();
  shape[0] = count;
  const std::size_t stride = tensor.size(0) == 0 ? 0 : tensor.numel() / tensor.size(0);
  // Tensor::borrow float* ister. Dilim yalnızca okunur; yazmak çağıranın tamponuna gider.
  return borrow_shape(const_cast<float*>(tensor.data()) + row * stride, shape);
}

inline BatchTensors view_batch(nexusdata::Batch& batch) {
  BatchTensors view;
  view.inputs = as_float_tensor(batch.inputs);
  view.labels = as_float_tensor(batch.labels);
  return view;
}

}  // namespace nexus_train
