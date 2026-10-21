#pragma once

/**
 * @file tensor_bridge.hpp
 * @brief MatrixFlash, NexusModel ve NexusData arasinda sifir kopyali gorunum.
 *
 * `view(...)` kaynak tamponu kopyalamaz. Donen `TensorView::data()` ile
 * kaynagin `data()` / `host_ptr()` / `device_data()` adresi ayni bayti gosterir.
 * `share` yalnizca `std::shared_ptr` sayacini artirir; float dizisi kopyalanmaz,
 * boylece kaynak NDArray yok olduktan sonra da bellek ayakta kalir.
 *
 * MatrixFlash gorunumu host aynasi doluysa host isaretcisine, degilse cihaz
 * isaretcisine bakar. `narrow` gibi cihaz gorunumleri host aynasi tasimadigi
 * icin kopru cihaz adresini secer ve ofseti ekler.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/mirrored_tensor.hpp"
#include "nexus/core/tensor.hpp"

#include "nexus_model/core/tensor.hpp"
#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/loading/batch.hpp"

#include <type_traits>
#include <vector>

#if defined(NEXUS_HAS_MATRIX_PRO)
#include "matrix_pro/core/tensor.hpp"
#endif

namespace nexus {
namespace detail {

inline Shape shape_from_rank(const std::size_t* dims, std::size_t rank) {
  if (rank > kMaxRank) {
    throw TensorError("Nexus: kopru rank degeri 8 sinirini asiyor");
  }
  return Shape(dims, static_cast<std::uint8_t>(rank));
}

inline TensorView<float> make_float_view(float* data, Shape shape, MemorySpace space, DeviceRef device) {
  const bool contiguous = true;
  return TensorView<float>(data, shape, shape.contiguous_strides(), contiguous, space, device);
}

inline TensorView<const float> make_float_view(const float* data, Shape shape, MemorySpace space, DeviceRef device) {
  return TensorView<const float>(data, shape, shape.contiguous_strides(), true, space, device);
}

}  // namespace detail

[[nodiscard]] inline Shape shape_of(const nexus_model::Tensor& tensor) {
  return detail::shape_from_rank(tensor.dims(), tensor.rank());
}

[[nodiscard]] inline TensorView<float> view(nexus_model::Tensor& tensor) {
  return detail::make_float_view(tensor.data(), shape_of(tensor), MemorySpace::Host, DeviceRef::cpu());
}

[[nodiscard]] inline TensorView<const float> view(const nexus_model::Tensor& tensor) {
  return detail::make_float_view(tensor.data(), shape_of(tensor), MemorySpace::Host, DeviceRef::cpu());
}

/**
 * @brief nexus_model::Tensor'un host deposunu kopyalamadan cihaza aynalar.
 *
 * Host tarafi modelin kendi `data()` adresidir; yalnizca cihaz tamponu
 * ayrilir. Model tensörunun yeniden boyutlanmasi adresi gecersiz kilar, bu
 * yuzden ayna o adim boyunca yasar. Durum HostAhead baslar.
 */
[[nodiscard]] inline MirroredTensor mirror(nexus_model::Tensor& tensor, DeviceRef device = DeviceRef::cuda(0)) {
  return MirroredTensor::wrap_host(tensor.data(), shape_of(tensor), device);
}

[[nodiscard]] inline DeviceRef device_of(const nexusdata::Device& device) {
  if (device.is_cuda()) {
    return DeviceRef::cuda(device.index);
  }
  if (device.is_auto()) {
    throw TensorError("Nexus: Auto cihazli NDArray somut bir adrese baglanmadan goruntulenemez");
  }
  return DeviceRef::cpu();
}

[[nodiscard]] inline MemorySpace space_of(const nexusdata::NDArray& array) {
  return array.device().is_cuda() ? MemorySpace::Device : MemorySpace::Host;
}

[[nodiscard]] inline Shape shape_of(const nexusdata::NDArray& array) {
  const nexusdata::Shape& shape = array.shape();
  return detail::shape_from_rank(shape.data(), shape.size());
}

[[nodiscard]] inline TensorView<float> view(nexusdata::NDArray& array) {
  if (array.dtype() != nexusdata::DType::Float32) {
    throw TensorError("Nexus: float gorunumu yalnizca Float32 NDArray kabul eder");
  }
  // data<float>() CUDA isaretcisinde bilerek hata verir. Ham isaretci hem
  // host hem cihaz icin ayni adresi tasir; tip kontrolunu yukarida yaptik.
  auto* data = static_cast<float*>(array.data());
  return detail::make_float_view(data, shape_of(array), space_of(array), device_of(array.device()));
}

[[nodiscard]] inline TensorView<const float> view(const nexusdata::NDArray& array) {
  if (array.dtype() != nexusdata::DType::Float32) {
    throw TensorError("Nexus: float gorunumu yalnizca Float32 NDArray kabul eder");
  }
  const auto* data = static_cast<const float*>(array.data());
  return detail::make_float_view(data, shape_of(array), space_of(array), device_of(array.device()));
}

/// Veri kopyalanmaz. NDArray'in shared_ptr'i tensörde tutulur.
[[nodiscard]] inline Tensor share(nexusdata::NDArray& array) {
  TensorView<float> seen = view(array);
  return Tensor::adopt_shared(seen.data(), seen.shape(), seen.space(), seen.device(), array.shared_storage());
}

[[nodiscard]] inline TensorView<float> view_inputs(nexusdata::Batch& batch) { return view(batch.inputs); }
[[nodiscard]] inline TensorView<const float> view_inputs(const nexusdata::Batch& batch) { return view(batch.inputs); }
[[nodiscard]] inline TensorView<float> view_labels(nexusdata::Batch& batch) { return view(batch.labels); }
[[nodiscard]] inline TensorView<const float> view_labels(const nexusdata::Batch& batch) { return view(batch.labels); }

[[nodiscard]] inline TensorView<float> view_mask(nexusdata::Batch& batch) {
  if (batch.mask.empty()) {
    throw TensorError("Nexus: batch maskesi bos");
  }
  return view(batch.mask);
}

#if defined(NEXUS_HAS_MATRIX_PRO)

namespace detail {

template <typename MatrixTensor>
auto view_matrix(MatrixTensor& tensor) {
  using Pointer = std::remove_pointer_t<decltype(tensor.host_ptr())>;
  if (tensor.rank() > kMaxRank) {
    throw TensorError("Nexus: MatrixFlash rank degeri 8 sinirini asiyor");
  }
  const std::vector<std::size_t>& raw_shape = tensor.shape();
  const std::vector<std::size_t>& raw_strides = tensor.strides();
  if (raw_strides.size() != raw_shape.size()) {
    throw TensorError("Nexus: MatrixFlash stride sayisi rank ile uyusmuyor");
  }
  Shape shape = shape_from_rank(raw_shape.data(), raw_shape.size());
  std::array<std::size_t, kMaxRank> strides{};
  for (std::size_t axis = 0; axis < raw_shape.size(); ++axis) {
    strides[axis] = raw_strides[axis];
  }
  const bool contiguous = is_row_major_contiguous(raw_shape.data(), raw_strides.data(),
                                                  static_cast<std::uint8_t>(raw_shape.size()));

  Pointer* base = nullptr;
  MemorySpace space = MemorySpace::Host;
  DeviceRef device = DeviceRef::cpu();
  if (tensor.host_materialized()) {
    base = tensor.host_ptr();
  } else if (tensor.device_data() != nullptr) {
    base = tensor.device_data();
    space = MemorySpace::Device;
    // MatrixFlash tensöru aygit indeksini saklamaz; calistigi anlik aygit
    // cogu kurulumda 0'dır. Baska aygit icin view(...).with_device kullanin.
    device = DeviceRef::cuda(0);
  } else if (tensor.size() != 0) {
    throw TensorError("Nexus: MatrixFlash tensörunde host aynasi da cihaz tamponu da yok");
  }
  Pointer* logical = base == nullptr ? nullptr : base + tensor.storage_offset();
  return TensorView<Pointer>(logical, shape, strides, contiguous, space, device);
}

}  // namespace detail

[[nodiscard]] inline TensorView<float> view(matrix_pro::Tensor& tensor) { return detail::view_matrix(tensor); }

[[nodiscard]] inline TensorView<const float> view(const matrix_pro::Tensor& tensor) {
  return detail::view_matrix(tensor);
}

#if defined(NEXUS_ENABLE_CUDA)

/**
 * @brief MatrixFlash tensörunun host aynasini ve cihaz tamponunu tek bir
 *        MirroredTensor altinda sarar. Hicbir bayt ayrilmaz veya kopyalanmaz.
 *
 * Bitisik ve host aynasi dolu tensör ister. `host_current()` false ise cihaz
 * ileridedir. Ayna uzerinden yapilan senkron MatrixFlash'in kendi
 * `host_current` bayragini guncellemez; ikisini birlikte kullanan kod
 * `mark_host_current()` / `mark_host_stale()` cagrisini kendisi yapar.
 */
[[nodiscard]] inline MirroredTensor mirror(matrix_pro::Tensor& tensor, int cuda_index = 0) {
  if (!tensor.is_contiguous() || tensor.storage_offset() != 0) {
    throw TensorError("Nexus: MatrixFlash aynasi bitisik ve ofsetsiz tensör ister");
  }
  if (!tensor.host_materialized() || tensor.device_data() == nullptr) {
    throw TensorError("Nexus: MatrixFlash aynasi hem host hem cihaz tamponu ister");
  }
  const Shape shape = detail::shape_from_rank(tensor.shape().data(), tensor.shape().size());
  const MirrorState state = tensor.host_current() ? MirrorState::Synced : MirrorState::DeviceAhead;
  return MirroredTensor::wrap(tensor.host_ptr(), tensor.device_data(), shape, DeviceRef::cuda(cuda_index), state);
}

#endif

#endif

}  // namespace nexus
