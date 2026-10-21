#pragma once

/**
 * @file tensor_view.hpp
 * @brief Sahip olmayan, sifir kopyali tensör gorunumu.
 *
 * `TensorView` bir isaretci, bir sekil ve adimlardan ibarettir. Kopyasi
 * bayt tasimaz; yalnizca basligi kopyalar. Kaynak tensör olmeden gorunum
 * kullanilmaz. Bitisik gorunum `elements()` ile `std::span` olur ve bu span
 * da ayirma yapmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/device.hpp"
#include "nexus/core/error.hpp"
#include "nexus/core/shape.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <utility>

namespace nexus {

template <typename T>
class TensorView {
 public:
  using value_type = T;
  using pointer = T*;

  TensorView() = default;

  TensorView(pointer data, Shape shape, std::array<std::size_t, kMaxRank> strides, bool contiguous,
             MemorySpace space, DeviceRef device) noexcept
      : data_(data),
        shape_(std::move(shape)),
        strides_(strides),
        contiguous_(contiguous),
        space_(space),
        device_(device) {}

  [[nodiscard]] pointer data() const noexcept { return data_; }
  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] std::size_t numel() const noexcept { return shape_.numel(); }
  [[nodiscard]] std::uint8_t rank() const noexcept { return shape_.rank(); }
  [[nodiscard]] bool empty() const noexcept { return numel() == 0; }
  [[nodiscard]] bool contiguous() const noexcept { return contiguous_; }
  [[nodiscard]] MemorySpace space() const noexcept { return space_; }
  [[nodiscard]] DeviceRef device() const noexcept { return device_; }
  [[nodiscard]] const std::array<std::size_t, kMaxRank>& strides() const noexcept { return strides_; }

  [[nodiscard]] std::size_t size(std::size_t axis) const { return shape_[axis]; }

  /// `alignment` 2'nin kuvveti olmalidir. Bos gorunum hizali kabul edilir.
  [[nodiscard]] bool aligned_to(std::size_t alignment) const noexcept {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      return false;
    }
    if (data_ == nullptr) {
      return true;
    }
    return (reinterpret_cast<std::uintptr_t>(data_) % alignment) == 0;
  }

  /// Bitisik elemanlarin span'i. Adimli gorunumde hata verir; yanlis duzlestirme yok.
  [[nodiscard]] std::span<T> elements() const {
    if (!contiguous_) {
      throw TensorError("Nexus: elements() yalnizca bitisik gorunumde gecerlidir");
    }
    return std::span<T>(data_, numel());
  }

  [[nodiscard]] T& operator[](std::size_t linear) const {
    if (!contiguous_) {
      throw TensorError("Nexus: dogrusal indeks yalnizca bitisik gorunumde gecerlidir");
    }
    if (linear >= numel()) {
      throw TensorError("Nexus: dogrusal indeks aralik disinda");
    }
    return data_[linear];
  }

  [[nodiscard]] std::size_t offset(std::initializer_list<std::size_t> index) const {
    if (index.size() != shape_.rank()) {
      throw TensorError("Nexus: indeks sayisi rank ile uyusmuyor");
    }
    std::size_t element = 0;
    std::size_t axis = 0;
    for (std::size_t coordinate : index) {
      if (coordinate >= shape_[axis]) {
        throw TensorError("Nexus: eksen indeksi aralik disinda");
      }
      element += coordinate * strides_[axis];
      ++axis;
    }
    return element;
  }

  [[nodiscard]] T& at(std::initializer_list<std::size_t> index) const { return data_[offset(index)]; }

  /// Ayni isaretci, yeni cihaz etiketi. Bayt kopyalanmaz.
  [[nodiscard]] TensorView with_device(DeviceRef device) const noexcept {
    TensorView copy = *this;
    copy.device_ = device;
    return copy;
  }

 private:
  pointer data_ = nullptr;
  Shape shape_{};
  std::array<std::size_t, kMaxRank> strides_{};
  bool contiguous_ = true;
  MemorySpace space_ = MemorySpace::Host;
  DeviceRef device_{};
};

}  // namespace nexus
