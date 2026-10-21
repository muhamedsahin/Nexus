#pragma once

/**
 * @file shape.hpp
 * @brief En fazla 8 eksenli, yiginda duran sekil.
 *
 * Sekil nesnesi yigin uzerindedir: gorunum olusturmak `std::vector` ayirmaz.
 * Sinir, `nexus_model::Tensor` ile ayni 8 eksendir. Daha yuksek rank acik
 * hata verir; sessizce kesilmez.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace nexus {

inline constexpr std::size_t kAlignment = 64;
inline constexpr std::uint8_t kMaxRank = 8;

class Shape {
 public:
  Shape() = default;

  Shape(std::initializer_list<std::size_t> dims) { assign(dims.begin(), static_cast<std::uint8_t>(dims.size())); }

  explicit Shape(const std::vector<std::size_t>& dims) {
    assign(dims.data(), static_cast<std::uint8_t>(dims.size()));
  }

  Shape(const std::size_t* dims, std::uint8_t rank) { assign(dims, rank); }

  [[nodiscard]] std::uint8_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::size_t numel() const noexcept { return numel_; }
  [[nodiscard]] bool empty() const noexcept { return numel_ == 0; }
  [[nodiscard]] const std::size_t* dims() const noexcept { return dims_.data(); }

  [[nodiscard]] std::size_t operator[](std::size_t axis) const {
    if (axis >= rank_) {
      throw TensorError("Nexus: eksen seklin rank degerinin disinda");
    }
    return dims_[axis];
  }

  [[nodiscard]] bool operator==(const Shape& other) const noexcept {
    if (rank_ != other.rank_ || numel_ != other.numel_) {
      return false;
    }
    for (std::uint8_t axis = 0; axis < rank_; ++axis) {
      if (dims_[axis] != other.dims_[axis]) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool operator!=(const Shape& other) const noexcept { return !(*this == other); }

  /// Satir-major bitisik adimlar. Son eksen 1'dir. Rank 0 ise adimlar sifirdir.
  [[nodiscard]] std::array<std::size_t, kMaxRank> contiguous_strides() const noexcept {
    std::array<std::size_t, kMaxRank> strides{};
    if (rank_ == 0) {
      return strides;
    }
    strides[rank_ - 1] = 1;
    for (int axis = static_cast<int>(rank_) - 2; axis >= 0; --axis) {
      strides[static_cast<std::size_t>(axis)] =
          strides[static_cast<std::size_t>(axis) + 1] * dims_[static_cast<std::size_t>(axis) + 1];
    }
    return strides;
  }

 private:
  void assign(const std::size_t* dims, std::uint8_t rank) {
    if (rank > kMaxRank) {
      throw TensorError("Nexus: tensör rank degeri 8 sinirini asiyor");
    }
    std::size_t count = 1;
    if (rank == 0) {
      count = 0;
    }
    for (std::uint8_t axis = 0; axis < rank; ++axis) {
      const std::size_t dim = dims == nullptr ? 0 : dims[axis];
      if (dim != 0 && count > (static_cast<std::size_t>(-1) / dim)) {
        throw TensorError("Nexus: seklin eleman sayisi tasiyor");
      }
      count *= dim;
      dims_[axis] = dim;
    }
    for (std::uint8_t axis = rank; axis < kMaxRank; ++axis) {
      dims_[axis] = 0;
    }
    rank_ = rank;
    numel_ = count;
  }

  std::array<std::size_t, kMaxRank> dims_{};
  std::uint8_t rank_ = 0;
  std::size_t numel_ = 0;
};

/// Satir-major bitisiklik. Boyutu 1 olan eksenin adimi serbesttir (yayin).
[[nodiscard]] inline bool is_row_major_contiguous(const std::size_t* dims, const std::size_t* strides,
                                                  std::uint8_t rank) noexcept {
  std::size_t expected = 1;
  for (int axis = static_cast<int>(rank) - 1; axis >= 0; --axis) {
    const std::size_t dim = dims[static_cast<std::size_t>(axis)];
    if (dim > 1 && strides[static_cast<std::size_t>(axis)] != expected) {
      return false;
    }
    if (dim != 0 && expected > (static_cast<std::size_t>(-1) / dim)) {
      return false;
    }
    expected *= dim;
  }
  return true;
}

}  // namespace nexus
