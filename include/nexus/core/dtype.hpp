#pragma once

/**
 * @file dtype.hpp
 * @brief Tek sayi bicimi tanimi (`nexus::DType`) ve tip yukseltme kurallari — FAZ 1.1.
 *
 * Yukseltme (promote) PyTorch'un `torch.promote_types` tablosunu izler:
 *  - Ayni tur -> kendisi. Bool en alttadir: Bool + X = X.
 *  - Tamsayi + tamsayi -> daha genis olan; isaretsiz U8 ile isaretli I8 -> I16
 *    olmasi gerekir, Nexus'ta I16 olmadigi icin I32 (belgelenmis tek sapma).
 *  - Tamsayi/Bool + kayan nokta -> kayan nokta turu.
 *  - F16 + BF16 -> F32 (ortak ustun en kucugu). Kayan noktalarda genis olan kazanir.
 *  - FP8 turleri (F8E4M3, F8E5M2) PyTorch'ta oldugu gibi kendileri disinda bir
 *    turle ortuk yukseltilmez: acik `TensorError` ("acik donusum gerekli").
 *
 * test_phase1_dtype tablonun tamamini (11 x 11) bu kurallara karsi dogrular.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace nexus {

enum class DType : std::uint8_t { F32, F16, BF16, F64, I32, I64, I8, U8, Bool, F8E4M3, F8E5M2 };

inline constexpr DType kAllDTypes[] = {DType::F32, DType::F16, DType::BF16, DType::F64,    DType::I32,   DType::I64,
                                       DType::I8,  DType::U8,  DType::Bool, DType::F8E4M3, DType::F8E5M2};

[[nodiscard]] constexpr std::size_t size_of(DType dtype) noexcept {
  switch (dtype) {
    case DType::F64:
    case DType::I64:
      return 8;
    case DType::F32:
    case DType::I32:
      return 4;
    case DType::F16:
    case DType::BF16:
      return 2;
    case DType::I8:
    case DType::U8:
    case DType::Bool:
    case DType::F8E4M3:
    case DType::F8E5M2:
      return 1;
  }
  return 0;
}

[[nodiscard]] constexpr bool is_floating(DType dtype) noexcept {
  return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::BF16 || dtype == DType::F64 ||
         dtype == DType::F8E4M3 || dtype == DType::F8E5M2;
}

[[nodiscard]] constexpr bool is_integral(DType dtype) noexcept {
  return dtype == DType::I32 || dtype == DType::I64 || dtype == DType::I8 || dtype == DType::U8;
}

[[nodiscard]] constexpr bool is_float8(DType dtype) noexcept {
  return dtype == DType::F8E4M3 || dtype == DType::F8E5M2;
}

[[nodiscard]] constexpr std::string_view to_string(DType dtype) noexcept {
  switch (dtype) {
    case DType::F32:
      return "float32";
    case DType::F16:
      return "float16";
    case DType::BF16:
      return "bfloat16";
    case DType::F64:
      return "float64";
    case DType::I32:
      return "int32";
    case DType::I64:
      return "int64";
    case DType::I8:
      return "int8";
    case DType::U8:
      return "uint8";
    case DType::Bool:
      return "bool";
    case DType::F8E4M3:
      return "float8_e4m3fn";
    case DType::F8E5M2:
      return "float8_e5m2";
  }
  return "unknown";
}

namespace detail {

/// Kayan nokta sirasi (genislik); F16 ve BF16 esit sira, aralarinda F32'ye gidilir.
[[nodiscard]] constexpr int float_rank(DType dtype) noexcept {
  switch (dtype) {
    case DType::F16:
    case DType::BF16:
      return 1;
    case DType::F32:
      return 2;
    case DType::F64:
      return 3;
    default:
      return 0;
  }
}

[[nodiscard]] constexpr int int_rank(DType dtype) noexcept {
  switch (dtype) {
    case DType::U8:
    case DType::I8:
      return 1;
    case DType::I32:
      return 3;
    case DType::I64:
      return 4;
    default:
      return 0;
  }
}

}  // namespace detail

/// PyTorch uyumlu tip yukseltme. FP8 kendisi disiyla karisirsa TensorError.
[[nodiscard]] inline DType promote(DType a, DType b) {
  if (a == b) {
    return a;
  }
  if (is_float8(a) || is_float8(b)) {
    throw TensorError(std::string("Nexus: ") + std::string(to_string(a)) + " ile " + std::string(to_string(b)) +
                      " ortuk yukseltilemez; FP8 icin acik donusum gerekli");
  }
  if (a == DType::Bool) {
    return b;
  }
  if (b == DType::Bool) {
    return a;
  }
  const bool fa = is_floating(a);
  const bool fb = is_floating(b);
  if (fa && !fb) {
    return a;
  }
  if (fb && !fa) {
    return b;
  }
  if (fa && fb) {
    const int ra = detail::float_rank(a);
    const int rb = detail::float_rank(b);
    if (ra == rb) {
      return DType::F32;  // F16 + BF16
    }
    return ra > rb ? a : b;
  }
  // Iki tamsayi.
  if ((a == DType::U8 && b == DType::I8) || (a == DType::I8 && b == DType::U8)) {
    return DType::I32;  // PyTorch: int16; Nexus'ta I16 yok (belgelenmis sapma)
  }
  return detail::int_rank(a) >= detail::int_rank(b) ? a : b;
}

}  // namespace nexus
