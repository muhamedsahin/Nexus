#pragma once

/**
 * @file half.hpp
 * @brief Host tarafi IEEE 754 binary16 (`nexus::half`) ve bfloat16 (`nexus::bfloat16`) — FAZ 1.1.
 *
 * Donusumler bit duzeyinde tanimlidir ve donanimdan bagimsizdir:
 *  - float -> half: round-to-nearest-even, alt-normal (subnormal) dogru, tasmada
 *    +-Inf, NaN korunur (sessiz NaN, isaret biti korunur, yuk ust bitleri tasinir).
 *  - float -> bfloat16: ust 16 bit + round-to-nearest-even; NaN sessiz NaN'a
 *    sabitlenir (yuvarlama NaN'i Inf'e cevirmesin diye).
 *  - half/bfloat16 -> float: kayipsizdir.
 *
 * test_phase1_dtype tum 65536 binary16 ve bfloat16 bit desenini gidis-donus ile,
 * float -> half yolunu F16C donanim komutuna (varsa) ve NexusOptim'in
 * `nexus_optim::float16/bfloat16` turlerine karsi bit-bit dogrular.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <bit>
#include <cstdint>

namespace nexus {

/// float -> binary16 bitleri (RNE).
[[nodiscard]] constexpr std::uint16_t float_to_half_bits(float value) noexcept {
  const std::uint32_t f = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (f >> 16) & 0x8000u;
  const std::uint32_t abs = f & 0x7FFFFFFFu;
  if (abs >= 0x7F800000u) {
    if (abs > 0x7F800000u) {
      // NaN: sessiz bit + yukun ust 9 biti (sifirlanirsa yine NaN kalsin).
      return static_cast<std::uint16_t>(sign | 0x7E00u | ((abs >> 13) & 0x01FFu));
    }
    return static_cast<std::uint16_t>(sign | 0x7C00u);  // Inf
  }
  if (abs >= 0x477FF000u) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);  // 65520 ve ustu -> Inf (RNE)
  }
  if (abs < 0x38800000u) {
    // Alt-normal half (|x| < 2^-14) veya sifir: mantisi gizli bitle kaydir, RNE.
    if (abs < 0x33000000u) {
      return static_cast<std::uint16_t>(sign);  // < 2^-25 -> +-0 (tam yarisi da cift olan 0'a)
    }
    const std::uint32_t exponent = abs >> 23;
    const std::uint32_t mantissa = (abs & 0x007FFFFFu) | 0x00800000u;
    const std::uint32_t shift = 126u - exponent;  // 14..24 araliginda
    const std::uint32_t halfway = 1u << (shift - 1u);
    const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
    std::uint32_t result = mantissa >> shift;
    if (remainder > halfway || (remainder == halfway && (result & 1u) != 0u)) {
      result += 1u;
    }
    return static_cast<std::uint16_t>(sign | result);
  }
  // Normal: us yeniden yanli, 13 bit kaydirma, RNE (tasma usse dogal tasinir).
  std::uint32_t result = ((abs >> 13) - (112u << 10));
  const std::uint32_t remainder = abs & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0u)) {
    result += 1u;
  }
  return static_cast<std::uint16_t>(sign | result);
}

/// binary16 bitleri -> float (kayipsiz).
[[nodiscard]] constexpr float half_bits_to_float(std::uint16_t bits) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
  const std::uint32_t exponent = (bits >> 10) & 0x1Fu;
  std::uint32_t mantissa = bits & 0x03FFu;
  if (exponent == 0x1Fu) {
    return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
  }
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return std::bit_cast<float>(sign);
    }
    // Alt-normal: normallestir.
    std::uint32_t e = 113u;
    while ((mantissa & 0x0400u) == 0u) {
      mantissa <<= 1u;
      e -= 1u;
    }
    mantissa &= 0x03FFu;
    return std::bit_cast<float>(sign | (e << 23) | (mantissa << 13));
  }
  return std::bit_cast<float>(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

/// float -> bfloat16 bitleri (RNE, NaN sessiz).
[[nodiscard]] constexpr std::uint16_t float_to_bfloat16_bits(float value) noexcept {
  const std::uint32_t f = std::bit_cast<std::uint32_t>(value);
  if ((f & 0x7FFFFFFFu) > 0x7F800000u) {
    return static_cast<std::uint16_t>((f >> 16) | 0x0040u);
  }
  const std::uint32_t rounding = 0x7FFFu + ((f >> 16) & 1u);
  return static_cast<std::uint16_t>((f + rounding) >> 16);
}

[[nodiscard]] constexpr float bfloat16_bits_to_float(std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

/// IEEE 754 binary16 depolama turu.
struct alignas(2) half {
  std::uint16_t bits = 0;

  constexpr half() noexcept = default;
  constexpr explicit half(float value) noexcept : bits(float_to_half_bits(value)) {}
  [[nodiscard]] static constexpr half from_bits(std::uint16_t raw) noexcept {
    half h;
    h.bits = raw;
    return h;
  }
  [[nodiscard]] constexpr explicit operator float() const noexcept { return half_bits_to_float(bits); }
  [[nodiscard]] constexpr float to_float() const noexcept { return half_bits_to_float(bits); }
};

/// bfloat16 depolama turu (FP32'nin ust 16 biti).
struct alignas(2) bfloat16 {
  std::uint16_t bits = 0;

  constexpr bfloat16() noexcept = default;
  constexpr explicit bfloat16(float value) noexcept : bits(float_to_bfloat16_bits(value)) {}
  [[nodiscard]] static constexpr bfloat16 from_bits(std::uint16_t raw) noexcept {
    bfloat16 b;
    b.bits = raw;
    return b;
  }
  [[nodiscard]] constexpr explicit operator float() const noexcept { return bfloat16_bits_to_float(bits); }
  [[nodiscard]] constexpr float to_float() const noexcept { return bfloat16_bits_to_float(bits); }
};

static_assert(sizeof(half) == 2 && sizeof(bfloat16) == 2, "dar turler 2 bayt olmali");
static_assert(float_to_half_bits(1.0f) == 0x3C00u, "1.0 -> 0x3C00");
static_assert(float_to_half_bits(65504.0f) == 0x7BFFu, "half en buyuk sonlu");
static_assert(float_to_bfloat16_bits(1.0f) == 0x3F80u, "bf16 1.0 -> 0x3F80");

}  // namespace nexus
