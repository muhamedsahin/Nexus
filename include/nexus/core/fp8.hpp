#pragma once

/**
 * @file fp8.hpp
 * @brief FP8 E4M3 (fn) ve E5M2 host turleri + gecikmeli olcekleme — FAZ 4.5.
 *
 * - E4M3 (OCP "e4m3fn"): us yanliligi 7, 3 mantis biti, sonsuz YOK, en buyuk sonlu 448,
 *   NaN = 0x7F/0xFF. Ileri gecis (agirlik/aktivasyon) icin.
 * - E5M2: us yanliligi 15, 2 mantis biti, IEEE gibi +-Inf (0x7C) ve NaN, en buyuk sonlu
 *   57344. Geri gecis (gradyan) icin.
 * Donusum round-to-nearest-even; `saturate` acikken aralik disi degerler en buyuk sonluya
 * kirpilir (Transformer Engine / __NV_SATFINITE davranisi), kapaliyken E5M2'de Inf,
 * E4M3'te NaN olur. Kodlama 128 pozitif desenin tablosundan yapilir: her desen tam
 * temsil edildigi icin yuvarlanmis deger tabloda birebir bulunur.
 *
 * `Fp8DelayedScaling`: tensör basina amax gecmisi (son N adim), olcek =
 * fp8_max / max(gecmis) / 2^margin (Transformer Engine "delayed scaling").
 *
 * Bu makinede (RTX 3070, sm_86) FP8 Tensor Core YOKTUR; GPU tarafinda yalniz emulasyon
 * (FP8'e yuvarla -> BF16 matmul) calisir. Bkz. nexus/cuda/fp8.hpp.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <vector>

namespace nexus {

enum class Fp8Format : std::uint8_t { E4M3 = 0, E5M2 = 1 };

namespace detail {

struct Fp8Spec {
  int exp_bits;
  int man_bits;
  int bias;
  bool has_inf;
};

constexpr Fp8Spec fp8_spec(Fp8Format f) noexcept {
  return f == Fp8Format::E4M3 ? Fp8Spec{4, 3, 7, false} : Fp8Spec{5, 2, 15, true};
}

/// Pozitif desenin (0..0x7F) degeri; NaN/Inf icin NaN/Inf.
inline double fp8_positive_value(std::uint8_t code, Fp8Format f) {
  const Fp8Spec s = fp8_spec(f);
  const int exp = (code >> s.man_bits) & ((1 << s.exp_bits) - 1);
  const int man = code & ((1 << s.man_bits) - 1);
  const int max_exp = (1 << s.exp_bits) - 1;
  if (f == Fp8Format::E4M3 && exp == max_exp && man == (1 << s.man_bits) - 1) {
    return std::nan("");
  }
  if (f == Fp8Format::E5M2 && exp == max_exp) {
    return man == 0 ? INFINITY : std::nan("");
  }
  if (exp == 0) {
    return std::ldexp(static_cast<double>(man), 1 - s.bias - s.man_bits);
  }
  return std::ldexp(static_cast<double>((1 << s.man_bits) | man), exp - s.bias - s.man_bits);
}

}  // namespace detail

[[nodiscard]] constexpr double fp8_max(Fp8Format f) noexcept { return f == Fp8Format::E4M3 ? 448.0 : 57344.0; }

[[nodiscard]] inline float fp8_to_float(std::uint8_t bits, Fp8Format f) {
  const double v = detail::fp8_positive_value(static_cast<std::uint8_t>(bits & 0x7F), f);
  return static_cast<float>((bits & 0x80) ? -v : v);
}

[[nodiscard]] inline std::uint8_t float_to_fp8(float value, Fp8Format f, bool saturate = true) {
  const std::uint8_t sign = std::signbit(value) ? 0x80 : 0x00;
  const std::uint8_t nan_code = f == Fp8Format::E4M3 ? 0x7F : 0x7E;
  if (std::isnan(value)) {
    return static_cast<std::uint8_t>(sign | nan_code);
  }
  const detail::Fp8Spec s = detail::fp8_spec(f);
  const std::uint8_t max_code = f == Fp8Format::E4M3 ? 0x7E : 0x7B;
  const double mag = std::fabs(static_cast<double>(value));
  if (std::isinf(value)) {
    if (saturate) {
      return static_cast<std::uint8_t>(sign | max_code);
    }
    return static_cast<std::uint8_t>(sign | (s.has_inf ? 0x7C : 0x7F));
  }
  // Kuantum: normal aralikta 2^(floor(log2 |x|) - man), alt-normalde sabit 2^(1-bias-man).
  const int min_normal_exp = 1 - s.bias;
  int e = 0;
  (void)std::frexp(mag, &e);  // mag = m * 2^e, m in [0.5, 1)
  const int unbiased = e - 1;
  const int quantum_exp = std::max(unbiased, min_normal_exp) - s.man_bits;
  const double scaled = std::ldexp(mag, -quantum_exp);
  // Host'un fesetround() ayarindan bagimsiz RNE. scaled negatif degildir; yarida
  // alt komsu cift ise yukari, tek ise asagi yuvarlanir.
  const double lower = std::floor(scaled);
  const double fraction = scaled - lower;
  const bool round_up = fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0);
  const double rounded = std::ldexp(lower + (round_up ? 1.0 : 0.0), quantum_exp);
  if (rounded > fp8_max(f)) {
    if (saturate) {
      return static_cast<std::uint8_t>(sign | max_code);
    }
    return static_cast<std::uint8_t>(sign | (s.has_inf ? 0x7C : 0x7F));
  }
  for (int code = 0; code <= max_code; ++code) {
    if (detail::fp8_positive_value(static_cast<std::uint8_t>(code), f) == rounded) {
      return static_cast<std::uint8_t>(sign | code);
    }
  }
  throw std::logic_error("Nexus fp8: yuvarlanmis deger tabloda yok");
}

/// Gecikmeli olcekleme: amax gecmisinden tensör basina olcek.
class Fp8DelayedScaling {
 public:
  explicit Fp8DelayedScaling(Fp8Format format, int history_length = 16, int margin = 0)
      : format_(format), history_length_(history_length), margin_(margin) {
    if (history_length < 1 || margin < 0) {
      throw std::invalid_argument("Nexus fp8: history_length >= 1, margin >= 0 olmali");
    }
  }

  void push_amax(float amax) {
    if (!(amax >= 0.0f) || !std::isfinite(amax)) {
      throw std::invalid_argument("Nexus fp8: amax sonlu ve negatif olmayan olmali");
    }
    history_.push_back(amax);
    while (static_cast<int>(history_.size()) > history_length_) {
      history_.pop_front();
    }
  }

  /// q = fp8(x * scale); x ~= q / scale. Gecmis bos ya da amax 0 ise 1.
  [[nodiscard]] float scale() const {
    float amax = 0.0f;
    for (const float a : history_) {
      amax = std::max(amax, a);
    }
    if (amax == 0.0f) {
      return 1.0f;
    }
    return static_cast<float>(fp8_max(format_) / static_cast<double>(amax) / std::ldexp(1.0, margin_));
  }

  [[nodiscard]] const std::deque<float>& history() const noexcept { return history_; }
  [[nodiscard]] Fp8Format format() const noexcept { return format_; }

 private:
  Fp8Format format_;
  int history_length_;
  int margin_;
  std::deque<float> history_;
};

/// MX/blok olcekleme hazirligi: her blok icin bagimsiz delayed-scale gecmisi.
/// CUDA tensor-basina emulasyonu degistirmez; sonraki blok quantization cekirdekleri
/// ayni blok indeksleme ve olcek politikasini kullanir.
class Fp8BlockScaling {
 public:
  Fp8BlockScaling(Fp8Format format, std::size_t numel, std::size_t block_size = 32, int history_length = 16,
                  int margin = 0)
      : block_size_(block_size), numel_(numel) {
    if (numel == 0 || block_size == 0) {
      throw std::invalid_argument("Nexus fp8 blok olcek: numel ve block_size pozitif olmali");
    }
    const std::size_t count = (numel + block_size - 1) / block_size;
    blocks_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) blocks_.emplace_back(format, history_length, margin);
  }
  void push_amax(std::size_t block, float amax) { blocks_.at(block).push_amax(amax); }
  [[nodiscard]] float scale(std::size_t block) const { return blocks_.at(block).scale(); }
  [[nodiscard]] std::size_t block_index(std::size_t offset) const {
    if (offset >= numel_) throw std::out_of_range("Nexus fp8 blok olcek: eleman ofseti disarida");
    return offset / block_size_;
  }
  [[nodiscard]] std::size_t block_count() const noexcept { return blocks_.size(); }
  [[nodiscard]] std::size_t block_size() const noexcept { return block_size_; }

 private:
  std::size_t block_size_;
  std::size_t numel_;
  std::vector<Fp8DelayedScaling> blocks_;
};

}  // namespace nexus
