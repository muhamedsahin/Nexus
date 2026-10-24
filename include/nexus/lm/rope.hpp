#pragma once

/**
 * @file rope.hpp
 * @brief FAZ 6.2: doner konum kodu (RoPE) ve uzunluk olceklemeleri.
 *
 * Duzen HF LLaMA ile ayni ("rotate_half"): bas icinde ilk rot_dim eleman, cift (i, i + rot/2)
 * olarak doner; rot_dim < head_dim ise kalan boyutlar degismez (GPT-NeoX kismi dondurme).
 *   y1 = a (x1 cos - x2 sin),  y2 = a (x2 cos + x1 sin),  aci = konum * inv_freq[i]
 * a = 1, YaRN'da attention_factor. Geri gecis matrisin transpozudur: ayni a, ters aci.
 *
 * inv_freq (r = rot_dim, b = theta, s = factor, L0 = original_max_position):
 *   none:     1 / b^(2i/r)
 *   linear:   none / s                          (konum enterpolasyonu: pos / s)
 *   ntk:      b' = b s^(r/(r-2))                (NTK-aware, sabit)
 *   dynamic:  b' = b ((s n / L0) - (s - 1))^(r/(r-2)),  n = max(guncel uzunluk, L0)
 *   yarn:     dusuk/yuksek duzeltme boyutlari arasinda enterpolasyon <-> ekstrapolasyon
 *             rampasi; a = 0.1 ln(s) + 1 (verilmemisse)
 *   llama3:   dalga boyu > L0/low ise /s, < L0/high ise ayni, arada yumusak gecis
 * Her biri HF transformers `modeling_rope_utils` ile birebir (tests/parity/reference_lm_torch.py).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/config.hpp"

#include <vector>

namespace nexus {
namespace lm {

/// [rot_dim / 2] ters frekanslar. seq_len yalniz dinamik NTK'da kullanilir.
[[nodiscard]] std::vector<float> rope_inv_freq(int rot_dim, float theta, const RopeScaling& scaling, int max_position,
                                               int seq_len);
/// cos/sin carpani (YaRN attention_factor; digerlerinde 1).
[[nodiscard]] float rope_attention_scale(const RopeScaling& scaling);

/// cos/sin tablosu [konum][rot/2], `ensure` ile tembel buyur (dinamik NTK'da uzunluga gore yeniden kurulur).
class RopeCache {
 public:
  RopeCache() = default;
  RopeCache(int rot_dim, float theta, RopeScaling scaling, int max_position);
  /// [0, length) konumlarini hazirlar.
  void ensure(int length);
  [[nodiscard]] int rot_dim() const noexcept { return rot_; }
  [[nodiscard]] int length() const noexcept { return length_; }
  [[nodiscard]] const float* cos(int pos) const noexcept { return cos_.data() + static_cast<std::size_t>(pos) * half(); }
  [[nodiscard]] const float* sin(int pos) const noexcept { return sin_.data() + static_cast<std::size_t>(pos) * half(); }
  [[nodiscard]] const std::vector<float>& inv_freq() const noexcept { return inv_freq_; }

 private:
  [[nodiscard]] std::size_t half() const noexcept { return static_cast<std::size_t>(rot_ / 2); }
  int rot_ = 0;
  float theta_ = 10000.0f;
  RopeScaling scaling_{};
  int max_position_ = 0;
  int length_ = 0;
  int built_for_ = -1;  // dinamik NTK: tablonun kuruldugu etkin uzunluk
  std::vector<float> inv_freq_, cos_, sin_;
};

/// x [rows, heads * head_dim] yerinde dondurulur; satir r'nin konumu positions[r].
/// transpose = true: geri gecis (ters aci).
void rope_apply(float* x, int rows, int heads, int head_dim, const int* positions, const RopeCache& cache,
                bool transpose);

}  // namespace lm
}  // namespace nexus
