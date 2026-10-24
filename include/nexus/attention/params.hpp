#pragma once

/**
 * @file params.hpp
 * @brief FAZ 5 dikkat parametreleri (CPU ve CUDA ortak). Sozlesme: nexus/cuda/attention.hpp.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"

#include <cstdint>

namespace nexus {

struct AttentionParams {
  int batch = 1;
  int heads = 1;
  int kv_heads = 1;
  int tq = 0;
  int tk = 0;
  int head_dim = 64;
  DType dtype = DType::BF16;
  float scale = 0.0f;  // 0: 1/sqrt(head_dim)
  bool causal = false;
  int window = 0;        // >0: kayan pencere boyu
  float softcap = 0.0f;  // >0: logit soft-cap
  const float* alibi_slopes = nullptr;  // aygit, [heads]
  float dropout_p = 0.0f;
  std::uint64_t seed = 0;
  std::uint64_t offset = 0;
  const std::int32_t* cu_seqlens_q = nullptr;  // aygit, [batch+1]
  const std::int32_t* cu_seqlens_k = nullptr;
  bool deterministic = false;  // geri gecis: atomiksiz dQ (kismi tampon + sabit sirali toplam)
  // Adimlar (eleman). D bitisik. Q ve O ayni adimlari paylasir, K ve V ayni adimlari.
  long long q_batch_stride = 0, q_row_stride = 0, q_head_stride = 0;
  long long k_batch_stride = 0, k_row_stride = 0, k_head_stride = 0;
  long long o_batch_stride = 0, o_row_stride = 0, o_head_stride = 0;

  /// Bitisik [B, T, H, D] duzeni icin adimlari doldurur.
  void set_contiguous_bthd();
  [[nodiscard]] float effective_scale() const noexcept;
  void validate() const;
};

}  // namespace nexus
