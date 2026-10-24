#pragma once

/**
 * @file ops.hpp
 * @brief FAZ 6.A/6.B: dil modeli icin fuzyonlu autograd op'lari (CPU, FP32).
 *
 *  - `rope`: Q veya K [N, heads*D] uzerinde RoPE; geri gecis transpoz (ters aci).
 *  - `swiglu` / `geglu`: x [N, 2F] = [W1 x | W3 x] (tek birlesik GEMM ciktisi) ->
 *    act(a) * b [N, F]; ileri ve geri tek dugum.
 *  - `attention`: GQA/MQA FlashAttention (nexus::attention CPU flash); [B,H,T,T] yok.
 *  - `rms_norm_heads`: bas boyutunda RMSNorm (qk_norm).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/lm/rope.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace nexus {
namespace lm {

/// positions: satir basina konum (N). cache.ensure(max konum + 1) cagrilmis olmali.
[[nodiscard]] autograd::Variable rope(const autograd::Variable& x, int heads, int head_dim,
                                      std::shared_ptr<const std::vector<int>> positions,
                                      std::shared_ptr<const RopeCache> cache);

/// y = x W (+ b). x [N, in], w [in, out]; NexusModel AVX2 GEMM (otograd matmul naif uclu dongu).
[[nodiscard]] autograd::Variable linear(const autograd::Variable& x, const autograd::Variable& w,
                                        const autograd::Variable& b);
/// y = x W^T. w [out, in] (gomme / lm_head duzeni).
[[nodiscard]] autograd::Variable linear_t(const autograd::Variable& x, const autograd::Variable& w);

[[nodiscard]] autograd::Variable swiglu(const autograd::Variable& x);
[[nodiscard]] autograd::Variable geglu(const autograd::Variable& x);

struct AttentionSpec {
  int batch = 1;
  int time = 1;
  int heads = 1;
  int kv_heads = 1;
  int head_dim = 64;
  bool causal = true;
  int window = 0;
  float softcap = 0.0f;
  float scale = 0.0f;  // 0: 1/sqrt(D); muP: 1/D
  float dropout_p = 0.0f;
  std::uint64_t seed = 0;
  std::uint64_t offset = 0;
  std::shared_ptr<const std::vector<float>> alibi_slopes;  // [heads] ya da bos
};

/// q [B*T, H*D], k/v [B*T, Hkv*D] -> [B*T, H*D].
[[nodiscard]] autograd::Variable attention(const autograd::Variable& q, const autograd::Variable& k,
                                           const autograd::Variable& v, const AttentionSpec& spec);

/// x [N, heads*D], gamma [D].
[[nodiscard]] autograd::Variable rms_norm_heads(const autograd::Variable& x, const autograd::Variable& gamma,
                                                int heads, int head_dim, float eps);

/// ALiBi egimleri (Press ve ark.): 2^(-8 (h+1) / H); H 2'nin kuvveti degilse BLOOM araya ekleme kurali.
[[nodiscard]] std::vector<float> alibi_slopes(int heads);

}  // namespace lm
}  // namespace nexus
