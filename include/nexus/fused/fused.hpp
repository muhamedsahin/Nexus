#pragma once

/**
 * @file fused.hpp
 * @brief Füzyonlu çekirdekler: ileri+geri tek dugumde, ara sonucu belleğe yazmadan — FAZ 3.
 *
 * FAZ 2'nin autograd motoru uzerine kurulu "CustomFunction" deseni: her füzyonlu
 * islem tek bir `Node` uretir; ileri geçis ara aktivasyonu (orneğin matmul ciktisi)
 * ayri bir tensor olarak dolastirmaz, geri geçis de girdi gradyanlarini tek geciste
 * hesaplar. Amac bellek trafigini dusurmektir; sayisal sonuc, füzyonsuz bilesimle
 * birebir aynidir (test_phase3_fusion bunu dogrular).
 *
 * Kapsam:
 *  - fused_linear(x, w, b, act)          : y = act(x @ w + b)
 *  - fused_layernorm(x, g, b, eps)       : y = (x - mean)*rstd*g + b
 *  - fused_residual_layernorm(x, r, g, b): y = layernorm(x + r)
 *  - fused_softmax(x)                    : kararli/cevrimici softmax (son boyut)
 *
 * Dikkat skorunu belleğe yazmayan tam füzyonlu dikkat (flash-attention tarzi)
 * sonraki fazdadir; buradaki fused_softmax onun yapi tasidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"

#include <cstdint>

namespace nexus {
namespace fused {

/// fused_linear icin aktivasyon secimi.
enum class Activation : std::uint8_t { Identity = 0, Relu = 1, Gelu = 2, Silu = 3 };

/// y = act(x @ w + b). x: [M,K], w: [K,N], b: [N]. Tek dugumde ileri+geri.
[[nodiscard]] autograd::Variable fused_linear(const autograd::Variable& x, const autograd::Variable& w,
                                              const autograd::Variable& b, Activation act);

/// y = (x - mean) / sqrt(var + eps) * gamma + beta. x: [M,N], gamma/beta: [N].
[[nodiscard]] autograd::Variable fused_layernorm(const autograd::Variable& x, const autograd::Variable& gamma,
                                                 const autograd::Variable& beta, float eps = 1.0e-5f);

/// y = layernorm(x + residual); artık toplamini ayri tensor olarak dolastirmaz.
[[nodiscard]] autograd::Variable fused_residual_layernorm(const autograd::Variable& x,
                                                          const autograd::Variable& residual,
                                                          const autograd::Variable& gamma,
                                                          const autograd::Variable& beta,
                                                          float eps = 1.0e-5f);

/// Kararli softmax (son boyut), füzyonlu geri. x: [M,N].
[[nodiscard]] autograd::Variable fused_softmax(const autograd::Variable& x);

}  // namespace fused
}  // namespace nexus
