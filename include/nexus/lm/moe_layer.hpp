#pragma once

/**
 * @file moe_layer.hpp
 * @brief FAZ 14.C: lm otograd icin MoE FFN — nexus::moe cekirdegi iki ozel otograd dugumu olarak:
 *        (1) y = MoEFFN(x) (girdiler x, W_gate, [W_noise], up, down), (2) yonlendirici kaybi aux + z (girdiler x, W_gate).
 *        Ayni yonlendirme onbellegi ikisinde paylasilir; kayip dugumu kapi gradyanini tasimaz (cift sayim yok).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/moe/moe.hpp"

#include <cstdint>
#include <vector>

namespace nexus {
namespace lm {

/// Katman basina son ileri gecisin uzman kullanimi (metrik, 14.5).
struct MoeLayerStats {
  int layer = 0;
  std::vector<std::int64_t> routed;  // kapasite oncesi atama
  std::vector<std::int64_t> kept;    // islenen satir
  std::int64_t capacity = 0, dropped = 0, rerouted = 0;
  int idle = 0;
  float aux = 0.0f, z = 0.0f;
};

/// noise bos Variable olabilir (gurultusuz). router_loss: aux + z skaler dugumu (nullptr: eklenmez).
[[nodiscard]] autograd::Variable moe_ffn(const autograd::Variable& x, const autograd::Variable& router,
                                         const autograd::Variable& noise, const autograd::Variable& up,
                                         const autograd::Variable& down, const moe::Spec& spec, bool training,
                                         std::uint64_t seed, autograd::Variable* router_loss, MoeLayerStats* stats);

}  // namespace lm
}  // namespace nexus
