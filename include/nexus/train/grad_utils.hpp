#pragma once

/**
 * @file grad_utils.hpp
 * @brief FAZ 9.8: CPU egitmenlerinin (dil modeli v2 CPU arka ucu ve siniflandirma Trainer'i) ortak
 *        gradyan islemleri: global norm, sonluluk denetimi, kirpma, olcekleme.
 *
 * Kirpma katsayisi torch.nn.utils.clip_grad_norm_ ve GPU AdamW cekirdegi ile ayni:
 * c = min(1, max_norm / (norm + 1e-6)). Norm FP64'te biriktirilir. Sonlu olmayan normda
 * gradyan degistirilmez; cagiran adimi atlar.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus_model/core/parameter.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nexus {
namespace train {

struct GradClipResult {
  double norm = 0.0;   // kirpma oncesi global L2 norm
  bool finite = true;  // false: inf/NaN, gradyan degistirilmedi
};

inline void scale_gradients(const std::vector<nexus_model::Parameter*>& params, float factor) {
  for (nexus_model::Parameter* p : params) {
    float* g = p->grad.data();
    for (std::size_t i = 0; i < p->grad.numel(); ++i) {
      g[i] *= factor;
    }
  }
}

/// max_norm <= 0: yalniz norm ve sonluluk.
inline GradClipResult clip_global_norm(const std::vector<nexus_model::Parameter*>& params, double max_norm) {
  double sq = 0.0;
  for (const nexus_model::Parameter* p : params) {
    const float* g = p->grad.data();
    for (std::size_t i = 0; i < p->grad.numel(); ++i) {
      sq += static_cast<double>(g[i]) * g[i];
    }
  }
  GradClipResult r;
  r.norm = std::sqrt(sq);
  r.finite = std::isfinite(r.norm);
  if (r.finite && max_norm > 0.0) {
    const auto coef = static_cast<float>(std::min(1.0, max_norm / (r.norm + 1.0e-6)));
    if (coef < 1.0f) {
      scale_gradients(params, coef);
    }
  }
  return r;
}

}  // namespace train
}  // namespace nexus
