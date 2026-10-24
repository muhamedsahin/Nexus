#pragma once

/**
 * @file loss_scale_policy.hpp
 * @brief Tek dinamik kayip olcegi politikasi — Nexus FAZ 4.3.
 *
 * Varsayilanlar (promt.txt 4.3, PyTorch GradScaler ile ayni): baslangic 2^16, buyume
 * araligi 2000 temiz adim, buyume 2x, geri cekilme 0.5x. Inf/NaN gorulen adim atlanir.
 *
 * Bu durum makinesi uc yerde ayni kurallarla kullanilir:
 *  - `nexus_train::LossScaler` (NexusTrain egitmeni),
 *  - `nexus::detail::LossScaler` (nexus::Trainer, mixed_precision(true)),
 *  - CUDA egitim yolu: `nexus::cuda::loss_scale_update` ayni `LossScalePolicy`
 *    alanlarini aygitta uygular (test_fp16_loss_scale host/aygit esitligini sinar).
 *
 * Bagimliligi yoktur (nvcc ve host derleyicisi ayni basligi okur).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace nexus_train {

struct LossScalePolicy {
  double init_scale = 65536.0;
  double growth_factor = 2.0;
  double backoff_factor = 0.5;
  int growth_interval = 2000;
  double min_scale = 1.0;
  double max_scale = 65536.0 * 65536.0;

  void validate() const {
    const double float_max = static_cast<double>(std::numeric_limits<float>::max());
    const bool finite = std::isfinite(init_scale) && std::isfinite(growth_factor) && std::isfinite(backoff_factor) &&
                        std::isfinite(min_scale) && std::isfinite(max_scale);
    if (!finite || !(init_scale > 0.0) || !(growth_factor > 1.0) || !(backoff_factor > 0.0) ||
        !(backoff_factor < 1.0) || growth_interval < 1 || min_scale < 1.0 / float_max || max_scale > float_max ||
        !(min_scale <= init_scale && init_scale <= max_scale)) {
      throw std::invalid_argument("Nexus: LossScalePolicy sonlu, FP32'de temsil edilebilir ve min <= init <= max olmali");
    }
  }
};

/// Politikanin host durumu: olcek, buyume sayaci, atlanan adim ve son olcekler.
class LossScaleState {
 public:
  explicit LossScaleState(LossScalePolicy policy = {}) : policy_(policy), scale_(policy.init_scale) {
    policy_.validate();
  }

  /// Bir adimin sonucu: inf/NaN goruldu mu. true donerse adim uygulanmamalidir.
  bool update(bool found_inf) {
    steps_ += 1;
    if (found_inf) {
      scale_ = scale_ * policy_.backoff_factor < policy_.min_scale ? policy_.min_scale : scale_ * policy_.backoff_factor;
      growth_tracker_ = 0;
      skipped_ += 1;
      return true;
    }
    growth_tracker_ += 1;
    if (growth_tracker_ >= policy_.growth_interval) {
      const double grown = scale_ * policy_.growth_factor;
      scale_ = grown > policy_.max_scale ? policy_.max_scale : grown;
      growth_tracker_ = 0;
    }
    return false;
  }

  [[nodiscard]] double scale() const noexcept { return scale_; }
  [[nodiscard]] int growth_tracker() const noexcept { return growth_tracker_; }
  [[nodiscard]] std::int64_t skipped_steps() const noexcept { return skipped_; }
  [[nodiscard]] std::int64_t steps() const noexcept { return steps_; }
  [[nodiscard]] const LossScalePolicy& policy() const noexcept { return policy_; }

 private:
  LossScalePolicy policy_;
  double scale_ = 65536.0;
  int growth_tracker_ = 0;
  std::int64_t skipped_ = 0;
  std::int64_t steps_ = 0;
};

}  // namespace nexus_train
