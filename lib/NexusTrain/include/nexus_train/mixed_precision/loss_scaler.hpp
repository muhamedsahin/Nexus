#pragma once

/**
 * @file loss_scaler.hpp
 * @brief FP16 underflow'una karşı dinamik loss ölçeği (Apex / GradScaler).
 *
 * NexusLoss skaler loss'tan otomatik türev almaz; dL/dy vektörünü açıkça verir.
 * Ölçek, `Module::backward` öncesinde bu vektöre uygulanır. Bu, skaler loss'u
 * çarpıp ters modda taşımakla aynıdır. Optimizer adımından önce parametre
 * gradyanları aynı katsayıya bölünür.
 *
 * Inf/NaN görülürse adım atlanır, ölçek `backoff_factor` ile küçülür.
 * `growth_interval` ardışık temiz adımdan sonra ölçek `growth_factor` ile büyür.
 *
 * Thread-safety: yalnızca eğitim thread'i. İç durum kilitsizdir.
 */

#include "nexus_model/core/tensor.hpp"
#include "nexus_optim/core/param_group.hpp"
#include "nexus_optim/processing/gradient_accumulation.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nexus_train {

[[nodiscard]] inline bool gradients_are_finite(const std::vector<nexus_optim::ParamGroup<float>>& groups) {
  for (const auto& group : groups) {
    for (std::size_t i = 0; i < group.grads.size(); ++i) {
      const float* grad = group.grads[i];
      if (grad == nullptr || group.devices[i] != nexus_optim::Device::CPU) {
        continue;
      }
      for (std::size_t k = 0; k < group.numels[i]; ++k) {
        if (!std::isfinite(grad[k])) {
          return false;
        }
      }
    }
  }
  return true;
}

class LossScaler {
 public:
  explicit LossScaler(double init_scale = 65536.0, double growth_factor = 2.0, double backoff_factor = 0.5,
                      int growth_interval = 2000)
      : scale_(init_scale),
        growth_factor_(growth_factor),
        backoff_factor_(backoff_factor),
        growth_interval_(growth_interval) {
    if (!(scale_ > 0.0) || !(growth_factor_ > 1.0) || !(backoff_factor_ > 0.0) || !(backoff_factor_ < 1.0) ||
        growth_interval_ < 1) {
      throw std::invalid_argument("NexusTrain: LossScaler factors are out of range");
    }
  }

  /// Skaler loss tensörünü ölçekler (API sözleşmesi). Sıcak yol gradyan vektörünü kullanır.
  [[nodiscard]] nexus_model::Tensor scale(const nexus_model::Tensor& loss) const {
    nexus_model::Tensor out = loss.clone();
    const float factor = static_cast<float>(scale_);
    float* data = out.data();
    for (std::size_t i = 0; i < out.numel(); ++i) data[i] *= factor;
    return out;
  }

  void scale_grad(nexus_model::Tensor& grad) const {
    const float factor = static_cast<float>(scale_);
    float* data = grad.data();
    for (std::size_t i = 0; i < grad.numel(); ++i) data[i] *= factor;
  }

  /**
   * Parametre gradyanlarını ölçeğe böler.
   * @return true ise Inf/NaN yok, optimizer.step() çağrılabilir.
   */
  [[nodiscard]] bool unscale_and_check(std::vector<nexus_optim::ParamGroup<float>>& groups) const {
    if (scale_ != 1.0) {
      nexus_optim::scale_gradients_(groups, 1.0 / scale_);
    }
    return gradients_are_finite(groups);
  }

  void update(bool found_inf) {
    if (found_inf) {
      scale_ *= backoff_factor_;
      if (scale_ < 1.0) {
        scale_ = 1.0;
      }
      growth_tracker_ = 0;
      return;
    }
    ++growth_tracker_;
    if (growth_tracker_ >= growth_interval_) {
      const double grown = scale_ * growth_factor_;
      scale_ = grown > kMaxScale ? kMaxScale : grown;
      growth_tracker_ = 0;
    }
  }

  [[nodiscard]] double scale_factor() const noexcept { return scale_; }
  [[nodiscard]] int growth_tracker() const noexcept { return growth_tracker_; }

 private:
  static constexpr double kMaxScale = 65536.0 * 1024.0;
  double scale_ = 65536.0;
  double growth_factor_ = 2.0;
  double backoff_factor_ = 0.5;
  int growth_interval_ = 2000;
  int growth_tracker_ = 0;
};

}  // namespace nexus_train
