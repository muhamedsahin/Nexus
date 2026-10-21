#pragma once

/**
 * @file loss_driver.hpp
 * @brief NexusLoss çağrısını tek batch adımına kilitler.
 *
 * LossBase bir şablondur; CrossEntropyLoss ondan türemez ve sınıf indisi ister.
 * Trainer bu farkı bilmez. Sürücü, tahmini ve hedefi alıp skaler loss ile dL/dy
 * tensörünü doldurur. Gradyan matematiği NexusLoss'ta kalır.
 *
 * Sparse cross-entropy NexusLoss'ta satır başına tanımlıdır. Döngü orkestrasyondur:
 * her satır `forward` + `backward`, sonra batch ortalaması (e2e örnekle aynı ölçek).
 *
 * Thread-safety: bir sürücü nesnesi aynı anda tek thread. Veri paralel adım
 * `clone()` ile replika başına kopya alır; önbellekli loss paylaşılmaz.
 */

#include "nexus_model/core/tensor.hpp"
#include "nexusloss/losses/regression.hpp"
#include "nexusloss/losses/classification.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nexus_train {

class LossDriver {
 public:
  virtual ~LossDriver() = default;

  /// @return batch üzerinde ortalama loss. `grad` prediction ile aynı şekle gelir.
  virtual double forward_and_backward(const nexus_model::Tensor& prediction, const nexus_model::Tensor& target,
                                      nexus_model::Tensor& grad) = 0;

  [[nodiscard]] virtual std::unique_ptr<LossDriver> clone() const = 0;
};

class MseLossDriver final : public LossDriver {
 public:
  explicit MseLossDriver(nexusloss::core::ReductionType reduction = nexusloss::core::ReductionType::Mean)
      : loss_(reduction) {}

  double forward_and_backward(const nexus_model::Tensor& prediction, const nexus_model::Tensor& target,
                              nexus_model::Tensor& grad) override {
    if (prediction.numel() == 0 || prediction.numel() != target.numel()) {
      throw std::invalid_argument("NexusTrain: MSE prediction and target numel differ");
    }
    const std::span<const float> pred(prediction.data(), prediction.numel());
    const std::span<const float> y(target.data(), target.numel());
    grad.resize_like(prediction);
    return loss_.forward_and_backward(pred, y, std::span<float>(grad.data(), grad.numel()));
  }

  [[nodiscard]] std::unique_ptr<LossDriver> clone() const override { return std::make_unique<MseLossDriver>(loss_.get_reduction()); }

 private:
  nexusloss::MSELoss<float> loss_;
};

class SparseCrossEntropyDriver final : public LossDriver {
 public:
  explicit SparseCrossEntropyDriver(float label_smoothing = 0.f) : loss_(label_smoothing), smoothing_(label_smoothing) {}

  double forward_and_backward(const nexus_model::Tensor& prediction, const nexus_model::Tensor& target,
                              nexus_model::Tensor& grad) override {
    if (prediction.rank() < 2) {
      throw std::invalid_argument("NexusTrain: sparse CE expects logits [N, C]");
    }
    const std::size_t rows = prediction.size(0);
    const std::size_t classes = prediction.size(prediction.rank() - 1);
    const std::size_t stride = prediction.numel() / rows;
    if (stride != classes) {
      throw std::invalid_argument("NexusTrain: sparse CE only supports a class axis at the end with no extra packed dims");
    }
    if (target.numel() != rows) {
      throw std::invalid_argument("NexusTrain: sparse CE target must be class indices [N]");
    }
    grad.resize_like(prediction);
    grad.zero();
    double sum = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
      const std::span<const float> logits(prediction.data() + row * classes, classes);
      const auto label = static_cast<std::size_t>(target.data()[row] + 0.5f);
      if (label >= classes) {
        throw std::invalid_argument("NexusTrain: class index out of range");
      }
      sum += static_cast<double>(loss_.forward(logits, label));
      row_grad_.resize(classes);
      loss_.backward_into(row_grad_);
      float* dst = grad.data() + row * classes;
      for (std::size_t c = 0; c < classes; ++c) dst[c] = row_grad_[c];
    }
    const float inv = rows == 0 ? 0.f : 1.f / static_cast<float>(rows);
    float* g = grad.data();
    for (std::size_t i = 0; i < grad.numel(); ++i) g[i] *= inv;
    return rows == 0 ? 0.0 : sum / static_cast<double>(rows);
  }

  [[nodiscard]] std::unique_ptr<LossDriver> clone() const override {
    return std::make_unique<SparseCrossEntropyDriver>(smoothing_);
  }

 private:
  nexusloss::classification::CrossEntropyLoss<float> loss_;
  float smoothing_ = 0.f;
  std::vector<float> row_grad_;
};

[[nodiscard]] inline std::shared_ptr<LossDriver> make_mse_loss() { return std::make_shared<MseLossDriver>(); }

[[nodiscard]] inline std::shared_ptr<LossDriver> make_sparse_cross_entropy(float label_smoothing = 0.f) {
  return std::make_shared<SparseCrossEntropyDriver>(label_smoothing);
}

}  // namespace nexus_train
