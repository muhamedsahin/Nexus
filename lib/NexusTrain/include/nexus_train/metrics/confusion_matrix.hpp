#pragma once

/**
 * @file confusion_matrix.hpp
 * @brief Satır = gerçek sınıf, sütun = tahmin. `compute` iz / toplam (accuracy).
 */

#include "nexus_train/metrics/metric_base.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nexus_train {

class ConfusionMatrix final : public MetricBase {
 public:
  void update(const nexus_model::Tensor& prediction, const nexus_model::Tensor& target) override {
    if (prediction.rank() < 2 || prediction.numel() == 0) {
      throw std::invalid_argument("NexusTrain: confusion matrix expects logits [N, C]");
    }
    const std::size_t rows = prediction.size(0);
    const std::size_t classes = prediction.numel() / rows;
    if (target.numel() != rows) {
      throw std::invalid_argument("NexusTrain: confusion matrix target must be [N]");
    }
    if (classes > 8192) {
      throw std::invalid_argument("NexusTrain: confusion matrix class count is too large");
    }
    if (counts_.size() != classes * classes) {
      counts_.assign(classes * classes, 0);
      classes_ = classes;
    }
    for (std::size_t r = 0; r < rows; ++r) {
      const std::size_t pred = argmax_stride(prediction.data() + r * classes, classes);
      const auto label = static_cast<std::size_t>(target.data()[r] + 0.5f);
      if (label >= classes) {
        throw std::invalid_argument("NexusTrain: confusion matrix class index out of range");
      }
      ++counts_[label * classes + pred];
      ++total_;
    }
  }

  [[nodiscard]] double compute() const override {
    if (total_ == 0 || classes_ == 0) {
      return 0.0;
    }
    std::int64_t trace = 0;
    for (std::size_t c = 0; c < classes_; ++c) trace += counts_[c * classes_ + c];
    return static_cast<double>(trace) / static_cast<double>(total_);
  }

  void reset() override {
    std::fill(counts_.begin(), counts_.end(), 0);
    total_ = 0;
  }

  [[nodiscard]] std::string name() const override { return "confusion_accuracy"; }

  [[nodiscard]] std::int64_t at(std::size_t truth, std::size_t pred) const {
    if (truth >= classes_ || pred >= classes_) {
      throw std::out_of_range("NexusTrain: confusion matrix index");
    }
    return counts_[truth * classes_ + pred];
  }

  [[nodiscard]] std::size_t classes() const noexcept { return classes_; }

 private:
  std::size_t classes_ = 0;
  std::int64_t total_ = 0;
  std::vector<std::int64_t> counts_;
};

}  // namespace nexus_train
