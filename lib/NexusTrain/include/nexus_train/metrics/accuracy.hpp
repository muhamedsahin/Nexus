#pragma once

/**
 * @file accuracy.hpp
 * @brief argmax(prediction) ile sınıf indisinin eşleşme oranı.
 *
 * Logit şekli [N, C], hedef [N] (sınıf indisi) ya da [N, C] (one-hot / logit).
 */

#include "nexus_train/metrics/metric_base.hpp"

#include <cstdint>
#include <stdexcept>

namespace nexus_train {

class Accuracy final : public MetricBase {
 public:
  void update(const nexus_model::Tensor& prediction, const nexus_model::Tensor& target) override {
    if (prediction.numel() == 0) {
      return;
    }
    const std::size_t rows = row_count(prediction);
    if (prediction.rank() >= 2 && target.numel() == rows) {
      const std::size_t classes = prediction.numel() / rows;
      for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t pred = argmax_stride(prediction.data() + r * classes, classes);
        const auto label = static_cast<std::size_t>(target.data()[r] + 0.5f);
        if (pred == label) {
          ++correct_;
        }
        ++total_;
      }
      return;
    }
    if (prediction.same_shape(target) && prediction.rank() >= 2) {
      const std::size_t classes = prediction.size(prediction.rank() - 1);
      if (classes == 0 || prediction.numel() % classes != 0) {
        throw std::invalid_argument("NexusTrain: accuracy class axis is empty");
      }
      const std::size_t n = prediction.numel() / classes;
      for (std::size_t r = 0; r < n; ++r) {
        const std::size_t pred = argmax_stride(prediction.data() + r * classes, classes);
        const std::size_t label = argmax_stride(target.data() + r * classes, classes);
        if (pred == label) {
          ++correct_;
        }
        ++total_;
      }
      return;
    }
    throw std::invalid_argument("NexusTrain: accuracy shape is neither [N,C] vs [N] nor matching one-hot");
  }

  [[nodiscard]] double compute() const override {
    return total_ == 0 ? 0.0 : static_cast<double>(correct_) / static_cast<double>(total_);
  }

  void reset() override {
    correct_ = 0;
    total_ = 0;
  }

  [[nodiscard]] std::string name() const override { return "accuracy"; }

 private:
  std::uint64_t correct_ = 0;
  std::uint64_t total_ = 0;
};

}  // namespace nexus_train
