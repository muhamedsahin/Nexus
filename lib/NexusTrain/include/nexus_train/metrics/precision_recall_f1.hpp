#pragma once

/**
 * @file precision_recall_f1.hpp
 * @brief Çok sınıflı precision / recall / F1. Makro boş sınıfları ortalamaya katmaz.
 *
 * `compute()` makro ya da mikro F1 döndürür. `collect` precision ve recall'u da yazar.
 */

#include "nexus_train/metrics/metric_base.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nexus_train {

enum class Average { Macro, Micro };

class PrecisionRecallF1 final : public MetricBase {
 public:
  explicit PrecisionRecallF1(Average average = Average::Macro) : average_(average) {}

  void update(const nexus_model::Tensor& prediction, const nexus_model::Tensor& target) override {
    if (prediction.rank() < 2 || prediction.numel() == 0) {
      throw std::invalid_argument("NexusTrain: F1 expects logits [N, C]");
    }
    const std::size_t rows = prediction.size(0);
    const std::size_t classes = prediction.numel() / rows;
    if (target.numel() != rows) {
      throw std::invalid_argument("NexusTrain: F1 target must be class indices [N]");
    }
    if (tp_.size() < classes) {
      tp_.resize(classes);
      fp_.resize(classes);
      fn_.resize(classes);
    }
    for (std::size_t r = 0; r < rows; ++r) {
      const std::size_t pred = argmax_stride(prediction.data() + r * classes, classes);
      const auto label = static_cast<std::size_t>(target.data()[r] + 0.5f);
      if (label >= classes) {
        throw std::invalid_argument("NexusTrain: F1 class index out of range");
      }
      if (pred == label) {
        ++tp_[label];
      } else {
        ++fp_[pred];
        ++fn_[label];
      }
    }
  }

  [[nodiscard]] double compute() const override { return f1(); }

  void reset() override {
    std::fill(tp_.begin(), tp_.end(), 0);
    std::fill(fp_.begin(), fp_.end(), 0);
    std::fill(fn_.begin(), fn_.end(), 0);
  }

  [[nodiscard]] std::string name() const override { return "f1"; }

  void collect(std::unordered_map<std::string, double>& out) const override {
    out["precision"] = precision();
    out["recall"] = recall();
    out["f1"] = f1();
  }

  [[nodiscard]] double precision() const { return reduce(/*precision=*/true); }
  [[nodiscard]] double recall() const { return reduce(/*precision=*/false); }
  [[nodiscard]] double f1() const {
    if (average_ == Average::Micro) {
      const double p = micro(true);
      const double r = micro(false);
      return harmonic(p, r);
    }
    double sum = 0.0;
    int seen = 0;
    for (std::size_t c = 0; c < tp_.size(); ++c) {
      const std::int64_t support = tp_[c] + fp_[c] + fn_[c];
      if (support == 0) {
        continue;
      }
      const double p = ratio(tp_[c], tp_[c] + fp_[c]);
      const double r = ratio(tp_[c], tp_[c] + fn_[c]);
      sum += harmonic(p, r);
      ++seen;
    }
    return seen == 0 ? 0.0 : sum / static_cast<double>(seen);
  }

 private:
  [[nodiscard]] static double ratio(std::int64_t num, std::int64_t den) {
    return den == 0 ? 0.0 : static_cast<double>(num) / static_cast<double>(den);
  }

  [[nodiscard]] static double harmonic(double p, double r) {
    return (p + r) == 0.0 ? 0.0 : 2.0 * p * r / (p + r);
  }

  [[nodiscard]] double micro(bool precision) const {
    std::int64_t tp = 0;
    std::int64_t other = 0;
    for (std::size_t c = 0; c < tp_.size(); ++c) {
      tp += tp_[c];
      other += precision ? fp_[c] : fn_[c];
    }
    return ratio(tp, tp + other);
  }

  [[nodiscard]] double reduce(bool precision) const {
    if (average_ == Average::Micro) {
      return micro(precision);
    }
    double sum = 0.0;
    int seen = 0;
    for (std::size_t c = 0; c < tp_.size(); ++c) {
      if (tp_[c] + fp_[c] + fn_[c] == 0) {
        continue;
      }
      sum += precision ? ratio(tp_[c], tp_[c] + fp_[c]) : ratio(tp_[c], tp_[c] + fn_[c]);
      ++seen;
    }
    return seen == 0 ? 0.0 : sum / static_cast<double>(seen);
  }

  Average average_ = Average::Macro;
  std::vector<std::int64_t> tp_;
  std::vector<std::int64_t> fp_;
  std::vector<std::int64_t> fn_;
};

}  // namespace nexus_train
