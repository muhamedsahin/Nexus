#pragma once

/**
 * @file metric_aggregator.hpp
 * @brief Birden fazla metriği tek update / compute / reset ile yürütür.
 */

#include "nexus_train/metrics/metric_base.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace nexus_train {

class MetricAggregator {
 public:
  void add(std::shared_ptr<MetricBase> metric) {
    if (metric) {
      metrics_.push_back(std::move(metric));
    }
  }

  void update(const nexus_model::Tensor& prediction, const nexus_model::Tensor& target) {
    for (const auto& metric : metrics_) metric->update(prediction, target);
  }

  void collect(std::unordered_map<std::string, double>& out) const {
    for (const auto& metric : metrics_) metric->collect(out);
  }

  void reset() {
    for (const auto& metric : metrics_) metric->reset();
  }

  [[nodiscard]] bool empty() const noexcept { return metrics_.empty(); }

 private:
  std::vector<std::shared_ptr<MetricBase>> metrics_;
};

}  // namespace nexus_train
