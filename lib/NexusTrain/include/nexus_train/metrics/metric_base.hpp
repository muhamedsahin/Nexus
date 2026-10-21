#pragma once

/**
 * @file metric_base.hpp
 * @brief Batch tahmin/hedef çiftinden biriken metrik.
 *
 * `update` sıcak yolda çağrılır, `compute` epoch sonunda. Argmax gibi indirgemeler
 * nexus_model::Tensor üzerinde tanımlı değildir (MatrixFlash ile aynı düzende
 * ham float32). İndirgeme, ardışık satır taramasıdır; formül burada yeniden
 * türetilmez, yalnızca karşılaştırma sayılır.
 */

#include "nexus_model/core/tensor.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace nexus_train {

class MetricBase {
 public:
  virtual void update(const nexus_model::Tensor& prediction, const nexus_model::Tensor& target) = 0;
  [[nodiscard]] virtual double compute() const = 0;
  virtual void reset() = 0;
  [[nodiscard]] virtual std::string name() const = 0;
  virtual void collect(std::unordered_map<std::string, double>& out) const { out[name()] = compute(); }
  virtual ~MetricBase() = default;
};

[[nodiscard]] inline std::size_t row_count(const nexus_model::Tensor& tensor) {
  return tensor.rank() == 0 ? 0 : tensor.size(0);
}

[[nodiscard]] inline std::size_t argmax_stride(const float* row, std::size_t n) {
  std::size_t best = 0;
  float max_value = row[0];
  for (std::size_t i = 1; i < n; ++i) {
    if (row[i] > max_value) {
      max_value = row[i];
      best = i;
    }
  }
  return best;
}

}  // namespace nexus_train
