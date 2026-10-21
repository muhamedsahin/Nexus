#pragma once

/**
 * @file early_stopping.hpp
 * @brief İzlenen metrik `patience` epoch boyunca `min_delta` kadar iyileşmezse durur.
 *
 * Min modda iyileşme `current < best - min_delta`, max modda `current > best + min_delta`.
 * İlk sonlu gözlem her zaman en iyidir. Metrik haritada yoksa o epoch sayılmaz.
 */

#include "nexus_train/callbacks/callback_base.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace nexus_train {

enum class MonitorMode { Min, Max };

class EarlyStopping final : public CallbackBase {
 public:
  EarlyStopping(std::string metric, int patience, double min_delta = 0.0, MonitorMode mode = MonitorMode::Min)
      : metric_(std::move(metric)), patience_(patience), min_delta_(min_delta), mode_(mode) {
    if (patience_ < 1) {
      throw std::invalid_argument("NexusTrain: EarlyStopping patience must be >= 1");
    }
  }

  void on_epoch_end(TrainingState& state) override {
    const auto it = state.metrics.find(metric_);
    if (it == state.metrics.end() || !std::isfinite(it->second)) {
      return;
    }
    const double current = it->second;
    if (!std::isfinite(best_)) {
      best_ = current;
      bad_epochs_ = 0;
      state.best_metric = best_;
      return;
    }
    const bool improved = mode_ == MonitorMode::Min ? current < best_ - min_delta_ : current > best_ + min_delta_;
    if (improved) {
      best_ = current;
      bad_epochs_ = 0;
      state.best_metric = best_;
      return;
    }
    ++bad_epochs_;
    if (bad_epochs_ >= patience_) {
      state.should_stop = true;
    }
  }

  [[nodiscard]] int bad_epochs() const noexcept { return bad_epochs_; }
  [[nodiscard]] double best() const noexcept { return best_; }

 private:
  std::string metric_;
  int patience_ = 1;
  double min_delta_ = 0.0;
  MonitorMode mode_ = MonitorMode::Min;
  double best_ = std::numeric_limits<double>::quiet_NaN();
  int bad_epochs_ = 0;
};

}  // namespace nexus_train
