#pragma once

/**
 * @file console_logger.hpp
 * @brief İnsan okur satır. İlerleme çubuğu ProgressLoggerCallback'tedir;
 * bu sınıf deney kaydı için düz metin basar.
 */

#include "nexus_train/logging/logger_base.hpp"

#include <iostream>

namespace nexus_train {

class ConsoleLogger final : public LoggerBase {
 public:
  explicit ConsoleLogger(std::ostream& out = std::cout) : out_(out) {}

  void log_step(const TrainingState& state) override {
    out_ << "step epoch=" << state.epoch << " step=" << state.global_step << " loss=" << state.current_loss << "\n";
  }

  void log_epoch(const TrainingState& state) override {
    out_ << "epoch " << state.epoch << " loss=" << state.current_loss;
    for (const auto& [name, value] : state.metrics) {
      out_ << " " << name << "=" << value;
    }
    out_ << "\n";
  }

 private:
  std::ostream& out_;
};

}  // namespace nexus_train
