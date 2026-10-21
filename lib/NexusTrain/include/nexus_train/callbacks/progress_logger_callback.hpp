#pragma once

/**
 * @file progress_logger_callback.hpp
 * @brief Konsola epoch ilerlemesi, anlık loss ve kaba ETA yazar.
 *
 * Hesap yoluna girmez. Satır başına bir yazım; renk isteğe bağlı ANSI'dir.
 */

#include "nexus_train/callbacks/callback_base.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

namespace nexus_train {

class ProgressLoggerCallback final : public CallbackBase {
 public:
  explicit ProgressLoggerCallback(bool color = false, std::ostream& out = std::cout) : color_(color), out_(out) {}

  void on_epoch_start(TrainingState& state) override {
    state.epoch_start_time = std::chrono::steady_clock::now();
    if (color_) {
      out_ << "\033[36m";
    }
    out_ << "epoch " << state.epoch << " start\n";
    if (color_) {
      out_ << "\033[0m";
    }
  }

  void on_batch_end(TrainingState& state) override {
    if (state.batches_per_epoch <= 0) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - state.epoch_start_time).count();
    const int done = state.batch_index + 1;
    const double eta = done > 0 ? elapsed * static_cast<double>(state.batches_per_epoch - done) / static_cast<double>(done) : 0.0;
    out_ << "  step " << state.global_step << " batch " << done << "/" << state.batches_per_epoch
         << " loss " << state.current_loss << " eta_s " << eta << "\n";
  }

  void on_epoch_end(TrainingState& state) override {
    out_ << "epoch " << state.epoch << " done";
    for (const auto& [name, value] : state.metrics) {
      out_ << " " << name << "=" << value;
    }
    out_ << "\n";
  }

 private:
  bool color_ = false;
  std::ostream& out_;
};

}  // namespace nexus_train
