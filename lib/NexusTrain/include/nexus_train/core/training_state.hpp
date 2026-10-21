#pragma once

/**
 * @file training_state.hpp
 * @brief Callback'lerin okuyup yazdığı ortak eğitim durumu.
 *
 * `should_stop` EarlyStopping tarafından kurulur; Trainer epoch sonunda (ve batch
 * sonunda) bu bayrağa bakar. `best_metric` izlenen moda göre küçülen ya da büyüyen
 * en iyi değerdir. Saat noktası ETA içindir; checkpoint onu diske yazmaz.
 */

#include <chrono>
#include <limits>
#include <string>
#include <unordered_map>

namespace nexus_train {

struct TrainingState {
  int epoch = 0;
  int global_step = 0;
  int batch_index = 0;
  int batches_per_epoch = 0;
  double current_loss = 0.0;
  double best_metric = std::numeric_limits<double>::infinity();
  std::unordered_map<std::string, double> metrics;
  bool should_stop = false;
  std::chrono::steady_clock::time_point epoch_start_time{};
};

}  // namespace nexus_train
