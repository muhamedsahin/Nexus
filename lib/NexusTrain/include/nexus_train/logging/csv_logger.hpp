#pragma once

/**
 * @file csv_logger.hpp
 * @brief Uzun biçim: kind,epoch,step,name,value. Şema epoch'tan epoch'a değişmez.
 */

#include "nexus_train/logging/logger_base.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace nexus_train {

class CsvLogger final : public LoggerBase {
 public:
  explicit CsvLogger(std::string path) : out_(path, std::ios::out | std::ios::trunc) {
    if (!out_) {
      throw std::runtime_error("NexusTrain: cannot open csv log " + path);
    }
    out_ << "kind,epoch,step,name,value\n";
  }

  void log_step(const TrainingState& state) override { write("step", state); }

  void log_epoch(const TrainingState& state) override { write("epoch", state); }

 private:
  void write(const char* kind, const TrainingState& state) {
    out_ << kind << ',' << state.epoch << ',' << state.global_step << ",loss," << state.current_loss << '\n';
    for (const auto& [name, value] : state.metrics) {
      out_ << kind << ',' << state.epoch << ',' << state.global_step << ',' << name << ',' << value << '\n';
    }
    out_.flush();
  }

  std::ofstream out_;
};

}  // namespace nexus_train
