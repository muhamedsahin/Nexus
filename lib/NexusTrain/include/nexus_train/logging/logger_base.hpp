#pragma once

/**
 * @file logger_base.hpp
 * @brief Adım ve epoch kayıtlarının ortak arayüzü.
 *
 * Birden fazla logger aynı anda durabilir. Yazma eğitim thread'indedir; dosya
 * logger'ları satır basar, tensör kopyalamaz.
 */

#include "nexus_train/core/training_state.hpp"

namespace nexus_train {

class LoggerBase {
 public:
  virtual void log_step(const TrainingState& state) = 0;
  virtual void log_epoch(const TrainingState& state) = 0;
  virtual ~LoggerBase() = default;
};

}  // namespace nexus_train
