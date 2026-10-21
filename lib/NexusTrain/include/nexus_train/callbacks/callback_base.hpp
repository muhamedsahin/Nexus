#pragma once

/**
 * @file callback_base.hpp
 * @brief Eğitim döngüsünün kanca noktaları.
 *
 * Varsayılan gövdeler boştur; kullanıcı yalnızca ihtiyacı olan anı override eder.
 * Kancalar eğitim thread'inde, batch hesapları bittikten sonra çalışır. Ağır I/O
 * kancanın içinde yapılmamalı; checkpoint kancası anlık kopyayı arka plana bırakır.
 *
 * Thread-safety: bütün `on_*` metodları eğitim thread'indedir. `TrainingState`
 * bu thread'e aittir; başka thread state'e yazmasın.
 */

#include "nexus_train/core/training_state.hpp"

namespace nexus_train {

class CallbackBase {
 public:
  virtual void on_train_start(TrainingState&) {}
  virtual void on_epoch_start(TrainingState&) {}
  virtual void on_batch_start(TrainingState&) {}
  virtual void on_batch_end(TrainingState&) {}
  virtual void on_epoch_end(TrainingState&) {}
  virtual void on_validation_end(TrainingState&) {}
  virtual void on_train_end(TrainingState&) {}
  virtual ~CallbackBase() = default;
};

}  // namespace nexus_train
