#pragma once

/**
 * @file train_config.hpp
 * @brief Eğitim döngüsünün zamanlama ve kaynak kararları.
 *
 * NexusTrain hesap yapmaz; bu yapı hangi adımın ne sıklıkla, hangi örtüşmeyle
 * çalışacağını söyler. `deterministic_mode` prefetch thread'ini kapatır ki aynı
 * seed iki çalıştırmada aynı loss eğrisini üretsin.
 */

#include <cstdint>

namespace nexus_train {

struct TrainConfig {
  int num_epochs = 10;
  /// 1 = her mikro-batch bir optimizer adımı. >1 ise gradyan birikir, `step` seyrelir.
  int gradient_accumulation_steps = 1;
  /// 0 = clipping kapalı. Matematik NexusOptim::clip_grad_norm_ içindedir.
  double grad_clip_norm = 0.0;
  bool mixed_precision = false;
  /// Dinamik loss scale başlangıcı (NVIDIA GradScaler ile aynı varsayılan).
  double loss_scale_init = 65536.0;
  double loss_scale_growth = 2.0;
  double loss_scale_backoff = 0.5;
  int loss_scale_growth_interval = 2000;
  int log_every_n_steps = 50;
  int validate_every_n_epochs = 1;
  bool shuffle_each_epoch = true;
  std::uint64_t seed = 42;
  /// Async prefetch halka derinliği. 0 veya deterministic_mode = senkron çekim.
  int prefetch_queue_depth = 2;
  bool deterministic_mode = false;
  /// Faz 6: veri-paralel replika sayısı. Tek Trainer bunu yok sayar.
  int num_worker_threads = 1;
};

}  // namespace nexus_train
