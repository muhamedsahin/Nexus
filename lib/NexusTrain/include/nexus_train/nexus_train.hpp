#pragma once

/**
 * @file nexus_train.hpp
 * @brief Eğitim döngüsü orkestrasyonu. Hesap MatrixFlash düzenindeki
 * nexus_model::Tensor, NexusLoss, NexusOptim ve NexusData üzerindedir.
 *
 * LossBase bir şablon olduğu ve CrossEntropyLoss ondan türemediği için Trainer
 * `LossDriver` görür. Optimizer CRTP olduğu için `wrap_optimizer` tek sanal
 * `step()` üretir; bu çağrı batch başına bir kezdir.
 */

#include "nexus_train/bridge/loss_driver.hpp"
#include "nexus_train/bridge/optimizer_handle.hpp"
#include "nexus_train/callbacks/callback_base.hpp"
#include "nexus_train/callbacks/early_stopping.hpp"
#include "nexus_train/callbacks/lr_scheduler_callback.hpp"
#include "nexus_train/callbacks/model_checkpoint_callback.hpp"
#include "nexus_train/callbacks/progress_logger_callback.hpp"
#include "nexus_train/checkpoint/checkpoint_manager.hpp"
#include "nexus_train/core/loop_profile.hpp"
#include "nexus_train/core/tensor_batch.hpp"
#include "nexus_train/core/train_config.hpp"
#include "nexus_train/core/trainer.hpp"
#include "nexus_train/core/training_state.hpp"
#include "nexus_train/distributed/data_parallel_trainer.hpp"
#include "nexus_train/distributed/gradient_sync.hpp"
#include "nexus_train/logging/console_logger.hpp"
#include "nexus_train/logging/csv_logger.hpp"
#include "nexus_train/logging/json_logger.hpp"
#include "nexus_train/logging/logger_base.hpp"
#include "nexus_train/metrics/accuracy.hpp"
#include "nexus_train/metrics/confusion_matrix.hpp"
#include "nexus_train/metrics/metric_aggregator.hpp"
#include "nexus_train/metrics/metric_base.hpp"
#include "nexus_train/metrics/precision_recall_f1.hpp"
#include "nexus_train/mixed_precision/loss_scaler.hpp"
#include "nexus_train/prefetch/async_prefetcher.hpp"
