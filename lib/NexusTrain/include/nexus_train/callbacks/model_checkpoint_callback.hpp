#pragma once

/**
 * @file model_checkpoint_callback.hpp
 * @brief Epoch sonunda model + optimizer + eğitim durumunu kaydeder.
 *
 * `save_best_only` yalnızca izlenen metrik iyileşince yazar. Aksi halde her epoch
 * `last` güncellenir, iyileşmede ayrıca `best` yazılır. Yazma CheckpointManager
 * içinde anlık kopyadan ve istenirse arka plan thread'inden yapılır.
 */

#include "nexus_train/callbacks/callback_base.hpp"
#include "nexus_train/checkpoint/checkpoint_manager.hpp"

#include <memory>
#include <string>

namespace nexus_train {

class ModelCheckpointCallback final : public CallbackBase {
 public:
  ModelCheckpointCallback(std::shared_ptr<CheckpointManager> manager, std::shared_ptr<nexus_model::Module> model,
                          std::shared_ptr<OptimizerHandle> optimizer, std::string monitor = "val_loss",
                          bool save_best_only = false, bool async_write = true)
      : manager_(std::move(manager)),
        model_(std::move(model)),
        optimizer_(std::move(optimizer)),
        monitor_(std::move(monitor)),
        save_best_only_(save_best_only),
        async_write_(async_write) {}

  void on_epoch_end(TrainingState& state) override {
    if (!manager_ || !model_ || !optimizer_) {
      return;
    }
    const auto it = state.metrics.find(monitor_);
    const bool have = it != state.metrics.end();
    const bool best = have && manager_->is_best(it->second);
    if (best) {
      state.best_metric = it->second;
      manager_->save(*model_, *optimizer_, state, manager_->best_directory(), async_write_);
    }
    if (!save_best_only_) {
      manager_->save(*model_, *optimizer_, state, manager_->last_directory(), async_write_);
    }
  }

 private:
  std::shared_ptr<CheckpointManager> manager_;
  std::shared_ptr<nexus_model::Module> model_;
  std::shared_ptr<OptimizerHandle> optimizer_;
  std::string monitor_;
  bool save_best_only_ = false;
  bool async_write_ = true;
};

}  // namespace nexus_train
