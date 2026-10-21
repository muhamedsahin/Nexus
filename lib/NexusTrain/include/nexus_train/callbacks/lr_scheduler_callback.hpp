#pragma once

/**
 * @file lr_scheduler_callback.hpp
 * @brief NexusOptim scheduler'ını doğru anda `step()`'ler.
 *
 * Epoch ve batch kipleri metriksiz `step()` çağırır. Plateau kipi doğrulama
 * sonundaki izlenen metriği geçirir (ReduceLROnPlateau bunu okur).
 * Scheduler, optimizer gruplarının learning_rate alanına işaretçi tutar;
 * optimizer scheduler'dan daha uzun yaşamalıdır.
 */

#include "nexus_optim/scheduler/lr_scheduler_base.hpp"
#include "nexus_train/callbacks/callback_base.hpp"

#include <memory>
#include <optional>
#include <string>

namespace nexus_train {

enum class SchedulerStep { Epoch, Batch, Plateau };

class LRSchedulerCallback final : public CallbackBase {
 public:
  LRSchedulerCallback(std::shared_ptr<nexus_optim::LRSchedulerBase> scheduler, SchedulerStep when,
                      std::string plateau_metric = "val_loss")
      : scheduler_(std::move(scheduler)), when_(when), plateau_metric_(std::move(plateau_metric)) {}

  void on_batch_end(TrainingState&) override {
    if (when_ == SchedulerStep::Batch && scheduler_) {
      scheduler_->step();
    }
  }

  void on_epoch_end(TrainingState&) override {
    if (when_ == SchedulerStep::Epoch && scheduler_) {
      scheduler_->step();
    }
  }

  void on_validation_end(TrainingState& state) override {
    if (when_ != SchedulerStep::Plateau || !scheduler_) {
      return;
    }
    const auto it = state.metrics.find(plateau_metric_);
    if (it == state.metrics.end()) {
      scheduler_->step(std::nullopt);
      return;
    }
    scheduler_->step(it->second);
  }

 private:
  std::shared_ptr<nexus_optim::LRSchedulerBase> scheduler_;
  SchedulerStep when_ = SchedulerStep::Epoch;
  std::string plateau_metric_;
};

}  // namespace nexus_train
