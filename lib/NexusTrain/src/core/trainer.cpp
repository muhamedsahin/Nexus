/**
 * @file trainer.cpp
 * @brief Senkron ve prefetch'li eğitim döngüsü.
 *
 * Prefetch kapalıyken DataLoader aynı thread'de çekilir. Açıkken üretici thread
 * bir sonraki batch'i doldururken bu thread forward/backward çalıştırır.
 * `next_fn` bir şablon lambda'dır; batch başına sanal çağrı yoktur.
 */

#include "nexus_train/core/trainer.hpp"

#include "nexus_model/core/initializer.hpp"
#include "nexus_optim/processing/gradient_accumulation.hpp"
#include "nexus_optim/processing/gradient_clipping.hpp"
#include "nexus_train/checkpoint/checkpoint_manager.hpp"
#include "nexus_train/core/tensor_batch.hpp"
#include "nexus_train/prefetch/async_prefetcher.hpp"

#include <chrono>
#include <optional>
#include <utility>

namespace nexus_train {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

Trainer::Trainer(std::shared_ptr<nexus_model::Module> model, std::shared_ptr<LossDriver> loss,
                 std::shared_ptr<OptimizerHandle> optimizer, TrainConfig config)
    : model_(std::move(model)), loss_fn_(std::move(loss)), optimizer_(std::move(optimizer)), config_(config) {
  if (!model_ || !loss_fn_ || !optimizer_) {
    fail("Trainer requires a model, a loss driver and an optimizer");
  }
  if (config_.num_epochs < 0 || config_.gradient_accumulation_steps < 1) {
    fail("num_epochs must be >= 0 and gradient_accumulation_steps must be >= 1");
  }
  if (config_.mixed_precision) {
    loss_scaler_ = std::make_unique<LossScaler>(config_.loss_scale_init, config_.loss_scale_growth, config_.loss_scale_backoff,
                                                config_.loss_scale_growth_interval);
  }
}

void Trainer::add_callback(std::shared_ptr<CallbackBase> callback) {
  if (callback) {
    callbacks_.push_back(std::move(callback));
  }
}

void Trainer::add_metric(std::shared_ptr<MetricBase> metric) { metrics_.add(std::move(metric)); }

void Trainer::add_logger(std::shared_ptr<LoggerBase> logger) {
  if (logger) {
    loggers_.push_back(std::move(logger));
  }
}

void Trainer::run_callbacks(void (CallbackBase::*event_fn)(TrainingState&)) {
  const auto start = Clock::now();
  for (const auto& callback : callbacks_) {
    (callback.get()->*event_fn)(state_);
  }
  profile_.callback_s += seconds_since(start);
}

void Trainer::publish_metrics(const std::string& prefix) {
  std::unordered_map<std::string, double> values;
  metrics_.collect(values);
  for (const auto& [name, value] : values) {
    state_.metrics[prefix + name] = value;
  }
}

void Trainer::maybe_log_step(bool stepped) {
  if (!stepped || config_.log_every_n_steps <= 0) {
    return;
  }
  if (state_.global_step % config_.log_every_n_steps != 0) {
    return;
  }
  const auto start = Clock::now();
  for (const auto& logger : loggers_) logger->log_step(state_);
  profile_.callback_s += seconds_since(start);
}

void Trainer::log_epoch() {
  const auto start = Clock::now();
  for (const auto& logger : loggers_) logger->log_epoch(state_);
  profile_.callback_s += seconds_since(start);
}

bool Trainer::flush_accumulation() {
  if (accumulation_ == 0) {
    return false;
  }
  const auto start = Clock::now();
  auto& groups = optimizer_->param_groups();
  bool ok = true;
  if (loss_scaler_) {
    ok = loss_scaler_->unscale_and_check(groups);
  }
  if (ok && accumulation_ != 1) {
    nexus_optim::scale_gradients_(groups, 1.0 / static_cast<double>(accumulation_));
  }
  if (ok && config_.grad_clip_norm > 0.0) {
    nexus_optim::clip_grad_norm_(groups, config_.grad_clip_norm);
  }
  if (ok) {
    optimizer_->step();
    ++state_.global_step;
    ++profile_.steps;
  }
  optimizer_->zero_grad();
  if (loss_scaler_) {
    loss_scaler_->update(!ok);
  }
  accumulation_ = 0;
  profile_.optimizer_s += seconds_since(start);
  return ok;
}

void Trainer::on_batch(nexusdata::Batch batch, bool training) {
  if (training) {
    run_callbacks(&CallbackBase::on_batch_start);
  }
  const BatchTensors view = view_batch(batch);

  const auto forward_start = Clock::now();
  const nexus_model::Tensor prediction = model_->forward(view.inputs);
  profile_.forward_s += seconds_since(forward_start);

  const auto loss_start = Clock::now();
  const double loss = loss_fn_->forward_and_backward(prediction, view.labels, grad_);
  profile_.loss_s += seconds_since(loss_start);
  state_.current_loss = loss;
  const double rows = prediction.rank() == 0 ? 1.0 : static_cast<double>(prediction.size(0));
  epoch_loss_sum_ += loss * rows;
  epoch_weight_ += rows;

  bool stepped = false;
  if (training) {
    if (loss_scaler_) {
      loss_scaler_->scale_grad(grad_);
    }
    const auto backward_start = Clock::now();
    model_->backward(grad_);
    profile_.backward_s += seconds_since(backward_start);
    ++accumulation_;
    if (accumulation_ >= config_.gradient_accumulation_steps) {
      stepped = flush_accumulation();
    }
  }

  metrics_.update(prediction, view.labels);
  if (training) {
    run_callbacks(&CallbackBase::on_batch_end);
    maybe_log_step(stepped);
  }
  ++state_.batch_index;
  ++profile_.batches;
}

void Trainer::train_one_epoch(nexusdata::DataLoader& loader) {
  model_->train();
  state_.batch_index = 0;
  state_.batches_per_epoch = static_cast<int>(loader.size());
  state_.epoch_start_time = Clock::now();
  metrics_.reset();
  epoch_loss_sum_ = 0.0;
  epoch_weight_ = 0.0;
  accumulation_ = 0;
  run_callbacks(&CallbackBase::on_epoch_start);

  const std::uint64_t epoch_seed =
      config_.shuffle_each_epoch ? config_.seed + static_cast<std::uint64_t>(state_.epoch) : config_.seed;
  loader.reset_epoch(epoch_seed);

  auto consume = [&](auto&& next_fn) {
    while (!state_.should_stop) {
      const auto wait_start = Clock::now();
      std::optional<nexusdata::Batch> batch = next_fn();
      profile_.data_wait_s += seconds_since(wait_start);
      if (!batch) {
        break;
      }
      on_batch(std::move(*batch), true);
    }
  };

  const bool prefetch = !config_.deterministic_mode && config_.prefetch_queue_depth > 0;
  if (prefetch) {
    AsyncPrefetcher prefetcher(loader, static_cast<std::size_t>(config_.prefetch_queue_depth));
    consume([&] { return prefetcher.next(); });
  } else {
    auto cursor = loader.begin();
    const auto end = loader.end();
    consume([&]() -> std::optional<nexusdata::Batch> {
      if (cursor == end) {
        return std::nullopt;
      }
      nexusdata::Batch batch = *cursor;
      ++cursor;
      return batch;
    });
  }
  maybe_log_step(flush_accumulation());
}

void Trainer::validate(nexusdata::DataLoader& loader) {
  model_->eval();
  metrics_.reset();
  const double saved_sum = epoch_loss_sum_;
  const double saved_weight = epoch_weight_;
  epoch_loss_sum_ = 0.0;
  epoch_weight_ = 0.0;
  state_.batch_index = 0;
  state_.batches_per_epoch = static_cast<int>(loader.size());

  const std::uint64_t val_seed = config_.seed + 0x9E3779B97F4A7C15ull;
  loader.reset_epoch(val_seed);

  auto consume = [&](auto&& next_fn) {
    while (!state_.should_stop) {
      const auto wait_start = Clock::now();
      std::optional<nexusdata::Batch> batch = next_fn();
      profile_.data_wait_s += seconds_since(wait_start);
      if (!batch) {
        break;
      }
      on_batch(std::move(*batch), false);
    }
  };

  const bool prefetch = !config_.deterministic_mode && config_.prefetch_queue_depth > 0;
  if (prefetch) {
    AsyncPrefetcher prefetcher(loader, static_cast<std::size_t>(config_.prefetch_queue_depth));
    consume([&] { return prefetcher.next(); });
  } else {
    auto cursor = loader.begin();
    const auto end = loader.end();
    consume([&]() -> std::optional<nexusdata::Batch> {
      if (cursor == end) {
        return std::nullopt;
      }
      nexusdata::Batch batch = *cursor;
      ++cursor;
      return batch;
    });
  }

  state_.metrics["val_loss"] = epoch_weight_ > 0.0 ? epoch_loss_sum_ / epoch_weight_ : 0.0;
  publish_metrics("val_");
  epoch_loss_sum_ = saved_sum;
  epoch_weight_ = saved_weight;
  model_->train();
}

void Trainer::fit(nexusdata::DataLoader& train_loader, nexusdata::DataLoader* val_loader) {
  if (running_) {
    fail("Trainer::fit is not re-entrant");
  }
  running_ = true;
  struct Reset {
    bool& flag;
    ~Reset() { flag = false; }
  } reset{running_};

  const auto wall = Clock::now();
  nexus_model::init::manual_seed(config_.seed);
  run_callbacks(&CallbackBase::on_train_start);

  for (int epoch = resume_epoch_; epoch < config_.num_epochs; ++epoch) {
    state_.epoch = epoch;
    train_one_epoch(train_loader);
    publish_metrics("");
    state_.metrics["loss"] = epoch_weight_ > 0.0 ? epoch_loss_sum_ / epoch_weight_ : state_.current_loss;

    if (val_loader != nullptr && !state_.should_stop && config_.validate_every_n_epochs > 0 &&
        epoch % config_.validate_every_n_epochs == 0) {
      validate(*val_loader);
      run_callbacks(&CallbackBase::on_validation_end);
    }
    log_epoch();
    run_callbacks(&CallbackBase::on_epoch_end);
    resume_epoch_ = epoch + 1;
    if (state_.should_stop) {
      break;
    }
  }

  run_callbacks(&CallbackBase::on_train_end);
  profile_.wall_s += seconds_since(wall);
}

void Trainer::resume_from_checkpoint(const std::string& path) {
  if (running_) {
    fail("cannot resume while fit() is running");
  }
  CheckpointManager manager(path);
  manager.load(*model_, *optimizer_, state_, path);
  resume_epoch_ = state_.epoch + 1;
  state_.should_stop = false;
}

}  // namespace nexus_train
