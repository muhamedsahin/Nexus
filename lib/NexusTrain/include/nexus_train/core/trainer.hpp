#pragma once

/**
 * @file trainer.hpp
 * @brief Epoch / batch orkestratörü. Hesap kütüphanelere aittir.
 *
 * Sıra (biriktirme penceresi kapanınca): forward → loss → (loss scale) →
 * backward → (unscale + 1/N + clip) → optimizer.step → zero_grad.
 * `zero_grad` pencereden önce çağrılmaz; mikro-batch gradyanları Parameter::grad
 * içinde toplanır. Epoch sonunda kalan pencere `1/gerçek_sayı` ile kapatılır.
 *
 * Kayıp değeri, mikro-batch ortalamalarının örnek ağırlıklı ortalamasıdır.
 * Gradyan ayrıca `gradient_accumulation_steps` ile bölünür; böylece efektif adım
 * büyük batch ortalamasına yaklaşır. İki şeyi birden loss sayısına bölmek,
 * raporlanan loss'u accumulation ayarları arasında kıyaslanamaz yapardı.
 */

#include "nexus_train/bridge/loss_driver.hpp"
#include "nexus_train/bridge/optimizer_handle.hpp"
#include "nexus_train/callbacks/callback_base.hpp"
#include "nexus_train/core/loop_profile.hpp"
#include "nexus_train/core/train_config.hpp"
#include "nexus_train/core/training_state.hpp"
#include "nexus_train/logging/logger_base.hpp"
#include "nexus_train/metrics/metric_aggregator.hpp"
#include "nexus_train/mixed_precision/loss_scaler.hpp"

#include "nexus_model/core/module.hpp"
#include "nexusdata/loading/dataloader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nexus_train {

class Trainer {
 public:
  Trainer(std::shared_ptr<nexus_model::Module> model, std::shared_ptr<LossDriver> loss,
          std::shared_ptr<OptimizerHandle> optimizer, TrainConfig config);

  void add_callback(std::shared_ptr<CallbackBase> callback);
  void add_metric(std::shared_ptr<MetricBase> metric);
  void add_logger(std::shared_ptr<LoggerBase> logger);

  void fit(nexusdata::DataLoader& train_loader, nexusdata::DataLoader* val_loader = nullptr);
  void resume_from_checkpoint(const std::string& path);

  [[nodiscard]] TrainingState& state() noexcept { return state_; }
  [[nodiscard]] const TrainingState& state() const noexcept { return state_; }
  [[nodiscard]] const LoopProfile& profile() const noexcept { return profile_; }
  [[nodiscard]] const LossScaler* loss_scaler() const noexcept { return loss_scaler_.get(); }

 private:
  void train_one_epoch(nexusdata::DataLoader& loader);
  void validate(nexusdata::DataLoader& loader);
  void on_batch(nexusdata::Batch batch, bool training);
  /// @return optimizer.step çağrıldıysa true (Inf/NaN atlanırsa false).
  bool flush_accumulation();
  void run_callbacks(void (CallbackBase::*event_fn)(TrainingState&));
  void publish_metrics(const std::string& prefix);
  void maybe_log_step(bool stepped);
  void log_epoch();

  std::shared_ptr<nexus_model::Module> model_;
  std::shared_ptr<LossDriver> loss_fn_;
  std::shared_ptr<OptimizerHandle> optimizer_;
  TrainConfig config_;
  TrainingState state_;
  LoopProfile profile_;
  std::vector<std::shared_ptr<CallbackBase>> callbacks_;
  MetricAggregator metrics_;
  std::vector<std::shared_ptr<LoggerBase>> loggers_;
  std::unique_ptr<LossScaler> loss_scaler_;
  nexus_model::Tensor grad_;
  int accumulation_ = 0;
  double epoch_loss_sum_ = 0.0;
  double epoch_weight_ = 0.0;
  int resume_epoch_ = 0;
  bool running_ = false;
};

}  // namespace nexus_train
