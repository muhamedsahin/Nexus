#pragma once

/**
 * @file checkpoint_manager.hpp
 * @brief Model, optimizer ve eğitim durumunun anlık kopyasını diske yazar.
 *
 * `save` / `publish` eğitim thread'inde yalnızca `state_dict` klonunu alır
 * (referans değil: yazıcı çalışırken ağırlıklar değişir). Dosya I/O'su tek
 * arka plan thread'indedir. Kuyruk derinliği 2'dir; doluysa eğitim thread'i
 * bir slot boşalana kadar bekler (bellek sınırı).
 *
 * Thread-safety:
 * - `capture`, `publish`, `save`, `is_best`, `load`, `flush` eğitim thread'inden.
 * - Yazıcı thread'i yalnızca kuyruktaki kopyayı okur, canlı modeli görmez.
 * - `flush` kuyruk ve yazım bitene kadar bloklar. Yıkıcı `flush` + join yapar.
 * - `load` ile eşzamanlı `publish` tanımsızdır; önce `flush`.
 */

#include "nexus_model/core/module.hpp"
#include "nexus_optim/core/optimizer_state.hpp"
#include "nexus_train/bridge/optimizer_handle.hpp"
#include "nexus_train/core/training_state.hpp"

#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace nexus_train {

struct CheckpointSnapshot {
  nexus_model::StateDict model;
  nexus_optim::OptimizerState optimizer;
  TrainingState state;
};

class CheckpointManager {
 public:
  explicit CheckpointManager(std::string checkpoint_dir, std::string monitor_metric = "val_loss",
                             bool monitor_mode_min = true);
  ~CheckpointManager();

  CheckpointManager(const CheckpointManager&) = delete;
  CheckpointManager& operator=(const CheckpointManager&) = delete;

  [[nodiscard]] std::shared_ptr<const CheckpointSnapshot> capture(const nexus_model::Module& model,
                                                                  const OptimizerHandle& optimizer,
                                                                  const TrainingState& state) const;

  /// Anlık kopyayı `directory` altına yazar. `is_async` false ise dönmeden flush eder.
  void publish(std::shared_ptr<const CheckpointSnapshot> snapshot, const std::filesystem::path& directory, bool is_async);

  void save(const nexus_model::Module& model, const OptimizerHandle& optimizer, const TrainingState& state,
            bool is_async = true);

  void save(const nexus_model::Module& model, const OptimizerHandle& optimizer, const TrainingState& state,
            const std::filesystem::path& directory, bool is_async);

  void load(nexus_model::Module& model, OptimizerHandle& optimizer, TrainingState& state, const std::string& path);

  /// İyileşme varsa içteki en iyi değeri günceller ve true döner.
  bool is_best(double current_metric);

  void flush();

  [[nodiscard]] std::filesystem::path best_directory() const;
  [[nodiscard]] std::filesystem::path last_directory() const;
  [[nodiscard]] const std::string& monitor_metric() const noexcept { return monitor_metric_; }
  [[nodiscard]] double best_value() const noexcept { return best_value_; }

 private:
  struct Job {
    std::shared_ptr<const CheckpointSnapshot> snapshot;
    std::filesystem::path directory;
  };

  void writer_main();
  static void write_job(const Job& job);
  static void read_into(const std::filesystem::path& directory, nexus_model::Module& model, OptimizerHandle& optimizer,
                        TrainingState& state);

  std::filesystem::path root_;
  std::string monitor_metric_;
  bool monitor_mode_min_ = true;
  double best_value_ = 0.0;

  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable drained_;
  std::queue<Job> queue_;
  bool stop_ = false;
  bool busy_ = false;
  std::exception_ptr error_;
  std::thread writer_;
};

}  // namespace nexus_train
