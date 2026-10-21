#pragma once

/**
 * @file data_parallel_trainer.hpp
 * @brief Aynı adımı birden fazla modül kopyasında veri parçalayarak çalıştırır.
 *
 * Her replika kendi forward/backward tamponuna sahiptir (NexusModel aynı nesnede
 * eşzamanlı forward'a izin vermez). Batch 0. eksende bölünür, dilimler ödünç
 * alınır. Join sonrası gradyanlar ortalanır, optimizer yalnızca replika 0'ı
 * günceller, parametreler diğerlerine yayınlanır.
 *
 * Thread-safety: `step` tek çağıran thread'den. İç thread'ler yalnızca kendi
 * replika ve loss kopyasına dokunur. LossDriver önbelleği paylaşılmaz (`clone`).
 * Optimizer, replika 0'ın Parameter depolamasına bağlı olmalıdır.
 */

#include "nexus_train/bridge/loss_driver.hpp"
#include "nexus_train/bridge/optimizer_handle.hpp"
#include "nexus_train/core/tensor_batch.hpp"
#include "nexus_train/distributed/gradient_sync.hpp"

#include <exception>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace nexus_train {

class DataParallelStep {
 public:
  DataParallelStep(std::vector<std::shared_ptr<nexus_model::Module>> replicas, std::shared_ptr<LossDriver> loss,
                   std::shared_ptr<OptimizerHandle> optimizer)
      : replicas_(std::move(replicas)), optimizer_(std::move(optimizer)) {
    if (replicas_.empty() || !loss || !optimizer_) {
      fail("data parallel step needs replicas, a loss and an optimizer");
    }
    drivers_.reserve(replicas_.size());
    parameters_.reserve(replicas_.size());
    for (const auto& replica : replicas_) {
      if (!replica) {
        fail("null replica");
      }
      drivers_.push_back(loss->clone());
      parameters_.push_back(replica->parameters());
    }
    broadcast_parameters(parameters_.front(), parameters_);
  }

  /// Ortalama loss. Gradyan senkronu backward bitince tek seferde yapılır.
  double step(const nexus_model::Tensor& inputs, const nexus_model::Tensor& labels) {
    if (inputs.rank() < 1 || labels.rank() < 1 || inputs.size(0) != labels.size(0)) {
      fail("data parallel batch rows must match");
    }
    for (const auto& replica : replicas_) replica->zero_grad();

    const std::size_t rows = inputs.size(0);
    const std::size_t width = replicas_.size();
    const std::size_t base = rows / width;
    const std::size_t extra = rows % width;
    std::vector<double> losses(width, 0.0);
    std::vector<std::size_t> counts(width, 0);
    std::vector<std::exception_ptr> errors(width);
    std::vector<std::thread> threads;
    threads.reserve(width);

    std::size_t cursor = 0;
    for (std::size_t replica = 0; replica < width; ++replica) {
      const std::size_t count = base + (replica < extra ? 1 : 0);
      counts[replica] = count;
      const std::size_t row = cursor;
      cursor += count;
      if (count == 0) {
        continue;
      }
      threads.emplace_back([&, replica, row, count] {
        try {
          const nexus_model::Tensor x = slice_rows(inputs, row, count);
          const nexus_model::Tensor y = slice_rows(labels, row, count);
          const nexus_model::Tensor prediction = replicas_[replica]->forward(x);
          nexus_model::Tensor grad;
          losses[replica] = drivers_[replica]->forward_and_backward(prediction, y, grad);
          replicas_[replica]->backward(grad);
        } catch (...) {
          errors[replica] = std::current_exception();
        }
      });
    }
    for (auto& thread : threads) thread.join();
    for (const auto& error : errors) {
      if (error) {
        std::rethrow_exception(error);
      }
    }

    allreduce_mean_gradients(parameters_);
    optimizer_->step();
    broadcast_parameters(parameters_.front(), parameters_);
    for (const auto& replica : replicas_) replica->zero_grad();

    double weighted = 0.0;
    std::size_t seen = 0;
    for (std::size_t replica = 0; replica < width; ++replica) {
      weighted += losses[replica] * static_cast<double>(counts[replica]);
      seen += counts[replica];
    }
    return seen == 0 ? 0.0 : weighted / static_cast<double>(seen);
  }

 private:
  std::vector<std::shared_ptr<nexus_model::Module>> replicas_;
  std::vector<std::unique_ptr<LossDriver>> drivers_;
  std::vector<std::vector<nexus_model::Parameter*>> parameters_;
  std::shared_ptr<OptimizerHandle> optimizer_;
};

}  // namespace nexus_train
