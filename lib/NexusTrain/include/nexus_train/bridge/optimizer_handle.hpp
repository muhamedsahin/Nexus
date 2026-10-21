#pragma once

/**
 * @file optimizer_handle.hpp
 * @brief CRTP optimizer'ı Trainer'ın tek sanal çağrısına indirger.
 *
 * NexusOptim `step()` sanal değildir; eleman döngüsü içine girmez. Buradaki sanal
 * çağrı batch başına bir kezdir (ileri/geri yayılıma göre gürültüdür). İşaretçiler
 * Parameter depolamasına ödünçtür: optimizer, model yaşadığı sürece yaşar.
 *
 * Thread-safety: `step` / `zero_grad` / `param_groups` yalnızca eğitim thread'inden.
 * Checkpoint yazıcısı `state_dict()` kopyasını alır, canlı nesneyi tutmaz.
 */

#include "nexus_optim/core/optimizer_state.hpp"
#include "nexus_optim/core/param_group.hpp"

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nexus_train {

class OptimizerHandle {
 public:
  virtual ~OptimizerHandle() = default;
  virtual void step() = 0;
  virtual void zero_grad() = 0;
  [[nodiscard]] virtual nexus_optim::OptimizerState state_dict() const = 0;
  virtual void load_state_dict(const nexus_optim::OptimizerState& state) = 0;
  [[nodiscard]] virtual std::vector<nexus_optim::ParamGroup<float>>& param_groups() = 0;
};

template <typename Optim>
class ErasedOptimizer final : public OptimizerHandle {
 public:
  explicit ErasedOptimizer(std::shared_ptr<Optim> optimizer) : optimizer_(std::move(optimizer)) {
    using scalar = typename Optim::scalar_type;
    static_assert(std::is_same_v<scalar, float>, "NexusTrain drives FP32 parameter storage");
    if (!optimizer_) {
      throw std::invalid_argument("NexusTrain: null optimizer");
    }
  }

  void step() override { optimizer_->step(); }
  void zero_grad() override { optimizer_->zero_grad(); }
  [[nodiscard]] nexus_optim::OptimizerState state_dict() const override { return optimizer_->state_dict(); }
  void load_state_dict(const nexus_optim::OptimizerState& state) override { optimizer_->load_state_dict(state); }
  [[nodiscard]] std::vector<nexus_optim::ParamGroup<float>>& param_groups() override {
    return optimizer_->param_groups();
  }

 private:
  std::shared_ptr<Optim> optimizer_;
};

template <typename Optim>
[[nodiscard]] std::shared_ptr<OptimizerHandle> wrap_optimizer(std::shared_ptr<Optim> optimizer) {
  return std::make_shared<ErasedOptimizer<Optim>>(std::move(optimizer));
}

}  // namespace nexus_train
