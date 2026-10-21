#include "check.hpp"
#include "toy.hpp"

#include "nexus_model/core/parameter.hpp"
#include "nexus_optim/core/param_group.hpp"
#include "nexus_train/core/trainer.hpp"
#include "nexus_train/mixed_precision/loss_scaler.hpp"

#include <cmath>
#include <limits>

namespace {

class InfLoss final : public nexus_train::LossDriver {
 public:
  double forward_and_backward(const nexus_model::Tensor& prediction, const nexus_model::Tensor&,
                              nexus_model::Tensor& grad) override {
    grad.resize_like(prediction);
    grad.fill(std::numeric_limits<float>::max());
    return 1.0;
  }
  std::unique_ptr<nexus_train::LossDriver> clone() const override { return std::make_unique<InfLoss>(); }
};

}  // namespace

int test_mixed_precision() {
  int failed = 0;
  {
    nexus_train::LossScaler scaler(4.0, 2.0, 0.5, 2);
    scaler.update(false);
    CHECK(scaler.growth_tracker() == 1);
    CHECK(scaler.scale_factor() == 4.0);
    scaler.update(false);
    CHECK(scaler.scale_factor() == 8.0);
    CHECK(scaler.growth_tracker() == 0);
    scaler.update(true);
    CHECK(scaler.scale_factor() == 4.0);
    CHECK(scaler.growth_tracker() == 0);
  }
  {
    nexus_model::Parameter parameter(nexus_model::Tensor::full({4}, 1.f));
    parameter.grad.fill(10.f);
    auto group = nexus_optim::make_group<float>(parameter.data.data(), parameter.grad.data(), parameter.data.numel());
    std::vector<nexus_optim::ParamGroup<float>> groups{group};
    nexus_train::LossScaler scaler(5.0, 2.0, 0.5, 100);
    CHECK(scaler.unscale_and_check(groups));
    CHECK(std::fabs(parameter.grad[0] - 2.f) < 1e-6f);
    parameter.grad[1] = std::numeric_limits<float>::infinity();
    nexus_train::LossScaler identity(1.0, 2.0, 0.5, 100);
    CHECK(!identity.unscale_and_check(groups));
  }
  {
    RegressionToy toy(4, 2, 4, 1e-3, 8, false);
    nexus_train::TrainConfig config;
    config.num_epochs = 1;
    config.deterministic_mode = true;
    config.mixed_precision = true;
    config.loss_scale_init = 65536.0;
    config.loss_scale_backoff = 0.5;
    config.log_every_n_steps = 0;
    config.validate_every_n_epochs = 0;
    config.seed = 8;
    nexus_train::Trainer trainer(toy.model, std::make_shared<InfLoss>(), toy.opt, config);
    trainer.fit(toy.loader);
    CHECK(trainer.state().global_step == 0);
    CHECK(trainer.loss_scaler() != nullptr);
    CHECK(std::fabs(trainer.loss_scaler()->scale_factor() - 32768.0) < 1e-6);
  }
  return failed;
}
