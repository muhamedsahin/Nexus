#include "check.hpp"

#include "nexus_model/adapters/nexus_optim.hpp"
#include "nexus_model/core/module.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_optim/algorithms/sgd.hpp"
#include "nexus_train/bridge/loss_driver.hpp"
#include "nexus_train/bridge/optimizer_handle.hpp"
#include "nexus_train/core/trainer.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/loading/dataloader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace {

class ProbeModule final : public nexus_model::Module {
 public:
  explicit ProbeModule(std::string* log) : log_(log) {}
  nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
    log_->push_back('F');
    return input;
  }
  nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override {
    log_->push_back('B');
    return grad_output;
  }

 private:
  std::string* log_;
};

class ProbeLoss final : public nexus_train::LossDriver {
 public:
  explicit ProbeLoss(std::string* log) : log_(log) {}
  double forward_and_backward(const nexus_model::Tensor& prediction, const nexus_model::Tensor&,
                              nexus_model::Tensor& grad) override {
    log_->push_back('L');
    grad.resize_like(prediction);
    grad.zero();
    return 0.25;
  }
  std::unique_ptr<nexus_train::LossDriver> clone() const override { return std::make_unique<ProbeLoss>(log_); }

 private:
  std::string* log_;
};

class ProbeOptim final : public nexus_train::OptimizerHandle {
 public:
  explicit ProbeOptim(std::string* log) : log_(log) {}
  void step() override { log_->push_back('S'); }
  void zero_grad() override { log_->push_back('Z'); }
  nexus_optim::OptimizerState state_dict() const override { return {}; }
  void load_state_dict(const nexus_optim::OptimizerState&) override {}
  std::vector<nexus_optim::ParamGroup<float>>& param_groups() override { return groups_; }

 private:
  std::string* log_;
  std::vector<nexus_optim::ParamGroup<float>> groups_;
};

struct TinyData {
  nexusdata::InMemoryDataset dataset;
  nexusdata::DataLoader loader;

  TinyData()
      : dataset(make_features(), make_labels()),
        loader(dataset, make_options()) {}

  static nexusdata::NDArray make_features() {
    nexusdata::NDArray x(nexusdata::Shape{2, 2}, nexusdata::DType::Float32);
    float* xf = x.data<float>();
    for (int i = 0; i < 4; ++i) xf[i] = 0.1f * static_cast<float>(i + 1);
    return x;
  }
  static nexusdata::NDArray make_labels() {
    nexusdata::NDArray y(nexusdata::Shape{2}, nexusdata::DType::Float32);
    y.data<float>()[0] = 0.f;
    y.data<float>()[1] = 1.f;
    return y;
  }
  static nexusdata::DataLoaderOptions make_options() {
    nexusdata::DataLoaderOptions options;
    options.batch_size = 1;
    options.shuffle = false;
    options.num_workers = 0;
    return options;
  }
};

}  // namespace

int test_trainer_basic_loop() {
  int failed = 0;
  {
    std::string log;
    auto model = std::make_shared<ProbeModule>(&log);
    auto loss = std::make_shared<ProbeLoss>(&log);
    auto opt = std::make_shared<ProbeOptim>(&log);
    nexus_train::TrainConfig config;
    config.num_epochs = 1;
    config.gradient_accumulation_steps = 1;
    config.deterministic_mode = true;
    config.log_every_n_steps = 0;
    config.validate_every_n_epochs = 0;
    nexus_train::Trainer trainer(model, loss, opt, config);
    TinyData data;
    trainer.fit(data.loader);
    CHECK(log == "FLBSZFLBSZ");
    CHECK(trainer.state().global_step == 2);
  }
  {
    std::string log;
    auto model = std::make_shared<ProbeModule>(&log);
    auto loss = std::make_shared<ProbeLoss>(&log);
    auto opt = std::make_shared<ProbeOptim>(&log);
    nexus_train::TrainConfig config;
    config.num_epochs = 1;
    config.gradient_accumulation_steps = 2;
    config.deterministic_mode = true;
    config.log_every_n_steps = 0;
    config.validate_every_n_epochs = 0;
    nexus_train::Trainer trainer(model, loss, opt, config);
    TinyData data;
    trainer.fit(data.loader);
    CHECK(log == "FLBFLBSZ");
    CHECK(trainer.state().global_step == 1);
  }
  return failed;
}
