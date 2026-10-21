#include "check.hpp"
#include "toy.hpp"

#include "nexus_model/adapters/nexus_optim.hpp"
#include "nexus_train/core/trainer.hpp"

#include <cmath>
#include <vector>

namespace {

class FreezeOptimizer final : public nexus_train::OptimizerHandle {
 public:
  explicit FreezeOptimizer(nexus_model::Module& model) : model_(&model) {
    groups_.push_back(nexus_model::as_param_group(model));
  }

  void step() override {
    std::vector<float> flat;
    for (nexus_model::Parameter* param : model_->parameters()) {
      flat.insert(flat.end(), param->grad.data(), param->grad.data() + param->grad.numel());
    }
    snapshots_.push_back(std::move(flat));
  }

  void zero_grad() override { model_->zero_grad(); }
  nexus_optim::OptimizerState state_dict() const override { return {}; }
  void load_state_dict(const nexus_optim::OptimizerState&) override {}
  std::vector<nexus_optim::ParamGroup<float>>& param_groups() override { return groups_; }

  const std::vector<std::vector<float>>& snapshots() const { return snapshots_; }

 private:
  nexus_model::Module* model_;
  std::vector<nexus_optim::ParamGroup<float>> groups_;
  std::vector<std::vector<float>> snapshots_;
};

}  // namespace

int test_gradient_accumulation() {
  int failed = 0;
  auto model_sum = RegressionToy::make_model(4, 11);
  auto model_avg = RegressionToy::make_model(4, 99);
  model_avg->load_state_dict(model_sum->state_dict());

  const auto features = RegressionToy::make_features(16, 4);
  const auto labels = RegressionToy::make_labels(16);
  nexusdata::InMemoryDataset data_sum(features, labels);
  nexusdata::InMemoryDataset data_avg(features, labels);
  auto options = RegressionToy::make_options(4, 11, false);
  nexusdata::DataLoader loader_sum(data_sum, options);
  nexusdata::DataLoader loader_avg(data_avg, options);

  auto freeze_sum = std::make_shared<FreezeOptimizer>(*model_sum);
  auto freeze_avg = std::make_shared<FreezeOptimizer>(*model_avg);
  auto loss_sum = nexus_train::make_mse_loss();
  auto loss_avg = nexus_train::make_mse_loss();

  nexus_train::TrainConfig accumulated;
  accumulated.num_epochs = 1;
  accumulated.gradient_accumulation_steps = 4;
  accumulated.deterministic_mode = true;
  accumulated.log_every_n_steps = 0;
  accumulated.validate_every_n_epochs = 0;
  accumulated.seed = 11;

  nexus_train::TrainConfig single = accumulated;
  single.gradient_accumulation_steps = 1;

  nexus_train::Trainer trainer_sum(model_sum, loss_sum, freeze_sum, accumulated);
  trainer_sum.fit(loader_sum);
  nexus_train::Trainer trainer_avg(model_avg, loss_avg, freeze_avg, single);
  trainer_avg.fit(loader_avg);

  CHECK(freeze_sum->snapshots().size() == 1);
  CHECK(freeze_avg->snapshots().size() == 4);
  const auto& once = freeze_sum->snapshots().front();
  CHECK(once.size() == freeze_avg->snapshots().front().size());
  for (std::size_t i = 0; i < once.size(); ++i) {
    double mean = 0.0;
    for (const auto& snap : freeze_avg->snapshots()) mean += static_cast<double>(snap[i]);
    mean /= static_cast<double>(freeze_avg->snapshots().size());
    CHECK(std::fabs(static_cast<double>(once[i]) - mean) < 1e-4);
  }
  return failed;
}
