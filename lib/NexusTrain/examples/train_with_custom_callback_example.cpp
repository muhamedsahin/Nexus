/**
 * @file train_with_custom_callback_example.cpp
 * @brief CallbackBase türetip döngüyü dışarıdan kesmek.
 */

#include "nexus_model/adapters/nexus_optim.hpp"
#include "nexus_model/containers/sequential.hpp"
#include "nexus_model/core/initializer.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_optim/algorithms/sgd.hpp"
#include "nexus_train/nexus_train.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/loading/dataloader.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace {

class StopAfterEpoch final : public nexus_train::CallbackBase {
 public:
  explicit StopAfterEpoch(int last_epoch) : last_epoch_(last_epoch) {}
  void on_epoch_end(nexus_train::TrainingState& state) override {
    std::cout << "custom callback epoch " << state.epoch << " loss " << state.current_loss << "\n";
    if (state.epoch >= last_epoch_) {
      state.should_stop = true;
    }
  }

 private:
  int last_epoch_ = 0;
};

}  // namespace

int main() {
  nexus_model::init::manual_seed(1);
  nexusdata::NDArray x(nexusdata::Shape{16, 4}, nexusdata::DType::Float32);
  nexusdata::NDArray y(nexusdata::Shape{16}, nexusdata::DType::Float32);
  for (int i = 0; i < 16 * 4; ++i) x.data<float>()[i] = 0.02f * static_cast<float>(i);
  for (int i = 0; i < 16; ++i) y.data<float>()[i] = 0.1f * static_cast<float>(i % 4);

  nexusdata::InMemoryDataset dataset(x, y);
  nexusdata::DataLoaderOptions options;
  options.batch_size = 4;
  options.shuffle = false;
  options.num_workers = 0;
  nexusdata::DataLoader loader(dataset, options);

  auto model = std::make_shared<nexus_model::Sequential>(std::vector<std::shared_ptr<nexus_model::Module>>{
      std::make_shared<nexus_model::Linear>(4, 1)});
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-2;
  nexus_optim::SGD<float>::Options sgd;
  sgd.lr = 1e-2;
  sgd.weight_decay = 0.0;
  auto optimizer = std::make_shared<nexus_optim::SGD<float>>(
      std::vector<nexus_optim::ParamGroup<float>>{nexus_model::as_param_group(*model, group)}, sgd);

  nexus_train::TrainConfig config;
  config.num_epochs = 20;
  config.deterministic_mode = true;
  config.log_every_n_steps = 0;
  config.validate_every_n_epochs = 0;
  nexus_train::Trainer trainer(model, nexus_train::make_mse_loss(), nexus_train::wrap_optimizer(optimizer), config);
  trainer.add_callback(std::make_shared<StopAfterEpoch>(1));
  trainer.fit(loader);
  std::cout << trainer.profile().report() << "\n";
  return trainer.state().epoch == 1 ? 0 : 1;
}
