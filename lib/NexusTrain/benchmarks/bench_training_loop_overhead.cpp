/**
 * @file bench_training_loop_overhead.cpp
 * @brief Küçük bir modelde döngünün kendi süresini ve aşama dökümünü basar.
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

int main() {
  constexpr int samples = 256;
  constexpr int features = 32;
  nexus_model::init::manual_seed(1);
  nexusdata::NDArray x(nexusdata::Shape{samples, features}, nexusdata::DType::Float32);
  nexusdata::NDArray y(nexusdata::Shape{samples}, nexusdata::DType::Float32);
  for (int i = 0; i < samples * features; ++i) x.data<float>()[i] = 0.01f * static_cast<float>(i % 17);
  for (int i = 0; i < samples; ++i) y.data<float>()[i] = 0.01f * static_cast<float>(i % 5);
  nexusdata::InMemoryDataset dataset(x, y);
  nexusdata::DataLoaderOptions options;
  options.batch_size = 32;
  options.shuffle = false;
  options.num_workers = 0;
  nexusdata::DataLoader loader(dataset, options);

  auto model = std::make_shared<nexus_model::Sequential>(std::vector<std::shared_ptr<nexus_model::Module>>{
      std::make_shared<nexus_model::Linear>(features, 16),
      std::make_shared<nexus_model::Linear>(16, 1)});
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-3;
  nexus_optim::SGD<float>::Options sgd;
  sgd.lr = 1e-3;
  auto optimizer = std::make_shared<nexus_optim::SGD<float>>(
      std::vector<nexus_optim::ParamGroup<float>>{nexus_model::as_param_group(*model, group)}, sgd);

  nexus_train::TrainConfig config;
  config.num_epochs = 2;
  config.deterministic_mode = true;
  config.log_every_n_steps = 0;
  config.validate_every_n_epochs = 0;
  nexus_train::Trainer trainer(model, nexus_train::make_mse_loss(), nexus_train::wrap_optimizer(optimizer), config);
  trainer.fit(loader);
  std::cout << trainer.profile().report() << "\n";
  return 0;
}
