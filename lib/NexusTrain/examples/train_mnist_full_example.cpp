/**
 * @file train_mnist_full_example.cpp
 * @brief Uçtan uca orkestrasyon. MNIST baytları NexusData'nın işidir; burada
 * aynı Trainer yolu sentetik sınıf etiketleriyle çalışır (dosya olmadan da biter).
 */

#include "nexus_model/adapters/nexus_optim.hpp"
#include "nexus_model/activations/activations.hpp"
#include "nexus_model/containers/sequential.hpp"
#include "nexus_model/core/initializer.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_optim/algorithms/adamw.hpp"
#include "nexus_train/nexus_train.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/loading/dataloader.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main() {
  constexpr int samples = 96;
  constexpr int features = 8;
  constexpr int classes = 3;

  nexus_model::init::manual_seed(7);
  nexusdata::NDArray x(nexusdata::Shape{samples, features}, nexusdata::DType::Float32);
  nexusdata::NDArray y(nexusdata::Shape{samples}, nexusdata::DType::Float32);
  float* xf = x.data<float>();
  float* yf = y.data<float>();
  for (int n = 0; n < samples; ++n) {
    const int label = n % classes;
    yf[n] = static_cast<float>(label);
    for (int f = 0; f < features; ++f) {
      xf[n * features + f] = (f % classes == label) ? 1.f : 0.05f * static_cast<float>((n + f) % 5);
    }
  }

  nexusdata::InMemoryDataset dataset(x, y);
  nexusdata::DataLoaderOptions train_options;
  train_options.batch_size = 16;
  train_options.shuffle = true;
  train_options.seed = 7;
  train_options.num_workers = 0;
  nexusdata::DataLoader train_loader(dataset, train_options);

  nexusdata::DataLoaderOptions val_options = train_options;
  val_options.shuffle = false;
  nexusdata::DataLoader val_loader(dataset, val_options);

  auto model = std::make_shared<nexus_model::Sequential>(std::vector<std::shared_ptr<nexus_model::Module>>{
      std::make_shared<nexus_model::Linear>(features, 16),
      std::make_shared<nexus_model::ReLU>(),
      std::make_shared<nexus_model::Linear>(16, classes),
  });

  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-2;
  nexus_optim::AdamW<float>::Options adam;
  adam.lr = 1e-2;
  adam.weight_decay = 1e-4;
  auto optimizer = std::make_shared<nexus_optim::AdamW<float>>(
      std::vector<nexus_optim::ParamGroup<float>>{nexus_model::as_param_group(*model, group)}, adam);
  auto handle = nexus_train::wrap_optimizer(optimizer);

  nexus_train::TrainConfig config;
  config.num_epochs = 8;
  config.gradient_accumulation_steps = 2;
  config.grad_clip_norm = 1.0;
  config.mixed_precision = true;
  config.seed = 7;
  config.deterministic_mode = true;
  config.log_every_n_steps = 1;
  config.prefetch_queue_depth = 2;

  nexus_train::Trainer trainer(model, nexus_train::make_sparse_cross_entropy(), handle, config);
  trainer.add_metric(std::make_shared<nexus_train::Accuracy>());
  trainer.add_callback(std::make_shared<nexus_train::EarlyStopping>("val_loss", 6));
  auto checkpoints = std::make_shared<nexus_train::CheckpointManager>("checkpoints_example");
  trainer.add_callback(std::make_shared<nexus_train::ModelCheckpointCallback>(checkpoints, model, handle, "val_loss"));
  trainer.add_callback(std::make_shared<nexus_train::ProgressLoggerCallback>());
  trainer.fit(train_loader, &val_loader);
  checkpoints->flush();

  const double loss = trainer.state().metrics.count("loss") ? trainer.state().metrics.at("loss") : trainer.state().current_loss;
  std::cout << trainer.profile().report() << "\n";
  std::cout << "final loss " << loss << "\n";
  return loss < 1.2 ? 0 : 1;
}
