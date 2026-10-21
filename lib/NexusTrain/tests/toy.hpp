#pragma once

#include "nexus_model/adapters/nexus_optim.hpp"
#include "nexus_model/containers/sequential.hpp"
#include "nexus_model/core/initializer.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_optim/algorithms/sgd.hpp"
#include "nexus_train/bridge/loss_driver.hpp"
#include "nexus_train/bridge/optimizer_handle.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/loading/dataloader.hpp"

#include <cstdint>
#include <memory>
#include <vector>

struct RegressionToy {
  std::shared_ptr<nexus_model::Sequential> model;
  std::shared_ptr<nexus_optim::SGD<float>> sgd;
  std::shared_ptr<nexus_train::OptimizerHandle> opt;
  std::shared_ptr<nexus_train::LossDriver> loss;
  nexusdata::InMemoryDataset dataset;
  nexusdata::DataLoader loader;

  RegressionToy(int samples, int features, int batch, double lr, std::uint64_t seed, bool shuffle)
      : model(make_model(features, seed)),
        sgd(make_sgd(*model, lr)),
        opt(nexus_train::wrap_optimizer(sgd)),
        loss(nexus_train::make_mse_loss()),
        dataset(make_features(samples, features), make_labels(samples)),
        loader(dataset, make_options(batch, seed, shuffle)) {}

  static std::shared_ptr<nexus_model::Sequential> make_model(int features, std::uint64_t seed) {
    nexus_model::init::manual_seed(seed);
    return std::make_shared<nexus_model::Sequential>(std::vector<std::shared_ptr<nexus_model::Module>>{
        std::make_shared<nexus_model::Linear>(static_cast<std::size_t>(features), 1)});
  }

  static std::shared_ptr<nexus_optim::SGD<float>> make_sgd(nexus_model::Module& model, double lr) {
    nexus_optim::ParamGroupOptions group;
    group.learning_rate = lr;
    nexus_optim::SGD<float>::Options options;
    options.lr = lr;
    options.weight_decay = 0.0;
    return std::make_shared<nexus_optim::SGD<float>>(
        std::vector<nexus_optim::ParamGroup<float>>{nexus_model::as_param_group(model, group)}, options);
  }

  static nexusdata::NDArray make_features(int samples, int features) {
    nexusdata::NDArray x(nexusdata::Shape{static_cast<std::size_t>(samples), static_cast<std::size_t>(features)},
                         nexusdata::DType::Float32);
    float* data = x.data<float>();
    for (int n = 0; n < samples; ++n) {
      for (int f = 0; f < features; ++f) {
        data[n * features + f] = 0.05f * static_cast<float>((n + 1) * (f + 1));
      }
    }
    return x;
  }

  static nexusdata::NDArray make_labels(int samples) {
    nexusdata::NDArray y(nexusdata::Shape{static_cast<std::size_t>(samples)}, nexusdata::DType::Float32);
    float* data = y.data<float>();
    for (int n = 0; n < samples; ++n) data[n] = 0.1f * static_cast<float>(n % 3);
    return y;
  }

  static nexusdata::DataLoaderOptions make_options(int batch, std::uint64_t seed, bool shuffle) {
    nexusdata::DataLoaderOptions options;
    options.batch_size = static_cast<std::size_t>(batch);
    options.shuffle = shuffle;
    options.seed = seed;
    options.num_workers = 0;
    return options;
  }
};
