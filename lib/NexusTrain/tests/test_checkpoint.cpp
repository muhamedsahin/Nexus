#include "check.hpp"
#include "toy.hpp"

#include "nexus_model/adapters/nexus_optim.hpp"
#include "nexus_optim/algorithms/adamw.hpp"
#include "nexus_train/checkpoint/checkpoint_manager.hpp"
#include "nexus_train/core/trainer.hpp"

#include <cstring>
#include <filesystem>
#include <vector>

int test_checkpoint() {
  int failed = 0;
  RegressionToy toy(8, 3, 4, 1e-2, 4, false);
  nexus_optim::ParamGroupOptions group;
  group.learning_rate = 1e-2;
  nexus_optim::AdamW<float>::Options adam_options;
  adam_options.lr = 1e-2;
  adam_options.weight_decay = 0.0;
  auto adam = std::make_shared<nexus_optim::AdamW<float>>(
      std::vector<nexus_optim::ParamGroup<float>>{nexus_model::as_param_group(*toy.model, group)}, adam_options);
  auto handle = nexus_train::wrap_optimizer(adam);

  nexus_train::TrainConfig config;
  config.num_epochs = 1;
  config.deterministic_mode = true;
  config.log_every_n_steps = 0;
  config.validate_every_n_epochs = 0;
  config.seed = 4;
  nexus_train::Trainer trainer(toy.model, toy.loss, handle, config);
  trainer.fit(toy.loader);

  std::vector<float> saved_weights;
  for (nexus_model::Parameter* param : toy.model->parameters()) {
    saved_weights.insert(saved_weights.end(), param->data.data(), param->data.data() + param->data.numel());
  }
  const auto saved_state = adam->state_dict();
  const std::uint64_t saved_step = adam->step_count();
  CHECK(saved_step > 0);

  const auto directory = std::filesystem::temp_directory_path() / "nexus_train_ckpt_test";
  std::filesystem::remove_all(directory);
  nexus_train::CheckpointManager manager(directory.string());
  manager.save(*toy.model, *handle, trainer.state(), true);
  toy.model->parameters().front()->data[0] = 42.f;
  adam->step();
  manager.flush();
  toy.model->parameters().front()->data[0] = 0.f;
  manager.load(*toy.model, *handle, trainer.state(), manager.last_directory().string());

  std::size_t cursor = 0;
  for (nexus_model::Parameter* param : toy.model->parameters()) {
    for (std::size_t i = 0; i < param->data.numel(); ++i, ++cursor) {
      CHECK(param->data[i] == saved_weights[cursor]);
    }
  }
  CHECK(adam->step_count() == saved_step);
  const auto restored = adam->state_dict();
  CHECK(restored.blobs.size() == saved_state.blobs.size());
  CHECK(!restored.blobs.empty());
  CHECK(restored.blobs.front().size() == saved_state.blobs.front().size());
  CHECK(std::memcmp(restored.blobs.front().data(), saved_state.blobs.front().data(), restored.blobs.front().size()) == 0);
  std::filesystem::remove_all(directory);
  return failed;
}
