#include "check.hpp"
#include "toy.hpp"

#include "nexus_train/callbacks/callback_base.hpp"
#include "nexus_train/core/trainer.hpp"

#include <vector>

namespace {

class LossTap final : public nexus_train::CallbackBase {
 public:
  std::vector<double> losses;
  void on_batch_end(nexus_train::TrainingState& state) override { losses.push_back(state.current_loss); }
};

}  // namespace

int test_reproducibility() {
  int failed = 0;
  auto run = [] {
    RegressionToy toy(12, 3, 4, 1e-2, 21, false);
    auto tap = std::make_shared<LossTap>();
    nexus_train::TrainConfig config;
    config.num_epochs = 2;
    config.deterministic_mode = true;
    config.shuffle_each_epoch = false;
    config.seed = 21;
    config.log_every_n_steps = 0;
    config.validate_every_n_epochs = 0;
    nexus_train::Trainer trainer(toy.model, toy.loss, toy.opt, config);
    trainer.add_callback(tap);
    trainer.fit(toy.loader);
    return tap->losses;
  };
  const auto first = run();
  const auto second = run();
  CHECK(first.size() == second.size());
  CHECK(!first.empty());
  for (std::size_t i = 0; i < first.size() && i < second.size(); ++i) {
    CHECK(first[i] == second[i]);
  }
  return failed;
}
