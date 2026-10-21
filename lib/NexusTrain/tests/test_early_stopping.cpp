#include "check.hpp"
#include "toy.hpp"

#include "nexus_train/callbacks/early_stopping.hpp"
#include "nexus_train/core/trainer.hpp"

int test_early_stopping() {
  int failed = 0;
  RegressionToy train(8, 2, 4, 0.0, 5, false);
  RegressionToy val(8, 2, 4, 0.0, 5, false);
  nexus_train::TrainConfig config;
  config.num_epochs = 8;
  config.deterministic_mode = true;
  config.log_every_n_steps = 0;
  config.seed = 5;
  config.validate_every_n_epochs = 1;
  nexus_train::Trainer trainer(train.model, train.loss, train.opt, config);
  trainer.add_callback(std::make_shared<nexus_train::EarlyStopping>("val_loss", 2, 0.0, nexus_train::MonitorMode::Min));
  trainer.fit(train.loader, &val.loader);
  CHECK(trainer.state().should_stop);
  CHECK(trainer.state().epoch == 2);
  return failed;
}
