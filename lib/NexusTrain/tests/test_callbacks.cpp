#include "check.hpp"
#include "toy.hpp"

#include "nexus_optim/scheduler/step_lr.hpp"
#include "nexus_train/callbacks/callback_base.hpp"
#include "nexus_train/callbacks/lr_scheduler_callback.hpp"
#include "nexus_train/core/trainer.hpp"
#include "nexus_train/logging/csv_logger.hpp"

#include <fstream>
#include <iterator>
#include <string>

namespace {

class CountCallback final : public nexus_train::CallbackBase {
 public:
  int train_start = 0;
  int epoch_start = 0;
  int batch_start = 0;
  int batch_end = 0;
  int epoch_end = 0;
  int validation_end = 0;
  int train_end = 0;

  void on_train_start(nexus_train::TrainingState&) override { ++train_start; }
  void on_epoch_start(nexus_train::TrainingState&) override { ++epoch_start; }
  void on_batch_start(nexus_train::TrainingState&) override { ++batch_start; }
  void on_batch_end(nexus_train::TrainingState&) override { ++batch_end; }
  void on_epoch_end(nexus_train::TrainingState&) override { ++epoch_end; }
  void on_validation_end(nexus_train::TrainingState&) override { ++validation_end; }
  void on_train_end(nexus_train::TrainingState&) override { ++train_end; }
};

}  // namespace

int test_callbacks() {
  int failed = 0;
  RegressionToy train(4, 2, 2, 0.01, 3, false);
  RegressionToy val(4, 2, 2, 0.0, 3, false);
  auto counter = std::make_shared<CountCallback>();
  auto scheduler = std::make_shared<nexus_optim::StepLR>(*train.sgd, 1, 0.5);
  const double initial_lr = train.sgd->param_groups().front().options.learning_rate;

  nexus_train::TrainConfig config;
  config.num_epochs = 2;
  config.deterministic_mode = true;
  config.log_every_n_steps = 1;
  config.seed = 3;
  config.validate_every_n_epochs = 1;
  const std::string csv_path = "nexus_train_callbacks.csv";
  nexus_train::Trainer trainer(train.model, train.loss, train.opt, config);
  trainer.add_callback(counter);
  trainer.add_callback(std::make_shared<nexus_train::LRSchedulerCallback>(scheduler, nexus_train::SchedulerStep::Epoch));
  trainer.add_logger(std::make_shared<nexus_train::CsvLogger>(csv_path));
  trainer.fit(train.loader, &val.loader);

  CHECK(counter->train_start == 1);
  CHECK(counter->epoch_start == 2);
  CHECK(counter->batch_start == 4);
  CHECK(counter->batch_end == 4);
  CHECK(counter->validation_end == 2);
  CHECK(counter->epoch_end == 2);
  CHECK(counter->train_end == 1);
  CHECK(initial_lr == 0.01);
  const double lr = train.sgd->param_groups().front().options.learning_rate;
  CHECK(lr > 0.004 && lr < 0.006);

  std::ifstream csv(csv_path);
  std::string contents((std::istreambuf_iterator<char>(csv)), std::istreambuf_iterator<char>());
  CHECK(contents.find("kind,epoch,step,name,value") != std::string::npos);
  CHECK(contents.find("loss") != std::string::npos);
  return failed;
}
