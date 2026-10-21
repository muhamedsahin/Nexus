#include "check.hpp"

#include "nexus_model/core/tensor.hpp"
#include "nexus_train/metrics/accuracy.hpp"
#include "nexus_train/metrics/confusion_matrix.hpp"
#include "nexus_train/metrics/precision_recall_f1.hpp"

#include <cmath>

int test_metrics() {
  int failed = 0;
  const auto prediction = nexus_model::Tensor::from_values({2, 3}, {0.1f, 0.8f, 0.1f, 0.7f, 0.2f, 0.1f});
  const auto target = nexus_model::Tensor::from_values({2}, {1.f, 0.f});

  nexus_train::Accuracy accuracy;
  accuracy.update(prediction, target);
  CHECK(std::fabs(accuracy.compute() - 1.0) < 1e-12);

  const auto wrong = nexus_model::Tensor::from_values({2}, {1.f, 1.f});
  accuracy.reset();
  accuracy.update(prediction, wrong);
  CHECK(std::fabs(accuracy.compute() - 0.5) < 1e-12);

  nexus_train::PrecisionRecallF1 f1(nexus_train::Average::Macro);
  f1.update(prediction, target);
  CHECK(std::fabs(f1.f1() - 1.0) < 1e-12);
  CHECK(std::fabs(f1.precision() - 1.0) < 1e-12);

  nexus_train::PrecisionRecallF1 micro(nexus_train::Average::Micro);
  micro.update(prediction, wrong);
  CHECK(micro.f1() > 0.0 && micro.f1() < 1.0);

  nexus_train::ConfusionMatrix matrix;
  matrix.update(prediction, target);
  CHECK(matrix.at(1, 1) == 1);
  CHECK(matrix.at(0, 0) == 1);
  CHECK(matrix.at(0, 1) == 0);
  CHECK(std::fabs(matrix.compute() - 1.0) < 1e-12);
  return failed;
}
