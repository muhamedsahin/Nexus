#include "check.hpp"
#include "toy.hpp"

#include "nexus_model/core/tensor.hpp"
#include "nexus_train/distributed/data_parallel_trainer.hpp"

#include <cmath>
#include <vector>

int test_distributed() {
  int failed = 0;
  auto first = RegressionToy::make_model(4, 3);
  auto second = RegressionToy::make_model(4, 9);
  auto sgd = RegressionToy::make_sgd(*first, 0.05);
  auto opt = nexus_train::wrap_optimizer(sgd);
  nexus_train::DataParallelStep parallel({first, second}, nexus_train::make_mse_loss(), opt);

  const auto* left = first->parameters().front();
  const auto* right = second->parameters().front();
  CHECK(left->data.numel() == right->data.numel());
  for (std::size_t i = 0; i < left->data.numel(); ++i) CHECK(left->data[i] == right->data[i]);

  std::vector<float> before(left->data.data(), left->data.data() + left->data.numel());
  auto inputs = nexus_model::Tensor::zeros({8, 4});
  auto labels = nexus_model::Tensor::zeros({8, 1});
  for (std::size_t i = 0; i < inputs.numel(); ++i) inputs[i] = 0.1f * static_cast<float>(i + 1);
  for (std::size_t i = 0; i < labels.numel(); ++i) labels[i] = static_cast<float>(i % 2);

  const double loss = parallel.step(inputs, labels);
  CHECK(std::isfinite(loss));
  bool changed = false;
  for (std::size_t i = 0; i < left->data.numel(); ++i) {
    CHECK(left->data[i] == right->data[i]);
    if (left->data[i] != before[i]) changed = true;
  }
  CHECK(changed);
  return failed;
}
