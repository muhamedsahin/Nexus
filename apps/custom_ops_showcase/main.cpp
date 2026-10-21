/**
 * @file main.cpp
 * Ozel Swish, ozel kayip ve yuksek seviye egitmen ayni programda.
 */

#include "nexus/nexus.hpp"

#include <cmath>
#include <iostream>
#include <span>
#include <vector>

namespace {

nexus::data::Table samples(int count) {
    nexus::data::Table table;
    table.features_per_sample = 8;
    table.features.assign(static_cast<std::size_t>(count * 8), 0.0f);
    table.labels.resize(static_cast<std::size_t>(count));
    for (int sample = 0; sample < count; ++sample) {
        const int label = sample % 2;
        table.labels[static_cast<std::size_t>(sample)] = label;
        table.features[static_cast<std::size_t>(sample * 8 + label)] = 1.0f;
    }
    return table;
}

}  // namespace

int main() {
    const std::vector<float> prediction{1.0f, 3.0f};
    const std::vector<float> target{0.0f, 1.0f};
    nexus::Tensor predicted(nexus::Shape{2}, prediction);
    nexus::Tensor truth(nexus::Shape{2}, target);
    nexus::Tensor grad = nexus::Tensor::zeros(nexus::Shape{2});
    auto custom = nexus::make_loss_tape(
        "TapeMSE", [](nexus::MicroTape& tape, std::span<const nexus::MicroTape::Var> pred, std::span<const float> y) {
            nexus::MicroTape::Var acc = tape.constant(0.0f);
            for (std::size_t i = 0; i < pred.size(); ++i) {
                const nexus::MicroTape::Var diff = tape.sub(pred[i], tape.constant(y[i]));
                acc = tape.add(acc, tape.square(diff));
            }
            return tape.mul(acc, tape.constant(0.5f));
        });
    const double value = custom(predicted, truth, grad);
    if (!(std::fabs(value - 2.5) < 1.0e-4) || !(std::fabs(grad[0] - 1.0f) < 1.0e-4f)) {
        std::cout << "custom loss mismatch\n";
        return 1;
    }

    auto train = nexus::data::make_loader(samples(64), 32, true, 3);
    auto validation = nexus::data::make_loader(samples(32), 32, false, 4);
    auto model = nexus::nn::Sequential()
                     .add<nexus::nn::Linear>(8, 16)
                     .add(nexus::nn::make_activation(
                         "Swish", [](float x) { return x / (1.0f + std::exp(-x)); },
                         [](float x, float upstream) {
                             const float sigmoid = 1.0f / (1.0f + std::exp(-x));
                             return upstream * (sigmoid + x * sigmoid * (1.0f - sigmoid));
                         }))
                     .add<nexus::nn::Linear>(16, 2);
    auto trainer = nexus::TrainerBuilder(model)
                       .optimizer(nexus::optim::AdamW(model->parameters(), {.lr = 0.08f, .weight_decay = 0.0f}))
                       .loss(nexus::loss::CrossEntropy(0.0f))
                       .metric<nexus::metrics::Accuracy>()
                       .build();
    trainer.fit(train, validation, 10);
    const float accuracy = trainer.history().back().accuracy;
    std::cout << "custom_swish_acc=" << accuracy << " tape_mse=" << value << "\n";
    return accuracy >= 0.9f ? 0 : 1;
}
