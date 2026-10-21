/**
 * @file main.cpp
 * Tabular piksel siniflandirma. Her ornek 8x8, etiket ilk uc pikselden okunur.
 */

#include "nexus/nexus.hpp"

#include <iostream>
#include <vector>

namespace {

nexus::data::Table samples(int count, int salt) {
    constexpr int kWidth = 64;
    constexpr int kClasses = 3;
    nexus::data::Table table;
    table.features_per_sample = kWidth;
    table.features.assign(static_cast<std::size_t>(count * kWidth), 0.0f);
    table.labels.resize(static_cast<std::size_t>(count));
    for (int sample = 0; sample < count; ++sample) {
        const int label = (sample + salt) % kClasses;
        table.labels[static_cast<std::size_t>(sample)] = label;
        table.features[static_cast<std::size_t>(sample * kWidth + label)] = 1.0f;
        table.features[static_cast<std::size_t>(sample * kWidth + 8 + (sample % 8))] = 0.1f;
    }
    return table;
}

}  // namespace

int main() {
    auto train = nexus::data::make_loader(samples(96, 0), 32, true, 1);
    auto validation = nexus::data::make_loader(samples(32, 2), 32, false, 1);
    auto model = nexus::nn::Sequential()
                     .add<nexus::nn::Linear>(64, 32)
                     .add<nexus::nn::ReLU>()
                     .add<nexus::nn::Linear>(32, 3);
    auto trainer = nexus::TrainerBuilder(model)
                       .optimizer(nexus::optim::AdamW(model->parameters(), {.lr = 0.05f, .weight_decay = 1.0e-4f}))
                       .loss(nexus::loss::CrossEntropy(0.05f))
                       .execution_policy(nexus::ExecutionPolicy::cpu())
                       .gradient_clip_norm(1.0f)
                       .metric<nexus::metrics::Accuracy>()
                       .metric<nexus::metrics::F1Score>()
                       .callback<nexus::callbacks::EarlyStopping>(8)
                       .build();
    trainer.fit(train, validation, 8);
    const nexus::EpochStats& last = trainer.history().back();
    std::cout << "val_acc=" << last.accuracy << " val_f1=" << last.f1 << " train_loss=" << last.train_loss << "\n";
    return last.accuracy >= 0.75f ? 0 : 1;
}
