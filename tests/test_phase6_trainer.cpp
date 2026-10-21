/**
 * @file test_phase6_trainer.cpp
 * @brief Akici TrainerBuilder, metrik, zamanlayici ve all-reduce.
 */

#include "nexus/nexus.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void expect(bool condition, const char* name) {
    if (condition) {
        std::cout << "  PASS  " << name << "\n";
        return;
    }
    std::cout << "  FAIL  " << name << "\n";
    ++g_failed;
}

nexus::data::Table make_table(int samples, int features, int classes, int salt) {
    nexus::data::Table table;
    table.features_per_sample = features;
    table.features.assign(static_cast<std::size_t>(samples * features), 0.0f);
    table.labels.resize(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample) {
        const int label = (sample + salt) % classes;
        table.labels[static_cast<std::size_t>(sample)] = label;
        table.features[static_cast<std::size_t>(sample * features + label)] = 1.0f;
        table.features[static_cast<std::size_t>(sample * features + classes)] = 0.05f * static_cast<float>(sample % 4);
    }
    return table;
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 6 yuksek seviye API testi\n";
    try {
        expect(nexus::Device::cpu().is_host(), "device cpu");
        expect(nexus::Device::cuda(0).is_accelerator(), "device cuda");
        expect(nexus::Device::hybrid().is_hybrid(), "device hybrid");
        expect(nexus::ExecutionPolicy::hybrid_cpu_multi_gpu().hybrid(), "execution policy hybrid");

        float probs[3] = {};
        const float logits[3] = {1.0f, 2.0f, 0.0f};
        nexus::nn::functional::softmax(logits, probs, 1, 3);
        expect(std::fabs((probs[0] + probs[1] + probs[2]) - 1.0f) < 1.0e-5f, "softmax rows sum to one");

        const nexus::data::Table parsed = nexus::data::parse_csv("f0,f1,label\n1,0,0\n0,1,1\n");
        expect(parsed.samples() == 2 && parsed.labels[1] == 1 && parsed.features_per_sample == 2, "csv parser");

        nexus::optim::CosineAnnealing cosine(1.0f, 2);
        const float lr0 = cosine.step();
        const float lr1 = cosine.step();
        const float lr2 = cosine.step();
        expect(std::fabs(lr0 - 1.0f) < 1.0e-5f && lr2 < lr1 && lr1 < lr0, "cosine annealing falls");

        nexus::nn::Linear clip_layer(2, 1);
        clip_layer.parameters()[0]->grad.fill(3.0f);
        nexus::optim::clip_grad_norm(clip_layer.parameters(), 1.0f);
        double square = 0.0;
        for (std::size_t i = 0; i < clip_layer.parameters()[0]->grad.numel(); ++i) {
            const float value = clip_layer.parameters()[0]->grad[i];
            square += static_cast<double>(value) * value;
        }
        expect(std::sqrt(square) <= 1.0f + 1.0e-4f, "gradient clip");

        std::vector<std::vector<float>> ranks{{1.0f, 3.0f}, {3.0f, 5.0f}};
        nexus::train::all_reduce_mean(ranks);
        expect(std::fabs(ranks[0][0] - 2.0f) < 1.0e-5f && std::fabs(ranks[1][1] - 4.0f) < 1.0e-5f, "ring allreduce mean");

        nexus::callbacks::EarlyStopping early(2);
        nexus::callbacks::EpochState epoch;
        epoch.val_loss = 1.0f;
        early.on_epoch_end(epoch);
        epoch.val_loss = 1.0f;
        early.on_epoch_end(epoch);
        epoch.val_loss = 1.0f;
        early.on_epoch_end(epoch);
        expect(early.stop(), "early stopping patience");

        constexpr int kFeatures = 16;
        constexpr int kClasses = 3;
        const nexus::data::Table train_table = make_table(72, kFeatures, kClasses, 0);
        const nexus::data::Table val_table = make_table(24, kFeatures, kClasses, 1);
        auto train = nexus::data::make_loader(train_table, 24, true, 5);
        auto validation = nexus::data::make_loader(val_table, 24, false, 1);
        auto model = nexus::nn::Sequential()
                         .add<nexus::nn::Linear>(kFeatures, 32)
                         .add(nexus::nn::swish())
                         .add<nexus::nn::LayerNorm>(32)
                         .add<nexus::nn::Linear>(32, kClasses);
        auto trainer = nexus::TrainerBuilder(model)
                           .optimizer(nexus::optim::AdamW(model->parameters(), {.lr = 0.05f, .weight_decay = 0.0f}))
                           .loss(nexus::loss::CrossEntropy(0.1f))
                           .execution_policy(nexus::ExecutionPolicy::cpu())
                           .mixed_precision(true)
                           .gradient_clip_norm(1.0f)
                           .metric<nexus::metrics::Accuracy>()
                           .metric<nexus::metrics::F1Score>()
                           .callback<nexus::callbacks::EarlyStopping>(30)
                           .build();
        trainer.fit(train, validation, 12);
        const nexus::EpochStats& last = trainer.history().back();
        const float first_loss = trainer.history().front().train_loss;
        std::cout << "  info  loss " << first_loss << " -> " << last.train_loss << " acc=" << last.accuracy
                  << " f1=" << last.f1 << "\n";
        expect(last.train_loss < first_loss, "trainer loss decreases");
        expect(last.accuracy > 0.9f, "validation accuracy");
        expect(last.f1 > 0.9f, "validation f1");
    } catch (const std::exception& error) {
        std::cout << "  FAIL  istisna: " << error.what() << "\n";
        return 1;
    }
    if (g_failed != 0) {
        std::cout << g_failed << " kontrol basarisiz\n";
        return 1;
    }
    std::cout << "tum kontroller gecti\n";
    return 0;
}
