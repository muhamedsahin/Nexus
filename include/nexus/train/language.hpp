#pragma once

/**
 * @file language.hpp
 * @brief Kaydirilmis token kaybi. Duz satir siniflandirmasi trainer.hpp icinde kalir.
 *
 * Girdi [batch, time] tamsayi tokendir. Kayip, logitlerin [:, :-1] dilimi ile
 * tokenlarin [:, 1:] dilimi arasindaki capraz entropidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/loss/losses.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/optim/optimizers.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nexus {

struct TokenRows {
    std::vector<int> values;
    int batch = 0;
    int time = 0;
};

class LanguageTrainer {
 public:
    LanguageTrainer(nn::Module& model, float learning_rate, float weight_decay = 0.0f)
        : model_(&model),
          optimizer_(std::make_unique<optim::AdamW>(model.parameters(),
                                                    optim::AdamWOptions{learning_rate, weight_decay, 0.9f, 0.999f, 1.0e-8f})) {}

    float fit(const TokenRows& rows, int epochs) {
        if (epochs < 1 || rows.batch < 1 || rows.time < 2) {
            throw std::invalid_argument("language batch needs at least two tokens");
        }
        if (static_cast<int>(rows.values.size()) != rows.batch * rows.time) {
            throw std::invalid_argument("language token shape does not match batch and time");
        }
        float loss = 0.0f;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            loss = step(rows);
        }
        last_ = loss;
        return loss;
    }

    [[nodiscard]] float last_loss() const noexcept { return last_; }

 private:
    float step(const TokenRows& rows) {
        optimizer_->zero_grad();
        nexus_model::Tensor tokens = nexus_model::Tensor::zeros(
            {static_cast<std::size_t>(rows.batch), static_cast<std::size_t>(rows.time)});
        for (int index = 0; index < rows.batch * rows.time; ++index) {
            tokens[static_cast<std::size_t>(index)] = static_cast<float>(rows.values[static_cast<std::size_t>(index)]);
        }
        const nexus_model::Tensor logits = model_->forward(tokens);
        if (logits.rank() != 3 || logits.size(0) != static_cast<std::size_t>(rows.batch) ||
            logits.size(1) != static_cast<std::size_t>(rows.time)) {
            throw std::invalid_argument("language logits must be [batch, time, vocab]");
        }
        const int vocab = static_cast<int>(logits.size(2));
        const int steps = rows.time - 1;
        const int count = rows.batch * steps;
        std::vector<float> flat(static_cast<std::size_t>(count) * static_cast<std::size_t>(vocab));
        std::vector<int> target(static_cast<std::size_t>(count));
        for (int batch = 0; batch < rows.batch; ++batch) {
            for (int time = 0; time < steps; ++time) {
                const float* source =
                    logits.data() + (static_cast<std::size_t>(batch * rows.time + time) * static_cast<std::size_t>(vocab));
                float* destination = flat.data() + (static_cast<std::size_t>(batch * steps + time) * static_cast<std::size_t>(vocab));
                std::copy(source, source + vocab, destination);
                target[static_cast<std::size_t>(batch * steps + time)] =
                    rows.values[static_cast<std::size_t>(batch * rows.time + time + 1)];
            }
        }
        std::vector<float> grad(flat.size(), 0.0f);
        const float loss = loss_.forward_backward(flat.data(), target.data(), count, vocab, grad.data());
        nexus_model::Tensor full = nexus_model::Tensor::zeros({static_cast<std::size_t>(rows.batch),
                                                               static_cast<std::size_t>(rows.time),
                                                               static_cast<std::size_t>(vocab)});
        for (int batch = 0; batch < rows.batch; ++batch) {
            for (int time = 0; time < steps; ++time) {
                const float* source = grad.data() + (static_cast<std::size_t>(batch * steps + time) * static_cast<std::size_t>(vocab));
                float* destination =
                    full.data() + (static_cast<std::size_t>(batch * rows.time + time) * static_cast<std::size_t>(vocab));
                std::copy(source, source + vocab, destination);
            }
        }
        model_->backward(full);
        optimizer_->step();
        return loss;
    }

    nn::Module* model_ = nullptr;
    std::unique_ptr<optim::AdamW> optimizer_;
    loss::CrossEntropy loss_{};
    float last_ = 0.0f;
};

}  // namespace nexus
