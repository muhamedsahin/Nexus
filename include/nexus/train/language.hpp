#pragma once

/**
 * @file language.hpp
 * @brief Eski kaydirilmis-token egitmeni. FAZ 9: govde dil modeli egitmeni v2'ye yonlenir.
 *
 * Girdi [batch, time] tamsayi tokendir. Kayip, logitlerin [:, :-1] dilimi ile
 * tokenlarin [:, 1:] dilimi arasindaki capraz entropidir (FAZ 3.E fuzyonlu CE, kopyasiz).
 *
 * `fit(rows, epochs)`: "epoch" = AYNI yiginda bir optimizer adimi (veri tekrar okunmaz,
 * biriktirme yok). Icerde LanguageModelTrainer: CPU nn::Module arka ucu, sabit satir kaynagi,
 * sabit LR, kirpma yok, AdamW(lr, wd, 0.9, 0.999, 1e-8) ve sonum TUM parametrelerde (FAZ 9
 * oncesi davranis; v2 varsayilani bias/norm'u sonumlemez). Kayiplar FAZ 9 oncesi kodla ayni
 * (test_old_api_still_works altin degerleri). Tek fark: sonlu olmayan gradyanli adim artik
 * uygulanmaz, atlanir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/fused/training.hpp"
#include "nexus/loss/losses.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/optim/optimizers.hpp"
#include "nexus/train/lm_trainer.hpp"

#include <algorithm>
#include <cstdint>
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
        : model_(&model), learning_rate_(learning_rate), weight_decay_(weight_decay) {}

    float fit(const TokenRows& rows, int epochs) {
        if (epochs < 1 || rows.batch < 1 || rows.time < 2) {
            throw std::invalid_argument("language batch needs at least two tokens");
        }
        if (static_cast<int>(rows.values.size()) != rows.batch * rows.time) {
            throw std::invalid_argument("language token shape does not match batch and time");
        }
        std::vector<std::int32_t> tokens(rows.values.begin(), rows.values.end());
        if (!trainer_) {
            source_ = std::make_unique<train::FixedRowsSource>(std::move(tokens), rows.batch, rows.time);
            train::ParamGroupPolicy groups;
            groups.decay = train::ParamGroupPolicy::Decay::All;
            train::SpikeOptions no_spike;
            no_spike.enabled = false;
            // FAZ 9 oncesi AdamWOptions{lr, wd, 0.9f, 0.999f, 1e-8f}: float sabitler ayni degerle.
            trainer_ = train::LanguageTrainerBuilder(*model_)
                           .data(*source_)
                           .optimizer(train::AdamW{learning_rate_, 0.9f, 0.999f, 1.0e-8f, weight_decay_}, groups)
                           .schedule(train::ConstantLr{})
                           .micro_batch(rows.batch)
                           .seq_len(rows.time)
                           .spike_detector(no_spike)
                           .log_console(false)
                           .build();
        } else {
            if (rows.batch != trainer_->backend().micro_batch() || rows.time != trainer_->backend().seq_len()) {
                trainer_->reshape(rows.batch, rows.time);
            }
            source_->set(std::move(tokens), rows.batch, rows.time);
        }
        const train::TrainSummary summary = trainer_->train(trainer_->step() + epochs);
        last_ = static_cast<float>(summary.last_loss);
        return last_;
    }

    [[nodiscard]] float last_loss() const noexcept { return last_; }
    /// FAZ 9: icteki v2 egitmeni (ilk fit'ten sonra; oncesinde nullptr).
    [[nodiscard]] train::LanguageModelTrainer* v2() noexcept { return trainer_.get(); }

 private:
    nn::Module* model_ = nullptr;
    float learning_rate_ = 1.0e-3f;
    float weight_decay_ = 0.0f;
    std::unique_ptr<train::FixedRowsSource> source_;
    std::unique_ptr<train::LanguageModelTrainer> trainer_;
    float last_ = 0.0f;
};

}  // namespace nexus
