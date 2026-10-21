#pragma once

/**
 * @file trainer.hpp
 * @brief Builder ile kurulan egitim dongusu.
 * Ileri, kayip, geri, gradyan kirpma ve AdamW adimi ayni yerde biter.
 * Karisik hassasiyet bayragi kayip olcegi uygular; matmul FP32 kalir.
 * Hibrit politika Linear carpimini MatrixFlash zamanlayicisina baglar.
 */

#include "matrix_pro/hardware.hpp"
#include "nexus/core/config.hpp"
#include "nexus/data/datasets.hpp"
#include "nexus/loss/losses.hpp"
#include "nexus/metrics/metrics.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/optim/optimizers.hpp"
#include "nexus/model/nxs_format.hpp"
#include "nexus/train/callbacks.hpp"
#include "nexus_model/core/execution_policy.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace nexus {

struct EpochStats {
    int epoch = 0;
    float train_loss = 0.0f;
    float val_loss = 0.0f;
    float accuracy = 0.0f;
    float f1 = 0.0f;
};

namespace detail {

inline void hybrid_packed_gemm(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n, int k) {
    matrix_pro::HeterogeneousScheduler::instance().gemm(a, lda, b, ldb, c, ldc, m, n, k, 1.0f, 0.0f);
}

inline void apply_execution_policy(const ExecutionPolicy& policy) {
    if (policy.hybrid()) {
        nexus_model::set_packed_gemm(&hybrid_packed_gemm);
        nexus_model::set_matmul_policy(nexus_model::MatmulPolicy::Hybrid);
        return;
    }
    nexus_model::set_matmul_policy(nexus_model::MatmulPolicy::CpuKernel);
}

struct MetricBox {
    virtual ~MetricBox() = default;
    virtual void reset() = 0;
    virtual void update(const float* logits, const int* labels, int batch, int classes) = 0;
    [[nodiscard]] virtual double value() const = 0;
    [[nodiscard]] virtual const char* name() const = 0;
};

template <typename Metric>
struct MetricModel final : MetricBox {
    template <typename... Args>
    explicit MetricModel(Args&&... args) : metric(std::forward<Args>(args)...) {}
    void reset() override { metric.reset(); }
    void update(const float* logits, const int* labels, int batch, int classes) override {
        metric.update(logits, labels, batch, classes);
    }
    [[nodiscard]] double value() const override { return metric.value(); }
    [[nodiscard]] const char* name() const override { return metric.name(); }
    Metric metric;
};

class LossScaler {
 public:
    explicit LossScaler(bool enabled) : enabled_(enabled) {}
    void scale(float* grad, int n) const {
        if (!enabled_) {
            return;
        }
        for (int i = 0; i < n; ++i) {
            grad[i] *= scale_;
        }
    }
    bool unscale(const std::vector<nexus_model::Parameter*>& parameters) {
        if (!enabled_) {
            return true;
        }
        const float inv = 1.0f / scale_;
        for (nexus_model::Parameter* parameter : parameters) {
            float* grad = parameter->grad.data();
            for (std::size_t i = 0; i < parameter->grad.numel(); ++i) {
                if (!std::isfinite(grad[i])) {
                    scale_ = std::max(1.0f, scale_ * 0.5f);
                    return false;
                }
                grad[i] *= inv;
            }
        }
        return true;
    }
    [[nodiscard]] float scale() const noexcept { return scale_; }

 private:
    bool enabled_ = false;
    float scale_ = 128.0f;
};

}  // namespace detail

class Trainer {
 public:
    Trainer(nn::Sequential model, std::unique_ptr<optim::Optimizer> optimizer, loss::CrossEntropy loss, ExecutionPolicy policy,
            bool mixed_precision, float clip_norm, std::vector<std::unique_ptr<detail::MetricBox>> metrics,
            std::vector<std::unique_ptr<callbacks::Callback>> callbacks, std::function<float()> scheduler = {})
        : model_(std::move(model)), optimizer_(std::move(optimizer)), loss_(std::move(loss)), policy_(policy),
          mixed_precision_(mixed_precision), clip_norm_(clip_norm), scaler_(mixed_precision), metrics_(std::move(metrics)),
          callbacks_(std::move(callbacks)), scheduler_(std::move(scheduler)) {}

    [[nodiscard]] const std::vector<EpochStats>& history() const noexcept { return history_; }
    [[nodiscard]] int epochs_ran() const noexcept { return static_cast<int>(history_.size()); }
    nn::Sequential& model() noexcept { return model_; }

    void save_nxs(const std::string& path) const {
        nxs::Archive archive;
        const nxs::LayerSpec root = model_.nxs_spec();
        archive.layers = root.children;
        archive.weights = nxs::collect_weights(model_.parameters());
        archive.has_optim = true;
        archive.optim = optimizer_->state_dict();
        archive.has_trainer = true;
        archive.trainer.epoch = start_epoch_;
        archive.trainer.global_step = static_cast<std::int32_t>(history_.size());
        if (!history_.empty()) {
            archive.trainer.current_loss = history_.back().train_loss;
            archive.trainer.best_metric = history_.back().val_loss;
        }
        for (const EpochStats& stats : history_) {
            archive.trainer.metrics.emplace_back("hist_train", stats.train_loss);
            archive.trainer.metrics.emplace_back("hist_val", stats.val_loss);
            archive.trainer.metrics.emplace_back("hist_acc", stats.accuracy);
            archive.trainer.metrics.emplace_back("hist_f1", stats.f1);
        }
        for (const std::unique_ptr<callbacks::Callback>& callback : callbacks_) {
            if (const auto* early = dynamic_cast<const callbacks::EarlyStopping*>(callback.get())) {
                archive.trainer.metrics.emplace_back("early_best", early->best());
                archive.trainer.metrics.emplace_back("early_bad", static_cast<double>(early->bad_epochs()));
                archive.trainer.metrics.emplace_back("early_stop", early->stop() ? 1.0 : 0.0);
            }
        }
        nxs::save_archive(path, archive);
    }

    void load_nxs(const std::string& path) {
        const nxs::Archive archive = nxs::load_archive(path);
        const nxs::LayerSpec root = model_.nxs_spec();
        if (root.children.size() != archive.layers.size()) {
            throw std::runtime_error("nxs graph does not match the trainer model");
        }
        for (std::size_t i = 0; i < archive.layers.size(); ++i) {
            if (root.children[i].type != archive.layers[i].type) {
                throw std::runtime_error("nxs graph does not match the trainer model");
            }
        }
        nxs::apply_weights(model_.parameters(), archive.weights);
        if (archive.has_optim) {
            optimizer_->load_state_dict(archive.optim);
        }
        if (!archive.has_trainer) {
            return;
        }
        start_epoch_ = archive.trainer.epoch;
        history_.clear();
        std::size_t cursor = 0;
        const auto& metrics = archive.trainer.metrics;
        while (cursor + 3 < metrics.size() && metrics[cursor].first == "hist_train") {
            EpochStats stats;
            stats.epoch = static_cast<int>(history_.size());
            stats.train_loss = static_cast<float>(metrics[cursor].second);
            stats.val_loss = static_cast<float>(metrics[cursor + 1].second);
            stats.accuracy = static_cast<float>(metrics[cursor + 2].second);
            stats.f1 = static_cast<float>(metrics[cursor + 3].second);
            history_.push_back(stats);
            cursor += 4;
        }
        float early_best = std::numeric_limits<float>::quiet_NaN();
        int early_bad = 0;
        bool early_stop = false;
        bool saw_early = false;
        for (const auto& [name, value] : metrics) {
            if (name == "early_best") {
                early_best = static_cast<float>(value);
                saw_early = true;
            } else if (name == "early_bad") {
                early_bad = static_cast<int>(value);
            } else if (name == "early_stop") {
                early_stop = value > 0.5;
            }
        }
        if (saw_early) {
            for (const std::unique_ptr<callbacks::Callback>& callback : callbacks_) {
                if (auto* early = dynamic_cast<callbacks::EarlyStopping*>(callback.get())) {
                    early->restore(early_best, early_bad, early_stop);
                }
            }
        }
    }

    void fit(data::DataLoader& train, data::DataLoader& validation, int epochs) {
        if (epochs < 1) {
            throw std::invalid_argument("Trainer fit epochs must be positive");
        }
        const nexus_model::MatmulPolicy previous_policy = nexus_model::matmul_policy();
        const nexus_model::PackedGemmFn previous_hook = nexus_model::packed_gemm();
        detail::apply_execution_policy(policy_);
        if (start_epoch_ == 0) {
            history_.clear();
        }
        for (int epoch = start_epoch_; epoch < epochs; ++epoch) {
            EpochStats stats;
            stats.epoch = epoch;
            stats.train_loss = run_epoch(train, true);
            stats.val_loss = run_epoch(validation, false);
            for (const std::unique_ptr<detail::MetricBox>& metric : metrics_) {
                if (std::string(metric->name()) == "accuracy") {
                    stats.accuracy = static_cast<float>(metric->value());
                }
                if (std::string(metric->name()) == "f1") {
                    stats.f1 = static_cast<float>(metric->value());
                }
            }
            history_.push_back(stats);
            if (scheduler_) {
                optimizer_->set_learning_rate(scheduler_());
            }
            callbacks::EpochState state;
            state.epoch = epoch;
            state.train_loss = stats.train_loss;
            state.val_loss = stats.val_loss;
            state.accuracy = stats.accuracy;
            state.f1 = stats.f1;
            bool halt = false;
            for (const std::unique_ptr<callbacks::Callback>& callback : callbacks_) {
                callback->on_epoch_end(state);
                halt = halt || callback->stop();
            }
            if (halt) {
                break;
            }
        }
        start_epoch_ = static_cast<int>(history_.size());
        nexus_model::set_packed_gemm(previous_hook);
        nexus_model::set_matmul_policy(previous_policy);
    }

 private:
    float run_epoch(data::DataLoader& loader, bool update) {
        loader.reset();
        for (const std::unique_ptr<detail::MetricBox>& metric : metrics_) {
            if (!update) {
                metric->reset();
            }
        }
        data::Batch batch;
        double loss_sum = 0.0;
        int seen = 0;
        while (loader.next(batch)) {
            nexus_model::Tensor input = nexus_model::Tensor::zeros(
                {static_cast<std::size_t>(batch.batch_size), static_cast<std::size_t>(batch.features_per_sample)});
            std::memcpy(input.data(), batch.features.data(), batch.features.size() * sizeof(float));
            if (update) {
                model_.zero_grad();
            }
            nexus_model::Tensor logits = model_.forward(input);
            const int classes = static_cast<int>(logits.size(logits.rank() - 1));
            nexus_model::Tensor grad = nexus_model::Tensor::zeros(
                {static_cast<std::size_t>(batch.batch_size), static_cast<std::size_t>(classes)});
            const float loss = loss_.forward_backward(logits.data(), batch.labels.data(), batch.batch_size, classes, grad.data());
            if (!update) {
                for (const std::unique_ptr<detail::MetricBox>& metric : metrics_) {
                    metric->update(logits.data(), batch.labels.data(), batch.batch_size, classes);
                }
            }
            if (update) {
                scaler_.scale(grad.data(), static_cast<int>(grad.numel()));
                model_.backward(grad);
                if (scaler_.unscale(model_->parameters())) {
                    if (clip_norm_ > 0.0f) {
                        optim::clip_grad_norm(model_->parameters(), clip_norm_);
                    }
                    optimizer_->step();
                }
            }
            loss_sum += static_cast<double>(loss) * batch.batch_size;
            seen += batch.batch_size;
        }
        return seen == 0 ? 0.0f : static_cast<float>(loss_sum / static_cast<double>(seen));
    }

    nn::Sequential model_;
    std::unique_ptr<optim::Optimizer> optimizer_;
    loss::CrossEntropy loss_;
    ExecutionPolicy policy_ = ExecutionPolicy::cpu();
    bool mixed_precision_ = false;
    float clip_norm_ = 0.0f;
    detail::LossScaler scaler_;
    std::vector<std::unique_ptr<detail::MetricBox>> metrics_;
    std::vector<std::unique_ptr<callbacks::Callback>> callbacks_;
    std::function<float()> scheduler_;
    std::vector<EpochStats> history_;
    int start_epoch_ = 0;
};

class TrainerBuilder {
 public:
    explicit TrainerBuilder(nn::Sequential model) : model_(std::move(model)) {}

    TrainerBuilder& optimizer(optim::AdamW optimizer) {
        optimizer_ = std::make_unique<optim::AdamW>(std::move(optimizer));
        return *this;
    }

    TrainerBuilder& optimizer(optim::SGD optimizer) {
        optimizer_ = std::make_unique<optim::SGD>(std::move(optimizer));
        return *this;
    }

    TrainerBuilder& loss(loss::CrossEntropy loss) {
        loss_ = std::move(loss);
        has_loss_ = true;
        return *this;
    }

    TrainerBuilder& execution_policy(ExecutionPolicy policy) {
        policy_ = policy;
        return *this;
    }

    TrainerBuilder& mixed_precision(bool enabled) {
        mixed_precision_ = enabled;
        return *this;
    }

    TrainerBuilder& gradient_clip_norm(float norm) {
        clip_norm_ = norm;
        return *this;
    }

    template <typename Metric, typename... Args>
    TrainerBuilder& metric(Args&&... args) {
        metrics_.push_back(std::make_unique<detail::MetricModel<Metric>>(std::forward<Args>(args)...));
        return *this;
    }

    template <typename Callback, typename... Args>
    TrainerBuilder& callback(Args&&... args) {
        callbacks_.push_back(std::make_unique<Callback>(std::forward<Args>(args)...));
        return *this;
    }

    TrainerBuilder& callback(std::shared_ptr<callbacks::Callback> callback) {
        struct Holder : callbacks::Callback {
            explicit Holder(std::shared_ptr<callbacks::Callback> inner) : inner_(std::move(inner)) {}
            void on_epoch_end(const callbacks::EpochState& state) override { inner_->on_epoch_end(state); }
            [[nodiscard]] bool stop() const override { return inner_->stop(); }
            std::shared_ptr<callbacks::Callback> inner_;
        };
        callbacks_.push_back(std::unique_ptr<callbacks::Callback>(new Holder(std::move(callback))));
        return *this;
    }

    template <typename Scheduler>
    TrainerBuilder& learning_rate_schedule(Scheduler scheduler) {
        scheduler_ = [scheduler]() mutable { return scheduler.step(); };
        return *this;
    }

    [[nodiscard]] Trainer build() {
        if (!optimizer_ || !has_loss_) {
            throw std::logic_error("TrainerBuilder requires an optimizer and a loss");
        }
        return Trainer(std::move(model_), std::move(optimizer_), std::move(loss_), policy_, mixed_precision_, clip_norm_,
                       std::move(metrics_), std::move(callbacks_), std::move(scheduler_));
    }

 private:
    nn::Sequential model_;
    std::unique_ptr<optim::Optimizer> optimizer_;
    loss::CrossEntropy loss_{};
    bool has_loss_ = false;
    ExecutionPolicy policy_ = ExecutionPolicy::cpu();
    bool mixed_precision_ = false;
    float clip_norm_ = 0.0f;
    std::vector<std::unique_ptr<detail::MetricBox>> metrics_;
    std::vector<std::unique_ptr<callbacks::Callback>> callbacks_;
    std::function<float()> scheduler_;
};

}  // namespace nexus
