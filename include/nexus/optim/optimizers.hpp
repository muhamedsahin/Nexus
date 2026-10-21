#pragma once

/**
 * @file optimizers.hpp
 * @brief Parametre isaretclerini NexusOptim AdamW ve SGD cekirdeklerine baglar.
 * Ogrenme hizi grup alanina yazilir; zamanlayici ayni alani gunceller.
 */

#include "nexus_model/core/parameter.hpp"
#include "nexus_optim/algorithms/adam.hpp"
#include "nexus_optim/algorithms/sgd.hpp"
#include "nexus_optim/core/optimizer_state.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nexus {
namespace optim {

struct AdamWOptions {
    float lr = 1.0e-3f;
    float weight_decay = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1.0e-8f;
};

struct SgdOptions {
    float lr = 1.0e-2f;
    float momentum = 0.0f;
    float weight_decay = 0.0f;
};

class Optimizer {
 public:
    virtual ~Optimizer() = default;
    virtual void zero_grad() = 0;
    virtual void step() = 0;
    virtual void set_learning_rate(float learning_rate) = 0;
    [[nodiscard]] virtual float learning_rate() const = 0;
    [[nodiscard]] virtual nexus_optim::OptimizerState state_dict() const = 0;
    virtual void load_state_dict(const nexus_optim::OptimizerState& state) = 0;
};

inline void clip_grad_norm(const std::vector<nexus_model::Parameter*>& parameters, float max_norm) {
    if (!(max_norm > 0.0f)) {
        throw std::invalid_argument("clip_grad_norm max_norm must be positive");
    }
    double square = 0.0;
    for (nexus_model::Parameter* parameter : parameters) {
        const float* grad = parameter->grad.data();
        for (std::size_t i = 0; i < parameter->grad.numel(); ++i) {
            square += static_cast<double>(grad[i]) * grad[i];
        }
    }
    const float norm = std::sqrt(static_cast<float>(square));
    if (!(norm > max_norm) || norm == 0.0f) {
        return;
    }
    const float scale = max_norm / norm;
    for (nexus_model::Parameter* parameter : parameters) {
        float* grad = parameter->grad.data();
        for (std::size_t i = 0; i < parameter->grad.numel(); ++i) {
            grad[i] *= scale;
        }
    }
}

namespace detail {

inline nexus_optim::ParamGroup<float> group_from(const std::vector<nexus_model::Parameter*>& parameters, float learning_rate) {
    std::vector<float*> values;
    std::vector<float*> grads;
    std::vector<std::size_t> sizes;
    values.reserve(parameters.size());
    grads.reserve(parameters.size());
    sizes.reserve(parameters.size());
    for (nexus_model::Parameter* parameter : parameters) {
        values.push_back(parameter->data.data());
        grads.push_back(parameter->grad.data());
        sizes.push_back(parameter->data.numel());
    }
    nexus_optim::ParamGroupOptions options;
    options.learning_rate = learning_rate;
    return nexus_optim::make_group<float>(std::move(values), std::move(grads), std::move(sizes), options);
}

}  // namespace detail

class AdamW : public Optimizer {
 public:
    AdamW(std::vector<nexus_model::Parameter*> parameters, AdamWOptions options)
        : parameters_(std::move(parameters)), learning_rate_(options.lr) {
        if (parameters_.empty()) {
            throw std::invalid_argument("AdamW requires parameters");
        }
        nexus_optim::AdamW<float>::Options native;
        native.lr = options.lr;
        native.beta1 = options.beta1;
        native.beta2 = options.beta2;
        native.eps = options.eps;
        native.weight_decay = options.weight_decay;
        impl_ = std::make_unique<nexus_optim::AdamW<float>>(
            std::vector<nexus_optim::ParamGroup<float>>{detail::group_from(parameters_, learning_rate_)}, native);
    }

    void zero_grad() override {
        for (nexus_model::Parameter* parameter : parameters_) {
            parameter->zero_grad();
        }
    }

    void step() override { impl_->step(); }

    void set_learning_rate(float learning_rate) override {
        learning_rate_ = learning_rate;
        impl_->param_groups().front().options.learning_rate = learning_rate;
    }

    [[nodiscard]] float learning_rate() const override { return learning_rate_; }
    [[nodiscard]] nexus_optim::OptimizerState state_dict() const override { return impl_->state_dict(); }
    void load_state_dict(const nexus_optim::OptimizerState& state) override { impl_->load_state_dict(state); }
    [[nodiscard]] const std::vector<nexus_model::Parameter*>& parameters() const noexcept { return parameters_; }

 private:
    std::vector<nexus_model::Parameter*> parameters_;
    float learning_rate_ = 1.0e-3f;
    std::unique_ptr<nexus_optim::AdamW<float>> impl_;
};

class SGD : public Optimizer {
 public:
    SGD(std::vector<nexus_model::Parameter*> parameters, SgdOptions options)
        : parameters_(std::move(parameters)), learning_rate_(options.lr) {
        if (parameters_.empty()) {
            throw std::invalid_argument("SGD requires parameters");
        }
        nexus_optim::SGD<float>::Options native;
        native.lr = options.lr;
        native.momentum = options.momentum;
        native.weight_decay = options.weight_decay;
        impl_ = std::make_unique<nexus_optim::SGD<float>>(
            std::vector<nexus_optim::ParamGroup<float>>{detail::group_from(parameters_, learning_rate_)}, native);
    }

    void zero_grad() override {
        for (nexus_model::Parameter* parameter : parameters_) {
            parameter->zero_grad();
        }
    }
    void step() override { impl_->step(); }
    void set_learning_rate(float learning_rate) override {
        learning_rate_ = learning_rate;
        impl_->param_groups().front().options.learning_rate = learning_rate;
    }
    [[nodiscard]] float learning_rate() const override { return learning_rate_; }
    [[nodiscard]] nexus_optim::OptimizerState state_dict() const override { return impl_->state_dict(); }
    void load_state_dict(const nexus_optim::OptimizerState& state) override { impl_->load_state_dict(state); }

 private:
    std::vector<nexus_model::Parameter*> parameters_;
    float learning_rate_ = 1.0e-2f;
    std::unique_ptr<nexus_optim::SGD<float>> impl_;
};

}  // namespace optim
}  // namespace nexus
