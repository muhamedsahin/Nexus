#pragma once

/**
 * @file optimizer.hpp
 * @brief Parametre isaretcileri odunc alinir. Guncelleme kurali kullanicinindir.
 *
 * Kural her adimda bir parametre tamponu gorur: `param[i] -= ...`.
 * `sgd_rule` yerinde SGD'dir; agirlik cozunumu veya kirpma ayni imzaya yazilir.
 */

#include "nexus_model/core/parameter.hpp"

#include <functional>
#include <vector>

namespace nexus {

class CustomOptimizer {
 public:
    using Rule = std::function<void(float* param, const float* grad, std::size_t n, float lr)>;

    CustomOptimizer(std::vector<nexus_model::Parameter*> parameters, float learning_rate, Rule rule)
        : parameters_(std::move(parameters)), learning_rate_(learning_rate), rule_(std::move(rule)) {
        if (parameters_.empty()) {
            throw nexus_model::ModelError("CustomOptimizer requires a parameter");
        }
        if (!rule_) {
            throw nexus_model::ModelError("CustomOptimizer rule is empty");
        }
        if (!(learning_rate_ > 0.0f)) {
            throw nexus_model::ModelError("CustomOptimizer learning rate must be positive");
        }
    }

    void set_learning_rate(float learning_rate) {
        if (!(learning_rate > 0.0f)) {
            throw nexus_model::ModelError("CustomOptimizer learning rate must be positive");
        }
        learning_rate_ = learning_rate;
    }

    [[nodiscard]] float learning_rate() const noexcept { return learning_rate_; }

    void zero_grad() {
        for (nexus_model::Parameter* parameter : parameters_) {
            parameter->zero_grad();
        }
    }

    void step() {
        for (nexus_model::Parameter* parameter : parameters_) {
            if (!parameter->requires_grad) {
                continue;
            }
            if (parameter->grad.numel() != parameter->data.numel()) {
                throw nexus_model::ModelError("CustomOptimizer grad numel mismatch");
            }
            rule_(parameter->data.data(), parameter->grad.data(), parameter->data.numel(), learning_rate_);
        }
    }

 private:
    std::vector<nexus_model::Parameter*> parameters_;
    float learning_rate_ = 0.0f;
    Rule rule_;
};

inline CustomOptimizer::Rule sgd_rule() {
    return [](float* param, const float* grad, std::size_t n, float lr) {
        for (std::size_t i = 0; i < n; ++i) {
            param[i] -= lr * grad[i];
        }
    };
}

}  // namespace nexus
