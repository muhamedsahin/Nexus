#pragma once

/**
 * @file activations.hpp
 * @brief Isimli aktivasyonlar ve lambda ile kurulan ozel aktivasyon.
 */

#include "nexus/nn/modules.hpp"
#include "nexus_model/activations/activations.hpp"

#include <cmath>
#include <string>

namespace nexus {
namespace nn {

class GELU : public Module {
 public:
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }

 private:
    mutable nexus_model::GELU impl_;
};

class SiLU : public Module {
 public:
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }

 private:
    mutable nexus_model::SiLU impl_;
};

class Mish : public Module {
 public:
    void train() override { impl_.train(); }
    void eval() override { impl_.eval(); }
    nexus_model::Tensor forward(const nexus_model::Tensor& input) override { return impl_.forward(input); }
    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override { return impl_.backward(grad_output); }
    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override { return {}; }

 private:
    mutable nexus_model::Mish impl_;
};

inline std::shared_ptr<Module> swish(const std::string& name = "Swish") {
    return make_activation(
        name, [](float x) { return x / (1.0f + std::exp(-x)); },
        [](float x, float grad_out) {
            const float sigmoid = 1.0f / (1.0f + std::exp(-x));
            return grad_out * (sigmoid + x * sigmoid * (1.0f - sigmoid));
        });
}

}  // namespace nn
}  // namespace nexus
