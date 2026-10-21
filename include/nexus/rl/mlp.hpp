#pragma once

/**
 * @file mlp.hpp
 * @brief Kucuk ileri beslemeli ag. Gizli katmanlar ReLU, son katman dogrusaldır.
 * Adam adimi yerinde calisir. `backward` giris gradyanini da yazabilir.
 */

#include "nexus/model/nxs_format.hpp"
#include "nexus/rl/rng.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nexus {
namespace rl {

class Mlp {
 public:
    Mlp() = default;

    Mlp(std::vector<int> sizes, std::uint32_t seed) : sizes_(std::move(sizes)) {
        if (sizes_.size() < 2) {
            throw std::invalid_argument("Mlp needs an input and an output size");
        }
        Rng rng(seed);
        layers_.resize(sizes_.size() - 1);
        for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
            Layer& slot = layers_[layer];
            slot.in = sizes_[layer];
            slot.out = sizes_[layer + 1];
            slot.relu = layer + 1 < layers_.size();
            const int count = slot.in * slot.out;
            const float scale = 1.0f / std::sqrt(static_cast<float>(slot.in));
            slot.weight.resize(static_cast<std::size_t>(count));
            slot.bias.assign(static_cast<std::size_t>(slot.out), 0.0f);
            slot.weight_m.assign(slot.weight.size(), 0.0f);
            slot.weight_v.assign(slot.weight.size(), 0.0f);
            slot.bias_m.assign(slot.bias.size(), 0.0f);
            slot.bias_v.assign(slot.bias.size(), 0.0f);
            slot.weight_grad.assign(slot.weight.size(), 0.0f);
            slot.bias_grad.assign(slot.bias.size(), 0.0f);
            for (float& value : slot.weight) {
                value = (rng.uniform() * 2.0f - 1.0f) * scale;
            }
        }
    }

    [[nodiscard]] int input_size() const { return sizes_.empty() ? 0 : sizes_.front(); }
    [[nodiscard]] int output_size() const { return sizes_.empty() ? 0 : sizes_.back(); }
    [[nodiscard]] int batch() const noexcept { return batch_; }

    void forward(const float* input, int batch, float* output) {
        if (batch <= 0) {
            throw std::invalid_argument("Mlp batch must be positive");
        }
        batch_ = batch;
        const float* current = input;
        for (std::size_t index = 0; index < layers_.size(); ++index) {
            Layer& layer = layers_[index];
            layer.input.resize(static_cast<std::size_t>(batch * layer.in));
            std::memcpy(layer.input.data(), current, layer.input.size() * sizeof(float));
            layer.pre.resize(static_cast<std::size_t>(batch * layer.out));
            layer.activation.resize(layer.pre.size());
            nexus_model::kernels::linear_forward(layer.input.data(), layer.weight.data(), layer.bias.data(), layer.pre.data(),
                                                 batch, layer.in, layer.out);
            if (layer.relu) {
                nexus_model::kernels::relu(layer.pre.data(), layer.activation.data(), layer.pre.size());
            } else {
                std::memcpy(layer.activation.data(), layer.pre.data(), layer.pre.size() * sizeof(float));
            }
            current = layer.activation.data();
        }
        std::memcpy(output, layers_.back().activation.data(), static_cast<std::size_t>(batch * output_size()) * sizeof(float));
    }

    void zero_grad() {
        for (Layer& layer : layers_) {
            std::fill(layer.weight_grad.begin(), layer.weight_grad.end(), 0.0f);
            std::fill(layer.bias_grad.begin(), layer.bias_grad.end(), 0.0f);
        }
    }

    /// `dy` son cikis boyundadir. `input_grad` ilk girisin gradyanini isterse dolu olmalidir.
    void backward(const float* dy, float* input_grad = nullptr) {
        scratch_.resize(static_cast<std::size_t>(batch_ * output_size()));
        std::memcpy(scratch_.data(), dy, scratch_.size() * sizeof(float));
        for (std::size_t reverse = layers_.size(); reverse > 0; --reverse) {
            Layer& layer = layers_[reverse - 1];
            layer.masked.resize(scratch_.size());
            for (std::size_t i = 0; i < scratch_.size(); ++i) {
                layer.masked[i] = (layer.relu && layer.pre[i] <= 0.0f) ? 0.0f : scratch_[i];
            }
            layer.dx.assign(static_cast<std::size_t>(batch_ * layer.in), 0.0f);
            nexus_model::kernels::linear_backward(layer.input.data(), layer.weight.data(), layer.masked.data(), layer.dx.data(),
                                                  layer.weight_grad.data(), layer.bias_grad.data(), batch_, layer.in, layer.out);
            scratch_.swap(layer.dx);
        }
        if (input_grad != nullptr) {
            std::memcpy(input_grad, scratch_.data(), static_cast<std::size_t>(batch_ * input_size()) * sizeof(float));
        }
    }

    void step(float learning_rate, float clip_norm = 5.0f) {
        ++adam_t_;
        const float bias1 = 1.0f - std::pow(0.9f, static_cast<float>(adam_t_));
        const float bias2 = 1.0f - std::pow(0.999f, static_cast<float>(adam_t_));
        double norm = 0.0;
        for (const Layer& layer : layers_) {
            for (float grad : layer.weight_grad) {
                norm += static_cast<double>(grad) * grad;
            }
            for (float grad : layer.bias_grad) {
                norm += static_cast<double>(grad) * grad;
            }
        }
        const float scale = norm > static_cast<double>(clip_norm) * clip_norm
                                ? clip_norm / (std::sqrt(static_cast<float>(norm)) + 1.0e-8f)
                                : 1.0f;
        for (Layer& layer : layers_) {
            adam(layer.weight, layer.weight_m, layer.weight_v, layer.weight_grad, learning_rate, bias1, bias2, scale);
            adam(layer.bias, layer.bias_m, layer.bias_v, layer.bias_grad, learning_rate, bias1, bias2, scale);
        }
    }

    void copy_weights_from(const Mlp& other) {
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            layers_[i].weight = other.layers_[i].weight;
            layers_[i].bias = other.layers_[i].bias;
        }
    }

    void soft_update_from(const Mlp& source, float tau) {
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            mix(layers_[i].weight, source.layers_[i].weight, tau);
            mix(layers_[i].bias, source.layers_[i].bias, tau);
        }
    }

    [[nodiscard]] std::vector<float> parameters() const {
        std::vector<float> values;
        for (const Layer& layer : layers_) {
            values.insert(values.end(), layer.weight.begin(), layer.weight.end());
            values.insert(values.end(), layer.bias.begin(), layer.bias.end());
        }
        return values;
    }

    [[nodiscard]] nxs::MlpBlob export_blob(std::string name) const {
        nxs::MlpBlob blob;
        blob.name = std::move(name);
        blob.sizes.assign(sizes_.begin(), sizes_.end());
        blob.adam_step = adam_t_;
        blob.parameters = parameters();
        for (const Layer& layer : layers_) {
            blob.moments.insert(blob.moments.end(), layer.weight_m.begin(), layer.weight_m.end());
            blob.moments.insert(blob.moments.end(), layer.weight_v.begin(), layer.weight_v.end());
            blob.moments.insert(blob.moments.end(), layer.bias_m.begin(), layer.bias_m.end());
            blob.moments.insert(blob.moments.end(), layer.bias_v.begin(), layer.bias_v.end());
        }
        return blob;
    }

    void import_blob(const nxs::MlpBlob& blob) {
        if (blob.sizes.size() != sizes_.size()) {
            throw std::invalid_argument("Mlp blob rank does not match");
        }
        for (std::size_t i = 0; i < sizes_.size(); ++i) {
            if (blob.sizes[i] != sizes_[i]) {
                throw std::invalid_argument("Mlp blob shape does not match");
            }
        }
        set_parameters(blob.parameters);
        std::size_t cursor = 0;
        auto take = [&](std::vector<float>& slot) {
            if (cursor + slot.size() > blob.moments.size()) {
                throw std::invalid_argument("Mlp moment blob is short");
            }
            std::copy(blob.moments.begin() + static_cast<std::ptrdiff_t>(cursor),
                      blob.moments.begin() + static_cast<std::ptrdiff_t>(cursor + slot.size()), slot.begin());
            cursor += slot.size();
        };
        for (Layer& layer : layers_) {
            take(layer.weight_m);
            take(layer.weight_v);
            take(layer.bias_m);
            take(layer.bias_v);
        }
        adam_t_ = blob.adam_step;
    }

    void set_parameters(const std::vector<float>& values) {
        std::size_t cursor = 0;
        for (Layer& layer : layers_) {
            if (cursor + layer.weight.size() + layer.bias.size() > values.size()) {
                throw std::invalid_argument("Mlp parameter blob is short");
            }
            std::copy(values.begin() + static_cast<std::ptrdiff_t>(cursor),
                      values.begin() + static_cast<std::ptrdiff_t>(cursor + layer.weight.size()), layer.weight.begin());
            cursor += layer.weight.size();
            std::copy(values.begin() + static_cast<std::ptrdiff_t>(cursor),
                      values.begin() + static_cast<std::ptrdiff_t>(cursor + layer.bias.size()), layer.bias.begin());
            cursor += layer.bias.size();
        }
    }

 private:
    struct Layer {
        int in = 0;
        int out = 0;
        bool relu = false;
        std::vector<float> weight;
        std::vector<float> bias;
        std::vector<float> weight_m;
        std::vector<float> weight_v;
        std::vector<float> bias_m;
        std::vector<float> bias_v;
        std::vector<float> weight_grad;
        std::vector<float> bias_grad;
        std::vector<float> input;
        std::vector<float> pre;
        std::vector<float> activation;
        std::vector<float> masked;
        std::vector<float> dx;
    };

    std::vector<float> scratch_;

    static void adam(std::vector<float>& value, std::vector<float>& moment, std::vector<float>& second,
                     const std::vector<float>& grad, float learning_rate, float bias1, float bias2, float scale) {
        constexpr float kBeta1 = 0.9f;
        constexpr float kBeta2 = 0.999f;
        constexpr float kEps = 1.0e-8f;
        for (std::size_t i = 0; i < value.size(); ++i) {
            const float g = grad[i] * scale;
            moment[i] = kBeta1 * moment[i] + (1.0f - kBeta1) * g;
            second[i] = kBeta2 * second[i] + (1.0f - kBeta2) * g * g;
            const float mhat = moment[i] / bias1;
            const float vhat = second[i] / bias2;
            value[i] -= learning_rate * mhat / (std::sqrt(vhat) + kEps);
        }
    }

    static void mix(std::vector<float>& target, const std::vector<float>& source, float tau) {
        for (std::size_t i = 0; i < target.size(); ++i) {
            target[i] = (1.0f - tau) * target[i] + tau * source[i];
        }
    }

    std::vector<int> sizes_;
    std::vector<Layer> layers_;
    int batch_ = 0;
    std::uint64_t adam_t_ = 0;
};

}  // namespace rl
}  // namespace nexus
