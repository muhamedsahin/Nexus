/**
 * @file test_phase4_custom_extensions.cpp
 * @brief Ozel aktivasyon, kayip, katman ve ucundan uca egitim.
 */

#include "nexus/custom/activation_simd.hpp"
#include "nexus/custom/custom.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
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

float swish(float x) { return x / (1.0f + std::exp(-x)); }

float swish_backward(float x, float grad_out) {
    const float sigmoid = 1.0f / (1.0f + std::exp(-x));
    return grad_out * (sigmoid + x * sigmoid * (1.0f - sigmoid));
}

class ScaleBias : public nexus::CustomModule {
 public:
    explicit ScaleBias(std::size_t features) : features_(features) {
        auto weight = std::make_shared<nexus_model::Parameter>(nexus_model::Tensor::zeros({features}));
        auto bias = std::make_shared<nexus_model::Parameter>(nexus_model::Tensor::zeros({features}));
        weight->data.fill(0.1f);
        bias->data.fill(0.0f);
        add_parameter("weight", weight);
        add_parameter("bias", bias);
        weight_ = std::move(weight);
        bias_ = std::move(bias);
    }

    nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
        cache_.copy_from(input);
        output_.resize_like(input);
        const int features = static_cast<int>(features_);
        const int rows = static_cast<int>(input.numel() / features_);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < features; ++col) {
                const std::size_t index = static_cast<std::size_t>(row * features + col);
                output_[index] = cache_[index] * weight_->data[static_cast<std::size_t>(col)] +
                                 bias_->data[static_cast<std::size_t>(col)];
            }
        }
        return output_;
    }

    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override {
        grad_input_.resize_like(cache_);
        const int features = static_cast<int>(features_);
        const int rows = static_cast<int>(grad_output.numel() / features_);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < features; ++col) {
                const std::size_t index = static_cast<std::size_t>(row * features + col);
                const float upstream = grad_output[index];
                weight_->grad[static_cast<std::size_t>(col)] += upstream * cache_[index];
                bias_->grad[static_cast<std::size_t>(col)] += upstream;
                grad_input_[index] = upstream * weight_->data[static_cast<std::size_t>(col)];
            }
        }
        return grad_input_;
    }

 private:
    std::size_t features_;
    std::shared_ptr<nexus_model::Parameter> weight_;
    std::shared_ptr<nexus_model::Parameter> bias_;
    nexus_model::Tensor cache_;
    nexus_model::Tensor output_;
    nexus_model::Tensor grad_input_;
};

class StepCounter : public nexus::CustomCallback {
 public:
    int steps = 0;
    void on_step_end(int, double) override { ++steps; }
};

double asymmetric_huber(const nexus::Tensor& pred, const nexus::Tensor& target, nexus::Tensor& grad) {
    constexpr float kDelta = 1.0f;
    constexpr float kPositive = 2.0f;
    constexpr float kNegative = 0.5f;
    const float inv_n = 1.0f / static_cast<float>(pred.numel());
    double sum = 0.0;
    for (std::size_t i = 0; i < pred.numel(); ++i) {
        const float diff = pred[i] - target[i];
        const float magnitude = std::fabs(diff);
        const float weight = diff >= 0.0f ? kPositive : kNegative;
        float local = 0.0f;
        float local_grad = 0.0f;
        if (magnitude <= kDelta) {
            local = 0.5f * diff * diff;
            local_grad = diff;
        } else {
            local = kDelta * (magnitude - 0.5f * kDelta);
            local_grad = diff > 0.0f ? kDelta : -kDelta;
        }
        sum += static_cast<double>(weight * local);
        grad[i] = weight * local_grad * inv_n;
    }
    return sum * static_cast<double>(inv_n);
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 4 ozel genisletme testi\n";
    try {
        auto analytic = nexus::make_activation("SwishCustom", swish, swish_backward);
        expect(analytic.name() == "SwishCustom", "activation name");
        float y = 0.0f;
        float grad = 0.0f;
        const float x = 1.0f;
        analytic.forward_buffer(&x, &y, 1);
        analytic.backward_buffer(&x, &x, &grad, 1);
        const float expected_y = swish(x);
        const float expected_grad = swish_backward(x, x);
        expect(std::fabs(y - expected_y) < 1.0e-6f, "swish forward");
        expect(std::fabs(grad - expected_grad) < 1.0e-6f, "swish backward");

        auto finite = nexus::make_activation("SwishFinite", swish);
        float finite_grad = 0.0f;
        const float unit = 1.0f;
        finite.backward_buffer(&x, &unit, &finite_grad, 1);
        expect(std::fabs(finite_grad - swish_backward(x, 1.0f)) < 2.0e-3f, "finite-difference swish");

        auto scaled = nexus::make_activation("ScaleTwo", [](float value) { return value; });
        nexus::bind_avx2_elementwise(
            scaled, [](__m256 value) { return _mm256_mul_ps(value, _mm256_set1_ps(2.0f)); },
            [](__m256) { return _mm256_set1_ps(2.0f); });
        std::vector<float> wide(17);
        std::vector<float> wide_out(17);
        std::vector<float> wide_grad(17);
        std::vector<float> ones(17, 1.0f);
        for (int i = 0; i < 17; ++i) {
            wide[static_cast<std::size_t>(i)] = 0.25f * static_cast<float>(i - 8);
        }
        scaled.forward_buffer(wide.data(), wide_out.data(), wide.size());
        scaled.backward_buffer(wide.data(), ones.data(), wide_grad.data(), wide.size());
        float avx_worst = 0.0f;
        for (std::size_t i = 0; i < wide.size(); ++i) {
            avx_worst = std::max(avx_worst, std::fabs(wide_out[i] - 2.0f * wide[i]));
            avx_worst = std::max(avx_worst, std::fabs(wide_grad[i] - 2.0f));
        }
        std::cout << "  info  avx_max_abs=" << avx_worst << "\n";
        expect(avx_worst < 1.0e-6f, "avx2 batch activation");

        std::vector<float> parallel_in(4096);
        std::vector<float> parallel_out(4096);
        for (std::size_t i = 0; i < parallel_in.size(); ++i) {
            parallel_in[i] = 0.001f * static_cast<float>(static_cast<int>(i) - 2000);
        }
        analytic.forward_buffer(parallel_in.data(), parallel_out.data(), parallel_in.size());
        float parallel_worst = 0.0f;
        for (std::size_t i = 0; i < parallel_in.size(); ++i) {
            parallel_worst = std::max(parallel_worst, std::fabs(parallel_out[i] - swish(parallel_in[i])));
        }
        expect(parallel_worst < 1.0e-5f, "parallel swish");

        const std::vector<float> pred_values{2.0f};
        const std::vector<float> target_values{0.0f};
        nexus::Tensor pred(nexus::Shape{1}, pred_values);
        nexus::Tensor target(nexus::Shape{1}, target_values);
        nexus::Tensor grad_out = nexus::Tensor::zeros(nexus::Shape{1});
        auto huber = nexus::make_loss("AsymmetricHuber", asymmetric_huber);
        const double huber_value = huber(pred, target, grad_out);
        expect(std::fabs(huber_value - 3.0) < 1.0e-5, "asymmetric huber loss");
        expect(std::fabs(grad_out[0] - 2.0f) < 1.0e-5f, "asymmetric huber grad");

        auto numeric = nexus::make_loss_autograd("NumericMSE", [](std::span<const float> prediction, std::span<const float> truth) {
            double sum = 0.0;
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const double diff = static_cast<double>(prediction[i] - truth[i]);
                sum += diff * diff;
            }
            return sum / static_cast<double>(prediction.size());
        });
        const std::vector<float> mse_pred{1.0f, 3.0f};
        const std::vector<float> mse_target{0.0f, 1.0f};
        nexus::Tensor mse_p(nexus::Shape{2}, mse_pred);
        nexus::Tensor mse_t(nexus::Shape{2}, mse_target);
        nexus::Tensor mse_g = nexus::Tensor::zeros(nexus::Shape{2});
        const double mse_value = numeric(mse_p, mse_t, mse_g);
        expect(std::fabs(mse_value - 2.5) < 1.0e-3, "numeric mse value");
        expect(std::fabs(mse_g[0] - 1.0f) < 2.0e-2f && std::fabs(mse_g[1] - 2.0f) < 2.0e-2f, "numeric mse grad");

        auto taped = nexus::make_loss_tape(
            "TapeMSE", [](nexus::MicroTape& tape, std::span<const nexus::MicroTape::Var> prediction, std::span<const float> truth) {
                nexus::MicroTape::Var acc = tape.constant(0.0f);
                for (std::size_t i = 0; i < prediction.size(); ++i) {
                    const nexus::MicroTape::Var diff = tape.sub(prediction[i], tape.constant(truth[i]));
                    acc = tape.add(acc, tape.square(diff));
                }
                return tape.mul(acc, tape.constant(1.0f / static_cast<float>(prediction.size())));
            });
        mse_g.zero();
        const double tape_value = taped(mse_p, mse_t, mse_g);
        expect(std::fabs(tape_value - 2.5) < 1.0e-5, "tape mse value");
        expect(std::fabs(mse_g[0] - 1.0f) < 1.0e-5f && std::fabs(mse_g[1] - 2.0f) < 1.0e-5f, "tape mse grad");

        nexus::CustomMetric mae("MAE", [](std::span<const float> prediction, std::span<const float> truth) {
            double sum = 0.0;
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                sum += std::fabs(static_cast<double>(prediction[i] - truth[i]));
            }
            return sum / static_cast<double>(prediction.size());
        });
        mae.update(mse_pred, mse_target);
        expect(std::fabs(mae.value() - 1.5) < 1.0e-6, "mae metric");

        constexpr int kBatch = 32;
        constexpr float kTrueWeight = 1.0f;
        constexpr float kTrueBias = 0.2f;
        ScaleBias layer(1);
        auto activation = nexus::make_activation("SwishTrain", swish, swish_backward);
        auto train_loss = nexus::make_loss("AsymmetricHuber", asymmetric_huber);
        nexus::CustomOptimizer optimizer(layer.parameters(), 0.2f, [](float* param, const float* parameter_grad, std::size_t n, float lr) {
            for (std::size_t i = 0; i < n; ++i) {
                param[i] -= lr * parameter_grad[i];
            }
        });
        auto counter = std::make_shared<StepCounter>();
        nexus::CallbackList callbacks;
        callbacks.add(counter);
        callbacks.on_train_start();

        nexus_model::Tensor input = nexus_model::Tensor::zeros({kBatch, 1});
        std::vector<float> truth(static_cast<std::size_t>(kBatch));
        for (int i = 0; i < kBatch; ++i) {
            const float sample = 0.15f * static_cast<float>(i + 1);
            input[static_cast<std::size_t>(i)] = sample;
            truth[static_cast<std::size_t>(i)] = swish(kTrueWeight * sample + kTrueBias);
        }
        nexus::Tensor target_tensor = nexus::Tensor::alias(truth.data(), nexus::Shape{static_cast<std::size_t>(kBatch)});
        nexus::Tensor loss_grad = nexus::Tensor::zeros(nexus::Shape{static_cast<std::size_t>(kBatch)});
        nexus_model::Tensor upstream = nexus_model::Tensor::zeros({kBatch, 1});

        auto step_loss = [&] {
            optimizer.zero_grad();
            nexus_model::Tensor hidden = layer.forward(input);
            nexus_model::Tensor prediction = activation.forward(hidden);
            nexus::Tensor predicted = nexus::Tensor::alias(prediction.data(), nexus::Shape{static_cast<std::size_t>(kBatch)});
            const double value = train_loss(predicted, target_tensor, loss_grad);
            std::memcpy(upstream.data(), loss_grad.data(), truth.size() * sizeof(float));
            nexus_model::Tensor hidden_grad = activation.backward(upstream);
            layer.backward(hidden_grad);
            optimizer.step();
            return value;
        };

        const double initial = step_loss();
        callbacks.on_step_end(0, initial);
        double final_loss = initial;
        for (int step = 1; step < 80; ++step) {
            final_loss = step_loss();
            callbacks.on_step_end(step, final_loss);
        }
        callbacks.on_train_end(final_loss);
        const float learned = layer.find_parameter("weight")->data[0];
        std::cout << "  info  loss " << initial << " -> " << final_loss << " weight=" << learned << "\n";
        expect(final_loss < initial * 0.5, "training loss decreases");
        expect(learned > 0.4f, "custom layer weight moves toward target");
        expect(counter->steps == 80, "callback sees every step");
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
