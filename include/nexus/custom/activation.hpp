#pragma once

/**
 * @file activation.hpp
 * @brief Kullanicinin iki satirdaki lambda ile tanimladigi aktivasyon.
 *
 * Geri turev verilmezse merkezil sonlu fark kullanilir:
 *   f'(x) ~= (f(x+eps) - f(x-eps)) / (2 eps)
 * Uzunluk esigi asilinca ileri ve geri gecis cekirdek sayisina bolunur.
 * Batch kancasi varsa o, skaler lambdanin yerini alir. Kullanici bu kancanin
 * icinde AVX2 yazabilir; yardimci `activation_simd.hpp` icindedir.
 *
 * Modul, diger NexusModel katmanlari gibi girisi onbellekler. Ayni nesne
 * uzerinde eszamanli forward tanimsizdir.
 */

#include "nexus/custom/detail/parallel.hpp"
#include "nexus_model/core/module.hpp"

#include <cmath>
#include <functional>
#include <span>
#include <string>
#include <utility>

namespace nexus {

class CustomActivation : public nexus_model::Module {
 public:
    using ScalarForward = std::function<float(float)>;
    using ScalarBackward = std::function<float(float x, float grad_out)>;
    using BatchForward = std::function<void(std::span<const float> in, std::span<float> out)>;
    using BatchBackward = std::function<void(std::span<const float> x, std::span<const float> grad_out, std::span<float> grad_in)>;

    static constexpr std::size_t kParallelThreshold = 2048;
    static constexpr float kFiniteDifferenceStep = 1.0e-3f;

    CustomActivation(std::string name, ScalarForward forward, ScalarBackward backward = {})
        : name_(std::move(name)), forward_(std::move(forward)), backward_(std::move(backward)) {
        if (!forward_) {
            throw nexus_model::ModelError("CustomActivation forward is empty");
        }
    }

    CustomActivation(const CustomActivation&) = delete;
    CustomActivation& operator=(const CustomActivation&) = delete;
    CustomActivation(CustomActivation&&) noexcept = default;
    CustomActivation& operator=(CustomActivation&&) noexcept = default;

    /// Span kancasi skaler lambdayi gecer. Bos geri kanca skaler/sonlu farka duser.
    void set_batch(BatchForward forward, BatchBackward backward = {}) {
        if (!forward) {
            throw nexus_model::ModelError("CustomActivation batch forward is empty");
        }
        batch_forward_ = std::move(forward);
        batch_backward_ = std::move(backward);
    }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] bool has_analytic_backward() const noexcept { return static_cast<bool>(backward_); }
    [[nodiscard]] bool has_batch_forward() const noexcept { return static_cast<bool>(batch_forward_); }

    nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
        cache_.copy_from(input);
        output_.resize_like(input);
        forward_buffer(cache_.data(), output_.data(), input.numel());
        return output_;
    }

    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override {
        if (grad_output.numel() != cache_.numel()) {
            throw nexus_model::ModelError("CustomActivation backward size mismatch");
        }
        grad_input_.resize_like(cache_);
        backward_buffer(cache_.data(), grad_output.data(), grad_input_.data(), cache_.numel());
        return grad_input_;
    }

    void forward_buffer(const float* in, float* out, std::size_t n) const {
        if (batch_forward_) {
            batch_forward_(std::span<const float>(in, n), std::span<float>(out, n));
            return;
        }
        const ScalarForward& fn = forward_;
        detail::parallel_for_range(n, kParallelThreshold, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                out[i] = fn(in[i]);
            }
        });
    }

    void backward_buffer(const float* x, const float* grad_out, float* grad_in, std::size_t n) const {
        if (batch_backward_) {
            batch_backward_(std::span<const float>(x, n), std::span<const float>(grad_out, n), std::span<float>(grad_in, n));
            return;
        }
        if (backward_) {
            const ScalarBackward& fn = backward_;
            detail::parallel_for_range(n, kParallelThreshold, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    grad_in[i] = fn(x[i], grad_out[i]);
                }
            });
            return;
        }
        const ScalarForward& fn = forward_;
        constexpr float step = kFiniteDifferenceStep;
        constexpr float inv = 1.0f / (2.0f * step);
        detail::parallel_for_range(n, kParallelThreshold, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                const float slope = (fn(x[i] + step) - fn(x[i] - step)) * inv;
                grad_in[i] = grad_out[i] * slope;
            }
        });
    }

 private:
    std::string name_;
    ScalarForward forward_;
    ScalarBackward backward_;
    BatchForward batch_forward_;
    BatchBackward batch_backward_;
    nexus_model::Tensor cache_;
    nexus_model::Tensor output_;
    nexus_model::Tensor grad_input_;
};

/// Ileri ve istege bagli geri. Geri bos birakilirsa sonlu fark devreye girer.
inline CustomActivation make_activation(std::string name, CustomActivation::ScalarForward forward,
                                        CustomActivation::ScalarBackward backward = {}) {
    return CustomActivation(std::move(name), std::move(forward), std::move(backward));
}

}  // namespace nexus
