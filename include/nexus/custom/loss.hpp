#pragma once

/**
 * @file loss.hpp
 * @brief Uc kayit yolu: acik gradyan, sayisal turev, MicroTape.
 *
 * Acik lambda hem kaybi dondurur hem `grad_out` doldurur.
 * Sayisal mod yalnizca ileri fonksiyonu ister; her koordinat icin merkezil fark alir.
 * Teyp modu ters moddur ve sonlu farkin bellek maliyetini tasimaz.
 */

#include "nexus/core/tensor.hpp"
#include "nexus/custom/microtape.hpp"

#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nexus {

class CustomLoss {
 public:
    using ExplicitFn = std::function<double(const Tensor& pred, const Tensor& target, Tensor& grad_out)>;
    using ForwardFn = std::function<double(std::span<const float> pred, std::span<const float> target)>;
    using TapeFn = std::function<MicroTape::Var(MicroTape& tape, std::span<const MicroTape::Var> pred, std::span<const float> target)>;

    enum class Mode { Explicit, Numeric, Tape };

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] Mode mode() const noexcept { return mode_; }

    double operator()(const Tensor& pred, const Tensor& target, Tensor& grad_out) const {
        if (pred.space() != MemorySpace::Host || target.space() != MemorySpace::Host || grad_out.space() != MemorySpace::Host) {
            throw TensorError("CustomLoss yalnizca host tensör kabul eder");
        }
        if (pred.numel() == 0 || pred.numel() != target.numel() || pred.numel() != grad_out.numel()) {
            throw TensorError("CustomLoss pred, target ve grad ayni uzunlukta olmali");
        }
        switch (mode_) {
            case Mode::Explicit:
                return explicit_(pred, target, grad_out);
            case Mode::Numeric:
                return numeric(pred, target, grad_out);
            case Mode::Tape:
                return tape_loss(pred, target, grad_out);
        }
        throw TensorError("CustomLoss: bilinmeyen mod");
    }

    static CustomLoss from_explicit(std::string name, ExplicitFn fn) {
        if (!fn) {
            throw TensorError("CustomLoss explicit fonksiyon bos");
        }
        CustomLoss loss;
        loss.name_ = std::move(name);
        loss.mode_ = Mode::Explicit;
        loss.explicit_ = std::move(fn);
        return loss;
    }

    static CustomLoss from_numeric(std::string name, ForwardFn fn, float epsilon = 1.0e-3f) {
        if (!fn) {
            throw TensorError("CustomLoss ileri fonksiyon bos");
        }
        if (!(epsilon > 0.0f)) {
            throw TensorError("CustomLoss epsilon pozitif olmali");
        }
        CustomLoss loss;
        loss.name_ = std::move(name);
        loss.mode_ = Mode::Numeric;
        loss.forward_ = std::move(fn);
        loss.epsilon_ = epsilon;
        return loss;
    }

    static CustomLoss from_tape(std::string name, TapeFn fn) {
        if (!fn) {
            throw TensorError("CustomLoss teyp fonksiyonu bos");
        }
        CustomLoss loss;
        loss.name_ = std::move(name);
        loss.mode_ = Mode::Tape;
        loss.tape_ = std::move(fn);
        return loss;
    }

 private:
    double numeric(const Tensor& pred, const Tensor& target, Tensor& grad_out) const {
        const std::size_t n = pred.numel();
        std::vector<float> values(pred.data(), pred.data() + n);
        const std::span<const float> target_span(target.data(), n);
        const float inv = 1.0f / (2.0f * epsilon_);
        for (std::size_t i = 0; i < n; ++i) {
            const float saved = values[i];
            values[i] = saved + epsilon_;
            const double plus = forward_(values, target_span);
            values[i] = saved - epsilon_;
            const double minus = forward_(values, target_span);
            values[i] = saved;
            grad_out[i] = static_cast<float>((plus - minus) * static_cast<double>(inv));
        }
        return forward_(std::span<const float>(pred.data(), n), target_span);
    }

    double tape_loss(const Tensor& pred, const Tensor& target, Tensor& grad_out) const {
        const std::size_t n = pred.numel();
        MicroTape tape;
        std::vector<MicroTape::Var> inputs;
        inputs.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            inputs.push_back(tape.input(pred[i]));
        }
        const MicroTape::Var root = tape_(tape, inputs, std::span<const float>(target.data(), n));
        tape.backward(root);
        for (std::size_t i = 0; i < n; ++i) {
            grad_out[i] = tape.grad(inputs[i]);
        }
        return static_cast<double>(tape.value(root));
    }

    std::string name_;
    Mode mode_ = Mode::Explicit;
    ExplicitFn explicit_;
    ForwardFn forward_;
    TapeFn tape_;
    float epsilon_ = 1.0e-3f;
};

inline CustomLoss make_loss(std::string name, CustomLoss::ExplicitFn fn) {
    return CustomLoss::from_explicit(std::move(name), std::move(fn));
}

inline CustomLoss make_loss_autograd(std::string name, CustomLoss::ForwardFn fn, float epsilon = 1.0e-3f) {
    return CustomLoss::from_numeric(std::move(name), std::move(fn), epsilon);
}

inline CustomLoss make_loss_tape(std::string name, CustomLoss::TapeFn fn) {
    return CustomLoss::from_tape(std::move(name), std::move(fn));
}

}  // namespace nexus
