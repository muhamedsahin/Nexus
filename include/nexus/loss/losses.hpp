#pragma once

/**
 * @file losses.hpp
 * @brief Batch cross-entropy ve MSE. Gradyan cagiranin tamponuna yazilir.
 * Cross-entropy logit bekler. Mean indirgeme gradyani 1/N ile boler.
 */

#include "nexusloss/losses/classification.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace nexus {
namespace loss {

class CrossEntropy {
 public:
    explicit CrossEntropy(float label_smoothing = 0.0f) : smoothing_(label_smoothing) {
        if (!(label_smoothing >= 0.0f) || label_smoothing >= 1.0f) {
            throw std::invalid_argument("CrossEntropy label smoothing must be in [0, 1)");
        }
    }

    [[nodiscard]] float label_smoothing() const noexcept { return smoothing_; }

    float forward_backward(const float* logits, const int* targets, int batch, int classes, float* grad) const {
        if (batch <= 0 || classes <= 1) {
            throw std::invalid_argument("CrossEntropy batch and class count are invalid");
        }
        nexusloss::classification::CrossEntropyLoss<float> loss(smoothing_);
        double sum = 0.0;
        for (int row = 0; row < batch; ++row) {
            const std::span<const float> row_logits(logits + static_cast<std::size_t>(row * classes), static_cast<std::size_t>(classes));
            const int target = targets[row];
            if (target < 0 || target >= classes) {
                throw std::invalid_argument("CrossEntropy target class is outside the logit width");
            }
            sum += static_cast<double>(loss.forward(row_logits, static_cast<std::size_t>(target)));
            loss.backward_into(std::span<float>(grad + static_cast<std::size_t>(row * classes), static_cast<std::size_t>(classes)));
        }
        const float inv = 1.0f / static_cast<float>(batch);
        for (int i = 0; i < batch * classes; ++i) {
            grad[i] *= inv;
        }
        return static_cast<float>(sum / static_cast<double>(batch));
    }

 private:
    float smoothing_ = 0.0f;
};

class MSE {
 public:
    float forward_backward(const float* prediction, const float* target, int n, float* grad) const {
        if (n <= 0) {
            throw std::invalid_argument("MSE requires a non-empty batch");
        }
        double sum = 0.0;
        const float inv = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) {
            const float diff = prediction[i] - target[i];
            sum += static_cast<double>(diff) * diff;
            grad[i] = 2.0f * diff * inv;
        }
        return static_cast<float>(sum / static_cast<double>(n));
    }
};

}  // namespace loss
}  // namespace nexus
