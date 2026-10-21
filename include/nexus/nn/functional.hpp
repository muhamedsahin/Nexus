#pragma once

/**
 * @file functional.hpp
 * @brief Durumsuz eleman fonksiyonlari. Katman tamponu tutmaz.
 */

#include <cmath>
#include <cstddef>

namespace nexus {
namespace nn {
namespace functional {

inline void relu(const float* input, float* output, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

inline void silu(const float* input, float* output, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        output[i] = input[i] * sigmoid;
    }
}

inline void softmax(const float* logits, float* probabilities, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const float* source = logits + static_cast<std::size_t>(row * cols);
        float* destination = probabilities + static_cast<std::size_t>(row * cols);
        float peak = source[0];
        for (int col = 1; col < cols; ++col) {
            peak = source[col] > peak ? source[col] : peak;
        }
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            destination[col] = std::exp(source[col] - peak);
            sum += destination[col];
        }
        const float inv = 1.0f / sum;
        for (int col = 0; col < cols; ++col) {
            destination[col] *= inv;
        }
    }
}

}  // namespace functional
}  // namespace nn
}  // namespace nexus
