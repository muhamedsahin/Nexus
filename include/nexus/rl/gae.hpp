#pragma once

/**
 * @file gae.hpp
 * @brief Generalized Advantage Estimation.
 * terminated adimi bootstrap etmez. Son adim horizon ile bittiyse `last_value`
 * V(s_T) olarak verilir.
 */

#include <cmath>
#include <cstddef>
#include <vector>

namespace nexus {
namespace rl {

inline void compute_gae(const std::vector<float>& rewards, const std::vector<float>& values,
                        const std::vector<std::uint8_t>& terminated, float gamma, float lambda, float last_value,
                        std::vector<float>& advantages, std::vector<float>& returns) {
    const std::size_t n = rewards.size();
    advantages.assign(n, 0.0f);
    returns.assign(n, 0.0f);
    float next_value = last_value;
    float accumulator = 0.0f;
    for (std::size_t k = n; k > 0; --k) {
        const std::size_t t = k - 1;
        const float non_terminal = terminated[t] ? 0.0f : 1.0f;
        const float delta = rewards[t] + gamma * next_value * non_terminal - values[t];
        accumulator = delta + gamma * lambda * non_terminal * accumulator;
        advantages[t] = accumulator;
        returns[t] = accumulator + values[t];
        next_value = values[t];
    }
}

inline void normalize_advantages(std::vector<float>& advantages) {
    if (advantages.empty()) {
        return;
    }
    double sum = 0.0;
    for (float value : advantages) {
        sum += static_cast<double>(value);
    }
    const float mean = static_cast<float>(sum / static_cast<double>(advantages.size()));
    double var = 0.0;
    for (float value : advantages) {
        const double diff = static_cast<double>(value - mean);
        var += diff * diff;
    }
    const float inv = 1.0f / (std::sqrt(static_cast<float>(var / static_cast<double>(advantages.size()))) + 1.0e-8f);
    for (float& value : advantages) {
        value = (value - mean) * inv;
    }
}

}  // namespace rl
}  // namespace nexus
