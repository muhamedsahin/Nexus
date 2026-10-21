#pragma once

/**
 * @file rollout_buffer.hpp
 * @brief On-policy toplama. GAE bu tamponun odul, deger ve bitis bayraklarina bakar.
 */

#include "nexus/rl/gae.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nexus {
namespace rl {

class RolloutBuffer {
 public:
    void reset(int horizon, int observation_dim) {
        if (horizon <= 0 || observation_dim <= 0) {
            throw std::invalid_argument("RolloutBuffer horizon and observation dim must be positive");
        }
        horizon_ = horizon;
        observation_dim_ = observation_dim;
        cursor_ = 0;
        observations_.assign(static_cast<std::size_t>(horizon * observation_dim), 0.0f);
        actions_.assign(static_cast<std::size_t>(horizon), 0);
        log_probs_.assign(static_cast<std::size_t>(horizon), 0.0f);
        values_.assign(static_cast<std::size_t>(horizon), 0.0f);
        rewards_.assign(static_cast<std::size_t>(horizon), 0.0f);
        terminated_.assign(static_cast<std::size_t>(horizon), 0);
        advantages_.assign(static_cast<std::size_t>(horizon), 0.0f);
        returns_.assign(static_cast<std::size_t>(horizon), 0.0f);
    }

    void add(const float* observation, int action, float log_prob, float value, float reward, bool terminated) {
        if (cursor_ >= horizon_) {
            throw std::logic_error("RolloutBuffer is full");
        }
        std::memcpy(observations_.data() + static_cast<std::size_t>(cursor_ * observation_dim_), observation,
                    static_cast<std::size_t>(observation_dim_) * sizeof(float));
        actions_[static_cast<std::size_t>(cursor_)] = action;
        log_probs_[static_cast<std::size_t>(cursor_)] = log_prob;
        values_[static_cast<std::size_t>(cursor_)] = value;
        rewards_[static_cast<std::size_t>(cursor_)] = reward;
        terminated_[static_cast<std::size_t>(cursor_)] = terminated ? 1 : 0;
        ++cursor_;
    }

    void finish(float gamma, float lambda, float last_value) {
        compute_gae(rewards_, values_, terminated_, gamma, lambda, last_value, advantages_, returns_);
        normalize_advantages(advantages_);
    }

    [[nodiscard]] int size() const noexcept { return cursor_; }
    [[nodiscard]] int observation_dim() const noexcept { return observation_dim_; }
    [[nodiscard]] const float* observations() const noexcept { return observations_.data(); }
    [[nodiscard]] const std::vector<int>& actions() const noexcept { return actions_; }
    [[nodiscard]] const std::vector<float>& log_probs() const noexcept { return log_probs_; }
    [[nodiscard]] const std::vector<float>& values() const noexcept { return values_; }
    [[nodiscard]] const std::vector<float>& advantages() const noexcept { return advantages_; }
    [[nodiscard]] const std::vector<float>& returns() const noexcept { return returns_; }

 private:
    int horizon_ = 0;
    int observation_dim_ = 0;
    int cursor_ = 0;
    std::vector<float> observations_;
    std::vector<int> actions_;
    std::vector<float> log_probs_;
    std::vector<float> values_;
    std::vector<float> rewards_;
    std::vector<std::uint8_t> terminated_;
    std::vector<float> advantages_;
    std::vector<float> returns_;
};

}  // namespace rl
}  // namespace nexus
