#pragma once

/**
 * @file replay_buffer.hpp
 * @brief Sabit kapasiteli halka tampon. Gozlem baytlari 64 bayt hizalidir.
 * Doluysa en eski gecis ezilir.
 */

#include "nexus/rl/aligned_alloc.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nexus {
namespace rl {

class ReplayBuffer {
 public:
    ReplayBuffer(int capacity, int observation_dim)
        : capacity_(capacity), observation_dim_(observation_dim), observations_(static_cast<std::size_t>(capacity) * observation_dim),
          next_observations_(static_cast<std::size_t>(capacity) * observation_dim), actions_(static_cast<std::size_t>(capacity)),
          rewards_(static_cast<std::size_t>(capacity)), terminated_(static_cast<std::size_t>(capacity)) {
        if (capacity <= 0 || observation_dim <= 0) {
            throw std::invalid_argument("ReplayBuffer capacity and observation dim must be positive");
        }
    }

    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] int observation_dim() const noexcept { return observation_dim_; }
    [[nodiscard]] int capacity() const noexcept { return capacity_; }
    [[nodiscard]] const float* observation_data() const noexcept { return observations_.data(); }

    void add(const float* observation, int action, float reward, const float* next_observation, bool terminated) {
        const std::size_t slot = static_cast<std::size_t>(cursor_);
        std::memcpy(observations_.data() + slot * observation_dim_, observation, static_cast<std::size_t>(observation_dim_) * sizeof(float));
        std::memcpy(next_observations_.data() + slot * observation_dim_, next_observation,
                    static_cast<std::size_t>(observation_dim_) * sizeof(float));
        actions_[slot] = action;
        rewards_[slot] = reward;
        terminated_[slot] = terminated ? 1 : 0;
        cursor_ = (cursor_ + 1) % capacity_;
        if (size_ < capacity_) {
            ++size_;
        }
    }

    void sample(const std::vector<int>& indices, float* observations, float* next_observations, int* actions, float* rewards,
                std::uint8_t* terminated) const {
        for (std::size_t row = 0; row < indices.size(); ++row) {
            const int index = indices[row];
            if (index < 0 || index >= size_) {
                throw std::invalid_argument("ReplayBuffer sample index is outside the live range");
            }
            const std::size_t slot = static_cast<std::size_t>(index);
            std::memcpy(observations + row * observation_dim_, observations_.data() + slot * observation_dim_,
                        static_cast<std::size_t>(observation_dim_) * sizeof(float));
            std::memcpy(next_observations + row * observation_dim_, next_observations_.data() + slot * observation_dim_,
                        static_cast<std::size_t>(observation_dim_) * sizeof(float));
            actions[row] = actions_[slot];
            rewards[row] = rewards_[slot];
            terminated[row] = terminated_[slot];
        }
    }

    [[nodiscard]] int action_at(int index) const { return actions_[static_cast<std::size_t>(index)]; }

 private:
    int capacity_ = 0;
    int observation_dim_ = 0;
    int size_ = 0;
    int cursor_ = 0;
    AlignedVector<float> observations_;
    AlignedVector<float> next_observations_;
    std::vector<int> actions_;
    std::vector<float> rewards_;
    std::vector<std::uint8_t> terminated_;
};

}  // namespace rl
}  // namespace nexus
