#pragma once

/**
 * @file prioritized_replay.hpp
 * @brief Toplam agaci. Ornekleme O(log N). Oncelik (|TD| + eps)^alpha.
 * Importance sampling agirligi batch icindeki en buyuge normalize edilir.
 */

#include "nexus/rl/aligned_alloc.hpp"
#include "nexus/rl/rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nexus {
namespace rl {

class PrioritizedReplayBuffer {
 public:
    PrioritizedReplayBuffer(int capacity, int observation_dim, float alpha)
        : capacity_(capacity), observation_dim_(observation_dim), alpha_(alpha),
          observations_(static_cast<std::size_t>(capacity) * observation_dim),
          next_observations_(static_cast<std::size_t>(capacity) * observation_dim), actions_(static_cast<std::size_t>(capacity)),
          rewards_(static_cast<std::size_t>(capacity)), terminated_(static_cast<std::size_t>(capacity)),
          tree_(static_cast<std::size_t>(2 * capacity), 0.0f) {
        if (capacity <= 1 || observation_dim <= 0 || !(alpha >= 0.0f)) {
            throw std::invalid_argument("PER capacity, dim, or alpha is invalid");
        }
    }

    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] float total() const noexcept { return tree_[1]; }
    [[nodiscard]] int observation_dim() const noexcept { return observation_dim_; }
    [[nodiscard]] const float* observation_data() const noexcept { return observations_.data(); }

    void add(const float* observation, int action, float reward, const float* next_observation, bool terminated, float priority) {
        write(cursor_, observation, action, reward, next_observation, terminated);
        set_priority(cursor_, priority);
        cursor_ = (cursor_ + 1) % capacity_;
        if (size_ < capacity_) {
            ++size_;
        }
    }

    struct Sample {
        std::vector<int> indices;
        std::vector<float> weights;
        std::vector<float> observations;
        std::vector<float> next_observations;
        std::vector<int> actions;
        std::vector<float> rewards;
        std::vector<std::uint8_t> terminated;
    };

    Sample sample(int batch, float beta, Rng& rng) const {
        if (size_ < batch) {
            throw std::invalid_argument("PER batch exceeds the number of stored transitions");
        }
        Sample out;
        out.indices.resize(static_cast<std::size_t>(batch));
        out.weights.resize(static_cast<std::size_t>(batch));
        out.observations.resize(static_cast<std::size_t>(batch * observation_dim_));
        out.next_observations.resize(static_cast<std::size_t>(batch * observation_dim_));
        out.actions.resize(static_cast<std::size_t>(batch));
        out.rewards.resize(static_cast<std::size_t>(batch));
        out.terminated.resize(static_cast<std::size_t>(batch));
        const float mass = std::max(total(), 1.0e-8f);
        float max_weight = 0.0f;
        for (int row = 0; row < batch; ++row) {
            const int index = sample_leaf(rng.uniform() * mass);
            out.indices[static_cast<std::size_t>(row)] = index;
            const float probability = tree_[static_cast<std::size_t>(capacity_ + index)] / mass;
            const float weight = std::pow(static_cast<float>(size_) * std::max(probability, 1.0e-8f), -beta);
            out.weights[static_cast<std::size_t>(row)] = weight;
            max_weight = std::max(max_weight, weight);
            copy_row(index, row, out);
        }
        if (max_weight > 0.0f) {
            for (float& weight : out.weights) {
                weight /= max_weight;
            }
        }
        return out;
    }

    void update_priorities(const std::vector<int>& indices, const std::vector<float>& absolute_td) {
        for (std::size_t i = 0; i < indices.size(); ++i) {
            set_priority(indices[i], absolute_td[i]);
        }
    }

 private:
    void write(int index, const float* observation, int action, float reward, const float* next_observation, bool terminated) {
        const std::size_t slot = static_cast<std::size_t>(index);
        const std::size_t bytes = static_cast<std::size_t>(observation_dim_) * sizeof(float);
        std::memcpy(observations_.data() + slot * observation_dim_, observation, bytes);
        std::memcpy(next_observations_.data() + slot * observation_dim_, next_observation, bytes);
        actions_[slot] = action;
        rewards_[slot] = reward;
        terminated_[slot] = terminated ? 1 : 0;
    }

    void copy_row(int index, int row, Sample& out) const {
        const std::size_t bytes = static_cast<std::size_t>(observation_dim_) * sizeof(float);
        const std::size_t slot = static_cast<std::size_t>(index);
        const std::size_t dst = static_cast<std::size_t>(row) * observation_dim_;
        std::memcpy(out.observations.data() + dst, observations_.data() + slot * observation_dim_, bytes);
        std::memcpy(out.next_observations.data() + dst, next_observations_.data() + slot * observation_dim_, bytes);
        out.actions[static_cast<std::size_t>(row)] = actions_[slot];
        out.rewards[static_cast<std::size_t>(row)] = rewards_[slot];
        out.terminated[static_cast<std::size_t>(row)] = terminated_[slot];
    }

    void set_priority(int index, float priority) {
        const float scaled = std::pow(std::fabs(priority) + 1.0e-5f, alpha_);
        int node = capacity_ + index;
        tree_[static_cast<std::size_t>(node)] = scaled;
        while (node > 1) {
            node /= 2;
            tree_[static_cast<std::size_t>(node)] =
                tree_[static_cast<std::size_t>(2 * node)] + tree_[static_cast<std::size_t>(2 * node + 1)];
        }
    }

    [[nodiscard]] int sample_leaf(float mass) const {
        int node = 1;
        while (node < capacity_) {
            const int left = node * 2;
            const float left_mass = tree_[static_cast<std::size_t>(left)];
            if (mass <= left_mass) {
                node = left;
            } else {
                mass -= left_mass;
                node = left + 1;
            }
        }
        int index = node - capacity_;
        if (index < 0 || index >= size_ || tree_[static_cast<std::size_t>(capacity_ + index)] <= 0.0f) {
            for (int live = 0; live < size_; ++live) {
                if (tree_[static_cast<std::size_t>(capacity_ + live)] > 0.0f) {
                    return live;
                }
            }
            return 0;
        }
        return index;
    }

    int capacity_ = 0;
    int observation_dim_ = 0;
    float alpha_ = 0.6f;
    int size_ = 0;
    int cursor_ = 0;
    AlignedVector<float> observations_;
    AlignedVector<float> next_observations_;
    std::vector<int> actions_;
    std::vector<float> rewards_;
    std::vector<std::uint8_t> terminated_;
    std::vector<float> tree_;
};

}  // namespace rl
}  // namespace nexus
