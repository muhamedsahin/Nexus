#pragma once

/**
 * @file vector_env.hpp
 * @brief Ortam kopyalarini ayri nesnelerde kosar ve gozlemleri tek tensörde istifler.
 * Kopya sayisi 8 ve uzerinde adimlar kalici iscilere dagitilir; her adim thread acmaz.
 * Biten bolum, istenirse ayni cagri icinde yeniden baslar.
 */

#include "nexus/rl/detail_parallel.hpp"
#include "nexus/rl/environment.hpp"
#include "nexus/rl/multi_agent.hpp"

#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nexus {
namespace rl {

struct VectorStep {
    Tensor observations;
    std::vector<float> rewards;
    std::vector<std::uint8_t> terminated;
    std::vector<std::uint8_t> truncated;
};

class VectorEnv {
 public:
    explicit VectorEnv(std::vector<std::unique_ptr<Environment>> envs, bool auto_reset = true)
        : envs_(std::move(envs)), auto_reset_(auto_reset) {
        if (envs_.empty()) {
            throw std::invalid_argument("VectorEnv requires at least one environment");
        }
        observation_dim_ = envs_.front()->observation_dim();
        action_count_ = envs_.front()->action_count();
    }

    [[nodiscard]] int count() const { return static_cast<int>(envs_.size()); }
    [[nodiscard]] int observation_dim() const { return observation_dim_; }
    [[nodiscard]] int action_count() const { return action_count_; }

    Tensor reset(std::uint32_t seed) {
        const Shape shape{envs_.size(), static_cast<std::size_t>(observation_dim_)};
        if (reset_obs_.numel() != shape.numel()) {
            reset_obs_ = Tensor::zeros(shape);
        } else {
            reset_obs_.zero();
        }
        Tensor& stacked = reset_obs_;
        detail::parallel_indexed(envs_.size(), [&](std::size_t index) {
            Tensor observation = envs_[index]->reset(seed + static_cast<std::uint32_t>(index));
            copy_obs(stacked, index, observation);
        });
        return stacked;
    }

    VectorStep step(const std::vector<int>& actions) {
        if (static_cast<int>(actions.size()) != count()) {
            throw std::invalid_argument("VectorEnv action count mismatch");
        }
        const Shape shape{envs_.size(), static_cast<std::size_t>(observation_dim_)};
        if (step_batch_.observations.numel() != shape.numel()) {
            step_batch_.observations = Tensor::zeros(shape);
        } else {
            step_batch_.observations.zero();
        }
        VectorStep& batch = step_batch_;
        batch.rewards.assign(envs_.size(), 0.0f);
        batch.terminated.assign(envs_.size(), 0);
        batch.truncated.assign(envs_.size(), 0);
        detail::parallel_indexed(envs_.size(), [&](std::size_t index) {
            StepResult result = envs_[index]->step(discrete_action_tensor(actions[index]));
            batch.rewards[index] = result.reward;
            batch.terminated[index] = result.terminated ? 1 : 0;
            batch.truncated[index] = result.truncated ? 1 : 0;
            if (auto_reset_ && (result.terminated || result.truncated)) {
                Tensor fresh = envs_[index]->reset(seed_counter_ + static_cast<std::uint32_t>(index));
                copy_obs(batch.observations, index, fresh);
            } else {
                copy_obs(batch.observations, index, result.observation);
            }
        });
        seed_counter_ += static_cast<std::uint32_t>(envs_.size());
        return batch;
    }

 private:
    static void copy_obs(Tensor& stacked, std::size_t index, const Tensor& observation) {
        const std::size_t dim = observation.numel();
        float* dst = stacked.data() + index * dim;
        std::memcpy(dst, observation.data(), dim * sizeof(float));
    }

    std::vector<std::unique_ptr<Environment>> envs_;
    int observation_dim_ = 0;
    int action_count_ = 0;
    bool auto_reset_ = true;
    std::uint32_t seed_counter_ = 1000;
    Tensor reset_obs_;
    VectorStep step_batch_;
};

struct MultiVectorStep {
    std::unordered_map<AgentID, Tensor> observations;
    std::unordered_map<AgentID, std::vector<float>> rewards;
    std::vector<std::uint8_t> done;
};

class MultiAgentVectorEnv {
 public:
    explicit MultiAgentVectorEnv(std::vector<std::unique_ptr<MultiAgentEnvironment>> envs, bool auto_reset = true)
        : envs_(std::move(envs)), auto_reset_(auto_reset) {
        if (envs_.empty()) {
            throw std::invalid_argument("MultiAgentVectorEnv requires at least one environment");
        }
        agents_ = envs_.front()->agents();
    }

    [[nodiscard]] int count() const { return static_cast<int>(envs_.size()); }
    [[nodiscard]] const std::vector<AgentID>& agents() const { return agents_; }

    std::unordered_map<AgentID, Tensor> reset(std::uint32_t seed) {
        std::unordered_map<AgentID, Tensor> stacked;
        for (const AgentID& agent : agents_) {
            const int dim = envs_.front()->observation_space(agent).flat_dim();
            stacked.emplace(agent, Tensor::zeros(Shape{envs_.size(), static_cast<std::size_t>(dim)}));
        }
        detail::parallel_indexed(envs_.size(), [&](std::size_t index) {
            auto observations = envs_[index]->reset(seed + static_cast<std::uint32_t>(index));
            for (const AgentID& agent : agents_) {
                const std::size_t dim = observations.at(agent).numel();
                std::memcpy(stacked.at(agent).data() + index * dim, observations.at(agent).data(), dim * sizeof(float));
            }
        });
        return stacked;
    }

    MultiVectorStep step(const std::unordered_map<AgentID, std::vector<int>>& actions) {
        MultiVectorStep batch;
        batch.done.assign(envs_.size(), 0);
        for (const AgentID& agent : agents_) {
            const int dim = envs_.front()->observation_space(agent).flat_dim();
            batch.observations.emplace(agent, Tensor::zeros(Shape{envs_.size(), static_cast<std::size_t>(dim)}));
            batch.rewards.emplace(agent, std::vector<float>(envs_.size(), 0.0f));
        }
        detail::parallel_indexed(envs_.size(), [&](std::size_t index) {
            std::unordered_map<AgentID, Tensor> action_map;
            for (const AgentID& agent : agents_) {
                action_map.emplace(agent, discrete_action_tensor(actions.at(agent)[index]));
            }
            MultiStepResult result = envs_[index]->step(action_map);
            batch.done[index] = result.done ? 1 : 0;
            if (auto_reset_ && result.done) {
                auto fresh = envs_[index]->reset(seed_counter_ + static_cast<std::uint32_t>(index));
                for (const AgentID& agent : agents_) {
                    const std::size_t dim = fresh.at(agent).numel();
                    std::memcpy(batch.observations.at(agent).data() + index * dim, fresh.at(agent).data(), dim * sizeof(float));
                }
            } else {
                for (const AgentID& agent : agents_) {
                    const std::size_t dim = result.observations.at(agent).numel();
                    std::memcpy(batch.observations.at(agent).data() + index * dim, result.observations.at(agent).data(),
                                dim * sizeof(float));
                }
            }
            for (const AgentID& agent : agents_) {
                batch.rewards.at(agent)[index] = result.rewards.at(agent);
            }
        });
        seed_counter_ += static_cast<std::uint32_t>(envs_.size());
        return batch;
    }

 private:
    std::vector<std::unique_ptr<MultiAgentEnvironment>> envs_;
    std::vector<AgentID> agents_;
    bool auto_reset_ = true;
    std::uint32_t seed_counter_ = 2000;
};

}  // namespace rl
}  // namespace nexus
