#pragma once

/**
 * @file multi_agent_trainer.hpp
 * @brief N ajanli IPPO ve MAPPO.
 *
 * IPPO her ajana kendi aktorunu ve kritigini verir. MAPPO aktorleri yalnizca
 * kendi gozlemini kullanir; merkezi kritik gozlemleri art arda birlestirir.
 * Ajanlarin gozlem boyu ve eylem sayisi ayni olmalidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/rl/envs.hpp"
#include "nexus/rl/ppo.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nexus {
namespace rl {

struct MarlReport {
    float first_return = 0.0f;
    float last_return = 0.0f;
};

class MultiAgentTrainer {
 public:
    MarlReport train_independent(MultiAgentEnvironment& env, int iterations, int horizon, std::uint32_t seed) const {
        return run(env, iterations, horizon, seed, false);
    }

    MarlReport train_centralized(MultiAgentEnvironment& env, int iterations, int horizon, std::uint32_t seed) const {
        return run(env, iterations, horizon, seed, true);
    }

 private:
    struct Slot {
        PPOAgent agent;
        std::vector<float> state;
        std::vector<float> boot;
        RolloutBuffer local;

        Slot(int observation_dim, int actions, PPOAgent::Config config)
            : agent(observation_dim, actions, config), state(static_cast<std::size_t>(observation_dim)),
              boot(static_cast<std::size_t>(observation_dim)) {}
    };

    static MarlReport run(MultiAgentEnvironment& env, int iterations, int horizon, std::uint32_t seed, bool centralized) {
        const std::vector<AgentID>& names = env.agents();
        if (names.empty()) {
            throw std::invalid_argument("MultiAgentTrainer requires at least one agent");
        }
        const int obs = env.observation_space(names.front()).flat_dim();
        const int actions = env.action_space(names.front()).n();
        if (obs <= 0 || actions <= 1) {
            throw std::invalid_argument("MultiAgentTrainer spaces are invalid");
        }
        for (const AgentID& name : names) {
            if (env.observation_space(name).flat_dim() != obs || env.action_space(name).n() != actions) {
                throw std::invalid_argument("MultiAgentTrainer agents must share observation and action size");
            }
        }
        const int count = static_cast<int>(names.size());
        PPOAgent::Config config;
        config.hidden = 32;
        config.learning_rate = 1.0e-2f;
        config.epochs = 3;
        config.entropy_coef = 0.02f;
        std::vector<Slot> slots;
        slots.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
            config.seed = seed + static_cast<std::uint32_t>(index);
            slots.emplace_back(obs, actions, config);
        }
        std::unique_ptr<PPOAgent> critic;
        if (centralized) {
            config.seed = seed + static_cast<std::uint32_t>(count);
            critic = std::make_unique<PPOAgent>(obs * count, actions, config);
        }

        auto observations = env.reset(seed);
        for (int index = 0; index < count; ++index) {
            copy_obs(slots[static_cast<std::size_t>(index)].state, observations.at(names[static_cast<std::size_t>(index)]));
        }

        MarlReport report;
        const int joint_dim = obs * count;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (Slot& slot : slots) {
                slot.local.reset(horizon, obs);
            }
            RolloutBuffer joint;
            if (centralized) {
                joint.reset(horizon, joint_dim);
            }
            float episode = 0.0f;
            std::vector<float> returns;
            bool bootstrap = true;
            for (int step = 0; step < horizon; ++step) {
                std::unordered_map<AgentID, Tensor> action_map;
                std::vector<float> joint_obs(static_cast<std::size_t>(joint_dim));
                float central_value = 0.0f;
                for (int index = 0; index < count; ++index) {
                    Slot& slot = slots[static_cast<std::size_t>(index)];
                    std::memcpy(joint_obs.data() + static_cast<std::size_t>(index * obs), slot.state.data(),
                                slot.state.size() * sizeof(float));
                }
                if (centralized) {
                    central_value = critic->value_of(joint_obs.data());
                }
                std::vector<int> chosen(static_cast<std::size_t>(count));
                std::vector<float> log_probs(static_cast<std::size_t>(count));
                std::vector<float> values(static_cast<std::size_t>(count));
                for (int index = 0; index < count; ++index) {
                    Slot& slot = slots[static_cast<std::size_t>(index)];
                    chosen[static_cast<std::size_t>(index)] =
                        slot.agent.act(slot.state.data(), log_probs[static_cast<std::size_t>(index)],
                                       values[static_cast<std::size_t>(index)], false);
                    action_map.emplace(names[static_cast<std::size_t>(index)],
                                       discrete_action_tensor(chosen[static_cast<std::size_t>(index)]));
                }
                MultiStepResult result = env.step(action_map);
                float reward_sum = 0.0f;
                bool ended = result.done;
                for (int index = 0; index < count; ++index) {
                    const AgentID& name = names[static_cast<std::size_t>(index)];
                    reward_sum += result.rewards.at(name);
                    ended = ended || result.terminated.at(name) || result.truncated.at(name);
                }
                const float reward = reward_sum / static_cast<float>(count);
                for (int index = 0; index < count; ++index) {
                    Slot& slot = slots[static_cast<std::size_t>(index)];
                    const float logged_value = centralized ? central_value : values[static_cast<std::size_t>(index)];
                    slot.local.add(slot.state.data(), chosen[static_cast<std::size_t>(index)],
                                   log_probs[static_cast<std::size_t>(index)], logged_value, reward, ended);
                    copy_obs(slot.boot, result.observations.at(names[static_cast<std::size_t>(index)]));
                }
                if (centralized) {
                    joint.add(joint_obs.data(), 0, 0.0f, central_value, reward, ended);
                }
                episode += reward;
                bootstrap = !ended;
                if (result.done || ended) {
                    returns.push_back(episode);
                    episode = 0.0f;
                    auto restarted = env.reset(seed + static_cast<std::uint32_t>(iteration * horizon + step + 3));
                    for (int index = 0; index < count; ++index) {
                        copy_obs(slots[static_cast<std::size_t>(index)].state,
                                 restarted.at(names[static_cast<std::size_t>(index)]));
                    }
                } else {
                    for (Slot& slot : slots) {
                        slot.state.swap(slot.boot);
                        slot.boot = slot.state;
                    }
                }
            }
            const float mean = mean_of(returns);
            if (iteration == 0) {
                report.first_return = mean;
            }
            report.last_return = mean;
            if (centralized) {
                std::vector<float> joint_boot(static_cast<std::size_t>(joint_dim));
                for (int index = 0; index < count; ++index) {
                    const Slot& slot = slots[static_cast<std::size_t>(index)];
                    std::memcpy(joint_boot.data() + static_cast<std::size_t>(index * obs), slot.boot.data(),
                                slot.boot.size() * sizeof(float));
                }
                const float bootstrap_value = bootstrap ? critic->value_of(joint_boot.data()) : 0.0f;
                joint.finish(0.99f, 0.95f, bootstrap_value);
                for (Slot& slot : slots) {
                    slot.agent.update_policy(slot.local.observations(), slot.local.actions().data(), slot.local.log_probs().data(),
                                             joint.advantages().data(), slot.local.size());
                }
                critic->update_value(joint.observations(), joint.returns().data(), joint.values().data(), joint.size());
            } else {
                for (Slot& slot : slots) {
                    const float bootstrap_value = bootstrap ? slot.agent.value_of(slot.boot.data()) : 0.0f;
                    slot.local.finish(0.99f, 0.95f, bootstrap_value);
                    slot.agent.update_policy(slot.local.observations(), slot.local.actions().data(), slot.local.log_probs().data(),
                                             slot.local.advantages().data(), slot.local.size());
                    slot.agent.update_value(slot.local.observations(), slot.local.returns().data(), slot.local.values().data(),
                                            slot.local.size());
                }
            }
        }
        return report;
    }

    static void copy_obs(std::vector<float>& dst, const Tensor& observation) {
        if (observation.numel() != dst.size()) {
            throw std::invalid_argument("MultiAgentTrainer observation size mismatch");
        }
        std::memcpy(dst.data(), observation.data(), dst.size() * sizeof(float));
    }

    static float mean_of(const std::vector<float>& values) {
        if (values.empty()) {
            return 0.0f;
        }
        double sum = 0.0;
        for (float value : values) {
            sum += value;
        }
        return static_cast<float>(sum / static_cast<double>(values.size()));
    }
};

}  // namespace rl
}  // namespace nexus
