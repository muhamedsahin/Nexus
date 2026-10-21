#pragma once

/**
 * @file actor_critic.hpp
 * @brief Kesikli A2C, PPO'nun kirpmasiz ve tek epokluk halidir.
 * Surekli A2C, Gauss politikasinin ortalamasini avantaja gore ceker.
 */

#include "nexus/rl/gae.hpp"
#include "nexus/rl/ppo.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexus {
namespace rl {

class ActorCriticAgent {
 public:
    static ActorCriticAgent discrete(int observation_dim, int actions, std::uint32_t seed) {
        PPOAgent::Config config;
        config.seed = seed;
        config.epochs = 1;
        config.clip_epsilon = 10.0f;
        config.entropy_coef = 0.01f;
        config.learning_rate = 1.0e-2f;
        config.hidden = 32;
        return ActorCriticAgent(observation_dim, actions, config);
    }

    int act(const float* observation, float& log_prob, float& value, bool greedy) {
        return agent_.act(observation, log_prob, value, greedy);
    }
    float learn(Environment& env, int iterations, int horizon) { return agent_.learn(env, iterations, horizon); }
    float evaluate(Environment& env, int episodes) { return agent_.evaluate(env, episodes); }
    void save_nxs(const std::string& path) const { agent_.save_nxs(path); }
    [[nodiscard]] static ActorCriticAgent load_nxs(const std::string& path) {
        PPOAgent agent = PPOAgent::load_nxs(path);
        return ActorCriticAgent(std::move(agent));
    }

 private:
    explicit ActorCriticAgent(PPOAgent agent) : agent_(std::move(agent)) {}
    ActorCriticAgent(int observation_dim, int actions, PPOAgent::Config config) : agent_(observation_dim, actions, config) {}

    PPOAgent agent_;
};

class ContinuousActorCritic {
 public:
    ContinuousActorCritic(int observation_dim, std::uint32_t seed)
        : rng_(seed), mean_({observation_dim, 32, 1}, seed), critic_({observation_dim, 32, 1}, seed + 3) {}

    float act(const float* observation, bool greedy) {
        float mean = 0.0f;
        mean_.forward(observation, 1, &mean);
        if (greedy) {
            return std::tanh(mean);
        }
        return std::clamp(std::tanh(mean + 0.3f * rng_.normal()), -1.0f, 1.0f);
    }

    float learn(Environment& env, int iterations, int horizon) {
        float last = 0.0f;
        Tensor observation = env.reset(7);
        std::vector<float> current(1);
        std::memcpy(current.data(), observation.data(), sizeof(float));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            std::vector<float> observations(static_cast<std::size_t>(horizon));
            std::vector<float> actions(static_cast<std::size_t>(horizon));
            std::vector<float> rewards(static_cast<std::size_t>(horizon));
            std::vector<float> values(static_cast<std::size_t>(horizon));
            std::vector<std::uint8_t> terminated(static_cast<std::size_t>(horizon), 0);
            float episode = 0.0f;
            bool bootstrap = true;
            float bootstrap_state = current[0];
            for (int step = 0; step < horizon; ++step) {
                float mean = 0.0f;
                float value = 0.0f;
                mean_.forward(current.data(), 1, &mean);
                critic_.forward(current.data(), 1, &value);
                const float noise = rng_.normal();
                const float action = std::clamp(std::tanh(mean + 0.3f * noise), -1.0f, 1.0f);
                const float stored[1] = {action};
                StepResult result = env.step(Tensor(Shape{1}, std::span<const float>(stored, 1)));
                observations[static_cast<std::size_t>(step)] = current[0];
                actions[static_cast<std::size_t>(step)] = action;
                rewards[static_cast<std::size_t>(step)] = result.reward;
                values[static_cast<std::size_t>(step)] = value;
                terminated[static_cast<std::size_t>(step)] = result.terminated ? 1 : 0;
                episode += result.reward;
                bootstrap = !result.terminated;
                bootstrap_state = result.observation[0];
                if (result.terminated || result.truncated) {
                    last = episode;
                    episode = 0.0f;
                    Tensor restarted = env.reset(8u + static_cast<std::uint32_t>(iteration * horizon + step));
                    current[0] = restarted[0];
                } else {
                    current[0] = result.observation[0];
                }
            }
            float last_value = 0.0f;
            if (bootstrap) {
                critic_.forward(&bootstrap_state, 1, &last_value);
            }
            std::vector<float> advantages;
            std::vector<float> returns;
            compute_gae(rewards, values, terminated, 0.99f, 0.95f, last_value, advantages, returns);
            normalize_advantages(advantages);
            std::vector<float> mean_grad(static_cast<std::size_t>(horizon), 0.0f);
            std::vector<float> means(static_cast<std::size_t>(horizon));
            mean_.forward(observations.data(), horizon, means.data());
            constexpr float kStd = 0.3f;
            for (int step = 0; step < horizon; ++step) {
                const float mean = means[static_cast<std::size_t>(step)];
                const float action = actions[static_cast<std::size_t>(step)];
                const float d_log = (action - std::tanh(mean)) / (kStd * kStd);
                mean_grad[static_cast<std::size_t>(step)] = -advantages[static_cast<std::size_t>(step)] * d_log / static_cast<float>(horizon);
            }
            mean_.zero_grad();
            mean_.backward(mean_grad.data());
            mean_.step(1.0e-2f);
            std::vector<float> predicted(static_cast<std::size_t>(horizon));
            critic_.forward(observations.data(), horizon, predicted.data());
            std::vector<float> value_grad(static_cast<std::size_t>(horizon));
            for (int step = 0; step < horizon; ++step) {
                value_grad[static_cast<std::size_t>(step)] =
                    (predicted[static_cast<std::size_t>(step)] - returns[static_cast<std::size_t>(step)]) / static_cast<float>(horizon);
            }
            critic_.zero_grad();
            critic_.backward(value_grad.data());
            critic_.step(1.0e-2f);
        }
        return last;
    }

    void save_nxs(const std::string& path) const {
        nxs::Archive archive;
        archive.has_rl = true;
        archive.rl.kind = "continuous_actor_critic";
        archive.rl.nets = {mean_.export_blob("mean"), critic_.export_blob("critic")};
        nxs::save_archive(path, archive);
    }

    [[nodiscard]] static ContinuousActorCritic load_nxs(const std::string& path) {
        const nxs::Archive archive = nxs::load_archive(path);
        if (!archive.has_rl || archive.rl.kind != "continuous_actor_critic" || archive.rl.nets.size() < 2) {
            throw std::runtime_error("nxs file is not a continuous actor critic");
        }
        ContinuousActorCritic agent(archive.rl.nets[0].sizes.empty() ? 1 : archive.rl.nets[0].sizes.front(), 1);
        agent.mean_.import_blob(archive.rl.nets[0]);
        agent.critic_.import_blob(archive.rl.nets[1]);
        return agent;
    }

    float evaluate(Environment& env, int episodes) {
        double sum = 0.0;
        for (int episode = 0; episode < episodes; ++episode) {
            Tensor observation = env.reset(20u + static_cast<std::uint32_t>(episode));
            float state = observation[0];
            float reward_sum = 0.0f;
            for (int step = 0; step < 10; ++step) {
                float mean = 0.0f;
                mean_.forward(&state, 1, &mean);
                const float action = std::tanh(mean);
                const float stored[1] = {action};
                StepResult result = env.step(Tensor(Shape{1}, std::span<const float>(stored, 1)));
                reward_sum += result.reward;
                if (result.terminated || result.truncated) {
                    break;
                }
                state = result.observation[0];
            }
            sum += reward_sum;
        }
        return static_cast<float>(sum / static_cast<double>(episodes));
    }

 private:
    Rng rng_;
    Mlp mean_;
    Mlp critic_;
};

}  // namespace rl
}  // namespace nexus
