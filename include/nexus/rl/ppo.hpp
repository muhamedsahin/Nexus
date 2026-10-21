#pragma once

/**
 * @file ppo.hpp
 * @brief Kesikli PPO. Kirpilmis vekil amac, deger kirpmasi, entropi ve normalize GAE.
 * `clip_epsilon` cok buyuk verilirse kirpma kapanir; A2C bu yolu kullanir.
 */

#include "nexus/model/nxs_format.hpp"
#include "nexus/rl/environment.hpp"
#include "nexus/rl/mlp.hpp"
#include "nexus/rl/rollout_buffer.hpp"

#include <string>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace nexus {
namespace rl {

class PPOAgent {
 public:
    struct Config {
        int hidden = 32;
        float gamma = 0.99f;
        float lambda = 0.95f;
        float clip_epsilon = 0.2f;
        float learning_rate = 1.0e-2f;
        float entropy_coef = 0.01f;
        float value_coef = 0.5f;
        int epochs = 4;
        std::uint32_t seed = 1;
    };

    PPOAgent(int observation_dim, int actions, Config config = {})
        : observation_dim_(observation_dim), actions_(actions), config_(config), rng_(config.seed),
          actor_({observation_dim, config.hidden, actions}, config.seed),
          critic_({observation_dim, config.hidden, 1}, config.seed + 17) {
        if (observation_dim <= 0 || actions <= 1) {
            throw std::invalid_argument("PPOAgent dimensions are invalid");
        }
    }

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] std::vector<float> policy_parameters() const { return actor_.parameters(); }
    void set_policy_parameters(const std::vector<float>& values) { actor_.set_parameters(values); }

    void save_nxs(const std::string& path) const {
        nxs::Archive archive;
        archive.has_rl = true;
        archive.rl.kind = "ppo";
        archive.rl.config = {nxs::i32("observation", observation_dim_),
                             nxs::i32("actions", actions_),
                             nxs::i32("hidden", config_.hidden),
                             nxs::f32("gamma", config_.gamma),
                             nxs::f32("lambda", config_.lambda),
                             nxs::f32("clip", config_.clip_epsilon),
                             nxs::f32("lr", config_.learning_rate),
                             nxs::f32("entropy", config_.entropy_coef),
                             nxs::f32("value", config_.value_coef),
                             nxs::i32("epochs", config_.epochs),
                             nxs::i32("seed", static_cast<int>(config_.seed))};
        archive.rl.nets.push_back(actor_.export_blob("actor"));
        archive.rl.nets.push_back(critic_.export_blob("critic"));
        nxs::save_archive(path, archive);
    }

    [[nodiscard]] static PPOAgent load_nxs(const std::string& path) {
        const nxs::Archive archive = nxs::load_archive(path);
        if (!archive.has_rl || archive.rl.kind != "ppo" || archive.rl.nets.size() < 2) {
            throw std::runtime_error("nxs file is not a ppo agent");
        }
        Config config;
        config.hidden = nxs::attrs_i32(archive.rl.config, "hidden", config.hidden);
        config.gamma = nxs::attrs_f32(archive.rl.config, "gamma", config.gamma);
        config.lambda = nxs::attrs_f32(archive.rl.config, "lambda", config.lambda);
        config.clip_epsilon = nxs::attrs_f32(archive.rl.config, "clip", config.clip_epsilon);
        config.learning_rate = nxs::attrs_f32(archive.rl.config, "lr", config.learning_rate);
        config.entropy_coef = nxs::attrs_f32(archive.rl.config, "entropy", config.entropy_coef);
        config.value_coef = nxs::attrs_f32(archive.rl.config, "value", config.value_coef);
        config.epochs = nxs::attrs_i32(archive.rl.config, "epochs", config.epochs);
        config.seed = static_cast<std::uint32_t>(nxs::attrs_i32(archive.rl.config, "seed", static_cast<int>(config.seed)));
        const int observation = nxs::attrs_i32(archive.rl.config, "observation", 0);
        const int actions = nxs::attrs_i32(archive.rl.config, "actions", 0);
        PPOAgent agent(observation, actions, config);
        agent.actor_.import_blob(archive.rl.nets[0]);
        agent.critic_.import_blob(archive.rl.nets[1]);
        return agent;
    }

    int act(const float* observation, float& log_prob, float& value, bool greedy) {
        std::vector<float> logits(static_cast<std::size_t>(actions_));
        std::vector<float> probs(logits.size());
        actor_.forward(observation, 1, logits.data());
        const float value_scalar = predict_value(observation);
        softmax(logits.data(), probs.data(), actions_);
        int action = 0;
        if (greedy) {
            for (int candidate = 1; candidate < actions_; ++candidate) {
                if (probs[static_cast<std::size_t>(candidate)] > probs[static_cast<std::size_t>(action)]) {
                    action = candidate;
                }
            }
        } else {
            float cursor = rng_.uniform();
            float sum = 0.0f;
            for (int candidate = 0; candidate < actions_; ++candidate) {
                sum += probs[static_cast<std::size_t>(candidate)];
                action = candidate;
                if (cursor <= sum) {
                    break;
                }
            }
        }
        log_prob = std::log(probs[static_cast<std::size_t>(action)] + 1.0e-8f);
        value = value_scalar;
        return action;
    }

    float value_of(const float* observation) { return predict_value(observation); }

    void update_policy(const float* observations, const int* actions, const float* old_log_probs, const float* advantages, int batch) {
        for (int epoch = 0; epoch < config_.epochs; ++epoch) {
            logits_.resize(static_cast<std::size_t>(batch * actions_));
            actor_.forward(observations, batch, logits_.data());
            grad_.assign(logits_.size(), 0.0f);
            probs_.resize(static_cast<std::size_t>(actions_));
            std::vector<float>& logits = logits_;
            std::vector<float>& grad = grad_;
            for (int row = 0; row < batch; ++row) {
                softmax(logits.data() + static_cast<std::size_t>(row * actions_), probs_.data(), actions_);
                const int action = actions[row];
                const float log_prob = std::log(probs_[static_cast<std::size_t>(action)] + 1.0e-8f);
                const float ratio = std::exp(log_prob - old_log_probs[row]);
                const float advantage = advantages[row];
                float d_log = 0.0f;
                const bool clipped_high = advantage >= 0.0f && ratio > 1.0f + config_.clip_epsilon;
                const bool clipped_low = advantage < 0.0f && ratio < 1.0f - config_.clip_epsilon;
                if (!clipped_high && !clipped_low) {
                    d_log = -ratio * advantage;
                }
                d_log /= static_cast<float>(batch);
                float entropy = 0.0f;
                for (int action_index = 0; action_index < actions_; ++action_index) {
                    const float probability = probs_[static_cast<std::size_t>(action_index)];
                    entropy -= probability * std::log(probability + 1.0e-8f);
                }
                for (int action_index = 0; action_index < actions_; ++action_index) {
                    const float probability = probs_[static_cast<std::size_t>(action_index)];
                    const float indicator = action_index == action ? 1.0f : 0.0f;
                    float d_logit = d_log * (indicator - probability);
                    d_logit += config_.entropy_coef * probability * (std::log(probability + 1.0e-8f) + entropy) / static_cast<float>(batch);
                    grad[static_cast<std::size_t>(row * actions_ + action_index)] = d_logit;
                }
            }
            actor_.zero_grad();
            actor_.backward(grad.data());
            actor_.step(config_.learning_rate);
        }
    }

    void update_value(const float* observations, const float* returns, const float* old_values, int batch) {
        for (int epoch = 0; epoch < config_.epochs; ++epoch) {
            values_.resize(static_cast<std::size_t>(batch));
            critic_.forward(observations, batch, values_.data());
            value_grad_.resize(static_cast<std::size_t>(batch));
            std::vector<float>& values = values_;
            std::vector<float>& grad = value_grad_;
            for (int row = 0; row < batch; ++row) {
                const float value = values[static_cast<std::size_t>(row)];
                const float delta = value - old_values[row];
                const float clipped = old_values[row] + std::clamp(delta, -config_.clip_epsilon, config_.clip_epsilon);
                const float unclipped = value - returns[row];
                const float clipped_error = clipped - returns[row];
                float local = unclipped;
                if (clipped_error * clipped_error > unclipped * unclipped && std::fabs(delta) <= config_.clip_epsilon) {
                    local = clipped_error;
                }
                if (clipped_error * clipped_error > unclipped * unclipped && std::fabs(delta) > config_.clip_epsilon) {
                    local = 0.0f;
                }
                grad[static_cast<std::size_t>(row)] = config_.value_coef * local / static_cast<float>(batch);
            }
            critic_.zero_grad();
            critic_.backward(grad.data());
            critic_.step(config_.learning_rate);
        }
    }

    float learn(Environment& env, int iterations, int horizon) {
        float last_mean = 0.0f;
        Tensor observation = env.reset(config_.seed);
        std::vector<float> current(static_cast<std::size_t>(observation_dim_));
        std::memcpy(current.data(), observation.data(), current.size() * sizeof(float));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            RolloutBuffer rollout;
            rollout.reset(horizon, observation_dim_);
            std::vector<float> episode_returns;
            float episode = 0.0f;
            bool bootstrap_from_value = true;
            std::vector<float> bootstrap_state = current;
            for (int step = 0; step < horizon; ++step) {
                float log_prob = 0.0f;
                float value = 0.0f;
                const int action = act(current.data(), log_prob, value, false);
                StepResult result = env.step(discrete_action_tensor(action));
                rollout.add(current.data(), action, log_prob, value, result.reward, result.terminated || result.truncated);
                episode += result.reward;
                std::memcpy(bootstrap_state.data(), result.observation.data(), bootstrap_state.size() * sizeof(float));
                bootstrap_from_value = !(result.terminated || result.truncated);
                if (result.terminated || result.truncated) {
                    episode_returns.push_back(episode);
                    episode = 0.0f;
                    Tensor restarted = env.reset(config_.seed + static_cast<std::uint32_t>(iteration * horizon + step + 1));
                    std::memcpy(current.data(), restarted.data(), current.size() * sizeof(float));
                } else {
                    current.swap(bootstrap_state);
                    bootstrap_state = current;
                }
            }
            const float bootstrap = bootstrap_from_value ? value_of(bootstrap_state.data()) : 0.0f;
            rollout.finish(config_.gamma, config_.lambda, bootstrap);
            update_policy(rollout.observations(), rollout.actions().data(), rollout.log_probs().data(), rollout.advantages().data(),
                          rollout.size());
            update_value(rollout.observations(), rollout.returns().data(), rollout.values().data(), rollout.size());
            if (!episode_returns.empty()) {
                double sum = 0.0;
                for (float value : episode_returns) {
                    sum += value;
                }
                last_mean = static_cast<float>(sum / static_cast<double>(episode_returns.size()));
            }
        }
        return last_mean;
    }

    float evaluate(Environment& env, int episodes) {
        double sum = 0.0;
        for (int episode = 0; episode < episodes; ++episode) {
            Tensor observation = env.reset(4000u + static_cast<std::uint32_t>(episode));
            std::vector<float> current(static_cast<std::size_t>(observation_dim_));
            std::memcpy(current.data(), observation.data(), current.size() * sizeof(float));
            float reward_sum = 0.0f;
            for (int step = 0; step < 64; ++step) {
                float log_prob = 0.0f;
                float value = 0.0f;
                const int action = act(current.data(), log_prob, value, true);
                StepResult result = env.step(discrete_action_tensor(action));
                reward_sum += result.reward;
                if (result.terminated || result.truncated) {
                    break;
                }
                std::memcpy(current.data(), result.observation.data(), current.size() * sizeof(float));
            }
            sum += reward_sum;
        }
        return static_cast<float>(sum / static_cast<double>(episodes));
    }

 private:
    float predict_value(const float* observation) {
        float value = 0.0f;
        critic_.forward(observation, 1, &value);
        return value;
    }

    static void softmax(const float* logits, float* probs, int n) {
        float peak = logits[0];
        for (int i = 1; i < n; ++i) {
            peak = std::max(peak, logits[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            probs[i] = std::exp(logits[i] - peak);
            sum += probs[i];
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) {
            probs[i] *= inv;
        }
    }

    int observation_dim_ = 0;
    int actions_ = 0;
    Config config_;
    Rng rng_;
    Mlp actor_;
    Mlp critic_;
    std::vector<float> logits_;
    std::vector<float> grad_;
    std::vector<float> probs_;
    std::vector<float> values_;
    std::vector<float> value_grad_;
};

}  // namespace rl
}  // namespace nexus
