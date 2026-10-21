#pragma once

/**
 * @file dqn.hpp
 * @brief DQN, Double DQN, Dueling DQN ve istege bagli PER.
 * Hedef ag periyodik kopyalanir. Bootstrap yalnizca terminated olmayan adimda vardir.
 */

#include "nexus/model/nxs_format.hpp"
#include "nexus/rl/environment.hpp"

#include <string>
#include "nexus/rl/mlp.hpp"
#include "nexus/rl/prioritized_replay.hpp"
#include "nexus/rl/replay_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace nexus {
namespace rl {

class DQNAgent {
 public:
    struct Config {
        int hidden = 64;
        float gamma = 0.99f;
        float learning_rate = 1.0e-2f;
        float epsilon_start = 1.0f;
        float epsilon_end = 0.05f;
        int epsilon_decay_steps = 300;
        int batch = 64;
        int warmup = 64;
        int target_interval = 40;
        int buffer_capacity = 2048;
        bool double_q = true;
        bool dueling = true;
        bool prioritized = true;
        float per_alpha = 0.6f;
        float per_beta = 0.4f;
        std::uint32_t seed = 1;
    };

    DQNAgent(int observation_dim, int actions, Config config = {})
        : observation_dim_(observation_dim), actions_(actions), config_(config), rng_(config.seed),
          uniform_(config.buffer_capacity, observation_dim),
          prioritized_(config.buffer_capacity, observation_dim, config.per_alpha) {
        if (observation_dim <= 0 || actions <= 1) {
            throw std::invalid_argument("DQNAgent dimensions are invalid");
        }
        if (config_.dueling) {
            feature_ = Mlp({observation_dim_, config_.hidden}, config.seed);
            value_ = Mlp({config_.hidden, 1}, config.seed + 1);
            advantage_ = Mlp({config_.hidden, actions_}, config.seed + 2);
            feature_target_ = feature_;
            value_target_ = value_;
            advantage_target_ = advantage_;
            feature_target_.copy_weights_from(feature_);
            value_target_.copy_weights_from(value_);
            advantage_target_.copy_weights_from(advantage_);
        } else {
            q_ = Mlp({observation_dim_, config_.hidden, actions_}, config.seed);
            q_target_ = q_;
            q_target_.copy_weights_from(q_);
        }
    }

    void save_nxs(const std::string& path) const {
        nxs::Archive archive;
        archive.has_rl = true;
        archive.rl.kind = "dqn";
        archive.rl.config = {nxs::i32("observation", observation_dim_),
                             nxs::i32("actions", actions_),
                             nxs::i32("hidden", config_.hidden),
                             nxs::i32("seed", static_cast<int>(config_.seed)),
                             nxs::boolean("dueling", config_.dueling),
                             nxs::boolean("double_q", config_.double_q),
                             nxs::boolean("prioritized", config_.prioritized)};
        archive.rl.nets = {q_.export_blob("q"),           q_target_.export_blob("q_target"),
                           feature_.export_blob("feature"), feature_target_.export_blob("feature_target"),
                           value_.export_blob("value"),   value_target_.export_blob("value_target"),
                           advantage_.export_blob("advantage"), advantage_target_.export_blob("advantage_target")};
        nxs::save_archive(path, archive);
    }

    [[nodiscard]] static DQNAgent load_nxs(const std::string& path) {
        const nxs::Archive archive = nxs::load_archive(path);
        if (!archive.has_rl || archive.rl.kind != "dqn") {
            throw std::runtime_error("nxs file is not a dqn agent");
        }
        Config config;
        config.hidden = nxs::attrs_i32(archive.rl.config, "hidden", config.hidden);
        config.seed = static_cast<std::uint32_t>(nxs::attrs_i32(archive.rl.config, "seed", 1));
        config.dueling = false;
        config.double_q = false;
        config.prioritized = false;
        for (const nxs::Attr& attr : archive.rl.config) {
            if (attr.key == "dueling") config.dueling = attr.b;
            if (attr.key == "double_q") config.double_q = attr.b;
            if (attr.key == "prioritized") config.prioritized = attr.b;
        }
        DQNAgent agent(nxs::attrs_i32(archive.rl.config, "observation", 1), nxs::attrs_i32(archive.rl.config, "actions", 2), config);
        auto take = [&](Mlp& net, const char* name) {
            for (const nxs::MlpBlob& blob : archive.rl.nets) {
                if (blob.name == name && !blob.parameters.empty()) {
                    net.import_blob(blob);
                }
            }
        };
        take(agent.q_, "q");
        take(agent.q_target_, "q_target");
        take(agent.feature_, "feature");
        take(agent.feature_target_, "feature_target");
        take(agent.value_, "value");
        take(agent.value_target_, "value_target");
        take(agent.advantage_, "advantage");
        take(agent.advantage_target_, "advantage_target");
        return agent;
    }

    int act(const float* observation, bool greedy) {
        const float progress = std::min(1.0f, static_cast<float>(env_steps_) / static_cast<float>(std::max(config_.epsilon_decay_steps, 1)));
        const float epsilon = config_.epsilon_start + (config_.epsilon_end - config_.epsilon_start) * progress;
        if (!greedy && rng_.uniform() < epsilon) {
            return rng_.uniform_int(actions_);
        }
        std::vector<float> q(static_cast<std::size_t>(actions_));
        q_values(false, observation, 1, q.data());
        int best = 0;
        for (int action = 1; action < actions_; ++action) {
            if (q[static_cast<std::size_t>(action)] > q[static_cast<std::size_t>(best)]) {
                best = action;
            }
        }
        return best;
    }

    void observe(const float* observation, int action, float reward, const float* next_observation, bool terminated) {
        const float priority = 1.0f;
        if (config_.prioritized) {
            prioritized_.add(observation, action, reward, next_observation, terminated, priority);
        } else {
            uniform_.add(observation, action, reward, next_observation, terminated);
        }
    }

    float update() {
        const int live = config_.prioritized ? prioritized_.size() : uniform_.size();
        if (live < config_.batch) {
            return 0.0f;
        }
        const int batch = config_.batch;
        const int dim = observation_dim_;
        observations_.resize(static_cast<std::size_t>(batch * dim));
        next_observations_.resize(observations_.size());
        sampled_actions_.resize(static_cast<std::size_t>(batch));
        rewards_.resize(static_cast<std::size_t>(batch));
        terminated_.resize(static_cast<std::size_t>(batch));
        weights_.assign(static_cast<std::size_t>(batch), 1.0f);
        indices_.resize(static_cast<std::size_t>(batch));
        std::vector<float>& observations = observations_;
        std::vector<float>& next_observations = next_observations_;
        std::vector<int>& actions = sampled_actions_;
        std::vector<float>& rewards = rewards_;
        std::vector<std::uint8_t>& terminated = terminated_;
        std::vector<float>& weights = weights_;
        std::vector<int>& indices = indices_;
        if (config_.prioritized) {
            auto sample = prioritized_.sample(batch, config_.per_beta, rng_);
            observations.swap(sample.observations);
            next_observations.swap(sample.next_observations);
            actions.swap(sample.actions);
            rewards.swap(sample.rewards);
            terminated.swap(sample.terminated);
            weights.swap(sample.weights);
            indices.swap(sample.indices);
        } else {
            for (int row = 0; row < batch; ++row) {
                indices[static_cast<std::size_t>(row)] = rng_.uniform_int(live);
            }
            uniform_.sample(indices, observations.data(), next_observations.data(), actions.data(), rewards.data(), terminated.data());
        }

        std::vector<float> q(static_cast<std::size_t>(batch * actions_));
        std::vector<float> next_online(q.size());
        std::vector<float> next_target(q.size());
        q_values(false, observations.data(), batch, q.data());
        q_values(true, next_observations.data(), batch, next_target.data());
        if (config_.double_q) {
            q_values(false, next_observations.data(), batch, next_online.data());
        }

        std::vector<float> dy(q.size(), 0.0f);
        std::vector<float> td(static_cast<std::size_t>(batch));
        double loss = 0.0;
        for (int row = 0; row < batch; ++row) {
            const int action = actions[static_cast<std::size_t>(row)];
            float bootstrap = 0.0f;
            if (!terminated[static_cast<std::size_t>(row)]) {
                if (config_.double_q) {
                    int best = 0;
                    for (int candidate = 1; candidate < actions_; ++candidate) {
                        if (next_online[static_cast<std::size_t>(row * actions_ + candidate)] >
                            next_online[static_cast<std::size_t>(row * actions_ + best)]) {
                            best = candidate;
                        }
                    }
                    bootstrap = next_target[static_cast<std::size_t>(row * actions_ + best)];
                } else {
                    bootstrap = next_target[static_cast<std::size_t>(row * actions_)];
                    for (int candidate = 1; candidate < actions_; ++candidate) {
                        bootstrap = std::max(bootstrap, next_target[static_cast<std::size_t>(row * actions_ + candidate)]);
                    }
                }
            }
            const float target = rewards[static_cast<std::size_t>(row)] + config_.gamma * bootstrap;
            const float predicted = q[static_cast<std::size_t>(row * actions_ + action)];
            const float diff = predicted - target;
            td[static_cast<std::size_t>(row)] = std::fabs(diff);
            const float huber = std::fabs(diff) <= 1.0f ? diff : (diff > 0.0f ? 1.0f : -1.0f);
            dy[static_cast<std::size_t>(row * actions_ + action)] = huber / static_cast<float>(batch) * weights[static_cast<std::size_t>(row)];
            loss += static_cast<double>(diff * diff);
        }

        q_values(false, observations.data(), batch, q.data());
        backward_q(dy.data(), batch);
        ++updates_;
        if (updates_ % config_.target_interval == 0) {
            copy_to_target();
        }
        if (config_.prioritized) {
            prioritized_.update_priorities(indices, td);
        }
        return static_cast<float>(loss / static_cast<double>(batch));
    }

    float learn(Environment& env, int steps) {
        Tensor observation = env.reset(config_.seed);
        std::vector<float> current(static_cast<std::size_t>(observation_dim_));
        std::memcpy(current.data(), observation.data(), current.size() * sizeof(float));
        float episode = 0.0f;
        std::vector<float> returns;
        for (int step = 0; step < steps; ++step) {
            const int action = act(current.data(), false);
            StepResult result = env.step(discrete_action_tensor(action));
            std::vector<float> next(current.size());
            std::memcpy(next.data(), result.observation.data(), next.size() * sizeof(float));
            observe(current.data(), action, result.reward, next.data(), result.terminated);
            episode += result.reward;
            ++env_steps_;
            if (result.terminated || result.truncated) {
                returns.push_back(episode);
                episode = 0.0f;
                Tensor restarted = env.reset(config_.seed + static_cast<std::uint32_t>(returns.size()));
                std::memcpy(current.data(), restarted.data(), current.size() * sizeof(float));
            } else {
                current.swap(next);
            }
            if (env_steps_ >= config_.warmup) {
                update();
            }
        }
        if (returns.empty()) {
            return episode;
        }
        const std::size_t window = std::min<std::size_t>(returns.size(), 10);
        double sum = 0.0;
        for (std::size_t i = returns.size() - window; i < returns.size(); ++i) {
            sum += returns[i];
        }
        return static_cast<float>(sum / static_cast<double>(window));
    }

    float evaluate(Environment& env, int episodes) {
        double sum = 0.0;
        for (int episode = 0; episode < episodes; ++episode) {
            Tensor observation = env.reset(1000u + static_cast<std::uint32_t>(episode));
            std::vector<float> current(static_cast<std::size_t>(observation_dim_));
            std::memcpy(current.data(), observation.data(), current.size() * sizeof(float));
            float reward_sum = 0.0f;
            for (int step = 0; step < 64; ++step) {
                const int action = act(current.data(), true);
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
    void q_values(bool target, const float* observations, int batch, float* out) {
        if (!config_.dueling) {
            (target ? q_target_ : q_).forward(observations, batch, out);
            return;
        }
        std::vector<float> hidden(static_cast<std::size_t>(batch * config_.hidden));
        std::vector<float> value(static_cast<std::size_t>(batch));
        std::vector<float> advantage(static_cast<std::size_t>(batch * actions_));
        Mlp& feature = target ? feature_target_ : feature_;
        Mlp& value_net = target ? value_target_ : value_;
        Mlp& advantage_net = target ? advantage_target_ : advantage_;
        feature.forward(observations, batch, hidden.data());
        value_net.forward(hidden.data(), batch, value.data());
        advantage_net.forward(hidden.data(), batch, advantage.data());
        for (int row = 0; row < batch; ++row) {
            float mean = 0.0f;
            for (int action = 0; action < actions_; ++action) {
                mean += advantage[static_cast<std::size_t>(row * actions_ + action)];
            }
            mean /= static_cast<float>(actions_);
            for (int action = 0; action < actions_; ++action) {
                out[static_cast<std::size_t>(row * actions_ + action)] =
                    value[static_cast<std::size_t>(row)] + advantage[static_cast<std::size_t>(row * actions_ + action)] - mean;
            }
        }
    }

    void backward_q(const float* dy, int batch) {
        if (!config_.dueling) {
            q_.zero_grad();
            q_.backward(dy);
            q_.step(config_.learning_rate);
            return;
        }
        std::vector<float> d_value(static_cast<std::size_t>(batch), 0.0f);
        std::vector<float> d_advantage(static_cast<std::size_t>(batch * actions_), 0.0f);
        for (int row = 0; row < batch; ++row) {
            float sum = 0.0f;
            for (int action = 0; action < actions_; ++action) {
                sum += dy[static_cast<std::size_t>(row * actions_ + action)];
            }
            d_value[static_cast<std::size_t>(row)] = sum;
            for (int action = 0; action < actions_; ++action) {
                d_advantage[static_cast<std::size_t>(row * actions_ + action)] =
                    dy[static_cast<std::size_t>(row * actions_ + action)] - sum / static_cast<float>(actions_);
            }
        }
        std::vector<float> from_value(static_cast<std::size_t>(batch * config_.hidden), 0.0f);
        std::vector<float> from_advantage(from_value.size(), 0.0f);
        value_.zero_grad();
        advantage_.zero_grad();
        value_.backward(d_value.data(), from_value.data());
        advantage_.backward(d_advantage.data(), from_advantage.data());
        for (std::size_t i = 0; i < from_value.size(); ++i) {
            from_value[i] += from_advantage[i];
        }
        feature_.zero_grad();
        feature_.backward(from_value.data());
        value_.step(config_.learning_rate);
        advantage_.step(config_.learning_rate);
        feature_.step(config_.learning_rate);
    }

    void copy_to_target() {
        if (config_.dueling) {
            feature_target_.copy_weights_from(feature_);
            value_target_.copy_weights_from(value_);
            advantage_target_.copy_weights_from(advantage_);
        } else {
            q_target_.copy_weights_from(q_);
        }
    }

    int observation_dim_ = 0;
    int actions_ = 0;
    Config config_;
    Rng rng_;
    ReplayBuffer uniform_;
    PrioritizedReplayBuffer prioritized_;
    Mlp q_;
    Mlp q_target_;
    Mlp feature_;
    Mlp value_;
    Mlp advantage_;
    Mlp feature_target_;
    Mlp value_target_;
    Mlp advantage_target_;
    int env_steps_ = 0;
    int updates_ = 0;
    std::vector<float> observations_;
    std::vector<float> next_observations_;
    std::vector<int> sampled_actions_;
    std::vector<float> rewards_;
    std::vector<std::uint8_t> terminated_;
    std::vector<float> weights_;
    std::vector<int> indices_;
};

}  // namespace rl
}  // namespace nexus
