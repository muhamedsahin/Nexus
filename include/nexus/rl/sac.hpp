#pragma once

/**
 * @file sac.hpp
 * @brief Tek boyutlu surekli SAC. Iki Q, yumusak hedef ve tanh yeniden parametreleme.
 * Alfa sabittir. Hedef, min Q eksi alfa log pi ile bootstrapping yapar.
 */

#include "nexus/model/nxs_format.hpp"
#include "nexus/rl/aligned_alloc.hpp"

#include <string>
#include "nexus/rl/environment.hpp"
#include "nexus/rl/mlp.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nexus {
namespace rl {

class SACAgent {
 public:
    struct Config {
        int hidden = 32;
        float gamma = 0.99f;
        float learning_rate = 1.0e-3f;
        float tau = 0.02f;
        float alpha = 0.2f;
        int batch = 64;
        int warmup = 128;
        int buffer_capacity = 2048;
        std::uint32_t seed = 1;
    };

    explicit SACAgent(int observation_dim, Config config = {})
        : observation_dim_(observation_dim), config_(config), rng_(config.seed),
          actor_({observation_dim, config.hidden, 2}, config.seed), q1_({observation_dim + 1, config.hidden, 1}, config.seed + 4),
          q2_({observation_dim + 1, config.hidden, 1}, config.seed + 5), observations_(static_cast<std::size_t>(config.buffer_capacity)),
          next_(observations_.size()), actions_(observations_.size()), rewards_(observations_.size()),
          terminated_(observations_.size()) {
        if (observation_dim != 1) {
            throw std::invalid_argument("SACAgent currently trains a 1D observation");
        }
        q1_target_ = q1_;
        q2_target_ = q2_;
        q1_target_.copy_weights_from(q1_);
        q2_target_.copy_weights_from(q2_);
    }

    void save_nxs(const std::string& path) const {
        nxs::Archive archive;
        archive.has_rl = true;
        archive.rl.kind = "sac";
        archive.rl.config = {nxs::i32("observation", observation_dim_), nxs::i32("hidden", config_.hidden),
                             nxs::i32("seed", static_cast<int>(config_.seed)), nxs::f32("gamma", config_.gamma),
                             nxs::f32("lr", config_.learning_rate), nxs::f32("tau", config_.tau), nxs::f32("alpha", config_.alpha)};
        archive.rl.nets = {actor_.export_blob("actor"), q1_.export_blob("q1"), q2_.export_blob("q2"), q1_target_.export_blob("q1_target"),
                           q2_target_.export_blob("q2_target")};
        nxs::save_archive(path, archive);
    }

    [[nodiscard]] static SACAgent load_nxs(const std::string& path) {
        const nxs::Archive archive = nxs::load_archive(path);
        if (!archive.has_rl || archive.rl.kind != "sac") {
            throw std::runtime_error("nxs file is not a sac agent");
        }
        Config config;
        config.hidden = nxs::attrs_i32(archive.rl.config, "hidden", config.hidden);
        config.seed = static_cast<std::uint32_t>(nxs::attrs_i32(archive.rl.config, "seed", 1));
        config.gamma = nxs::attrs_f32(archive.rl.config, "gamma", config.gamma);
        config.learning_rate = nxs::attrs_f32(archive.rl.config, "lr", config.learning_rate);
        config.tau = nxs::attrs_f32(archive.rl.config, "tau", config.tau);
        config.alpha = nxs::attrs_f32(archive.rl.config, "alpha", config.alpha);
        SACAgent agent(nxs::attrs_i32(archive.rl.config, "observation", 1), config);
        auto take = [&](Mlp& net, const char* name) {
            for (const nxs::MlpBlob& blob : archive.rl.nets) {
                if (blob.name == name) {
                    net.import_blob(blob);
                }
            }
        };
        take(agent.actor_, "actor");
        take(agent.q1_, "q1");
        take(agent.q2_, "q2");
        take(agent.q1_target_, "q1_target");
        take(agent.q2_target_, "q2_target");
        return agent;
    }

    float act(const float* observation, bool greedy) {
        float out[2];
        actor_.forward(observation, 1, out);
        const float mean = out[0];
        const float log_std = std::clamp(out[1], -5.0f, 2.0f);
        if (greedy) {
            return std::tanh(mean);
        }
        const float u = mean + std::exp(log_std) * rng_.normal();
        return std::tanh(u);
    }

    float learn(Environment& env, int steps) {
        Tensor observation = env.reset(config_.seed);
        float state = observation[0];
        float episode = 0.0f;
        float last = 0.0f;
        for (int step = 0; step < steps; ++step) {
            const float action = act(&state, false);
            const float stored[1] = {action};
            StepResult result = env.step(Tensor(Shape{1}, std::span<const float>(stored, 1)));
            push(state, action, result.reward, result.observation[0], result.terminated);
            episode += result.reward;
            if (size_ >= config_.warmup) {
                update();
            }
            if (result.terminated || result.truncated) {
                last = episode;
                episode = 0.0f;
                Tensor restarted = env.reset(config_.seed + static_cast<std::uint32_t>(step + 1));
                state = restarted[0];
            } else {
                state = result.observation[0];
            }
        }
        return last;
    }

    float evaluate(Environment& env, int episodes) {
        double sum = 0.0;
        for (int episode = 0; episode < episodes; ++episode) {
            Tensor observation = env.reset(50u + static_cast<std::uint32_t>(episode));
            float state = observation[0];
            float reward_sum = 0.0f;
            for (int step = 0; step < 10; ++step) {
                const float action = act(&state, true);
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
    void push(float observation, float action, float reward, float next, bool terminated) {
        const std::size_t slot = static_cast<std::size_t>(cursor_);
        observations_[slot] = observation;
        actions_[slot] = action;
        rewards_[slot] = reward;
        next_[slot] = next;
        terminated_[slot] = terminated ? 1 : 0;
        cursor_ = (cursor_ + 1) % config_.buffer_capacity;
        if (size_ < config_.buffer_capacity) {
            ++size_;
        }
    }

    static float correction(float action) { return std::log(std::max(1.0f - action * action, 1.0e-6f)); }

    void update() {
        const int batch = config_.batch;
        batch_obs_.resize(static_cast<std::size_t>(batch));
        batch_next_.resize(batch_obs_.size());
        batch_action_.resize(batch_obs_.size());
        batch_reward_.resize(batch_obs_.size());
        batch_done_.resize(batch_obs_.size());
        std::vector<float>& obs = batch_obs_;
        std::vector<float>& next = batch_next_;
        std::vector<float>& action = batch_action_;
        std::vector<float>& reward = batch_reward_;
        std::vector<std::uint8_t>& done = batch_done_;
        for (int row = 0; row < batch; ++row) {
            const int index = rng_.uniform_int(size_);
            const std::size_t slot = static_cast<std::size_t>(index);
            obs[static_cast<std::size_t>(row)] = observations_[slot];
            next[static_cast<std::size_t>(row)] = next_[slot];
            action[static_cast<std::size_t>(row)] = actions_[slot];
            reward[static_cast<std::size_t>(row)] = rewards_[slot];
            done[static_cast<std::size_t>(row)] = terminated_[slot];
        }
        actor_next_.resize(static_cast<std::size_t>(batch * 2));
        actor_.forward(next.data(), batch, actor_next_.data());
        std::vector<float>& actor_next = actor_next_;
        next_action_.resize(static_cast<std::size_t>(batch));
        next_logp_.resize(static_cast<std::size_t>(batch));
        noise_.resize(static_cast<std::size_t>(batch));
        std::vector<float>& next_action = next_action_;
        std::vector<float>& next_logp = next_logp_;
        std::vector<float>& noise = noise_;
        for (int row = 0; row < batch; ++row) {
            noise[static_cast<std::size_t>(row)] = rng_.normal();
            const float mean = actor_next[static_cast<std::size_t>(row * 2)];
            const float log_std = std::clamp(actor_next[static_cast<std::size_t>(row * 2 + 1)], -5.0f, 2.0f);
            const float u = mean + std::exp(log_std) * noise[static_cast<std::size_t>(row)];
            const float squashed = std::tanh(u);
            next_action[static_cast<std::size_t>(row)] = squashed;
            const float gauss = -0.5f * noise[static_cast<std::size_t>(row)] * noise[static_cast<std::size_t>(row)] - log_std -
                                0.5f * std::log(2.0f * 3.14159265f);
            next_logp[static_cast<std::size_t>(row)] = gauss - correction(squashed);
        }
        q_input_.resize(static_cast<std::size_t>(batch * 2));
        std::vector<float>& q_input = q_input_;
        for (int row = 0; row < batch; ++row) {
            q_input[static_cast<std::size_t>(row * 2)] = next[static_cast<std::size_t>(row)];
            q_input[static_cast<std::size_t>(row * 2 + 1)] = next_action[static_cast<std::size_t>(row)];
        }
        t1_.resize(static_cast<std::size_t>(batch));
        t2_.resize(t1_.size());
        std::vector<float>& t1 = t1_;
        std::vector<float>& t2 = t2_;
        q1_target_.forward(q_input.data(), batch, t1.data());
        q2_target_.forward(q_input.data(), batch, t2.data());
        target_.resize(static_cast<std::size_t>(batch));
        std::vector<float>& target = target_;
        for (int row = 0; row < batch; ++row) {
            const float qmin = std::min(t1[static_cast<std::size_t>(row)], t2[static_cast<std::size_t>(row)]);
            const float bootstrap = done[static_cast<std::size_t>(row)]
                                         ? 0.0f
                                         : qmin - config_.alpha * next_logp[static_cast<std::size_t>(row)];
            target[static_cast<std::size_t>(row)] = reward[static_cast<std::size_t>(row)] + config_.gamma * bootstrap;
        }
        for (int row = 0; row < batch; ++row) {
            q_input[static_cast<std::size_t>(row * 2)] = obs[static_cast<std::size_t>(row)];
            q_input[static_cast<std::size_t>(row * 2 + 1)] = action[static_cast<std::size_t>(row)];
        }
        fit_q(q1_, q_input.data(), target.data(), batch);
        fit_q(q2_, q_input.data(), target.data(), batch);

        actor_now_.resize(static_cast<std::size_t>(batch * 2));
        actor_.forward(obs.data(), batch, actor_now_.data());
        std::vector<float>& actor_now = actor_now_;
        d_actor_.assign(actor_now.size(), 0.0f);
        policy_action_.resize(static_cast<std::size_t>(batch));
        policy_u_.resize(static_cast<std::size_t>(batch));
        std::vector<float>& d_actor = d_actor_;
        std::vector<float>& policy_action = policy_action_;
        std::vector<float>& policy_u = policy_u_;
        for (int row = 0; row < batch; ++row) {
            const float mean = actor_now[static_cast<std::size_t>(row * 2)];
            const float log_std = std::clamp(actor_now[static_cast<std::size_t>(row * 2 + 1)], -5.0f, 2.0f);
            const float stddev = std::exp(log_std);
            const float epsilon = rng_.normal();
            const float u = mean + stddev * epsilon;
            const float squashed = std::tanh(u);
            policy_action[static_cast<std::size_t>(row)] = squashed;
            policy_u[static_cast<std::size_t>(row)] = u;
            q_input[static_cast<std::size_t>(row * 2)] = obs[static_cast<std::size_t>(row)];
            q_input[static_cast<std::size_t>(row * 2 + 1)] = squashed;
        }
        q1_values_.resize(static_cast<std::size_t>(batch));
        q2_values_.resize(q1_values_.size());
        std::vector<float>& q1_values = q1_values_;
        std::vector<float>& q2_values = q2_values_;
        q1_.forward(q_input.data(), batch, q1_values.data());
        q2_.forward(q_input.data(), batch, q2_values.data());
        d_q1_.assign(static_cast<std::size_t>(batch), 0.0f);
        d_q2_.assign(d_q1_.size(), 0.0f);
        std::vector<float>& d_q1 = d_q1_;
        std::vector<float>& d_q2 = d_q2_;
        for (int row = 0; row < batch; ++row) {
            if (q1_values[static_cast<std::size_t>(row)] <= q2_values[static_cast<std::size_t>(row)]) {
                d_q1[static_cast<std::size_t>(row)] = -1.0f / static_cast<float>(batch);
            } else {
                d_q2[static_cast<std::size_t>(row)] = -1.0f / static_cast<float>(batch);
            }
        }
        grad_q1_.assign(static_cast<std::size_t>(batch * 2), 0.0f);
        grad_q2_.assign(grad_q1_.size(), 0.0f);
        std::vector<float>& grad_q1 = grad_q1_;
        std::vector<float>& grad_q2 = grad_q2_;
        q1_.zero_grad();
        q2_.zero_grad();
        q1_.backward(d_q1.data(), grad_q1.data());
        q2_.backward(d_q2.data(), grad_q2.data());
        for (int row = 0; row < batch; ++row) {
            const float d_action = grad_q1[static_cast<std::size_t>(row * 2 + 1)] + grad_q2[static_cast<std::size_t>(row * 2 + 1)];
            const float mean = actor_now[static_cast<std::size_t>(row * 2)];
            const float u = policy_u[static_cast<std::size_t>(row)];
            const float squashed = policy_action[static_cast<std::size_t>(row)];
            const float da_du = std::max(1.0f - squashed * squashed, 1.0e-6f);
            const float d_logp_du = 2.0f * squashed * da_du / da_du;
            const float d_q_da = -d_action * static_cast<float>(batch);
            const float d_loss_du = (config_.alpha * (d_logp_du / da_du) - d_q_da) * da_du;
            d_actor[static_cast<std::size_t>(row * 2)] = d_loss_du / static_cast<float>(batch);
            d_actor[static_cast<std::size_t>(row * 2 + 1)] =
                (config_.alpha * (-1.0f + d_logp_du * (u - mean)) - d_q_da * da_du * (u - mean)) / static_cast<float>(batch);
        }
        actor_.zero_grad();
        actor_.backward(d_actor.data());
        actor_.step(config_.learning_rate);
        q1_target_.soft_update_from(q1_, config_.tau);
        q2_target_.soft_update_from(q2_, config_.tau);
    }

    void fit_q(Mlp& net, const float* input, const float* target, int batch) {
        q_pred_.resize(static_cast<std::size_t>(batch));
        net.forward(input, batch, q_pred_.data());
        q_grad_.resize(q_pred_.size());
        std::vector<float>& predicted = q_pred_;
        std::vector<float>& grad = q_grad_;
        for (int row = 0; row < batch; ++row) {
            grad[static_cast<std::size_t>(row)] =
                (predicted[static_cast<std::size_t>(row)] - target[row]) / static_cast<float>(batch);
        }
        net.zero_grad();
        net.backward(grad.data());
        net.step(config_.learning_rate);
    }

    int observation_dim_ = 1;
    Config config_;
    Rng rng_;
    Mlp actor_;
    Mlp q1_;
    Mlp q2_;
    Mlp q1_target_;
    Mlp q2_target_;
    AlignedVector<float> observations_;
    AlignedVector<float> next_;
    AlignedVector<float> actions_;
    AlignedVector<float> rewards_;
    std::vector<std::uint8_t> terminated_;
    int cursor_ = 0;
    int size_ = 0;
    std::vector<float> batch_obs_;
    std::vector<float> batch_next_;
    std::vector<float> batch_action_;
    std::vector<float> batch_reward_;
    std::vector<std::uint8_t> batch_done_;
    std::vector<float> actor_next_;
    std::vector<float> next_action_;
    std::vector<float> next_logp_;
    std::vector<float> noise_;
    std::vector<float> q_input_;
    std::vector<float> t1_;
    std::vector<float> t2_;
    std::vector<float> target_;
    std::vector<float> actor_now_;
    std::vector<float> d_actor_;
    std::vector<float> policy_action_;
    std::vector<float> policy_u_;
    std::vector<float> q1_values_;
    std::vector<float> q2_values_;
    std::vector<float> d_q1_;
    std::vector<float> d_q2_;
    std::vector<float> grad_q1_;
    std::vector<float> grad_q2_;
    std::vector<float> q_pred_;
    std::vector<float> q_grad_;
};

}  // namespace rl
}  // namespace nexus
