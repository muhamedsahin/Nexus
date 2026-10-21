#pragma once

/**
 * @file envs.hpp
 * @brief Yerlesik ortamlar.
 *
 * LineWorld ve GridWorld ayrik klasik kontrol. CartPole, OpenAI Gym
 * denklemlerini kullanir. ButtonWorld isbirligi, Duel sifir toplam,
 * TurnLine AEC, PointMass surekli eylemdir.
 */

#include "nexus/rl/environment.hpp"
#include "nexus/rl/multi_agent.hpp"
#include "nexus/rl/rng.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace nexus {
namespace rl {

inline Tensor one_hot(int index, int n) {
    Tensor tensor = Tensor::zeros(Shape{static_cast<std::size_t>(n)});
    if (index >= 0 && index < n) {
        tensor[static_cast<std::size_t>(index)] = 1.0f;
    }
    return tensor;
}

class LineWorld final : public Environment {
 public:
    explicit LineWorld(int length = 5, int horizon = 12) : length_(length), horizon_(horizon) {
        if (length < 2) {
            throw std::invalid_argument("LineWorld length must be at least 2");
        }
        observation_ = Space::box(length, 0.0f, 1.0f);
        action_ = Space::discrete(2);
    }

    [[nodiscard]] const Space& observation_space() const override { return observation_; }
    [[nodiscard]] const Space& action_space() const override { return action_; }

    Tensor reset(std::uint32_t) override {
        position_ = 0;
        steps_ = 0;
        return one_hot(position_, length_);
    }

    StepResult step(const Tensor& action) override {
        const int choice = read_discrete_action(action, 2);
        StepResult result;
        if (choice == 1 && position_ < length_ - 1) {
            ++position_;
            result.reward = 0.1f;
        } else if (choice == 0 && position_ > 0) {
            --position_;
        }
        ++steps_;
        if (position_ == length_ - 1) {
            result.reward += 1.0f;
            result.terminated = true;
            result.info = "goal";
        } else if (steps_ >= horizon_) {
            result.truncated = true;
            result.info = "horizon";
        }
        result.observation = one_hot(position_, length_);
        return result;
    }

 private:
    int length_;
    int horizon_;
    int position_ = 0;
    int steps_ = 0;
    Space observation_ = Space::box(5, 0.0f, 1.0f);
    Space action_ = Space::discrete(2);
};

class GridWorld final : public Environment {
 public:
    explicit GridWorld(int width = 4, int horizon = 24) : width_(width), horizon_(horizon) {
        observation_ = Space::box(width * width, 0.0f, 1.0f);
        action_ = Space::discrete(4);
    }

    [[nodiscard]] const Space& observation_space() const override { return observation_; }
    [[nodiscard]] const Space& action_space() const override { return action_; }

    Tensor reset(std::uint32_t) override {
        row_ = 0;
        col_ = 0;
        steps_ = 0;
        return observe();
    }

    StepResult step(const Tensor& action) override {
        const int choice = read_discrete_action(action, 4);
        if (choice == 0 && col_ + 1 < width_) {
            ++col_;
        } else if (choice == 1 && col_ > 0) {
            --col_;
        } else if (choice == 2 && row_ + 1 < width_) {
            ++row_;
        } else if (choice == 3 && row_ > 0) {
            --row_;
        }
        ++steps_;
        StepResult result;
        result.reward = -0.01f;
        if (row_ == width_ - 1 && col_ == width_ - 1) {
            result.reward = 1.0f;
            result.terminated = true;
            result.info = "goal";
        } else if (steps_ >= horizon_) {
            result.truncated = true;
        }
        result.observation = observe();
        return result;
    }

 private:
    Tensor observe() const { return one_hot(row_ * width_ + col_, width_ * width_); }

    int width_;
    int horizon_;
    int row_ = 0;
    int col_ = 0;
    int steps_ = 0;
    Space observation_;
    Space action_;
};

class CartPole final : public Environment {
 public:
    CartPole() {
        observation_ = Space::box(4, -10.0f, 10.0f);
        action_ = Space::discrete(2);
    }

    [[nodiscard]] const Space& observation_space() const override { return observation_; }
    [[nodiscard]] const Space& action_space() const override { return action_; }

    Tensor reset(std::uint32_t seed) override {
        rng_ = Rng(seed);
        x_ = (rng_.uniform() - 0.5f) * 0.1f;
        x_dot_ = (rng_.uniform() - 0.5f) * 0.1f;
        theta_ = (rng_.uniform() - 0.5f) * 0.1f;
        theta_dot_ = (rng_.uniform() - 0.5f) * 0.1f;
        steps_ = 0;
        return observe();
    }

    StepResult step(const Tensor& action) override {
        const int choice = read_discrete_action(action, 2);
        constexpr float kGravity = 9.8f;
        constexpr float kMassCart = 1.0f;
        constexpr float kMassPole = 0.1f;
        constexpr float kTotal = kMassCart + kMassPole;
        constexpr float kLength = 0.5f;
        constexpr float kPoleMassLength = kMassPole * kLength;
        constexpr float kForceMag = 10.0f;
        constexpr float kTau = 0.02f;
        constexpr float kThetaThreshold = 12.0f * 3.14159265f / 180.0f;
        constexpr float kXThreshold = 2.4f;
        const float force = choice == 1 ? kForceMag : -kForceMag;
        const float costheta = std::cos(theta_);
        const float sintheta = std::sin(theta_);
        const float temp = (force + kPoleMassLength * theta_dot_ * theta_dot_ * sintheta) / kTotal;
        const float theta_acc = (kGravity * sintheta - costheta * temp) /
                                 (kLength * (4.0f / 3.0f - kMassPole * costheta * costheta / kTotal));
        const float x_acc = temp - kPoleMassLength * theta_acc * costheta / kTotal;
        x_ += kTau * x_dot_;
        x_dot_ += kTau * x_acc;
        theta_ += kTau * theta_dot_;
        theta_dot_ += kTau * theta_acc;
        ++steps_;
        StepResult result;
        result.reward = 1.0f;
        result.terminated = std::fabs(x_) > kXThreshold || std::fabs(theta_) > kThetaThreshold;
        result.truncated = steps_ >= 500;
        result.observation = observe();
        return result;
    }

    [[nodiscard]] float x() const noexcept { return x_; }

 private:
    Tensor observe() const {
        const float values[4] = {x_, x_dot_, theta_, theta_dot_};
        return Tensor(Shape{4}, std::span<const float>(values, 4));
    }

    Space observation_ = Space::box(4, -10.0f, 10.0f);
    Space action_ = Space::discrete(2);
    Rng rng_{1};
    float x_ = 0.0f;
    float x_dot_ = 0.0f;
    float theta_ = 0.0f;
    float theta_dot_ = 0.0f;
    int steps_ = 0;
};

class PointMass final : public Environment {
 public:
    PointMass() {
        observation_ = Space::box(1, -2.0f, 2.0f);
        action_ = Space::box(1, -1.0f, 1.0f);
    }

    [[nodiscard]] const Space& observation_space() const override { return observation_; }
    [[nodiscard]] const Space& action_space() const override { return action_; }
    [[nodiscard]] int action_count() const override { return 1; }

    Tensor reset(std::uint32_t) override {
        x_ = 1.0f;
        steps_ = 0;
        return observe();
    }

    StepResult step(const Tensor& action) override {
        if (action.numel() != 1) {
            throw std::invalid_argument("PointMass action is a single float");
        }
        const float clipped = std::clamp(action[0], -1.0f, 1.0f);
        x_ = std::clamp(x_ + 0.5f * clipped, -2.0f, 2.0f);
        ++steps_;
        StepResult result;
        result.reward = -x_ * x_;
        result.truncated = steps_ >= 10;
        result.observation = observe();
        return result;
    }

 private:
    Tensor observe() const {
        const float value = x_;
        return Tensor(Shape{1}, std::span<const float>(&value, 1));
    }

    Space observation_ = Space::box(1, -2.0f, 2.0f);
    Space action_ = Space::box(1, -1.0f, 1.0f);
    float x_ = 1.0f;
    int steps_ = 0;
};

class TeamButton final : public MultiAgentEnvironment {
 public:
    explicit TeamButton(int agent_count, int horizon = 6) : horizon_(horizon) {
        if (agent_count < 1) {
            throw std::invalid_argument("TeamButton requires at least one agent");
        }
        agents_.reserve(static_cast<std::size_t>(agent_count));
        pressed_.assign(static_cast<std::size_t>(agent_count), false);
        for (int index = 0; index < agent_count; ++index) {
            agents_.push_back("a" + std::to_string(index));
        }
        observation_ = Space::box(2, 0.0f, 1.0f);
        action_ = Space::discrete(2);
    }

    [[nodiscard]] const std::vector<AgentID>& agents() const override { return agents_; }
    [[nodiscard]] const Space& observation_space(const AgentID&) const override { return observation_; }
    [[nodiscard]] const Space& action_space(const AgentID&) const override { return action_; }

    std::unordered_map<AgentID, Tensor> reset(std::uint32_t) override {
        std::fill(pressed_.begin(), pressed_.end(), false);
        steps_ = 0;
        return observations();
    }

    MultiStepResult step(const std::unordered_map<AgentID, Tensor>& actions) override {
        bool all_pressed = true;
        for (std::size_t index = 0; index < agents_.size(); ++index) {
            pressed_[index] = read_discrete_action(actions.at(agents_[index]), 2) == 1;
            all_pressed = all_pressed && pressed_[index];
        }
        ++steps_;
        const float reward = all_pressed ? 1.0f : 0.0f;
        const bool stop = steps_ >= horizon_;
        MultiStepResult result;
        result.observations = observations();
        result.done = stop;
        for (const AgentID& name : agents_) {
            result.rewards.emplace(name, reward);
            result.terminated.emplace(name, false);
            result.truncated.emplace(name, stop);
        }
        return result;
    }

 private:
    std::unordered_map<AgentID, Tensor> observations() const {
        std::unordered_map<AgentID, Tensor> view;
        for (std::size_t index = 0; index < agents_.size(); ++index) {
            view.emplace(agents_[index], one_hot(pressed_[index] ? 1 : 0, 2));
        }
        return view;
    }

    std::vector<AgentID> agents_;
    std::vector<bool> pressed_;
    Space observation_ = Space::box(2, 0.0f, 1.0f);
    Space action_ = Space::discrete(2);
    int steps_ = 0;
    int horizon_ = 6;
};

class ButtonWorld final : public MultiAgentEnvironment {
 public:
    ButtonWorld() : agents_{"a", "b"} {
        observation_ = Space::box(2, 0.0f, 1.0f);
        action_ = Space::discrete(2);
    }

    [[nodiscard]] const std::vector<AgentID>& agents() const override { return agents_; }
    [[nodiscard]] const Space& observation_space(const AgentID&) const override { return observation_; }
    [[nodiscard]] const Space& action_space(const AgentID&) const override { return action_; }

    std::unordered_map<AgentID, Tensor> reset(std::uint32_t) override {
        on_a_ = false;
        on_b_ = false;
        steps_ = 0;
        return observations();
    }

    MultiStepResult step(const std::unordered_map<AgentID, Tensor>& actions) override {
        on_a_ = read_discrete_action(actions.at("a"), 2) == 1;
        on_b_ = read_discrete_action(actions.at("b"), 2) == 1;
        ++steps_;
        const float reward = (on_a_ && on_b_) ? 1.0f : 0.0f;
        MultiStepResult result;
        result.observations = observations();
        result.rewards = {{"a", reward}, {"b", reward}};
        const bool stop = steps_ >= horizon_;
        result.terminated = {{"a", false}, {"b", false}};
        result.truncated = {{"a", stop}, {"b", stop}};
        result.done = stop;
        return result;
    }

 private:
    std::unordered_map<AgentID, Tensor> observations() const {
        return {{"a", one_hot(on_a_ ? 1 : 0, 2)}, {"b", one_hot(on_b_ ? 1 : 0, 2)}};
    }

    std::vector<AgentID> agents_;
    Space observation_ = Space::box(2, 0.0f, 1.0f);
    Space action_ = Space::discrete(2);
    bool on_a_ = false;
    bool on_b_ = false;
    int steps_ = 0;
    int horizon_ = 6;
};

class Duel final : public MultiAgentEnvironment {
 public:
    Duel() : agents_{"a", "b"} {
        observation_ = Space::box(1, 0.0f, 1.0f);
        action_ = Space::discrete(2);
    }

    [[nodiscard]] const std::vector<AgentID>& agents() const override { return agents_; }
    [[nodiscard]] const Space& observation_space(const AgentID&) const override { return observation_; }
    [[nodiscard]] const Space& action_space(const AgentID&) const override { return action_; }

    std::unordered_map<AgentID, Tensor> reset(std::uint32_t) override { return observations(); }

    MultiStepResult step(const std::unordered_map<AgentID, Tensor>& actions) override {
        const int a = read_discrete_action(actions.at("a"), 2);
        const int b = read_discrete_action(actions.at("b"), 2);
        const float score = a == b ? 1.0f : -1.0f;
        MultiStepResult result;
        result.observations = observations();
        result.rewards = {{"a", score}, {"b", -score}};
        result.terminated = {{"a", true}, {"b", true}};
        result.truncated = {{"a", false}, {"b", false}};
        result.done = true;
        return result;
    }

 private:
    std::unordered_map<AgentID, Tensor> observations() const {
        const float value = 1.0f;
        Tensor obs(Shape{1}, std::span<const float>(&value, 1));
        return {{"a", obs.clone()}, {"b", obs.clone()}};
    }

    std::vector<AgentID> agents_;
    Space observation_ = Space::box(1, 0.0f, 1.0f);
    Space action_ = Space::discrete(2);
};

class TurnLine final : public MultiAgentEnvironment {
 public:
    TurnLine() : agents_{"a", "b"} {
        observation_ = Space::box(1, 0.0f, 4.0f);
        action_ = Space::discrete(2);
    }

    [[nodiscard]] const std::vector<AgentID>& agents() const override { return agents_; }
    [[nodiscard]] const Space& observation_space(const AgentID&) const override { return observation_; }
    [[nodiscard]] const Space& action_space(const AgentID&) const override { return action_; }
    [[nodiscard]] bool simultaneous() const override { return false; }
    [[nodiscard]] AgentID current_agent() const override { return agents_[static_cast<std::size_t>(turn_)]; }

    std::unordered_map<AgentID, Tensor> reset(std::uint32_t) override {
        position_ = 0;
        turn_ = 0;
        pending_reward_ = 0.0f;
        return observations();
    }

    MultiStepResult step(const std::unordered_map<AgentID, Tensor>&) override {
        throw std::logic_error("TurnLine is AEC; use step_one");
    }

    void step_one(const Tensor& action) override {
        const int choice = read_discrete_action(action, 2);
        if (choice == 0) {
            position_ = std::min(4, position_ + 1);
        } else {
            position_ = std::max(0, position_ - 1);
        }
        pending_reward_ = position_ == 4 ? 1.0f : 0.0f;
        turn_ = 1 - turn_;
    }

    [[nodiscard]] int position() const noexcept { return position_; }
    [[nodiscard]] float last_reward() const noexcept { return pending_reward_; }

 private:
    std::unordered_map<AgentID, Tensor> observations() const {
        const float value = static_cast<float>(position_);
        Tensor obs(Shape{1}, std::span<const float>(&value, 1));
        return {{"a", obs.clone()}, {"b", obs.clone()}};
    }

    std::vector<AgentID> agents_;
    Space observation_ = Space::box(1, 0.0f, 4.0f);
    Space action_ = Space::discrete(2);
    int position_ = 0;
    int turn_ = 0;
    float pending_reward_ = 0.0f;
};

}  // namespace rl
}  // namespace nexus
