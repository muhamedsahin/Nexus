/**
 * @file test_phase5_rl_and_marl.cpp
 * @brief GAE, PER, CartPole, DQN, PPO, IPPO, MAPPO, SAC ve ELO.
 */

#include "nexus/rl/rl.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

int g_failed = 0;

void expect(bool condition, const char* name) {
    if (condition) {
        std::cout << "  PASS  " << name << "\n";
        return;
    }
    std::cout << "  FAIL  " << name << "\n";
    ++g_failed;
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 5 pekistirmeli ogrenme testi\n";
    try {
        std::vector<float> rewards{1.0f, 1.0f};
        std::vector<float> values{0.0f, 0.0f};
        std::vector<std::uint8_t> terminated{0, 0};
        std::vector<float> advantages;
        std::vector<float> returns;
        nexus::rl::compute_gae(rewards, values, terminated, 1.0f, 1.0f, 0.0f, advantages, returns);
        expect(std::fabs(advantages[0] - 2.0f) < 1.0e-5f && std::fabs(advantages[1] - 1.0f) < 1.0e-5f, "gae advantages");

        nexus::rl::ReplayBuffer replay(2, 1);
        const float first_obs[1] = {1.0f};
        const float second_obs[1] = {2.0f};
        const float third_obs[1] = {3.0f};
        replay.add(first_obs, 1, 0.0f, first_obs, false);
        replay.add(second_obs, 2, 0.0f, second_obs, false);
        replay.add(third_obs, 3, 0.0f, third_obs, false);
        expect(replay.size() == 2 && replay.action_at(0) == 3 && replay.action_at(1) == 2, "replay ring overwrite");
        const auto address = reinterpret_cast<std::uintptr_t>(replay.observation_data());
        expect(address % 64 == 0, "replay 64-byte alignment");

        nexus::rl::PrioritizedReplayBuffer prioritized(8, 1, 0.6f);
        const float priorities[4] = {1.0f, 1.0f, 1.0f, 10.0f};
        for (int i = 0; i < 4; ++i) {
            const float value = static_cast<float>(i);
            prioritized.add(&value, i, 0.0f, &value, false, priorities[i]);
        }
        nexus::rl::Rng rng(7);
        int hits = 0;
        constexpr int kDraws = 2000;
        for (int draw = 0; draw < kDraws; ++draw) {
            auto sample = prioritized.sample(1, 0.4f, rng);
            if (sample.indices[0] == 3) {
                ++hits;
            }
        }
        const float fraction = static_cast<float>(hits) / static_cast<float>(kDraws);
        std::cout << "  info  per_top_fraction=" << fraction << "\n";
        expect(fraction > 0.45f, "prioritized sample favors large td");

        nexus::rl::CartPole right_cart;
        right_cart.reset(1);
        for (int step = 0; step < 20; ++step) {
            right_cart.step(nexus::rl::discrete_action_tensor(1));
        }
        nexus::rl::CartPole left_cart;
        left_cart.reset(1);
        for (int step = 0; step < 20; ++step) {
            left_cart.step(nexus::rl::discrete_action_tensor(0));
        }
        expect(std::isfinite(right_cart.x()) && right_cart.x() > left_cart.x(), "cartpole force pushes right");

        nexus::rl::GridWorld grid;
        grid.reset(1);
        nexus::rl::StepResult right = grid.step(nexus::rl::discrete_action_tensor(0));
        expect(right.observation[1] == 1.0f, "gridworld moves right");

        std::vector<std::unique_ptr<nexus::rl::Environment>> copies;
        for (int i = 0; i < 16; ++i) {
            copies.push_back(std::make_unique<nexus::rl::LineWorld>());
        }
        nexus::rl::VectorEnv vector_env(std::move(copies));
        nexus::Tensor stacked = vector_env.reset(3);
        expect(stacked.numel() == 16 * 5 && stacked[0] == 1.0f, "vector env stacks observations");

        nexus::rl::TurnLine turn;
        turn.reset(1);
        expect(turn.current_agent() == "a", "aec starts at agent a");
        turn.step_one(nexus::rl::discrete_action_tensor(0));
        expect(turn.current_agent() == "b" && turn.position() == 1, "aec alternates");

        nexus::rl::League league;
        const int strong = league.add_player();
        const int weak = league.add_player();
        league.save(strong, {1.0f});
        league.save(weak, {0.0f});
        nexus::rl::Duel duel;
        for (int game = 0; game < 12; ++game) {
            const float score = nexus::rl::scripted_duel(duel, 1, 1);
            league.result(strong, weak, score > 0.0f ? 1.0f : 0.0f);
        }
        expect(league.elo(strong) > league.elo(weak), "self-play elo rewards the winner");
        expect(league.weights(strong)[0] == 1.0f, "historical checkpoint roundtrip");

        nexus::rl::LineWorld line;
        nexus::rl::DQNAgent::Config dqn_config;
        dqn_config.seed = 2;
        dqn_config.hidden = 32;
        dqn_config.buffer_capacity = 1024;
        dqn_config.batch = 32;
        dqn_config.warmup = 64;
        nexus::rl::DQNAgent dqn(line.observation_dim(), line.action_count(), dqn_config);
        const float dqn_before = dqn.evaluate(line, 20);
        dqn.learn(line, 900);
        const float dqn_after = dqn.evaluate(line, 20);
        std::cout << "  info  dqn " << dqn_before << " -> " << dqn_after << "\n";
        expect(dqn_after > dqn_before + 0.2f, "dqn return increases");

        nexus::rl::PPOAgent::Config ppo_config;
        ppo_config.seed = 3;
        ppo_config.hidden = 32;
        ppo_config.epochs = 4;
        nexus::rl::PPOAgent ppo(line.observation_dim(), line.action_count(), ppo_config);
        const float ppo_before = ppo.evaluate(line, 15);
        ppo.learn(line, 8, 64);
        const float ppo_after = ppo.evaluate(line, 15);
        std::cout << "  info  ppo " << ppo_before << " -> " << ppo_after << "\n";
        expect(ppo_after > ppo_before + 0.15f, "ppo return increases");

        auto a2c = nexus::rl::ActorCriticAgent::discrete(line.observation_dim(), line.action_count(), 4);
        const float a2c_before = a2c.evaluate(line, 15);
        a2c.learn(line, 8, 64);
        const float a2c_after = a2c.evaluate(line, 15);
        std::cout << "  info  a2c " << a2c_before << " -> " << a2c_after << "\n";
        expect(a2c_after > a2c_before + 0.1f, "actor-critic return increases");

        nexus::rl::ButtonWorld buttons;
        nexus::rl::MultiAgentTrainer trainer;
        const nexus::rl::MarlReport ippo = trainer.train_independent(buttons, 14, 36, 5);
        std::cout << "  info  ippo " << ippo.first_return << " -> " << ippo.last_return << "\n";
        expect(ippo.last_return > ippo.first_return + 0.4f, "ippo cooperative return increases");
        const nexus::rl::MarlReport mappo = trainer.train_centralized(buttons, 14, 36, 6);
        std::cout << "  info  mappo " << mappo.first_return << " -> " << mappo.last_return << "\n";
        expect(mappo.last_return > mappo.first_return + 0.4f, "mappo centralized return increases");

        nexus::rl::TeamButton trio(3);
        const nexus::rl::MarlReport trio_ippo = trainer.train_independent(trio, 2, 12, 11);
        const nexus::rl::MarlReport trio_mappo = trainer.train_centralized(trio, 2, 12, 12);
        std::cout << "  info  trio ippo " << trio_ippo.last_return << " mappo " << trio_mappo.last_return << "\n";
        expect(std::isfinite(trio_ippo.last_return) && std::isfinite(trio_mappo.last_return), "three-agent ippo and mappo run");
        expect(trio.agents().size() == 3, "team button has three agents");

        nexus::rl::PointMass point;
        nexus::rl::SACAgent sac(1);
        const float sac_before = sac.evaluate(point, 10);
        sac.learn(point, 500);
        const float sac_after = sac.evaluate(point, 10);
        std::cout << "  info  sac " << sac_before << " -> " << sac_after << "\n";
        expect(sac_after > sac_before + 0.5f, "sac return increases");

        nexus::rl::ContinuousActorCritic continuous(1, 9);
        const float continuous_before = continuous.evaluate(point, 10);
        continuous.learn(point, 25, 20);
        const float continuous_after = continuous.evaluate(point, 10);
        std::cout << "  info  continuous_a2c " << continuous_before << " -> " << continuous_after << "\n";
        expect(continuous_after > continuous_before + 0.5f, "continuous actor-critic return increases");
    } catch (const std::exception& error) {
        std::cout << "  FAIL  istisna: " << error.what() << "\n";
        return 1;
    }
    if (g_failed != 0) {
        std::cout << g_failed << " kontrol basarisiz\n";
        return 1;
    }
    std::cout << "tum kontroller gecti\n";
    return 0;
}
