/**
 * @file bench_phase5_rl_throughput.cpp
 * @brief Vektor ortam adimi ve DQN guncelleme hizi.
 */

#include "nexus/rl/rl.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

int main() {
    constexpr int kEnvs = 128;
    constexpr int kSteps = 200;
    std::vector<std::unique_ptr<nexus::rl::Environment>> copies;
    copies.reserve(kEnvs);
    for (int i = 0; i < kEnvs; ++i) {
        copies.push_back(std::make_unique<nexus::rl::LineWorld>());
    }
    nexus::rl::VectorEnv env(std::move(copies));
    env.reset(1);
    std::vector<int> actions(static_cast<std::size_t>(kEnvs), 0);
    const auto step_start = std::chrono::steady_clock::now();
    for (int step = 0; step < kSteps; ++step) {
        env.step(actions);
    }
    const double step_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    const double transitions = static_cast<double>(kEnvs) * kSteps;
    const double steps_per_second = transitions / step_seconds;

    nexus::rl::DQNAgent::Config config;
    config.batch = 128;
    config.warmup = 128;
    config.buffer_capacity = 2048;
    config.hidden = 64;
    config.prioritized = false;
    nexus::rl::LineWorld line;
    nexus::rl::DQNAgent agent(line.observation_dim(), line.action_count(), config);
    agent.learn(line, config.warmup + 1);
    constexpr int kUpdates = 40;
    const auto update_start = std::chrono::steady_clock::now();
    for (int update = 0; update < kUpdates; ++update) {
        agent.update();
    }
    const double update_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
    const double updates_per_second = static_cast<double>(kUpdates) / update_seconds;

    std::cout << "Nexus Faz 5 benchmark\n";
    std::cout << "| islem | hiz | not |\n";
    std::cout << "| VectorEnv adim | " << steps_per_second << " adim/s | " << kEnvs << " x LineWorld |\n";
    std::cout << "| DQN guncelleme | " << updates_per_second << " guncelleme/s | batch 128 |\n";
    return std::isfinite(steps_per_second) && std::isfinite(updates_per_second) ? 0 : 1;
}
