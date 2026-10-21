#pragma once

/**
 * @file multi_agent.hpp
 * @brief Es zamanli adim ve AEC (sirayla ajan) sozlesmesi.
 * `step` butun ajanlarin eylemini ayni anda uygular.
 * `step_one` yalnizca `current_agent()` icin gecerlidir.
 */

#include "nexus/rl/environment.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace nexus {
namespace rl {

using AgentID = std::string;

struct MultiStepResult {
    std::unordered_map<AgentID, Tensor> observations;
    std::unordered_map<AgentID, float> rewards;
    std::unordered_map<AgentID, bool> terminated;
    std::unordered_map<AgentID, bool> truncated;
    bool done = false;
};

class MultiAgentEnvironment {
 public:
    virtual ~MultiAgentEnvironment() = default;

    [[nodiscard]] virtual const std::vector<AgentID>& agents() const = 0;
    [[nodiscard]] virtual const Space& observation_space(const AgentID& agent) const = 0;
    [[nodiscard]] virtual const Space& action_space(const AgentID& agent) const = 0;
    [[nodiscard]] virtual bool simultaneous() const { return true; }
    [[nodiscard]] virtual AgentID current_agent() const { return agents().front(); }

    virtual std::unordered_map<AgentID, Tensor> reset(std::uint32_t seed) = 0;
    virtual MultiStepResult step(const std::unordered_map<AgentID, Tensor>& actions) = 0;

    /// AEC. Es zamanli ortamlar bunu desteklemez.
    virtual void step_one(const Tensor&) {
        throw std::logic_error("step_one is only valid for an AEC environment");
    }
};

}  // namespace rl
}  // namespace nexus
