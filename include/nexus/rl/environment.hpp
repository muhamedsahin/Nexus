#pragma once

/**
 * @file environment.hpp
 * @brief Tek ajanli ortam. `reset` ve `step` sahip olunan host tensör dondurur.
 * Ayrık eylem, tek elemanli tensörde tamsayi indekstir.
 */

#include "nexus/rl/space.hpp"

#include <string>

namespace nexus {
namespace rl {

struct StepResult {
    Tensor observation;
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
    std::string info;
};

class Environment {
 public:
    virtual ~Environment() = default;

    [[nodiscard]] virtual const Space& observation_space() const = 0;
    [[nodiscard]] virtual const Space& action_space() const = 0;
    [[nodiscard]] virtual int observation_dim() const { return observation_space().flat_dim(); }
    [[nodiscard]] virtual int action_count() const { return action_space().n(); }

    virtual Tensor reset(std::uint32_t seed) = 0;
    virtual StepResult step(const Tensor& action) = 0;
};

inline int read_discrete_action(const Tensor& action, int count) {
    if (action.numel() != 1) {
        throw std::invalid_argument("discrete action tensor must have one element");
    }
    const int index = static_cast<int>(std::lround(action[0]));
    if (index < 0 || index >= count) {
        throw std::invalid_argument("discrete action is outside the space");
    }
    return index;
}

inline Tensor discrete_action_tensor(int index) {
    const float value = static_cast<float>(index);
    return Tensor(Shape{1}, std::span<const float>(&value, 1));
}

}  // namespace rl
}  // namespace nexus
