#pragma once

/**
 * @file space.hpp
 * @brief Gymnasium tarzi gozlem ve eylem uzayi. Ornek, host tensördur.
 */

#include "nexus/core/tensor.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexus {
namespace rl {

enum class SpaceKind { Discrete, Box, MultiDiscrete };

class Space {
 public:
    static Space discrete(int n) {
        if (n <= 0) {
            throw std::invalid_argument("DiscreteSpace n must be positive");
        }
        Space space;
        space.kind_ = SpaceKind::Discrete;
        space.nvec_ = {n};
        return space;
    }

    static Space box(int dim, float low, float high) {
        if (dim <= 0 || !(high > low)) {
            throw std::invalid_argument("BoxSpace dim and bounds are invalid");
        }
        Space space;
        space.kind_ = SpaceKind::Box;
        space.nvec_ = {dim};
        space.low_ = low;
        space.high_ = high;
        return space;
    }

    static Space multi_discrete(std::vector<int> nvec) {
        if (nvec.empty()) {
            throw std::invalid_argument("MultiDiscreteSpace is empty");
        }
        for (int n : nvec) {
            if (n <= 0) {
                throw std::invalid_argument("MultiDiscreteSpace entry must be positive");
            }
        }
        Space space;
        space.kind_ = SpaceKind::MultiDiscrete;
        space.nvec_ = std::move(nvec);
        return space;
    }

    [[nodiscard]] SpaceKind kind() const noexcept { return kind_; }
    [[nodiscard]] int n() const { return nvec_.front(); }
    [[nodiscard]] const std::vector<int>& nvec() const noexcept { return nvec_; }
    [[nodiscard]] float low() const noexcept { return low_; }
    [[nodiscard]] float high() const noexcept { return high_; }

    [[nodiscard]] int flat_dim() const {
        if (kind_ == SpaceKind::Box) {
            return nvec_.front();
        }
        if (kind_ == SpaceKind::Discrete) {
            return nvec_.front();
        }
        return static_cast<int>(nvec_.size());
    }

    [[nodiscard]] bool contains(const Tensor& sample) const {
        if (sample.space() != MemorySpace::Host || sample.numel() == 0) {
            return false;
        }
        if (kind_ == SpaceKind::Discrete) {
            if (sample.numel() != 1) {
                return false;
            }
            const int action = static_cast<int>(std::lround(sample[0]));
            return action >= 0 && action < nvec_.front();
        }
        if (kind_ == SpaceKind::Box) {
            if (static_cast<int>(sample.numel()) != nvec_.front()) {
                return false;
            }
            for (std::size_t i = 0; i < sample.numel(); ++i) {
                if (sample[i] < low_ || sample[i] > high_ || !std::isfinite(sample[i])) {
                    return false;
                }
            }
            return true;
        }
        if (static_cast<int>(sample.numel()) != static_cast<int>(nvec_.size())) {
            return false;
        }
        for (std::size_t i = 0; i < nvec_.size(); ++i) {
            const int action = static_cast<int>(std::lround(sample[i]));
            if (action < 0 || action >= nvec_[i]) {
                return false;
            }
        }
        return true;
    }

 private:
    SpaceKind kind_ = SpaceKind::Discrete;
    std::vector<int> nvec_{1};
    float low_ = 0.0f;
    float high_ = 1.0f;
};

}  // namespace rl
}  // namespace nexus
