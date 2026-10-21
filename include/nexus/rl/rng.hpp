#pragma once

/**
 * @file rng.hpp
 * @brief xorshift32. Oyun ve ag baslangici ayni tohumla tekrarlanir.
 */

#include <cstdint>

namespace nexus {
namespace rl {

class Rng {
 public:
    explicit Rng(std::uint32_t seed = 1) : state_(seed == 0 ? 1u : seed) {}

    std::uint32_t next_u32() {
        std::uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        return x;
    }

    float uniform() { return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f); }

    int uniform_int(int n) {
        if (n <= 1) {
            return 0;
        }
        return static_cast<int>(next_u32() % static_cast<std::uint32_t>(n));
    }

    float normal() {
        const float u1 = uniform() + 1.0e-7f;
        const float u2 = uniform();
        constexpr float kPi = 3.14159265f;
        const float r = std::sqrt(-2.0f * std::log(u1));
        return r * std::cos(2.0f * kPi * u2);
    }

 private:
    std::uint32_t state_;
};

}  // namespace rl
}  // namespace nexus
