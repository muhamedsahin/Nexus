#pragma once

/**
 * @file config.hpp
 * @brief Tohum ve calistirma politikasi. Politika, Linear carpiminin CPU cekirdeginde
 * mi yoksa MatrixFlash hibrit zamanlayicisinda mi kosacagini soyler.
 */

#include "nexus/core/device.hpp"

#include <cstdint>

namespace nexus {

inline std::uint32_t& global_seed() {
    static std::uint32_t seed = 1;
    return seed;
}

inline void set_global_seed(std::uint32_t seed) { global_seed() = seed == 0 ? 1u : seed; }

class ExecutionPolicy {
 public:
    static ExecutionPolicy cpu() { return ExecutionPolicy{Kind::Cpu, 0}; }
    static ExecutionPolicy cuda(int index = 0) { return ExecutionPolicy{Kind::Cuda, index}; }
    static ExecutionPolicy hybrid_cpu_multi_gpu() { return ExecutionPolicy{Kind::Hybrid, 0}; }

    [[nodiscard]] bool use_cuda() const noexcept { return kind_ == Kind::Cuda; }
    [[nodiscard]] bool hybrid() const noexcept { return kind_ == Kind::Hybrid; }
    [[nodiscard]] int device_index() const noexcept { return index_; }

    [[nodiscard]] Device device() const {
        if (kind_ == Kind::Hybrid) {
            return Device::hybrid();
        }
        if (kind_ == Kind::Cuda) {
            return Device::cuda(index_);
        }
        return Device::cpu();
    }

 private:
    enum class Kind { Cpu, Cuda, Hybrid };
    ExecutionPolicy(Kind kind, int index) : kind_(kind), index_(index) {}
    Kind kind_ = Kind::Cpu;
    int index_ = 0;
};

}  // namespace nexus
