#pragma once

/**
 * @file aligned_alloc.hpp
 * @brief 64 bayt hizali std::vector allocator. Replay halkasi bunu kullanir.
 */

#include "nexus/core/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexus {
namespace rl {

template <typename T>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        return static_cast<T*>(nexus::allocate_aligned(n * sizeof(T)));
    }

    void deallocate(T* pointer, std::size_t) noexcept { nexus::release_owned(pointer); }
};

template <typename T, typename U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept {
    return true;
}

template <typename T, typename U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept {
    return false;
}

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

}  // namespace rl
}  // namespace nexus
