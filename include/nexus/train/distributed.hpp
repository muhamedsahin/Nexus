#pragma once

/**
 * @file distributed.hpp
 * @brief Surec ici rank tamponlarinin ortalamasi. MatrixFlash halka all-reduce kullanir.
 * Tek rank kimliktir. Gradyan senkronu egitim adimindan once cagrilir.
 */

#include "matrix_pro/distributed/collectives.hpp"

#include <stdexcept>
#include <vector>

namespace nexus {
namespace train {

inline void all_reduce_mean(std::vector<std::vector<float>>& ranks) {
    if (ranks.empty()) {
        throw std::invalid_argument("all_reduce_mean requires a rank");
    }
    const std::size_t count = ranks.front().size();
    if (count == 0) {
        return;
    }
    std::vector<matrix_pro::CollectiveBuffer> buffers(ranks.size());
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        if (ranks[rank].size() != count) {
            throw std::invalid_argument("all_reduce_mean rank lengths differ");
        }
        buffers[rank].data = ranks[rank].data();
        buffers[rank].count = count;
        buffers[rank].device = matrix_pro::DeviceId::cpu();
    }
    matrix_pro::all_reduce(buffers.data(), static_cast<int>(buffers.size()), matrix_pro::ReduceOp::Mean,
                           matrix_pro::CollectiveAlgorithm::Ring);
}

}  // namespace train
}  // namespace nexus
