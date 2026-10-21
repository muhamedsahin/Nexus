#pragma once

/**
 * @file gradient_sync.hpp
 * @brief Replika gradyanlarını toplar ve ortalar, parametreleri yayınlar.
 *
 * Thread-safety: her iki fonksiyon da bütün replikaların backward'ı bittikten
 * sonra tek thread'den çağrılır. Replika içi forward eşzamanlıdır; bu fonksiyonlar
 * o pencerenin dışındadır. `accumulate_grad_locked` burada gerekmez çünkü
 * çağıran zaten bir bariyerin (join) ardındadır.
 */

#include "nexus_model/core/parameter.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace nexus_train {

inline void allreduce_mean_gradients(const std::vector<std::vector<nexus_model::Parameter*>>& replicas) {
  if (replicas.empty()) {
    return;
  }
  const std::size_t width = replicas.front().size();
  for (const auto& replica : replicas) {
    if (replica.size() != width) {
      throw std::invalid_argument("NexusTrain: replica parameter counts differ");
    }
  }
  const float inv = 1.f / static_cast<float>(replicas.size());
  for (std::size_t index = 0; index < width; ++index) {
    nexus_model::Parameter* root = replicas.front()[index];
    if (root == nullptr || !root->requires_grad) {
      continue;
    }
    float* acc = root->grad.data();
    const std::size_t n = root->grad.numel();
    for (std::size_t replica = 1; replica < replicas.size(); ++replica) {
      nexus_model::Parameter* other = replicas[replica][index];
      if (other == nullptr || other->grad.numel() != n) {
        throw std::invalid_argument("NexusTrain: replica gradient shapes differ");
      }
      const float* src = other->grad.data();
      for (std::size_t k = 0; k < n; ++k) acc[k] += src[k];
    }
    for (std::size_t k = 0; k < n; ++k) acc[k] *= inv;
    for (std::size_t replica = 1; replica < replicas.size(); ++replica) {
      float* dst = replicas[replica][index]->grad.data();
      std::memcpy(dst, acc, n * sizeof(float));
    }
  }
}

inline void broadcast_parameters(const std::vector<nexus_model::Parameter*>& source,
                                 const std::vector<std::vector<nexus_model::Parameter*>>& replicas) {
  if (replicas.empty()) {
    return;
  }
  for (std::size_t index = 0; index < source.size(); ++index) {
    const float* src = source[index]->data.data();
    const std::size_t n = source[index]->data.numel();
    for (std::size_t replica = 1; replica < replicas.size(); ++replica) {
      nexus_model::Parameter* dst = replicas[replica][index];
      if (dst->data.numel() != n) {
        throw std::invalid_argument("NexusTrain: replica parameter shapes differ");
      }
      std::memcpy(dst->data.data(), src, n * sizeof(float));
    }
  }
}

}  // namespace nexus_train
