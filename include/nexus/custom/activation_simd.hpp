#pragma once

/**
 * @file activation_simd.hpp
 * @brief AVX2 seritli aktivasyon kancasi. Bu basligi iceren ceviri birimi
 * AVX2 uretebilmelidir (`/arch:AVX2` veya `-mavx2`).
 *
 * `map` ve `dfdx` serit-serit calisir: bir serit digerine bakmaz.
 * Kuyruk, sifirlanmis bir seritte tek eleman olarak ayni fonksiyonlara gider.
 */

#include "nexus/custom/activation.hpp"

#include <immintrin.h>

#include <span>

namespace nexus {

template <typename MapFn, typename DerivFn>
void bind_avx2_elementwise(CustomActivation& activation, MapFn map, DerivFn dfdx) {
    activation.set_batch(
        [map](std::span<const float> in, std::span<float> out) {
            std::size_t i = 0;
            for (; i + 8 <= in.size(); i += 8) {
                const __m256 y = map(_mm256_loadu_ps(in.data() + i));
                _mm256_storeu_ps(out.data() + i, y);
            }
            for (; i < in.size(); ++i) {
                alignas(32) float lane[8] = {};
                lane[0] = in[i];
                const __m256 y = map(_mm256_load_ps(lane));
                _mm256_store_ps(lane, y);
                out[i] = lane[0];
            }
        },
        [dfdx](std::span<const float> x, std::span<const float> grad_out, std::span<float> grad_in) {
            std::size_t i = 0;
            for (; i + 8 <= x.size(); i += 8) {
                const __m256 slope = dfdx(_mm256_loadu_ps(x.data() + i));
                const __m256 dx = _mm256_mul_ps(_mm256_loadu_ps(grad_out.data() + i), slope);
                _mm256_storeu_ps(grad_in.data() + i, dx);
            }
            for (; i < x.size(); ++i) {
                alignas(32) float lane[8] = {};
                lane[0] = x[i];
                const __m256 slope = dfdx(_mm256_load_ps(lane));
                _mm256_store_ps(lane, slope);
                grad_in[i] = grad_out[i] * lane[0];
            }
        });
}

}  // namespace nexus
