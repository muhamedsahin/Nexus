/**
 * @file bench_phase4_custom.cpp
 * @brief Ozel aktivasyon: tek thread, is havuzu ve AVX2 seridi.
 *
 * Olcek cekirdegi bellek tavanini, Swish ise exp agirlikli havuz kazancini olcer.
 * AVX2 Swish sarmalayicisi seridi skalere acip exp cagirdigi icin ayri raporlanmaz;
 * gercek serit hizi olcek cekirdegindedir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/custom/activation_simd.hpp"
#include "nexus/custom/custom.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

float swish(float x) { return x / (1.0f + std::exp(-x)); }

double seconds_of(int warmup, int repeats, const auto& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        fn();
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / static_cast<double>(repeats);
}

}  // namespace

int main() {
    constexpr std::size_t kN = 1u << 20;
    std::vector<float> input(kN);
    std::vector<float> output(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        input[i] = 0.001f * static_cast<float>(static_cast<int>(i % 4000) - 2000);
    }

    auto scale = nexus::make_activation("Scale", [](float value) { return value * 2.0f; },
                                        [](float, float grad) { return grad * 2.0f; });
    auto scale_avx = nexus::make_activation("ScaleAvx", [](float value) { return value * 2.0f; });
    nexus::bind_avx2_elementwise(
        scale_avx, [](__m256 value) { return _mm256_mul_ps(value, _mm256_set1_ps(2.0f)); },
        [](__m256) { return _mm256_set1_ps(2.0f); });
    auto swish_act = nexus::make_activation("Swish", swish, [](float x, float grad) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-x));
        return grad * (sigmoid + x * sigmoid * (1.0f - sigmoid));
    });

    const double serial_scale = seconds_of(1, 6, [&] {
        for (std::size_t i = 0; i < kN; ++i) {
            output[i] = input[i] * 2.0f;
        }
    });
    const double pool_scale = seconds_of(1, 6, [&] { scale.forward_buffer(input.data(), output.data(), kN); });
    const double avx_scale = seconds_of(1, 6, [&] { scale_avx.forward_buffer(input.data(), output.data(), kN); });
    const double serial_swish = seconds_of(1, 3, [&] {
        for (std::size_t i = 0; i < kN; ++i) {
            output[i] = swish(input[i]);
        }
    });
    const double pool_swish = seconds_of(1, 3, [&] { swish_act.forward_buffer(input.data(), output.data(), kN); });

    volatile float sink = output[kN / 2];
    const double bytes = static_cast<double>(kN) * 2.0 * static_cast<double>(sizeof(float));
    auto row = [&](const char* name, double seconds, double baseline) {
        std::cout << "| " << name << " | " << seconds * 1.0e6 << " | " << (1.0 / seconds) << " | " << (bytes / seconds / 1.0e9)
                  << " | " << (baseline / seconds) << " |\n";
    };
    std::cout << "Nexus Faz 4 benchmark  N=" << kN << " sink=" << sink << "\n";
    std::cout << "| yol | us | cagri/s | GB/s | hizlanma |\n";
    row("skaler olcek", serial_scale, serial_scale);
    row("havuz olcek", pool_scale, serial_scale);
    row("AVX2 olcek", avx_scale, serial_scale);
    row("skaler swish", serial_swish, serial_swish);
    row("havuz swish", pool_swish, serial_swish);
    return std::isfinite(sink) ? 0 : 1;
}
