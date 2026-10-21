/**
 * @file main.cpp
 * CPU GEMM, hibrit GEMM ve iki ranklik halka ortalama.
 */

#include "nexus/nexus.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr int kN = 256;
    const std::size_t cells = static_cast<std::size_t>(kN) * kN;
    std::vector<float> a(cells, 0.01f);
    std::vector<float> b(cells, 0.02f);
    std::vector<float> c(cells, 0.0f);
    matrix_pro::gemm_f32(kN, kN, kN, a.data(), kN, b.data(), kN, c.data(), kN, 1.0f, 0.0f);
    constexpr int kRepeats = 4;
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRepeats; ++i) {
        matrix_pro::gemm_f32(kN, kN, kN, a.data(), kN, b.data(), kN, c.data(), kN, 1.0f, 0.0f);
    }
    const double cpu_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_start).count() / kRepeats;
    const double flops = 2.0 * kN * kN * kN;

    matrix_pro::HeterogeneousScheduler::instance().gemm(a.data(), kN, b.data(), kN, c.data(), kN, kN, kN, kN, 1.0f, 0.0f);
    const auto hybrid_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRepeats; ++i) {
        matrix_pro::HeterogeneousScheduler::instance().gemm(a.data(), kN, b.data(), kN, c.data(), kN, kN, kN, kN, 1.0f,
                                                           0.0f);
    }
    const double hybrid_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - hybrid_start).count() / kRepeats;

    std::vector<std::vector<float>> ranks{{2.0f, 8.0f, 4.0f}, {6.0f, 0.0f, 2.0f}};
    nexus::train::all_reduce_mean(ranks);
    const bool reduced = std::fabs(ranks[0][0] - 4.0f) < 1.0e-5f && std::fabs(ranks[1][2] - 3.0f) < 1.0e-5f;

    std::cout << "Nexus hibrit kume\n";
    std::cout << "| islem | sure us | GFLOPS |\n";
    std::cout << "| CPU GEMM | " << cpu_seconds * 1.0e6 << " | " << flops / cpu_seconds / 1.0e9 << " |\n";
    std::cout << "| Hibrit GEMM | " << hybrid_seconds * 1.0e6 << " | " << flops / hybrid_seconds / 1.0e9 << " |\n";
    std::cout << "allreduce_mean=" << ranks[0][0] << "\n";
    return std::isfinite(c[0]) && reduced ? 0 : 1;
}
