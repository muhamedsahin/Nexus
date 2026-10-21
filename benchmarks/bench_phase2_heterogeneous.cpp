/**
 * @file bench_phase2_heterogeneous.cpp
 * @brief Saf CPU, saf GPU ve hibrit GEMM hizlari.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/detail/cuda_hooks.hpp"
#include "matrix_pro/hardware.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

double gflops_of(int n, double seconds) {
    const double flops = 2.0 * static_cast<double>(n) * n * n;
    return seconds > 0.0 ? flops / seconds / 1.0e9 : 0.0;
}

}  // namespace

int main() {
    constexpr int kN = 512;
    const std::size_t cells = static_cast<std::size_t>(kN) * kN;
    std::vector<float> a(cells), b(cells), c(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        a[i] = 0.01f;
        b[i] = 0.02f;
    }
    matrix_pro::gemm_f32(kN, kN, kN, a.data(), kN, b.data(), kN, c.data(), kN, 1.0f, 0.0f);
    const auto cpu_start = std::chrono::steady_clock::now();
    constexpr int kRepeats = 3;
    for (int i = 0; i < kRepeats; ++i) {
        matrix_pro::gemm_f32(kN, kN, kN, a.data(), kN, b.data(), kN, c.data(), kN, 1.0f, 0.0f);
    }
    const double cpu_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_start).count() / kRepeats;

    double gpu_seconds = 0;
    double hybrid_seconds = 0;
    int cpu_rows = 0;
    int gpu_rows = 0;
#if defined(MATRIX_PRO_ENABLE_CUDA)
    if (matrix_pro::hw::cuda_device_count() > 0) {
        float* device_a = matrix_pro::hw::cuda_alloc(0, cells);
        float* device_b = matrix_pro::hw::cuda_alloc(0, cells);
        float* device_c = matrix_pro::hw::cuda_alloc(0, cells);
        matrix_pro::hw::cuda_copy_h2d(0, device_a, a.data(), cells);
        matrix_pro::hw::cuda_copy_h2d(0, device_b, b.data(), cells);
        matrix_pro::hw::cuda_sync(0);
        matrix_pro::hw::cuda_gemm(0, kN, kN, kN, device_a, kN, device_b, kN, device_c, kN, 1.0f, 0.0f);
        matrix_pro::hw::cuda_sync(0);
        const auto gpu_start = std::chrono::steady_clock::now();
        for (int i = 0; i < kRepeats; ++i) {
            matrix_pro::hw::cuda_gemm(0, kN, kN, kN, device_a, kN, device_b, kN, device_c, kN, 1.0f, 0.0f);
            matrix_pro::hw::cuda_sync(0);
        }
        gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_start).count() / kRepeats;
        matrix_pro::hw::cuda_free(0, device_a);
        matrix_pro::hw::cuda_free(0, device_b);
        matrix_pro::hw::cuda_free(0, device_c);

        matrix_pro::HeterogeneousScheduler& scheduler = matrix_pro::HeterogeneousScheduler::instance();
        scheduler.calibrate();
        scheduler.gemm(a.data(), kN, b.data(), kN, c.data(), kN, kN, kN, kN, 1.0f, 0.0f);
        const auto hybrid_start = std::chrono::steady_clock::now();
        for (int i = 0; i < kRepeats; ++i) {
            scheduler.gemm(a.data(), kN, b.data(), kN, c.data(), kN, kN, kN, kN, 1.0f, 0.0f);
        }
        hybrid_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - hybrid_start).count() / kRepeats;
        cpu_rows = scheduler.last_split().cpu_rows;
        gpu_rows = scheduler.last_split().gpu_rows;
    }
#endif
    const double bytes = static_cast<double>(cells) * 3.0 * sizeof(float);
    std::vector<float> x(cells, 1.0f), y(cells, 2.0f), z(cells);
    const auto add_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 8; ++i) {
        matrix_pro::elementwise_add(x.data(), y.data(), z.data(), static_cast<int>(cells));
    }
    const double add_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - add_start).count() / 8.0;

    std::cout << "Nexus Faz 2 benchmark  N=" << kN << "\n";
    std::cout << "| mod | sure us | GFLOPS | GB/s | not |\n";
    std::cout << "| CPU GEMM | " << cpu_seconds * 1.0e6 << " | " << gflops_of(kN, cpu_seconds) << " |  | AVX |\n";
    std::cout << "| GPU GEMM | " << gpu_seconds * 1.0e6 << " | " << gflops_of(kN, gpu_seconds) << " |  | FP32 pedantic |\n";
    std::cout << "| Hybrid GEMM | " << hybrid_seconds * 1.0e6 << " | " << gflops_of(kN, hybrid_seconds)
              << " |  | cpu_rows=" << cpu_rows << " gpu_rows=" << gpu_rows << " |\n";
    std::cout << "| CPU add | " << add_seconds * 1.0e6 << " |  | " << (bytes / add_seconds / 1.0e9) << " | elementwise |\n";
    std::cout << "sink " << c[0] << " " << z[0] << "\n";
    return std::isfinite(c[0]) && std::isfinite(z[0]) ? 0 : 1;
}
