/**
 * @file test_phase13_isa.cpp
 * @brief GEMM cekirdek adi ve referansa yakinlik.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/core/device.hpp"
#include "matrix_pro/cpu/gemm.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void expect(bool condition, const char* name) {
    if (condition) {
        std::cout << "  PASS  " << name << "\n";
        return;
    }
    std::cout << "  FAIL  " << name << "\n";
    ++g_failed;
}

float max_abs(const std::vector<float>& left, const std::vector<float>& right) {
    float worst = 0.0f;
    for (std::size_t i = 0; i < left.size(); ++i) {
        worst = std::max(worst, std::fabs(left[i] - right[i]));
    }
    return worst;
}

void check_gemm(int m, int n, int k, float beta, float tolerance = 1.0e-4f) {
    std::vector<float> a(static_cast<std::size_t>(m) * k);
    std::vector<float> b(static_cast<std::size_t>(k) * n);
    std::vector<float> c(static_cast<std::size_t>(m) * n, 0.25f);
    std::vector<float> reference = c;
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = 0.01f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = 0.02f * static_cast<float>(static_cast<int>(i % 7) - 3);
    }
    matrix_pro::gemm_f32(m, n, k, a.data(), k, b.data(), n, c.data(), n, 1.0f, beta);
    matrix_pro::gemm_reference(m, n, k, a.data(), k, b.data(), n, reference.data(), n, 1.0f, beta);
    const float error = max_abs(c, reference);
    const bool ok = error < tolerance && std::isfinite(error);
    if (!ok) {
        std::cout << "  hata " << m << "x" << n << "x" << k << " = " << error << "\n";
    }
    expect(ok, "gemm matches the reference");
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 13 ISA testi\n";
    const matrix_pro::CpuIsa isa = matrix_pro::detect_cpu_isa();
    const std::string name = matrix_pro::gemm_kernel_name();
    std::cout << "  cekirdek " << name << "\n";
    if (isa.avx2 && isa.fma) {
        expect(name == "avx2-4x16", "avx2 kernel is selected ahead of the serial avx512 path");
    } else {
        expect(name == "reference", "scalar kernel is selected without avx2");
    }
    expect(name != "avx512-8x16", "serial avx512 kernel is not selected");
    check_gemm(64, 64, 64, 0.0f);
    check_gemm(65, 17, 33, 0.0f);
    check_gemm(48, 40, 24, 0.5f);
    check_gemm(80, 1024, 128, 0.0f, 1.0e-3f);
    check_gemm(70, 1032, 160, 0.5f, 2.0e-3f);
    check_gemm(128, 2048, 256, 0.0f, 2.0e-3f);
    check_gemm(72, 1024, 130, 0.0f, 2.0e-3f);
    if (g_failed != 0) {
        std::cout << g_failed << " kontrol basarisiz\n";
        return 1;
    }
    std::cout << "tum kontroller gecti\n";
    return 0;
}
