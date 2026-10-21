/**
 * @file bench_phase3_optim_model.cpp
 * @brief Adam adimi ve Linear ileri gecis hizlari.
 */

#include "matrix_pro/hardware.hpp"
#include "nexus_model/core/execution_policy.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_optim/device/hybrid_update.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"

#include <chrono>
#include <iostream>
#include <vector>

namespace {

void hybrid_gemm(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n, int k) {
    matrix_pro::HeterogeneousScheduler::instance().gemm(a, lda, b, ldb, c, ldc, m, n, k, 1.0f, 0.0f);
}

}  // namespace

int main() {
    constexpr std::size_t kN = 1u << 20;
    std::vector<float> params(kN, 0.1f), grads(kN, 0.01f), moment(kN, 0.0f), second(kN, 0.0f);
    nexus_optim::AdamCoeffs coeffs;
    coeffs.lr = 1.0e-3f;
    coeffs.one_minus_beta1 = 0.1f;
    coeffs.one_minus_beta2 = 0.001f;
    coeffs.inv_bc1 = 1.0f / 0.1f;
    coeffs.inv_bc2 = 1.0f / 0.001f;
    nexus_optim::adam_update(params.data(), grads.data(), moment.data(), second.data(), nullptr, kN, coeffs);
    const auto cpu_start = std::chrono::steady_clock::now();
    constexpr int kRepeats = 8;
    for (int i = 0; i < kRepeats; ++i) {
        nexus_optim::adam_update(params.data(), grads.data(), moment.data(), second.data(), nullptr, kN, coeffs);
    }
    const double cpu_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_start).count() / kRepeats;
    const auto hybrid_counts = nexus_optim::adam_update_hybrid(params.data(), grads.data(), moment.data(), second.data(),
                                                              nullptr, kN, coeffs);
    const auto hybrid_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRepeats; ++i) {
        nexus_optim::adam_update_hybrid(params.data(), grads.data(), moment.data(), second.data(), nullptr, kN, coeffs);
    }
    const double hybrid_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - hybrid_start).count() / kRepeats;
    const double bytes = static_cast<double>(kN) * 4.0 * 4.0;

    nexus_model::Linear layer(256, 256, true);
    nexus_model::Tensor input = nexus_model::Tensor::zeros({128, 256});
    for (std::size_t i = 0; i < input.numel(); ++i) {
        input.data()[i] = 0.01f;
    }
    nexus_model::set_matmul_policy(nexus_model::MatmulPolicy::CpuKernel);
    layer.forward(input);
    const auto linear_cpu_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRepeats; ++i) {
        layer.forward(input);
    }
    const double linear_cpu =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - linear_cpu_start).count() / kRepeats;
    nexus_model::set_packed_gemm(&hybrid_gemm);
    nexus_model::set_matmul_policy(nexus_model::MatmulPolicy::Hybrid);
    layer.forward(input);
    const auto linear_hybrid_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kRepeats; ++i) {
        layer.forward(input);
    }
    const double linear_hybrid =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - linear_hybrid_start).count() / kRepeats;
    nexus_model::set_matmul_policy(nexus_model::MatmulPolicy::CpuKernel);
    const double linear_flops = 2.0 * 128.0 * 256.0 * 256.0;

    std::cout << "Nexus Faz 3 benchmark\n";
    std::cout << "| islem | sure us | GB/s veya GFLOPS | not |\n";
    std::cout << "| Adam CPU | " << cpu_seconds * 1.0e6 << " | " << bytes / cpu_seconds / 1.0e9 << " GB/s | n=" << kN
              << " |\n";
    std::cout << "| Adam hibrit | " << hybrid_seconds * 1.0e6 << " | " << bytes / hybrid_seconds / 1.0e9
              << " GB/s | cpu=" << hybrid_counts.cpu << " gpu=" << hybrid_counts.gpu << " |\n";
    std::cout << "| Linear CPU | " << linear_cpu * 1.0e6 << " | " << linear_flops / linear_cpu / 1.0e9
              << " GFLOPS | 128x256x256 |\n";
    std::cout << "| Linear hibrit | " << linear_hybrid * 1.0e6 << " | " << linear_flops / linear_hybrid / 1.0e9
              << " GFLOPS | MatrixFlash scheduler |\n";
    return 0;
}
