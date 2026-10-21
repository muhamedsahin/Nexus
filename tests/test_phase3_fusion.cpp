/**
 * @file test_phase3_fusion.cpp
 * @brief Hibrit Adam, Linear GEMM politikasi ve tek gecis MSE.
 */

#include "matrix_pro/hardware.hpp"
#include "nexus_model/core/execution_policy.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_optim/device/hybrid_update.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"
#include "nexusloss/losses/regression.hpp"
#include "nexusloss/losses/classification.hpp"

#include <cmath>
#include <iostream>
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

void hybrid_gemm(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n, int k) {
    matrix_pro::HeterogeneousScheduler::instance().gemm(a, lda, b, ldb, c, ldc, m, n, k, 1.0f, 0.0f);
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 3 fuzyon testi\n";
    try {
        constexpr std::size_t kN = 8192;
        std::vector<float> params(kN), grads(kN), m(kN, 0.0f), v(kN, 0.0f);
        for (std::size_t i = 0; i < kN; ++i) {
            params[i] = 0.01f * static_cast<float>(static_cast<int>(i % 50) - 25);
            grads[i] = 0.001f * static_cast<float>(static_cast<int>(i % 17) - 8);
        }
        auto params_ref = params;
        auto m_ref = m;
        auto v_ref = v;
        nexus_optim::AdamCoeffs coeffs;
        coeffs.lr = 1.0e-3f;
        coeffs.beta1 = 0.9f;
        coeffs.one_minus_beta1 = 0.1f;
        coeffs.beta2 = 0.999f;
        coeffs.one_minus_beta2 = 0.001f;
        coeffs.eps = 1.0e-8f;
        coeffs.inv_bc1 = 1.0f / (1.0f - coeffs.beta1);
        coeffs.inv_bc2 = 1.0f / (1.0f - coeffs.beta2);
        nexus_optim::adam_update(params_ref.data(), grads.data(), m_ref.data(), v_ref.data(), nullptr, kN, coeffs);
        const nexus_optim::HybridCounts split = nexus_optim::adam_update_hybrid(
            params.data(), grads.data(), m.data(), v.data(), nullptr, kN, coeffs);
        std::cout << "  info  adam cpu=" << split.cpu << " gpu=" << split.gpu << "\n";
        float worst = 0.0f;
        for (std::size_t i = 0; i < kN; ++i) {
            worst = std::max(worst, std::fabs(params[i] - params_ref[i]));
        }
        std::cout << "  info  adam_max_abs=" << worst << "\n";
        expect(split.cpu > 0, "adam keeps a CPU slice");
#if defined(NEXUS_OPTIM_WITH_CUDA)
        expect(split.gpu > 0, "adam keeps a CUDA slice");
#endif
        expect(worst < 1.0e-5f, "hybrid Adam matches CPU Adam");

        nexus_model::Linear layer(64, 48, true);
        nexus_model::Tensor input = nexus_model::Tensor::zeros({32, 64});
        for (std::size_t i = 0; i < input.numel(); ++i) {
            input.data()[i] = 0.01f * static_cast<float>(static_cast<int>(i % 20) - 10);
        }
        nexus_model::set_matmul_policy(nexus_model::MatmulPolicy::CpuKernel);
        nexus_model::Tensor cpu_out = layer.forward(input).clone();
        nexus_model::set_packed_gemm(&hybrid_gemm);
        nexus_model::set_matmul_policy(nexus_model::MatmulPolicy::Hybrid);
        nexus_model::Tensor hybrid_out = layer.forward(input).clone();
        nexus_model::set_matmul_policy(nexus_model::MatmulPolicy::CpuKernel);
        float linear_worst = 0.0f;
        for (std::size_t i = 0; i < cpu_out.numel(); ++i) {
            linear_worst = std::max(linear_worst, std::fabs(cpu_out.data()[i] - hybrid_out.data()[i]));
        }
        std::cout << "  info  linear_max_abs=" << linear_worst << "\n";
        expect(linear_worst < 1.0e-4f, "hybrid Linear matches CPU Linear");

        const std::vector<float> pred{1.0f, 3.0f};
        const std::vector<float> target{0.0f, 1.0f};
        nexusloss::MSELoss<float> loss;
        std::vector<float> grad(2);
        const double value = loss.forward_and_backward(pred, target, grad);
        expect(std::fabs(value - 2.5) < 1.0e-6, "MSE fused loss");
        expect(grad[0] == 1.0f && grad[1] == 2.0f, "MSE fused gradient");

        constexpr std::size_t kWide = 32;
        std::vector<float> wide_pred(kWide), wide_target(kWide), wide_grad(kWide);
        for (std::size_t i = 0; i < kWide; ++i) {
            wide_pred[i] = 0.05f * static_cast<float>(static_cast<int>(i) - 8);
            wide_target[i] = 0.02f * static_cast<float>(static_cast<int>(i % 7));
        }
        auto check_fused = [&](auto& fused_loss, const char* name) {
            const double reference = static_cast<double>(fused_loss.compute(wide_pred, wide_target)[0]);
            const std::vector<float> reference_grad = fused_loss.gradient(wide_pred, wide_target);
            const double fused_value = fused_loss.forward_and_backward(wide_pred, wide_target, wide_grad);
            float grad_worst = 0.0f;
            for (std::size_t i = 0; i < kWide; ++i) {
                grad_worst = std::max(grad_worst, std::fabs(wide_grad[i] - reference_grad[i]));
            }
            std::cout << "  info  " << name << "_loss_abs=" << std::fabs(fused_value - reference)
                      << " grad_abs=" << grad_worst << "\n";
            expect(std::fabs(fused_value - reference) < 1.0e-5, name);
            expect(grad_worst < 1.0e-6f, name);
        };
        nexusloss::MSELoss<float> mse;
        nexusloss::MAELoss<float> mae;
        nexusloss::HuberLoss<float> huber(0.5f);
        nexusloss::BCEWithLogitsLoss<float> bce;
        for (float& y : wide_target) {
            y = std::fabs(y);
            if (y > 1.0f) {
                y = 1.0f;
            }
        }
        check_fused(mse, "MSE AVX matches gradient");
        check_fused(mae, "MAE fused matches gradient");
        check_fused(huber, "Huber fused matches gradient");
        check_fused(bce, "BCE fused matches gradient");
    } catch (const std::exception& error) {
        std::cout << "  FAIL  istisna: " << error.what() << "\n";
        return 1;
    }
    if (g_failed != 0) {
        std::cout << g_failed << " kontrol basarisiz\n";
        return 1;
    }
    std::cout << "tum kontroller gecti\n";
    return 0;
}
