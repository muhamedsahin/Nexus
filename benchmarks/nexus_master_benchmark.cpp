/**
 * @file nexus_master_benchmark.cpp
 * @brief Faz 7 olcumu. Isinma turu sureye girmez. Cikti tek satirlk kayitlardir.
 */

#include "matrix_pro/detail/cuda_hooks.hpp"
#include "matrix_pro/hardware.hpp"
#include "nexus/nexus.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void row(const char* name, double value, const char* unit, const std::string& note) {
    std::cout << name << " | " << value << " | " << unit << " | " << note << "\n";
}

// noinline: MSVC, govdesini gordugu `return 0` fonksiyonunu sabit katlar ve
// `if (gpus > 0)` icin C4127 uretir. Sayim her cagrida gercekten yapilir.
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
int observed_gpu_count() { return matrix_pro::hw::cuda_device_count(); }

}  // namespace

int main() {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "NEXUS_MASTER_BENCHMARK\n";
    const int gpus = observed_gpu_count();
    std::cout << "gpu_count | " << gpus << " | count | ";
#if defined(NEXUS_ENABLE_CUDA)
    if (gpus > 0) {
        std::cout << matrix_pro::hw::cuda_device_name(0);
    }
#else
    std::cout << "cuda kapali";
#endif
    std::cout << "\n";

    constexpr int kN = 512;
    const std::size_t cells = static_cast<std::size_t>(kN) * kN;
    std::vector<float> a(cells, 0.01f);
    std::vector<float> b(cells, 0.02f);
    std::vector<float> c(cells, 0.0f);
    const double gemm_flops = 2.0 * kN * kN * kN;
    matrix_pro::gemm_f32(kN, kN, kN, a.data(), kN, b.data(), kN, c.data(), kN, 1.0f, 0.0f);
    constexpr int kGemmRepeats = 3;
    const auto cpu_gemm_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kGemmRepeats; ++i) {
        matrix_pro::gemm_f32(kN, kN, kN, a.data(), kN, b.data(), kN, c.data(), kN, 1.0f, 0.0f);
    }
    const double cpu_gemm = seconds_since(cpu_gemm_start) / kGemmRepeats;
    row("cpu_gemm_512", cpu_gemm * 1.0e6, "us", "AVX");
    row("cpu_gemm_512_gflops", gemm_flops / cpu_gemm / 1.0e9, "GFLOPS", "AVX");

    double gpu_gemm = 0.0;
    if (gpus > 0) {
        float* device_a = matrix_pro::hw::cuda_alloc(0, cells);
        float* device_b = matrix_pro::hw::cuda_alloc(0, cells);
        float* device_c = matrix_pro::hw::cuda_alloc(0, cells);
        matrix_pro::hw::cuda_copy_h2d(0, device_a, a.data(), cells);
        matrix_pro::hw::cuda_copy_h2d(0, device_b, b.data(), cells);
        matrix_pro::hw::cuda_gemm(0, kN, kN, kN, device_a, kN, device_b, kN, device_c, kN, 1.0f, 0.0f);
        matrix_pro::hw::cuda_sync(0);
        const auto gpu_start = std::chrono::steady_clock::now();
        for (int i = 0; i < kGemmRepeats; ++i) {
            matrix_pro::hw::cuda_gemm(0, kN, kN, kN, device_a, kN, device_b, kN, device_c, kN, 1.0f, 0.0f);
            matrix_pro::hw::cuda_sync(0);
        }
        gpu_gemm = seconds_since(gpu_start) / kGemmRepeats;
        matrix_pro::hw::cuda_free(0, device_a);
        matrix_pro::hw::cuda_free(0, device_b);
        matrix_pro::hw::cuda_free(0, device_c);
        row("gpu_gemm_512", gpu_gemm * 1.0e6, "us", "FP32 pedantic device-resident");
        row("gpu_gemm_512_gflops", gemm_flops / gpu_gemm / 1.0e9, "GFLOPS", "H2D haric");
    }

    matrix_pro::HeterogeneousScheduler::instance().calibrate();
    matrix_pro::HeterogeneousScheduler::instance().gemm(a.data(), kN, b.data(), kN, c.data(), kN, kN, kN, kN, 1.0f, 0.0f);
    const auto hybrid_start = std::chrono::steady_clock::now();
    matrix_pro::HeterogeneousScheduler::instance().gemm(a.data(), kN, b.data(), kN, c.data(), kN, kN, kN, kN, 1.0f, 0.0f);
    const double hybrid_gemm = seconds_since(hybrid_start);
    row("hybrid_gemm_512", hybrid_gemm * 1.0e6, "us", "tek cagri, kopya dahil");
    row("hybrid_gemm_512_gflops", gemm_flops / hybrid_gemm / 1.0e9, "GFLOPS", "N=512");

    std::vector<float> x(cells, 1.0f);
    std::vector<float> y(cells, 2.0f);
    std::vector<float> z(cells, 0.0f);
    matrix_pro::elementwise_add(x.data(), y.data(), z.data(), static_cast<int>(cells));
    constexpr int kAddRepeats = 8;
    const auto add_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kAddRepeats; ++i) {
        matrix_pro::elementwise_add(x.data(), y.data(), z.data(), static_cast<int>(cells));
    }
    const double add_seconds = seconds_since(add_start) / kAddRepeats;
    const double add_bytes = static_cast<double>(cells) * 3.0 * sizeof(float);
    row("cpu_add_gbs", add_bytes / add_seconds / 1.0e9, "GB/s", "N=512^2");

    constexpr std::size_t kParams = 1u << 20;
    std::vector<float> params(kParams, 0.1f);
    std::vector<float> grads(kParams, 0.01f);
    std::vector<float> moment(kParams, 0.0f);
    std::vector<float> second(kParams, 0.0f);
    nexus_optim::AdamCoeffs adam;
    adam.lr = 1.0e-3f;
    adam.beta1 = 0.9f;
    adam.one_minus_beta1 = 0.1f;
    adam.beta2 = 0.999f;
    adam.one_minus_beta2 = 0.001f;
    adam.eps = 1.0e-8f;
    adam.inv_bc1 = 1.0f / 0.1f;
    adam.inv_bc2 = 1.0f / 0.001f;
    nexus_optim::adam_update(params.data(), grads.data(), moment.data(), second.data(), nullptr, kParams, adam);
    constexpr int kOptimRepeats = 8;
    const auto adam_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kOptimRepeats; ++i) {
        nexus_optim::adam_update(params.data(), grads.data(), moment.data(), second.data(), nullptr, kParams, adam);
    }
    const double adam_seconds = seconds_since(adam_start) / kOptimRepeats;
    const double adam_bytes = static_cast<double>(kParams) * 4.0 * sizeof(float);
    row("adam_step", adam_seconds * 1.0e6, "us", "n=1048576 AVX2");
    row("adam_gbs", adam_bytes / adam_seconds / 1.0e9, "GB/s", "4 dizi");
    row("adam_steps_per_s", 1.0 / adam_seconds, "step/s", "1M parametre");

    nexus_optim::ParamGroupOptions group;
    group.learning_rate = 1.0e-3;
    nexus_optim::AdamW<float>::Options adamw_options;
    adamw_options.lr = 1.0e-3;
    adamw_options.weight_decay = 0.01;
    nexus_optim::AdamW<float> adamw({nexus_optim::make_group(params.data(), grads.data(), kParams, group)}, adamw_options);
    adamw.step();
    const auto adamw_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kOptimRepeats; ++i) {
        adamw.step();
    }
    const double adamw_seconds = seconds_since(adamw_start) / kOptimRepeats;
    row("adamw_step", adamw_seconds * 1.0e6, "us", "weight_decay=0.01");
    row("adamw_steps_per_s", 1.0 / adamw_seconds, "step/s", "1M parametre");

    nexus_optim::SGD<float>::Options sgd_options;
    sgd_options.lr = 1.0e-2;
    nexus_optim::SGD<float> sgd({nexus_optim::make_group(params.data(), grads.data(), kParams, group)}, sgd_options);
    sgd.step();
    const auto sgd_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kOptimRepeats; ++i) {
        sgd.step();
    }
    const double sgd_seconds = seconds_since(sgd_start) / kOptimRepeats;
    row("sgd_step", sgd_seconds * 1.0e6, "us", "momentum=0");
    row("sgd_steps_per_s", 1.0 / sgd_seconds, "step/s", "1M parametre");

    nexus_model::Linear layer(256, 256, true);
    nexus_model::Tensor input = nexus_model::Tensor::zeros({128, 256});
    input.fill(0.01f);
    layer.forward(input);
    constexpr int kLinearRepeats = 8;
    const auto linear_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kLinearRepeats; ++i) {
        layer.forward(input);
    }
    const double linear_seconds = seconds_since(linear_start) / kLinearRepeats;
    const double linear_flops = 2.0 * 128.0 * 256.0 * 256.0;
    row("linear_128_256_256", linear_seconds * 1.0e6, "us", "CPU cekirdek");
    row("linear_gflops", linear_flops / linear_seconds / 1.0e9, "GFLOPS", "bias dahil degil");

    constexpr std::size_t kLoss = 1u << 20;
    std::vector<float> pred(kLoss, 0.2f);
    std::vector<float> target(kLoss, 0.1f);
    std::vector<float> loss_grad(kLoss, 0.0f);
    nexusloss::fused::mse(pred.data(), target.data(), loss_grad.data(), kLoss, 0);
    const auto mse_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kOptimRepeats; ++i) {
        nexusloss::fused::mse(pred.data(), target.data(), loss_grad.data(), kLoss, 0);
    }
    const double mse_seconds = seconds_since(mse_start) / kOptimRepeats;
    const double mse_bytes = static_cast<double>(kLoss) * 3.0 * sizeof(float);
    row("mse_fused", mse_seconds * 1.0e6, "us", "1M eleman mean");
    row("mse_gbs", mse_bytes / mse_seconds / 1.0e9, "GB/s", "pred+target+grad");

    constexpr int kSamples = 100000;
    constexpr int kWidth = 32;
    std::vector<float> features(static_cast<std::size_t>(kSamples * kWidth), 0.1f);
    std::vector<int> labels(static_cast<std::size_t>(kSamples), 0);
    nexus::data::DataLoader loader(features, labels, kWidth, 256, true, 1);
    nexus::data::Batch batch;
    while (loader.next(batch)) {
    }
    loader.reset();
    const auto load_start = std::chrono::steady_clock::now();
    int loaded = 0;
    while (loader.next(batch)) {
        loaded += batch.batch_size;
    }
    const double load_seconds = seconds_since(load_start);
    row("dataloader_samples_per_s", static_cast<double>(loaded) / load_seconds, "sample/s", "100k x 32 shuffle");

    constexpr int kEnvs = 64;
    constexpr int kEnvSteps = 100;
    std::vector<std::unique_ptr<nexus::rl::Environment>> envs;
    for (int i = 0; i < kEnvs; ++i) {
        envs.push_back(std::make_unique<nexus::rl::LineWorld>());
    }
    nexus::rl::VectorEnv vector_env(std::move(envs));
    vector_env.reset(1);
    std::vector<int> actions(static_cast<std::size_t>(kEnvs), 1);
    const auto env_start = std::chrono::steady_clock::now();
    for (int step = 0; step < kEnvSteps; ++step) {
        vector_env.step(actions);
    }
    const double env_seconds = seconds_since(env_start);
    row("vector_env_sps", static_cast<double>(kEnvs * kEnvSteps) / env_seconds, "step/s", "64 LineWorld");

    std::vector<float> left(kParams, 1.0f);
    std::vector<float> right(kParams, 3.0f);
    std::vector<std::vector<float>> ranks{left, right};
    const auto reduce_start = std::chrono::steady_clock::now();
    nexus::train::all_reduce_mean(ranks);
    const double reduce_seconds = seconds_since(reduce_start);
    row("allreduce_1m", reduce_seconds * 1.0e6, "us", "2 rank mean");

    const bool finite = std::isfinite(c[0]) && std::isfinite(z[0]) && std::isfinite(loss_grad[0]) && std::isfinite(ranks[0][0]);
    std::cout << "sink | " << c[0] << " | f32 | " << (finite ? "ok" : "nan") << "\n";
    return finite ? 0 : 1;
}
