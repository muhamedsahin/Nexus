/**
 * @file bench_phase13_baseline.cpp
 * @brief Faz 14-23 icin tek taban olcumu. Rakamlar bu surecin ciktisidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/cpu/gemm.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"
#include "nexus_train/bridge/loss_driver.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

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

void gemm_row(int n) {
    const std::size_t cells = static_cast<std::size_t>(n) * n;
    std::vector<float> a(cells, 0.01f);
    std::vector<float> b(cells, 0.02f);
    std::vector<float> c(cells, 0.0f);
    const double flops = 2.0 * n * n * n;
    const double seconds = seconds_of(1, 3, [&] { matrix_pro::gemm_f32(n, n, n, a.data(), n, b.data(), n, c.data(), n, 1.0f, 0.0f); });
    std::cout << "| GEMM N" << n << " | " << seconds * 1.0e6 << " | " << (flops / seconds / 1.0e9) << " | GFLOPS |\n";
    if (!std::isfinite(c[0])) {
        std::cout << "nan\n";
    }
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 13 taban\n";
    std::cout << "cekirdek | " << matrix_pro::gemm_kernel_name() << "\n";
    std::cout << "| olcu | us | deger | birim |\n";
    gemm_row(256);
    gemm_row(512);
    gemm_row(1024);
    gemm_row(2048);

    nexus::nn::Conv2d conv(3, 16, 3, 1, 1);
    nexus_model::Tensor image = nexus_model::Tensor::zeros({8, 3, 32, 32});
    image.fill(0.02f);
    const double conv_seconds = seconds_of(1, 4, [&] { conv.forward(image); });
    const double conv_flops = 2.0 * 8 * 16 * 32 * 32 * 3 * 3 * 3;
    std::cout << "| Conv2d 8x3x32x32 k3 | " << conv_seconds * 1.0e6 << " | " << (conv_flops / conv_seconds / 1.0e9) << " | GFLOPS |\n";

    nexus::nn::DepthwiseConv2d depth(32, 3, 1, 1);
    nexus_model::Tensor field = nexus_model::Tensor::zeros({8, 32, 16, 16});
    field.fill(0.02f);
    const double depth_seconds = seconds_of(1, 4, [&] { depth.forward(field); });
    const double depth_flops = 2.0 * 8 * 32 * 16 * 16 * 9;
    std::cout << "| Depthwise 8x32x16x16 k3 | " << depth_seconds * 1.0e6 << " | " << (depth_flops / depth_seconds / 1.0e9)
              << " | GFLOPS |\n";

    nexus::nn::MultiHeadAttention attention(64, 4);
    nexus_model::Tensor tokens = nexus_model::Tensor::zeros({4, 32, 64});
    tokens.fill(0.01f);
    const double attn_seconds = seconds_of(1, 4, [&] { attention.forward(tokens); });
    std::cout << "| Attention B4 T32 E64 H4 | " << attn_seconds * 1.0e6 << " | " << (1.0 / attn_seconds) << " | cagri/s |\n";

    nexus::nn::LSTM lstm(32, 64);
    nexus::nn::GRU gru(32, 64);
    nexus_model::Tensor sequence = nexus_model::Tensor::zeros({4, 16, 32});
    sequence.fill(0.01f);
    const double lstm_seconds = seconds_of(1, 4, [&] { lstm.forward(sequence); });
    const double gru_seconds = seconds_of(1, 4, [&] { gru.forward(sequence); });
    std::cout << "| LSTM B4 T16 C32 H64 | " << lstm_seconds * 1.0e6 << " | " << (1.0 / lstm_seconds) << " | cagri/s |\n";
    std::cout << "| GRU B4 T16 C32 H64 | " << gru_seconds * 1.0e6 << " | " << (1.0 / gru_seconds) << " | cagri/s |\n";

    nexus_model::Tensor pred = nexus_model::Tensor::zeros({1 << 20});
    nexus_model::Tensor target = nexus_model::Tensor::zeros({1 << 20});
    nexus_model::Tensor grad = nexus_model::Tensor::zeros({1 << 20});
    pred.fill(0.2f);
    target.fill(0.1f);
    nexus_train::MseLossDriver mse;
    const double mse_seconds = seconds_of(1, 4, [&] { mse.forward_and_backward(pred, target, grad); });
    const double mse_bytes = static_cast<double>(1 << 20) * 3.0 * sizeof(float);
    std::cout << "| MSE driver 1M | " << mse_seconds * 1.0e6 << " | " << (mse_bytes / mse_seconds / 1.0e9) << " | GB/s |\n";

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
    adam.inv_bc1 = 10.0f;
    adam.inv_bc2 = 1000.0f;
    const double adam_seconds =
        seconds_of(1, 4, [&] { nexus_optim::adam_update(params.data(), grads.data(), moment.data(), second.data(), nullptr, kParams, adam); });
    const double adam_bytes = static_cast<double>(kParams) * 4.0 * sizeof(float);
    std::cout << "| Adam CPU 1M | " << adam_seconds * 1.0e6 << " | " << (adam_bytes / adam_seconds / 1.0e9) << " | GB/s |\n";

    const float sink = pred[0] + grad[0] + params[0] + image[0];
    return std::isfinite(sink) ? 0 : 1;
}
