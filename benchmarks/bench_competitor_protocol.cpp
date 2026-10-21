/**
 * @file bench_competitor_protocol.cpp
 * @brief Rakip protokolu ile ayni boyutlarda Nexus olcumu. Stdout yalniz JSON.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/core/thread_pool.hpp"
#include "matrix_pro/cpu/gemm.hpp"
#include "matrix_pro/detail/cuda_hooks.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus_optim/simd/simd_dispatch.hpp"
#include "nexus_train/bridge/loss_driver.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Row {
    std::string name;
    double seconds = 0.0;
    double value = 0.0;
    const char* unit = "";
};

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

void add_row(std::vector<Row>& rows, const std::string& name, double seconds, double value, const char* unit, float probe) {
    if (!std::isfinite(probe) || !std::isfinite(seconds) || !std::isfinite(value) || seconds <= 0.0) {
        std::cerr << "dusen satir " << name << "\n";
        return;
    }
    rows.push_back(Row{name, seconds, value, unit});
}

void print_json(const char* device, int pool_workers, const char* kernel, const char* gpu_name, const std::vector<Row>& rows) {
    std::cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
    std::cout.precision(17);
    std::cout << "{\n";
    std::cout << "  \"library\": \"nexus\",\n";
    std::cout << "  \"device\": \"" << device << "\",\n";
    std::cout << "  \"threads\": \"" << (std::string(device) == "gpu" ? "device" : "8") << "\",\n";
    std::cout << "  \"pool_workers\": " << pool_workers << ",\n";
    std::cout << "  \"kernel\": \"" << kernel << "\",\n";
#if defined(_MSC_VER)
    std::cout << "  \"compiler\": \"MSVC " << _MSC_FULL_VER << "\",\n";
#else
    std::cout << "  \"compiler\": \"other\",\n";
#endif
    std::cout << "  \"gpu_name\": \"" << gpu_name << "\",\n";
    std::cout << "  \"rows\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Row& row = rows[i];
        std::cout << "    {\"library\": \"nexus\", \"device\": \"" << device << "\", \"threads\": \""
                  << (std::string(device) == "gpu" ? "device" : "8") << "\", \"name\": \"" << row.name
                  << "\", \"seconds\": " << row.seconds << ", \"value\": " << row.value << ", \"unit\": \"" << row.unit
                  << "\"}";
        if (i + 1 < rows.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]\n}\n";
}

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
int observed_gpu_count() { return matrix_pro::hw::cuda_device_count(); }

void cpu_rows(std::vector<Row>& rows) {
    for (int n : {256, 512, 1024, 2048}) {
        const std::size_t cells = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        std::vector<float> a(cells, 0.01f);
        std::vector<float> b(cells, 0.02f);
        std::vector<float> c(cells, 0.0f);
        const double flops = 2.0 * static_cast<double>(n) * n * n;
        const double seconds = seconds_of(1, 3, [&] { matrix_pro::gemm_f32(n, n, n, a.data(), n, b.data(), n, c.data(), n, 1.0f, 0.0f); });
        add_row(rows, "gemm_n" + std::to_string(n), seconds, flops / seconds / 1.0e9, "GFLOPS", c[0]);
    }

    nexus::nn::Conv2d conv(3, 16, 3, 1, 1);
    nexus_model::Tensor image = nexus_model::Tensor::zeros({8, 3, 32, 32});
    image.fill(0.02f);
    nexus_model::Tensor conv_out;
    const double conv_seconds = seconds_of(1, 4, [&] { conv_out = conv.forward(image); });
    const double conv_flops = 2.0 * 8 * 16 * 32 * 32 * 3 * 3 * 3;
    add_row(rows, "conv2d", conv_seconds, conv_flops / conv_seconds / 1.0e9, "GFLOPS", conv_out[0]);

    nexus::nn::DepthwiseConv2d depth(32, 3, 1, 1);
    nexus_model::Tensor field = nexus_model::Tensor::zeros({8, 32, 16, 16});
    field.fill(0.02f);
    nexus_model::Tensor depth_out;
    const double depth_seconds = seconds_of(1, 4, [&] { depth_out = depth.forward(field); });
    const double depth_flops = 2.0 * 8 * 32 * 16 * 16 * 9;
    add_row(rows, "depthwise", depth_seconds, depth_flops / depth_seconds / 1.0e9, "GFLOPS", depth_out[0]);

    nexus::nn::MultiHeadAttention attention(64, 4);
    nexus_model::Tensor tokens = nexus_model::Tensor::zeros({4, 32, 64});
    tokens.fill(0.01f);
    nexus_model::Tensor attn_out;
    const double attn_seconds = seconds_of(1, 4, [&] { attn_out = attention.forward(tokens); });
    add_row(rows, "attention", attn_seconds, 1.0 / attn_seconds, "cagri/s", attn_out[0]);

    nexus::nn::LSTM lstm(32, 64);
    nexus::nn::GRU gru(32, 64);
    nexus_model::Tensor sequence = nexus_model::Tensor::zeros({4, 16, 32});
    sequence.fill(0.01f);
    nexus_model::Tensor lstm_out;
    nexus_model::Tensor gru_out;
    const double lstm_seconds = seconds_of(1, 4, [&] { lstm_out = lstm.forward(sequence); });
    const double gru_seconds = seconds_of(1, 4, [&] { gru_out = gru.forward(sequence); });
    add_row(rows, "lstm", lstm_seconds, 1.0 / lstm_seconds, "cagri/s", lstm_out[0]);
    add_row(rows, "gru", gru_seconds, 1.0 / gru_seconds, "cagri/s", gru_out[0]);

    nexus_model::Tensor pred = nexus_model::Tensor::zeros({1 << 20});
    nexus_model::Tensor target = nexus_model::Tensor::zeros({1 << 20});
    nexus_model::Tensor grad = nexus_model::Tensor::zeros({1 << 20});
    pred.fill(0.2f);
    target.fill(0.1f);
    nexus_train::MseLossDriver mse;
    const double mse_seconds = seconds_of(1, 4, [&] { mse.forward_and_backward(pred, target, grad); });
    const double mse_bytes = static_cast<double>(1 << 20) * 3.0 * sizeof(float);
    add_row(rows, "mse", mse_seconds, mse_bytes / mse_seconds / 1.0e9, "GB/s", grad[0]);

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
    const double adam_seconds = seconds_of(
        1, 4, [&] { nexus_optim::adam_update(params.data(), grads.data(), moment.data(), second.data(), nullptr, kParams, adam); });
    const double adam_bytes = static_cast<double>(kParams) * 4.0 * sizeof(float);
    add_row(rows, "adam", adam_seconds, adam_bytes / adam_seconds / 1.0e9, "GB/s", params[0]);
}

void gpu_rows(std::vector<Row>& rows) {
#if defined(NEXUS_ENABLE_CUDA)
    if (observed_gpu_count() <= 0) {
        std::cerr << "gpu yok\n";
        return;
    }
    for (int n : {256, 512, 1024, 2048}) {
        const std::size_t cells = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        std::vector<float> a(cells, 0.01f);
        std::vector<float> b(cells, 0.02f);
        float* device_a = matrix_pro::hw::cuda_alloc(0, cells);
        float* device_b = matrix_pro::hw::cuda_alloc(0, cells);
        float* device_c = matrix_pro::hw::cuda_alloc(0, cells);
        matrix_pro::hw::cuda_copy_h2d(0, device_a, a.data(), cells);
        matrix_pro::hw::cuda_copy_h2d(0, device_b, b.data(), cells);
        matrix_pro::hw::cuda_gemm(0, n, n, n, device_a, n, device_b, n, device_c, n, 1.0f, 0.0f);
        matrix_pro::hw::cuda_sync(0);
        const double flops = 2.0 * static_cast<double>(n) * n * n;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 3; ++i) {
            matrix_pro::hw::cuda_gemm(0, n, n, n, device_a, n, device_b, n, device_c, n, 1.0f, 0.0f);
            matrix_pro::hw::cuda_sync(0);
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 3.0;
        float host = 0.0f;
        matrix_pro::hw::cuda_copy_d2h(0, &host, device_c, 1);
        matrix_pro::hw::cuda_sync(0);
        add_row(rows, "gemm_n" + std::to_string(n), seconds, flops / seconds / 1.0e9, "GFLOPS", host);
        matrix_pro::hw::cuda_free(0, device_a);
        matrix_pro::hw::cuda_free(0, device_b);
        matrix_pro::hw::cuda_free(0, device_c);
    }
#else
    (void)rows;
    std::cerr << "cuda kapali\n";
#endif
}

}  // namespace

int main(int argc, char** argv) {
    const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
    const int pool = matrix_pro::WorkStealingPool::instance().worker_count();
    const char* kernel = matrix_pro::gemm_kernel_name();
    const char* gpu_name = "";
#if defined(NEXUS_ENABLE_CUDA)
    if (observed_gpu_count() > 0) {
        gpu_name = matrix_pro::hw::cuda_device_name(0);
    }
#endif
    std::vector<Row> rows;
    if (gpu) {
        gpu_rows(rows);
        print_json("gpu", pool, kernel, gpu_name, rows);
    } else {
        cpu_rows(rows);
        print_json("cpu", pool, kernel, gpu_name, rows);
    }
    return 0;
}
