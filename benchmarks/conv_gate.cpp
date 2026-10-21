/**
 * Faz 7 olcumu. Protokol sekli: N=8, C=3, 32x32, 16 filtre, 3x3, pad 1.
 * CPU nn::Conv2d ile aygitta kalan matrix_pro::conv2d karsilastirilir.
 */

#include "matrix_pro/nn/conv.hpp"
#include "nexus/nn/modules.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

namespace {

double seconds(int times, const std::function<void()>& body) {
    body();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < times; ++i) {
        body();
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / static_cast<double>(times);
}

}  // namespace

int main() {
    constexpr double kFlops = 2.0 * 8.0 * 16.0 * 32.0 * 32.0 * 3.0 * 3.0 * 3.0;
    nexus::nn::Conv2d conv(3, 16, 3, 1, 1);
    nexus_model::Tensor image = nexus_model::Tensor::zeros({8, 3, 32, 32});
    image.fill(0.02f);
    nexus_model::Tensor cpu_out;
    const double cpu = seconds(8, [&] { cpu_out = conv.forward(image); });

    const std::vector<float> input_values(8 * 3 * 32 * 32, 0.02f);
    const std::vector<float> weight_values(16 * 3 * 3 * 3, 0.01f);
    const std::vector<float> bias_values(16, 0.0f);
    matrix_pro::Tensor input({8, 3, 32, 32}, input_values);
    matrix_pro::Tensor weights({16, 3, 3, 3}, weight_values);
    matrix_pro::Tensor bias({16}, bias_values);
    input.upload();
    weights.upload();
    bias.upload();
    matrix_pro::Tensor gpu_out;
    const double gpu = seconds(8, [&] {
        gpu_out = matrix_pro::conv2d(input, weights, bias, 1, 1);
        gpu_out.synchronize();
    });
    const double roundtrip = seconds(4, [&] {
        input.upload();
        weights.upload();
        bias.upload();
        gpu_out = matrix_pro::conv2d(input, weights, bias, 1, 1);
        gpu_out.download();
        gpu_out.synchronize();
    });

    matrix_pro::Tensor tiny_in({1, 1, 1, 1}, std::vector<float>{2.0f});
    matrix_pro::Tensor tiny_w({1, 1, 1, 1}, std::vector<float>{3.0f});
    matrix_pro::Tensor tiny_b({1}, std::vector<float>{0.5f});
    tiny_in.upload();
    tiny_w.upload();
    tiny_b.upload();
    matrix_pro::Tensor tiny_out = matrix_pro::conv2d(tiny_in, tiny_w, tiny_b, 1, 0);
    tiny_out.download();
    const float tiny = tiny_out.data().empty() ? 0.0f : tiny_out.data()[0];

    const double cpu_gflops = kFlops / cpu / 1.0e9;
    const double gpu_gflops = kFlops / gpu / 1.0e9;
    const double round_gflops = kFlops / roundtrip / 1.0e9;
    std::cout << "cpu_s " << cpu << " cpu_gflops " << cpu_gflops << "\n";
    std::cout << "gpu_s " << gpu << " gpu_gflops " << gpu_gflops << "\n";
    std::cout << "roundtrip_s " << roundtrip << " roundtrip_gflops " << round_gflops << "\n";
    std::cout << "cpu_check " << cpu_out[0] << " tiny " << tiny << "\n";
    std::cout << (gpu < cpu && std::fabs(tiny - 6.5f) < 1.0e-3f ? "keep\n" : "drop\n");
    return 0;
}
