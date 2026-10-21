/**
 * @file bench_phase10_structure.cpp
 * @brief Havuz, duzlestirme, dropout ve BatchNorm2d ileri gecis suresi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/nn/modules.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

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

}  // namespace

int main() {
    nexus_model::Tensor image = nexus_model::Tensor::zeros({8, 16, 32, 32});
    image.fill(0.25f);
    nexus::nn::MaxPool2d max_pool(2);
    nexus::nn::AvgPool2d avg_pool(2);
    nexus::nn::GlobalAvgPool2d global_pool;
    nexus::nn::Flatten flatten;
    const double max_seconds = seconds_of(1, 8, [&] { max_pool.forward(image); });
    const double avg_seconds = seconds_of(1, 8, [&] { avg_pool.forward(image); });
    const double global_seconds = seconds_of(1, 8, [&] { global_pool.forward(image); });
    const double flat_seconds = seconds_of(1, 8, [&] { flatten.forward(image); });

    nexus_model::Tensor wide = nexus_model::Tensor::zeros({32, 1024});
    wide.fill(0.5f);
    nexus::nn::Dropout dropout(0.1f);
    dropout.train();
    const double drop_seconds = seconds_of(1, 8, [&] { dropout.forward(wide); });

    nexus_model::Tensor batch = nexus_model::Tensor::zeros({8, 32, 16, 16});
    batch.fill(0.2f);
    nexus::nn::BatchNorm2d norm(32);
    const double norm_seconds = seconds_of(1, 8, [&] { norm.forward(batch); });

    volatile float sink = max_pool.forward(image)[0] + avg_pool.forward(image)[0] + global_pool.forward(image)[0] +
                          flatten.forward(image)[0] + dropout.forward(wide)[0] + norm.forward(batch)[0];
    std::cout << "Nexus Faz 10 benchmark\n";
    std::cout << "| katman | us | cagri/s |\n";
    std::cout << "| MaxPool2d 8x16x32x32 k2 | " << max_seconds * 1.0e6 << " | " << (1.0 / max_seconds) << " |\n";
    std::cout << "| AvgPool2d 8x16x32x32 k2 | " << avg_seconds * 1.0e6 << " | " << (1.0 / avg_seconds) << " |\n";
    std::cout << "| GlobalAvgPool2d 8x16x32x32 | " << global_seconds * 1.0e6 << " | " << (1.0 / global_seconds) << " |\n";
    std::cout << "| Flatten 8x16x32x32 | " << flat_seconds * 1.0e6 << " | " << (1.0 / flat_seconds) << " |\n";
    std::cout << "| Dropout 32x1024 p0.1 | " << drop_seconds * 1.0e6 << " | " << (1.0 / drop_seconds) << " |\n";
    std::cout << "| BatchNorm2d 8x32x16x16 | " << norm_seconds * 1.0e6 << " | " << (1.0 / norm_seconds) << " |\n";
    return std::isfinite(sink) ? 0 : 1;
}
