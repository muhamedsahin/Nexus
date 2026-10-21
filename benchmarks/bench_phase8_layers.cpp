/**
 * @file bench_phase8_layers.cpp
 * @brief Conv2d ve cok basli dikkat ileri gecis suresi.
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
    nexus::nn::Conv2d conv(3, 16, 3, 1, 1);
    nexus_model::Tensor image = nexus_model::Tensor::zeros({8, 3, 32, 32});
    image.fill(0.02f);
    const double conv_seconds = seconds_of(2, 6, [&] { conv.forward(image); });
    const double conv_flops = 2.0 * 8.0 * 16.0 * 32.0 * 32.0 * 3.0 * 3.0 * 3.0;

    nexus::nn::MultiHeadAttention attention(64, 4);
    nexus_model::Tensor tokens = nexus_model::Tensor::zeros({4, 32, 64});
    tokens.fill(0.01f);
    const double attention_seconds = seconds_of(2, 6, [&] { attention.forward(tokens); });

    nexus::nn::LSTM lstm(32, 64);
    nexus_model::Tensor sequence = nexus_model::Tensor::zeros({4, 16, 32});
    sequence.fill(0.01f);
    const double lstm_seconds = seconds_of(2, 6, [&] { lstm.forward(sequence); });

    volatile float sink = conv.forward(image)[0] + attention.forward(tokens)[0] + lstm.forward(sequence)[0];
    std::cout << "Nexus Faz 8 benchmark\n";
    std::cout << "| katman | us | cagri/s | GFLOPS |\n";
    std::cout << "| Conv2d 8x3x32x32 k=3 | " << conv_seconds * 1.0e6 << " | " << (1.0 / conv_seconds) << " | "
              << (conv_flops / conv_seconds / 1.0e9) << " |\n";
    std::cout << "| Attention B4 T32 E64 H4 | " << attention_seconds * 1.0e6 << " | " << (1.0 / attention_seconds)
              << " |  |\n";
    std::cout << "| LSTM B4 T16 C32 H64 | " << lstm_seconds * 1.0e6 << " | " << (1.0 / lstm_seconds) << " |  |\n";
    return std::isfinite(sink) ? 0 : 1;
}
