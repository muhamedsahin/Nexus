/**
 * @file bench_phase9_blocks.cpp
 * @brief Transformer, GRU, gomme ve BatchNorm ileri gecis suresi.
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
    nexus::nn::TransformerEncoder encoder(64, 4, 128);
    nexus_model::Tensor tokens = nexus_model::Tensor::zeros({4, 32, 64});
    tokens.fill(0.01f);
    const double encoder_seconds = seconds_of(1, 4, [&] { encoder.forward(tokens); });

    nexus::nn::GRU gru(32, 64);
    nexus_model::Tensor sequence = nexus_model::Tensor::zeros({4, 16, 32});
    sequence.fill(0.01f);
    const double gru_seconds = seconds_of(1, 4, [&] { gru.forward(sequence); });

    nexus::nn::Embedding embedding(1024, 64);
    nexus_model::Tensor indices = nexus_model::Tensor::zeros({32, 64});
    for (std::size_t i = 0; i < indices.numel(); ++i) {
        indices[i] = static_cast<float>(i % 1024);
    }
    const double embed_seconds = seconds_of(1, 8, [&] { embedding.forward(indices); });

    nexus::nn::BatchNorm1d norm(128);
    nexus_model::Tensor batch = nexus_model::Tensor::zeros({64, 128});
    batch.fill(0.2f);
    const double norm_seconds = seconds_of(1, 8, [&] { norm.forward(batch); });

    volatile float sink = encoder.forward(tokens)[0] + gru.forward(sequence)[0] + embedding.forward(indices)[0] +
                          norm.forward(batch)[0];
    std::cout << "Nexus Faz 9 benchmark\n";
    std::cout << "| katman | us | cagri/s |\n";
    std::cout << "| Transformer B4 T32 E64 | " << encoder_seconds * 1.0e6 << " | " << (1.0 / encoder_seconds) << " |\n";
    std::cout << "| GRU B4 T16 C32 H64 | " << gru_seconds * 1.0e6 << " | " << (1.0 / gru_seconds) << " |\n";
    std::cout << "| Embedding 32x64 vocab 1024 | " << embed_seconds * 1.0e6 << " | " << (1.0 / embed_seconds) << " |\n";
    std::cout << "| BatchNorm1d 64x128 | " << norm_seconds * 1.0e6 << " | " << (1.0 / norm_seconds) << " |\n";
    return std::isfinite(sink) ? 0 : 1;
}
