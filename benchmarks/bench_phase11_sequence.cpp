/**
 * @file bench_phase11_sequence.cpp
 * @brief Konum kodlama, cozucu, RNN ve yeniden sekil ileri gecis suresi.
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
    nexus::nn::PositionalEncoding encoding(64, 64, false);
    nexus_model::Tensor tokens = nexus_model::Tensor::zeros({4, 32, 64});
    tokens.fill(0.01f);
    const double encode_seconds = seconds_of(1, 8, [&] { encoding.forward(tokens); });

    nexus::nn::TransformerDecoder decoder(64, 4, 128);
    nexus_model::Tensor target = nexus_model::Tensor::zeros({4, 32, 64});
    nexus_model::Tensor memory = nexus_model::Tensor::zeros({4, 32, 64});
    target.fill(0.01f);
    memory.fill(0.01f);
    const double decoder_seconds = seconds_of(1, 4, [&] { decoder.forward(target, memory); });

    nexus::nn::RNN rnn(32, 64);
    nexus_model::Tensor sequence = nexus_model::Tensor::zeros({4, 16, 32});
    sequence.fill(0.01f);
    const double rnn_seconds = seconds_of(1, 4, [&] { rnn.forward(sequence); });

    nexus::nn::Reshape reshape({4, 2048});
    const double reshape_seconds = seconds_of(1, 8, [&] { reshape.forward(tokens); });

    volatile float sink = encoding.forward(tokens)[0] + decoder.forward(target, memory)[0] + rnn.forward(sequence)[0] +
                          reshape.forward(tokens)[0];
    std::cout << "Nexus Faz 11 benchmark\n";
    std::cout << "| katman | us | cagri/s |\n";
    std::cout << "| PositionalEncoding B4 T32 E64 | " << encode_seconds * 1.0e6 << " | " << (1.0 / encode_seconds) << " |\n";
    std::cout << "| TransformerDecoder B4 T32 E64 | " << decoder_seconds * 1.0e6 << " | " << (1.0 / decoder_seconds) << " |\n";
    std::cout << "| RNN B4 T16 C32 H64 | " << rnn_seconds * 1.0e6 << " | " << (1.0 / rnn_seconds) << " |\n";
    std::cout << "| Reshape 4x32x64 -> 4x2048 | " << reshape_seconds * 1.0e6 << " | " << (1.0 / reshape_seconds) << " |\n";
    return std::isfinite(sink) ? 0 : 1;
}
