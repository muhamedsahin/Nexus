/**
 * @file test_phase11_sequence.cpp
 * @brief Konum kodlama, cozucu, RNN ve yeniden sekil.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/nn/modules.hpp"

#include <cmath>
#include <iostream>

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

bool finite_tensor(const nexus_model::Tensor& tensor) {
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        if (!std::isfinite(tensor[i])) {
            return false;
        }
    }
    return true;
}

float parameter_grad_energy(const nexus::nn::Module& layer) {
    double energy = 0.0;
    for (nexus_model::Parameter* parameter : layer.parameters()) {
        for (std::size_t i = 0; i < parameter->grad.numel(); ++i) {
            const double value = parameter->grad[i];
            energy += value * value;
        }
    }
    return static_cast<float>(energy);
}

void fill_pattern(nexus_model::Tensor& tensor, float scale) {
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        tensor[i] = scale * static_cast<float>(static_cast<int>(i % 9) - 4);
    }
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 11 dizi testi\n";
    try {
        nexus::nn::PositionalEncoding encoding(4, 8, false);
        nexus_model::Tensor tokens = nexus_model::Tensor::zeros({2, 3, 4});
        nexus_model::Tensor encoded = encoding.forward(tokens);
        expect(encoded.rank() == 3 && encoded.size(0) == 2 && encoded.size(1) == 3 && encoded.size(2) == 4, "positional shape [B,T,E]");
        expect(encoded[0] == 0.0f && encoded[1] == 1.0f && encoded[2] == 0.0f && encoded[3] == 1.0f, "position 0 is sin/cos of zero");
        expect(finite_tensor(encoded) && encoded[4] != 0.0f, "later positions leave zero");

        nexus::nn::PositionalEncoding learned(4, 8, true);
        learned.zero_grad();
        learned.forward(tokens);
        nexus_model::Tensor position_grad = nexus_model::Tensor::zeros({2, 3, 4});
        position_grad.fill(0.2f);
        expect(finite_tensor(learned.backward(position_grad)) && parameter_grad_energy(learned) > 0.0f, "learnable positions receive gradient");

        nexus::nn::TransformerDecoder decoder(8, 2, 16);
        nexus_model::Tensor target = nexus_model::Tensor::zeros({2, 4, 8});
        nexus_model::Tensor memory = nexus_model::Tensor::zeros({2, 5, 8});
        fill_pattern(target, 0.05f);
        fill_pattern(memory, 0.04f);
        bool single_rejected = false;
        try {
            decoder.forward(target);
        } catch (const nexus_model::ModelError&) {
            single_rejected = true;
        }
        expect(single_rejected, "decoder rejects a single tensor");
        nexus_model::Tensor decoded = decoder.forward(target, memory);
        expect(decoded.rank() == 3 && decoded.size(0) == 2 && decoded.size(1) == 4 && decoded.size(2) == 8, "decoder keeps the target shape");
        expect(finite_tensor(decoded), "decoder forward finite");
        nexus_model::Tensor decoder_grad = nexus_model::Tensor::zeros({2, 4, 8});
        decoder_grad.fill(0.1f);
        decoder.zero_grad();
        expect(finite_tensor(decoder.backward(decoder_grad)) && parameter_grad_energy(decoder) > 0.0f, "decoder backward finite");

        nexus::nn::RNN rnn(4, 6);
        nexus_model::Tensor sequence = nexus_model::Tensor::zeros({2, 5, 4});
        fill_pattern(sequence, 0.04f);
        nexus_model::Tensor hidden = rnn.forward(sequence);
        expect(hidden.rank() == 3 && hidden.size(0) == 2 && hidden.size(1) == 5 && hidden.size(2) == 6, "rnn output [B,T,H]");
        expect(finite_tensor(hidden), "rnn forward finite");
        nexus_model::Tensor rnn_grad = nexus_model::Tensor::zeros({2, 5, 6});
        rnn_grad.fill(0.1f);
        rnn.zero_grad();
        expect(finite_tensor(rnn.backward(rnn_grad)) && parameter_grad_energy(rnn) > 0.0f, "rnn backward finite");

        nexus_model::Tensor volume = nexus_model::Tensor::zeros({2, 3, 4});
        volume.fill(0.5f);
        nexus::nn::Reshape reshape({2, 12});
        nexus_model::Tensor flat = reshape.forward(volume);
        expect(flat.rank() == 2 && flat.size(0) == 2 && flat.size(1) == 12 && flat[0] == 0.5f, "reshape writes the requested shape");
        nexus_model::Tensor flat_grad = nexus_model::Tensor::zeros({2, 12});
        flat_grad.fill(0.25f);
        nexus_model::Tensor restored = reshape.backward(flat_grad);
        expect(restored.rank() == 3 && restored.size(1) == 3 && restored.size(2) == 4 && restored[0] == 0.25f, "reshape restores the input rank");
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
