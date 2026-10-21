/**
 * @file test_phase9_blocks.cpp
 * @brief Transformer, GRU, gomme ve BatchNorm yuksek seviye kabugu.
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
    std::cout << "Nexus Faz 9 blok testi\n";
    try {
        nexus::nn::TransformerEncoder encoder(8, 2, 16);
        nexus_model::Tensor tokens = nexus_model::Tensor::zeros({2, 4, 8});
        fill_pattern(tokens, 0.05f);
        nexus_model::Tensor encoded = encoder.forward(tokens);
        expect(encoded.rank() == 3 && encoded.size(0) == 2 && encoded.size(1) == 4 && encoded.size(2) == 8,
               "transformer keeps [B,T,E]");
        expect(finite_tensor(encoded), "transformer forward finite");
        nexus_model::Tensor encoder_grad = nexus_model::Tensor::zeros({2, 4, 8});
        encoder_grad.fill(0.1f);
        encoder.zero_grad();
        expect(finite_tensor(encoder.backward(encoder_grad)) && parameter_grad_energy(encoder) > 0.0f,
               "transformer backward finite");

        nexus::nn::GRU gru(4, 6);
        nexus_model::Tensor sequence = nexus_model::Tensor::zeros({2, 5, 4});
        fill_pattern(sequence, 0.04f);
        nexus_model::Tensor hidden = gru.forward(sequence);
        expect(hidden.rank() == 3 && hidden.size(0) == 2 && hidden.size(1) == 5 && hidden.size(2) == 6, "gru output [B,T,H]");
        expect(finite_tensor(hidden), "gru forward finite");
        nexus_model::Tensor gru_grad = nexus_model::Tensor::zeros({2, 5, 6});
        gru_grad.fill(0.1f);
        gru.zero_grad();
        expect(finite_tensor(gru.backward(gru_grad)) && parameter_grad_energy(gru) > 0.0f, "gru backward finite");

        nexus::nn::Embedding embedding(11, 4, 0);
        nexus_model::Tensor indices = nexus_model::Tensor::zeros({2, 3});
        indices[0] = 0.0f;
        indices[1] = 2.0f;
        indices[2] = 5.0f;
        indices[3] = 0.0f;
        indices[4] = 7.0f;
        indices[5] = 1.0f;
        nexus_model::Tensor looked = embedding.forward(indices);
        expect(looked.rank() == 3 && looked.size(2) == 4, "embedding adds the vector axis");
        expect(looked[0] == 0.0f && looked[1] == 0.0f && looked[2] == 0.0f && looked[3] == 0.0f, "padding index is zero");
        expect(finite_tensor(looked), "embedding forward finite");
        nexus_model::Tensor embed_grad = nexus_model::Tensor::zeros({2, 3, 4});
        embed_grad.fill(0.2f);
        embedding.zero_grad();
        embedding.backward(embed_grad);
        expect(parameter_grad_energy(embedding) > 0.0f, "embedding scatters the gradient");

        nexus::nn::BatchNorm1d norm(8);
        nexus_model::Tensor batch = nexus_model::Tensor::zeros({4, 8});
        fill_pattern(batch, 0.3f);
        nexus_model::Tensor normalized = norm.forward(batch);
        expect(normalized.numel() == batch.numel() && finite_tensor(normalized), "batchnorm forward finite");
        nexus_model::Tensor norm_grad = nexus_model::Tensor::zeros({4, 8});
        norm_grad.fill(0.1f);
        norm.zero_grad();
        expect(finite_tensor(norm.backward(norm_grad)) && parameter_grad_energy(norm) > 0.0f, "batchnorm backward finite");
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
