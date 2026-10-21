/**
 * @file test_phase8_sequence_vision.cpp
 * @brief Yuksek seviye Conv2d, dikkat ve LSTM ileri-geri gecisi.
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

}  // namespace

int main() {
    std::cout << "Nexus Faz 8 katman testi\n";
    try {
        nexus::nn::Conv2d conv(1, 4, 3, 1, 1);
        nexus_model::Tensor image = nexus_model::Tensor::zeros({2, 1, 8, 8});
        for (std::size_t i = 0; i < image.numel(); ++i) {
            image[i] = 0.05f * static_cast<float>(static_cast<int>(i % 9) - 4);
        }
        nexus_model::Tensor conv_out = conv.forward(image);
        expect(conv_out.rank() == 4 && conv_out.size(0) == 2 && conv_out.size(1) == 4 && conv_out.size(2) == 8 &&
                   conv_out.size(3) == 8,
               "conv2d keeps spatial size with padding");
        expect(finite_tensor(conv_out), "conv2d forward finite");
        nexus_model::Tensor conv_grad =
            nexus_model::Tensor::zeros({conv_out.size(0), conv_out.size(1), conv_out.size(2), conv_out.size(3)});
        conv_grad.fill(1.0f);
        conv.zero_grad();
        nexus_model::Tensor conv_input_grad = conv.backward(conv_grad);
        expect(finite_tensor(conv_input_grad) && parameter_grad_energy(conv) > 0.0f, "conv2d backward finite");

        nexus::nn::MultiHeadAttention attention(8, 2);
        nexus_model::Tensor tokens = nexus_model::Tensor::zeros({2, 4, 8});
        for (std::size_t i = 0; i < tokens.numel(); ++i) {
            tokens[i] = 0.02f * static_cast<float>(static_cast<int>(i % 11) - 5);
        }
        nexus_model::Tensor attended = attention.forward(tokens);
        expect(attended.rank() == 3 && attended.size(0) == 2 && attended.size(1) == 4 && attended.size(2) == 8,
               "attention keeps [B,T,E]");
        expect(finite_tensor(attended), "attention forward finite");
        nexus_model::Tensor attention_grad = nexus_model::Tensor::zeros({attended.size(0), attended.size(1), attended.size(2)});
        attention_grad.fill(0.1f);
        attention.zero_grad();
        nexus_model::Tensor attention_input_grad = attention.backward(attention_grad);
        expect(finite_tensor(attention_input_grad) && parameter_grad_energy(attention) > 0.0f, "attention backward finite");

        nexus::nn::LSTM lstm(4, 6);
        nexus_model::Tensor sequence = nexus_model::Tensor::zeros({2, 5, 4});
        for (std::size_t i = 0; i < sequence.numel(); ++i) {
            sequence[i] = 0.03f * static_cast<float>(static_cast<int>(i % 7) - 3);
        }
        nexus_model::Tensor hidden = lstm.forward(sequence);
        expect(hidden.rank() == 3 && hidden.size(0) == 2 && hidden.size(1) == 5 && hidden.size(2) == 6, "lstm output [B,T,H]");
        expect(finite_tensor(hidden), "lstm forward finite");
        nexus_model::Tensor lstm_grad = nexus_model::Tensor::zeros({hidden.size(0), hidden.size(1), hidden.size(2)});
        lstm_grad.fill(0.1f);
        lstm.zero_grad();
        nexus_model::Tensor lstm_input_grad = lstm.backward(lstm_grad);
        expect(finite_tensor(lstm_input_grad) && parameter_grad_energy(lstm) > 0.0f, "lstm backward finite");
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
