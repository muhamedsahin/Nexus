/**
 * @file test_phase10_structure.cpp
 * @brief Havuz, duzlestirme, dropout, BatchNorm2d ve artik baglanti.
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

bool same_values(const nexus_model::Tensor& left, const nexus_model::Tensor& right) {
    if (left.numel() != right.numel()) {
        return false;
    }
    for (std::size_t i = 0; i < left.numel(); ++i) {
        if (left[i] != right[i]) {
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
    std::cout << "Nexus Faz 10 yapi testi\n";
    try {
        nexus_model::Tensor image = nexus_model::Tensor::zeros({1, 1, 2, 2});
        image[0] = 1.0f;
        image[1] = 3.0f;
        image[2] = 2.0f;
        image[3] = 0.0f;
        nexus::nn::MaxPool2d max_pool(2);
        nexus_model::Tensor pooled = max_pool.forward(image);
        expect(pooled.numel() == 1 && pooled[0] == 3.0f, "maxpool keeps the first maximum");
        nexus_model::Tensor pool_grad = nexus_model::Tensor::zeros({1, 1, 1, 1});
        pool_grad.fill(1.0f);
        nexus_model::Tensor max_input_grad = max_pool.backward(pool_grad);
        expect(max_input_grad[0] == 0.0f && max_input_grad[1] == 1.0f && max_input_grad[2] == 0.0f && max_input_grad[3] == 0.0f,
               "maxpool routes the gradient to argmax");

        image.fill(4.0f);
        nexus::nn::AvgPool2d avg_pool(2);
        nexus_model::Tensor averaged = avg_pool.forward(image);
        expect(averaged.numel() == 1 && averaged[0] == 4.0f, "avgpool mean");
        nexus_model::Tensor avg_grad = nexus_model::Tensor::zeros({1, 1, 1, 1});
        avg_grad.fill(4.0f);
        nexus_model::Tensor avg_input_grad = avg_pool.backward(avg_grad);
        expect(avg_input_grad[0] == 1.0f && avg_input_grad[1] == 1.0f && avg_input_grad[2] == 1.0f && avg_input_grad[3] == 1.0f,
               "avgpool spreads the gradient");

        nexus_model::Tensor field = nexus_model::Tensor::zeros({1, 1, 2, 2});
        field[0] = 1.0f;
        field[1] = 2.0f;
        field[2] = 3.0f;
        field[3] = 4.0f;
        nexus::nn::GlobalAvgPool2d global_pool;
        nexus_model::Tensor collapsed = global_pool.forward(field);
        expect(collapsed.rank() == 2 && collapsed.numel() == 1 && collapsed[0] == 2.5f, "global pool is the spatial mean");
        nexus_model::Tensor global_grad = nexus_model::Tensor::zeros({1, 1});
        global_grad.fill(4.0f);
        nexus_model::Tensor global_input_grad = global_pool.backward(global_grad);
        expect(global_input_grad[0] == 1.0f && global_input_grad[3] == 1.0f, "global pool broadcasts the gradient");

        nexus_model::Tensor volume = nexus_model::Tensor::zeros({2, 3, 4});
        volume.fill(0.5f);
        nexus::nn::Flatten flatten;
        nexus_model::Tensor flat = flatten.forward(volume);
        expect(flat.rank() == 2 && flat.size(0) == 2 && flat.size(1) == 12, "flatten keeps the batch axis");
        nexus_model::Tensor flat_grad = nexus_model::Tensor::zeros({2, 12});
        flat_grad.fill(0.25f);
        nexus_model::Tensor restored = flatten.backward(flat_grad);
        expect(restored.rank() == 3 && restored.size(1) == 3 && restored.size(2) == 4 && restored[0] == 0.25f, "flatten restores the rank");

        nexus_model::Tensor values = nexus_model::Tensor::zeros({4, 8});
        values.fill(1.5f);
        nexus::nn::Dropout dropout(1.0f);
        dropout.eval();
        expect(same_values(dropout.forward(values), values), "dropout eval is the identity");
        dropout.train();
        nexus_model::Tensor dropped = dropout.forward(values);
        bool all_zero = true;
        for (std::size_t i = 0; i < dropped.numel(); ++i) {
            all_zero = all_zero && dropped[i] == 0.0f;
        }
        expect(all_zero, "dropout train with p=1 zeroes the tensor");
        nexus::nn::Sequential sequence;
        sequence.add<nexus::nn::Dropout>(1.0f);
        sequence.eval();
        expect(same_values(sequence.forward(values), values), "sequential eval reaches dropout");
        sequence.train();
        nexus_model::Tensor sequence_dropped = sequence.forward(values);
        all_zero = true;
        for (std::size_t i = 0; i < sequence_dropped.numel(); ++i) {
            all_zero = all_zero && sequence_dropped[i] == 0.0f;
        }
        expect(all_zero, "sequential train reaches dropout");

        nexus::nn::BatchNorm2d norm(4);
        nexus_model::Tensor batch = nexus_model::Tensor::zeros({4, 4, 2, 2});
        for (std::size_t i = 0; i < batch.numel(); ++i) {
            batch[i] = 0.2f * static_cast<float>(static_cast<int>(i % 7) - 3);
        }
        nexus_model::Tensor normalized = norm.forward(batch);
        expect(normalized.rank() == 4 && normalized.size(1) == 4 && finite_tensor(normalized), "batchnorm2d keeps NCHW");
        nexus_model::Tensor norm_grad = nexus_model::Tensor::zeros({4, 4, 2, 2});
        norm_grad.fill(0.1f);
        norm.zero_grad();
        expect(finite_tensor(norm.backward(norm_grad)) && parameter_grad_energy(norm) > 0.0f, "batchnorm2d backward finite");
        norm.eval();
        expect(finite_tensor(norm.forward(batch)), "batchnorm2d eval uses running statistics");

        nexus::nn::Residual residual(std::make_shared<nexus::nn::Linear>(4, 4));
        nexus_model::Tensor row = nexus_model::Tensor::zeros({2, 4});
        row.fill(0.3f);
        nexus_model::Tensor skip = residual.forward(row);
        expect(skip.rank() == 2 && skip.size(1) == 4 && finite_tensor(skip), "residual keeps the linear shape");
        nexus_model::Tensor residual_grad = nexus_model::Tensor::zeros({2, 4});
        residual_grad.fill(0.1f);
        residual.zero_grad();
        expect(finite_tensor(residual.backward(residual_grad)) && parameter_grad_energy(residual) > 0.0f, "residual backward reaches the branch");

        nexus::nn::Residual identity(std::make_shared<nexus::nn::ReLU>());
        nexus_model::Tensor positive = nexus_model::Tensor::zeros({2, 3});
        positive.fill(1.25f);
        nexus_model::Tensor doubled = identity.forward(positive);
        expect(doubled[0] == 2.5f && doubled[5] == 2.5f, "residual adds the identity path");
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
