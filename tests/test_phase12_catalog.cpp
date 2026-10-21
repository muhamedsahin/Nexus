/**
 * @file test_phase12_catalog.cpp
 * @brief Conv1d, Conv3d, depthwise, GroupNorm ve InstanceNorm.
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

bool backward_ok(nexus::nn::Module& layer, const nexus_model::Tensor& grad) {
    layer.zero_grad();
    return finite_tensor(layer.backward(grad)) && parameter_grad_energy(layer) > 0.0f;
}

}  // namespace

int main() {
    std::cout << "Nexus Faz 12 katalog testi\n";
    try {
        nexus::nn::Conv1d conv1d(1, 4, 3, 1, 1);
        nexus_model::Tensor signal = nexus_model::Tensor::zeros({2, 1, 8});
        fill_pattern(signal, 0.2f);
        nexus_model::Tensor wave = conv1d.forward(signal);
        expect(wave.rank() == 3 && wave.size(0) == 2 && wave.size(1) == 4 && wave.size(2) == 8, "conv1d keeps the length");
        expect(finite_tensor(wave), "conv1d forward finite");
        nexus_model::Tensor wave_grad = nexus_model::Tensor::zeros({2, 4, 8});
        wave_grad.fill(0.1f);
        expect(backward_ok(conv1d, wave_grad), "conv1d backward finite");

        nexus::nn::Conv3d conv3d(1, 2, 3, 1, 1);
        nexus_model::Tensor volume = nexus_model::Tensor::zeros({1, 1, 4, 4, 4});
        fill_pattern(volume, 0.15f);
        nexus_model::Tensor cube = conv3d.forward(volume);
        expect(cube.rank() == 5 && cube.size(1) == 2 && cube.size(2) == 4 && cube.size(4) == 4, "conv3d keeps the volume");
        expect(finite_tensor(cube), "conv3d forward finite");
        nexus_model::Tensor cube_grad = nexus_model::Tensor::zeros({1, 2, 4, 4, 4});
        cube_grad.fill(0.1f);
        expect(backward_ok(conv3d, cube_grad), "conv3d backward finite");

        nexus::nn::DepthwiseConv2d depthwise(4, 3, 1, 1);
        nexus_model::Tensor image = nexus_model::Tensor::zeros({2, 4, 8, 8});
        fill_pattern(image, 0.1f);
        nexus_model::Tensor filtered = depthwise.forward(image);
        expect(filtered.rank() == 4 && filtered.size(1) == 4 && filtered.size(2) == 8 && filtered.size(3) == 8, "depthwise keeps NCHW");
        expect(finite_tensor(filtered), "depthwise forward finite");
        nexus_model::Tensor image_grad = nexus_model::Tensor::zeros({2, 4, 8, 8});
        image_grad.fill(0.1f);
        expect(backward_ok(depthwise, image_grad), "depthwise backward finite");

        nexus::nn::GroupNorm group(4, 8);
        nexus_model::Tensor grouped = nexus_model::Tensor::zeros({2, 8, 4, 4});
        fill_pattern(grouped, 0.3f);
        nexus_model::Tensor group_out = group.forward(grouped);
        expect(group_out.numel() == grouped.numel() && finite_tensor(group_out), "groupnorm keeps the tensor");
        nexus_model::Tensor group_grad = nexus_model::Tensor::zeros({2, 8, 4, 4});
        group_grad.fill(0.1f);
        expect(backward_ok(group, group_grad), "groupnorm backward finite");

        nexus::nn::InstanceNorm instance(4);
        nexus_model::Tensor sample = nexus_model::Tensor::zeros({2, 4, 3, 3});
        fill_pattern(sample, 0.25f);
        nexus_model::Tensor instance_out = instance.forward(sample);
        expect(instance_out.rank() == 4 && instance_out.size(1) == 4 && finite_tensor(instance_out), "instancenorm keeps NCHW");
        nexus_model::Tensor instance_grad = nexus_model::Tensor::zeros({2, 4, 3, 3});
        instance_grad.fill(0.1f);
        expect(backward_ok(instance, instance_grad), "instancenorm backward finite");
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
