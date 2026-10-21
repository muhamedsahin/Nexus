/**
 * @file bench_phase12_catalog.cpp
 * @brief Conv1d, Conv3d, depthwise ve normalizasyon ileri gecis suresi.
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
    nexus::nn::Conv1d conv1d(16, 32, 3, 1, 1);
    nexus_model::Tensor signal = nexus_model::Tensor::zeros({8, 16, 128});
    signal.fill(0.02f);
    const double conv1d_seconds = seconds_of(1, 4, [&] { conv1d.forward(signal); });

    nexus::nn::Conv3d conv3d(4, 8, 3, 1, 1);
    nexus_model::Tensor volume = nexus_model::Tensor::zeros({2, 4, 8, 8, 8});
    volume.fill(0.02f);
    const double conv3d_seconds = seconds_of(1, 4, [&] { conv3d.forward(volume); });

    nexus::nn::DepthwiseConv2d depthwise(32, 3, 1, 1);
    nexus_model::Tensor image = nexus_model::Tensor::zeros({8, 32, 16, 16});
    image.fill(0.02f);
    const double depth_seconds = seconds_of(1, 4, [&] { depthwise.forward(image); });

    nexus::nn::GroupNorm group(8, 32);
    nexus_model::Tensor batch = nexus_model::Tensor::zeros({8, 32, 16, 16});
    batch.fill(0.2f);
    const double group_seconds = seconds_of(1, 8, [&] { group.forward(batch); });

    nexus::nn::InstanceNorm instance(32);
    const double instance_seconds = seconds_of(1, 8, [&] { instance.forward(batch); });

    volatile float sink = conv1d.forward(signal)[0] + conv3d.forward(volume)[0] + depthwise.forward(image)[0] +
                          group.forward(batch)[0] + instance.forward(batch)[0];
    std::cout << "Nexus Faz 12 benchmark\n";
    std::cout << "| katman | us | cagri/s |\n";
    std::cout << "| Conv1d 8x16x128 k3 | " << conv1d_seconds * 1.0e6 << " | " << (1.0 / conv1d_seconds) << " |\n";
    std::cout << "| Conv3d 2x4x8x8x8 k3 | " << conv3d_seconds * 1.0e6 << " | " << (1.0 / conv3d_seconds) << " |\n";
    std::cout << "| DepthwiseConv2d 8x32x16x16 k3 | " << depth_seconds * 1.0e6 << " | " << (1.0 / depth_seconds) << " |\n";
    std::cout << "| GroupNorm 8x32x16x16 g8 | " << group_seconds * 1.0e6 << " | " << (1.0 / group_seconds) << " |\n";
    std::cout << "| InstanceNorm 8x32x16x16 | " << instance_seconds * 1.0e6 << " | " << (1.0 / instance_seconds) << " |\n";
    return std::isfinite(sink) ? 0 : 1;
}
