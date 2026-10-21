/**
 * @file 06_custom_activation.cpp
 * Lambda ile Swish. Geri turev verilmedigi icin sonlu fark kullanilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/custom/activation.hpp"

#include <cmath>
#include <iostream>

int main() {
    auto swish = nexus::make_activation("SwishCustom", [](float x) { return x / (1.0f + std::exp(-x)); });
    const float x = 1.0f;
    float y = 0.0f;
    float grad = 0.0f;
    const float upstream = 1.0f;
    swish.forward_buffer(&x, &y, 1);
    swish.backward_buffer(&x, &upstream, &grad, 1);
    const float sigmoid = 1.0f / (1.0f + std::exp(-x));
    const float analytic = sigmoid + x * sigmoid * (1.0f - sigmoid);
    std::cout << "y=" << y << " finite_grad=" << grad << " analytic=" << analytic << "\n";
    return std::fabs(y - x * sigmoid) < 1.0e-6f && std::fabs(grad - analytic) < 2.0e-3f ? 0 : 1;
}
