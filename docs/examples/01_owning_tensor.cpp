/**
 * @file 01_owning_tensor.cpp
 * Sahip olunan host tensöru. zeros bellegi ayirir ve sifirlar.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor.hpp"

#include <iostream>

int main() {
  nexus::Tensor weights = nexus::Tensor::zeros(nexus::Shape{2, 3});
  weights[0] = 0.5f;
  weights[5] = -1.0f;
  std::cout << "numel=" << weights.numel()
            << " align64=" << weights.view().aligned_to(64)
            << " w0=" << weights[0] << " w5=" << weights[5] << "\n";
  return weights.numel() == 6 && weights.view().aligned_to(64) ? 0 : 1;
}
