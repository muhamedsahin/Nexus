/**
 * @file 02_zero_copy_model.cpp
 * nexus_model::Tensor ile ayni adresi paylasan gorunum.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor_bridge.hpp"

#include <iostream>

int main() {
  nexus_model::Tensor layer = nexus_model::Tensor::from_values({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
  nexus::TensorView<float> view = nexus::view(layer);
  view[2] = 30.0f;
  const bool same = view.data() == layer.data();
  std::cout << "same_address=" << same << " layer2=" << layer[2] << "\n";
  return same && layer[2] == 30.0f ? 0 : 1;
}
