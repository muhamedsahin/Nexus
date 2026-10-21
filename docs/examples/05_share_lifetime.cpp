/**
 * @file 05_share_lifetime.cpp
 * share(), NDArray'in shared_ptr sayacini artirir. Float dizisi kopyalanmaz.
 * Kaynak nesne olunce bellek, tensör durdukca gecerli kalir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor_bridge.hpp"

#include <iostream>

int main() {
  nexus::Tensor held;
  float* raw = nullptr;
  {
    nexusdata::NDArray array(nexusdata::Shape{3}, nexusdata::DType::Float32);
    array.data<float>()[1] = 6.0f;
    raw = array.data<float>();
    held = nexus::share(array);
  }
  const bool alive = held.data() == raw && held.data()[1] == 6.0f;
  std::cout << "alive=" << alive << " mid=" << held.data()[1] << "\n";
  return alive ? 0 : 1;
}
