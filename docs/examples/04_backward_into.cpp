/**
 * @file 04_backward_into.cpp
 * MSE turevi cagiranin tamponuna yazilir. Yeni sonuc vektoru ayirilmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/loss/backward_into.hpp"
#include "nexusloss/losses/regression.hpp"

#include <iostream>
#include <vector>

int main() {
  const std::vector<float> prediction{1.0f, 3.0f};
  const std::vector<float> target{0.0f, 1.0f};
  nexusloss::MSELoss<float> loss;
  loss.forward(prediction, target);

  std::vector<float> gradient(prediction.size());
  nexus::backward_into(loss, std::span<float>(gradient));
  std::cout << "dL=" << gradient[0] << " " << gradient[1] << "\n";
  // d/dx (x-y)^2 = 2(x-y), Mean bunu 1/N ile carpar. N=2.
  return gradient[0] == 1.0f && gradient[1] == 2.0f ? 0 : 1;
}
