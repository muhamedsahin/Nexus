/**
 * @file 03_ndarray_and_batch.cpp
 * DataLoader ciktisindaki NDArray ve Batch, kopyalanmadan goruntulenir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor_bridge.hpp"

#include <iostream>

int main() {
  nexusdata::Batch batch;
  batch.inputs = nexusdata::NDArray(nexusdata::Shape{2, 2}, nexusdata::DType::Float32);
  batch.labels = nexusdata::NDArray(nexusdata::Shape{2}, nexusdata::DType::Float32);
  batch.inputs.data<float>()[0] = 1.0f;
  batch.labels.data<float>()[1] = 3.0f;

  nexus::TensorView<float> inputs = nexus::view_inputs(batch);
  nexus::TensorView<float> labels = nexus::view_labels(batch);
  const bool same_in = inputs.data() == batch.inputs.data<float>();
  const bool same_y = labels.data() == batch.labels.data<float>();
  inputs[0] = 8.0f;
  std::cout << "inputs_same=" << same_in << " labels_same=" << same_y
            << " written=" << batch.inputs.data<float>()[0] << "\n";
  return same_in && same_y && batch.inputs.data<float>()[0] == 8.0f ? 0 : 1;
}
