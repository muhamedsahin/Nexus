/**
 * @file test_phase1_bridge.cpp
 * @brief Faz 1: alti kutuphane tek yurutulebilir dosyada, sifir kopya, sifir tahsisat.
 *
 * Kayip ailelerinin tamami `test_phase1_loss_families.cpp`, host/cihaz
 * aynasi `test_phase1_mirrored_tensor.cpp` icindedir. Bu dosya kopru
 * katmanini (gorunum, paylasim, ayirma kaydi, MatrixFlash adresleri) dener.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/nexus.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(NEXUS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

using nexus_test::AllocationScope;
using nexus_test::expect;

bool same_address(const void* lhs, const void* rhs) { return lhs == rhs; }

void test_libraries_and_addresses() {
  nexus_test::section("kutuphaneler ve sifir kopya");

  nexus_model::Tensor model = nexus_model::Tensor::from_values({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  nexus::TensorView<float> model_view = nexus::view(model);
  expect(same_address(model_view.data(), model.data()), "nexus_model data() adresi ayni");
  model_view[1] = 9.0f;
  expect(model[1] == 9.0f, "gorunum uzerinden yazmak model tensörunu degistirir");
  expect(model_view.aligned_to(64), "nexus_model deposu 64 bayt hizali");

  nexusdata::NDArray array(nexusdata::Shape{2, 3}, nexusdata::DType::Float32);
  array.data<float>()[0] = 1.5f;
  nexus::TensorView<float> array_view = nexus::view(array);
  expect(same_address(array_view.data(), array.data<float>()), "NDArray data() adresi ayni");
  array_view[0] = 4.25f;
  expect(array.data<float>()[0] == 4.25f, "NDArray gorunumu ayni bellek");

  const std::uint64_t before_share = nexus::allocation_count();
  nexus::Tensor shared = nexus::share(array);
  expect(same_address(shared.data(), array.data<float>()), "share() float dizisini kopyalamaz");
  expect(nexus::allocation_count() == before_share, "share() Nexus ayirma sayacini artirmaz");

  float* raw = array.data<float>();
  {
    nexusdata::NDArray local(nexusdata::Shape{4}, nexusdata::DType::Float32);
    local.data<float>()[0] = 7.0f;
    raw = local.data<float>();
    shared = nexus::share(local);
    expect(same_address(shared.data(), raw), "paylasilan isaretci yerel NDArray ile ayni");
  }
  expect(shared.data()[0] == 7.0f, "NDArray olunce shared_ptr bellegi ayakta tutar");

  nexusdata::Batch batch;
  batch.inputs = nexusdata::NDArray(nexusdata::Shape{2, 3}, nexusdata::DType::Float32);
  batch.labels = nexusdata::NDArray(nexusdata::Shape{2}, nexusdata::DType::Float32);
  nexus::TensorView<float> inputs = nexus::view_inputs(batch);
  nexus::TensorView<float> labels = nexus::view_labels(batch);
  expect(same_address(inputs.data(), batch.inputs.data<float>()), "batch girdi adresi ayni");
  expect(same_address(labels.data(), batch.labels.data<float>()), "batch etiket adresi ayni");
  inputs[2] = 0.5f;
  expect(batch.inputs.data<float>()[2] == 0.5f, "batch gorunumu ayni bellek");

  const std::uint64_t before_alias = nexus::allocation_count();
  float stack_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  nexus::Tensor alias = nexus::Tensor::alias(stack_values, nexus::Shape{4});
  expect(same_address(alias.data(), stack_values), "alias yigin adresini korur");
  expect(nexus::allocation_count() == before_alias, "alias bellek ayirmaz");
  expect(!alias.owns_storage(), "alias sahibi degildir");

  const std::uint64_t before_owned = nexus::allocation_count();
  nexus::Tensor owned = nexus::Tensor::zeros(nexus::Shape{2, 64});
  expect(nexus::allocation_count() == before_owned + 1, "zeros tek host blogu ayirir");
  expect(owned.view().aligned_to(64), "sahip olunan host 64 bayt hizali");
  expect(owned.space() == nexus::MemorySpace::Host, "zeros host alanindadir");
  bool all_zero = true;
  for (std::size_t i = 0; i < owned.numel(); ++i) {
    if (owned.data()[i] != 0.0f) {
      all_zero = false;
    }
  }
  expect(all_zero, "zeros tum elemanlari sifirlar");

  nexus::Tensor pinned = nexus::Tensor::pinned(nexus::Shape{32});
  expect(pinned.space() == nexus::MemorySpace::PinnedHost, "pinned host alani");
  expect(pinned.view().aligned_to(64), "pinned bellek 64 bayt hizali");
  expect(nexus::owns_allocation(pinned.data()), "pinned kayit altinda");
  pinned[0] = 1.0f;
  expect(pinned[0] == 1.0f, "pinned host uzerinden okunur");

  const auto simd = nexus_optim::cpu_simd_level();
  std::cout << "  info  nexus_optim simd=" << nexus_optim::simd_level_name(simd) << "\n";
  float kernel_buffer[4] = {};
  nexus_model::kernels::fill(kernel_buffer, 2.0f, 4);
  expect(kernel_buffer[3] == 2.0f, "nexus_model cekirdegi calisti");

  using BestFn = bool (nexus_train::CheckpointManager::*)(double);
  BestFn best = &nexus_train::CheckpointManager::is_best;
  volatile unsigned char symbol_byte = *reinterpret_cast<const unsigned char*>(&best);
  std::cout << "  info  nexus_train symbol_byte=" << static_cast<int>(symbol_byte) << "\n";
  expect(sizeof(best) > 0, "nexus_train sembolu baglandi");

  nexus_train::TrainConfig config;
  expect(config.num_epochs == 10, "nexus_train varsayilan epoch");
}

void test_backward_into() {
  nexus_test::section("backward_into (LossBase kapisi)");

  const std::vector<float> pred{1.0f, 3.0f};
  const std::vector<float> target{0.0f, 1.0f};
  nexusloss::MSELoss<float> mse;
  const std::vector<float> loss_value = mse.forward(pred, target);
  expect(loss_value.size() == 1, "MSE forward skaler doner");
  expect(mse.gradient_size() == pred.size(), "MSE gradient_size tahmin uzunlugu");

  std::vector<float> into(pred.size());
  const std::vector<float> from_api = mse.backward();
  nexusloss::core::inplace_fallback_allocations().store(0, std::memory_order_relaxed);
  {
    AllocationScope scope;
    for (int step = 0; step < 32; ++step) {
      mse.backward_into(into);
    }
    expect(scope.count() == 0, "MSE backward_into 32 adimda 0 heap ayirma");
  }
  expect(nexusloss::core::inplace_fallback_allocations().load() == 0, "MSE fallback ayirma yoluna girmedi");
  expect(nexus_test::identical<float>(from_api, into), "MSE backward ve backward_into ayni");
  expect(into[0] == 1.0f && into[1] == 2.0f, "MSE mean turevi 2*(p-y)/N");
  expect(nexus_test::finite<float>(into), "MSE turevi sonlu");

  nexus::backward_into(mse, std::span<float>(into));
  expect(into[0] == 1.0f && into[1] == 2.0f, "nexus::backward_into ayni sonucu yazar");

  std::vector<float> reusable;
  (void)nexus::backward_into(mse, reusable);
  {
    AllocationScope scope;
    const std::span<float> written = nexus::backward_into(mse, reusable);
    expect(scope.count() == 0 && written.size() == 2, "nexus::backward_into(vector&) ikinci cagrida ayirmaz");
  }

  nexusloss::HuberLoss<float> huber(1.0f);
  const std::vector<float> huber_pred{0.25f, 3.0f};
  const std::vector<float> huber_target{0.0f, 0.0f};
  huber.forward(huber_pred, huber_target);
  const std::vector<float> huber_ref = huber.backward();
  std::vector<float> huber_out(huber_pred.size());
  {
    AllocationScope scope;
    huber.backward_into(huber_out);
    expect(scope.count() == 0, "Huber backward_into ayirma yapmaz");
  }
  expect(nexus_test::identical<float>(huber_ref, huber_out), "Huber backward ve backward_into ayni");

  nexusloss::BCEWithLogitsLoss<float> bce;
  const std::vector<float> logits{0.0f, 2.0f, -2.0f};
  const std::vector<float> labels{0.0f, 1.0f, 1.0f};
  bce.forward(logits, labels);
  const std::vector<float> bce_ref = bce.backward();
  std::vector<float> bce_out(logits.size());
  {
    AllocationScope scope;
    bce.backward_into(bce_out);
    expect(scope.count() == 0, "BCEWithLogits backward_into ayirma yapmaz");
  }
  expect(nexus_test::identical<float>(bce_ref, bce_out), "BCEWithLogits backward ve backward_into ayni");

  bool threw = false;
  try {
    std::vector<float> too_small(1);
    mse.backward_into(too_small);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  expect(threw, "kisa grad_out hata firlatir");
}

void test_matrixflash() {
#if !defined(NEXUS_ENABLE_CUDA)
  nexus_test::section("matrixflash: CUDA kapali, cihaz testi atlandi");
  expect(true, "CPU derlemesi MatrixFlash cihaz yolu olmadan da gecerli");
#else
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count <= 0) {
    nexus_test::section("matrixflash: CUDA aygiti yok, calisma testi atlandi");
    expect(true, "MatrixFlash derlendi, aygit yok");
    return;
  }
  std::cout << "matrixflash cihaz=" << device_count << "\n";
  matrix_pro::Tensor tensor(std::vector<std::size_t>{2, 4});
  tensor.host_ptr()[0] = 1.25f;
  nexus::TensorView<float> host_view = nexus::view(tensor);
  expect(same_address(host_view.data(), tensor.host_ptr()), "MatrixFlash host adresi ayni");
  expect(host_view.space() == nexus::MemorySpace::Host, "dolu host aynasi host alanidir");
  host_view[0] = 2.5f;
  expect(tensor.host_ptr()[0] == 2.5f, "MatrixFlash host gorunumu ayni bellek");

  matrix_pro::Tensor slice = tensor.narrow(0, 1, 1);
  nexus::TensorView<float> device_view = nexus::view(slice);
  expect(slice.device_data() == tensor.device_data(), "narrow ayni cihaz tamponunu paylasir");
  expect(same_address(device_view.data(), slice.device_data() + slice.storage_offset()),
         "narrow sifir kopya ofseti");
  expect(device_view.space() == nexus::MemorySpace::Device, "host aynasi olmayan dilim cihaz alanidir");

  const std::uint64_t before = nexus::allocation_count();
  nexus::Tensor device = nexus::Tensor::zeros(nexus::Shape{16}, nexus::MemorySpace::Device, nexus::DeviceRef::cuda(0));
  expect(nexus::allocation_count() == before + 1, "CUDA tamponu bir ayirma sayilir");
  expect(device.space() == nexus::MemorySpace::Device, "cihaz tensöru device alaninda");
  expect(device.data() != nullptr, "cudaMalloc isaretci dondurdu");
  expect(nexus::owns_allocation(device.data()), "cihaz tamponu kayitli");
#endif
}

}  // namespace

int main() {
  std::cout << "Nexus Faz 1 kopru testi\n";
  try {
    test_libraries_and_addresses();
    test_backward_into();
    test_matrixflash();
  } catch (const std::exception& error) {
    std::cout << "  FAIL  istisna: " << error.what() << "\n";
    return 1;
  }
  return nexus_test::finish("test_phase1_bridge");
}
