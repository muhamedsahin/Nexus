#pragma once

/**
 * @file tensor_optim.hpp
 * @brief Dalga 2: Tensor depolamali parametreler icin AdamW (her dtype parametre: F32 / BF16 / F16; CPU ve CUDA).
 *
 * Algoritma fused::adamw_multi_tensor ile ayni: katsayilar double'da hesaplanip FP32'ye tek yuvarlama
 * (cpu::adam_coeffs), eleman islem sirasi ayni, FMA yok -> F32 parametrede FAZ 2 yoluyla bit bit (CPU) ve
 * cihazda __f*_rn ile ayni sira. m / v her zaman F32 ve parametrenin cihazinda. Kirpma: PyTorch clip_grad_norm_
 * gibi tum gradyanlarin global normu (FP64, her parametre sq_norm toplami), katsayi min(1, max / (norm + 1e-6)).
 * Parametre FAZ 2 vector depolamaliysa acik hata (o yolda fused::adamw_multi_tensor / lm::AdamW).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace nexus {
namespace autograd {

class TensorAdamW {
 public:
  struct Config {
    double lr = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;
    double max_grad_norm = 0.0;  // 0: kirpma yok
  };
  struct Param {
    Variable var;       // Tensor depolamali yaprak (requires_grad)
    bool decay = true;  // weight_decay uygulansin mi (nanoGPT: 2B agirliklar evet, bias / LN hayir)
  };

  TensorAdamW(std::vector<Param> params, Config config);

  /// Bir adim (gradyani olmayan parametre atlanir). Kirpma acikken global gradyan normunu dondurur, degilse -1.
  double step();
  void zero_grad();
  void set_lr(double lr) noexcept { config_.lr = lr; }
  [[nodiscard]] std::int64_t steps() const noexcept { return step_; }
  [[nodiscard]] const Config& config() const noexcept { return config_; }

 private:
  std::vector<Param> params_;
  std::vector<nexus::Tensor> m_, v_;
  Config config_;
  std::int64_t step_ = 0;
};

}  // namespace autograd
}  // namespace nexus
