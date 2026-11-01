/**
 * @file tensor_optim.cpp
 * @brief Dalga 2: TensorAdamW (bkz. tensor_optim.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/tensor_optim.hpp"

#include "nexus/core/tensor_ops.hpp"
#include "nexus/cpu/adam.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nexus {
namespace autograd {

TensorAdamW::TensorAdamW(std::vector<Param> params, Config config) : params_(std::move(params)), config_(config) {
  for (const Param& p : params_) {
    if (!p.var.is_tensor_backed() || !p.var.requires_grad() || !p.var.is_leaf())
      throw std::invalid_argument("Nexus TensorAdamW: parametre Tensor depolamali, requires_grad yaprak olmali "
                                  "(FAZ 2 vector parametreleri icin fused::adamw_multi_tensor)");
    const nexus::Tensor t = p.var.tensor();
    if (!t.is_contiguous()) throw std::invalid_argument("Nexus TensorAdamW: parametre bitisik olmali");
    m_.push_back(nexus::Tensor::zeros(t.shape(), nexus::DType::F32, t.space(), t.device()));
    v_.push_back(nexus::Tensor::zeros(t.shape(), nexus::DType::F32, t.space(), t.device()));
  }
}

double TensorAdamW::step() {
  ++step_;
  double norm = -1.0;
  float clip = 1.0f;
  if (config_.max_grad_norm > 0.0) {
    double sq = 0.0;
    for (const Param& p : params_) {
      if (!p.var.meta() || !p.var.meta()->grad) continue;
      const nexus::Tensor s = nexus::ops::sq_norm(p.var.grad().tensor()).to(nexus::MemorySpace::Host, nexus::DeviceRef::cpu());
      sq += s.data_as<double>()[0];
    }
    norm = std::sqrt(sq);
    clip = static_cast<float>(std::min(1.0, config_.max_grad_norm / (norm + 1.0e-6)));  // clip_grad_norm_ / fused ile ayni
  }
  const cpu::AdamCoeffs c = cpu::adam_coeffs(config_.lr, config_.beta1, config_.beta2, config_.eps, config_.weight_decay, step_);
  const bool wd = static_cast<float>(config_.weight_decay) != 0.0f;
  for (std::size_t i = 0; i < params_.size(); ++i) {
    Param& p = params_[i];
    if (!p.var.meta() || !p.var.meta()->grad) continue;
    nexus::Tensor param = p.var.tensor();
    const nexus::Tensor grad = p.var.grad().tensor();
    nexus::ops::AdamWCoeffs k{c.b1, c.ob1, c.b2, c.ob2, c.inv1, c.inv2, c.lr, c.lr_wd, c.eps, clip, p.decay && wd};
    nexus::ops::adamw_(param, grad, m_[i], v_[i], k);
  }
  return norm;
}

void TensorAdamW::zero_grad() {
  for (Param& p : params_) p.var.zero_grad();
}

}  // namespace autograd
}  // namespace nexus
