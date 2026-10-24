#pragma once

/**
 * @file autocast.hpp
 * @brief Autocast baglami ve op politika tablosu — FAZ 4.1.
 *
 * PyTorch autocast listeleri referans alinmistir:
 *  - Dar (BF16/F16): matmul, linear, bmm, conv, attention.
 *  - FP32: softmax, log_softmax, cross_entropy, layer_norm, rms_norm, exp, log, pow,
 *    sum, mean, norm (FP32 biriktirici; sayisal guvenlik, 4.6).
 *  - En genis giris tipine yukselt: add, cat, where.
 *  - Degismez (cikti = giris tipi): reshape/view/permute/transpose, embedding, dropout.
 *
 * `AutocastGuard(device, dtype, enabled)` is parcacigi basina ve aygit turu basinadir;
 * ic ice kurulabilir, yikimda onceki durum geri gelir. Autocast kapaliyken her op giris
 * tiplerinin yukseltmesini (nexus::promote) dondurur.
 *
 * Nexus GPU GPT yolu dtype kararlarini bu tablodan alir (gpt.cpp); test_autocast_policy
 * tabloyu tum op siniflari icin dogrular.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/device.hpp"
#include "nexus/core/dtype.hpp"
#include "nexus/core/error.hpp"

#include <array>
#include <initializer_list>
#include <string>

namespace nexus {
namespace amp {

enum class OpClass : std::uint8_t {
  Matmul,
  Linear,
  Bmm,
  Conv,
  Attention,
  Softmax,
  LogSoftmax,
  CrossEntropy,
  LayerNorm,
  RmsNorm,
  Exp,
  Log,
  Pow,
  Sum,
  Mean,
  Norm,
  Add,
  Cat,
  Where,
  Reshape,
  Embedding,
  Dropout,
  kCount
};

enum class CastPolicy : std::uint8_t {
  Lower,    // autocast tipine dusur (Tensor Core)
  Fp32,     // FP32'de calis
  Promote,  // girislerin en genisi
  Keep,     // giris tipi korunur (tek girisli yapisal op)
};

[[nodiscard]] constexpr CastPolicy policy_for(OpClass op) noexcept {
  switch (op) {
    case OpClass::Matmul:
    case OpClass::Linear:
    case OpClass::Bmm:
    case OpClass::Conv:
    case OpClass::Attention:
      return CastPolicy::Lower;
    case OpClass::Softmax:
    case OpClass::LogSoftmax:
    case OpClass::CrossEntropy:
    case OpClass::LayerNorm:
    case OpClass::RmsNorm:
    case OpClass::Exp:
    case OpClass::Log:
    case OpClass::Pow:
    case OpClass::Sum:
    case OpClass::Mean:
    case OpClass::Norm:
      return CastPolicy::Fp32;
    case OpClass::Add:
    case OpClass::Cat:
    case OpClass::Where:
      return CastPolicy::Promote;
    case OpClass::Reshape:
    case OpClass::Embedding:
    case OpClass::Dropout:
    case OpClass::kCount:
      return CastPolicy::Keep;
  }
  return CastPolicy::Keep;
}

[[nodiscard]] constexpr const char* to_string(OpClass op) noexcept {
  constexpr const char* names[] = {"matmul", "linear",  "bmm",  "conv", "attention", "softmax", "log_softmax",
                                   "cross_entropy", "layer_norm", "rms_norm", "exp", "log", "pow", "sum",
                                   "mean", "norm", "add", "cat", "where", "reshape", "embedding", "dropout"};
  return op < OpClass::kCount ? names[static_cast<int>(op)] : "unknown";
}

struct AutocastState {
  bool enabled = false;
  DType dtype = DType::BF16;
};

namespace detail {
inline AutocastState& state_for(DeviceKind kind) {
  thread_local std::array<AutocastState, 8> states{};
  const auto index = static_cast<std::size_t>(kind);
  if (index >= states.size()) {
    throw TensorError("Nexus autocast: aygit turu desteklenmiyor");
  }
  return states[index];
}
}  // namespace detail

[[nodiscard]] inline AutocastState autocast_state(DeviceKind kind) { return detail::state_for(kind); }

class AutocastGuard {
 public:
  AutocastGuard(DeviceKind kind, DType dtype, bool enabled = true) : kind_(kind), previous_(detail::state_for(kind)) {
    if (enabled && dtype != DType::BF16 && dtype != DType::F16) {
      throw TensorError("Nexus autocast: autocast tipi BF16 veya F16 olmali");
    }
    detail::state_for(kind) = AutocastState{enabled, dtype};
  }
  ~AutocastGuard() { detail::state_for(kind_) = previous_; }
  AutocastGuard(const AutocastGuard&) = delete;
  AutocastGuard& operator=(const AutocastGuard&) = delete;

 private:
  DeviceKind kind_;
  AutocastState previous_;
};

/// Opun bu baglamdaki cikti (ve hesap girisi) tipi.
[[nodiscard]] inline DType compute_dtype(DeviceKind kind, OpClass op, std::initializer_list<DType> inputs) {
  if (inputs.size() == 0) {
    throw TensorError(std::string("Nexus autocast: ") + to_string(op) + " en az bir giris tipi ister");
  }
  DType widest = *inputs.begin();
  for (const DType t : inputs) {
    widest = promote(widest, t);
  }
  const AutocastState st = detail::state_for(kind);
  if (!st.enabled) {
    return widest;
  }
  const bool all_float = is_floating(widest);
  switch (policy_for(op)) {
    case CastPolicy::Lower:
      return all_float ? st.dtype : widest;
    case CastPolicy::Fp32:
      return all_float ? DType::F32 : widest;
    case CastPolicy::Promote:
      return widest;
    case CastPolicy::Keep:
      return *inputs.begin();
  }
  return widest;
}

}  // namespace amp
}  // namespace nexus
