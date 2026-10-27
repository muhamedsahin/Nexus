#pragma once

/**
 * @file adam.hpp
 * @brief FAZ 18.E: ZeRO-Offload CPU AdamW — AVX2, tek gecis (gradyan acma + kirpma + AdamW + BF16 RNE golge).
 *
 * Islem sirasi fused::adamw_multi_tensor (ve GPU cekirdegi) ile BIREBIR ayni; FMA YOK (carpma + toplama ayri,
 * her biri IEEE yuvarlama), sqrt ve bolme dogru yuvarlanmis (vsqrtps / vdivps) -> skaler yolla bit-bit esit.
 * Gradyan F32 ya da BF16 olabilir; BF16 golge (out) gradyan tamponuyla ayni bellek olabilir (oge basina once oku).
 * sq_norm: sabit 1M'lik parcalar (FP64) -> sonuc is parcacigi sayisindan bagimsiz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cpu/threads.hpp"

#include <cstddef>
#include <cstdint>

namespace nexus {
namespace cpu {

enum class GradType : std::uint8_t { F32, BF16 };

struct AdamCoeffs {
  float b1, ob1, b2, ob2, inv1, inv2, lr, lr_wd, eps;
};
/// fused::adamw_multi_tensor ile ayni katsayilar (double'da hesap, FP32'ye tek yuvarlama).
[[nodiscard]] AdamCoeffs adam_coeffs(double lr, double beta1, double beta2, double eps, double weight_decay,
                                     std::int64_t step);

/// sum(g^2), FP64; [b, e) bolgesi icin, tek is parcacigi (AVX2, 4 FP64 serit, sabit sira).
[[nodiscard]] double sq_norm_range(const void* grad, GradType t, std::size_t b, std::size_t e) noexcept;
/// p, m, v [b, e) guncelle; out != nullptr ise BF16 RNE golge yaz. decay: lr_wd uygula.
void adamw_range(float* p, float* m, float* v, const void* grad, GradType t, float clip, const AdamCoeffs& c,
                 bool decay, std::uint16_t* out, std::size_t b, std::size_t e) noexcept;

/// FP32 -> BF16 RNE (NaN sessiz), AVX2.
void to_bf16_range(const float* x, std::uint16_t* out, std::size_t n) noexcept;

}  // namespace cpu
}  // namespace nexus
