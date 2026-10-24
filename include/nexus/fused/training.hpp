#pragma once

/**
 * @file training.hpp
 * @brief FAZ 3 egitim yolu cekirdekleri (CPU referans + uretim yolu).
 *
 *  - `cross_entropy_shifted`: [B*T, V] logit uzerinde TEK geçiste satir max,
 *    log-sum-exp, kayip ve dlogits = softmax - hedef. Kaydirma kopya ile degil
 *    indeks ofsetiyle (hedef = tokens[b, t+1]); son konum ignore. `dlogits`
 *    `logits` ile ayni tampon olabilir (yerinde): logit boyutunda ikinci tampon
 *    ayrilmaz. ignore_index, label_smoothing ve z-loss (PaLM) destekli; kayip
 *    gecerli token sayisina bolunur.
 *  - `rms_norm`: autograd dugumu, FP32 istatistik, rstd kaydedilir.
 *  - `embedding_pos`: wte[tok] + wpe[pos] tek geçiste; geri geçiste iki tabloya
 *    deterministik (indeks sirali, atomiksiz) segment toplami.
 *  - `philox4x32_10` + `dropout_philox`: (seed, offset) ile yeniden uretilebilir
 *    maske; maske SAKLANMAZ, geri geçiste ayni sayaclarla yeniden uretilir.
 *  - `adamw_multi_tensor`: tum parametreleri tek cagrida guncelleyen foreach
 *    AdamW; global gradyan normu ayni cagrida tek indirgemeyle hesaplanir ve
 *    kirpma uygulanir. Matematik NexusOptim AdamW ile birebir aynidir.
 *
 * CUDA cekirdek karsiliklari (cuBLASLt epilog, warp-basi LN, multi-tensor-apply)
 * bu dosyanin kapsami disindadir; GPU yolu FAZ 3 raporunda acik is olarak yazilidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nexus {
namespace fused {

// ---- Kaydirilmis, fuzyonlu capraz entropi -------------------------------------------
struct CrossEntropyOptions {
  std::int32_t ignore_index = -100;
  float label_smoothing = 0.0f;
  float z_loss = 0.0f;  // PaLM: z_loss * log(Z)^2
};

struct CrossEntropyResult {
  float loss = 0.0f;          // gecerli tokenlar uzerinde ortalama (CE + z-loss)
  float z_loss = 0.0f;        // ortalama z-loss katkisi (rapor icin)
  std::int64_t valid = 0;     // kayba katilan token sayisi
};

/// logits: [batch*time, vocab] satir-ana. tokens: [batch*time]. dlogits null ise
/// yalnizca kayip; dlogits == logits ise yerinde. Gecersiz satirlarin dlogits'i 0.
/// Bellek ayirmaz (sicak yolda heap yok).
CrossEntropyResult cross_entropy_shifted(const float* logits, const std::int32_t* tokens, int batch, int time,
                                         int vocab, const CrossEntropyOptions& options, float* dlogits);

/// Autograd surumu: logits Variable [batch*time, vocab] -> skaler kayip.
[[nodiscard]] autograd::Variable cross_entropy_shifted(const autograd::Variable& logits,
                                                       const std::vector<std::int32_t>& tokens, int batch, int time,
                                                       const CrossEntropyOptions& options = {});

// ---- RMSNorm -------------------------------------------------------------------------
/// y = x / sqrt(mean(x^2) + eps) * gamma. x: [M,N], gamma: [N].
[[nodiscard]] autograd::Variable rms_norm(const autograd::Variable& x, const autograd::Variable& gamma,
                                          float eps = 1.0e-6f);

// ---- Gomme + konum fuzyonu --------------------------------------------------------------
/// out[b*T+t] = wte[tokens[b*T+t]] + wpe[t]. wte: [V,D], wpe: [P>=T, D] -> [B*T, D].
[[nodiscard]] autograd::Variable embedding_pos(const autograd::Variable& wte, const autograd::Variable& wpe,
                                               const std::vector<std::int32_t>& tokens, int batch, int time);

// ---- Philox4x32-10 dropout -------------------------------------------------------------
/// Random123 Philox4x32-10 (Salmon ve ark., SC'11). counter 128 bit, key 64 bit.
[[nodiscard]] std::array<std::uint32_t, 4> philox4x32_10(std::array<std::uint32_t, 4> counter,
                                                         std::array<std::uint32_t, 2> key) noexcept;

/// Eleman i icin [0,1) duzgun sayi: sayac = (offset + i/4), anahtar = seed.
[[nodiscard]] float philox_uniform(std::uint64_t seed, std::uint64_t offset, std::uint64_t index) noexcept;

/// Ters olcekli dropout; maske kaydedilmez, geri geçiste (seed, offset) ile yeniden uretilir.
[[nodiscard]] autograd::Variable dropout_philox(const autograd::Variable& x, float p, std::uint64_t seed,
                                                std::uint64_t offset, bool training = true);

// ---- Foreach / fuzyonlu AdamW -----------------------------------------------------------
struct ParamSlot {
  float* param = nullptr;
  const float* grad = nullptr;
  float* exp_avg = nullptr;
  float* exp_avg_sq = nullptr;
  std::size_t numel = 0;
};

/// Hiperparametreler NexusOptim/PyTorch gibi double tutulur; cekirdek katsayilari
/// (1-beta dahil) double'da hesaplanip FP32'ye bir kez yuvarlanir.
struct AdamWConfig {
  double lr = 1.0e-3;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps = 1.0e-8;
  double weight_decay = 0.0;
  double max_grad_norm = 0.0;  // <= 0: kirpma yok
};

struct AdamWStepInfo {
  double grad_norm = 0.0;    // kirpma oncesi global L2 norm (FP64 biriktirme)
  float clip_coef = 1.0f;    // uygulanan olcek (<=1)
};

/// Tum parametreleri tek cagrida gunceller. `step` 1'den baslar (bias duzeltmesi).
AdamWStepInfo adamw_multi_tensor(std::span<const ParamSlot> slots, const AdamWConfig& config, std::int64_t step);

/// Global L2 normu tek indirgemede hesaplar (FP64 biriktirme).
[[nodiscard]] double global_grad_norm(std::span<const ParamSlot> slots) noexcept;

}  // namespace fused
}  // namespace nexus
