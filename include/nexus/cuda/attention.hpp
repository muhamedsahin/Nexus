#pragma once

/**
 * @file attention.hpp
 * @brief FAZ 5: FlashAttention-2 (CUDA), referans "math" dikkat, ortak parametreler.
 *
 * Sozlesme (promt.txt 5.1): her sorgu satiri i icin anahtar bloklari j uzerinde online
 * softmax; skor matrisi [B,H,Tq,Tk] YAZILMAZ, yalniz LSE [B,H,Tq] (FP32) saklanir.
 *
 * Duzen: Q [.., Tq, H, D], K/V [.., Tk, Hkv, D]; son eksen (D) bitisik. Satir (token), bas
 * ve batch adimlari eleman cinsinden verilir: GPT'nin [B,T,3,H,D] qkv tamponu kopyasiz
 * okunur. O, Q ile ayni sekil; dQ/dK/dV de kendi giris adimlarini kullanir.
 *
 * Maskeler (5.5, 5.7):
 *  - causal: sorgu konumlari anahtarin SONUNA hizalanir, q_pos = tk - tq + i; j > q_pos maskeli.
 *  - window > 0: yalniz q_pos - window < j <= q_pos (kayan pencere; causal ile birlikte).
 *  - softcap > 0: s = softcap * tanh(s / softcap) (Gemma).
 *  - alibi_slopes: s += slope[h] * (j - q_pos) (yalniz causal ile anlamli).
 * GQA/MQA (5.6): kv_heads | heads; h_kv = h / (heads / kv_heads). K/V kopyalanmaz.
 * Varlen (5.4): cu_seqlens_q/k [batch+1] (aygit, int32) verilirse batch b'nin tokenlari
 * paketli tamponun [cu[b], cu[b+1]) satirlaridir; batch adimlari kullanilmaz, tq/tk o
 * zaman batch'teki en uzun boylardir. Belge sinirinda dikkat kesilir, pad yoktur.
 * Dropout (5.F): Philox(seed, offset) ile cekirdek icinde; maske saklanmaz.
 *
 * Desteklenen flash ornekleri: head_dim 64 ve 128, BF16 ve F16, sm_80+.
 * Diger durumlar `flash_supported` false doner; dispatcher (sdpa) "math"e duser ve nedeni
 * NEXUS_SDPA_DEBUG=1 ile stderr'e yazar (sessiz geri donus yok: secim kaydedilir).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/attention/params.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace nexus {
namespace cuda {

using AttentionParams = nexus::AttentionParams;

/// Flash yolu bu parametrelerle calisabilir mi; false ise neden.
[[nodiscard]] bool flash_supported(const AttentionParams& p, std::string* reason = nullptr);

/// O ve LSE [B, H, Tq] (FP32, varlen'de [H, total_q]) yazar. Skor matrisi yazilmaz.
void flash_attention_forward(void* o, float* lse, const void* q, const void* k, const void* v,
                             const AttentionParams& p, cudaStream_t stream);

[[nodiscard]] std::size_t flash_attention_backward_scratch_bytes(const AttentionParams& p);
/// dQ, dK, dV (giris tipinde, kendi adimlariyla: dq ~ q, dk/dv ~ k) yazar.
void flash_attention_backward(void* dq, void* dk, void* dv, const void* dout, const void* q, const void* k,
                              const void* v, const void* o, const float* lse, const AttentionParams& p, void* scratch,
                              cudaStream_t stream);

}  // namespace cuda
}  // namespace nexus
