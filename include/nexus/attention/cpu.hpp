#pragma once

/**
 * @file cpu.hpp
 * @brief FAZ 5.A referans "math" dikkat (FP64 biriktirici) ve 5.G CPU FlashAttention (FP32).
 *
 * Tamponlar host FP32, duzen/maske/GQA/varlen/dropout sozlesmesi nexus/cuda/attention.hpp ile
 * aynidir; tek fark cu_seqlens_q/k burada HOST isaretcileridir.
 *
 * - `math_attention_*`: her satir icin tam skor vektorunu FP64'te kurar (skoru "yazan" arka
 *   uc). Test kahini; yavastir.
 * - `flash_attention_*_cpu`: ayni online softmax algoritmasi, 64 anahtarlik bloklar, FP32;
 *   ileri (batch, head, q-blok) uzerinde, geri dK/dV icin (batch, kv-head, k-blok) ve dQ
 *   icin (batch, head, q-blok) uzerinde is parcacikli; atomik yok (belirlenimci).
 *
 * Dropout indeksi (CUDA ile ayni): e = (( (b*H + h) * Tq + i) * Tk + j) bitisik durumda,
 * varlen'de e = ((h * total_q + q_global) * max_tk + j). Philox: fused::philox_uniform.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/attention/params.hpp"

namespace nexus {
namespace attention {

/// O (q adimlariyla) ve LSE ([B,H,Tq] ya da varlen [H,total_q]) yazar.
void math_attention_forward(float* o, float* lse, const float* q, const float* k, const float* v,
                            const AttentionParams& p);
/// dQ, dK, dV (sifirdan yazilir). D_i = rowsum(dO o O) icin o ve lse kullanilir.
void math_attention_backward(float* dq, float* dk, float* dv, const float* dout, const float* q, const float* k,
                             const float* v, const float* o, const float* lse, const AttentionParams& p);

void flash_attention_forward_cpu(float* o, float* lse, const float* q, const float* k, const float* v,
                                 const AttentionParams& p, int threads = 0);
void flash_attention_backward_cpu(float* dq, float* dk, float* dv, const float* dout, const float* q, const float* k,
                                  const float* v, const float* o, const float* lse, const AttentionParams& p,
                                  int threads = 0);

}  // namespace attention
}  // namespace nexus
