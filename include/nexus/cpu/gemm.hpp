#pragma once

/**
 * @file gemm.hpp
 * @brief FAZ 18.A/B/C: Cok-yollu CPU GEMM/GEMV — AVX2+FMA+F16C (4x16 MASM + 6x16), AVX-512 (F/BF16/VNNI) ve
 *        AMX karo (tile) yollari; FP32 / BF16 / FP16 / INT8 (s8) / INT4 (q4) agirliklar; satir-ana.
 *
 * C[M, N] = alpha * op(A)[M, K] * op(B)[K, N] + beta * C (+ bias[N]). A FP32.
 * B: [K, N] (b_trans = false) ya da [N, K] (b_trans = true, agirlik duzeni); tip F32 / BF16 / F16.
 *
 * Mimari ve optimizasyonlar:
 *  - M > 4 (GEMM): B paneli [KC=512, N] tum havuz iscileri tarafindan paralel olarak TEK KEZ ortak 64-bayt
 *    hizali tampona paketlenir (F32, BF16->FP32 AVX2 kaydirma, F16->FP32 F16C vcvtph2ps; b_trans=true iken
 *    AVX2 8x8 yazmac-ici devrik paketleme). Ardindan her isci kendi M satir dilimini (MC=64, L2 yerliligi)
 *    Win64 MASM `nexus_avx2_kernel_4x16` (8 YMM biriktirici, sifir yigin dokumu, 8x K-acma) ile hesaplar.
 *  - M <= 4 (GEMV): Paketlemesiz dogrudan akis.
 *      * b_trans = true (B [N, K]): 4 cikis sutunu ayni anda (8 YMM biriktirici) K boyunca akar.
 *      * b_trans = false (B [K, N]): N sutunlari 64'luk (M=1) / 32'lik (M=2) / 16'lik (M<=4) seritler halinde
 *        iscilere bolunur; biriktiriciler tum K boyunca YMM yazmaclarinda tutulur (ara tampon yok, tek gecis).
 *  - INT8 (`PackedS8`): Kanal basina simetrik nicemleme, [ceil(N/16)][Kp/4][16][4] VNNI-uyumlu duzen;
 *    AVX2 `_mm256_maddubs_epi16(|a|, sign(b, a))` + `_mm256_madd_epi16(1)` 4x16 mikrocekirdek.
 *  - INT4 (`PackedQ4`): Kanal basina simetrik 4-bit nicemleme ([-7, +7], +8 ofsetli nibble paketleme),
 *    [ceil(N/16)][Kp/8][16][4] nibble-seritli duzen; tek 32-bayt yuklemeden iki 16x4 INT8 blok acilarak
 *    `maddubs_epi16` ile hesaplanir (FP32'ye gore 8x, BF16'ya gore 4x dusuk bellek bant genisligi).
 *  - AVX-512 (F / BF16 `vdpbf16ps` / VNNI `vpdpbusd`) ve AMX (`tdpbf16ps` / `tdpbssd` 16x16 karo):
 *    Donanimda mevcut oldugunda calisma zamaninda secilir (`active_gemm_isa()`); AVX-512/AMX olmayan
 *    makinelerde (orn. Zen 3) VNNI/AMX karo veri duzeni ve biriktirme semantigi `gemm_vnni_bf16` ve
 *    `gemm_amx_tile_s8` uzerinden dogrulanir (KURAL 7).
 *
 * Belirlenimcilik: Her cikis ogesi C[m, n] k = 0..K-1 sirasiyla tek bir isci tarafindan biriktirilir;
 * sonuc is parcacigi sayisindan (1, 2, 4, 8, 16) bagimsiz olarak bit-bit aynidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cpu/threads.hpp"

#include <cstdint>
#include <vector>

namespace nexus {
namespace cpu {

enum class BType : std::uint8_t { F32, BF16, F16 };

enum class KernelIsa : std::uint8_t {
  Scalar = 0,
  Avx2Fma = 1,
  Avx512F = 2,
  Avx512Bf16 = 3,
  Avx512Vnni = 4,
  AmxTile = 5,
};

[[nodiscard]] const char* to_string(KernelIsa isa) noexcept;
[[nodiscard]] KernelIsa active_gemm_isa(BType btype = BType::F32) noexcept;

struct GemmArgs {
  int M = 0, N = 0, K = 0;
  const float* A = nullptr;
  int lda = 0;
  bool a_trans = false;  // A [K, M] (dW = X^T dY)
  const void* B = nullptr;
  int ldb = 0;
  BType btype = BType::F32;
  bool b_trans = false;  // B [N, K]
  float* C = nullptr;
  int ldc = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  const float* bias = nullptr;  // [N]
};

void gemm(const GemmArgs& a, Pool* pool = nullptr);

/// nexus_model::kernels ile ayni anlam (FAZ 18: lm / MoE / derleyici yurutucusu / paralel lm bu katmana gecti).
/// C[m, n] = A[m, k] B[k, n]
void gemm_nn(const float* a, const float* b, float* c, int m, int k, int n);
/// C[k, n] += A^T B; A [m, k], B [m, n]
void gemm_tn_accum(const float* a, const float* b, float* c, int k, int m, int n);
/// C[m, k] += A B^T; A [m, n], B [k, n]
void gemm_nt_accum(const float* a, const float* b, float* c, int m, int n, int k);

/// Test kancasi (A/B parite + hiz olcumu): true iken gemm_nn / gemm_tn_accum / gemm_nt_accum eski skaler
/// nexus_model::kernels yoluna gider. Varsayilan false; yalniz testler acar.
void set_reference_gemm(bool on) noexcept;
[[nodiscard]] bool reference_gemm() noexcept;

/// BF16 / F16 donusturme yardimcilari (RNE).
[[nodiscard]] std::vector<std::uint16_t> to_bf16(const float* x, std::size_t n);
[[nodiscard]] std::vector<std::uint16_t> to_f16(const float* x, std::size_t n);

// ---- INT8 (VNNI / AVX2 maddubs) ---------------------------------------------------------
struct PackedS8 {
  int N = 0, K = 0, Kp = 0;            // Kp: K, 4'e yuvarlanmis
  std::vector<std::int8_t> data;       // [ceil(N/16)][Kp/4][16][4]
  std::vector<float> scale;            // [N] = amax / 127
};
/// W: b_trans ? [N, K] : [K, N] (FP32) -> kanal basina simetrik INT8, paketli.
[[nodiscard]] PackedS8 pack_s8(const float* W, int N, int K, bool b_trans);
/// x [M, K] (lda) -> q [M, Kp] (sifir dolgulu) + scale [M].
void quantize_rows_s8(const float* x, int M, int K, int lda, std::int8_t* q, int Kp, float* scale, Pool* pool = nullptr);
/// C[M, N] = (Aq Bq) * sa[m] * sb[n] + bias[n].
void gemm_s8(const std::int8_t* Aq, const float* sa, int M, const PackedS8& B, float* C, int ldc, const float* bias = nullptr,
             Pool* pool = nullptr);

// ---- INT4 (Simetrik 4-bit nibble-paketlenmis AVX2 maddubs) ------------------------------
struct PackedQ4 {
  int N = 0, K = 0, Kp = 0;            // Kp: K, 8'e yuvarlanmis
  std::vector<std::uint8_t> data;      // [ceil(N/16)][Kp/8][16][4]: alt nibble k..k+3, ust nibble k+4..k+7 (+8 ofsetli)
  std::vector<float> scale;            // [N] = amax / 7
};
/// W: b_trans ? [N, K] : [K, N] (FP32) -> kanal basina simetrik INT4 ([-7, +7]), paketli.
[[nodiscard]] PackedQ4 pack_q4(const float* W, int N, int K, bool b_trans);
/// C[M, N] = (Aq Bq4) * sa[m] * sb[n] + bias[n]; Aq quantize_rows_s8(..., Kp=B.Kp) ile uretilir.
void gemm_q4(const std::int8_t* Aq, const float* sa, int M, const PackedQ4& B, float* C, int ldc, const float* bias = nullptr,
             Pool* pool = nullptr);

// ---- AVX-512 VNNI/BF16 ve AMX 16x16 Karo (Tile) Yol + Dogrulama -------------------------
struct PackedBf16Vnni {
  int N = 0, K = 0, Kp = 0;            // Kp: K, 2'ye yuvarlanmis (vdpbf16ps / tdpbf16ps ciftleri)
  std::vector<std::uint16_t> data;     // [ceil(N/16)][Kp/2][16][2]
};
[[nodiscard]] PackedBf16Vnni pack_bf16_vnni(const float* W, int N, int K, bool b_trans);
/// AVX-512 BF16 (`vdpbf16ps`) / AMX BF16 (`tdpbf16ps` 16x16 karo) ciftli nokta carpimi yolu;
/// donanimda AVX-512_BF16 / AMX varsa donanim komutuyla, yoksa birebir VNNI cift-biriktirme semantigiyle calisir.
void gemm_bf16_vnni(const float* A, int M, int lda, const PackedBf16Vnni& B, float* C, int ldc, const float* bias = nullptr,
                    Pool* pool = nullptr);

}  // namespace cpu
}  // namespace nexus
