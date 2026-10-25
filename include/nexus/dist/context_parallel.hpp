#pragma once

/**
 * @file context_parallel.hpp
 * @brief FAZ 12.F: baglam (context) paralelligi — Ring Attention (zigzag/dengeli parca atamasi) ve
 *        Ulysses (all-to-all ile bas <-> dizi ekseni degisimi). CPU FlashAttention (FAZ 5 LSE) uzerinde.
 *
 * Dizi T, cp rank'e bolunur. Parca atamasi:
 *   Contiguous: cp parca (T / cp), rank r parca r'yi tutar. Nedensel maskede rank r, r + 1 blok
 *               hesaplar (son rank ilkinin cp kati) — dengesiz.
 *   Zigzag:     2 cp parca (T / 2cp), rank r parcalar {r, 2cp - 1 - r}'yi tutar; her rank ayni sayida
 *               blok hesaplar (dengeli).
 * Yerel duzen parca-ana: [nch][B][Tc][heads][D] (her parca bitisik [B, Tc, H, D] blogu; FlashAttention
 * dogrudan cagrilir). LSE [nch][B][H][Tc].
 *
 * Ring ileri: K/V bloklari halkada dolasir (send r+1 / recv r-1, hesapla ortusur); her adimda yerel
 * Flash + online softmax birlesimi: lse = log(e^l1 + e^l2), o = o1 e^(l1-lse) + o2 e^(l2-lse).
 * Parca cifti (q parcasi gi, k parcasi gj): gj < gi tam blok, gj == gi nedensel kosegen, gj > gi atlanir.
 * Ring geri: K/V yine dolasir; her blokta global O ve LSE ile Flash geri (D_i = rowsum(dO o O) global);
 * dK/dV biriktiricisi K/V ile birlikte dolasir ve sonunda sahibine doner.
 * Ulysses: q/k/v all-to-all ile [B, T, H/cp, D]'ye (tam dizi, bas parcasi), yerel nedensel Flash,
 * cikis all-to-all ile geri. H ve Hkv cp'ye bolunmeli.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/process_group.hpp"

#include <cstdint>
#include <vector>

namespace nexus {
namespace dist {

enum class CpLayout : std::uint8_t { Contiguous, Zigzag };
enum class CpAlgorithm : std::uint8_t { Ring, Ulysses };
[[nodiscard]] const char* to_string(CpLayout layout) noexcept;
[[nodiscard]] const char* to_string(CpAlgorithm algorithm) noexcept;

/// Toplam parca sayisi (Contiguous: cp, Zigzag: 2 cp).
[[nodiscard]] int cp_num_chunks(int cp, CpLayout layout) noexcept;
/// Rank'in global parca kimlikleri (yerel sirayla).
[[nodiscard]] std::vector<int> cp_chunks(int cp, int rank, CpLayout layout);

struct CpAttention {
  int batch = 1;
  int seq = 0;  // global T
  int heads = 1;
  int kv_heads = 1;
  int head_dim = 64;
  float scale = 0.0f;  // 0: 1/sqrt(D)
  CpLayout layout = CpLayout::Zigzag;
  int threads = 1;  // Flash CPU is parcacigi (rank'ler is parcacigi oldugunda 1)
  [[nodiscard]] int chunk_len(int cp) const;  // Tc
};

/// Bir rank'in hesapladigi (q parcasi, k parcasi) blok sayisi (nedensel yuk olcusu).
[[nodiscard]] int ring_attention_blocks(int cp, int rank, CpLayout layout);

/// q/o [nch,B,Tc,H,D], k/v [nch,B,Tc,Hk,D], lse [nch,B,H,Tc] (yerel). Kollektif (cp grubu).
void ring_attention_forward(ProcessGroup& cp, const CpAttention& a, const float* q, const float* k, const float* v,
                            float* o, float* lse);
/// dq/dk/dv sifirdan yazilir.
void ring_attention_backward(ProcessGroup& cp, const CpAttention& a, const float* q, const float* k, const float* v,
                             const float* o, const float* lse, const float* dout, float* dq, float* dk, float* dv);

/// Ulysses. lse: [B, H/cp, T] (bas parcali, geri gecis icin saklanir).
void ulysses_attention_forward(ProcessGroup& cp, const CpAttention& a, const float* q, const float* k, const float* v,
                               float* o, float* lse);
void ulysses_attention_backward(ProcessGroup& cp, const CpAttention& a, const float* q, const float* k,
                                const float* v, const float* o, const float* lse, const float* dout, float* dq,
                                float* dk, float* dv);
[[nodiscard]] std::size_t ulysses_lse_size(const CpAttention& a, int cp) noexcept;

/// Tam dizi [B, T, h, D] <-> yerel parca-ana [nch, B, Tc, h, D] (test ve model icin).
void cp_scatter_sequence(const float* full, float* local, int batch, int seq, int width, int cp, int rank,
                         CpLayout layout);
void cp_gather_sequence(const float* local, float* full, int batch, int seq, int width, int cp, int rank,
                        CpLayout layout);

}  // namespace dist
}  // namespace nexus
