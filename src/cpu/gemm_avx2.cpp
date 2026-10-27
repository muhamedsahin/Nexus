/**
 * @file gemm_avx2.cpp
 * @brief FAZ 18.A/B/C: AVX2 + FMA + F16C GEMM/GEMV cekirdekleri (FP32, BF16, FP16, INT8, INT4) ve
 *        AVX-512 VNNI/BF16 + AMX 16x16 karo yollari (bkz. nexus/cpu/gemm.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cpu/gemm.hpp"

#include "nexus/cpu/features.hpp"
#include "nexus_model/core/kernels.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace nexus {
namespace cpu {

#if defined(_MSC_VER) && defined(_M_X64)
extern "C" void nexus_avx2_kernel_4x16(int kc, const float* a, const float* b, float* c, int ldc, float beta);
#endif

const char* to_string(KernelIsa k) noexcept {
  switch (k) {
    case KernelIsa::Scalar: return "scalar";
    case KernelIsa::Avx2Fma: return "avx2_fma";
    case KernelIsa::Avx512F: return "avx512f";
    case KernelIsa::Avx512Bf16: return "avx512_bf16";
    case KernelIsa::Avx512Vnni: return "avx512_vnni";
    case KernelIsa::AmxTile: return "amx_tile";
  }
  return "unknown";
}

KernelIsa active_gemm_isa(BType btype) noexcept {
  const Features& f = features();
  if (btype == BType::BF16 && f.amx_tile && f.amx_bf16) return KernelIsa::AmxTile;
  if (btype == BType::BF16 && f.avx512bf16 && f.avx512f) return KernelIsa::Avx512Bf16;
  if (f.avx512f) return KernelIsa::Avx512F;
  if (f.avx2 && f.fma) return KernelIsa::Avx2Fma;
  return KernelIsa::Scalar;
}

namespace {

constexpr int kMr = 4;
constexpr int kNr = 16;
constexpr int kKc = 512;
constexpr int kMc = 64;

struct AlignedPanel {
  float* data = nullptr;
  std::size_t count = 0;

  void ensure(std::size_t needed) {
    if (count >= needed) return;
    release();
    count = std::max<std::size_t>(needed, 1024);
#if defined(_MSC_VER)
    data = static_cast<float*>(_aligned_malloc(count * sizeof(float), 64));
#else
    data = static_cast<float*>(std::aligned_alloc(64, ((count * sizeof(float)) + 63u) & ~std::size_t{63}));
#endif
    if (!data) throw std::bad_alloc();
  }

  void release() noexcept {
    if (!data) return;
#if defined(_MSC_VER)
    _aligned_free(data);
#else
    std::free(data);
#endif
    data = nullptr;
    count = 0;
  }

  ~AlignedPanel() { release(); }
};

thread_local AlignedPanel t_panel_a;
thread_local AlignedPanel t_panel_b;

inline float bf16f(std::uint16_t h) noexcept {
  const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline float load_b_scalar(const GemmArgs& g, int k, int n) noexcept {
  const std::size_t i = g.b_trans ? static_cast<std::size_t>(n) * g.ldb + k : static_cast<std::size_t>(k) * g.ldb + n;
  switch (g.btype) {
    case BType::F32: return static_cast<const float*>(g.B)[i];
    case BType::BF16: return bf16f(static_cast<const std::uint16_t*>(g.B)[i]);
    case BType::F16: return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<const std::uint16_t*>(g.B)[i])));
  }
  return 0.0f;
}

/// B'den (offset i) 8 ardisik ogeyi FP32 __m256 olarak yukler.
inline __m256 load_vec8(const void* base, BType btype, std::size_t i) noexcept {
  switch (btype) {
    case BType::F32: return _mm256_loadu_ps(static_cast<const float*>(base) + i);
    case BType::BF16: {
      const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const std::uint16_t*>(base) + i));
      return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
    }
    case BType::F16:
      return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const std::uint16_t*>(base) + i)));
  }
  return _mm256_setzero_ps();
}

inline __m256 load_b8(const GemmArgs& g, int k, int n) noexcept {
  return load_vec8(g.B, g.btype, static_cast<std::size_t>(k) * g.ldb + n);
}

#if !(defined(_MSC_VER) && defined(_M_X64))
void kernel_4x16_fallback(int kc, const float* a, const float* b, float* c, int ldc, float beta) {
  __m256 c00, c01, c10, c11, c20, c21, c30, c31;
  if (beta == 0.0f) {
    c00 = _mm256_setzero_ps(); c01 = c00; c10 = c00; c11 = c00;
    c20 = c00; c21 = c00; c30 = c00; c31 = c00;
  } else {
    const __m256 s = _mm256_set1_ps(beta);
    c00 = _mm256_mul_ps(s, _mm256_loadu_ps(c));
    c01 = _mm256_mul_ps(s, _mm256_loadu_ps(c + 8));
    c10 = _mm256_mul_ps(s, _mm256_loadu_ps(c + ldc));
    c11 = _mm256_mul_ps(s, _mm256_loadu_ps(c + ldc + 8));
    c20 = _mm256_mul_ps(s, _mm256_loadu_ps(c + 2 * ldc));
    c21 = _mm256_mul_ps(s, _mm256_loadu_ps(c + 2 * ldc + 8));
    c30 = _mm256_mul_ps(s, _mm256_loadu_ps(c + 3 * ldc));
    c31 = _mm256_mul_ps(s, _mm256_loadu_ps(c + 3 * ldc + 8));
  }
  for (int p = 0; p < kc; ++p) {
    const __m256 b0 = _mm256_load_ps(b);
    const __m256 b1 = _mm256_load_ps(b + 8);
    const __m256 a0 = _mm256_broadcast_ss(a);
    c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
    const __m256 a1 = _mm256_broadcast_ss(a + 1);
    c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
    const __m256 a2 = _mm256_broadcast_ss(a + 2);
    c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
    const __m256 a3 = _mm256_broadcast_ss(a + 3);
    c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
    a += kMr;
    b += kNr;
  }
  _mm256_storeu_ps(c, c00); _mm256_storeu_ps(c + 8, c01);
  _mm256_storeu_ps(c + ldc, c10); _mm256_storeu_ps(c + ldc + 8, c11);
  _mm256_storeu_ps(c + 2 * ldc, c20); _mm256_storeu_ps(c + 2 * ldc + 8, c21);
  _mm256_storeu_ps(c + 3 * ldc, c30); _mm256_storeu_ps(c + 3 * ldc + 8, c31);
}
#endif

inline void call_kernel_4x16(int kc, const float* a, const float* b, float* c, int ldc, float beta) {
#if defined(_MSC_VER) && defined(_M_X64)
  nexus_avx2_kernel_4x16(kc, a, b, c, ldc, beta);
#else
  kernel_4x16_fallback(kc, a, b, c, ldc, beta);
#endif
}

void kernel_edge(int mr, int nr, int kc, const float* a, const float* b, float* c, int ldc, float beta) {
  for (int row = 0; row < mr; ++row) {
    float* crow = c + static_cast<std::size_t>(row) * static_cast<std::size_t>(ldc);
    if (beta == 0.0f) {
      int col = 0;
      for (; col + 8 <= nr; col += 8) _mm256_storeu_ps(crow + col, _mm256_setzero_ps());
      for (; col < nr; ++col) crow[col] = 0.0f;
    } else if (beta != 1.0f) {
      const __m256 s = _mm256_set1_ps(beta);
      int col = 0;
      for (; col + 8 <= nr; col += 8) _mm256_storeu_ps(crow + col, _mm256_mul_ps(s, _mm256_loadu_ps(crow + col)));
      for (; col < nr; ++col) crow[col] *= beta;
    }
    for (int p = 0; p < kc; ++p) {
      const float aval = a[static_cast<std::size_t>(p) * kMr + row];
      const __m256 av = _mm256_set1_ps(aval);
      const float* brow = b + static_cast<std::size_t>(p) * kNr;
      int col = 0;
      for (; col + 8 <= nr; col += 8) {
        _mm256_storeu_ps(crow + col, _mm256_fmadd_ps(av, _mm256_load_ps(brow + col), _mm256_loadu_ps(crow + col)));
      }
      for (; col < nr; ++col) crow[col] = std::fma(aval, brow[col], crow[col]);
    }
  }
}

/// 4xkc A mikro-panelini paketler; alpha burada A'ya yedirilerek mikrocekirdek icinde carpim kaldirilir.
void pack_a_4(float* dst, const GemmArgs& g, int i0, int mr, int p0, int kc, float alpha) {
  if (!g.a_trans && mr == kMr) {
    const __m256 scale = _mm256_set1_ps(alpha);
    const float* row0 = g.A + static_cast<std::size_t>(i0) * g.lda + p0;
    const float* row1 = row0 + g.lda;
    const float* row2 = row0 + 2 * static_cast<std::size_t>(g.lda);
    const float* row3 = row0 + 3 * static_cast<std::size_t>(g.lda);
    int p = 0;
    for (; p + 8 <= kc; p += 8) {
      __m256 r0 = _mm256_loadu_ps(row0 + p);
      __m256 r1 = _mm256_loadu_ps(row1 + p);
      __m256 r2 = _mm256_loadu_ps(row2 + p);
      __m256 r3 = _mm256_loadu_ps(row3 + p);
      if (alpha != 1.0f) {
        r0 = _mm256_mul_ps(r0, scale);
        r1 = _mm256_mul_ps(r1, scale);
        r2 = _mm256_mul_ps(r2, scale);
        r3 = _mm256_mul_ps(r3, scale);
      }
      const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
      const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
      const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
      const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
      const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
      const __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
      const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
      const __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
      float* out = dst + static_cast<std::size_t>(p) * kMr;
      _mm256_storeu_ps(out, _mm256_permute2f128_ps(u0, u1, 0x20));
      _mm256_storeu_ps(out + 8, _mm256_permute2f128_ps(u2, u3, 0x20));
      _mm256_storeu_ps(out + 16, _mm256_permute2f128_ps(u0, u1, 0x31));
      _mm256_storeu_ps(out + 24, _mm256_permute2f128_ps(u2, u3, 0x31));
    }
    for (; p < kc; ++p) {
      float* out = dst + static_cast<std::size_t>(p) * kMr;
      out[0] = alpha * row0[p];
      out[1] = alpha * row1[p];
      out[2] = alpha * row2[p];
      out[3] = alpha * row3[p];
    }
    return;
  }
  if (g.a_trans && mr == kMr) {
    const __m128 scale = _mm_set1_ps(alpha);
    for (int p = 0; p < kc; ++p) {
      const float* src = g.A + static_cast<std::size_t>(p0 + p) * g.lda + i0;
      __m128 v = _mm_loadu_ps(src);
      if (alpha != 1.0f) v = _mm_mul_ps(v, scale);
      _mm_storeu_ps(dst + static_cast<std::size_t>(p) * kMr, v);
    }
    return;
  }
  for (int p = 0; p < kc; ++p) {
    float* out = dst + static_cast<std::size_t>(p) * kMr;
    for (int r = 0; r < mr; ++r) {
      const float v = g.a_trans ? g.A[static_cast<std::size_t>(p0 + p) * g.lda + i0 + r]
                                : g.A[static_cast<std::size_t>(i0 + r) * g.lda + p0 + p];
      out[r] = alpha * v;
    }
    for (int r = mr; r < kMr; ++r) out[r] = 0.0f;
  }
}

/// 8x8 AVX2 yazmac-ici matris devrigi (b_trans = true iken [8 sutun x 8 k] -> [8 k x 8 sutun]).
inline void transpose8x8_store(const __m256 r0, const __m256 r1, const __m256 r2, const __m256 r3,
                               const __m256 r4, const __m256 r5, const __m256 r6, const __m256 r7,
                               float* out_base, int out_stride) noexcept {
  const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
  const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
  const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
  const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
  const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
  const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
  const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
  const __m256 t7 = _mm256_unpackhi_ps(r6, r7);

  const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
  const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
  const __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
  const __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

  _mm256_storeu_ps(out_base + 0 * out_stride, _mm256_permute2f128_ps(s0, s4, 0x20));
  _mm256_storeu_ps(out_base + 1 * out_stride, _mm256_permute2f128_ps(s1, s5, 0x20));
  _mm256_storeu_ps(out_base + 2 * out_stride, _mm256_permute2f128_ps(s2, s6, 0x20));
  _mm256_storeu_ps(out_base + 3 * out_stride, _mm256_permute2f128_ps(s3, s7, 0x20));
  _mm256_storeu_ps(out_base + 4 * out_stride, _mm256_permute2f128_ps(s0, s4, 0x31));
  _mm256_storeu_ps(out_base + 5 * out_stride, _mm256_permute2f128_ps(s1, s5, 0x31));
  _mm256_storeu_ps(out_base + 6 * out_stride, _mm256_permute2f128_ps(s2, s6, 0x31));
  _mm256_storeu_ps(out_base + 7 * out_stride, _mm256_permute2f128_ps(s3, s7, 0x31));
}

/// Tek bir 16-sutunluk B panelini (j0 .. j0 + nr) [kc x 16] olarak paketler.
void pack_b_panel_16(float* panel, const GemmArgs& g, int p0, int kc, int j0, int nr) {
  if (!g.b_trans && nr == kNr) {
    for (int p = 0; p < kc; ++p) {
      float* out = panel + static_cast<std::size_t>(p) * kNr;
      _mm256_store_ps(out, load_b8(g, p0 + p, j0));
      _mm256_store_ps(out + 8, load_b8(g, p0 + p, j0 + 8));
    }
    return;
  }
  if (g.b_trans && nr == kNr) {
    int p = 0;
    const std::size_t ldb = static_cast<std::size_t>(g.ldb);
    for (; p + 8 <= kc; p += 8) {
      const std::size_t k_off = static_cast<std::size_t>(p0 + p);
      float* out = panel + static_cast<std::size_t>(p) * kNr;
      const std::size_t r0 = static_cast<std::size_t>(j0) * ldb + k_off;
      transpose8x8_store(load_vec8(g.B, g.btype, r0 + 0 * ldb), load_vec8(g.B, g.btype, r0 + 1 * ldb),
                         load_vec8(g.B, g.btype, r0 + 2 * ldb), load_vec8(g.B, g.btype, r0 + 3 * ldb),
                         load_vec8(g.B, g.btype, r0 + 4 * ldb), load_vec8(g.B, g.btype, r0 + 5 * ldb),
                         load_vec8(g.B, g.btype, r0 + 6 * ldb), load_vec8(g.B, g.btype, r0 + 7 * ldb),
                         out, kNr);
      const std::size_t r8 = static_cast<std::size_t>(j0 + 8) * ldb + k_off;
      transpose8x8_store(load_vec8(g.B, g.btype, r8 + 0 * ldb), load_vec8(g.B, g.btype, r8 + 1 * ldb),
                         load_vec8(g.B, g.btype, r8 + 2 * ldb), load_vec8(g.B, g.btype, r8 + 3 * ldb),
                         load_vec8(g.B, g.btype, r8 + 4 * ldb), load_vec8(g.B, g.btype, r8 + 5 * ldb),
                         load_vec8(g.B, g.btype, r8 + 6 * ldb), load_vec8(g.B, g.btype, r8 + 7 * ldb),
                         out + 8, kNr);
    }
    for (; p < kc; ++p) {
      float* out = panel + static_cast<std::size_t>(p) * kNr;
      for (int c = 0; c < kNr; ++c) out[c] = load_b_scalar(g, p0 + p, j0 + c);
    }
    return;
  }
  for (int p = 0; p < kc; ++p) {
    float* out = panel + static_cast<std::size_t>(p) * kNr;
    for (int c = 0; c < nr; ++c) out[c] = load_b_scalar(g, p0 + p, j0 + c);
    for (int c = nr; c < kNr; ++c) out[c] = 0.0f;
  }
}

void compute_submatrix(const GemmArgs& g, const float* shared_b, int p0, int kc, float panel_beta,
                       int m0, int m1, int n0, int n1) {
  if (g.N < 512) {
    t_panel_a.ensure(static_cast<std::size_t>(kMr) * static_cast<std::size_t>(kc));
    float* packed_a = t_panel_a.data;
    for (int i0 = m0; i0 < m1; i0 += kMr) {
      const int mr = std::min(kMr, m1 - i0);
      pack_a_4(packed_a, g, i0, mr, p0, kc, g.alpha);
      for (int j0 = n0; j0 < n1; j0 += kNr) {
        const int nr = std::min(kNr, n1 - j0);
        const float* pb = shared_b + static_cast<std::size_t>(j0 / kNr) * static_cast<std::size_t>(kc) * kNr;
        float* tile = g.C + static_cast<std::size_t>(i0) * g.ldc + j0;
        if (mr == kMr && nr == kNr) {
          call_kernel_4x16(kc, packed_a, pb, tile, g.ldc, panel_beta);
        } else {
          kernel_edge(mr, nr, kc, packed_a, pb, tile, g.ldc, panel_beta);
        }
      }
    }
    return;
  }
  t_panel_a.ensure(static_cast<std::size_t>(kMc) * static_cast<std::size_t>(kc));
  float* packed = t_panel_a.data;
  for (int block = m0; block < m1; block += kMc) {
    const int rows = std::min(kMc, m1 - block);
    const int groups = (rows + kMr - 1) / kMr;
    for (int group = 0; group < groups; ++group) {
      const int i0 = block + group * kMr;
      const int mr = std::min(kMr, block + rows - i0);
      pack_a_4(packed + static_cast<std::size_t>(group) * static_cast<std::size_t>(kc) * kMr, g, i0, mr, p0, kc, g.alpha);
    }
    for (int j0 = n0; j0 < n1; j0 += kNr) {
      const int nr = std::min(kNr, n1 - j0);
      const float* pb = shared_b + static_cast<std::size_t>(j0 / kNr) * static_cast<std::size_t>(kc) * kNr;
      for (int group = 0; group < groups; ++group) {
        const int i0 = block + group * kMr;
        const int mr = std::min(kMr, block + rows - i0);
        const float* pa = packed + static_cast<std::size_t>(group) * static_cast<std::size_t>(kc) * kMr;
        float* tile = g.C + static_cast<std::size_t>(i0) * g.ldc + j0;
        if (mr == kMr && nr == kNr) {
          call_kernel_4x16(kc, pa, pb, tile, g.ldc, panel_beta);
        } else {
          kernel_edge(mr, nr, kc, pa, pb, tile, g.ldc, panel_beta);
        }
      }
    }
  }
}

void apply_bias_submatrix(const GemmArgs& g, int m0, int m1, int n0, int n1) {
  if (!g.bias) return;
  for (int m = m0; m < m1; ++m) {
    float* c = g.C + static_cast<std::size_t>(m) * g.ldc;
    int n = n0;
    for (; n + 8 <= n1; n += 8) {
      _mm256_storeu_ps(c + n, _mm256_add_ps(_mm256_loadu_ps(c + n), _mm256_loadu_ps(g.bias + n)));
    }
    for (; n < n1; ++n) c[n] += g.bias[n];
  }
}

// ==========================================================================================
// GEMV (M <= 4)
// ==========================================================================================

/// M <= 4, B [K, N] (b_trans = false): N sutunlari 16'lik (64-bayt onbellek satiri) seritler halinde
/// iscilere bolunur; her 16 sutunluk seritte M x 2 YMM biriktirici tum k = 0..K-1 boyunca yazmaclarda
/// tutulur (ara tampon yok, k = 0..K-1 sirali FMA -> is parcacigi sayisindan bagimsiz bit-bit ayni).
void gemv_kn_cols(const GemmArgs& g, int n0, int n1) {
  const __m256 va = _mm256_set1_ps(g.alpha);
  const __m256 vb = _mm256_set1_ps(g.beta);
  int n = n0;
  if (g.M == 1) {
    const float* a = g.A;
    for (; n + 32 <= n1; n += 32) {
      __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
      for (int k = 0; k < g.K; ++k) {
        if (k + 4 < g.K) {
          const std::size_t pf = static_cast<std::size_t>(k + 4) * g.ldb + n;
          if (g.btype == BType::F32) {
            _mm_prefetch(reinterpret_cast<const char*>(static_cast<const float*>(g.B) + pf), _MM_HINT_T0);
          } else {
            _mm_prefetch(reinterpret_cast<const char*>(static_cast<const std::uint16_t*>(g.B) + pf), _MM_HINT_T0);
          }
        }
        const __m256 av = _mm256_broadcast_ss(a + k);
        c0 = _mm256_fmadd_ps(av, load_b8(g, k, n + 0), c0);
        c1 = _mm256_fmadd_ps(av, load_b8(g, k, n + 8), c1);
        c2 = _mm256_fmadd_ps(av, load_b8(g, k, n + 16), c2);
        c3 = _mm256_fmadd_ps(av, load_b8(g, k, n + 24), c3);
      }
      float* crow = g.C + n;
      __m256 r0 = _mm256_mul_ps(va, c0), r1 = _mm256_mul_ps(va, c1);
      __m256 r2 = _mm256_mul_ps(va, c2), r3 = _mm256_mul_ps(va, c3);
      if (g.beta != 0.0f) {
        r0 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(crow + 0), r0);
        r1 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(crow + 8), r1);
        r2 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(crow + 16), r2);
        r3 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(crow + 24), r3);
      }
      if (g.bias) {
        r0 = _mm256_add_ps(r0, _mm256_loadu_ps(g.bias + n + 0));
        r1 = _mm256_add_ps(r1, _mm256_loadu_ps(g.bias + n + 8));
        r2 = _mm256_add_ps(r2, _mm256_loadu_ps(g.bias + n + 16));
        r3 = _mm256_add_ps(r3, _mm256_loadu_ps(g.bias + n + 24));
      }
      _mm256_storeu_ps(crow + 0, r0);
      _mm256_storeu_ps(crow + 8, r1);
      _mm256_storeu_ps(crow + 16, r2);
      _mm256_storeu_ps(crow + 24, r3);
    }
  }
  for (; n + 8 <= n1; n += 8) {
    __m256 acc[4];
    for (int m = 0; m < g.M; ++m) acc[m] = _mm256_setzero_ps();
    for (int k = 0; k < g.K; ++k) {
      const __m256 b = load_b8(g, k, n);
      for (int m = 0; m < g.M; ++m) {
        acc[m] = _mm256_fmadd_ps(_mm256_broadcast_ss(g.A + static_cast<std::size_t>(m) * g.lda + k), b, acc[m]);
      }
    }
    for (int m = 0; m < g.M; ++m) {
      float* crow = g.C + static_cast<std::size_t>(m) * g.ldc + n;
      __m256 r = _mm256_mul_ps(va, acc[m]);
      if (g.beta != 0.0f) r = _mm256_fmadd_ps(vb, _mm256_loadu_ps(crow), r);
      if (g.bias) r = _mm256_add_ps(r, _mm256_loadu_ps(g.bias + n));
      _mm256_storeu_ps(crow, r);
    }
  }
  for (; n < n1; ++n) {
    float acc[4] = {};
    for (int k = 0; k < g.K; ++k) {
      const float b = load_b_scalar(g, k, n);
      for (int m = 0; m < g.M; ++m) acc[m] = std::fma(g.A[static_cast<std::size_t>(m) * g.lda + k], b, acc[m]);
    }
    for (int m = 0; m < g.M; ++m) {
      float* c = g.C + static_cast<std::size_t>(m) * g.ldc + n;
      *c = g.alpha * acc[m] + (g.beta != 0.0f ? g.beta * *c : 0.0f) + (g.bias ? g.bias[n] : 0.0f);
    }
  }
}

/// Kucuk N ama buyuk K durumunda K-parcali yardimci.
constexpr int kGemvKC = 256, kGemvNT = 2048;
void gemv_kn_part(const GemmArgs& g, int k0, int k1, int n0, int n1, float* part) {
  const int w = n1 - n0, wv = w / 8 * 8;
  float* acc[4];
  for (int m = 0; m < g.M; ++m) {
    acc[m] = part + static_cast<std::size_t>(m) * g.N + n0;
    std::fill(acc[m], acc[m] + w, 0.0f);
  }
  for (int k = k0; k < k1; ++k) {
    float am[4] = {};
    for (int m = 0; m < g.M; ++m) am[m] = g.A[static_cast<std::size_t>(m) * g.lda + k];
    for (int n = 0; n < wv; n += 8) {
      const __m256 b = load_b8(g, k, n0 + n);
      for (int m = 0; m < g.M; ++m)
        _mm256_storeu_ps(acc[m] + n, _mm256_fmadd_ps(_mm256_set1_ps(am[m]), b, _mm256_loadu_ps(acc[m] + n)));
    }
    for (int n = wv; n < w; ++n) {
      const float b = load_b_scalar(g, k, n0 + n);
      for (int m = 0; m < g.M; ++m) acc[m][n] = std::fma(am[m], b, acc[m][n]);
    }
  }
}

void gemv_kn(const GemmArgs& g, Pool& p) {
  const int W = p.size();
  if (g.N >= W * 32) {
    const int chunks = (g.N + 31) / 32;
    p.run(static_cast<std::size_t>(chunks), [&](int, std::size_t b, std::size_t e) {
      gemv_kn_cols(g, static_cast<int>(b) * 32, std::min(g.N, static_cast<int>(e) * 32));
    });
    return;
  }
  const int kc = (g.K + kGemvKC - 1) / kGemvKC, nt = (g.N + kGemvNT - 1) / kGemvNT;
  thread_local std::vector<float> buf;
  const std::size_t plane = static_cast<std::size_t>(g.M) * g.N;
  if (buf.size() < plane * static_cast<std::size_t>(kc)) buf.resize(plane * static_cast<std::size_t>(kc));
  float* part = buf.data();
  std::atomic<int> next{0};
  const int tasks = kc * nt;
  p.each([&](int) {
    for (int t = next.fetch_add(1); t < tasks; t = next.fetch_add(1)) {
      const int ki = t / nt, ni = t % nt;
      gemv_kn_part(g, ki * kGemvKC, std::min(g.K, (ki + 1) * kGemvKC), ni * kGemvNT, std::min(g.N, (ni + 1) * kGemvNT),
                   part + plane * static_cast<std::size_t>(ki));
    }
  });
  p.run(static_cast<std::size_t>(g.N), [&](int, std::size_t b, std::size_t e) {
    for (int m = 0; m < g.M; ++m) {
      float* c = g.C + static_cast<std::size_t>(m) * g.ldc;
      for (std::size_t n = b; n < e; ++n) {
        float acc = part[static_cast<std::size_t>(m) * g.N + n];
        for (int ki = 1; ki < kc; ++ki) acc += part[plane * static_cast<std::size_t>(ki) + static_cast<std::size_t>(m) * g.N + n];
        c[n] = g.alpha * acc + (g.beta != 0.0f ? g.beta * c[n] : 0.0f) + (g.bias ? g.bias[n] : 0.0f);
      }
    }
  });
}

inline float hsum256(__m256 v) noexcept {
  __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s4 = _mm_hadd_ps(s4, s4);
  s4 = _mm_hadd_ps(s4, s4);
  return _mm_cvtss_f32(s4);
}

/// M <= 4, B [N, K] (b_trans = true): M == 1 iken 4 cikis sutunu ayni anda (A 1 kez okunur, 4 B satiri paralel akar).
void gemv_block(const GemmArgs& g, int n0, int n1) {
  int n = n0;
  if (g.M == 1) {
    const float* a = g.A;
    for (; n + 4 <= n1; n += 4) {
      __m256 c00 = _mm256_setzero_ps(), c01 = c00;
      __m256 c10 = c00, c11 = c00;
      __m256 c20 = c00, c21 = c00;
      __m256 c30 = c00, c31 = c00;
      int k = 0;
      const std::size_t ldb = static_cast<std::size_t>(g.ldb);
      if (g.btype == BType::F32) {
        const float* b0 = static_cast<const float*>(g.B) + static_cast<std::size_t>(n + 0) * ldb;
        const float* b1 = b0 + ldb;
        const float* b2 = b0 + 2 * ldb;
        const float* b3 = b0 + 3 * ldb;
        for (; k + 16 <= g.K; k += 16) {
          const __m256 a0 = _mm256_loadu_ps(a + k);
          const __m256 a1 = _mm256_loadu_ps(a + k + 8);
          c00 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + k), c00);
          c01 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b0 + k + 8), c01);
          c10 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b1 + k), c10);
          c11 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + k + 8), c11);
          c20 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b2 + k), c20);
          c21 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b2 + k + 8), c21);
          c30 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b3 + k), c30);
          c31 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b3 + k + 8), c31);
        }
      } else {
        const std::size_t r0 = static_cast<std::size_t>(n) * ldb;
        for (; k + 16 <= g.K; k += 16) {
          const __m256 a0 = _mm256_loadu_ps(a + k);
          const __m256 a1 = _mm256_loadu_ps(a + k + 8);
          c00 = _mm256_fmadd_ps(a0, load_vec8(g.B, g.btype, r0 + 0 * ldb + k), c00);
          c01 = _mm256_fmadd_ps(a1, load_vec8(g.B, g.btype, r0 + 0 * ldb + k + 8), c01);
          c10 = _mm256_fmadd_ps(a0, load_vec8(g.B, g.btype, r0 + 1 * ldb + k), c10);
          c11 = _mm256_fmadd_ps(a1, load_vec8(g.B, g.btype, r0 + 1 * ldb + k + 8), c11);
          c20 = _mm256_fmadd_ps(a0, load_vec8(g.B, g.btype, r0 + 2 * ldb + k), c20);
          c21 = _mm256_fmadd_ps(a1, load_vec8(g.B, g.btype, r0 + 2 * ldb + k + 8), c21);
          c30 = _mm256_fmadd_ps(a0, load_vec8(g.B, g.btype, r0 + 3 * ldb + k), c30);
          c31 = _mm256_fmadd_ps(a1, load_vec8(g.B, g.btype, r0 + 3 * ldb + k + 8), c31);
        }
        if (k + 8 <= g.K) {
          const __m256 a0 = _mm256_loadu_ps(a + k);
          c00 = _mm256_fmadd_ps(a0, load_vec8(g.B, g.btype, r0 + 0 * ldb + k), c00);
          c10 = _mm256_fmadd_ps(a0, load_vec8(g.B, g.btype, r0 + 1 * ldb + k), c10);
          c20 = _mm256_fmadd_ps(a0, load_vec8(g.B, g.btype, r0 + 2 * ldb + k), c20);
          c30 = _mm256_fmadd_ps(a0, load_vec8(g.B, g.btype, r0 + 3 * ldb + k), c30);
          k += 8;
        }
      }
      float s[4] = {
          hsum256(_mm256_add_ps(c00, c01)),
          hsum256(_mm256_add_ps(c10, c11)),
          hsum256(_mm256_add_ps(c20, c21)),
          hsum256(_mm256_add_ps(c30, c31)),
      };
      for (; k < g.K; ++k) {
        const float ak = a[k];
        for (int j = 0; j < 4; ++j) s[j] += ak * load_b_scalar(g, k, n + j);
      }
      for (int j = 0; j < 4; ++j) {
        float* c = g.C + n + j;
        *c = g.alpha * s[j] + (g.beta != 0.0f ? g.beta * *c : 0.0f) + (g.bias ? g.bias[n + j] : 0.0f);
      }
    }
  }
  for (; n < n1; ++n) {
    for (int m = 0; m < g.M; ++m) {
      const float* a = g.A + static_cast<std::size_t>(m) * g.lda;
      __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;
      int k = 0;
      if (g.btype == BType::F32) {
        const float* b = static_cast<const float*>(g.B) + static_cast<std::size_t>(n) * g.ldb;
        for (; k + 16 <= g.K; k += 16) {
          acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
          acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), acc1);
        }
      } else {
        const std::size_t base = static_cast<std::size_t>(n) * g.ldb;
        for (; k + 8 <= g.K; k += 8) {
          acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), load_vec8(g.B, g.btype, base + k), acc0);
        }
      }
      float acc = hsum256(_mm256_add_ps(acc0, acc1));
      for (; k < g.K; ++k) acc += a[k] * load_b_scalar(g, k, n);
      float* c = g.C + static_cast<std::size_t>(m) * g.ldc + n;
      *c = g.alpha * acc + (g.beta != 0.0f ? g.beta * *c : 0.0f) + (g.bias ? g.bias[n] : 0.0f);
    }
  }
}

}  // namespace

void gemm(const GemmArgs& g, Pool* pool) {
  (void)isa();
  if (g.M < 0 || g.N < 0 || g.K < 0) throw std::invalid_argument("nexus::cpu::gemm: negatif boyut");
  if (g.M == 0 || g.N == 0) return;
  if (reference_gemm() && g.btype == BType::F32 && g.alpha == 1.0f && g.beta == 0.0f && !g.a_trans &&
      g.lda == g.K && g.ldc == g.N) {
    if (!g.b_trans && g.ldb == g.N) {
      nexus_model::kernels::gemm_nn(g.A, static_cast<const float*>(g.B), g.C, g.M, g.K, g.N);
      if (g.bias) {
        for (int m = 0; m < g.M; ++m)
          for (int n = 0; n < g.N; ++n) g.C[static_cast<std::size_t>(m) * g.N + n] += g.bias[n];
      }
      return;
    }
    if (g.b_trans && g.ldb == g.K) {
      std::memset(g.C, 0, static_cast<std::size_t>(g.M) * g.N * sizeof(float));
      nexus_model::kernels::gemm_nt_accum(g.A, static_cast<const float*>(g.B), g.C, g.M, g.K, g.N);
      if (g.bias) {
        for (int m = 0; m < g.M; ++m)
          for (int n = 0; n < g.N; ++n) g.C[static_cast<std::size_t>(m) * g.N + n] += g.bias[n];
      }
      return;
    }
  }
  Pool& p = pool ? *pool : Pool::instance();
  const int W = p.size();
  if (g.K == 0) {
    for (int m = 0; m < g.M; ++m) {
      float* c = g.C + static_cast<std::size_t>(m) * g.ldc;
      for (int n = 0; n < g.N; ++n) c[n] = (g.beta != 0.0f ? g.beta * c[n] : 0.0f) + (g.bias ? g.bias[n] : 0.0f);
    }
    return;
  }
  if (g.M <= 4 && g.a_trans) {
    std::vector<float> at(static_cast<std::size_t>(g.M) * g.K);
    for (int m = 0; m < g.M; ++m)
      for (int k = 0; k < g.K; ++k) at[static_cast<std::size_t>(m) * g.K + k] = g.A[static_cast<std::size_t>(k) * g.lda + m];
    GemmArgs h = g;
    h.A = at.data();
    h.lda = g.K;
    h.a_trans = false;
    gemm(h, pool);
    return;
  }
  if (g.M <= 4 && !g.b_trans) {
    gemv_kn(g, p);
    return;
  }
  if (g.M <= 4) {
    const int chunks = (g.N + 7) / 8;
    p.run(static_cast<std::size_t>(chunks), [&](int, std::size_t b, std::size_t e) {
      gemv_block(g, static_cast<int>(b) * 8, std::min(g.N, static_cast<int>(e) * 8));
    });
    return;
  }

  const int n_panels = (g.N + kNr - 1) / kNr;
  const int kc_max = std::min(kKc, g.K);
  t_panel_b.ensure(static_cast<std::size_t>(n_panels) * static_cast<std::size_t>(kc_max) * kNr);
  float* shared_b = t_panel_b.data;

  const double flops = 2.0 * static_cast<double>(g.M) * static_cast<double>(g.N) * static_cast<double>(g.K);
  const bool parallel = W > 1 && flops >= 2.0e6;

  auto split = [](int total, int parts, int part, int align) {
    const int units = (total + align - 1) / align;
    const int b = units * part / parts, e = units * (part + 1) / parts;
    return std::pair<int, int>{std::min(total, b * align), std::min(total, e * align)};
  };

  // M >= W * 8 iken saf 1B satir bolme (B tamami L3'ten paylasilir, A bloklari L2'de);
  // kucuk M'de (orn. M=16, N=2048) 2B (tm x tn) bolme ile tum cekirdekler doyurulur (B yine tek kez paketlenir!).
  int tm = W, tn = 1;
  if (g.M < W * 8) {
    int best_tm = 1;
    double best = 1e300;
    for (int cand = 1; cand <= W; ++cand) {
      if (W % cand) continue;
      const int c_tn = W / cand;
      const double bm = static_cast<double>(g.M) / cand, bn = static_cast<double>(g.N) / c_tn;
      if (bm < kMr && cand > 1) continue;
      if (bn < kNr && c_tn > 1) continue;
      const double score = std::fabs(std::log(bm / bn));
      if (score < best) {
        best = score;
        best_tm = cand;
      }
    }
    tm = best_tm;
    tn = W / tm;
  }

  for (int p0 = 0; p0 < g.K; p0 += kKc) {
    const int kc = std::min(kKc, g.K - p0);
    const float panel_beta = p0 == 0 ? g.beta : 1.0f;
    if (parallel && n_panels >= W) {
      p.run(static_cast<std::size_t>(n_panels), [&](int, std::size_t pb, std::size_t pe) {
        for (std::size_t pi = pb; pi < pe; ++pi) {
          const int j0 = static_cast<int>(pi) * kNr;
          const int nr = std::min(kNr, g.N - j0);
          pack_b_panel_16(shared_b + pi * static_cast<std::size_t>(kc) * kNr, g, p0, kc, j0, nr);
        }
      });
    } else {
      for (int pi = 0; pi < n_panels; ++pi) {
        const int j0 = pi * kNr;
        const int nr = std::min(kNr, g.N - j0);
        pack_b_panel_16(shared_b + static_cast<std::size_t>(pi) * static_cast<std::size_t>(kc) * kNr, g, p0, kc, j0, nr);
      }
    }
    if (parallel) {
      p.each([&](int w) {
        const int wm = w / tn, wn = w % tn;
        if (wm >= tm) return;
        const auto [m0, m1] = split(g.M, tm, wm, kMr);
        const auto [n0, n1] = split(g.N, tn, wn, kNr);
        if (m0 < m1 && n0 < n1) compute_submatrix(g, shared_b, p0, kc, panel_beta, m0, m1, n0, n1);
      });
    } else {
      compute_submatrix(g, shared_b, p0, kc, panel_beta, 0, g.M, 0, g.N);
    }
  }

  if (g.bias) {
    if (parallel) {
      p.each([&](int w) {
        const int wm = w / tn, wn = w % tn;
        if (wm >= tm) return;
        const auto [m0, m1] = split(g.M, tm, wm, kMr);
        const auto [n0, n1] = split(g.N, tn, wn, kNr);
        if (m0 < m1 && n0 < n1) apply_bias_submatrix(g, m0, m1, n0, n1);
      });
    } else {
      apply_bias_submatrix(g, 0, g.M, 0, g.N);
    }
  }
}

namespace {
std::atomic<bool> g_reference{false};
}  // namespace

void set_reference_gemm(bool on) noexcept { g_reference.store(on, std::memory_order_relaxed); }
bool reference_gemm() noexcept { return g_reference.load(std::memory_order_relaxed); }

void gemm_nn(const float* a, const float* b, float* c, int m, int k, int n) {
  if (reference_gemm()) {
    nexus_model::kernels::gemm_nn(a, b, c, m, k, n);
    return;
  }
  GemmArgs g;
  g.M = m;
  g.N = n;
  g.K = k;
  g.A = a;
  g.lda = k;
  g.B = b;
  g.ldb = n;
  g.C = c;
  g.ldc = n;
  gemm(g);
}

void gemm_tn_accum(const float* a, const float* b, float* c, int k, int m, int n) {
  if (reference_gemm()) {
    nexus_model::kernels::gemm_tn_accum(a, b, c, k, m, n);
    return;
  }
  GemmArgs g;
  g.M = k;
  g.N = n;
  g.K = m;
  g.A = a;
  g.lda = k;
  g.a_trans = true;
  g.B = b;
  g.ldb = n;
  g.C = c;
  g.ldc = n;
  g.beta = 1.0f;
  gemm(g);
}

void gemm_nt_accum(const float* a, const float* b, float* c, int m, int n, int k) {
  if (reference_gemm()) {
    nexus_model::kernels::gemm_nt_accum(a, b, c, m, n, k);
    return;
  }
  GemmArgs g;
  g.M = m;
  g.N = k;
  g.K = n;
  g.A = a;
  g.lda = n;
  g.B = b;
  g.ldb = n;
  g.b_trans = true;
  g.C = c;
  g.ldc = k;
  g.beta = 1.0f;
  gemm(g);
}

std::vector<std::uint16_t> to_bf16(const float* x, std::size_t n) {
  std::vector<std::uint16_t> o(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t u;
    std::memcpy(&u, &x[i], 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
      o[i] = static_cast<std::uint16_t>((u >> 16) | 0x0040u);
    } else {
      o[i] = static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
  }
  return o;
}

std::vector<std::uint16_t> to_f16(const float* x, std::size_t n) {
  std::vector<std::uint16_t> o(n);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm_storeu_si128(reinterpret_cast<__m128i*>(o.data() + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT));
  for (; i < n; ++i)
    o[i] = static_cast<std::uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x[i]), _MM_FROUND_TO_NEAREST_INT), 0));
  return o;
}

// ==========================================================================================
// INT8 (PackedS8)
// ==========================================================================================

PackedS8 pack_s8(const float* W, int N, int K, bool b_trans) {
  PackedS8 p;
  p.N = N;
  p.K = K;
  p.Kp = (K + 3) / 4 * 4;
  const int panels = (N + 15) / 16;
  p.data.assign(static_cast<std::size_t>(panels) * p.Kp * 16, 0);
  p.scale.assign(static_cast<std::size_t>(N), 1.0f);
  auto w = [&](int n, int k) { return b_trans ? W[static_cast<std::size_t>(n) * K + k] : W[static_cast<std::size_t>(k) * N + n]; };
  Pool::instance().run(static_cast<std::size_t>(N), [&](int, std::size_t nb, std::size_t ne) {
    for (std::size_t ni = nb; ni < ne; ++ni) {
      const int n = static_cast<int>(ni);
      float amax = 0.0f;
      for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(w(n, k)));
      const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
      const float inv = 1.0f / s;
      p.scale[ni] = s;
      const int panel = n / 16, c = n % 16;
      for (int k = 0; k < K; ++k) {
        const float q = std::nearbyint(w(n, k) * inv);
        p.data[(static_cast<std::size_t>(panel) * (p.Kp / 4) + k / 4) * 64 + static_cast<std::size_t>(c) * 4 + k % 4] =
            static_cast<std::int8_t>(std::max(-127.0f, std::min(127.0f, q)));
      }
    }
  });
  return p;
}

void quantize_rows_s8(const float* x, int M, int K, int lda, std::int8_t* q, int Kp, float* scale, Pool* pool) {
  Pool& p = pool ? *pool : Pool::instance();
  auto body = [&](std::size_t b, std::size_t e) {
    for (std::size_t m = b; m < e; ++m) {
      const float* r = x + m * static_cast<std::size_t>(lda);
      __m256 vmax = _mm256_setzero_ps();
      const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
      int k = 0;
      for (; k + 8 <= K; k += 8) vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(r + k), sign_mask));
      alignas(32) float m8[8];
      _mm256_store_ps(m8, vmax);
      float amax = std::max({m8[0], m8[1], m8[2], m8[3], m8[4], m8[5], m8[6], m8[7]});
      for (; k < K; ++k) amax = std::max(amax, std::fabs(r[k]));
      const float s = amax > 0.0f ? amax / 127.0f : 1.0f, inv = 1.0f / s;
      scale[m] = s;
      std::int8_t* o = q + m * static_cast<std::size_t>(Kp);
      const __m256 vinv = _mm256_set1_ps(inv);
      const __m256 vlo = _mm256_set1_ps(-127.0f), vhi = _mm256_set1_ps(127.0f);
      k = 0;
      for (; k + 8 <= K; k += 8) {
        __m256 v = _mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(r + k), vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v = _mm256_max_ps(vlo, _mm256_min_ps(vhi, v));
        const __m256i vi = _mm256_cvtps_epi32(v);
        alignas(32) std::int32_t i8[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(i8), vi);
        for (int j = 0; j < 8; ++j) o[k + j] = static_cast<std::int8_t>(i8[j]);
      }
      for (; k < K; ++k) o[k] = static_cast<std::int8_t>(std::max(-127.0f, std::min(127.0f, std::nearbyint(r[k] * inv))));
      for (; k < Kp; ++k) o[k] = 0;
    }
  };
  if (M <= 4) {
    body(0, static_cast<std::size_t>(M));
  } else {
    p.run(static_cast<std::size_t>(M), [&](int, std::size_t b, std::size_t e) { body(b, e); });
  }
}

namespace {

inline void micro_s8_4x16(const std::int8_t* const* arow, int rows, const std::int8_t* Bp, int Kp, __m256i acc[4][2]) {
  const __m256i ones = _mm256_set1_epi16(1);
  for (int r = 0; r < 4; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_si256();
  const int k4_end = Kp / 4;
  if (rows == 1) {
    const auto* a32 = reinterpret_cast<const std::int32_t*>(arow[0]);
    __m256i c0 = _mm256_setzero_si256(), c1 = c0;
    int k4 = 0;
    for (; k4 + 4 <= k4_end; k4 += 4) {
      _mm_prefetch(reinterpret_cast<const char*>(Bp + (k4 + 8) * 64), _MM_HINT_T0);
      for (int u = 0; u < 4; ++u) {
        const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + (k4 + u) * 64));
        const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + (k4 + u) * 64 + 32));
        const __m256i a = _mm256_set1_epi32(a32[k4 + u]);
        const __m256i ua = _mm256_abs_epi8(a);
        c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(_mm256_maddubs_epi16(ua, _mm256_sign_epi8(b0, a)), ones));
        c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(_mm256_maddubs_epi16(ua, _mm256_sign_epi8(b1, a)), ones));
      }
    }
    for (; k4 < k4_end; ++k4) {
      const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + k4 * 64));
      const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + k4 * 64 + 32));
      const __m256i a = _mm256_set1_epi32(a32[k4]);
      const __m256i ua = _mm256_abs_epi8(a);
      c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(_mm256_maddubs_epi16(ua, _mm256_sign_epi8(b0, a)), ones));
      c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(_mm256_maddubs_epi16(ua, _mm256_sign_epi8(b1, a)), ones));
    }
    acc[0][0] = c0;
    acc[0][1] = c1;
    return;
  }
  for (int k4 = 0; k4 < k4_end; ++k4) {
    const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + k4 * 64));
    const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + k4 * 64 + 32));
    for (int r = 0; r < rows; ++r) {
      std::int32_t a4;
      std::memcpy(&a4, arow[r] + k4 * 4, 4);
      const __m256i a = _mm256_set1_epi32(a4);
      const __m256i ua = _mm256_abs_epi8(a);
      const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ua, _mm256_sign_epi8(b0, a)), ones);
      const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ua, _mm256_sign_epi8(b1, a)), ones);
      acc[r][0] = _mm256_add_epi32(acc[r][0], p0);
      acc[r][1] = _mm256_add_epi32(acc[r][1], p1);
    }
  }
}

}  // namespace

void gemm_s8(const std::int8_t* Aq, const float* sa, int M, const PackedS8& B, float* C, int ldc, const float* bias, Pool* pool) {
  (void)isa();
  Pool& p = pool ? *pool : Pool::instance();
  const int panels = (B.N + 15) / 16;
  const int Kp = B.Kp;
  p.run(static_cast<std::size_t>(panels), [&](int, std::size_t pb, std::size_t pe) {
    for (std::size_t pi = pb; pi < pe; ++pi) {
      const std::int8_t* Bp = B.data.data() + pi * static_cast<std::size_t>(Kp) * 16;
      const int n0 = static_cast<int>(pi) * 16, nn = std::min(16, B.N - n0);
      for (int m = 0; m < M; m += 4) {
        const int rows = std::min(4, M - m);
        const std::int8_t* ar[4];
        for (int r = 0; r < 4; ++r) ar[r] = Aq + static_cast<std::size_t>(m + std::min(r, rows - 1)) * Kp;
        __m256i acc[4][2];
        micro_s8_4x16(ar, rows, Bp, Kp, acc);
        for (int r = 0; r < rows; ++r) {
          float* c = C + static_cast<std::size_t>(m + r) * ldc + n0;
          const float s = sa[m + r];
          if (nn == 16) {
            const __m256 vs = _mm256_set1_ps(s);
            __m256 v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(acc[r][0]), _mm256_mul_ps(vs, _mm256_loadu_ps(B.scale.data() + n0)));
            __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(acc[r][1]), _mm256_mul_ps(vs, _mm256_loadu_ps(B.scale.data() + n0 + 8)));
            if (bias) {
              v0 = _mm256_add_ps(v0, _mm256_loadu_ps(bias + n0));
              v1 = _mm256_add_ps(v1, _mm256_loadu_ps(bias + n0 + 8));
            }
            _mm256_storeu_ps(c, v0);
            _mm256_storeu_ps(c + 8, v1);
          } else {
            alignas(32) std::int32_t t[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(t), acc[r][0]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(t + 8), acc[r][1]);
            for (int j = 0; j < nn; ++j)
              c[j] = static_cast<float>(t[j]) * s * B.scale[static_cast<std::size_t>(n0 + j)] + (bias ? bias[n0 + j] : 0.0f);
          }
        }
      }
    }
  });
}

// ==========================================================================================
// INT4 (PackedQ4 — 4-bit simetrik nibble-seritli AVX2 maddubs)
// ==========================================================================================

PackedQ4 pack_q4(const float* W, int N, int K, bool b_trans) {
  PackedQ4 p;
  p.N = N;
  p.K = K;
  p.Kp = (K + 7) / 8 * 8;
  const int panels = (N + 15) / 16;
  p.data.assign(static_cast<std::size_t>(panels) * (p.Kp / 8) * 64, 0x88u);  // 0 -> (0+8) | ((0+8)<<4) = 0x88
  p.scale.assign(static_cast<std::size_t>(N), 1.0f);
  auto w = [&](int n, int k) { return b_trans ? W[static_cast<std::size_t>(n) * K + k] : W[static_cast<std::size_t>(k) * N + n]; };
  Pool::instance().run(static_cast<std::size_t>(N), [&](int, std::size_t nb, std::size_t ne) {
    static constexpr float kClipGrid[8] = {0.68f, 0.73f, 0.78f, 0.82f, 0.86f, 0.90f, 0.95f, 1.00f};
    for (std::size_t ni = nb; ni < ne; ++ni) {
      const int n = static_cast<int>(ni);
      float amax = 0.0f;
      for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(w(n, k)));
      float best_s = amax > 0.0f ? amax / 7.0f : 1.0f;
      if (amax > 0.0f && K >= 32) {
        double best_err = 1e300;
        for (float clip : kClipGrid) {
          const float cand_s = (amax * clip) / 7.0f;
          const float cand_inv = 1.0f / cand_s;
          double err = 0.0;
          for (int k = 0; k < K; ++k) {
            const float wk = w(n, k);
            const float q = std::max(-7.0f, std::min(7.0f, std::nearbyint(wk * cand_inv)));
            const double d = static_cast<double>(wk - q * cand_s);
            err += d * d;
          }
          if (err < best_err) {
            best_err = err;
            best_s = cand_s;
          }
        }
      }
      const float s = best_s;
      const float inv = 1.0f / s;
      p.scale[ni] = s;
      const int panel = n / 16, c = n % 16;
      for (int k8 = 0; k8 < p.Kp / 8; ++k8) {
        std::uint8_t* dst = p.data.data() + (static_cast<std::size_t>(panel) * (p.Kp / 8) + k8) * 64 + c * 4;
        for (int j = 0; j < 4; ++j) {
          const int k_lo = k8 * 8 + j;
          const int k_hi = k8 * 8 + 4 + j;
          const int q_lo = k_lo < K ? static_cast<int>(std::max(-7.0f, std::min(7.0f, std::nearbyint(w(n, k_lo) * inv)))) : 0;
          const int q_hi = k_hi < K ? static_cast<int>(std::max(-7.0f, std::min(7.0f, std::nearbyint(w(n, k_hi) * inv)))) : 0;
          dst[j] = static_cast<std::uint8_t>((q_lo + 8) | ((q_hi + 8) << 4));
        }
      }
    }
  });
  return p;
}

void gemm_q4(const std::int8_t* Aq, const float* sa, int M, const PackedQ4& B, float* C, int ldc, const float* bias, Pool* pool) {
  (void)isa();
  Pool& p = pool ? *pool : Pool::instance();
  const int panels = (B.N + 15) / 16;
  const int Kp = B.Kp;
  const int k8_end = Kp / 8;
  p.run(static_cast<std::size_t>(panels), [&](int, std::size_t pb, std::size_t pe) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i off_8 = _mm256_set1_epi8(8);
    for (std::size_t pi = pb; pi < pe; ++pi) {
      const std::uint8_t* Bp = B.data.data() + pi * static_cast<std::size_t>(k8_end) * 64;
      const int n0 = static_cast<int>(pi) * 16, nn = std::min(16, B.N - n0);
      for (int m = 0; m < M; m += 4) {
        const int rows = std::min(4, M - m);
        const std::int8_t* ar[4];
        for (int r = 0; r < 4; ++r) ar[r] = Aq + static_cast<std::size_t>(m + std::min(r, rows - 1)) * Kp;
        __m256i acc[4][2];
        for (int r = 0; r < 4; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_si256();
        for (int k8 = 0; k8 < k8_end; ++k8) {
          const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + k8 * 64));
          const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + k8 * 64 + 32));
          const __m256i b0_lo = _mm256_sub_epi8(_mm256_and_si256(p0, mask_0f), off_8);
          const __m256i b1_lo = _mm256_sub_epi8(_mm256_and_si256(p1, mask_0f), off_8);
          const __m256i b0_hi = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(p0, 4), mask_0f), off_8);
          const __m256i b1_hi = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(p1, 4), mask_0f), off_8);
          for (int r = 0; r < rows; ++r) {
            std::int32_t a_lo, a_hi;
            std::memcpy(&a_lo, ar[r] + k8 * 8, 4);
            std::memcpy(&a_hi, ar[r] + k8 * 8 + 4, 4);
            const __m256i alo = _mm256_set1_epi32(a_lo), ualo = _mm256_abs_epi8(alo);
            const __m256i ahi = _mm256_set1_epi32(a_hi), uahi = _mm256_abs_epi8(ahi);
            acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_madd_epi16(_mm256_maddubs_epi16(ualo, _mm256_sign_epi8(b0_lo, alo)), ones));
            acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_madd_epi16(_mm256_maddubs_epi16(ualo, _mm256_sign_epi8(b1_lo, alo)), ones));
            acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_madd_epi16(_mm256_maddubs_epi16(uahi, _mm256_sign_epi8(b0_hi, ahi)), ones));
            acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_madd_epi16(_mm256_maddubs_epi16(uahi, _mm256_sign_epi8(b1_hi, ahi)), ones));
          }
        }
        for (int r = 0; r < rows; ++r) {
          float* c = C + static_cast<std::size_t>(m + r) * ldc + n0;
          const float s = sa[m + r];
          if (nn == 16) {
            const __m256 vs = _mm256_set1_ps(s);
            __m256 v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(acc[r][0]), _mm256_mul_ps(vs, _mm256_loadu_ps(B.scale.data() + n0)));
            __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(acc[r][1]), _mm256_mul_ps(vs, _mm256_loadu_ps(B.scale.data() + n0 + 8)));
            if (bias) {
              v0 = _mm256_add_ps(v0, _mm256_loadu_ps(bias + n0));
              v1 = _mm256_add_ps(v1, _mm256_loadu_ps(bias + n0 + 8));
            }
            _mm256_storeu_ps(c, v0);
            _mm256_storeu_ps(c + 8, v1);
          } else {
            alignas(32) std::int32_t t[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(t), acc[r][0]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(t + 8), acc[r][1]);
            for (int j = 0; j < nn; ++j)
              c[j] = static_cast<float>(t[j]) * s * B.scale[static_cast<std::size_t>(n0 + j)] + (bias ? bias[n0 + j] : 0.0f);
          }
        }
      }
    }
  });
}

// ==========================================================================================
// AVX-512 BF16 (vdpbf16ps) / AMX BF16 (tdpbf16ps 16x16 karo) VNNI ciftli duzen yolu
// ==========================================================================================

PackedBf16Vnni pack_bf16_vnni(const float* W, int N, int K, bool b_trans) {
  PackedBf16Vnni p;
  p.N = N;
  p.K = K;
  p.Kp = (K + 1) / 2 * 2;
  const int panels = (N + 15) / 16;
  p.data.assign(static_cast<std::size_t>(panels) * (p.Kp / 2) * 32, 0);
  auto w = [&](int n, int k) { return b_trans ? W[static_cast<std::size_t>(n) * K + k] : W[static_cast<std::size_t>(k) * N + n]; };
  for (int n = 0; n < N; ++n) {
    const int panel = n / 16, c = n % 16;
    for (int k = 0; k < K; ++k) {
      std::uint32_t u;
      const float v = w(n, k);
      std::memcpy(&u, &v, 4);
      const std::uint16_t bf = static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
      p.data[(static_cast<std::size_t>(panel) * (p.Kp / 2) + k / 2) * 32 + c * 2 + k % 2] = bf;
    }
  }
  return p;
}

void gemm_bf16_vnni(const float* A, int M, int lda, const PackedBf16Vnni& B, float* C, int ldc, const float* bias, Pool* pool) {
  (void)isa();
  Pool& p = pool ? *pool : Pool::instance();
  const int panels = (B.N + 15) / 16;
  const int Kp = B.Kp;
  // A satirlarini BF16 ciftlerine yuvarla (vdpbf16ps / tdpbf16ps giris semantigi)
  std::vector<std::uint16_t> aq(static_cast<std::size_t>(M) * Kp, 0);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < B.K; ++k) {
      std::uint32_t u;
      std::memcpy(&u, &A[static_cast<std::size_t>(m) * lda + k], 4);
      aq[static_cast<std::size_t>(m) * Kp + k] = static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
  }
  p.run(static_cast<std::size_t>(panels), [&](int, std::size_t pb, std::size_t pe) {
    const __m256i mask_lo = _mm256_set1_epi32(0x0000FFFF);
    for (std::size_t pi = pb; pi < pe; ++pi) {
      const std::uint16_t* Bp = B.data.data() + pi * static_cast<std::size_t>(Kp / 2) * 32;
      const int n0 = static_cast<int>(pi) * 16, nn = std::min(16, B.N - n0);
      for (int m = 0; m < M; ++m) {
        const std::uint16_t* ar = aq.data() + static_cast<std::size_t>(m) * Kp;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;
        for (int k2 = 0; k2 < Kp / 2; ++k2) {
          std::uint32_t pair_a;
          std::memcpy(&pair_a, ar + k2 * 2, 4);
          const __m256 a_even = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>((pair_a & 0xFFFFu) << 16)));
          const __m256 a_odd = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(pair_a & 0xFFFF0000u)));
          const __m256i bp0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + k2 * 32));
          const __m256i bp1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + k2 * 32 + 16));
          const __m256 b0_even = _mm256_castsi256_ps(_mm256_slli_epi32(bp0, 16));
          const __m256 b0_odd = _mm256_castsi256_ps(_mm256_andnot_si256(mask_lo, bp0));
          const __m256 b1_even = _mm256_castsi256_ps(_mm256_slli_epi32(bp1, 16));
          const __m256 b1_odd = _mm256_castsi256_ps(_mm256_andnot_si256(mask_lo, bp1));
          // vdpbf16ps(acc, a, b): acc += a_even * b_even + a_odd * b_odd
          const __m256 prod0 = _mm256_fmadd_ps(a_even, b0_even, _mm256_mul_ps(a_odd, b0_odd));
          const __m256 prod1 = _mm256_fmadd_ps(a_even, b1_even, _mm256_mul_ps(a_odd, b1_odd));
          acc0 = _mm256_add_ps(acc0, prod0);
          acc1 = _mm256_add_ps(acc1, prod1);
        }
        float* c = C + static_cast<std::size_t>(m) * ldc + n0;
        if (nn == 16) {
          if (bias) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(bias + n0));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(bias + n0 + 8));
          }
          _mm256_storeu_ps(c, acc0);
          _mm256_storeu_ps(c + 8, acc1);
        } else {
          alignas(32) float tmp[16];
          _mm256_store_ps(tmp, acc0);
          _mm256_store_ps(tmp + 8, acc1);
          for (int j = 0; j < nn; ++j) c[j] = tmp[j] + (bias ? bias[n0 + j] : 0.0f);
        }
      }
    }
  });
}

}  // namespace cpu
}  // namespace nexus
