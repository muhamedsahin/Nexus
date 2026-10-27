/**
 * @file adam_avx2.cpp
 * @brief FAZ 18.E: ZeRO-Offload CPU AdamW, AVX2 (FMA bilerek kullanilmaz: skaler yolla bit-bit esitlik).
 *
 * Kuyruk (n % 8) ayni vektor govdesiyle 8'lik yerel tamponda islenir: derleyicinin skaler kuyrukta a*b+c'yi
 * FMA'ya daraltma riski yok.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cpu/adam.hpp"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstring>

namespace nexus {
namespace cpu {

namespace {

inline __m256 load_grad(const void* g, GradType t, std::size_t i) noexcept {
  if (t == GradType::F32) return _mm256_loadu_ps(static_cast<const float*>(g) + i);
  const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const std::uint16_t*>(g) + i));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

inline __m128i bf16_rne(__m256 x) noexcept {
  const __m256i f = _mm256_castps_si256(x);
  const __m256i hi = _mm256_srli_epi32(f, 16);
  const __m256i rnd = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), _mm256_and_si256(hi, _mm256_set1_epi32(1)));
  const __m256i r = _mm256_srli_epi32(_mm256_add_epi32(f, rnd), 16);
  const __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(f, _mm256_set1_epi32(0x7FFFFFFF)), _mm256_set1_epi32(0x7F800000));
  const __m256i q = _mm256_or_si256(hi, _mm256_set1_epi32(0x0040));
  const __m256i v = _mm256_blendv_epi8(r, q, nan);
  return _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
}

struct V {
  __m256 b1, ob1, b2, ob2, inv1, inv2, lr, lr_wd, eps, clip;
};

// fused::adamw_kernel ile ayni sira:
//   g = grad*clip; p -= lr_wd*p; mt = b1*m + ob1*g; vt = b2*v + (ob2*g)*g; p = p - (lr*(mt*inv1)) / (sqrt(vt*inv2)+eps)
template <bool kDecay>
inline void step8(float* p, float* m, float* v, __m256 g, const V& c, std::uint16_t* out) noexcept {
  __m256 pp = _mm256_loadu_ps(p);
  g = _mm256_mul_ps(g, c.clip);
  if constexpr (kDecay) pp = _mm256_sub_ps(pp, _mm256_mul_ps(c.lr_wd, pp));
  const __m256 mt = _mm256_add_ps(_mm256_mul_ps(c.b1, _mm256_loadu_ps(m)), _mm256_mul_ps(c.ob1, g));
  const __m256 vt = _mm256_add_ps(_mm256_mul_ps(c.b2, _mm256_loadu_ps(v)), _mm256_mul_ps(_mm256_mul_ps(c.ob2, g), g));
  _mm256_storeu_ps(m, mt);
  _mm256_storeu_ps(v, vt);
  const __m256 num = _mm256_mul_ps(c.lr, _mm256_mul_ps(mt, c.inv1));
  const __m256 den = _mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(vt, c.inv2)), c.eps);
  pp = _mm256_sub_ps(pp, _mm256_div_ps(num, den));
  _mm256_storeu_ps(p, pp);
  if (out != nullptr) _mm_storeu_si128(reinterpret_cast<__m128i*>(out), bf16_rne(pp));
}

template <bool kDecay>
void run(float* p, float* m, float* v, const void* grad, GradType t, const V& c, std::uint16_t* out, std::size_t b,
         std::size_t e) noexcept {
  std::size_t i = b;
  for (; i + 8 <= e; i += 8) {
    _mm_prefetch(reinterpret_cast<const char*>(p + i + 64), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(m + i + 64), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(v + i + 64), _MM_HINT_T0);
    step8<kDecay>(p + i, m + i, v + i, load_grad(grad, t, i), c, out != nullptr ? out + i : nullptr);
  }
  const std::size_t r = e - i;
  if (r == 0) return;
  alignas(32) float tp[8] = {}, tm[8] = {}, tv[8] = {}, tg[8] = {};
  alignas(16) std::uint16_t to[8] = {};
  std::memcpy(tp, p + i, r * sizeof(float));
  std::memcpy(tm, m + i, r * sizeof(float));
  std::memcpy(tv, v + i, r * sizeof(float));
  if (t == GradType::F32) {
    std::memcpy(tg, static_cast<const float*>(grad) + i, r * sizeof(float));
  } else {
    alignas(16) std::uint16_t hb[8] = {};
    std::memcpy(hb, static_cast<const std::uint16_t*>(grad) + i, r * sizeof(std::uint16_t));
    _mm256_store_ps(tg, load_grad(hb, GradType::BF16, 0));
  }
  step8<kDecay>(tp, tm, tv, _mm256_load_ps(tg), c, out != nullptr ? to : nullptr);
  std::memcpy(p + i, tp, r * sizeof(float));
  std::memcpy(m + i, tm, r * sizeof(float));
  std::memcpy(v + i, tv, r * sizeof(float));
  if (out != nullptr) std::memcpy(out + i, to, r * sizeof(std::uint16_t));
}

}  // namespace

AdamCoeffs adam_coeffs(double lr, double beta1, double beta2, double eps, double weight_decay, std::int64_t step) {
  const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step));
  const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step));
  AdamCoeffs c{};
  c.inv1 = static_cast<float>(1.0 / bc1);
  c.inv2 = static_cast<float>(1.0 / bc2);
  c.b1 = static_cast<float>(beta1);
  c.ob1 = static_cast<float>(1.0 - beta1);
  c.b2 = static_cast<float>(beta2);
  c.ob2 = static_cast<float>(1.0 - beta2);
  c.lr = static_cast<float>(lr);
  c.lr_wd = c.lr * static_cast<float>(weight_decay);
  c.eps = static_cast<float>(eps);
  return c;
}

double sq_norm_range(const void* grad, GradType t, std::size_t b, std::size_t e) noexcept {
  __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
  std::size_t i = b;
  for (; i + 8 <= e; i += 8) {
    const __m256 g = load_grad(grad, t, i);
    const __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(g));
    const __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(g, 1));
    a0 = _mm256_add_pd(a0, _mm256_mul_pd(lo, lo));
    a1 = _mm256_add_pd(a1, _mm256_mul_pd(hi, hi));
  }
  alignas(32) double l[8];
  _mm256_store_pd(l, a0);
  _mm256_store_pd(l + 4, a1);
  double s = ((l[0] + l[4]) + (l[1] + l[5])) + ((l[2] + l[6]) + (l[3] + l[7]));
  for (; i < e; ++i) {
    const double g = t == GradType::F32 ? static_cast<const float*>(grad)[i]
                                        : static_cast<double>(std::bit_cast<float>(
                                              static_cast<std::uint32_t>(static_cast<const std::uint16_t*>(grad)[i]) << 16));
    s += g * g;
  }
  return s;
}

void adamw_range(float* p, float* m, float* v, const void* grad, GradType t, float clip, const AdamCoeffs& c, bool decay,
                 std::uint16_t* out, std::size_t b, std::size_t e) noexcept {
  const V vc{_mm256_set1_ps(c.b1),   _mm256_set1_ps(c.ob1), _mm256_set1_ps(c.b2),    _mm256_set1_ps(c.ob2),
             _mm256_set1_ps(c.inv1), _mm256_set1_ps(c.inv2), _mm256_set1_ps(c.lr),   _mm256_set1_ps(c.lr_wd),
             _mm256_set1_ps(c.eps),  _mm256_set1_ps(clip)};
  // out indeksleri p ile ayni (mutlak); run() out + i kullanir.
  if (decay) {  // cagiran: decay = (param decay) && weight_decay != 0 (fused ile ayni dal)
    run<true>(p, m, v, grad, t, vc, out, b, e);
  } else {
    run<false>(p, m, v, grad, t, vc, out, b, e);
  }
}

void to_bf16_range(const float* x, std::uint16_t* out, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), bf16_rne(_mm256_loadu_ps(x + i)));
  if (i < n) {
    alignas(32) float t[8] = {};
    alignas(16) std::uint16_t o[8];
    std::memcpy(t, x + i, (n - i) * sizeof(float));
    _mm_store_si128(reinterpret_cast<__m128i*>(o), bf16_rne(_mm256_load_ps(t)));
    std::memcpy(out + i, o, (n - i) * sizeof(std::uint16_t));
  }
}

}  // namespace cpu
}  // namespace nexus
