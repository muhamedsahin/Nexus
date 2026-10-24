/**
 * @file cpu.cpp
 * @brief FAZ 5.A math dikkat (FP64 kahin) ve 5.G CPU FlashAttention (FP32, online softmax).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/attention/cpu.hpp"

#include "nexus/core/error.hpp"
#include "nexus/fused/training.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nexus {

// ---- AttentionParams --------------------------------------------------------------------
void AttentionParams::set_contiguous_bthd() {
  q_row_stride = static_cast<long long>(heads) * head_dim;
  q_head_stride = head_dim;
  q_batch_stride = static_cast<long long>(tq) * q_row_stride;
  k_row_stride = static_cast<long long>(kv_heads) * head_dim;
  k_head_stride = head_dim;
  k_batch_stride = static_cast<long long>(tk) * k_row_stride;
  o_row_stride = q_row_stride;
  o_head_stride = q_head_stride;
  o_batch_stride = q_batch_stride;
}

float AttentionParams::effective_scale() const noexcept {
  return scale != 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
}

void AttentionParams::validate() const {
  if (batch < 1 || heads < 1 || kv_heads < 1 || tq < 1 || tk < 1 || head_dim < 1) {
    throw TensorError("Nexus attention: batch, heads, kv_heads, tq, tk, head_dim pozitif olmali");
  }
  if (heads % kv_heads != 0) {
    throw TensorError("Nexus attention: heads, kv_heads'in kati olmali (GQA)");
  }
  if (window < 0 || softcap < 0.0f || !(dropout_p >= 0.0f && dropout_p < 1.0f)) {
    throw TensorError("Nexus attention: window >= 0, softcap >= 0, dropout_p [0,1) olmali");
  }
  if ((cu_seqlens_q == nullptr) != (cu_seqlens_k == nullptr)) {
    throw TensorError("Nexus attention: cu_seqlens_q ve cu_seqlens_k birlikte verilmeli");
  }
  if (q_row_stride < head_dim || k_row_stride < head_dim || o_row_stride < head_dim) {
    throw TensorError("Nexus attention: satir adimlari ayarlanmamis (set_contiguous_bthd?)");
  }
}

namespace attention {
namespace {

struct Geometry {
  int b, tq, tk;
  long long q_row0, k_row0;  // varlen: paketli satir; degilse batch ofseti eleman cinsinden
  bool varlen;
  int total_q;
};

Geometry geometry(const AttentionParams& p, int b) {
  Geometry g{};
  g.b = b;
  g.varlen = p.cu_seqlens_q != nullptr;
  if (g.varlen) {
    g.tq = p.cu_seqlens_q[b + 1] - p.cu_seqlens_q[b];
    g.tk = p.cu_seqlens_k[b + 1] - p.cu_seqlens_k[b];
    g.q_row0 = p.cu_seqlens_q[b];
    g.k_row0 = p.cu_seqlens_k[b];
    g.total_q = p.cu_seqlens_q[p.batch];
    if (g.tq > p.tq || g.tk > p.tk) {
      throw TensorError("Nexus attention: varlen dizisi tq/tk azami boyunu asiyor");
    }
  } else {
    g.tq = p.tq;
    g.tk = p.tk;
    g.total_q = p.batch * p.tq;
  }
  return g;
}

const float* q_row(const float* base, const AttentionParams& p, const Geometry& g, int h, int i) {
  return base + (g.varlen ? (g.q_row0 + i) * p.q_row_stride : g.b * p.q_batch_stride + i * p.q_row_stride) +
         h * p.q_head_stride;
}
const float* k_row(const float* base, const AttentionParams& p, const Geometry& g, int hk, int j) {
  return base + (g.varlen ? (g.k_row0 + j) * p.k_row_stride : g.b * p.k_batch_stride + j * p.k_row_stride) +
         hk * p.k_head_stride;
}
float* o_row(float* base, const AttentionParams& p, const Geometry& g, int h, int i) {
  return base + (g.varlen ? (g.q_row0 + i) * p.o_row_stride : g.b * p.o_batch_stride + i * p.o_row_stride) +
         h * p.o_head_stride;
}
std::size_t lse_index(const AttentionParams& p, const Geometry& g, int h, int i) {
  return g.varlen ? static_cast<std::size_t>(h) * g.total_q + static_cast<std::size_t>(g.q_row0 + i)
                  : (static_cast<std::size_t>(g.b) * p.heads + h) * p.tq + i;
}
std::uint64_t dropout_index(const AttentionParams& p, const Geometry& g, int h, int i, int j) {
  if (g.varlen) {
    return (static_cast<std::uint64_t>(h) * g.total_q + static_cast<std::uint64_t>(g.q_row0 + i)) * p.tk + j;
  }
  return ((static_cast<std::uint64_t>(g.b) * p.heads + h) * p.tq + i) * p.tk + j;
}

bool allowed(const AttentionParams& p, int qpos, int j) {
  if (p.causal && j > qpos) {
    return false;
  }
  if (p.window > 0) {
    if (qpos - j >= p.window) {
      return false;
    }
    if (!p.causal && j - qpos >= p.window) {
      return false;
    }
  }
  return true;
}

// Ham skor -> (softcap, alibi) uygulanmis skor; turev icin tanh degeri dondurulur.
double transform(const AttentionParams& p, double raw, int h, int qpos, int j, double* softcap_t) {
  double s = raw;
  *softcap_t = 0.0;
  if (p.softcap > 0.0f) {
    const double t = std::tanh(s / p.softcap);
    *softcap_t = t;
    s = p.softcap * t;
  }
  if (p.alibi_slopes != nullptr) {
    s += static_cast<double>(p.alibi_slopes[h]) * (j - qpos);
  }
  return s;
}

void check_cpu(const AttentionParams& p) {
  p.validate();
  if (p.dtype != DType::F32) {
    throw TensorError("Nexus CPU attention: yalniz F32 tamponlar (CPU yolu FP32)");
  }
}

void parallel_for(int count, int threads, const std::function<void(int)>& fn) {
  if (threads <= 0) {
    threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  }
  threads = std::min(threads, count);
  if (threads <= 1) {
    for (int i = 0; i < count; ++i) {
      fn(i);
    }
    return;
  }
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      for (int i = next.fetch_add(1); i < count; i = next.fetch_add(1)) {
        fn(i);
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }
}

constexpr int kBlock = 64;

}  // namespace

// ================================ math (FP64) ==========================================
void math_attention_forward(float* o, float* lse, const float* q, const float* k, const float* v,
                            const AttentionParams& p) {
  check_cpu(p);
  const int D = p.head_dim;
  const int group = p.heads / p.kv_heads;
  const double scale = p.effective_scale();
  const double keep_scale = 1.0 / (1.0 - p.dropout_p);
  std::vector<double> s(static_cast<std::size_t>(p.tk));
  std::vector<double> acc(static_cast<std::size_t>(D));
  for (int b = 0; b < p.batch; ++b) {
    const Geometry g = geometry(p, b);
    for (int h = 0; h < p.heads; ++h) {
      const int hk = h / group;
      for (int i = 0; i < g.tq; ++i) {
        const int qpos = g.tk - g.tq + i;
        const float* qi = q_row(q, p, g, h, i);
        double m = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < g.tk; ++j) {
          if (!allowed(p, qpos, j)) {
            s[static_cast<std::size_t>(j)] = -std::numeric_limits<double>::infinity();
            continue;
          }
          const float* kj = k_row(k, p, g, hk, j);
          double dot = 0.0;
          for (int d = 0; d < D; ++d) {
            dot += static_cast<double>(qi[d]) * kj[d];
          }
          double t = 0.0;
          s[static_cast<std::size_t>(j)] = transform(p, dot * scale, h, qpos, j, &t);
          m = std::max(m, s[static_cast<std::size_t>(j)]);
        }
        float* oi = o_row(o, p, g, h, i);
        if (!std::isfinite(m)) {  // tum anahtarlar maskeli: O = 0, LSE = -inf
          std::fill(oi, oi + D, 0.0f);
          lse[lse_index(p, g, h, i)] = -std::numeric_limits<float>::infinity();
          continue;
        }
        double l = 0.0;
        for (int j = 0; j < g.tk; ++j) {
          if (std::isfinite(s[static_cast<std::size_t>(j)])) {
            l += std::exp(s[static_cast<std::size_t>(j)] - m);
          }
        }
        const double row_lse = m + std::log(l);
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int j = 0; j < g.tk; ++j) {
          if (!std::isfinite(s[static_cast<std::size_t>(j)])) {
            continue;
          }
          double pij = std::exp(s[static_cast<std::size_t>(j)] - row_lse);
          if (p.dropout_p > 0.0f) {
            const bool drop = fused::philox_uniform(p.seed, p.offset, dropout_index(p, g, h, i, j)) < p.dropout_p;
            pij = drop ? 0.0 : pij * keep_scale;
          }
          const float* vj = k_row(v, p, g, hk, j);
          for (int d = 0; d < D; ++d) {
            acc[static_cast<std::size_t>(d)] += pij * vj[d];
          }
        }
        for (int d = 0; d < D; ++d) {
          oi[d] = static_cast<float>(acc[static_cast<std::size_t>(d)]);
        }
        lse[lse_index(p, g, h, i)] = static_cast<float>(row_lse);
      }
    }
  }
}

void math_attention_backward(float* dq, float* dk, float* dv, const float* dout, const float* q, const float* k,
                             const float* v, const float* o, const float* lse, const AttentionParams& p) {
  check_cpu(p);
  const int D = p.head_dim;
  const int group = p.heads / p.kv_heads;
  const double scale = p.effective_scale();
  const double keep_scale = 1.0 / (1.0 - p.dropout_p);
  // Cikti tamponlarini sifirla (adimli duzende yalniz kendi satirlari).
  for (int b = 0; b < p.batch; ++b) {
    const Geometry g = geometry(p, b);
    for (int h = 0; h < p.heads; ++h) {
      for (int i = 0; i < g.tq; ++i) {
        float* r = const_cast<float*>(q_row(dq, p, g, h, i));
        std::fill(r, r + D, 0.0f);
      }
    }
    for (int hk = 0; hk < p.kv_heads; ++hk) {
      for (int j = 0; j < g.tk; ++j) {
        float* rk = const_cast<float*>(k_row(dk, p, g, hk, j));
        float* rv = const_cast<float*>(k_row(dv, p, g, hk, j));
        std::fill(rk, rk + D, 0.0f);
        std::fill(rv, rv + D, 0.0f);
      }
    }
  }
  std::vector<double> dqi(static_cast<std::size_t>(D));
  for (int b = 0; b < p.batch; ++b) {
    const Geometry g = geometry(p, b);
    for (int h = 0; h < p.heads; ++h) {
      const int hk = h / group;
      for (int i = 0; i < g.tq; ++i) {
        const int qpos = g.tk - g.tq + i;
        const double row_lse = lse[lse_index(p, g, h, i)];
        if (!std::isfinite(row_lse)) {
          continue;
        }
        const float* qi = q_row(q, p, g, h, i);
        const float* doi = q_row(dout, p, g, h, i);  // dO, O ile ayni adimlarda (o_* = q_* varsayimi)
        const float* oi = q_row(o, p, g, h, i);
        double Di = 0.0;
        for (int d = 0; d < D; ++d) {
          Di += static_cast<double>(doi[d]) * oi[d];
        }
        std::fill(dqi.begin(), dqi.end(), 0.0);
        for (int j = 0; j < g.tk; ++j) {
          if (!allowed(p, qpos, j)) {
            continue;
          }
          const float* kj = k_row(k, p, g, hk, j);
          const float* vj = k_row(v, p, g, hk, j);
          double dot = 0.0;
          for (int d = 0; d < D; ++d) {
            dot += static_cast<double>(qi[d]) * kj[d];
          }
          double t = 0.0;
          const double sij = transform(p, dot * scale, h, qpos, j, &t);
          const double pij = std::exp(sij - row_lse);
          double z = 1.0;
          if (p.dropout_p > 0.0f) {
            z = fused::philox_uniform(p.seed, p.offset, dropout_index(p, g, h, i, j)) < p.dropout_p ? 0.0 : keep_scale;
          }
          double dpd = 0.0;
          for (int d = 0; d < D; ++d) {
            dpd += static_cast<double>(doi[d]) * vj[d];
          }
          const double dp = dpd * z;
          double ds = pij * (dp - Di);
          if (p.softcap > 0.0f) {
            ds *= 1.0 - t * t;
          }
          ds *= scale;
          float* dvj = const_cast<float*>(k_row(dv, p, g, hk, j));
          float* dkj = const_cast<float*>(k_row(dk, p, g, hk, j));
          for (int d = 0; d < D; ++d) {
            dvj[d] += static_cast<float>(pij * z * doi[d]);
            dkj[d] += static_cast<float>(ds * qi[d]);
            dqi[static_cast<std::size_t>(d)] += ds * kj[d];
          }
        }
        float* dqr = const_cast<float*>(q_row(dq, p, g, h, i));
        for (int d = 0; d < D; ++d) {
          dqr[d] = static_cast<float>(dqi[static_cast<std::size_t>(d)]);
        }
      }
    }
  }
}

// ================================ CPU flash (FP32) ======================================
// Doseme duzeni: Br x Bc = 64 x 64. K (ve V) dosemesi [D][64] olarak paketlenir; boylece
// skor ve dP icin ic dongu 64 bitisik eleman uzerinde (vektorlesir), PV / dK / dV / dQ icin ic
// dongu D uzerindedir. Maskeli elemanlar -inf skor tasir: exp(-inf - m) = 0, dal yok.
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// nc satiri [D][kBlock] olarak paketler; c >= nc sutunlari sifir.
void pack_t(float* bt, const float* base, const AttentionParams& p, const Geometry& g, int hk, int j0, int nc, int D) {
  for (int c = 0; c < nc; ++c) {
    const float* row = k_row(base, p, g, hk, j0 + c);
    for (int d = 0; d < D; ++d) {
      bt[static_cast<std::size_t>(d) * kBlock + c] = row[d];
    }
  }
  for (int d = 0; d < D; ++d) {
    std::fill(bt + static_cast<std::size_t>(d) * kBlock + nc, bt + static_cast<std::size_t>(d + 1) * kBlock, 0.0f);
  }
}

#if defined(__AVX2__)
// Cephes expf, 8 serit (hata <= 2 ULP). x < -87.34 (-inf dahil) tam 0 verir: maskeli skor = 0.
inline __m256 exp_avx2(__m256 x) {
  const __m256 zero_mask = _mm256_cmp_ps(x, _mm256_set1_ps(-87.33654f), _CMP_LT_OQ);
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.33654f)), _mm256_set1_ps(88.37626f));
  __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
  const __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_andnot_ps(zero_mask, _mm256_mul_ps(y, _mm256_castsi256_ps(n)));
}
#endif

// x[c] = exp(x[c] - m), c < kBlock.
void exp_shift(float* x, float m) {
#if defined(__AVX2__)
  const __m256 mv = _mm256_set1_ps(m);
  for (int c = 0; c < kBlock; c += 8) {
    _mm256_storeu_ps(x + c, exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + c), mv)));
  }
#else
  for (int c = 0; c < kBlock; ++c) {
    x[c] = std::exp(x[c] - m);
  }
#endif
}

// out[0..D) += sum_k coef[k * stride] * rows[k][0..D); sifir katsayilar atlanir.
void accum_rows(float* out, const float* coef, int stride, const float* const* rows, int n, int D) {
#if defined(__AVX2__)
  if (D % 64 == 0) {
    for (int d0 = 0; d0 < D; d0 += 64) {
      __m256 acc[8];
      for (int u = 0; u < 8; ++u) {
        acc[u] = _mm256_loadu_ps(out + d0 + 8 * u);
      }
      for (int kk = 0; kk < n; ++kk) {
        const float cf = coef[static_cast<std::size_t>(kk) * stride];
        if (cf == 0.0f) {
          continue;
        }
        const __m256 cv = _mm256_set1_ps(cf);
        const float* row = rows[kk] + d0;
        for (int u = 0; u < 8; ++u) {
          acc[u] = _mm256_fmadd_ps(cv, _mm256_loadu_ps(row + 8 * u), acc[u]);
        }
      }
      for (int u = 0; u < 8; ++u) {
        _mm256_storeu_ps(out + d0 + 8 * u, acc[u]);
      }
    }
    return;
  }
#endif
  for (int kk = 0; kk < n; ++kk) {
    const float cf = coef[static_cast<std::size_t>(kk) * stride];
    if (cf == 0.0f) {
      continue;
    }
    const float* row = rows[kk];
    for (int d = 0; d < D; ++d) {
      out[d] += cf * row[d];
    }
  }
}

// out[r][c] = a_r . bt[:, c]
void dot_tile(float* out, const float* const* arows, int nr, const float* bt, int D) {
#if defined(__AVX2__)
  for (int r = 0; r < nr; ++r) {
    __m256 acc[kBlock / 8];
    for (int u = 0; u < kBlock / 8; ++u) {
      acc[u] = _mm256_setzero_ps();
    }
    const float* a = arows[r];
    for (int d = 0; d < D; ++d) {
      const __m256 av = _mm256_set1_ps(a[d]);
      const float* b = bt + static_cast<std::size_t>(d) * kBlock;
      for (int u = 0; u < kBlock / 8; ++u) {
        acc[u] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + 8 * u), acc[u]);
      }
    }
    for (int u = 0; u < kBlock / 8; ++u) {
      _mm256_storeu_ps(out + static_cast<std::size_t>(r) * kBlock + 8 * u, acc[u]);
    }
  }
#else
  for (int r = 0; r < nr; ++r) {
    float* o = out + static_cast<std::size_t>(r) * kBlock;
    std::fill(o, o + kBlock, 0.0f);
    const float* a = arows[r];
    for (int d = 0; d < D; ++d) {
      const float ad = a[d];
      const float* b = bt + static_cast<std::size_t>(d) * kBlock;
      for (int c = 0; c < kBlock; ++c) {
        o[c] += ad * b[c];
      }
    }
  }
#endif
}

// Ham nokta carpimlarini skora cevirir (olcek, maske, softcap, alibi). tanh degeri t'ye.
void score_tile(float* s, float* t, const AttentionParams& p, const Geometry& g, int h, int i0, int nr, int j0, int nc,
                float scale) {
  const bool cap = p.softcap > 0.0f;
  const float slope = p.alibi_slopes != nullptr ? p.alibi_slopes[h] : 0.0f;
  for (int r = 0; r < nr; ++r) {
    const int qpos = g.tk - g.tq + i0 + r;
    float* sr = s + static_cast<std::size_t>(r) * kBlock;
    float* tr = t + static_cast<std::size_t>(r) * kBlock;
    for (int c = 0; c < kBlock; ++c) {
      const int j = j0 + c;
      if (c >= nc || !allowed(p, qpos, j)) {
        sr[c] = kNegInf;
        continue;
      }
      float v = sr[c] * scale;
      if (cap) {
        const float th = std::tanh(v / p.softcap);
        tr[c] = th;
        v = p.softcap * th;
      }
      sr[c] = v + slope * static_cast<float>(j - qpos);
    }
  }
}

// Sorgu satirlari [i0, i0+nr) icin gorunur anahtar araligi (koruyucu; eleman maskesi kesindir).
void key_span(const AttentionParams& p, const Geometry& g, int i0, int nr, int* lo, int* hi) {
  const int off = g.tk - g.tq;
  const int q_first = i0 + off;
  const int q_last = i0 + nr - 1 + off;
  *lo = 0;
  *hi = g.tk;
  if (p.causal) {
    *hi = std::min(g.tk, q_last + 1);
  } else if (p.window > 0) {
    *hi = std::min(g.tk, q_last + p.window);
  }
  if (p.window > 0) {
    *lo = std::max(0, q_first - p.window + 1);
  }
  *hi = std::max(*hi, *lo);
}

// Anahtar sutunlari [j0, j0+nc) icin gorunur sorgu araligi (koruyucu).
void query_span(const AttentionParams& p, const Geometry& g, int j0, int nc, int* lo, int* hi) {
  const int off = g.tk - g.tq;
  *lo = 0;
  *hi = g.tq;
  if (p.causal) {
    *lo = std::max(0, j0 - off);
  } else if (p.window > 0) {
    *lo = std::max(0, j0 - p.window + 1 - off);
  }
  if (p.window > 0) {
    *hi = std::min(g.tq, j0 + nc - 1 + p.window - off);
  }
  *hi = std::max(*hi, *lo);
}

}  // namespace

void flash_attention_forward_cpu(float* o, float* lse, const float* q, const float* k, const float* v,
                                 const AttentionParams& p, int threads) {
  check_cpu(p);
  const int D = p.head_dim;
  const int group = p.heads / p.kv_heads;
  const float scale = p.effective_scale();
  const float keep_scale = 1.0f / (1.0f - p.dropout_p);
  const int qblocks = (p.tq + kBlock - 1) / kBlock;
  parallel_for(p.batch * p.heads * qblocks, threads, [&](int task) {
    const int qb = task % qblocks;
    const int h = (task / qblocks) % p.heads;
    const int b = task / (qblocks * p.heads);
    const Geometry g = geometry(p, b);
    const int i0 = qb * kBlock;
    if (i0 >= g.tq) {
      return;
    }
    const int nr = std::min(kBlock, g.tq - i0);
    const int hk = h / group;
    std::vector<float> kt(static_cast<std::size_t>(D) * kBlock), s(kBlock * kBlock), t(kBlock * kBlock);
    std::vector<float> acc(static_cast<std::size_t>(nr) * D, 0.0f), m(nr, kNegInf), l(nr, 0.0f);
    const float* qrows[kBlock];
    for (int r = 0; r < nr; ++r) {
      qrows[r] = q_row(q, p, g, h, i0 + r);
    }
    int j_lo = 0;
    int j_hi = 0;
    key_span(p, g, i0, nr, &j_lo, &j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += kBlock) {
      const int nc = std::min(kBlock, j_hi - j0);
      pack_t(kt.data(), k, p, g, hk, j0, nc, D);
      const float* vrows[kBlock];
      for (int c = 0; c < nc; ++c) {
        vrows[c] = k_row(v, p, g, hk, j0 + c);
      }
      dot_tile(s.data(), qrows, nr, kt.data(), D);
      score_tile(s.data(), t.data(), p, g, h, i0, nr, j0, nc, scale);
      for (int r = 0; r < nr; ++r) {
        float* sr = s.data() + static_cast<std::size_t>(r) * kBlock;
        float mx = kNegInf;
        for (int c = 0; c < kBlock; ++c) {
          mx = std::max(mx, sr[c]);
        }
        if (mx == kNegInf) {
          continue;
        }
        const float m_new = std::max(m[r], mx);
        const float alpha = std::exp(m[r] - m_new);  // m = -inf -> 0
        exp_shift(sr, m_new);
        float sum = 0.0f;
        for (int c = 0; c < kBlock; ++c) {
          sum += sr[c];
        }
        l[r] = l[r] * alpha + sum;  // LSE dropout oncesi olasiliklarla
        m[r] = m_new;
        if (p.dropout_p > 0.0f) {
          for (int c = 0; c < nc; ++c) {
            sr[c] = fused::philox_uniform(p.seed, p.offset, dropout_index(p, g, h, i0 + r, j0 + c)) < p.dropout_p
                        ? 0.0f
                        : sr[c] * keep_scale;
          }
        }
        float* ar = acc.data() + static_cast<std::size_t>(r) * D;
        for (int d = 0; d < D; ++d) {
          ar[d] *= alpha;
        }
        accum_rows(ar, sr, 1, vrows, nc, D);
      }
    }
    for (int r = 0; r < nr; ++r) {
      float* oi = o_row(o, p, g, h, i0 + r);
      if (!(l[r] > 0.0f)) {
        std::fill(oi, oi + D, 0.0f);
        lse[lse_index(p, g, h, i0 + r)] = kNegInf;
        continue;
      }
      const float inv = 1.0f / l[r];
      const float* ar = acc.data() + static_cast<std::size_t>(r) * D;
      for (int d = 0; d < D; ++d) {
        oi[d] = ar[d] * inv;
      }
      lse[lse_index(p, g, h, i0 + r)] = m[r] + std::log(l[r]);
    }
  });
}

void flash_attention_backward_cpu(float* dq, float* dk, float* dv, const float* dout, const float* q, const float* k,
                                  const float* v, const float* o, const float* lse, const AttentionParams& p,
                                  int threads) {
  check_cpu(p);
  const int D = p.head_dim;
  const int group = p.heads / p.kv_heads;
  const float scale = p.effective_scale();
  const float keep_scale = 1.0f / (1.0f - p.dropout_p);
  // D_i = rowsum(dO o O), [lse duzeninde].
  const int total_rows = p.cu_seqlens_q != nullptr ? p.cu_seqlens_q[p.batch] : p.batch * p.tq;
  std::vector<float> Dvec(static_cast<std::size_t>(total_rows) * p.heads, 0.0f);
  for (int b = 0; b < p.batch; ++b) {
    const Geometry g = geometry(p, b);
    for (int h = 0; h < p.heads; ++h) {
      for (int i = 0; i < g.tq; ++i) {
        const float* doi = q_row(dout, p, g, h, i);
        const float* oi = q_row(o, p, g, h, i);
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
          acc += doi[d] * oi[d];
        }
        Dvec[lse_index(p, g, h, i)] = acc;
      }
    }
  }
  // Bir (sorgu dosemesi, anahtar dosemesi) icin P*Z (pz) ve olcekli dS (ds) [r][c].
  struct Scratch {
    std::vector<float> kt, vt, s, t, dp;
    explicit Scratch(int D)
        : kt(static_cast<std::size_t>(D) * kBlock), vt(kt.size()), s(kBlock * kBlock), t(kBlock * kBlock),
          dp(kBlock * kBlock) {}
  };
  // kt/vt paketli olmali; cikis: s <- pz, dp <- ds.
  auto tile = [&](Scratch& w, const Geometry& g, int h, int i0, int nr, int j0, int nc, const float* const* qrows,
                  const float* const* dorows) {
    dot_tile(w.s.data(), qrows, nr, w.kt.data(), D);
    score_tile(w.s.data(), w.t.data(), p, g, h, i0, nr, j0, nc, scale);
    dot_tile(w.dp.data(), dorows, nr, w.vt.data(), D);
    for (int r = 0; r < nr; ++r) {
      const std::size_t row = lse_index(p, g, h, i0 + r);
      const float row_lse = lse[row];
      const float di = Dvec[row];
      float* pr = w.s.data() + static_cast<std::size_t>(r) * kBlock;
      float* dr = w.dp.data() + static_cast<std::size_t>(r) * kBlock;
      const float* tr = w.t.data() + static_cast<std::size_t>(r) * kBlock;
      if (!std::isfinite(row_lse)) {
        std::fill(pr, pr + kBlock, 0.0f);
        std::fill(dr, dr + kBlock, 0.0f);
        continue;
      }
      exp_shift(pr, row_lse);  // maskeli: 0
      if (p.dropout_p > 0.0f) {
        for (int c = 0; c < nc; ++c) {
          const float z = fused::philox_uniform(p.seed, p.offset, dropout_index(p, g, h, i0 + r, j0 + c)) < p.dropout_p
                              ? 0.0f
                              : keep_scale;
          dr[c] = pr[c] * (dr[c] * z - di);
          pr[c] *= z;
        }
      } else {
        for (int c = 0; c < kBlock; ++c) {
          dr[c] = pr[c] * (dr[c] - di);
        }
      }
      if (p.softcap > 0.0f) {
        for (int c = 0; c < nc; ++c) {
          if (dr[c] != 0.0f) {
            dr[c] *= 1.0f - tr[c] * tr[c];  // maskeli eleman: dr = 0, t tanimsiz
          }
        }
      }
      for (int c = 0; c < kBlock; ++c) {
        dr[c] *= scale;
      }
    }
  };
  const int kblocks = (p.tk + kBlock - 1) / kBlock;
  // dK, dV: (batch, kv-head, k-blok); grup icindeki tum q-baslari ve sorgu dosemeleri sirayla.
  parallel_for(p.batch * p.kv_heads * kblocks, threads, [&](int task) {
    const int kb = task % kblocks;
    const int hk = (task / kblocks) % p.kv_heads;
    const int b = task / (kblocks * p.kv_heads);
    const Geometry g = geometry(p, b);
    const int j0 = kb * kBlock;
    if (j0 >= g.tk) {
      return;
    }
    const int nc = std::min(kBlock, g.tk - j0);
    Scratch w(D);
    pack_t(w.kt.data(), k, p, g, hk, j0, nc, D);
    pack_t(w.vt.data(), v, p, g, hk, j0, nc, D);
    std::vector<float> dk_acc(static_cast<std::size_t>(nc) * D, 0.0f), dv_acc(dk_acc.size(), 0.0f);
    int i_lo = 0;
    int i_hi = 0;
    query_span(p, g, j0, nc, &i_lo, &i_hi);
    const float* qrows[kBlock];
    const float* dorows[kBlock];
    for (int h = hk * group; h < (hk + 1) * group; ++h) {
      for (int i0 = i_lo; i0 < i_hi; i0 += kBlock) {
        const int nr = std::min(kBlock, i_hi - i0);
        for (int r = 0; r < nr; ++r) {
          qrows[r] = q_row(q, p, g, h, i0 + r);
          dorows[r] = q_row(dout, p, g, h, i0 + r);
        }
        tile(w, g, h, i0, nr, j0, nc, qrows, dorows);
        for (int c = 0; c < nc; ++c) {
          accum_rows(dv_acc.data() + static_cast<std::size_t>(c) * D, w.s.data() + c, kBlock, dorows, nr, D);
          accum_rows(dk_acc.data() + static_cast<std::size_t>(c) * D, w.dp.data() + c, kBlock, qrows, nr, D);
        }
      }
    }
    for (int c = 0; c < nc; ++c) {
      float* dkj = const_cast<float*>(k_row(dk, p, g, hk, j0 + c));
      float* dvj = const_cast<float*>(k_row(dv, p, g, hk, j0 + c));
      std::copy(dk_acc.begin() + static_cast<std::ptrdiff_t>(c) * D, dk_acc.begin() + static_cast<std::ptrdiff_t>(c + 1) * D,
                dkj);
      std::copy(dv_acc.begin() + static_cast<std::ptrdiff_t>(c) * D, dv_acc.begin() + static_cast<std::ptrdiff_t>(c + 1) * D,
                dvj);
    }
  });
  // dQ: (batch, head, q-blok); P ve dS yeniden hesaplanir (atomik yok).
  const int qblocks = (p.tq + kBlock - 1) / kBlock;
  parallel_for(p.batch * p.heads * qblocks, threads, [&](int task) {
    const int qb = task % qblocks;
    const int h = (task / qblocks) % p.heads;
    const int b = task / (qblocks * p.heads);
    const Geometry g = geometry(p, b);
    const int i0 = qb * kBlock;
    if (i0 >= g.tq) {
      return;
    }
    const int nr = std::min(kBlock, g.tq - i0);
    const int hk = h / group;
    Scratch w(D);
    std::vector<float> dq_acc(static_cast<std::size_t>(nr) * D, 0.0f);
    const float* qrows[kBlock];
    const float* dorows[kBlock];
    for (int r = 0; r < nr; ++r) {
      qrows[r] = q_row(q, p, g, h, i0 + r);
      dorows[r] = q_row(dout, p, g, h, i0 + r);
    }
    int j_lo = 0;
    int j_hi = 0;
    key_span(p, g, i0, nr, &j_lo, &j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += kBlock) {
      const int nc = std::min(kBlock, j_hi - j0);
      pack_t(w.kt.data(), k, p, g, hk, j0, nc, D);
      pack_t(w.vt.data(), v, p, g, hk, j0, nc, D);
      tile(w, g, h, i0, nr, j0, nc, qrows, dorows);
      const float* krows[kBlock];
      for (int c = 0; c < nc; ++c) {
        krows[c] = k_row(k, p, g, hk, j0 + c);
      }
      for (int r = 0; r < nr; ++r) {
        accum_rows(dq_acc.data() + static_cast<std::size_t>(r) * D, w.dp.data() + static_cast<std::size_t>(r) * kBlock,
                   1, krows, nc, D);
      }
    }
    for (int r = 0; r < nr; ++r) {
      float* dqi = const_cast<float*>(q_row(dq, p, g, h, i0 + r));
      std::copy(dq_acc.begin() + static_cast<std::ptrdiff_t>(r) * D, dq_acc.begin() + static_cast<std::ptrdiff_t>(r + 1) * D,
                dqi);
    }
  });
}

}  // namespace attention
}  // namespace nexus
