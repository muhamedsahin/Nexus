/**
 * @file cpu.cpp
 * @brief FAZ 5.A math dikkat (FP64 kahin) ve FAZ 5.G / FAZ 18.4 CPU FlashAttention (FP32 / BF16 / F16).
 *
 * FAZ 18.4 iyilestirmeleri:
 *  - `thread_local` cizik tamponlari: blok basina sifir yigin/heap ayirma (`malloc`/`free` yok).
 *  - `nexus::cpu::Pool` uzerinden sabitli fiziksel cekirdek dagitimi ve atomik is calma
 *    (nedensel maskede dengesiz sorgu bloklarini tum fiziksel cekirdeklere esit yayar).
 *  - AVX2 8x8 yazmac ici devrikleme ile K/V [D][64] paketleme (`pack_t`).
 *  - Birlestirilmis AVX2 `exp_shift_sum` ve satir-maksimum azaltmasi; ilk blokta veya
 *    maksimum degismediginde (`alpha == 1`) gereksiz `acc *= alpha` olceklemesini atlama.
 *  - Geri geciste `batch * kv_heads >= threads` oldugunda `(b, hk)` basina tek geciste
 *    hem dK/dV hem dQ biriktirme (atomiksiz, %100 belirlenimci, 7 yerine 5 karo GEMM'i).
 *  - FP32 yaninda BF16 ve F16 girdi/cikti destegi (yuklemede AVX2/F16C ile FP32'ye genisletme,
 *    FP32 biriktirme, cikista RNE ile BF16/F16'ya daraltma).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/attention/cpu.hpp"

#include "nexus/core/error.hpp"
#include "nexus/cpu/threads.hpp"
#include "nexus/fused/training.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
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

inline long long q_offset(const AttentionParams& p, const Geometry& g, int h, int i) noexcept {
  return (g.varlen ? (g.q_row0 + i) * p.q_row_stride : g.b * p.q_batch_stride + i * p.q_row_stride) +
         h * p.q_head_stride;
}
inline long long k_offset(const AttentionParams& p, const Geometry& g, int hk, int j) noexcept {
  return (g.varlen ? (g.k_row0 + j) * p.k_row_stride : g.b * p.k_batch_stride + j * p.k_row_stride) +
         hk * p.k_head_stride;
}
inline long long o_offset(const AttentionParams& p, const Geometry& g, int h, int i) noexcept {
  return (g.varlen ? (g.q_row0 + i) * p.o_row_stride : g.b * p.o_batch_stride + i * p.o_row_stride) +
         h * p.o_head_stride;
}

inline const float* q_row_f32(const void* base, const AttentionParams& p, const Geometry& g, int h, int i) noexcept {
  return static_cast<const float*>(base) + q_offset(p, g, h, i);
}
inline const float* k_row_f32(const void* base, const AttentionParams& p, const Geometry& g, int hk, int j) noexcept {
  return static_cast<const float*>(base) + k_offset(p, g, hk, j);
}
inline float* o_row_f32(void* base, const AttentionParams& p, const Geometry& g, int h, int i) noexcept {
  return static_cast<float*>(base) + o_offset(p, g, h, i);
}

// ---- BF16 / F16 <-> FP32 donusum yardimcilari -------------------------------------------
inline float bf16_to_f32_scalar(std::uint16_t b) noexcept {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

inline std::uint16_t f32_to_bf16_scalar(float f) noexcept {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(float));
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0u;  // NaN koruma
  const std::uint32_t lsb = (u >> 16) & 1u;
  return static_cast<std::uint16_t>((u + 0x7FFFu + lsb) >> 16);
}

inline float f16_to_f32_scalar(std::uint16_t h) noexcept {
#if defined(__F16C__) || defined(__AVX2__)
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
#else
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x03FFu;
  std::uint32_t u = 0;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;
    } else {
      float m = static_cast<float>(mant) * (1.0f / 16777216.0f);
      return sign ? -m : m;
    }
  } else if (exp == 31) {
    u = sign | 0x7F800000u | (mant << 13);
  } else {
    u = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
#endif
}

inline std::uint16_t f32_to_f16_scalar(float f) noexcept {
#if defined(__F16C__) || defined(__AVX2__)
  return static_cast<std::uint16_t>(
      _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
#else
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(float));
  const std::uint16_t sign = static_cast<std::uint16_t>((u >> 16) & 0x8000u);
  const int exp = static_cast<int>((u >> 23) & 0xFFu) - 127 + 15;
  std::uint32_t mant = u & 0x007FFFFFu;
  if (exp <= 0) return sign;
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
  return static_cast<std::uint16_t>(sign | (exp << 10) | ((mant + 0x1000u) >> 13));
#endif
}

inline void cvt_row_to_f32(float* dst, const void* base, DType dt, long long offset, int D) noexcept {
  if (dt == DType::F32) {
    std::memcpy(dst, static_cast<const float*>(base) + offset, static_cast<std::size_t>(D) * sizeof(float));
    return;
  }
  const std::uint16_t* src = static_cast<const std::uint16_t*>(base) + offset;
  int d = 0;
  if (dt == DType::BF16) {
#if defined(__AVX2__)
    for (; d + 8 <= D; d += 8) {
      const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + d));
      const __m256 f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
      _mm256_storeu_ps(dst + d, f);
    }
#endif
    for (; d < D; ++d) dst[d] = bf16_to_f32_scalar(src[d]);
  } else {
#if defined(__AVX2__)
    for (; d + 8 <= D; d += 8) {
      const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + d));
      _mm256_storeu_ps(dst + d, _mm256_cvtph_ps(h));
    }
#endif
    for (; d < D; ++d) dst[d] = f16_to_f32_scalar(src[d]);
  }
}

inline const float* get_row_f32(float* scratch, const void* base, DType dt, long long offset, int D) noexcept {
  if (dt == DType::F32) return static_cast<const float*>(base) + offset;
  cvt_row_to_f32(scratch, base, dt, offset, D);
  return scratch;
}

inline void store_row_from_f32(void* base, DType dt, long long offset, const float* src, int D) noexcept {
  if (dt == DType::F32) {
    std::memcpy(static_cast<float*>(base) + offset, src, static_cast<std::size_t>(D) * sizeof(float));
    return;
  }
  std::uint16_t* dst = static_cast<std::uint16_t*>(base) + offset;
  int d = 0;
  if (dt == DType::BF16) {
#if defined(__AVX2__)
    const __m256i bias = _mm256_set1_epi32(0x7FFF);
    const __m256i one = _mm256_set1_epi32(1);
    for (; d + 8 <= D; d += 8) {
      const __m256i u = _mm256_castps_si256(_mm256_loadu_ps(src + d));
      const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), one);
      const __m256i r = _mm256_srli_epi32(_mm256_add_epi32(u, _mm256_add_epi32(bias, lsb)), 16);
      const __m128i lo = _mm256_castsi256_si128(r);
      const __m128i hi = _mm256_extracti128_si256(r, 1);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + d), _mm_packus_epi32(lo, hi));
    }
#endif
    for (; d < D; ++d) dst[d] = f32_to_bf16_scalar(src[d]);
  } else {
#if defined(__AVX2__)
    for (; d + 8 <= D; d += 8) {
      const __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(src + d), _MM_FROUND_TO_NEAREST_INT);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + d), h);
    }
#endif
    for (; d < D; ++d) dst[d] = f32_to_f16_scalar(src[d]);
  }
}

inline void zero_row(void* base, DType dt, long long offset, int D) noexcept {
  if (dt == DType::F32) {
    std::memset(static_cast<float*>(base) + offset, 0, static_cast<std::size_t>(D) * sizeof(float));
  } else {
    std::memset(static_cast<std::uint16_t*>(base) + offset, 0, static_cast<std::size_t>(D) * sizeof(std::uint16_t));
  }
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
  if (p.dtype != DType::F32 && p.dtype != DType::BF16 && p.dtype != DType::F16) {
    throw TensorError("Nexus CPU attention: yalniz F32, BF16 ve F16 tamponlar desteklenir");
  }
}

// Yedek is parcacigi havuzu: yalniz kullanici `threads > cpu::Pool::instance().size()` istediginde devreye girer.
class WorkerPool {
 public:
  static WorkerPool& instance() {
    static WorkerPool pool;
    return pool;
  }
  void run(int count, int threads, const std::function<void(int)>& fn) {
    if (inside_) {
      for (int i = 0; i < count; ++i) fn(i);
      return;
    }
    std::lock_guard<std::mutex> job(job_mutex_);
    const int helpers = std::min(threads - 1, static_cast<int>(workers_.size()));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fn_ = &fn;
      count_ = count;
      next_.store(0, std::memory_order_relaxed);
      active_ = helpers;
      pending_ = helpers;
      ++generation_;
    }
    wake_.notify_all();
    drain();
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return pending_ == 0; });
    fn_ = nullptr;
  }

 private:
  WorkerPool() {
    const int n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency())) - 1;
    for (int t = 0; t < n; ++t) {
      workers_.emplace_back([this, t] { loop(t); });
    }
  }
  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    wake_.notify_all();
    for (auto& w : workers_) w.join();
  }
  void drain() {
    inside_ = true;
    for (int i = next_.fetch_add(1, std::memory_order_relaxed); i < count_;
         i = next_.fetch_add(1, std::memory_order_relaxed)) {
      (*fn_)(i);
    }
    inside_ = false;
  }
  void loop(int id) {
    std::uint64_t seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
      seen = generation_;
      if (stop_) return;
      if (id >= active_) continue;
      lock.unlock();
      drain();
      lock.lock();
      if (--pending_ == 0) done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex job_mutex_, mutex_;
  std::condition_variable wake_, done_;
  const std::function<void(int)>* fn_ = nullptr;
  int count_ = 0;
  int active_ = 0;
  int pending_ = 0;
  std::uint64_t generation_ = 0;
  bool stop_ = false;
  std::atomic<int> next_{0};
  static thread_local bool inside_;
};

thread_local bool WorkerPool::inside_ = false;

int resolve_threads(int threads) {
  const int pool_n = cpu::Pool::instance().size();
  return threads <= 0 ? pool_n : threads;
}

void parallel_for(int count, int threads, const std::function<void(int)>& fn) {
  if (count <= 0) return;
  const int pool_n = cpu::Pool::instance().size();
  if (threads <= 0) threads = pool_n;
  threads = std::min(threads, count);
  if (threads <= 1) {
    for (int i = 0; i < count; ++i) fn(i);
    return;
  }
  if (threads <= pool_n) {
    alignas(64) std::atomic<int> next{0};
    cpu::Pool::instance().each([&](int w) {
      if (w >= threads) return;
      for (int i = next.fetch_add(1, std::memory_order_relaxed); i < count;
           i = next.fetch_add(1, std::memory_order_relaxed)) {
        fn(i);
      }
    });
    return;
  }
  WorkerPool::instance().run(count, threads, fn);
}

constexpr int kBlock = 64;

}  // namespace

// ================================ math (FP64) ==========================================
void math_attention_forward(void* o, float* lse, const void* q, const void* k, const void* v,
                            const AttentionParams& p) {
  check_cpu(p);
  const int D = p.head_dim;
  const int group = p.heads / p.kv_heads;
  const double scale = p.effective_scale();
  const double keep_scale = 1.0 / (1.0 - p.dropout_p);
  std::vector<double> s(static_cast<std::size_t>(p.tk));
  std::vector<double> acc(static_cast<std::size_t>(D));
  std::vector<float> qbuf(static_cast<std::size_t>(D));
  std::vector<float> kbuf(static_cast<std::size_t>(D));
  std::vector<float> vbuf(static_cast<std::size_t>(D));
  std::vector<float> obuf(static_cast<std::size_t>(D));
  for (int b = 0; b < p.batch; ++b) {
    const Geometry g = geometry(p, b);
    for (int h = 0; h < p.heads; ++h) {
      const int hk = h / group;
      for (int i = 0; i < g.tq; ++i) {
        const int qpos = g.tk - g.tq + i;
        const float* qi = get_row_f32(qbuf.data(), q, p.dtype, q_offset(p, g, h, i), D);
        double m = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < g.tk; ++j) {
          if (!allowed(p, qpos, j)) {
            s[static_cast<std::size_t>(j)] = -std::numeric_limits<double>::infinity();
            continue;
          }
          const float* kj = get_row_f32(kbuf.data(), k, p.dtype, k_offset(p, g, hk, j), D);
          double dot = 0.0;
          for (int d = 0; d < D; ++d) {
            dot += static_cast<double>(qi[d]) * kj[d];
          }
          double t = 0.0;
          s[static_cast<std::size_t>(j)] = transform(p, dot * scale, h, qpos, j, &t);
          m = std::max(m, s[static_cast<std::size_t>(j)]);
        }
        if (!std::isfinite(m)) {
          zero_row(o, p.dtype, o_offset(p, g, h, i), D);
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
          const float* vj = get_row_f32(vbuf.data(), v, p.dtype, k_offset(p, g, hk, j), D);
          for (int d = 0; d < D; ++d) {
            acc[static_cast<std::size_t>(d)] += pij * vj[d];
          }
        }
        for (int d = 0; d < D; ++d) {
          obuf[static_cast<std::size_t>(d)] = static_cast<float>(acc[static_cast<std::size_t>(d)]);
        }
        store_row_from_f32(o, p.dtype, o_offset(p, g, h, i), obuf.data(), D);
        lse[lse_index(p, g, h, i)] = static_cast<float>(row_lse);
      }
    }
  }
}

void math_attention_forward(float* o, float* lse, const float* q, const float* k, const float* v,
                            const AttentionParams& p) {
  math_attention_forward(static_cast<void*>(o), lse, static_cast<const void*>(q), static_cast<const void*>(k),
                         static_cast<const void*>(v), p);
}

void math_attention_backward(void* dq, void* dk, void* dv, const void* dout, const void* q, const void* k,
                             const void* v, const void* o, const float* lse, const AttentionParams& p) {
  check_cpu(p);
  const int D = p.head_dim;
  const int group = p.heads / p.kv_heads;
  const double scale = p.effective_scale();
  const double keep_scale = 1.0 / (1.0 - p.dropout_p);
  std::vector<float> qbuf(static_cast<std::size_t>(D)), kbuf(static_cast<std::size_t>(D)),
      vbuf(static_cast<std::size_t>(D)), dobuf(static_cast<std::size_t>(D)), obuf(static_cast<std::size_t>(D)),
      outbuf(static_cast<std::size_t>(D));
  for (int b = 0; b < p.batch; ++b) {
    const Geometry g = geometry(p, b);
    for (int h = 0; h < p.heads; ++h) {
      for (int i = 0; i < g.tq; ++i) {
        zero_row(dq, p.dtype, q_offset(p, g, h, i), D);
      }
    }
    std::vector<double> dk_f64(static_cast<std::size_t>(p.kv_heads) * g.tk * D, 0.0);
    std::vector<double> dv_f64(static_cast<std::size_t>(p.kv_heads) * g.tk * D, 0.0);
    std::vector<double> dqi(static_cast<std::size_t>(D));
    for (int h = 0; h < p.heads; ++h) {
      const int hk = h / group;
      for (int i = 0; i < g.tq; ++i) {
        const int qpos = g.tk - g.tq + i;
        const double row_lse = lse[lse_index(p, g, h, i)];
        if (!std::isfinite(row_lse)) {
          continue;
        }
        const float* qi = get_row_f32(qbuf.data(), q, p.dtype, q_offset(p, g, h, i), D);
        const float* doi = get_row_f32(dobuf.data(), dout, p.dtype, q_offset(p, g, h, i), D);
        const float* oi = get_row_f32(obuf.data(), o, p.dtype, q_offset(p, g, h, i), D);
        double Di = 0.0;
        for (int d = 0; d < D; ++d) {
          Di += static_cast<double>(doi[d]) * oi[d];
        }
        std::fill(dqi.begin(), dqi.end(), 0.0);
        for (int j = 0; j < g.tk; ++j) {
          if (!allowed(p, qpos, j)) {
            continue;
          }
          const float* kj = get_row_f32(kbuf.data(), k, p.dtype, k_offset(p, g, hk, j), D);
          const float* vj = get_row_f32(vbuf.data(), v, p.dtype, k_offset(p, g, hk, j), D);
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
          double* dvj = dv_f64.data() + (static_cast<std::size_t>(hk) * g.tk + j) * D;
          double* dkj = dk_f64.data() + (static_cast<std::size_t>(hk) * g.tk + j) * D;
          for (int d = 0; d < D; ++d) {
            dvj[d] += pij * z * doi[d];
            dkj[d] += ds * qi[d];
            dqi[static_cast<std::size_t>(d)] += ds * kj[d];
          }
        }
        for (int d = 0; d < D; ++d) {
          outbuf[static_cast<std::size_t>(d)] = static_cast<float>(dqi[static_cast<std::size_t>(d)]);
        }
        store_row_from_f32(dq, p.dtype, q_offset(p, g, h, i), outbuf.data(), D);
      }
    }
    for (int hk = 0; hk < p.kv_heads; ++hk) {
      for (int j = 0; j < g.tk; ++j) {
        const double* dkj = dk_f64.data() + (static_cast<std::size_t>(hk) * g.tk + j) * D;
        const double* dvj = dv_f64.data() + (static_cast<std::size_t>(hk) * g.tk + j) * D;
        for (int d = 0; d < D; ++d) outbuf[static_cast<std::size_t>(d)] = static_cast<float>(dkj[d]);
        store_row_from_f32(dk, p.dtype, k_offset(p, g, hk, j), outbuf.data(), D);
        for (int d = 0; d < D; ++d) outbuf[static_cast<std::size_t>(d)] = static_cast<float>(dvj[d]);
        store_row_from_f32(dv, p.dtype, k_offset(p, g, hk, j), outbuf.data(), D);
      }
    }
  }
}

void math_attention_backward(float* dq, float* dk, float* dv, const float* dout, const float* q, const float* k,
                             const float* v, const float* o, const float* lse, const AttentionParams& p) {
  math_attention_backward(static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv),
                          static_cast<const void*>(dout), static_cast<const void*>(q), static_cast<const void*>(k),
                          static_cast<const void*>(v), static_cast<const void*>(o), lse, p);
}

// ================================ CPU flash (FP32 / BF16 / F16) =========================
// Doseme duzeni: Br x Bc = 64 x 64. K (ve V) dosemesi [D][64] olarak paketlenir; boylece
// skor ve dP icin ic dongu 64 bitisik eleman uzerinde (vektorlesir), PV / dK / dV / dQ icin ic
// dongu D uzerindedir. Maskeli elemanlar -inf skor tasir: exp(-inf - m) = 0, dal yok.
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

#if defined(__AVX2__)
// 8x8 FP32 devrikleme: 8 satir (r0..r7) -> [8][kBlock] hedefinde c0 sutun ofsetine yazar.
inline void transpose8x8_to_block(float* bt, int c0, const float* r0, const float* r1, const float* r2, const float* r3,
                                  const float* r4, const float* r5, const float* r6, const float* r7) noexcept {
  const __m256 row0 = _mm256_loadu_ps(r0);
  const __m256 row1 = _mm256_loadu_ps(r1);
  const __m256 row2 = _mm256_loadu_ps(r2);
  const __m256 row3 = _mm256_loadu_ps(r3);
  const __m256 row4 = _mm256_loadu_ps(r4);
  const __m256 row5 = _mm256_loadu_ps(r5);
  const __m256 row6 = _mm256_loadu_ps(r6);
  const __m256 row7 = _mm256_loadu_ps(r7);

  const __m256 t0 = _mm256_unpacklo_ps(row0, row1);
  const __m256 t1 = _mm256_unpackhi_ps(row0, row1);
  const __m256 t2 = _mm256_unpacklo_ps(row2, row3);
  const __m256 t3 = _mm256_unpackhi_ps(row2, row3);
  const __m256 t4 = _mm256_unpacklo_ps(row4, row5);
  const __m256 t5 = _mm256_unpackhi_ps(row4, row5);
  const __m256 t6 = _mm256_unpacklo_ps(row6, row7);
  const __m256 t7 = _mm256_unpackhi_ps(row6, row7);

  const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
  const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
  const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
  const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
  const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
  const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

  _mm256_storeu_ps(bt + 0 * kBlock + c0, _mm256_permute2f128_ps(s0, s4, 0x20));
  _mm256_storeu_ps(bt + 1 * kBlock + c0, _mm256_permute2f128_ps(s1, s5, 0x20));
  _mm256_storeu_ps(bt + 2 * kBlock + c0, _mm256_permute2f128_ps(s2, s6, 0x20));
  _mm256_storeu_ps(bt + 3 * kBlock + c0, _mm256_permute2f128_ps(s3, s7, 0x20));
  _mm256_storeu_ps(bt + 4 * kBlock + c0, _mm256_permute2f128_ps(s0, s4, 0x31));
  _mm256_storeu_ps(bt + 5 * kBlock + c0, _mm256_permute2f128_ps(s1, s5, 0x31));
  _mm256_storeu_ps(bt + 6 * kBlock + c0, _mm256_permute2f128_ps(s2, s6, 0x31));
  _mm256_storeu_ps(bt + 7 * kBlock + c0, _mm256_permute2f128_ps(s3, s7, 0x31));
}
#endif

// nc satiri (krows[0..nc)) [D][kBlock] olarak paketler; c >= nc sutunlari sifir.
void pack_t_rows(float* bt, const float* const* krows, int nc, int D) noexcept {
  int c = 0;
#if defined(__AVX2__)
  if (D % 8 == 0) {
    for (; c + 8 <= nc; c += 8) {
      const float* r0 = krows[c + 0];
      const float* r1 = krows[c + 1];
      const float* r2 = krows[c + 2];
      const float* r3 = krows[c + 3];
      const float* r4 = krows[c + 4];
      const float* r5 = krows[c + 5];
      const float* r6 = krows[c + 6];
      const float* r7 = krows[c + 7];
      for (int d0 = 0; d0 < D; d0 += 8) {
        transpose8x8_to_block(bt + static_cast<std::size_t>(d0) * kBlock, c, r0 + d0, r1 + d0, r2 + d0, r3 + d0,
                              r4 + d0, r5 + d0, r6 + d0, r7 + d0);
      }
    }
  }
#endif
  for (; c < nc; ++c) {
    const float* row = krows[c];
    for (int d = 0; d < D; ++d) {
      bt[static_cast<std::size_t>(d) * kBlock + c] = row[d];
    }
  }
  if (nc < kBlock) {
    for (int d = 0; d < D; ++d) {
      std::memset(bt + static_cast<std::size_t>(d) * kBlock + nc, 0, static_cast<std::size_t>(kBlock - nc) * sizeof(float));
    }
  }
}

#if defined(__AVX2__)
// Cephes expf, 8 serit (hata <= 2 ULP). x < -87.34 (-inf dahil) tam 0 verir: maskeli skor = 0.
inline __m256 exp_avx2(__m256 x) noexcept {
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

inline float hmax_avx2(__m256 v) noexcept {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  __m128 m = _mm_max_ps(lo, hi);
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
  return _mm_cvtss_f32(m);
}

inline float hsum_avx2(__m256 v) noexcept {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
#endif

inline float block_max(const float* x) noexcept {
#if defined(__AVX2__)
  __m256 m0 = _mm256_loadu_ps(x + 0);
  __m256 m1 = _mm256_loadu_ps(x + 8);
  m0 = _mm256_max_ps(m0, _mm256_loadu_ps(x + 16));
  m1 = _mm256_max_ps(m1, _mm256_loadu_ps(x + 24));
  m0 = _mm256_max_ps(m0, _mm256_loadu_ps(x + 32));
  m1 = _mm256_max_ps(m1, _mm256_loadu_ps(x + 40));
  m0 = _mm256_max_ps(m0, _mm256_loadu_ps(x + 48));
  m1 = _mm256_max_ps(m1, _mm256_loadu_ps(x + 56));
  return hmax_avx2(_mm256_max_ps(m0, m1));
#else
  float mx = kNegInf;
  for (int c = 0; c < kBlock; ++c) mx = std::max(mx, x[c]);
  return mx;
#endif
}

// x[c] = exp(x[c] - m), c < kBlock; ayni geciste yatay toplami dondurur.
inline float exp_shift_sum(float* x, float m) noexcept {
#if defined(__AVX2__)
  const __m256 mv = _mm256_set1_ps(m);
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  for (int c = 0; c < kBlock; c += 16) {
    const __m256 e0 = exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + c), mv));
    const __m256 e1 = exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + c + 8), mv));
    _mm256_storeu_ps(x + c, e0);
    _mm256_storeu_ps(x + c + 8, e1);
    sum0 = _mm256_add_ps(sum0, e0);
    sum1 = _mm256_add_ps(sum1, e1);
  }
  return hsum_avx2(_mm256_add_ps(sum0, sum1));
#else
  float sum = 0.0f;
  for (int c = 0; c < kBlock; ++c) {
    x[c] = std::exp(x[c] - m);
    sum += x[c];
  }
  return sum;
#endif
}

// x[c] = exp(x[c] - m), c < kBlock.
inline void exp_shift(float* x, float m) noexcept {
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

inline void scale_vec(float* dst, const float* src, float s, int D) noexcept {
  int d = 0;
#if defined(__AVX2__)
  const __m256 sv = _mm256_set1_ps(s);
  for (; d + 16 <= D; d += 16) {
    _mm256_storeu_ps(dst + d, _mm256_mul_ps(_mm256_loadu_ps(src + d), sv));
    _mm256_storeu_ps(dst + d + 8, _mm256_mul_ps(_mm256_loadu_ps(src + d + 8), sv));
  }
  for (; d + 8 <= D; d += 8) {
    _mm256_storeu_ps(dst + d, _mm256_mul_ps(_mm256_loadu_ps(src + d), sv));
  }
#endif
  for (; d < D; ++d) dst[d] = src[d] * s;
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

#if defined(__AVX2__)
// FAZ 18: R x 16 yazmac dosemesi (R <= 6: 12 biriktirici). out[r][c0 + 0..16) = a_r . bt[:, c0 + ..].
template <int R>
inline void dot_rows(float* out, const float* const* arows, const float* bt, int D) {
  const float* a[R];
  for (int r = 0; r < R; ++r) a[r] = arows[r];
  for (int c0 = 0; c0 < kBlock; c0 += 16) {
    __m256 acc0[R], acc1[R];
    for (int r = 0; r < R; ++r) acc0[r] = acc1[r] = _mm256_setzero_ps();
    for (int d = 0; d < D; ++d) {
      const float* b = bt + static_cast<std::size_t>(d) * kBlock + c0;
      const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + 8);
      for (int r = 0; r < R; ++r) {
        const __m256 av = _mm256_broadcast_ss(a[r] + d);
        acc0[r] = _mm256_fmadd_ps(av, b0, acc0[r]);
        acc1[r] = _mm256_fmadd_ps(av, b1, acc1[r]);
      }
    }
    for (int r = 0; r < R; ++r) {
      _mm256_storeu_ps(out + static_cast<std::size_t>(r) * kBlock + c0, acc0[r]);
      _mm256_storeu_ps(out + static_cast<std::size_t>(r) * kBlock + c0 + 8, acc1[r]);
    }
  }
}

// out[a * ldo + d] += sum_b coef[a * sa + b * sb] * rows[b][d], a < R, D % 16 == 0 (S V, dS K, P^T dO, dS^T Q).
template <int R>
inline void mm_rows(float* out, int ldo, const float* coef, int sa, int sb, const float* const* rows, int nb, int D) {
  for (int d0 = 0; d0 < D; d0 += 16) {
    __m256 acc0[R], acc1[R];
    for (int r = 0; r < R; ++r) {
      acc0[r] = _mm256_loadu_ps(out + static_cast<std::size_t>(r) * ldo + d0);
      acc1[r] = _mm256_loadu_ps(out + static_cast<std::size_t>(r) * ldo + d0 + 8);
    }
    for (int b = 0; b < nb; ++b) {
      const float* rw = rows[b] + d0;
      const __m256 v0 = _mm256_loadu_ps(rw), v1 = _mm256_loadu_ps(rw + 8);
      const float* cb = coef + static_cast<std::size_t>(b) * sb;
      for (int r = 0; r < R; ++r) {
        const __m256 cv = _mm256_broadcast_ss(cb + static_cast<std::size_t>(r) * sa);
        acc0[r] = _mm256_fmadd_ps(cv, v0, acc0[r]);
        acc1[r] = _mm256_fmadd_ps(cv, v1, acc1[r]);
      }
    }
    for (int r = 0; r < R; ++r) {
      _mm256_storeu_ps(out + static_cast<std::size_t>(r) * ldo + d0, acc0[r]);
      _mm256_storeu_ps(out + static_cast<std::size_t>(r) * ldo + d0 + 8, acc1[r]);
    }
  }
}

template <typename F>
inline void by_six(int n, F&& f) {
  int a = 0;
  for (; a + 6 <= n; a += 6) f(a, std::integral_constant<int, 6>{});
  switch (n - a) {
    case 5: f(a, std::integral_constant<int, 5>{}); break;
    case 4: f(a, std::integral_constant<int, 4>{}); break;
    case 3: f(a, std::integral_constant<int, 3>{}); break;
    case 2: f(a, std::integral_constant<int, 2>{}); break;
    case 1: f(a, std::integral_constant<int, 1>{}); break;
    default: break;
  }
}
#endif

// out[a][d] += sum_b coef[a * sa + b * sb] * rows[b][d], a < na, b < nb.
void mm_tile(float* out, int ldo, const float* coef, int sa, int sb, int na, const float* const* rows, int nb, int D) {
#if defined(__AVX2__)
  if (D % 16 == 0) {
    by_six(na, [&](int a, auto R) {
      mm_rows<decltype(R)::value>(out + static_cast<std::size_t>(a) * ldo, ldo, coef + static_cast<std::size_t>(a) * sa, sa, sb,
                                  rows, nb, D);
    });
    return;
  }
#endif
  for (int a = 0; a < na; ++a) {
    accum_rows(out + static_cast<std::size_t>(a) * ldo, coef + static_cast<std::size_t>(a) * sa, sb, rows, nb, D);
  }
}

// out[r][c] = a_r . bt[:, c]
void dot_tile(float* out, const float* const* arows, int nr, const float* bt, int D) {
#if defined(__AVX2__)
  by_six(nr, [&](int r, auto R) { dot_rows<decltype(R)::value>(out + static_cast<std::size_t>(r) * kBlock, arows + r, bt, D); });
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
// FAZ 18: tam gorunur karolarda (c_lo == 0 && c_hi == kBlock) AVX2 vektor olcekleme;
// kismi gorunur satirlarda [c_lo, c_hi) AVX2 olcekleme + maske.
void score_tile(float* s, float* t, const AttentionParams& p, const Geometry& g, int h, int i0, int nr, int j0, int nc,
                float scale, int* out_max_chi = nullptr) {
  const bool cap = p.softcap > 0.0f;
  const float slope = p.alibi_slopes != nullptr ? p.alibi_slopes[h] : 0.0f;
  const bool plain = !cap && p.alibi_slopes == nullptr;

  // Hizli yol: maskesiz tam blok (nc == kBlock, tum r'ler icin c_lo == 0 ve c_hi == kBlock).
  const long long qpos_first = g.tk - g.tq + i0;
  const long long qpos_last = qpos_first + nr - 1;
  const bool all_unmasked =
      nc == kBlock && (!p.causal || static_cast<long long>(j0) + kBlock - 1 <= qpos_first) &&
      (p.window <= 0 || (qpos_last - p.window + 1 <= j0 && (p.causal || j0 + kBlock - 1 <= qpos_first + p.window - 1)));
  if (all_unmasked && plain) {
    if (out_max_chi) *out_max_chi = kBlock;
    scale_vec(s, s, scale, nr * kBlock);
    return;
  }

  int max_chi = 0;
  for (int r = 0; r < nr; ++r) {
    const long long qpos = g.tk - g.tq + i0 + r;
    long long lo = 0, hi = static_cast<long long>(j0) + kBlock;
    if (p.causal) hi = qpos;
    if (p.window > 0) {
      lo = qpos - p.window + 1;
      if (!p.causal) hi = qpos + p.window - 1;
    }
    const int c_lo = static_cast<int>(std::clamp<long long>(lo - j0, 0, nc));
    const int c_hi = std::max(c_lo, static_cast<int>(std::clamp<long long>(hi - j0 + 1, 0, nc)));
    max_chi = std::max(max_chi, c_hi);
    float* sr = s + static_cast<std::size_t>(r) * kBlock;
    float* tr = t + static_cast<std::size_t>(r) * kBlock;
    for (int c = 0; c < c_lo; ++c) sr[c] = kNegInf;
    for (int c = c_hi; c < kBlock; ++c) sr[c] = kNegInf;
    if (plain) {
      scale_vec(sr + c_lo, sr + c_lo, scale, c_hi - c_lo);
      continue;
    }
    for (int c = c_lo; c < c_hi; ++c) {
      const int j = j0 + c;
      float v = sr[c] * scale;
      if (cap) {
        const float th = std::tanh(v / p.softcap);
        tr[c] = th;
        v = p.softcap * th;
      }
      sr[c] = v + slope * static_cast<float>(j - qpos);
    }
  }
  if (out_max_chi) *out_max_chi = max_chi;
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

// İş parçacığı başına kalıcı çizik tamponları (blok başına heap ayırmayı tamamen kaldırır).
struct FlashScratch {
  int cap_d = 0;
  std::vector<float> kt;
  std::vector<float> vt;
  std::vector<float> s;
  std::vector<float> t;
  std::vector<float> dp;
  std::vector<float> acc;
  std::vector<float> dk_acc;
  std::vector<float> dv_acc;
  std::vector<float> dq_acc;
  std::vector<float> q_tile;
  std::vector<float> do_tile;
  std::vector<float> k_tile;
  std::vector<float> v_tile;

  void ensure(int D, bool need_bwd, bool need_cvt) {
    if (s.empty()) {
      s.resize(kBlock * kBlock);
      t.resize(kBlock * kBlock);
      dp.resize(kBlock * kBlock);
    }
    if (D > cap_d) {
      const std::size_t bd = static_cast<std::size_t>(kBlock) * D;
      kt.resize(bd);
      acc.resize(bd);
      cap_d = D;
    }
    const std::size_t bd = static_cast<std::size_t>(kBlock) * D;
    if (need_bwd && vt.size() < bd) {
      vt.resize(bd);
      dk_acc.resize(bd);
      dv_acc.resize(bd);
      dq_acc.resize(bd);
    }
    if (need_cvt && q_tile.size() < bd) {
      q_tile.resize(bd);
      do_tile.resize(bd);
      k_tile.resize(bd);
      v_tile.resize(bd);
    }
  }
};

thread_local FlashScratch t_scratch;

}  // namespace

void flash_attention_forward_cpu(void* o, float* lse, const void* q, const void* k, const void* v,
                                 const AttentionParams& p, int threads) {
  check_cpu(p);
  const int D = p.head_dim;
  const int group = p.heads / p.kv_heads;
  const float scale = p.effective_scale();
  const float keep_scale = 1.0f / (1.0f - p.dropout_p);
  const int qblocks = (p.tq + kBlock - 1) / kBlock;
  const bool is_f32 = (p.dtype == DType::F32);

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

    FlashScratch& w = t_scratch;
    w.ensure(D, /*need_bwd=*/false, /*need_cvt=*/!is_f32);

    float m[kBlock];
    float l[kBlock];
    for (int r = 0; r < nr; ++r) {
      m[r] = kNegInf;
      l[r] = 0.0f;
    }
    std::memset(w.acc.data(), 0, static_cast<std::size_t>(nr) * D * sizeof(float));

    const float* qrows[kBlock];
    for (int r = 0; r < nr; ++r) {
      qrows[r] = get_row_f32(w.q_tile.data() + static_cast<std::size_t>(r) * D, q, p.dtype, q_offset(p, g, h, i0 + r), D);
    }

    int j_lo = 0;
    int j_hi = 0;
    key_span(p, g, i0, nr, &j_lo, &j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += kBlock) {
      const int nc = std::min(kBlock, j_hi - j0);
      const float* krows[kBlock];
      const float* vrows[kBlock];
      for (int c = 0; c < nc; ++c) {
        const long long koff = k_offset(p, g, hk, j0 + c);
        krows[c] = get_row_f32(w.k_tile.data() + static_cast<std::size_t>(c) * D, k, p.dtype, koff, D);
        vrows[c] = get_row_f32(w.v_tile.data() + static_cast<std::size_t>(c) * D, v, p.dtype, koff, D);
      }
      pack_t_rows(w.kt.data(), krows, nc, D);

      dot_tile(w.s.data(), qrows, nr, w.kt.data(), D);
      int max_chi = nc;
      score_tile(w.s.data(), w.t.data(), p, g, h, i0, nr, j0, nc, scale, &max_chi);

      for (int r = 0; r < nr; ++r) {
        float* sr = w.s.data() + static_cast<std::size_t>(r) * kBlock;
        const float mx = block_max(sr);
        if (mx == kNegInf) {
          std::memset(sr, 0, kBlock * sizeof(float));
          continue;
        }
        const float m_prev = m[r];
        const float l_prev = l[r];
        const float m_new = std::max(m_prev, mx);
        const float alpha = (m_prev == kNegInf) ? 0.0f : ((m_prev == m_new) ? 1.0f : std::exp(m_prev - m_new));
        const float sum = exp_shift_sum(sr, m_new);
        l[r] = l_prev * alpha + sum;
        m[r] = m_new;
        if (p.dropout_p > 0.0f) {
          for (int c = 0; c < nc; ++c) {
            sr[c] = fused::philox_uniform(p.seed, p.offset, dropout_index(p, g, h, i0 + r, j0 + c)) < p.dropout_p
                        ? 0.0f
                        : sr[c] * keep_scale;
          }
        }
        if (l_prev > 0.0f && alpha != 1.0f) {
          float* ar = w.acc.data() + static_cast<std::size_t>(r) * D;
          scale_vec(ar, ar, alpha, D);
        }
      }
      if (max_chi > 0) {
        mm_tile(w.acc.data(), D, w.s.data(), kBlock, 1, nr, vrows, max_chi, D);  // acc += P V
      }
    }
    for (int r = 0; r < nr; ++r) {
      const long long ooff = o_offset(p, g, h, i0 + r);
      if (!(l[r] > 0.0f)) {
        zero_row(o, p.dtype, ooff, D);
        lse[lse_index(p, g, h, i0 + r)] = kNegInf;
        continue;
      }
      const float inv = 1.0f / l[r];
      float* ar = w.acc.data() + static_cast<std::size_t>(r) * D;
      if (is_f32) {
        scale_vec(static_cast<float*>(o) + ooff, ar, inv, D);
      } else {
        scale_vec(ar, ar, inv, D);
        store_row_from_f32(o, p.dtype, ooff, ar, D);
      }
      lse[lse_index(p, g, h, i0 + r)] = m[r] + std::log(l[r]);
    }
  });
}

void flash_attention_forward_cpu(float* o, float* lse, const float* q, const float* k, const float* v,
                                 const AttentionParams& p, int threads) {
  flash_attention_forward_cpu(static_cast<void*>(o), lse, static_cast<const void*>(q), static_cast<const void*>(k),
                              static_cast<const void*>(v), p, threads);
}

void flash_attention_backward_cpu(void* dq, void* dk, void* dv, const void* dout, const void* q, const void* k,
                                  const void* v, const void* o, const float* lse, const AttentionParams& p,
                                  int threads) {
  check_cpu(p);
  const int D = p.head_dim;
  const int group = p.heads / p.kv_heads;
  const float scale = p.effective_scale();
  const float keep_scale = 1.0f / (1.0f - p.dropout_p);
  const bool is_f32 = (p.dtype == DType::F32);

  // D_i = rowsum(dO o O), [lse duzeninde].
  const int total_rows = p.cu_seqlens_q != nullptr ? p.cu_seqlens_q[p.batch] : p.batch * p.tq;
  std::vector<float> Dvec(static_cast<std::size_t>(total_rows) * p.heads, 0.0f);
  parallel_for(p.batch * p.heads, threads, [&](int bh) {
    const int h = bh % p.heads;
    const int b = bh / p.heads;
    const Geometry g = geometry(p, b);
    float do_row_buf[256];
    float o_row_buf[256];
    std::vector<float> do_big, o_big;
    if (!is_f32 && D > 256) {
      do_big.resize(static_cast<std::size_t>(D));
      o_big.resize(static_cast<std::size_t>(D));
    }
    float* do_tmp = D <= 256 ? do_row_buf : do_big.data();
    float* o_tmp = D <= 256 ? o_row_buf : o_big.data();
    for (int i = 0; i < g.tq; ++i) {
      const long long qoff = q_offset(p, g, h, i);
      const float* doi = get_row_f32(do_tmp, dout, p.dtype, qoff, D);
      const float* oi = get_row_f32(o_tmp, o, p.dtype, qoff, D);
      float acc = 0.0f;
      int d = 0;
#if defined(__AVX2__)
      __m256 vacc = _mm256_setzero_ps();
      for (; d + 8 <= D; d += 8) {
        vacc = _mm256_fmadd_ps(_mm256_loadu_ps(doi + d), _mm256_loadu_ps(oi + d), vacc);
      }
      acc = hsum_avx2(vacc);
#endif
      for (; d < D; ++d) {
        acc += doi[d] * oi[d];
      }
      Dvec[lse_index(p, g, h, i)] = acc;
    }
  });

  // kt/vt paketli olmali; cikis: s <- pz, dp <- ds.
  auto tile = [&](FlashScratch& w, const Geometry& g, int h, int i0, int nr, int j0, int nc, const float* const* qrows,
                  const float* const* dorows) {
    dot_tile(w.s.data(), qrows, nr, w.kt.data(), D);
    score_tile(w.s.data(), w.t.data(), p, g, h, i0, nr, j0, nc, scale);
    dot_tile(w.dp.data(), dorows, nr, w.vt.data(), D);
    const bool fast_bwd = (p.dropout_p == 0.0f && p.softcap == 0.0f);
    for (int r = 0; r < nr; ++r) {
      const std::size_t row = lse_index(p, g, h, i0 + r);
      const float row_lse = lse[row];
      const float di = Dvec[row];
      float* pr = w.s.data() + static_cast<std::size_t>(r) * kBlock;
      float* dr = w.dp.data() + static_cast<std::size_t>(r) * kBlock;
      const float* tr = w.t.data() + static_cast<std::size_t>(r) * kBlock;
      if (!std::isfinite(row_lse)) {
        std::memset(pr, 0, kBlock * sizeof(float));
        std::memset(dr, 0, kBlock * sizeof(float));
        continue;
      }
      if (fast_bwd) {
#if defined(__AVX2__)
        const __m256 lsev = _mm256_set1_ps(row_lse);
        const __m256 div = _mm256_set1_ps(di);
        const __m256 scv = _mm256_set1_ps(scale);
        for (int c = 0; c < kBlock; c += 8) {
          const __m256 pv = exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(pr + c), lsev));
          const __m256 dv = _mm256_loadu_ps(dr + c);
          _mm256_storeu_ps(pr + c, pv);
          _mm256_storeu_ps(dr + c, _mm256_mul_ps(_mm256_mul_ps(pv, _mm256_sub_ps(dv, div)), scv));
        }
        continue;
#endif
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
      scale_vec(dr, dr, scale, kBlock);
    }
  };

  const int eff_threads = resolve_threads(threads);
  // Tek gecisli geri yayilim: F32'de `(batch * kv_heads) >= eff_threads` (veya tek is parcacigi) oldugunda
  // her `(b, hk)` birimi kendi sorgu baslari `h in [hk*group, (hk+1)*group)` ve kendi `hk` anahtar/deger
  // baslari uzerinde tamamen bagimsizdir. Boylece `tile()` (Q K^T, exp, dO V^T) iki kez degil yalniz
  // BIR kez hesaplanir (atomiksiz, %100 belirlenimci, 7 yerine 5 karo GEMM'i).
  if (is_f32 && (p.batch * p.kv_heads >= eff_threads || eff_threads == 1)) {
    parallel_for(p.batch * p.kv_heads, threads, [&](int task) {
      const int hk = task % p.kv_heads;
      const int b = task / p.kv_heads;
      const Geometry g = geometry(p, b);
      FlashScratch& w = t_scratch;
      w.ensure(D, /*need_bwd=*/true, /*need_cvt=*/false);

      // Bu kv_head grubuna ait tum sorgu baslarinin dQ satirlarini sifirla.
      for (int h = hk * group; h < (hk + 1) * group; ++h) {
        for (int i = 0; i < g.tq; ++i) {
          std::memset(o_row_f32(dq, p, g, h, i), 0, static_cast<std::size_t>(D) * sizeof(float));
        }
      }

      const float* qrows[kBlock];
      const float* dorows[kBlock];
      const float* krows[kBlock];
      const float* vrows[kBlock];

      for (int j0 = 0; j0 < g.tk; j0 += kBlock) {
        const int nc = std::min(kBlock, g.tk - j0);
        for (int c = 0; c < nc; ++c) {
          krows[c] = k_row_f32(k, p, g, hk, j0 + c);
          vrows[c] = k_row_f32(v, p, g, hk, j0 + c);
        }
        pack_t_rows(w.kt.data(), krows, nc, D);
        pack_t_rows(w.vt.data(), vrows, nc, D);
        std::memset(w.dk_acc.data(), 0, static_cast<std::size_t>(nc) * D * sizeof(float));
        std::memset(w.dv_acc.data(), 0, static_cast<std::size_t>(nc) * D * sizeof(float));

        int i_lo = 0;
        int i_hi = 0;
        query_span(p, g, j0, nc, &i_lo, &i_hi);
        for (int h = hk * group; h < (hk + 1) * group; ++h) {
          for (int i0 = i_lo; i0 < i_hi; i0 += kBlock) {
            const int nr = std::min(kBlock, i_hi - i0);
            for (int r = 0; r < nr; ++r) {
              qrows[r] = q_row_f32(q, p, g, h, i0 + r);
              dorows[r] = q_row_f32(dout, p, g, h, i0 + r);
            }
            tile(w, g, h, i0, nr, j0, nc, qrows, dorows);
            mm_tile(w.dv_acc.data(), D, w.s.data(), 1, kBlock, nc, dorows, nr, D);  // dV += P^T dO
            mm_tile(w.dk_acc.data(), D, w.dp.data(), 1, kBlock, nc, qrows, nr, D);  // dK += dS^T Q

            // dQ += dS K: bitisik satirlarda dogrudan, adimli satirlarda gecici tampon uzerinden.
            if (p.q_row_stride == D) {
              float* dq_base = o_row_f32(dq, p, g, h, i0);
              mm_tile(dq_base, D, w.dp.data(), kBlock, 1, nr, krows, nc, D);
            } else {
              for (int r = 0; r < nr; ++r) {
                std::memcpy(w.dq_acc.data() + static_cast<std::size_t>(r) * D, o_row_f32(dq, p, g, h, i0 + r),
                            static_cast<std::size_t>(D) * sizeof(float));
              }
              mm_tile(w.dq_acc.data(), D, w.dp.data(), kBlock, 1, nr, krows, nc, D);
              for (int r = 0; r < nr; ++r) {
                std::memcpy(o_row_f32(dq, p, g, h, i0 + r), w.dq_acc.data() + static_cast<std::size_t>(r) * D,
                            static_cast<std::size_t>(D) * sizeof(float));
              }
            }
          }
        }
        for (int c = 0; c < nc; ++c) {
          std::memcpy(const_cast<float*>(k_row_f32(dk, p, g, hk, j0 + c)),
                      w.dk_acc.data() + static_cast<std::size_t>(c) * D, static_cast<std::size_t>(D) * sizeof(float));
          std::memcpy(const_cast<float*>(k_row_f32(dv, p, g, hk, j0 + c)),
                      w.dv_acc.data() + static_cast<std::size_t>(c) * D, static_cast<std::size_t>(D) * sizeof(float));
        }
      }
    });
    return;
  }

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
    FlashScratch& w = t_scratch;
    w.ensure(D, /*need_bwd=*/true, /*need_cvt=*/!is_f32);

    const float* krows[kBlock];
    const float* vrows[kBlock];
    for (int c = 0; c < nc; ++c) {
      const long long koff = k_offset(p, g, hk, j0 + c);
      krows[c] = get_row_f32(w.k_tile.data() + static_cast<std::size_t>(c) * D, k, p.dtype, koff, D);
      vrows[c] = get_row_f32(w.v_tile.data() + static_cast<std::size_t>(c) * D, v, p.dtype, koff, D);
    }
    pack_t_rows(w.kt.data(), krows, nc, D);
    pack_t_rows(w.vt.data(), vrows, nc, D);
    std::memset(w.dk_acc.data(), 0, static_cast<std::size_t>(nc) * D * sizeof(float));
    std::memset(w.dv_acc.data(), 0, static_cast<std::size_t>(nc) * D * sizeof(float));

    int i_lo = 0;
    int i_hi = 0;
    query_span(p, g, j0, nc, &i_lo, &i_hi);
    const float* qrows[kBlock];
    const float* dorows[kBlock];
    for (int h = hk * group; h < (hk + 1) * group; ++h) {
      for (int i0 = i_lo; i0 < i_hi; i0 += kBlock) {
        const int nr = std::min(kBlock, i_hi - i0);
        for (int r = 0; r < nr; ++r) {
          const long long qoff = q_offset(p, g, h, i0 + r);
          qrows[r] = get_row_f32(w.q_tile.data() + static_cast<std::size_t>(r) * D, q, p.dtype, qoff, D);
          dorows[r] = get_row_f32(w.do_tile.data() + static_cast<std::size_t>(r) * D, dout, p.dtype, qoff, D);
        }
        tile(w, g, h, i0, nr, j0, nc, qrows, dorows);
        mm_tile(w.dv_acc.data(), D, w.s.data(), 1, kBlock, nc, dorows, nr, D);  // dV += P^T dO
        mm_tile(w.dk_acc.data(), D, w.dp.data(), 1, kBlock, nc, qrows, nr, D);  // dK += dS^T Q
      }
    }
    for (int c = 0; c < nc; ++c) {
      const long long koff = k_offset(p, g, hk, j0 + c);
      store_row_from_f32(dk, p.dtype, koff, w.dk_acc.data() + static_cast<std::size_t>(c) * D, D);
      store_row_from_f32(dv, p.dtype, koff, w.dv_acc.data() + static_cast<std::size_t>(c) * D, D);
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
    FlashScratch& w = t_scratch;
    w.ensure(D, /*need_bwd=*/true, /*need_cvt=*/!is_f32);
    std::memset(w.dq_acc.data(), 0, static_cast<std::size_t>(nr) * D * sizeof(float));

    const float* qrows[kBlock];
    const float* dorows[kBlock];
    for (int r = 0; r < nr; ++r) {
      const long long qoff = q_offset(p, g, h, i0 + r);
      qrows[r] = get_row_f32(w.q_tile.data() + static_cast<std::size_t>(r) * D, q, p.dtype, qoff, D);
      dorows[r] = get_row_f32(w.do_tile.data() + static_cast<std::size_t>(r) * D, dout, p.dtype, qoff, D);
    }
    int j_lo = 0;
    int j_hi = 0;
    key_span(p, g, i0, nr, &j_lo, &j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += kBlock) {
      const int nc = std::min(kBlock, j_hi - j0);
      const float* krows[kBlock];
      const float* vrows[kBlock];
      for (int c = 0; c < nc; ++c) {
        const long long koff = k_offset(p, g, hk, j0 + c);
        krows[c] = get_row_f32(w.k_tile.data() + static_cast<std::size_t>(c) * D, k, p.dtype, koff, D);
        vrows[c] = get_row_f32(w.v_tile.data() + static_cast<std::size_t>(c) * D, v, p.dtype, koff, D);
      }
      pack_t_rows(w.kt.data(), krows, nc, D);
      pack_t_rows(w.vt.data(), vrows, nc, D);
      tile(w, g, h, i0, nr, j0, nc, qrows, dorows);
      mm_tile(w.dq_acc.data(), D, w.dp.data(), kBlock, 1, nr, krows, nc, D);  // dQ += dS K
    }
    for (int r = 0; r < nr; ++r) {
      store_row_from_f32(dq, p.dtype, q_offset(p, g, h, i0 + r), w.dq_acc.data() + static_cast<std::size_t>(r) * D, D);
    }
  });
}

void flash_attention_backward_cpu(float* dq, float* dk, float* dv, const float* dout, const float* q, const float* k,
                                  const float* v, const float* o, const float* lse, const AttentionParams& p,
                                  int threads) {
  flash_attention_backward_cpu(static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv),
                               static_cast<const void*>(dout), static_cast<const void*>(q), static_cast<const void*>(k),
                               static_cast<const void*>(v), static_cast<const void*>(o), lse, p, threads);
}

}  // namespace attention
}  // namespace nexus
