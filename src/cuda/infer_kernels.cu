/**
 * @file infer_kernels.cu
 * @brief FAZ 16 cikarim cekirdekleri: gomme, KV ekleme, sayfali dikkat, ince GEMM, ornekleme, NLL, nicemleme.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "infer_kernels.hpp"

#include "common.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <stdexcept>
#include <string>

namespace nexus {
namespace cuda {
namespace infer {
namespace k {

namespace {

using detail::block_max;
using detail::block_sum;
using detail::warp_sum;

__device__ __forceinline__ float ld(const __half* p, std::size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float ld(const float* p, std::size_t i) { return p[i]; }
__device__ __forceinline__ void st(__half* p, std::size_t i, float v) { p[i] = __float2half_rn(v); }
__device__ __forceinline__ void st(float* p, std::size_t i, float v) { p[i] = v; }

__device__ __forceinline__ float gelu_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("Nexus infer cekirdegi ") + what + ": " + cudaGetErrorString(e));
}

[[noreturn]] void bad_type(const char* what) { throw std::invalid_argument(std::string("Nexus infer: desteklenmeyen tip: ") + what); }

// ---- gomme --------------------------------------------------------------------------------------
template <typename W>
__global__ void embed_kernel(float* x, const std::int32_t* tok, const std::int32_t* pos, const W* wte, const W* wpe, int n,
                             int d) {
  const std::size_t total = static_cast<std::size_t>(n) * d;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(i / d), c = static_cast<int>(i % d);
    x[i] = ld(wte, static_cast<std::size_t>(tok[r]) * d + c) + ld(wpe, static_cast<std::size_t>(pos[r]) * d + c);
  }
}

// ---- KV ekleme --------------------------------------------------------------------------------
template <typename A, typename C>
__global__ void kv_append_kernel(C* kc, C* vc, const A* qkv, const std::int32_t* slot, int n, int H, int D) {
  const std::size_t per = static_cast<std::size_t>(H) * D;
  const std::size_t total = static_cast<std::size_t>(n) * per;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t r = i / per, e = i % per;
    const std::size_t dst = static_cast<std::size_t>(slot[r]) * per + e;
    st(kc, dst, ld(qkv, r * 3 * per + per + e));
    st(vc, dst, ld(qkv, r * 3 * per + 2 * per + e));
  }
}

/// INT8: warp basina (token, bas); D <= 128 (lane basina D/32 eleman).
template <typename A>
__global__ void kv_append_i8_kernel(std::int8_t* kc, std::int8_t* vc, float* ks, float* vs, const A* qkv,
                                    const std::int32_t* slot, int n, int H, int D) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= n * H) return;
  const int r = warp / H, h = warp % H;
  const std::size_t per = static_cast<std::size_t>(H) * D;
  for (int which = 0; which < 2; ++which) {
    const std::size_t src = static_cast<std::size_t>(r) * 3 * per + (1 + which) * per + static_cast<std::size_t>(h) * D;
    float v[4];
    float amax = 0.0f;
    const int epl = D / 32;
    for (int e = 0; e < epl; ++e) {
      v[e] = ld(qkv, src + lane * epl + e);
      amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const std::size_t dst = (static_cast<std::size_t>(slot[r]) * H + h) * D;
    std::int8_t* c = which == 0 ? kc : vc;
    for (int e = 0; e < epl; ++e) {
      const float q = rintf(v[e] / scale);
      c[dst + lane * epl + e] = static_cast<std::int8_t>(fmaxf(-127.0f, fminf(127.0f, q)));
    }
    if (lane == 0) (which == 0 ? ks : vs)[static_cast<std::size_t>(slot[r]) * H + h] = scale;
  }
}

// ---- sayfali dikkat ---------------------------------------------------------------------------
__device__ __forceinline__ float ldc(const __half* p, std::size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float ldc(const float* p, std::size_t i) { return p[i]; }
__device__ __forceinline__ float ldc(const std::int8_t* p, std::size_t i) { return static_cast<float>(p[i]); }

template <typename A, typename C, int D>
__global__ void __launch_bounds__(128) paged_kernel(PagedArgs a) {
  extern __shared__ float sc[];  // [max_ctx]
  __shared__ float qs[D];
  __shared__ float red[32];
  __shared__ float accs[4][D];
  constexpr int EPL = D / 32;
  const int i = blockIdx.x, h = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const A* qkv = static_cast<const A*>(a.qkv);
  const std::size_t qbase = static_cast<std::size_t>(i) * 3 * a.H * D + static_cast<std::size_t>(h) * D;
  for (int t = threadIdx.x; t < D; t += blockDim.x) qs[t] = ld(qkv, qbase + t) * a.scale;
  __syncthreads();
  const int seq = a.row_seq[i];
  const int ctx = a.row_pos[i] + 1;
  const std::int32_t* bt = a.block_table + static_cast<std::size_t>(seq) * a.max_blocks;
  const C* kc = static_cast<const C*>(a.kc);
  const C* vc = static_cast<const C*>(a.vc);
  const bool i8 = a.kscale != nullptr;
  for (int j = warp; j < ctx; j += 4) {
    const int slot = bt[j / a.block_size] * a.block_size + j % a.block_size;
    const std::size_t base = (static_cast<std::size_t>(slot) * a.H + h) * D;
    float dot = 0.0f;
#pragma unroll
    for (int e = 0; e < EPL; ++e) dot += qs[lane * EPL + e] * ldc(kc, base + lane * EPL + e);
    dot = warp_sum(dot);
    if (lane == 0) sc[j] = i8 ? dot * a.kscale[static_cast<std::size_t>(slot) * a.H + h] : dot;
  }
  __syncthreads();
  float m = -FLT_MAX;
  for (int j = threadIdx.x; j < ctx; j += blockDim.x) m = fmaxf(m, sc[j]);
  m = block_max(m, red);
  __syncthreads();
  float sum = 0.0f;
  for (int j = threadIdx.x; j < ctx; j += blockDim.x) {
    const float e = __expf(sc[j] - m);
    sc[j] = e;
    sum += e;
  }
  sum = block_sum(sum, red);
  __syncthreads();
  float acc[EPL];
#pragma unroll
  for (int e = 0; e < EPL; ++e) acc[e] = 0.0f;
  for (int j = warp; j < ctx; j += 4) {
    const int slot = bt[j / a.block_size] * a.block_size + j % a.block_size;
    const std::size_t base = (static_cast<std::size_t>(slot) * a.H + h) * D;
    const float p = i8 ? sc[j] * a.vscale[static_cast<std::size_t>(slot) * a.H + h] : sc[j];
#pragma unroll
    for (int e = 0; e < EPL; ++e) acc[e] += p * ldc(vc, base + lane * EPL + e);
  }
#pragma unroll
  for (int e = 0; e < EPL; ++e) accs[warp][lane * EPL + e] = acc[e];
  __syncthreads();
  A* out = static_cast<A*>(a.out);
  const float inv = 1.0f / sum;
  for (int t = threadIdx.x; t < D; t += blockDim.x) {
    const float o = ((accs[0][t] + accs[1][t]) + (accs[2][t] + accs[3][t])) * inv;
    st(out, static_cast<std::size_t>(i) * a.H * D + static_cast<std::size_t>(h) * D + t, o);
  }
}

// ---- satir toplama ------------------------------------------------------------------------------
__global__ void gather_kernel(float* dst, const float* src, const std::int32_t* idx, int rows, int cols) {
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t r = i / cols, c = i % cols;
    dst[i] = src[static_cast<std::size_t>(idx[r]) * cols + c];
  }
}

// ---- ince GEMM --------------------------------------------------------------------------------
template <typename Y>
__device__ __forceinline__ void finish(Y* y, std::size_t i, float v, const SkinnyArgs& a, int col) {
  if (a.bias != nullptr) v += a.bias[col];
  if (a.gelu) v = gelu_tanh(v);
  if (a.accumulate) {
    reinterpret_cast<float*>(y)[i] += v;
  } else {
    st(y, i, v);
  }
}

/// 16 baytlik vektor yukleme -> 8 (F16) ya da 4 (F32) eleman.
template <typename T>
struct Vec;
template <>
struct Vec<__half> {
  static constexpr int N = 8;
  static __device__ __forceinline__ void load(const __half* p, float* out) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(h[i]);
      out[2 * i] = f.x;
      out[2 * i + 1] = f.y;
    }
  }
};
template <>
struct Vec<float> {
  static constexpr int N = 4;
  static __device__ __forceinline__ void load(const float* p, float* out) {
    const float4 u = *reinterpret_cast<const float4*>(p);
    out[0] = u.x;
    out[1] = u.y;
    out[2] = u.z;
    out[3] = u.w;
  }
};

/// W [n, k]: warp basina bir cikis sutunu; lane'ler K uzerinde 16 baytlik adimlarla (birlesik), sabit kelebek
/// indirgeme. Her satirin toplama sirasi m'den bagimsiz.
template <typename X, typename W, typename Y>
__global__ void __launch_bounds__(256) skinny_nk_kernel(SkinnyArgs a) {
  constexpr int VN = Vec<W>::N;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int col = blockIdx.x * 8 + warp;
  if (col >= a.n) return;
  const X* x = static_cast<const X*>(a.x);
  const W* w = static_cast<const W*>(a.w) + static_cast<std::size_t>(col) * a.k;
  float acc[kSkinnyMaxRows];
#pragma unroll
  for (int r = 0; r < kSkinnyMaxRows; ++r) acc[r] = 0.0f;
  for (int kk = lane * VN; kk < a.k; kk += 32 * VN) {
    float wv[VN];
    Vec<W>::load(w + kk, wv);
#pragma unroll
    for (int r = 0; r < kSkinnyMaxRows; ++r) {
      if (r >= a.m) break;
      float xv[VN];
      Vec<X>::load(x + static_cast<std::size_t>(r) * a.k + kk, xv);
      float t = 0.0f;
#pragma unroll
      for (int e = 0; e < VN; ++e) t += xv[e] * wv[e];
      acc[r] += t;
    }
  }
  Y* y = static_cast<Y*>(a.y);
  const int ldy = a.ldy > 0 ? a.ldy : a.n;
#pragma unroll
  for (int r = 0; r < kSkinnyMaxRows; ++r) {
    if (r >= a.m) break;
    const float v = warp_sum(acc[r]);
    if (lane == 0) finish(y, static_cast<std::size_t>(r) * ldy + col, v, a, col);
  }
}

// ---- ornekleme ---------------------------------------------------------------------------------
__device__ __forceinline__ unsigned mulhilo(unsigned a, unsigned b, unsigned* hi) {
  const unsigned long long p = static_cast<unsigned long long>(a) * b;
  *hi = static_cast<unsigned>(p >> 32);
  return static_cast<unsigned>(p);
}

/// Philox4x32-10 (Salmon ve ark. 2011): sayac (offset, satir) + anahtar (seed) -> [0, 1) tekduze.
__device__ float philox_uniform(unsigned long long seed, unsigned long long offset, unsigned row) {
  unsigned c0 = static_cast<unsigned>(offset), c1 = static_cast<unsigned>(offset >> 32), c2 = row, c3 = 0;
  unsigned k0 = static_cast<unsigned>(seed), k1 = static_cast<unsigned>(seed >> 32);
  for (int i = 0; i < 10; ++i) {
    unsigned hi0, hi1;
    const unsigned lo0 = mulhilo(0xD2511F53u, c0, &hi0);
    const unsigned lo1 = mulhilo(0xCD9E8D57u, c2, &hi1);
    const unsigned n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
    c0 = n0;
    c1 = lo1;
    c2 = n2;
    c3 = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return static_cast<float>(c0 >> 8) * (1.0f / 16777216.0f);
}

constexpr int kSampleThreads = 1024;

__device__ __forceinline__ float penalized(float l, int c, int oc, const SampleRow& p) {
  if (c > 0 && p.repetition != 1.0f) l = l > 0.0f ? l / p.repetition : l * p.repetition;
  l -= p.frequency * static_cast<float>(oc) + (oc > 0 ? p.presence : 0.0f);
  return l;
}

__global__ void __launch_bounds__(kSampleThreads) sample_kernel(std::int32_t* out, const float* logits, float* scratch,
                                                                 int ld, int V,
                                                                 const SampleRow* rows, const std::int32_t* counts,
                                                                 const std::int32_t* out_counts) {
  __shared__ float red[32];
  __shared__ float part[kSampleThreads];
  __shared__ int sidx[32];
  const int r = blockIdx.x;
  const SampleRow p = rows[r];
  const float* lg = logits + static_cast<std::size_t>(r) * ld;
  float* s = scratch + static_cast<std::size_t>(r) * ld;
  const std::int32_t* cr = counts + static_cast<std::size_t>(p.seq) * V;
  const std::int32_t* ocr = out_counts + static_cast<std::size_t>(p.seq) * V;
  const bool greedy = p.temperature <= 0.0f;
  const float invt = greedy ? 1.0f : 1.0f / p.temperature;
  float mx = -FLT_MAX, mn = FLT_MAX;
  int amax = 0x7FFFFFFF;
  for (int j = threadIdx.x; j < V; j += blockDim.x) {
    const float v = penalized(lg[j], cr[j], ocr[j], p) * invt;
    s[j] = v;
    if (v > mx || (v == mx && j < amax)) {
      mx = v;
      amax = j;
    }
    mn = fminf(mn, v);
  }
  // blok argmax (esitlikte kucuk id)
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, mx, o);
    const int oi = __shfl_xor_sync(0xFFFFFFFFu, amax, o);
    if (ov > mx || (ov == mx && oi < amax)) {
      mx = ov;
      amax = oi;
    }
  }
  if (lane == 0) {
    red[warp] = mx;
    sidx[warp] = amax;
  }
  __syncthreads();
  if (warp == 0) {
    mx = lane < kSampleThreads / 32 ? red[lane] : -FLT_MAX;
    amax = lane < kSampleThreads / 32 ? sidx[lane] : 0x7FFFFFFF;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xFFFFFFFFu, mx, o);
      const int oi = __shfl_xor_sync(0xFFFFFFFFu, amax, o);
      if (ov > mx || (ov == mx && oi < amax)) {
        mx = ov;
        amax = oi;
      }
    }
    if (lane == 0) {
      red[0] = mx;
      sidx[0] = amax;
    }
  }
  __syncthreads();
  mx = red[0];
  amax = sidx[0];
  __syncthreads();
  if (greedy) {
    if (threadIdx.x == 0) out[r] = amax;
    return;
  }
  mn = -block_max(-mn, red);
  __syncthreads();
  float thr = -FLT_MAX;
  if (p.top_k > 0 && p.top_k < V) {  // en buyuk t: |{s >= t}| >= k  (k. en buyuk deger)
    float lo = mn, hi = mx;
    for (int it = 0; it < 48; ++it) {
      const float mid = 0.5f * (lo + hi);
      float c = 0.0f;
      for (int j = threadIdx.x; j < V; j += blockDim.x) c += s[j] >= mid ? 1.0f : 0.0f;
      c = block_sum(c, red);
      __syncthreads();
      if (c >= static_cast<float>(p.top_k)) lo = mid; else hi = mid;
    }
    thr = lo;
  }
  if (p.top_p < 1.0f) {  // top-k'dan kalanlar uzerinde: en buyuk t, kutle(s >= t) >= p * Z
    float z = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) z += s[j] >= thr ? __expf(s[j] - mx) : 0.0f;
    z = block_sum(z, red);
    __syncthreads();
    float lo = fmaxf(thr, mn), hi = mx;
    for (int it = 0; it < 48; ++it) {
      const float mid = 0.5f * (lo + hi);
      float c = 0.0f;
      for (int j = threadIdx.x; j < V; j += blockDim.x) c += (s[j] >= mid && s[j] >= thr) ? __expf(s[j] - mx) : 0.0f;
      c = block_sum(c, red);
      __syncthreads();
      if (c >= p.top_p * z) lo = mid; else hi = mid;
    }
    thr = fmaxf(thr, lo);
  }
  if (p.min_p > 0.0f) thr = fmaxf(thr, mx + __logf(p.min_p));
  // cok terimli cekilis: indeks sirasinda kume toplam (thread basina bitisik dilim)
  const int chunk = (V + kSampleThreads - 1) / kSampleThreads;
  const int b = threadIdx.x * chunk, e = min(V, b + chunk);
  float local = 0.0f;
  for (int j = b; j < e; ++j) local += s[j] >= thr ? __expf(s[j] - mx) : 0.0f;
  part[threadIdx.x] = local;
  __syncthreads();
  if (threadIdx.x == 0) {
    float acc = 0.0f;
    for (int t = 0; t < kSampleThreads; ++t) {
      const float v = part[t];
      part[t] = acc;  // dislayici on ek
      acc += v;
    }
    red[0] = acc;
  }
  __syncthreads();
  const float total = red[0];
  const float u = philox_uniform(p.seed, p.offset, static_cast<unsigned>(r)) * total;
  const float start = part[threadIdx.x];
  const bool last_thread = threadIdx.x == kSampleThreads - 1;
  if ((u >= start && u < start + local) || (last_thread && u >= start + local)) {
    float acc = start;
    int pick = -1, last = -1;
    for (int j = b; j < e; ++j) {
      if (s[j] < thr) continue;
      last = j;
      acc += __expf(s[j] - mx);
      if (u < acc) {
        pick = j;
        break;
      }
    }
    if (pick < 0) pick = last;
    if (pick >= 0) out[r] = pick;
  }
  if (threadIdx.x == 0 && total <= 0.0f) out[r] = amax;
}

__global__ void count_kernel(std::int32_t* counts, std::int32_t* out_counts, const std::int32_t* seq, const std::int32_t* tok,
                             const std::uint8_t* is_output, int n, int V, int delta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const std::size_t at = static_cast<std::size_t>(seq[i]) * V + tok[i];
  atomicAdd(counts + at, delta);
  if (is_output[i]) atomicAdd(out_counts + at, delta);
}

__global__ void __launch_bounds__(512) nll_kernel(float* out, const float* logits, int ld, int V, const std::int32_t* tg) {
  __shared__ float red[32];
  const int r = blockIdx.x;
  const float* s = logits + static_cast<std::size_t>(r) * ld;
  float m = -FLT_MAX;
  for (int j = threadIdx.x; j < V; j += blockDim.x) m = fmaxf(m, s[j]);
  m = block_max(m, red);
  __syncthreads();
  float z = 0.0f;
  for (int j = threadIdx.x; j < V; j += blockDim.x) z += __expf(s[j] - m);
  z = block_sum(z, red);
  if (threadIdx.x == 0) out[r] = __logf(z) + m - s[tg[r]];
}

// ---- nicemleme ---------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) quant_rows_kernel(std::int8_t* q, float* sx, const __half* x, const float* inv_s,
                                                          int k) {
  __shared__ float red[32];
  const int r = blockIdx.x;
  float amax = 0.0f;
  for (int j = threadIdx.x; j < k; j += blockDim.x) {
    float v = __half2float(x[static_cast<std::size_t>(r) * k + j]);
    if (inv_s) v *= inv_s[j];
    amax = fmaxf(amax, fabsf(v));
  }
  amax = block_max(amax, red);
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  for (int j = threadIdx.x; j < k; j += blockDim.x) {
    float v = __half2float(x[static_cast<std::size_t>(r) * k + j]);
    if (inv_s) v *= inv_s[j];
    q[static_cast<std::size_t>(r) * k + j] = static_cast<std::int8_t>(fmaxf(-127.0f, fminf(127.0f, rintf(v / scale))));
  }
  if (threadIdx.x == 0) sx[r] = scale;
}

__global__ void dequant_kernel(void* y, bool f32, bool accumulate, bool gelu, const std::int32_t* acc, const float* sx,
                               const float* sw, const float* bias, int m, int n) {
  const std::size_t total = static_cast<std::size_t>(m) * n;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(i / n), c = static_cast<int>(i % n);
    float v = static_cast<float>(acc[i]) * sx[r] * sw[c] + (bias ? bias[c] : 0.0f);
    if (gelu) v = gelu_tanh(v);
    if (accumulate) {
      static_cast<float*>(y)[i] += v;
    } else if (f32) {
      static_cast<float*>(y)[i] = v;
    } else {
      static_cast<__half*>(y)[i] = __float2half_rn(v);
    }
  }
}

/// W4: warp basina cikis sutunu; lane basina 8 yarim bayt (uint32) adimlariyla K.
__global__ void __launch_bounds__(256) w4_gemv_kernel(W4Args a) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int col = blockIdx.x * 8 + warp;
  if (col >= a.n) return;
  const __half* x = static_cast<const __half*>(a.x);
  const std::uint32_t* pw = reinterpret_cast<const std::uint32_t*>(a.packed + static_cast<std::size_t>(col) * (a.k / 2));
  const int groups = a.k / a.group;
  const float* sc = a.scale + static_cast<std::size_t>(col) * groups;
  const std::uint8_t* zp = a.zero + static_cast<std::size_t>(col) * groups;
  float acc[kSkinnyMaxRows];
#pragma unroll
  for (int r = 0; r < kSkinnyMaxRows; ++r) acc[r] = 0.0f;
  for (int u = lane; u < a.k / 8; u += 32) {
    const std::uint32_t word = pw[u];
    const int kk = u * 8;
    const int g = kk / a.group;
    const float s = sc[g], z = static_cast<float>(zp[g]);
    float wv[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      wv[e] = (static_cast<float>((word >> (4 * e)) & 15u) - z) * s;
      if (a.inv_s) wv[e] *= a.inv_s[kk + e];
    }
#pragma unroll
    for (int r = 0; r < kSkinnyMaxRows; ++r) {
      if (r >= a.m) break;
      const __half* xr = x + static_cast<std::size_t>(r) * a.k + kk;
      float t = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) t += __half2float(xr[e]) * wv[e];
      acc[r] += t;
    }
  }
#pragma unroll
  for (int r = 0; r < kSkinnyMaxRows; ++r) {
    if (r >= a.m) break;
    float v = warp_sum(acc[r]);
    if (lane == 0) {
      if (a.bias) v += a.bias[col];
      if (a.gelu) v = gelu_tanh(v);
      const std::size_t i = static_cast<std::size_t>(r) * a.n + col;
      if (a.accumulate) static_cast<float*>(a.y)[i] += v;
      else if (a.ytype == DType::F32) static_cast<float*>(a.y)[i] = v;
      else static_cast<__half*>(a.y)[i] = __float2half_rn(v);
    }
  }
}

__global__ void w4_dequant_kernel(__half* w, const std::uint8_t* packed, const float* scale, const std::uint8_t* zero,
                                  int group, int n, int k) {
  const std::size_t total = static_cast<std::size_t>(n) * k;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int col = static_cast<int>(i / k), kk = static_cast<int>(i % k);
    const std::uint8_t b = packed[static_cast<std::size_t>(col) * (k / 2) + kk / 2];
    const float q = static_cast<float>((kk & 1) ? (b >> 4) : (b & 15));
    const int g = kk / group, groups = k / group;
    const float v = (q - static_cast<float>(zero[static_cast<std::size_t>(col) * groups + g])) *
                    scale[static_cast<std::size_t>(col) * groups + g];
    w[static_cast<std::size_t>(kk) * n + col] = __float2half_rn(v);
  }
}

__global__ void scale_cols_kernel(__half* x, const float* inv_s, int m, int k) {
  const std::size_t total = static_cast<std::size_t>(m) * k;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    x[i] = __float2half_rn(__half2float(x[i]) * inv_s[i % k]);
  }
}

unsigned grid_for(std::size_t n, int threads) { return detail::blocks_for(n, threads); }

}  // namespace

void embed(float* x, const std::int32_t* tok, const std::int32_t* pos, const void* wte, const void* wpe, DType wtype, int n,
           int d, cudaStream_t s) {
  const unsigned g = grid_for(static_cast<std::size_t>(n) * d, 256);
  if (wtype == DType::F16) {
    embed_kernel<<<g, 256, 0, s>>>(x, tok, pos, static_cast<const __half*>(wte), static_cast<const __half*>(wpe), n, d);
  } else if (wtype == DType::F32) {
    embed_kernel<<<g, 256, 0, s>>>(x, tok, pos, static_cast<const float*>(wte), static_cast<const float*>(wpe), n, d);
  } else {
    bad_type("embed");
  }
  check_launch("embed");
}

void kv_append(void* kc, void* vc, float* kscale, float* vscale, DType cache, const void* qkv, DType act,
               const std::int32_t* slot, int n, int H, int D, cudaStream_t s) {
  if (n == 0) return;
  const unsigned g = grid_for(static_cast<std::size_t>(n) * H * D, 256);
  if (cache == DType::I8) {
    const unsigned gw = static_cast<unsigned>((static_cast<std::size_t>(n) * H * 32 + 255) / 256);
    if (D % 32 != 0 || D > 128) bad_type("kv_append INT8 D");
    if (act == DType::F16) {
      kv_append_i8_kernel<<<gw, 256, 0, s>>>(static_cast<std::int8_t*>(kc), static_cast<std::int8_t*>(vc), kscale, vscale,
                                             static_cast<const __half*>(qkv), slot, n, H, D);
    } else {
      kv_append_i8_kernel<<<gw, 256, 0, s>>>(static_cast<std::int8_t*>(kc), static_cast<std::int8_t*>(vc), kscale, vscale,
                                             static_cast<const float*>(qkv), slot, n, H, D);
    }
  } else if (cache == DType::F16 && act == DType::F16) {
    kv_append_kernel<<<g, 256, 0, s>>>(static_cast<__half*>(kc), static_cast<__half*>(vc), static_cast<const __half*>(qkv),
                                       slot, n, H, D);
  } else if (cache == DType::F32 && act == DType::F32) {
    kv_append_kernel<<<g, 256, 0, s>>>(static_cast<float*>(kc), static_cast<float*>(vc), static_cast<const float*>(qkv), slot,
                                       n, H, D);
  } else {
    bad_type("kv_append");
  }
  check_launch("kv_append");
}

void paged_attention(const PagedArgs& a, cudaStream_t s) {
  if (a.n == 0) return;
  const dim3 grid(static_cast<unsigned>(a.n), static_cast<unsigned>(a.H));
  const std::size_t smem = static_cast<std::size_t>(a.max_ctx) * sizeof(float);
  if (a.D != 64) bad_type("paged_attention: yalniz head_dim 64");
  if (a.act == DType::F16 && a.cache == DType::F16) {
    paged_kernel<__half, __half, 64><<<grid, 128, smem, s>>>(a);
  } else if (a.act == DType::F16 && a.cache == DType::I8) {
    paged_kernel<__half, std::int8_t, 64><<<grid, 128, smem, s>>>(a);
  } else if (a.act == DType::F32 && a.cache == DType::F32) {
    paged_kernel<float, float, 64><<<grid, 128, smem, s>>>(a);
  } else if (a.act == DType::F32 && a.cache == DType::I8) {
    paged_kernel<float, std::int8_t, 64><<<grid, 128, smem, s>>>(a);
  } else {
    bad_type("paged_attention");
  }
  check_launch("paged_attention");
}

void gather_rows(float* dst, const float* src, const std::int32_t* idx, int rows, int cols, cudaStream_t s) {
  if (rows == 0) return;
  gather_kernel<<<grid_for(static_cast<std::size_t>(rows) * cols, 256), 256, 0, s>>>(dst, src, idx, rows, cols);
  check_launch("gather_rows");
}

void skinny_gemm(const SkinnyArgs& a, cudaStream_t s) {
  if (a.m == 0) return;
  if (a.m > kSkinnyMaxRows) throw std::invalid_argument("Nexus infer: ince GEMM m <= 16");
  if (a.accumulate && a.ytype != DType::F32) bad_type("ince GEMM birikim F32 ister");
  const bool f16 = a.xtype == DType::F16 && a.wtype == DType::F16;
  const bool f32 = a.xtype == DType::F32 && a.wtype == DType::F32;
  if (!f16 && !f32) bad_type("ince GEMM x/w");
  if (!a.w_nk) bad_type("ince GEMM agirlik duzeni [n, k] olmali");
  if (a.k % 8 != 0) bad_type("ince GEMM k % 8");
  const unsigned g = static_cast<unsigned>((a.n + 7) / 8);
  if (f16 && a.ytype == DType::F16) skinny_nk_kernel<__half, __half, __half><<<g, 256, 0, s>>>(a);
  else if (f16) skinny_nk_kernel<__half, __half, float><<<g, 256, 0, s>>>(a);
  else skinny_nk_kernel<float, float, float><<<g, 256, 0, s>>>(a);
  check_launch("skinny_gemm");
}

void sample(std::int32_t* out, const float* logits, float* scratch, int ld, int V, const SampleRow* rows, int nrows,
            const std::int32_t* counts, const std::int32_t* out_counts, cudaStream_t s) {
  if (nrows == 0) return;
  sample_kernel<<<nrows, kSampleThreads, 0, s>>>(out, logits, scratch, ld, V, rows, counts, out_counts);
  check_launch("sample");
}

void count_tokens(std::int32_t* counts, std::int32_t* out_counts, const std::int32_t* seq, const std::int32_t* tok,
                  const std::uint8_t* is_output, int n, int V, int delta, cudaStream_t s) {
  if (n == 0) return;
  count_kernel<<<(n + 255) / 256, 256, 0, s>>>(counts, out_counts, seq, tok, is_output, n, V, delta);
  check_launch("count_tokens");
}

void row_nll(float* out, const float* logits, int ld, int V, const std::int32_t* targets, int rows, cudaStream_t s) {
  if (rows == 0) return;
  nll_kernel<<<rows, 512, 0, s>>>(out, logits, ld, V, targets);
  check_launch("row_nll");
}

void quant_rows_i8(std::int8_t* q, float* sx, const void* x, const float* inv_smooth, int m, int k, cudaStream_t s) {
  if (m == 0) return;
  quant_rows_kernel<<<m, 256, 0, s>>>(q, sx, static_cast<const __half*>(x), inv_smooth, k);
  check_launch("quant_rows_i8");
}

void dequant_i32(void* y, DType ytype, bool accumulate, bool gelu, const std::int32_t* acc, const float* sx,
                 const float* sw, const float* bias, int m, int n, cudaStream_t s) {
  if (m == 0) return;
  dequant_kernel<<<grid_for(static_cast<std::size_t>(m) * n, 256), 256, 0, s>>>(y, ytype == DType::F32, accumulate, gelu,
                                                                                acc, sx, sw, bias, m, n);
  check_launch("dequant_i32");
}

void w4_gemv(const W4Args& a, cudaStream_t s) {
  if (a.m == 0) return;
  if (a.m > kSkinnyMaxRows || a.k % 8 != 0 || a.k % a.group != 0) bad_type("w4_gemv sekli");
  w4_gemv_kernel<<<static_cast<unsigned>((a.n + 7) / 8), 256, 0, s>>>(a);
  check_launch("w4_gemv");
}

void w4_dequant(void* w_kn, const std::uint8_t* packed, const float* scale, const std::uint8_t* zero, int group, int n,
                int k, cudaStream_t s) {
  w4_dequant_kernel<<<grid_for(static_cast<std::size_t>(n) * k, 256), 256, 0, s>>>(static_cast<__half*>(w_kn), packed, scale,
                                                                                   zero, group, n, k);
  check_launch("w4_dequant");
}

void scale_cols(void* x, const float* inv_s, int m, int k, cudaStream_t s) {
  if (m == 0) return;
  scale_cols_kernel<<<grid_for(static_cast<std::size_t>(m) * k, 256), 256, 0, s>>>(static_cast<__half*>(x), inv_s, m, k);
  check_launch("scale_cols");
}

}  // namespace k
}  // namespace infer
}  // namespace cuda
}  // namespace nexus
