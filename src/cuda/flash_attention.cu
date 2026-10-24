/**
 * @file flash_attention.cu
 * @brief FAZ 5.B/5.C FlashAttention-2 ileri ve geri (Ampere, mma.sync m16n8k16, BF16/F16).
 *
 * Ileri: blok = 4 warp = 64 sorgu satiri (warp basina 16), anahtar dosemesi Bc = 64.
 * Q, K, V dosemeleri paylasimli bellekte (satir dolgusu +8 eleman: bank cakismasini azaltir).
 * S = Q K^T ve O += P V mma.sync ile; online softmax yazmaclarda (C fragman duzeni bilinir:
 * ip, satir lane/4 ve lane/4+8, sutun (lane%4)*2 + {0,1}). Skor matrisi yazilmaz, LSE yazilir.
 *
 * Geri: blok = 4 warp = 64 anahtar (warp basina 16), GQA grubu ve sorgu dosemeleri uzerinde
 * dongu. S^T = K Q^T, P^T LSE'den yeniden; dV += (P Z)^T dO, dP^T = V dO^T,
 * dS^T = P^T (dP^T Z - D), dK += dS^T Q, dQ += dS K (FP32 atomik ya da belirlenimci kismi).
 *
 * Nedensel ve pencere maskelerinde tamamen maskeli dosemeler atlanir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "common.cuh"
#include "nexus/cuda/attention.hpp"
#include "nexus/cuda/caching_allocator.hpp"

#include <cmath>
#include <type_traits>

namespace nexus {
namespace cuda {
namespace {

using detail::Num;

constexpr int kBr = 64;
constexpr int kBc = 64;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kPad = 8;

// ---- Tip ozellikleri: mma ve paketleme -------------------------------------------------
template <typename T>
struct Mma;

template <>
struct Mma<__nv_bfloat16> {
  static __device__ __forceinline__ void run(float c[4], const unsigned a[4], const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  static __device__ __forceinline__ unsigned pack(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
  }
};

template <>
struct Mma<__half> {
  static __device__ __forceinline__ void run(float c[4], const unsigned a[4], const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  static __device__ __forceinline__ unsigned pack(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
  }
};

template <typename T>
__device__ __forceinline__ unsigned load_pair(const T* p) {
  return *reinterpret_cast<const unsigned*>(p);
}

// Ayri satirlardaki iki eleman (B operandi "k" yonunde bitisik degilse).
template <typename T>
__device__ __forceinline__ unsigned load_two(const T* lo, const T* hi) {
  const unsigned short a = *reinterpret_cast<const unsigned short*>(lo);
  const unsigned short b = *reinterpret_cast<const unsigned short*>(hi);
  return static_cast<unsigned>(a) | (static_cast<unsigned>(b) << 16);
}

// ---- Philox (kernels_train.cu ile ayni; CPU fused::philox_uniform ile bit-bit) ------------
__device__ __forceinline__ uint4 philox(uint4 ctr, uint2 key) {
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    const unsigned hi0 = __umulhi(0xD2511F53u, ctr.x);
    const unsigned lo0 = 0xD2511F53u * ctr.x;
    const unsigned hi1 = __umulhi(0xCD9E8D57u, ctr.z);
    const unsigned lo1 = 0xCD9E8D57u * ctr.z;
    ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
  }
  return ctr;
}

__device__ __forceinline__ float philox_u(std::uint64_t seed, std::uint64_t offset, std::uint64_t index) {
  const std::uint64_t block = offset + index / 4;
  const uint4 r = philox(make_uint4(static_cast<unsigned>(block), static_cast<unsigned>(block >> 32), 0u, 0u),
                         make_uint2(static_cast<unsigned>(seed), static_cast<unsigned>(seed >> 32)));
  const unsigned lanes[4] = {r.x, r.y, r.z, r.w};
  return static_cast<float>(lanes[index % 4] >> 8) * (1.0f / 16777216.0f);
}

// ---- Ortak geometri ------------------------------------------------------------------------
struct DeviceParams {
  int batch, heads, kv_heads, tq, tk, group;
  float scale, softcap, dropout_p, keep_scale;
  int causal, window;
  const float* alibi;
  const std::int32_t* cu_q;
  const std::int32_t* cu_k;
  std::uint64_t seed, offset;
  long long qb, qr, qh, kb, kr, kh, ob, orow, oh;
};

struct Seq {
  int tq, tk;
  long long q0, k0;  // varlen paketli ilk satir; degilse -1
  int total_q;
};

__device__ __forceinline__ Seq seq_of(const DeviceParams& p, int b) {
  Seq s;
  if (p.cu_q != nullptr) {
    s.q0 = p.cu_q[b];
    s.k0 = p.cu_k[b];
    s.tq = p.cu_q[b + 1] - p.cu_q[b];
    s.tk = p.cu_k[b + 1] - p.cu_k[b];
    s.total_q = p.cu_q[p.batch];
  } else {
    s.q0 = -1;
    s.k0 = -1;
    s.tq = p.tq;
    s.tk = p.tk;
    s.total_q = p.batch * p.tq;
  }
  return s;
}

__device__ __forceinline__ long long q_off(const DeviceParams& p, const Seq& s, int b, int h, int i) {
  return (s.q0 >= 0 ? (s.q0 + i) * p.qr : b * p.qb + i * p.qr) + h * p.qh;
}
__device__ __forceinline__ long long o_off(const DeviceParams& p, const Seq& s, int b, int h, int i) {
  return (s.q0 >= 0 ? (s.q0 + i) * p.orow : b * p.ob + i * p.orow) + h * p.oh;
}
__device__ __forceinline__ long long k_off(const DeviceParams& p, const Seq& s, int b, int hk, int j) {
  return (s.k0 >= 0 ? (s.k0 + j) * p.kr : b * p.kb + j * p.kr) + hk * p.kh;
}
__device__ __forceinline__ long long lse_idx(const DeviceParams& p, const Seq& s, int b, int h, int i) {
  return s.q0 >= 0 ? static_cast<long long>(h) * s.total_q + s.q0 + i
                   : (static_cast<long long>(b) * p.heads + h) * p.tq + i;
}
__device__ __forceinline__ std::uint64_t drop_idx(const DeviceParams& p, const Seq& s, int b, int h, int i, int j) {
  if (s.q0 >= 0) {
    return (static_cast<std::uint64_t>(h) * s.total_q + static_cast<std::uint64_t>(s.q0 + i)) * p.tk + j;
  }
  return ((static_cast<std::uint64_t>(b) * p.heads + h) * p.tq + i) * p.tk + j;
}

__device__ __forceinline__ bool allowed(const DeviceParams& p, int qpos, int j, int tk) {
  if (j >= tk) {
    return false;
  }
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

// Ham skor -> softcap + alibi; t = tanh degeri (softcap turevi icin).
__device__ __forceinline__ float transform(const DeviceParams& p, float raw, int h, int qpos, int j, float* t) {
  float s = raw;
  *t = 0.0f;
  if (p.softcap > 0.0f) {
    *t = tanhf(s / p.softcap);
    s = p.softcap * *t;
  }
  if (p.alibi != nullptr) {
    s += p.alibi[h] * static_cast<float>(j - qpos);
  }
  return s;
}

// Sorgu satir araligi [i0, i1) icin gorunur anahtar araligi [j0, j1).
__device__ __forceinline__ void key_range(const DeviceParams& p, int i0, int i1, int tq, int tk, int* j0, int* j1) {
  const int pos_lo = tk - tq + i0;
  const int pos_hi = tk - tq + (i1 - 1);
  *j1 = p.causal ? min(tk, pos_hi + 1) : tk;
  if (!p.causal && p.window > 0) {
    *j1 = min(tk, pos_hi + p.window);
  }
  *j0 = p.window > 0 ? max(0, pos_lo - p.window + 1) : 0;
  if (*j1 < 0) {
    *j1 = 0;
  }
}

// Global -> paylasimli dosema yukleme: rows satir, D sutun; sinir disi sifir. 16 bayt vektor.
template <typename T, int D>
__device__ __forceinline__ void load_tile(T* smem, const T* base, long long row_stride, int rows_valid, int rows) {
  constexpr int kVec = 8;  // 8 x 16 bit = 16 bayt
  constexpr int kPerRow = D / kVec;
  for (int idx = threadIdx.x; idx < rows * kPerRow; idx += kThreads) {
    const int r = idx / kPerRow;
    const int c = (idx % kPerRow) * kVec;
    uint4 val = make_uint4(0u, 0u, 0u, 0u);
    if (r < rows_valid) {
      val = *reinterpret_cast<const uint4*>(base + r * row_stride + c);
    }
    *reinterpret_cast<uint4*>(smem + r * (D + kPad) + c) = val;
  }
}

// ================================ Ileri ============================================
template <typename T, int D>
__global__ void __launch_bounds__(kThreads) flash_fwd_kernel(T* __restrict__ out, float* __restrict__ lse,
                                                             const T* __restrict__ q, const T* __restrict__ k,
                                                             const T* __restrict__ v, DeviceParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* Qs = reinterpret_cast<T*>(smem_raw);
  T* Ks = Qs + kBr * (D + kPad);
  T* Vs = Ks + kBc * (D + kPad);

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int hk = h / p.group;
  const Seq s = seq_of(p, b);
  const int i_block = blockIdx.x * kBr;
  if (i_block >= s.tq) {
    return;
  }
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane >> 2;
  const int t4 = lane & 3;

  load_tile<T, D>(Qs, q + q_off(p, s, b, h, i_block), p.qr, min(kBr, s.tq - i_block), kBr);
  __syncthreads();

  // Q fragmanlari (warp'in 16 satiri) yazmaclarda.
  unsigned qa[D / 16][4];
  const T* qrow0 = Qs + (warp * 16 + g) * (D + kPad);
  const T* qrow1 = qrow0 + 8 * (D + kPad);
#pragma unroll
  for (int ks = 0; ks < D / 16; ++ks) {
    qa[ks][0] = load_pair(qrow0 + ks * 16 + t4 * 2);
    qa[ks][1] = load_pair(qrow1 + ks * 16 + t4 * 2);
    qa[ks][2] = load_pair(qrow0 + ks * 16 + t4 * 2 + 8);
    qa[ks][3] = load_pair(qrow1 + ks * 16 + t4 * 2 + 8);
  }

  float o_acc[D / 8][4];
#pragma unroll
  for (int dn = 0; dn < D / 8; ++dn) {
    o_acc[dn][0] = o_acc[dn][1] = o_acc[dn][2] = o_acc[dn][3] = 0.0f;
  }
  float m_row[2] = {-INFINITY, -INFINITY};
  float l_row[2] = {0.0f, 0.0f};
  const int qi[2] = {i_block + warp * 16 + g, i_block + warp * 16 + g + 8};
  const int qpos[2] = {s.tk - s.tq + qi[0], s.tk - s.tq + qi[1]};

  int j_begin = 0;
  int j_end = 0;
  key_range(p, i_block, min(s.tq, i_block + kBr), s.tq, s.tk, &j_begin, &j_end);
  j_begin = (j_begin / kBc) * kBc;

  for (int j_tile = j_begin; j_tile < j_end; j_tile += kBc) {
    __syncthreads();
    load_tile<T, D>(Ks, k + k_off(p, s, b, hk, j_tile), p.kr, min(kBc, s.tk - j_tile), kBc);
    load_tile<T, D>(Vs, v + k_off(p, s, b, hk, j_tile), p.kr, min(kBc, s.tk - j_tile), kBc);
    __syncthreads();

    float sc[kBc / 8][4];
#pragma unroll
    for (int nt = 0; nt < kBc / 8; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.0f;
      const T* krow = Ks + (nt * 8 + g) * (D + kPad);
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        const unsigned bfrag[2] = {load_pair(krow + ks * 16 + t4 * 2), load_pair(krow + ks * 16 + t4 * 2 + 8)};
        Mma<T>::run(sc[nt], qa[ks], bfrag);
      }
    }
    // Olcek, softcap, alibi, maske; satir maksimumu.
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < kBc / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = e / 2;
        const int j = j_tile + nt * 8 + t4 * 2 + (e & 1);
        float tv = 0.0f;
        float val = -INFINITY;
        if (qi[r] < s.tq && allowed(p, qpos[r], j, s.tk)) {
          val = transform(p, sc[nt][e] * p.scale, h, qpos[r], j, &tv);
        }
        sc[nt][e] = val;
        tile_max[r] = fmaxf(tile_max[r], val);
      }
    }
    float alpha[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xFFFFFFFFu, tile_max[r], 1));
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xFFFFFFFFu, tile_max[r], 2));
      const float m_new = fmaxf(m_row[r], tile_max[r]);
      const float m_use = m_new == -INFINITY ? 0.0f : m_new;
      alpha[r] = __expf(m_row[r] - m_use);
      m_row[r] = m_new;
      l_row[r] *= alpha[r];
#pragma unroll
      for (int nt = 0; nt < kBc / 8; ++nt) {
        sc[nt][2 * r] = __expf(sc[nt][2 * r] - m_use);
        sc[nt][2 * r + 1] = __expf(sc[nt][2 * r + 1] - m_use);
        l_row[r] += sc[nt][2 * r] + sc[nt][2 * r + 1];
      }
    }
#pragma unroll
    for (int dn = 0; dn < D / 8; ++dn) {
      o_acc[dn][0] *= alpha[0];
      o_acc[dn][1] *= alpha[0];
      o_acc[dn][2] *= alpha[1];
      o_acc[dn][3] *= alpha[1];
    }
    if (p.dropout_p > 0.0f) {
#pragma unroll
      for (int nt = 0; nt < kBc / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = e / 2;
          const int j = j_tile + nt * 8 + t4 * 2 + (e & 1);
          if (qi[r] < s.tq && j < s.tk) {
            const bool drop = philox_u(p.seed, p.offset, drop_idx(p, s, b, h, qi[r], j)) < p.dropout_p;
            sc[nt][e] = drop ? 0.0f : sc[nt][e] * p.keep_scale;
          }
        }
      }
    }
    // O += P V. P'nin C duzeni dogrudan A fragmanina doner (iki komsu n-dosemesi = 16 anahtar).
#pragma unroll
    for (int kk = 0; kk < kBc / 16; ++kk) {
      const unsigned pa[4] = {Mma<T>::pack(sc[2 * kk][0], sc[2 * kk][1]), Mma<T>::pack(sc[2 * kk][2], sc[2 * kk][3]),
                              Mma<T>::pack(sc[2 * kk + 1][0], sc[2 * kk + 1][1]),
                              Mma<T>::pack(sc[2 * kk + 1][2], sc[2 * kk + 1][3])};
      const T* v0 = Vs + (kk * 16 + t4 * 2) * (D + kPad);
#pragma unroll
      for (int dn = 0; dn < D / 8; ++dn) {
        const int d = dn * 8 + g;
        const unsigned bfrag[2] = {load_two(v0 + d, v0 + (D + kPad) + d),
                                   load_two(v0 + 8 * (D + kPad) + d, v0 + 9 * (D + kPad) + d)};
        Mma<T>::run(o_acc[dn], pa, bfrag);
      }
    }
  }
  // Satir toplamlari dortlu grupta birlesir; O /= l, LSE yazilir.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l_row[r] += __shfl_xor_sync(0xFFFFFFFFu, l_row[r], 1);
    l_row[r] += __shfl_xor_sync(0xFFFFFFFFu, l_row[r], 2);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    if (qi[r] >= s.tq) {
      continue;
    }
    const bool empty = !(l_row[r] > 0.0f);
    const float inv = empty ? 0.0f : 1.0f / l_row[r];
    T* orow = out + o_off(p, s, b, h, qi[r]);
#pragma unroll
    for (int dn = 0; dn < D / 8; ++dn) {
      *reinterpret_cast<unsigned*>(orow + dn * 8 + t4 * 2) =
          Mma<T>::pack(o_acc[dn][2 * r] * inv, o_acc[dn][2 * r + 1] * inv);
    }
    if (t4 == 0) {
      lse[lse_idx(p, s, b, h, qi[r])] = empty ? -INFINITY : m_row[r] + __logf(l_row[r]);
    }
  }
}

// ================================ Geri ==============================================
// D_i = rowsum(dO o O) (FP32), dQ biriktiricisi sifirlanir.
template <typename T>
__global__ void flash_bwd_pre_kernel(float* __restrict__ dvec, float* __restrict__ dq_acc,
                                     const T* __restrict__ dout, const T* __restrict__ o, DeviceParams p, int D,
                                     long long dq_acc_elems) {
  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const Seq s = seq_of(p, b);
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < s.tq) {
    const long long off = o_off(p, s, b, h, i);
    float acc = 0.0f;
    for (int d = threadIdx.x; d < D; d += 32) {
      acc += Num<T>::load(dout[off + d]) * Num<T>::load(o[off + d]);
    }
    acc = detail::warp_sum(acc);
    const long long li = lse_idx(p, s, b, h, i);
    if (threadIdx.x == 0) {
      dvec[li] = acc;
    }
    for (int d = threadIdx.x; d < D; d += 32) {
      dq_acc[li * D + d] = 0.0f;
    }
  }
  (void)dq_acc_elems;
}

template <typename T, int D>
__global__ void __launch_bounds__(kThreads) flash_bwd_kernel(T* __restrict__ dk, T* __restrict__ dv,
                                                             float* __restrict__ dq_acc, float* __restrict__ dq_part,
                                                             const T* __restrict__ dout, const T* __restrict__ q,
                                                             const T* __restrict__ k, const T* __restrict__ v,
                                                             const float* __restrict__ lse,
                                                             const float* __restrict__ dvec, DeviceParams p,
                                                             long long dq_elems) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* Ks = reinterpret_cast<T*>(smem_raw);
  T* Vs = Ks + kBc * (D + kPad);
  T* Qs = Vs + kBc * (D + kPad);
  T* dOs = Qs + kBr * (D + kPad);
  T* dSt = dOs + kBr * (D + kPad);  // [kBc][kBr + kPad]
  float* lse_s = reinterpret_cast<float*>(dSt + kBc * (kBr + kPad));
  float* dvec_s = lse_s + kBr;

  const int b = blockIdx.z;
  const int hk = blockIdx.y;
  const Seq s = seq_of(p, b);
  const int j_block = blockIdx.x * kBc;
  if (j_block >= s.tk) {
    return;
  }
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane >> 2;
  const int t4 = lane & 3;

  load_tile<T, D>(Ks, k + k_off(p, s, b, hk, j_block), p.kr, min(kBc, s.tk - j_block), kBc);
  load_tile<T, D>(Vs, v + k_off(p, s, b, hk, j_block), p.kr, min(kBc, s.tk - j_block), kBc);
  __syncthreads();
  // Warp'in 16 anahtarinin K ve V A-fragmanlari.
  unsigned ka[D / 16][4];
  unsigned va[D / 16][4];
  {
    const T* k0 = Ks + (warp * 16 + g) * (D + kPad);
    const T* v0 = Vs + (warp * 16 + g) * (D + kPad);
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      ka[ks][0] = load_pair(k0 + ks * 16 + t4 * 2);
      ka[ks][1] = load_pair(k0 + 8 * (D + kPad) + ks * 16 + t4 * 2);
      ka[ks][2] = load_pair(k0 + ks * 16 + t4 * 2 + 8);
      ka[ks][3] = load_pair(k0 + 8 * (D + kPad) + ks * 16 + t4 * 2 + 8);
      va[ks][0] = load_pair(v0 + ks * 16 + t4 * 2);
      va[ks][1] = load_pair(v0 + 8 * (D + kPad) + ks * 16 + t4 * 2);
      va[ks][2] = load_pair(v0 + ks * 16 + t4 * 2 + 8);
      va[ks][3] = load_pair(v0 + 8 * (D + kPad) + ks * 16 + t4 * 2 + 8);
    }
  }
  float dk_acc[D / 8][4];
  float dv_acc[D / 8][4];
#pragma unroll
  for (int dn = 0; dn < D / 8; ++dn) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      dk_acc[dn][e] = 0.0f;
      dv_acc[dn][e] = 0.0f;
    }
  }
  const int kj[2] = {j_block + warp * 16 + g, j_block + warp * 16 + g + 8};

  // Bu anahtar dosemesini goren sorgu araligi.
  const int k_lo = j_block;
  const int k_hi = min(s.tk, j_block + kBc) - 1;
  int i_begin = 0;
  int i_end = s.tq;
  if (p.causal) {
    i_begin = max(0, k_lo - (s.tk - s.tq));  // q_pos >= k_lo
  }
  if (p.window > 0) {
    i_end = min(s.tq, k_hi + p.window - (s.tk - s.tq));  // q_pos - k_hi < window
    if (!p.causal) {
      i_begin = max(0, k_lo - p.window + 1 - (s.tk - s.tq));
    }
  }
  i_begin = (i_begin / kBr) * kBr;

  for (int h = hk * p.group; h < (hk + 1) * p.group; ++h) {
    for (int i_tile = i_begin; i_tile < i_end; i_tile += kBr) {
      __syncthreads();
      const int rows = min(kBr, s.tq - i_tile);
      load_tile<T, D>(Qs, q + q_off(p, s, b, h, i_tile), p.qr, rows, kBr);
      load_tile<T, D>(dOs, dout + o_off(p, s, b, h, i_tile), p.orow, rows, kBr);
      for (int r = threadIdx.x; r < kBr; r += kThreads) {
        const bool ok = r < rows;
        lse_s[r] = ok ? lse[lse_idx(p, s, b, h, i_tile + r)] : -INFINITY;
        dvec_s[r] = ok ? dvec[lse_idx(p, s, b, h, i_tile + r)] : 0.0f;
      }
      __syncthreads();

      // S^T = K Q^T  (16 anahtar x 64 sorgu)
      float st[kBr / 8][4];
      float dpt[kBr / 8][4];
#pragma unroll
      for (int nt = 0; nt < kBr / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          st[nt][e] = 0.0f;
          dpt[nt][e] = 0.0f;
        }
        const T* qrow = Qs + (nt * 8 + g) * (D + kPad);
        const T* dorow = dOs + (nt * 8 + g) * (D + kPad);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          const unsigned bq[2] = {load_pair(qrow + ks * 16 + t4 * 2), load_pair(qrow + ks * 16 + t4 * 2 + 8)};
          Mma<T>::run(st[nt], ka[ks], bq);
          const unsigned bd[2] = {load_pair(dorow + ks * 16 + t4 * 2), load_pair(dorow + ks * 16 + t4 * 2 + 8)};
          Mma<T>::run(dpt[nt], va[ks], bd);  // dP^T = V dO^T
        }
      }
      // P^T, Z, dS^T
#pragma unroll
      for (int nt = 0; nt < kBr / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = e / 2;  // anahtar satiri: kj[r]
          const int c = nt * 8 + t4 * 2 + (e & 1);
          const int i = i_tile + c;
          const int qpos = s.tk - s.tq + i;
          float pz = 0.0f;
          float ds = 0.0f;
          const float row_lse = lse_s[c];
          if (i < s.tq && row_lse != -INFINITY && allowed(p, qpos, kj[r], s.tk)) {
            float tv = 0.0f;
            const float sv = transform(p, st[nt][e] * p.scale, h, qpos, kj[r], &tv);
            const float pij = __expf(sv - row_lse);
            float z = 1.0f;
            if (p.dropout_p > 0.0f) {
              z = philox_u(p.seed, p.offset, drop_idx(p, s, b, h, i, kj[r])) < p.dropout_p ? 0.0f : p.keep_scale;
            }
            pz = pij * z;
            ds = pij * (dpt[nt][e] * z - dvec_s[c]);
            if (p.softcap > 0.0f) {
              ds *= 1.0f - tv * tv;
            }
            ds *= p.scale;
          }
          st[nt][e] = pz;
          dpt[nt][e] = ds;
        }
      }
      // dS^T'yi paylasimliya yaz (dQ icin), dV += PZ^T dO, dK += dS^T Q.
#pragma unroll
      for (int nt = 0; nt < kBr / 8; ++nt) {
        T* row0 = dSt + (warp * 16 + g) * (kBr + kPad) + nt * 8 + t4 * 2;
        T* row1 = row0 + 8 * (kBr + kPad);
        *reinterpret_cast<unsigned*>(row0) = Mma<T>::pack(dpt[nt][0], dpt[nt][1]);
        *reinterpret_cast<unsigned*>(row1) = Mma<T>::pack(dpt[nt][2], dpt[nt][3]);
      }
#pragma unroll
      for (int kk = 0; kk < kBr / 16; ++kk) {
        const unsigned pa[4] = {Mma<T>::pack(st[2 * kk][0], st[2 * kk][1]), Mma<T>::pack(st[2 * kk][2], st[2 * kk][3]),
                                Mma<T>::pack(st[2 * kk + 1][0], st[2 * kk + 1][1]),
                                Mma<T>::pack(st[2 * kk + 1][2], st[2 * kk + 1][3])};
        const unsigned da[4] = {Mma<T>::pack(dpt[2 * kk][0], dpt[2 * kk][1]),
                                Mma<T>::pack(dpt[2 * kk][2], dpt[2 * kk][3]),
                                Mma<T>::pack(dpt[2 * kk + 1][0], dpt[2 * kk + 1][1]),
                                Mma<T>::pack(dpt[2 * kk + 1][2], dpt[2 * kk + 1][3])};
        const T* do0 = dOs + (kk * 16 + t4 * 2) * (D + kPad);
        const T* q0 = Qs + (kk * 16 + t4 * 2) * (D + kPad);
#pragma unroll
        for (int dn = 0; dn < D / 8; ++dn) {
          const int d = dn * 8 + g;
          const unsigned bdo[2] = {load_two(do0 + d, do0 + (D + kPad) + d),
                                   load_two(do0 + 8 * (D + kPad) + d, do0 + 9 * (D + kPad) + d)};
          Mma<T>::run(dv_acc[dn], pa, bdo);
          const unsigned bqv[2] = {load_two(q0 + d, q0 + (D + kPad) + d),
                                   load_two(q0 + 8 * (D + kPad) + d, q0 + 9 * (D + kPad) + d)};
          Mma<T>::run(dk_acc[dn], da, bqv);
        }
      }
      __syncthreads();
      // dQ (64 sorgu x D) += dS (64 x 64 anahtar) K : warp basina 16 sorgu satiri.
      {
        float dq_frag[D / 8][4];
#pragma unroll
        for (int dn = 0; dn < D / 8; ++dn) {
          dq_frag[dn][0] = dq_frag[dn][1] = dq_frag[dn][2] = dq_frag[dn][3] = 0.0f;
        }
        const int qr0 = warp * 16 + g;
#pragma unroll
        for (int kk = 0; kk < kBc / 16; ++kk) {
          // A[row q][col key] = dSt[key][q]
          const int key0 = kk * 16 + t4 * 2;
          const unsigned aq[4] = {
              load_two(dSt + key0 * (kBr + kPad) + qr0, dSt + (key0 + 1) * (kBr + kPad) + qr0),
              load_two(dSt + key0 * (kBr + kPad) + qr0 + 8, dSt + (key0 + 1) * (kBr + kPad) + qr0 + 8),
              load_two(dSt + (key0 + 8) * (kBr + kPad) + qr0, dSt + (key0 + 9) * (kBr + kPad) + qr0),
              load_two(dSt + (key0 + 8) * (kBr + kPad) + qr0 + 8, dSt + (key0 + 9) * (kBr + kPad) + qr0 + 8)};
          const T* kt0 = Ks + key0 * (D + kPad);
#pragma unroll
          for (int dn = 0; dn < D / 8; ++dn) {
            const int d = dn * 8 + g;
            const unsigned bk[2] = {load_two(kt0 + d, kt0 + (D + kPad) + d),
                                    load_two(kt0 + 8 * (D + kPad) + d, kt0 + 9 * (D + kPad) + d)};
            Mma<T>::run(dq_frag[dn], aq, bk);
          }
        }
        const int rows_q[2] = {i_tile + qr0, i_tile + qr0 + 8};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          if (rows_q[r] >= s.tq) {
            continue;
          }
          const long long base = lse_idx(p, s, b, h, rows_q[r]) * D;
#pragma unroll
          for (int dn = 0; dn < D / 8; ++dn) {
            const int d = dn * 8 + t4 * 2;
            if (dq_part != nullptr) {
              // Belirlenimci: anahtar dosemesi basina ayri dilim, sabit sirali son toplam.
              float* dst = dq_part + static_cast<long long>(blockIdx.x) * dq_elems + base + d;
              dst[0] += dq_frag[dn][2 * r];
              dst[1] += dq_frag[dn][2 * r + 1];
            } else {
              atomicAdd(dq_acc + base + d, dq_frag[dn][2 * r]);
              atomicAdd(dq_acc + base + d + 1, dq_frag[dn][2 * r + 1]);
            }
          }
        }
      }
    }
  }
  // dK, dV yaz.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    if (kj[r] >= s.tk) {
      continue;
    }
    T* dkrow = dk + k_off(p, s, b, hk, kj[r]);
    T* dvrow = dv + k_off(p, s, b, hk, kj[r]);
#pragma unroll
    for (int dn = 0; dn < D / 8; ++dn) {
      *reinterpret_cast<unsigned*>(dkrow + dn * 8 + t4 * 2) = Mma<T>::pack(dk_acc[dn][2 * r], dk_acc[dn][2 * r + 1]);
      *reinterpret_cast<unsigned*>(dvrow + dn * 8 + t4 * 2) = Mma<T>::pack(dv_acc[dn][2 * r], dv_acc[dn][2 * r + 1]);
    }
  }
}

// dQ biriktiricisi (veya kismi dilimlerin sabit sirali toplami) -> dQ (giris tipinde, q adimlariyla).
template <typename T>
__global__ void flash_bwd_post_kernel(T* __restrict__ dq, const float* __restrict__ dq_acc,
                                      const float* __restrict__ dq_part, int parts, long long dq_elems,
                                      DeviceParams p, int D) {
  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const Seq s = seq_of(p, b);
  const int i = blockIdx.x;
  if (i >= s.tq) {
    return;
  }
  const long long base = lse_idx(p, s, b, h, i) * D;
  T* row = dq + q_off(p, s, b, h, i);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc;
    if (dq_part != nullptr) {
      acc = 0.0f;
      for (int part = 0; part < parts; ++part) {
        acc += dq_part[static_cast<long long>(part) * dq_elems + base + d];
      }
    } else {
      acc = dq_acc[base + d];
    }
    row[d] = Num<T>::store(acc);
  }
}

// ---- Yardimcilar ----------------------------------------------------------------------------
DeviceParams to_device(const AttentionParams& a) {
  DeviceParams p{};
  p.batch = a.batch;
  p.heads = a.heads;
  p.kv_heads = a.kv_heads;
  p.tq = a.tq;
  p.tk = a.tk;
  p.group = a.heads / a.kv_heads;
  p.scale = a.effective_scale();
  p.softcap = a.softcap;
  p.dropout_p = a.dropout_p;
  p.keep_scale = 1.0f / (1.0f - a.dropout_p);
  p.causal = a.causal ? 1 : 0;
  p.window = a.window;
  p.alibi = a.alibi_slopes;
  p.cu_q = a.cu_seqlens_q;
  p.cu_k = a.cu_seqlens_k;
  p.seed = a.seed;
  p.offset = a.offset;
  p.qb = a.q_batch_stride;
  p.qr = a.q_row_stride;
  p.qh = a.q_head_stride;
  p.kb = a.k_batch_stride;
  p.kr = a.k_row_stride;
  p.kh = a.k_head_stride;
  p.ob = a.o_batch_stride;
  p.orow = a.o_row_stride;
  p.oh = a.o_head_stride;
  return p;
}

std::size_t fwd_smem(int d) { return static_cast<std::size_t>(kBr + 2 * kBc) * (d + kPad) * 2; }
std::size_t bwd_smem(int d) {
  return static_cast<std::size_t>(2 * kBc + 2 * kBr) * (d + kPad) * 2 + static_cast<std::size_t>(kBc) * (kBr + kPad) * 2 +
         2 * kBr * sizeof(float);
}

long long total_q_rows(const AttentionParams& p, cudaStream_t stream) {
  if (p.cu_seqlens_q == nullptr) {
    return static_cast<long long>(p.batch) * p.tq;
  }
  std::int32_t total = 0;
  check(cudaMemcpyAsync(&total, p.cu_seqlens_q + p.batch, sizeof(total), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(cu_seqlens_q)");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  return total;
}

template <typename F>
void dispatch_flash(const AttentionParams& p, F&& f) {
  detail::dispatch_storage(p.dtype, "flash_attention", [&](auto tag) {
    using T = decltype(tag);
    if constexpr (std::is_same_v<T, float>) {
      throw TensorError("Nexus flash_attention: F32 desteklenmez (BF16/F16)");
    } else if (p.head_dim == 64) {
      f(T{}, std::integral_constant<int, 64>{});
    } else if (p.head_dim == 128) {
      f(T{}, std::integral_constant<int, 128>{});
    } else {
      throw TensorError("Nexus flash_attention: head_dim 64 veya 128 olmali");
    }
  });
}

}  // namespace

bool flash_supported(const AttentionParams& p, std::string* reason) {
  auto fail = [reason](const char* why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return false;
  };
  if (p.dtype != DType::BF16 && p.dtype != DType::F16) {
    return fail("dtype BF16/F16 degil");
  }
  if (p.head_dim != 64 && p.head_dim != 128) {
    return fail("head_dim 64/128 degil");
  }
  const long long strides[] = {p.q_row_stride, p.q_head_stride, p.q_batch_stride, p.k_row_stride, p.k_head_stride,
                               p.k_batch_stride, p.o_row_stride, p.o_head_stride, p.o_batch_stride};
  for (const long long s : strides) {
    if (s % 8 != 0) {
      return fail("adimlar 8 elemanin kati degil (16 bayt vektor yukleme)");
    }
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device());
  if (major < 8) {
    return fail("sm_80+ gerekli (mma.sync BF16)");
  }
  return true;
}

void flash_attention_forward(void* o, float* lse, const void* q, const void* k, const void* v,
                             const AttentionParams& a, cudaStream_t stream) {
  a.validate();
  std::string why;
  if (!flash_supported(a, &why)) {
    throw TensorError("Nexus flash_attention_forward: " + why);
  }
  const DeviceParams p = to_device(a);
  dispatch_flash(a, [&](auto tag, auto dim) {
    using T = decltype(tag);
    constexpr int D = decltype(dim)::value;
    const std::size_t smem = fwd_smem(D);
    check(cudaFuncSetAttribute(flash_fwd_kernel<T, D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem)),
          "cudaFuncSetAttribute(flash_fwd)");
    const dim3 grid((a.tq + kBr - 1) / kBr, a.heads, a.batch);
    flash_fwd_kernel<T, D><<<grid, kThreads, smem, stream>>>(static_cast<T*>(o), lse, static_cast<const T*>(q),
                                                              static_cast<const T*>(k), static_cast<const T*>(v), p);
  });
  detail::check_launch("flash_fwd_kernel");
}

std::size_t flash_attention_backward_scratch_bytes(const AttentionParams& p) {
  // D vektoru + dQ FP32 biriktirici; belirlenimci modda anahtar dosemesi basina dQ dilimi.
  // Varlen'de azami (batch * tq) satir varsayilir.
  const std::size_t rows = static_cast<std::size_t>(p.batch) * p.tq * p.heads;
  const std::size_t dq = rows * p.head_dim;
  const std::size_t parts = p.deterministic ? static_cast<std::size_t>((p.tk + kBc - 1) / kBc) : 0;
  return sizeof(float) * (rows + dq + parts * dq) + 512;
}

void flash_attention_backward(void* dq, void* dk, void* dv, const void* dout, const void* q, const void* k,
                              const void* v, const void* o, const float* lse, const AttentionParams& a, void* scratch,
                              cudaStream_t stream) {
  a.validate();
  std::string why;
  if (!flash_supported(a, &why)) {
    throw TensorError("Nexus flash_attention_backward: " + why);
  }
  if (scratch == nullptr) {
    throw TensorError("Nexus flash_attention_backward: scratch gerekli");
  }
  const DeviceParams p = to_device(a);
  const std::size_t rows = static_cast<std::size_t>(a.batch) * a.tq * a.heads;
  const long long dq_elems = static_cast<long long>(rows) * a.head_dim;
  auto* dvec = static_cast<float*>(scratch);
  float* dq_acc = dvec + rows;
  float* dq_part = a.deterministic ? dq_acc + dq_elems : nullptr;
  const int kblocks = (a.tk + kBc - 1) / kBc;
  (void)total_q_rows;
  if (dq_part != nullptr) {
    check(cudaMemsetAsync(dq_part, 0, sizeof(float) * static_cast<std::size_t>(kblocks) * dq_elems, stream),
          "cudaMemsetAsync(dq_part)");
  }
  dispatch_flash(a, [&](auto tag, auto dim) {
    using T = decltype(tag);
    constexpr int D = decltype(dim)::value;
    const dim3 pre_block(32, 8);
    const dim3 pre_grid((a.tq + 7) / 8, a.heads, a.batch);
    flash_bwd_pre_kernel<T><<<pre_grid, pre_block, 0, stream>>>(dvec, dq_acc, static_cast<const T*>(dout),
                                                               static_cast<const T*>(o), p, D, dq_elems);
    detail::check_launch("flash_bwd_pre_kernel");
    const std::size_t smem = bwd_smem(D);
    check(cudaFuncSetAttribute(flash_bwd_kernel<T, D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem)),
          "cudaFuncSetAttribute(flash_bwd)");
    const dim3 grid(kblocks, a.kv_heads, a.batch);
    flash_bwd_kernel<T, D><<<grid, kThreads, smem, stream>>>(
        static_cast<T*>(dk), static_cast<T*>(dv), dq_acc, dq_part, static_cast<const T*>(dout),
        static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v), lse, dvec, p, dq_elems);
    detail::check_launch("flash_bwd_kernel");
    const dim3 post_grid(a.tq, a.heads, a.batch);
    flash_bwd_post_kernel<T><<<post_grid, 64, 0, stream>>>(static_cast<T*>(dq), dq_acc, dq_part, kblocks, dq_elems,
                                                           p, D);
    detail::check_launch("flash_bwd_post_kernel");
  });
}

}  // namespace cuda
}  // namespace nexus
