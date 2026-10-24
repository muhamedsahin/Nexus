/**
 * @file kernels_elementwise.cu
 * @brief FAZ 3.3 nedensel softmax (maske cekirdek icinde) + dikkat yardimcilari,
 *        GELU (tanh), eleman eleman toplama ve belirlenimci toplam.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "common.cuh"
#include "nexus/core/error.hpp"
#include "nexus/cuda/kernels.hpp"

namespace nexus {
namespace cuda {
namespace {

using detail::kWarp;

constexpr int kRowThreads = 256;
constexpr int kRowsPerBlock = kRowThreads / kWarp;
constexpr float kGeluC = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float kGeluA = 0.044715f;

#define NEXUS_GRID_STRIDE(i, n)                                                                         \
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < (n); \
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)

// Satir basina bir warp. Satir r'nin sorgu konumu i = r % t; j > i maskelenir.
__global__ void softmax_causal_forward_kernel(float* __restrict__ probs, const float* __restrict__ scores, int rows,
                                              int t, float scale) {
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x % kWarp;
  const int i = row % t;
  const float* s = scores + static_cast<std::size_t>(row) * t;
  float* p = probs + static_cast<std::size_t>(row) * t;
  float maxv = -INFINITY;
  for (int j = lane; j <= i; j += kWarp) {
    maxv = fmaxf(maxv, s[j] * scale);
  }
  maxv = detail::warp_max(maxv);
  float total = 0.0f;
  for (int j = lane; j <= i; j += kWarp) {
    total += expf(s[j] * scale - maxv);
  }
  total = detail::warp_sum(total);
  const float inv = 1.0f / total;
  for (int j = lane; j < t; j += kWarp) {
    p[j] = j <= i ? expf(s[j] * scale - maxv) * inv : 0.0f;
  }
}

// dscores ve dprobs ayni tampon olabilir (yerinde): bu yuzden __restrict__ yok. Her satiri tek
// warp isler; nokta carpimi tum okumalar bittikten sonra (warp_sum) yazmaya gecilir.
__global__ void softmax_causal_backward_kernel(float* dscores, const float* dprobs, const float* __restrict__ probs,
                                               int rows, int t, float scale) {
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x % kWarp;
  const int i = row % t;
  const std::size_t base = static_cast<std::size_t>(row) * t;
  float dot = 0.0f;
  for (int j = lane; j <= i; j += kWarp) {
    dot += probs[base + j] * dprobs[base + j];
  }
  dot = detail::warp_sum(dot);
  for (int j = lane; j < t; j += kWarp) {
    dscores[base + j] = j <= i ? scale * probs[base + j] * (dprobs[base + j] - dot) : 0.0f;
  }
}

__global__ void gelu_forward_kernel(float* __restrict__ out, const float* __restrict__ x, std::size_t n) {
  NEXUS_GRID_STRIDE(i, n) {
    const float v = x[i];
    out[i] = 0.5f * v * (1.0f + tanhf(kGeluC * (v + kGeluA * v * v * v)));
  }
}

__global__ void gelu_backward_kernel(float* __restrict__ dx, const float* __restrict__ dout,
                                     const float* __restrict__ x, std::size_t n) {
  NEXUS_GRID_STRIDE(i, n) {
    const float v = x[i];
    const float u = kGeluC * (v + kGeluA * v * v * v);
    const float th = tanhf(u);
    const float sech2 = 1.0f - th * th;
    const float grad = 0.5f * (1.0f + th) + 0.5f * v * sech2 * kGeluC * (1.0f + 3.0f * kGeluA * v * v);
    dx[i] = dout[i] * grad;
  }
}

__global__ void add_kernel(float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b,
                           std::size_t n) {
  NEXUS_GRID_STRIDE(i, n) { out[i] = a[i] + b[i]; }
}

// 16 bayt hizali tamponlar icin float4 yukleme/yazma (128 bit bellek islemleri).
__global__ void add_vec4_kernel(float4* __restrict__ out, const float4* __restrict__ a, const float4* __restrict__ b,
                                std::size_t n4) {
  NEXUS_GRID_STRIDE(i, n4) {
    const float4 x = a[i];
    const float4 y = b[i];
    out[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

__global__ void add_inplace_kernel(float* __restrict__ y, const float* __restrict__ x, std::size_t n) {
  NEXUS_GRID_STRIDE(i, n) { y[i] += x[i]; }
}

__global__ void fill_kernel(float* __restrict__ out, float value, std::size_t n) {
  NEXUS_GRID_STRIDE(i, n) { out[i] = value; }
}

// qkv [B,T,3,H,D] -> q/k/v [B,H,T,D]
__global__ void split_qkv_kernel(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                                 const float* __restrict__ qkv, int batch, int time, int heads, int dim) {
  const std::size_t total = static_cast<std::size_t>(batch) * heads * time * dim;
  NEXUS_GRID_STRIDE(idx, total) {
    const int d = static_cast<int>(idx % dim);
    std::size_t rest = idx / dim;
    const int tt = static_cast<int>(rest % time);
    rest /= time;
    const int h = static_cast<int>(rest % heads);
    const int b = static_cast<int>(rest / heads);
    const std::size_t c = static_cast<std::size_t>(heads) * dim;
    const std::size_t src = (static_cast<std::size_t>(b) * time + tt) * 3 * c + static_cast<std::size_t>(h) * dim + d;
    q[idx] = qkv[src];
    k[idx] = qkv[src + c];
    v[idx] = qkv[src + 2 * c];
  }
}

__global__ void merge_qkv_kernel(float* __restrict__ dqkv, const float* __restrict__ dq, const float* __restrict__ dk,
                                 const float* __restrict__ dv, int batch, int time, int heads, int dim) {
  const std::size_t total = static_cast<std::size_t>(batch) * heads * time * dim;
  NEXUS_GRID_STRIDE(idx, total) {
    const int d = static_cast<int>(idx % dim);
    std::size_t rest = idx / dim;
    const int tt = static_cast<int>(rest % time);
    rest /= time;
    const int h = static_cast<int>(rest % heads);
    const int b = static_cast<int>(rest / heads);
    const std::size_t c = static_cast<std::size_t>(heads) * dim;
    const std::size_t dst = (static_cast<std::size_t>(b) * time + tt) * 3 * c + static_cast<std::size_t>(h) * dim + d;
    dqkv[dst] = dq[idx];
    dqkv[dst + c] = dk[idx];
    dqkv[dst + 2 * c] = dv[idx];
  }
}

// y [B,H,T,D] <-> out [B,T,H*D]
template <bool kMerge>
__global__ void heads_kernel(float* __restrict__ dst, const float* __restrict__ src, int batch, int time, int heads,
                             int dim) {
  const std::size_t total = static_cast<std::size_t>(batch) * heads * time * dim;
  NEXUS_GRID_STRIDE(idx, total) {
    const int d = static_cast<int>(idx % dim);
    std::size_t rest = idx / dim;
    const int tt = static_cast<int>(rest % time);
    rest /= time;
    const int h = static_cast<int>(rest % heads);
    const int b = static_cast<int>(rest / heads);
    const std::size_t flat =
        (static_cast<std::size_t>(b) * time + tt) * static_cast<std::size_t>(heads) * dim + static_cast<std::size_t>(h) * dim + d;
    if (kMerge) {
      dst[flat] = src[idx];
    } else {
      dst[idx] = src[flat];
    }
  }
}

constexpr int kSumThreads = 256;
constexpr int kSumBlocks = 1024;

// Ip basina FP32 kismi toplam (float4 yukleme, sabit eleman atamasi -> belirlenimci),
// blok ve son asama FP64. Izgara boyutu sabit oldugundan sonuc donanimdan bagimsizdir.
__global__ void sum_partial_kernel(double* __restrict__ partial, const float* __restrict__ x, std::size_t n,
                                   bool vec4) {
  __shared__ double shared[32];
  float local = 0.0f;
  if (vec4) {
    const std::size_t n4 = n / 4;
    const auto* x4 = reinterpret_cast<const float4*>(x);
    NEXUS_GRID_STRIDE(i, n4) {
      const float4 v = x4[i];
      local += (v.x + v.y) + (v.z + v.w);
    }
    const std::size_t tail = n4 * 4 + blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (tail < n) {
      local += x[tail];
    }
  } else {
    NEXUS_GRID_STRIDE(i, n) { local += x[i]; }
  }
  double acc = detail::block_sum(static_cast<double>(local), shared);
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = acc;
  }
}

__global__ void sum_final_kernel(float* __restrict__ out, const double* __restrict__ partial, int count) {
  __shared__ double shared[32];
  double acc = 0.0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    acc += partial[i];
  }
  acc = detail::block_sum(acc, shared);
  if (threadIdx.x == 0) {
    out[0] = static_cast<float>(acc);
  }
}

}  // namespace

void softmax_causal_forward(float* probs, const float* scores, int bh, int t, float scale, cudaStream_t stream) {
  if (bh <= 0 || t <= 0) {
    throw TensorError("Nexus CUDA softmax: bh ve t pozitif olmali");
  }
  const int rows = bh * t;
  softmax_causal_forward_kernel<<<(rows + kRowsPerBlock - 1) / kRowsPerBlock, kRowThreads, 0, stream>>>(
      probs, scores, rows, t, scale);
  detail::check_launch("softmax_causal_forward_kernel");
}

void softmax_causal_backward(float* dscores, const float* dprobs, const float* probs, int bh, int t, float scale,
                             cudaStream_t stream) {
  if (bh <= 0 || t <= 0) {
    throw TensorError("Nexus CUDA softmax geri: bh ve t pozitif olmali");
  }
  const int rows = bh * t;
  softmax_causal_backward_kernel<<<(rows + kRowsPerBlock - 1) / kRowsPerBlock, kRowThreads, 0, stream>>>(
      dscores, dprobs, probs, rows, t, scale);
  detail::check_launch("softmax_causal_backward_kernel");
}

void gelu_forward(float* out, const float* x, std::size_t n, cudaStream_t stream) {
  gelu_forward_kernel<<<detail::blocks_for(n, 256), 256, 0, stream>>>(out, x, n);
  detail::check_launch("gelu_forward_kernel");
}

void gelu_backward(float* dx, const float* dout, const float* x, std::size_t n, cudaStream_t stream) {
  gelu_backward_kernel<<<detail::blocks_for(n, 256), 256, 0, stream>>>(dx, dout, x, n);
  detail::check_launch("gelu_backward_kernel");
}

void add(float* out, const float* a, const float* b, std::size_t n, cudaStream_t stream) {
  if (n % 4 == 0 && aligned16(out) && aligned16(a) && aligned16(b)) {
    add_vec4_kernel<<<detail::blocks_for(n / 4, 256), 256, 0, stream>>>(
        reinterpret_cast<float4*>(out), reinterpret_cast<const float4*>(a), reinterpret_cast<const float4*>(b), n / 4);
  } else {
    add_kernel<<<detail::blocks_for(n, 256), 256, 0, stream>>>(out, a, b, n);
  }
  detail::check_launch("add_kernel");
}

void add_inplace(float* y, const float* x, std::size_t n, cudaStream_t stream) {
  add_inplace_kernel<<<detail::blocks_for(n, 256), 256, 0, stream>>>(y, x, n);
  detail::check_launch("add_inplace_kernel");
}

void fill(float* out, float value, std::size_t n, cudaStream_t stream) {
  fill_kernel<<<detail::blocks_for(n, 256), 256, 0, stream>>>(out, value, n);
  detail::check_launch("fill_kernel");
}

void split_qkv_heads(float* q, float* k, float* v, const float* qkv, int batch, int time, int heads, int head_dim,
                     cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(batch) * heads * time * head_dim;
  split_qkv_kernel<<<detail::blocks_for(total, 256), 256, 0, stream>>>(q, k, v, qkv, batch, time, heads, head_dim);
  detail::check_launch("split_qkv_kernel");
}

void merge_qkv_heads(float* dqkv, const float* dq, const float* dk, const float* dv, int batch, int time, int heads,
                     int head_dim, cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(batch) * heads * time * head_dim;
  merge_qkv_kernel<<<detail::blocks_for(total, 256), 256, 0, stream>>>(dqkv, dq, dk, dv, batch, time, heads, head_dim);
  detail::check_launch("merge_qkv_kernel");
}

void merge_heads(float* out, const float* y, int batch, int time, int heads, int head_dim, cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(batch) * heads * time * head_dim;
  heads_kernel<true><<<detail::blocks_for(total, 256), 256, 0, stream>>>(out, y, batch, time, heads, head_dim);
  detail::check_launch("merge_heads_kernel");
}

void split_heads(float* y, const float* in, int batch, int time, int heads, int head_dim, cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(batch) * heads * time * head_dim;
  heads_kernel<false><<<detail::blocks_for(total, 256), 256, 0, stream>>>(y, in, batch, time, heads, head_dim);
  detail::check_launch("split_heads_kernel");
}

std::size_t sum_scratch_bytes(std::size_t) { return sizeof(double) * kSumBlocks; }

void sum(float* out, const float* x, std::size_t n, void* scratch, cudaStream_t stream) {
  auto* partial = static_cast<double*>(scratch);
  sum_partial_kernel<<<kSumBlocks, kSumThreads, 0, stream>>>(partial, x, n, aligned16(x));
  detail::check_launch("sum_partial_kernel");
  sum_final_kernel<<<1, kSumThreads, 0, stream>>>(out, partial, kSumBlocks);
  detail::check_launch("sum_final_kernel");
}

}  // namespace cuda
}  // namespace nexus
