/**
 * @file kernels_lm.cu
 * @brief FAZ 6 GPU: RoPE (yerinde, qkv adimli) ve SwiGLU/GeGLU ileri/geri cekirdekleri.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "common.cuh"
#include "nexus/core/error.hpp"
#include "nexus/cuda/lm_kernels.hpp"

#include <algorithm>
#include <vector>

namespace nexus {
namespace cuda {
namespace {

using detail::Num;

constexpr int kPairThreads = 64;
constexpr int kHeadsPerBlock = 4;

// Blok: (64 cift, 4 bas). Izgara: (ceil(heads/4), rows).
template <typename T>
__global__ void rope_kernel(T* x, int rows, long long row_stride, int col0, int heads, int head_dim, int half,
                            int time, int offset, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
                            float sign) {
  const int row = blockIdx.y;
  const int h = blockIdx.x * kHeadsPerBlock + threadIdx.y;
  if (row >= rows || h >= heads) {
    return;
  }
  const int pos = row % time + offset;
  T* v = x + row * row_stride + col0 + static_cast<long long>(h) * head_dim;
  const float* c = cos_t + static_cast<long long>(pos) * half;
  const float* s = sin_t + static_cast<long long>(pos) * half;
  for (int i = threadIdx.x; i < half; i += kPairThreads) {
    const float x1 = Num<T>::load(v[i]);
    const float x2 = Num<T>::load(v[i + half]);
    const float si = sign * s[i];
    v[i] = Num<T>::store(x1 * c[i] - x2 * si);
    v[i + half] = Num<T>::store(x2 * c[i] + x1 * si);
  }
}

constexpr float kInvSqrt2 = 0.70710678118654752f;
constexpr float kInvSqrt2Pi = 0.39894228040143268f;

template <GatedAct A>
__device__ __forceinline__ void act(float a, float* f, float* df) {
  if constexpr (A == GatedAct::SiLU) {
    const float sg = 1.0f / (1.0f + __expf(-a));
    *f = a * sg;
    *df = sg * (1.0f + a * (1.0f - sg));
  } else {
    const float cdf = 0.5f * (1.0f + erff(a * kInvSqrt2));
    *f = a * cdf;
    *df = cdf + a * kInvSqrt2Pi * __expf(-0.5f * a * a);
  }
}

template <typename T, GatedAct A>
__global__ void gated_fwd_kernel(T* __restrict__ out, const T* __restrict__ in, long long n, int f) {
  for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<long long>(gridDim.x) * blockDim.x) {
    const long long r = idx / f;
    const int i = static_cast<int>(idx - r * f);
    const T* row = in + r * 2 * f;
    float fa = 0.0f;
    float dfa = 0.0f;
    act<A>(Num<T>::load(row[i]), &fa, &dfa);
    out[idx] = Num<T>::store(fa * Num<T>::load(row[i + f]));
  }
}

template <typename T, GatedAct A>
__global__ void gated_bwd_kernel(T* __restrict__ din, const T* __restrict__ dout, const T* __restrict__ in, long long n,
                                 int f) {
  for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<long long>(gridDim.x) * blockDim.x) {
    const long long r = idx / f;
    const int i = static_cast<int>(idx - r * f);
    const T* row = in + r * 2 * f;
    float fa = 0.0f;
    float dfa = 0.0f;
    act<A>(Num<T>::load(row[i]), &fa, &dfa);
    const float g = Num<T>::load(dout[idx]);
    din[r * 2 * f + i] = Num<T>::store(g * Num<T>::load(row[i + f]) * dfa);
    din[r * 2 * f + i + f] = Num<T>::store(g * fa);
  }
}

unsigned int grid_for(long long n) {
  const long long blocks = (n + 255) / 256;
  return static_cast<unsigned int>(std::min<long long>(blocks, 65535LL * 8));
}

}  // namespace

DeviceRopeTable::DeviceRopeTable(int rot_dim, float theta, lm::RopeScaling scaling, int max_position,
                                 const Stream& stream)
    : host_(rot_dim, theta, scaling, max_position), stream_(&stream) {}

void DeviceRopeTable::ensure(int length) {
  const int before = host_.length();
  const std::vector<float> freq_before = host_.inv_freq();
  host_.ensure(length);
  if (host_.length() == uploaded_ && before == host_.length() && freq_before == host_.inv_freq()) {
    return;  // degisiklik yok
  }
  const std::size_t bytes = static_cast<std::size_t>(host_.length()) * (host_.rot_dim() / 2) * sizeof(float);
  if (cos_.bytes() < bytes) {
    cos_ = DeviceBuffer(bytes, *stream_);
    sin_ = DeviceBuffer(bytes, *stream_);
  }
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(cos_.data(), host_.cos(0), bytes, cudaMemcpyHostToDevice, stream_->handle()));
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(sin_.data(), host_.sin(0), bytes, cudaMemcpyHostToDevice, stream_->handle()));
  stream_->synchronize();  // host tablosu sonraki ensure'da yeniden yazilabilir
  uploaded_ = host_.length();
}

void rope_inplace(const RopeArgs& a, const DeviceRopeTable& table, cudaStream_t stream) {
  const int half = table.rot_dim() / 2;
  if (table.rot_dim() > a.head_dim || a.rows < 0 || a.time < 1) {
    throw TensorError("Nexus CUDA rope: rot_dim <= head_dim, time >= 1 olmali");
  }
  if (std::min(a.time, a.rows) + a.offset > table.length()) {
    throw TensorError("Nexus CUDA rope: konum tablo disinda (ensure cagrilmadi)");
  }
  if (a.rows == 0) {
    return;
  }
  const dim3 block(kPairThreads, kHeadsPerBlock);
  const dim3 grid((a.heads + kHeadsPerBlock - 1) / kHeadsPerBlock, a.rows);
  detail::dispatch_storage(a.dtype, "rope", [&](auto tag) {
    using T = decltype(tag);
    rope_kernel<T><<<grid, block, 0, stream>>>(static_cast<T*>(a.x), a.rows, a.row_stride, a.col0, a.heads, a.head_dim,
                                               half, a.time, a.offset, table.cos(), table.sin(),
                                               a.transpose ? -1.0f : 1.0f);
  });
  detail::check_launch("rope_kernel");
}

void gated_forward(void* out, const void* in, DType dtype, int rows, int f, GatedAct a, cudaStream_t stream) {
  const long long n = static_cast<long long>(rows) * f;
  if (n == 0) {
    return;
  }
  detail::dispatch_storage(dtype, "gated_forward", [&](auto tag) {
    using T = decltype(tag);
    if (a == GatedAct::SiLU) {
      gated_fwd_kernel<T, GatedAct::SiLU><<<grid_for(n), 256, 0, stream>>>(static_cast<T*>(out), static_cast<const T*>(in), n, f);
    } else {
      gated_fwd_kernel<T, GatedAct::GeLU><<<grid_for(n), 256, 0, stream>>>(static_cast<T*>(out), static_cast<const T*>(in), n, f);
    }
  });
  detail::check_launch("gated_fwd_kernel");
}

void gated_backward(void* din, const void* dout, const void* in, DType dtype, int rows, int f, GatedAct a,
                    cudaStream_t stream) {
  const long long n = static_cast<long long>(rows) * f;
  if (n == 0) {
    return;
  }
  detail::dispatch_storage(dtype, "gated_backward", [&](auto tag) {
    using T = decltype(tag);
    if (a == GatedAct::SiLU) {
      gated_bwd_kernel<T, GatedAct::SiLU><<<grid_for(n), 256, 0, stream>>>(
          static_cast<T*>(din), static_cast<const T*>(dout), static_cast<const T*>(in), n, f);
    } else {
      gated_bwd_kernel<T, GatedAct::GeLU><<<grid_for(n), 256, 0, stream>>>(
          static_cast<T*>(din), static_cast<const T*>(dout), static_cast<const T*>(in), n, f);
    }
  });
  detail::check_launch("gated_bwd_kernel");
}

}  // namespace cuda
}  // namespace nexus
