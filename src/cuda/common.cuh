#pragma once

/**
 * @file common.cuh
 * @brief Nexus CUDA cekirdekleri icin ortak yardimcilar: warp/blok indirgeme, hata denetimi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace nexus {
namespace cuda {
namespace detail {

constexpr int kWarp = 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
  }
  return v;
}

/// Blok toplami (sabit agac sirasi: belirlenimci). `shared` en az 32 eleman.
template <typename T>
__device__ __forceinline__ T block_sum(T v, T* shared) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  const int warps = (blockDim.x + kWarp - 1) / kWarp;
  T total = lane < warps ? shared[lane] : T(0);
  total = warp_sum(total);
  return total;  // tum iplerde ayni deger
}

__device__ __forceinline__ float block_max(float v, float* shared) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  const int warps = (blockDim.x + kWarp - 1) / kWarp;
  float m = lane < warps ? shared[lane] : -INFINITY;
  return warp_max(m);
}

inline unsigned int blocks_for(std::size_t n, int threads) {
  const std::size_t b = (n + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);
  return static_cast<unsigned int>(b < 1 ? 1 : (b > 65535u * 8u ? 65535u * 8u : b));
}

/// Cekirdek baslatma hatasini hemen yakalar (asenkron hatalar sonraki senkronda gorunur).
inline void check_launch(const char* name) { check(cudaGetLastError(), name); }

}  // namespace detail
}  // namespace cuda
}  // namespace nexus
