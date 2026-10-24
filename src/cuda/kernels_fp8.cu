/**
 * @file kernels_fp8.cu
 * @brief FAZ 4.5 FP8 emulasyonu: niceleme (+amax), geri-niceleme, emule FP8 GEMM.
 *
 * Aygit donusumu cuda_fp8.h `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, ...)` ile yapilir
 * (RNE, doymali). test_fp8_emulation host tablosuyla (nexus/core/fp8.hpp) esitligi sinar.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "common.cuh"
#include "nexus/cuda/fp8.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/kernels.hpp"

#include <cuda_fp8.h>

namespace nexus {
namespace cuda {
namespace {

#define NEXUS_GRID_STRIDE(i, n)                                                                         \
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < (n); \
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)

__device__ __forceinline__ __nv_fp8_interpretation_t interp(int format) {
  return format == 0 ? __NV_E4M3 : __NV_E5M2;
}

__global__ void fp8_quantize_kernel(std::uint8_t* __restrict__ out, const float* __restrict__ x, std::size_t n,
                                    float scale, int format, float* __restrict__ amax_out) {
  float local = 0.0f;
  NEXUS_GRID_STRIDE(i, n) {
    const float v = x[i];
    local = fmaxf(local, fabsf(v));
    out[i] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(v * scale, __NV_SATFINITE, interp(format)));
  }
  if (amax_out != nullptr) {
    local = detail::warp_max(local);
    if (threadIdx.x % detail::kWarp == 0) {
      // Pozitif float'lar icin tamsayi sirasi = sayi sirasi: atomicMax sira-bagimsiz (belirlenimci).
      atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(local));
    }
  }
}

template <typename T>
__global__ void fp8_dequantize_kernel(T* __restrict__ out, const std::uint8_t* __restrict__ q, std::size_t n,
                                      float inv_scale, int format) {
  NEXUS_GRID_STRIDE(i, n) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(q[i]), interp(format));
    out[i] = detail::Num<T>::store(__half2float(__half(h)) * inv_scale);
  }
}

std::size_t align_up(std::size_t bytes) { return (bytes + 255u) & ~static_cast<std::size_t>(255u); }

}  // namespace

void fp8_quantize(std::uint8_t* out, const float* x, std::size_t n, float scale, Fp8Format format, float* amax_out,
                  cudaStream_t stream) {
  if (!(scale > 0.0f) || !std::isfinite(scale)) {
    throw TensorError("Nexus CUDA fp8_quantize: olcek pozitif ve sonlu olmali");
  }
  fp8_quantize_kernel<<<detail::blocks_for(n, 256), 256, 0, stream>>>(out, x, n, scale, static_cast<int>(format),
                                                                      amax_out);
  detail::check_launch("fp8_quantize_kernel");
}

void fp8_dequantize(void* out, DType out_type, const std::uint8_t* q, std::size_t n, float scale, Fp8Format format,
                    cudaStream_t stream) {
  if (!(scale > 0.0f) || !std::isfinite(scale)) {
    throw TensorError("Nexus CUDA fp8_dequantize: olcek pozitif ve sonlu olmali");
  }
  detail::dispatch_storage(out_type, "fp8_dequantize", [&](auto tag) {
    using T = decltype(tag);
    fp8_dequantize_kernel<T><<<detail::blocks_for(n, 256), 256, 0, stream>>>(static_cast<T*>(out), q, n, 1.0f / scale,
                                                                             static_cast<int>(format));
  });
  detail::check_launch("fp8_dequantize_kernel");
}

bool fp8_native_supported(int device) {
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  return prop.major > 8 || (prop.major == 8 && prop.minor >= 9);
}

void require_fp8_native(int device) {
  if (!fp8_native_supported(device)) {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    throw TensorError("Nexus FP8: yerel FP8 matmul sm_89+ (Ada/Hopper) Tensor Core gerektirir; bu aygit sm_" +
                      std::to_string(prop.major) + std::to_string(prop.minor) +
                      ". Sayisal davranis icin fp8_gemm_emulated kullanin.");
  }
  // Desteklenen aygitta fp8_gemm_native cuBLASLt yolunu kullanir.
}

std::size_t fp8_gemm_emulated_scratch_bytes(int m, int n, int k) {
  const std::size_t a = static_cast<std::size_t>(m) * k;
  const std::size_t b = static_cast<std::size_t>(k) * n;
  return align_up(a) + align_up(b) + align_up(a * 2) + align_up(b * 2);
}

void fp8_gemm_emulated(float* c, const float* a, const float* b, int m, int n, int k, float scale_a, float scale_b,
                       Fp8Format format, void* scratch, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || scratch == nullptr) {
    throw TensorError("Nexus fp8_gemm_emulated: gecersiz sekil veya scratch yok");
  }
  const std::size_t na = static_cast<std::size_t>(m) * k;
  const std::size_t nb = static_cast<std::size_t>(k) * n;
  auto* base = static_cast<char*>(scratch);
  auto* qa = reinterpret_cast<std::uint8_t*>(base);
  auto* qb = reinterpret_cast<std::uint8_t*>(base + align_up(na));
  void* ha = base + align_up(na) + align_up(nb);
  void* hb = static_cast<char*>(ha) + align_up(na * 2);
  fp8_quantize(qa, a, na, scale_a, format, nullptr, stream);
  fp8_quantize(qb, b, nb, scale_b, format, nullptr, stream);
  // FP8 degerleri BF16'da tam temsil edilir: olceksiz aktarim kayipsizdir. Olcek, donanim
  // FP8 matmul'unda oldugu gibi FP32 biriktiriciden sonra (alpha) uygulanir.
  fp8_dequantize(ha, DType::BF16, qa, na, 1.0f, format, stream);
  fp8_dequantize(hb, DType::BF16, qb, nb, 1.0f, format, stream);
  GemmArgs g;
  g.m = m;
  g.n = n;
  g.k = k;
  g.a = ha;
  g.lda = k;
  g.b = hb;
  g.ldb = n;
  g.c = c;
  g.ldc = n;
  g.a_type = DType::BF16;
  g.b_type = DType::BF16;
  g.c_type = DType::F32;
  g.alpha = 1.0f / (scale_a * scale_b);
  gemm(g, stream);
}

}  // namespace cuda
}  // namespace nexus
