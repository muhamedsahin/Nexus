#pragma once

/**
 * @file fp8.hpp
 * @brief FAZ 4.5 GPU FP8: niceleme/geri-niceleme, amax, emule FP8 GEMM, yerel FP8 kapisi.
 *
 * - `fp8_quantize`: q = fp8(x * scale) (RNE, doymali); ayni geciste amax = max|x|.
 * - `fp8_dequantize`: y = fp8_to_float(q) / scale, F32/BF16/F16 cikti.
 * - `fp8_gemm_emulated`: A ve B tensör basina olcekle FP8'e yuvarlanir, BF16'ya
 *   geri-nicelenir ve BF16 Tensor Core GEMM'i (FP32 biriktirici) ile carpilir. Sayisal
 *   davranis (FP8 yuvarlama hatasi) gercek FP8 matmul ile aynidir; hiz degildir.
 * - `fp8_native_supported`: cuBLASLt FP8 matmul sm_89+ (Ada/Hopper) ister. Bu kartta
 *   (sm_86) false; `require_fp8_native` acik hata verir. sm_89+ icin cuBLASLt yolu
 *   fp8_gemm_native'dedir; bu makinede yalniz derleme duzeyinde dogrulanabilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"
#include "nexus/core/fp8.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace nexus {
namespace cuda {

/// amax_out (aygit, 1 float, istege bagli) max|x| ile GUNCELLENIR (once 0'lanmali).
void fp8_quantize(std::uint8_t* out, const float* x, std::size_t n, float scale, Fp8Format format, float* amax_out,
                  cudaStream_t stream);
void fp8_dequantize(void* out, DType out_type, const std::uint8_t* q, std::size_t n, float scale, Fp8Format format,
                    cudaStream_t stream);

[[nodiscard]] bool fp8_native_supported(int device = 0);
/// sm_89+ degilse aciklayici TensorError.
void require_fp8_native(int device = 0);
/// sm_89+ cuBLASLt FP8 matmul, FP32 biriktirici/cikti. A [m,k], B [k,n] satir-ana
/// quantized FP8; q=fp8(x*scale) olcekleri FP32 sonuca alpha ile uygulanir.
void fp8_gemm_native(float* c, const std::uint8_t* a, const std::uint8_t* b, int m, int n, int k, float scale_a,
                     float scale_b, Fp8Format format, cudaStream_t stream);

/// C[m,n] (F32) = deq(q(A)) deq(q(B)); A [m,k], B [k,n] satir-ana FP32. Olcekler tensör
/// basina (q = fp8(x * scale)). scratch: fp8_gemm_emulated_scratch_bytes.
[[nodiscard]] std::size_t fp8_gemm_emulated_scratch_bytes(int m, int n, int k);
void fp8_gemm_emulated(float* c, const float* a, const float* b, int m, int n, int k, float scale_a, float scale_b,
                       Fp8Format format, void* scratch, cudaStream_t stream);

}  // namespace cuda
}  // namespace nexus
