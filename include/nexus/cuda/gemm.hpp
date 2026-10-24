#pragma once

/**
 * @file gemm.hpp
 * @brief cuBLASLt GEMM katmani — FAZ 3.1.
 *
 * Satir-ana (row-major) sozlesme: C[m,n] = alpha * op(A)[m,k] * op(B)[k,n] + beta * C.
 * `trans_a` true ise A bellekte [k,m] durur (op(A) = A^T); `trans_b` icin ayni. Boylece
 * dW = X^T * dY ve dX = dY * W^T kopyasiz hesaplanir.
 *
 * - cuBLASLt matmul tanimlayici + algoritma secimi (sezgisel, istege bagli otomatik
 *   ayar) sekil anahtarli plan onbelleginde tutulur.
 * - Calisma alani (workspace) onbellekli aygit ayiricisindan; tanitici (handle) aygit
 *   basina, akis argumani her cagrida verilir.
 * - Epilog fuzyonu: bias, GELU (tanh yaklasimi) + aux cikti (on-aktivasyon), ReLU,
 *   DGELU (geri), BGRADA/BGRADB (bias gradyani).
 * - Strided batched GEMM (dikkat icin).
 * - TF32 politikasi: "highest" (tam FP32), "high"/"medium" (TF32 Tensor Core).
 *
 * Veri tipleri (FAZ 4): A/B F32, BF16 veya F16; C/D F32 ya da A ile ayni dar tip. Dar
 * girislerde hesap FP32 biriktiricilidir (CUBLAS_COMPUTE_32F, Tensor Core). Bias ve aux
 * tipi varsayilan olarak D tipidir; `bias_type` ile FP32 bias + dar D secilebilir.
 *
 * Desteklenmeyen epilog/sekil bileşimi icin cuBLASLt algoritma bulamazsa acik
 * TensorError atilir; sessiz yedek yol yoktur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace nexus {
namespace cuda {

enum class Epilogue : std::uint8_t {
  None = 0,
  Bias,        // D = AB + bias (bias uzunlugu n)
  Relu,
  BiasRelu,
  Gelu,        // D = gelu(AB), aux = AB (aux verilirse)
  BiasGelu,    // D = gelu(AB + bias), aux = AB + bias
  DGelu,       // D = (AB) * gelu'(aux)   (geri gecis)
  BGradA,      // D = AB, bias_grad[m] = sum_k op(A)[m,k]   (satir-ana anlamiyla)
  BGradB,      // D = AB, bias_grad[n] = sum_k op(B)[k,n]   (Linear geri: db = sum dY)
};

[[nodiscard]] const char* to_string(Epilogue epilogue) noexcept;

struct GemmArgs {
  bool trans_a = false;
  bool trans_b = false;
  int m = 0;
  int n = 0;
  int k = 0;
  const void* a = nullptr;
  int lda = 0;  // A'nin bellekteki satir adimi (satir-ana)
  const void* b = nullptr;
  int ldb = 0;
  void* c = nullptr;
  int ldc = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  Epilogue epilogue = Epilogue::None;
  const void* bias = nullptr;   // Bias* epiloglari (tip: bias_type)
  void* bias_grad = nullptr;    // BGrad* epiloglari (tip: bias_type)
  void* aux = nullptr;          // Gelu aux cikti / DGelu aux girdi ([m,n], ldc adimi, tip: c_type)
  DType a_type = DType::F32;
  DType b_type = DType::F32;
  DType c_type = DType::F32;
  DType bias_type = DType::F32;  // c_type dar ise ve bu F32 ise FP32 bias kullanilir
  int batch = 1;
  long long stride_a = 0;
  long long stride_b = 0;
  long long stride_c = 0;
};

/// Satir-ana GEMM. `stream` uzerinde asenkron.
void gemm(const GemmArgs& args, cudaStream_t stream);

/// FP32 matmul hassasiyet politikasi (PyTorch set_float32_matmul_precision).
enum class MatmulPrecision : std::uint8_t { Highest = 0, High = 1, Medium = 2 };
void set_float32_matmul_precision(MatmulPrecision precision) noexcept;
[[nodiscard]] MatmulPrecision float32_matmul_precision() noexcept;
/// "highest" | "high" | "medium"; bilinmeyende TensorError.
void set_float32_matmul_precision(const std::string& name);

/// true: her yeni sekil icin aday algoritmalar olculup en hizlisi plan onbellegine yazilir.
void set_gemm_autotune(bool enabled) noexcept;

struct GemmCacheStats {
  std::uint64_t plans = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t autotuned = 0;
};
[[nodiscard]] GemmCacheStats gemm_cache_stats() noexcept;
void clear_gemm_cache();

}  // namespace cuda
}  // namespace nexus
