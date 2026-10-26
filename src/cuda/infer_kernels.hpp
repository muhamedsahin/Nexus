#pragma once

/**
 * @file infer_kernels.hpp
 * @brief FAZ 16 cikarim cekirdekleri (ic arayuz): gomme, KV ekleme (FP16/FP32/INT8), sayfali dikkat, ince GEMM
 *        (satir sonucu m'den bagimsiz), ornekleme, NLL, nicemleme (INT8 satir, INT32 -> cikis, W4 GEMV / acma).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace nexus {
namespace cuda {
namespace infer {
namespace k {

/// x[n, d] (FP32) = wte[tok[n]] + wpe[pos[n]]; w tipi F16 ya da F32.
void embed(float* x, const std::int32_t* tok, const std::int32_t* pos, const void* wte, const void* wpe, DType wtype, int n,
           int d, cudaStream_t s);

/// qkv [n, 3, H, D] (act tipi) -> K/V onbellek yuvasi slot[n]. INT8: (yuva, bas) basina simetrik olcek amax/127.
void kv_append(void* kc, void* vc, float* kscale, float* vscale, DType cache, const void* qkv, DType act,
               const std::int32_t* slot, int n, int H, int D, cudaStream_t s);

struct PagedArgs {
  const void* qkv = nullptr;   // [n, 3, H, D] (sorgu: ilk H*D)
  void* out = nullptr;         // [n, H, D] act tipi
  DType act = DType::F16;
  const void* kc = nullptr;
  const void* vc = nullptr;
  const float* kscale = nullptr;
  const float* vscale = nullptr;
  DType cache = DType::F16;
  const std::int32_t* row_seq = nullptr;   // [n] sira yuvasi (blok tablosu satiri)
  const std::int32_t* row_pos = nullptr;   // [n] sorgunun konumu: anahtarlar 0..pos
  const std::int32_t* block_table = nullptr;  // [sira, max_blocks]
  int max_blocks = 0;
  int block_size = 16;
  int n = 0, H = 12, D = 64;
  int max_ctx = 1024;
  float scale = 0.125f;
};
void paged_attention(const PagedArgs& a, cudaStream_t s);

/// dst[i] = src[idx[i]] (satir, FP32).
void gather_rows(float* dst, const float* src, const std::int32_t* idx, int rows, int cols, cudaStream_t s);

/// Ince GEMM: y[m, n] (= ya da +=) act(x[m, k] W^T + b), W [n, k] (w_nk zorunlu; k % 8 == 0).
/// Warp basina bir cikis, lane'ler K uzerinde 16 baytlik adimlar, sabit kelebek indirgeme -> sonuc m'den bagimsiz.
struct SkinnyArgs {
  const void* x = nullptr;   // [m, k] act tipi
  DType xtype = DType::F16;
  const void* w = nullptr;
  DType wtype = DType::F16;
  bool w_nk = false;
  const float* bias = nullptr;
  void* y = nullptr;
  DType ytype = DType::F16;
  bool accumulate = false;   // y (FP32) += ...
  bool gelu = false;
  int m = 0, n = 0, k = 0;
  int ldy = 0;               // 0: n
};
inline constexpr int kSkinnyMaxRows = 16;
void skinny_gemm(const SkinnyArgs& a, cudaStream_t s);

/// Ornekleme satiri (aygita kopyalanir).
struct SampleRow {
  float temperature, top_p, min_p, repetition, frequency, presence;
  int top_k;
  int seq;  // ceza sayaci satiri
  unsigned long long seed, offset;
};
/// counts / out_counts: [sira, V] (istem + cikti / yalniz cikti). scratch [nrows, ld]: cezali/olcekli logitler
/// (logits degismez: ayni satirdan tekrar ornek alinabilir).
void sample(std::int32_t* out, const float* logits, float* scratch, int ld, int V, const SampleRow* rows, int nrows,
            const std::int32_t* counts, const std::int32_t* out_counts, cudaStream_t s);
/// counts[seq[i], tok[i]] += delta (out_counts yalniz is_output[i] ise).
void count_tokens(std::int32_t* counts, std::int32_t* out_counts, const std::int32_t* seq, const std::int32_t* tok,
                  const std::uint8_t* is_output, int n, int V, int delta, cudaStream_t s);
/// out[i] = logsumexp(logits[i, :V]) - logits[i, targets[i]].
void row_nll(float* out, const float* logits, int ld, int V, const std::int32_t* targets, int rows, cudaStream_t s);

// ---- Nicemleme ---------------------------------------------------------------------------------
/// x [m, k] (F16) * inv_smooth[k] (nullable) -> q INT8 [m, k], sx[m] = amax / 127 (simetrik, satir basina).
void quant_rows_i8(std::int8_t* q, float* sx, const void* x, const float* inv_smooth, int m, int k, cudaStream_t s);
/// y[m, n] (= F16 / += F32) = act(acc[m, n] * sx[m] * sw[n] + b[n]).
void dequant_i32(void* y, DType ytype, bool accumulate, bool gelu, const std::int32_t* acc, const float* sx,
                 const float* sw, const float* bias, int m, int n, cudaStream_t s);
/// W4 (grup g, asimetrik): packed [n, k/2] (dusuk yarim tek indeks), scale/zero [n, k/g] (FP32 / uint8).
struct W4Args {
  const void* x = nullptr;  // [m, k] F16
  const float* inv_s = nullptr;  // [k] (AWQ: x / s), nullable
  const std::uint8_t* packed = nullptr;
  const float* scale = nullptr;
  const std::uint8_t* zero = nullptr;
  int group = 128;
  const float* bias = nullptr;
  void* y = nullptr;
  DType ytype = DType::F16;
  bool accumulate = false;
  bool gelu = false;
  int m = 0, n = 0, k = 0;
};
void w4_gemv(const W4Args& a, cudaStream_t s);
/// W4 -> FP16 [k, n] (prefill: sonra cuBLASLt); W' = (q - z) * s, AWQ olcegi geri alinmaz (x / s ayrica).
void w4_dequant(void* w_kn, const std::uint8_t* packed, const float* scale, const std::uint8_t* zero, int group, int n,
                int k, cudaStream_t s);
/// x[m, k] F16 *= inv_s[k] (yerinde).
void scale_cols(void* x, const float* inv_s, int m, int k, cudaStream_t s);

}  // namespace k
}  // namespace infer
}  // namespace cuda
}  // namespace nexus
