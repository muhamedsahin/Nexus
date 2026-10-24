#pragma once

/**
 * @file kernels.hpp
 * @brief FAZ 3 CUDA egitim cekirdekleri (FP32, satir-ana, aygit isaretcileri).
 *
 * Her cekirdegin CPU FP64/FP32 referansi tests/cuda altinda vardir. Tum cagrilar
 * verilen akista asenkrondur; host'a donus yoktur (kayip bile aygit bellegindedir).
 *
 * Belirlenimcilik: `deterministic=true` olan yollar atomik kullanmaz; ayni girdi
 * ayni bitleri uretir. Indirgemeler iki asamalidir (blok kismi toplamlari + sabit
 * sirali son toplam).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace nexus {
namespace cuda {

// ---- Normalizasyon (3.2) ---------------------------------------------------------------
/// out = (x - mean) * rstd * w + b; mean/rstd [N] kaydedilir (Welford, FP32).
void layernorm_forward(float* out, float* mean, float* rstd, const float* x, const float* weight, const float* bias,
                       int rows, int cols, float eps, cudaStream_t stream);

/// Gecici alan boyutu (bayt): blok basina dweight/dbias kismi toplamlari.
[[nodiscard]] std::size_t layernorm_backward_scratch_bytes(int rows, int cols);

/// dx (+= accumulate ise), dweight, dbias (+= ). Atomiksiz iki asamali indirgeme.
void layernorm_backward(float* dx, float* dweight, float* dbias, const float* dout, const float* x,
                        const float* weight, const float* mean, const float* rstd, int rows, int cols,
                        bool accumulate_dx, float* scratch, cudaStream_t stream);

/// RMSNorm: out = x * rstd * w; rstd [N].
void rmsnorm_forward(float* out, float* rstd, const float* x, const float* weight, int rows, int cols, float eps,
                     cudaStream_t stream);
void rmsnorm_backward(float* dx, float* dweight, const float* dout, const float* x, const float* weight,
                      const float* rstd, int rows, int cols, bool accumulate_dx, float* scratch, cudaStream_t stream);

/// add & norm: sum = a + b (sonraki residual icin yazilir), out = layernorm(sum).
void residual_layernorm_forward(float* sum, float* out, float* mean, float* rstd, const float* a, const float* b,
                                const float* weight, const float* bias, int rows, int cols, float eps,
                                cudaStream_t stream);

// ---- Softmax / nedensel maske (3.3) ------------------------------------------------------
/// scores [BH, T, T] -> probs; satir i icin j > i maskelenir (maske tensoru yazilmaz).
/// probs[i, j>i] = 0 yazilir. `scale` skora carpilir (1/sqrt(dh)).
void softmax_causal_forward(float* probs, const float* scores, int bh, int t, float scale, cudaStream_t stream);
/// dscores = scale * probs * (dprobs - sum_j probs*dprobs).
void softmax_causal_backward(float* dscores, const float* dprobs, const float* probs, int bh, int t, float scale,
                             cudaStream_t stream);

// ---- Gomme (3.4) -------------------------------------------------------------------------
/// out[b*T+t] = wte[tokens] + wpe[t].
void embedding_pos_forward(float* out, const std::int32_t* tokens, const float* wte, const float* wpe, int batch,
                           int time, int dim, cudaStream_t stream);
/// atomicAdd yolu (hizli, belirlenimci degil). dwte/dwpe'ye EKLER.
void embedding_backward_atomic(float* dwte, float* dwpe, const float* dout, const std::int32_t* tokens, int batch,
                               int time, int dim, cudaStream_t stream);
[[nodiscard]] std::size_t embedding_backward_scratch_bytes(int batch, int time);
/// Siralama + segment toplami (belirlenimci); padding_idx satiri gradyan almaz (-1: yok).
/// dwte/dwpe'ye EKLER.
void embedding_backward_deterministic(float* dwte, float* dwpe, const float* dout, const std::int32_t* tokens,
                                      int batch, int time, int dim, int vocab, int padding_idx, void* scratch,
                                      cudaStream_t stream);

// ---- Fuzyonlu kaydirilmis CE (3.5) --------------------------------------------------------
struct CrossEntropyDeviceOptions {
  std::int32_t ignore_index = -100;
  float label_smoothing = 0.0f;
  float z_loss = 0.0f;
};
[[nodiscard]] std::size_t cross_entropy_scratch_bytes(int batch, int time);
/// logits [B*T, V]; hedef tokens[b, t+1], son konum ignore. dlogits == logits olabilir
/// (yerinde). loss_out (aygit, 1 float) gecerli tokenlar uzerinde ortalama; valid_out
/// (aygit, 1 int, istege bagli). `grad_scale` dlogits'e ek carpan (gradyan biriktirme).
/// Sozluk disi bir hedef token varsa loss_out NaN yazilir (host senkronu olmadan gorunur hata).
void cross_entropy_shifted(float* dlogits, float* loss_out, std::int32_t* valid_out, const float* logits,
                           const std::int32_t* tokens, int batch, int time, int vocab,
                           const CrossEntropyDeviceOptions& options, float grad_scale, void* scratch,
                           cudaStream_t stream);

// ---- Philox dropout (3.7) --------------------------------------------------------------
/// CPU `fused::philox_uniform` ile bit-bit ayni maske. out = x * mask * 1/(1-p).
void dropout_philox_forward(float* out, const float* x, std::size_t n, float p, std::uint64_t seed,
                            std::uint64_t offset, cudaStream_t stream);
void dropout_philox_backward(float* dx, const float* dout, std::size_t n, float p, std::uint64_t seed,
                             std::uint64_t offset, cudaStream_t stream);

// ---- Foreach AdamW + global norm (3.6) --------------------------------------------------
struct DeviceParamSlot {
  float* param = nullptr;
  const float* grad = nullptr;
  float* exp_avg = nullptr;
  float* exp_avg_sq = nullptr;
  std::uint64_t numel = 0;
  /// Agirlik sonumu carpani: 1 = tam, 0 = yok (GPT-2: bias ve LayerNorm'da sonum yok).
  float weight_decay_scale = 1.0f;
};

struct DeviceAdamWConfig {
  double lr = 1.0e-3;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps = 1.0e-8;
  double weight_decay = 0.0;
  double max_grad_norm = 0.0;  // <= 0: kirpma yok
};

[[nodiscard]] std::size_t adamw_scratch_bytes(int num_slots, std::uint64_t total_numel);
/// Tum parametreler tek cagrida. Global norm aygitta hesaplanir, kirpma katsayisi
/// host'a donmeden guncelleme cekirdegine gecer. grad_norm_out (aygit, istege bagli).
/// `slots` HOST dizisidir; meta-veri scratch'e kopyalanir.
void adamw_multi_tensor(const DeviceParamSlot* slots, int num_slots, const DeviceAdamWConfig& config,
                        std::int64_t step, float* grad_norm_out, void* scratch, cudaStream_t stream);

// ---- Eleman eleman ve dikkat yardimcilari ----------------------------------------------
/// GELU (tanh yaklasimi, GPT-2). out = gelu(x).
void gelu_forward(float* out, const float* x, std::size_t n, cudaStream_t stream);
/// dx = dout * gelu'(x).
void gelu_backward(float* dx, const float* dout, const float* x, std::size_t n, cudaStream_t stream);
/// out = a + b.
void add(float* out, const float* a, const float* b, std::size_t n, cudaStream_t stream);
/// y += x.
void add_inplace(float* y, const float* x, std::size_t n, cudaStream_t stream);
void fill(float* out, float value, std::size_t n, cudaStream_t stream);
/// qkv [B, T, 3, H, D] -> q, k, v [B, H, T, D].
void split_qkv_heads(float* q, float* k, float* v, const float* qkv, int batch, int time, int heads, int head_dim,
                     cudaStream_t stream);
/// Ters: dq, dk, dv [B,H,T,D] -> dqkv [B,T,3,H,D].
void merge_qkv_heads(float* dqkv, const float* dq, const float* dk, const float* dv, int batch, int time, int heads,
                     int head_dim, cudaStream_t stream);
/// y [B, H, T, D] -> out [B, T, H*D] (ve tersi: unpermute_backward).
void merge_heads(float* out, const float* y, int batch, int time, int heads, int head_dim, cudaStream_t stream);
void split_heads(float* y, const float* in, int batch, int time, int heads, int head_dim, cudaStream_t stream);
/// Belirlenimci tam toplam (tek blok son asama): out[0] = sum(x).
[[nodiscard]] std::size_t sum_scratch_bytes(std::size_t n);
void sum(float* out, const float* x, std::size_t n, void* scratch, cudaStream_t stream);

}  // namespace cuda
}  // namespace nexus
