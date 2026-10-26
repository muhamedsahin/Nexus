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

#include "nexus/core/dtype.hpp"
#include "nexus_train/mixed_precision/loss_scale_policy.hpp"

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

/// FAZ 4: cikti dar tipte (F32/BF16/F16), istatistik FP32. Giris daima FP32 residual.
void layernorm_forward(void* out, DType out_type, float* mean, float* rstd, const float* x, const float* weight,
                       const float* bias, int rows, int cols, float eps, cudaStream_t stream);

/// add & norm: sum = a + b (sonraki residual icin yazilir), out = layernorm(sum).
void residual_layernorm_forward(float* sum, float* out, float* mean, float* rstd, const float* a, const float* b,
                                const float* weight, const float* bias, int rows, int cols, float eps,
                                cudaStream_t stream);
/// FAZ 4: sum FP32, out dar tip.
void residual_layernorm_forward(float* sum, void* out, DType out_type, float* mean, float* rstd, const float* a,
                                const float* b, const float* weight, const float* bias, int rows, int cols, float eps,
                                cudaStream_t stream);

// ---- Softmax / nedensel maske (3.3) ------------------------------------------------------
/// scores [BH, T, T] -> probs; satir i icin j > i maskelenir (maske tensoru yazilmaz).
/// probs[i, j>i] = 0 yazilir. `scale` skora carpilir (1/sqrt(dh)).
void softmax_causal_forward(float* probs, const float* scores, int bh, int t, float scale, cudaStream_t stream);
/// dscores = scale * probs * (dprobs - sum_j probs*dprobs).
void softmax_causal_backward(float* dscores, const float* dprobs, const float* probs, int bh, int t, float scale,
                             cudaStream_t stream);
/// FAZ 4: depolama tipi F32/BF16/F16, hesap ve satir indirgemesi FP32.
void softmax_causal_forward(void* probs, const void* scores, DType type, int bh, int t, float scale,
                            cudaStream_t stream);
void softmax_causal_backward(void* dscores, const void* dprobs, const void* probs, DType type, int bh, int t,
                             float scale, cudaStream_t stream);

// ---- Gomme (3.4) -------------------------------------------------------------------------
/// out[b*T+t] = wte[tokens] + wpe[t].
void embedding_pos_forward(float* out, const std::int32_t* tokens, const float* wte, const float* wpe, int batch,
                           int time, int dim, cudaStream_t stream);
/// FAZ 4: agirliklar dar tipte (ana kopyasiz mod), cikti FP32 residual.
void embedding_pos_forward(float* out, const std::int32_t* tokens, const void* wte, const void* wpe, DType weight_type,
                           int batch, int time, int dim, cudaStream_t stream);
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
  int row_stride = 0;  // FAZ 13: logit satir adimi (0: vocab). [vocab, row_stride) dolgu: yok sayilir, dlogits 0
  void* dlogits_bf16 = nullptr;
  bool online = false;  // FAZ 13: tek okumali cevrimici softmax + float4 (compile yolu); false: FAZ 3 toplama sirasi  // FAZ 13: verilirse dlogits FP32 yerine BF16 yazilir (ayri cast cekirdegi yok; ayni RN)
};
[[nodiscard]] std::size_t cross_entropy_scratch_bytes(int batch, int time);
/// FAZ 7.5 parcali CE: begin (gecerli sayim) -> her parca icin chunk (logits/dlogits parca
/// yerel, row_begin kuresel satir) -> end (ortalama kayip). Tam CE ile ayni satir matematigi.
void cross_entropy_chunked_begin(const std::int32_t* tokens, int batch, int time, int vocab,
                                 const CrossEntropyDeviceOptions& options, void* scratch, cudaStream_t stream);
void cross_entropy_chunk(float* dlogits, const float* logits, const std::int32_t* tokens, int batch, int time, int vocab,
                         const CrossEntropyDeviceOptions& options, float grad_scale, const float* device_grad_scale,
                         int row_begin, int rows, void* scratch, cudaStream_t stream);
void cross_entropy_chunked_end(float* loss_out, std::int32_t* valid_out, int batch, int time, void* scratch,
                               cudaStream_t stream);
/// logits [B*T, V]; hedef tokens[b, t+1], son konum ignore. dlogits == logits olabilir
/// (yerinde). loss_out (aygit, 1 float) gecerli tokenlar uzerinde ortalama; valid_out
/// (aygit, 1 int, istege bagli). `grad_scale` dlogits'e ek carpan (gradyan biriktirme).
/// Sozluk disi bir hedef token varsa loss_out NaN yazilir (host senkronu olmadan gorunur hata).
void cross_entropy_shifted(float* dlogits, float* loss_out, std::int32_t* valid_out, const float* logits,
                           const std::int32_t* tokens, int batch, int time, int vocab,
                           const CrossEntropyDeviceOptions& options, float grad_scale, void* scratch,
                           cudaStream_t stream);
/// FAZ 4.3: dlogits ek olarak *device_grad_scale ile carpilir (FP16 kayip olcegi, host senkronu yok).
void cross_entropy_shifted(float* dlogits, float* loss_out, std::int32_t* valid_out, const float* logits,
                           const std::int32_t* tokens, int batch, int time, int vocab,
                           const CrossEntropyDeviceOptions& options, float grad_scale, const float* device_grad_scale,
                           void* scratch, cudaStream_t stream);

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
  float lr_scale = 1.0f;  ///< FAZ 9.3: grup ogrenme hizi carpani (muP)
  /// FAZ 4.2: dar (BF16/F16) golge kopya. param != nullptr ise FP32 ana kopya guncellenir ve
  /// golgeye RN ile yazilir; param == nullptr ise parametre yalniz golgede yasar (ana kopyasiz mod).
  void* shadow = nullptr;
};

/// FAZ 13.6: CUDA Graph uyumlu AdamW: degisken hiperparametreler aygitta (yeniden oynatmada pinned bellekten
/// bir memcpy dugumu yeniler). inv1/inv2 = 1 / (1 - beta^step).
struct DeviceAdamDynamic {
  float lr = 0.0f;
  float lr_wd = 0.0f;
  float inv1 = 1.0f;
  float inv2 = 1.0f;
  long long step = 1;
  long long pad = 0;
};

struct DeviceAdamWConfig {
  double lr = 1.0e-3;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps = 1.0e-8;
  double weight_decay = 0.0;
  double max_grad_norm = 0.0;  // <= 0: kirpma yok
  // ---- FAZ 4 ----
  DType shadow_type = DType::F32;      // golge kopya tipi (F32 = golge yok)
  const float* loss_scale = nullptr;   // aygit: gradyanlar bu olcekle bolunur (unscale)
  int* found_inf = nullptr;            // aygit: inf/NaN bayragi (yazilir); 1 ise adim atlanir
  // Onceki atlanan adimlarin aygit sayaci. Bias correction deneme yerine kabul edilen
  // guncelleme sayisini kullanir; host senkronu gerekmez.
  const long long* skipped_steps = nullptr;
  bool stochastic_rounding = false;    // ana kopyasiz BF16 golgede stokastik yuvarlama
  std::uint64_t sr_seed = 0;
  // ---- FAZ 13.6 ----
  const DeviceAdamDynamic* dynamic = nullptr;  // aygit: lr/lr_wd/inv1/inv2/adim (host degerleri yok sayilir)
  bool descriptors_ready = false;              // scratch'teki slot/parca tanimlayicilari onceki cagridan gecerli
};

/// Aygittaki dinamik kayip olcegi durumu (FAZ 4.3). Son 64 olcek, host senkronu
/// olmadan aygitta halka tamponda tutulur; read_loss_scale() ile metrik olarak okunur.
inline constexpr int kLossScaleHistorySize = 64;
struct DeviceLossScaleState {
  float scale = 65536.0f;
  int growth_tracker = 0;
  int found_inf = 0;
  int pad = 0;
  long long skipped_steps = 0;
  long long steps = 0;
  float scale_history[kLossScaleHistorySize] = {};
};

void loss_scale_init(DeviceLossScaleState* state, const nexus_train::LossScalePolicy& policy, cudaStream_t stream);
/// state->found_inf'e gore olcegi gunceller; nexus_train::LossScaleState::update ile ayni kurallar.
void loss_scale_update(DeviceLossScaleState* state, const nexus_train::LossScalePolicy& policy, cudaStream_t stream);
/// FAZ 9.7: kayip olceksiz NaN korumasi. steps += 1; found_inf != 0 ise skipped_steps += 1
/// (AdamW cekirdegi found_inf'te adimi atlar, bias duzeltmesi skipped_steps'i cikarir).
void nonfinite_guard_update(DeviceLossScaleState* state, cudaStream_t stream);

[[nodiscard]] std::size_t adamw_scratch_bytes(int num_slots, std::uint64_t total_numel);

// ---- FAZ 7.3: 8-bit AdamW (blok bazli dinamik nicemleme, blok = 2048) ----------------------
inline constexpr int kQ8Block = 2048;
struct Device8bitSlot {
  float* param = nullptr;   // FP32 ana kopya (nullptr: ana kopyasiz, golge guncellenir)
  void* shadow = nullptr;   // dar golge (config.shadow_type)
  const float* grad = nullptr;
  std::uint8_t* qm = nullptr;      // [numel] isaretli dinamik kod
  std::uint8_t* qv = nullptr;      // [numel] isaretsiz dinamik kod
  float* absmax_m = nullptr;       // [ceil(numel / 2048)]
  float* absmax_v = nullptr;
  std::uint64_t numel = 0;
  float weight_decay_scale = 1.0f;
  float lr_scale = 1.0f;  ///< FAZ 9.3: grup ogrenme hizi carpani (muP)
};
[[nodiscard]] std::size_t adamw8bit_scratch_bytes(int n32, int n8, std::uint64_t total32, std::uint64_t total8);
/// 32-bit ve 8-bit slotlar tek adimda: global norm/kirpma hepsi uzerinde. Momentler sifirla
/// baslar (absmax = 0 -> kod ne olursa olsun 0).
void adamw_mixed_multi_tensor(const DeviceParamSlot* slots32, int n32, const Device8bitSlot* slots8, int n8,
                              const DeviceAdamWConfig& config, std::int64_t step, float* grad_norm_out, void* scratch,
                              cudaStream_t stream);
/// FAZ 7: geri geciste katman katman guncelleme. `clip` (aygit) onceden hesaplanmis kirpma
/// carpanidir (bir adim gecikmeli); bu gradyanlarin kare toplami `norm_accum`'a (aygit, double)
/// eklenir. Scratch: adamw8bit_scratch_bytes.
void adamw_fixed_clip_update(const DeviceParamSlot* slots32, int n32, const Device8bitSlot* slots8, int n8,
                             const DeviceAdamWConfig& config, std::int64_t step, const float* clip, double* norm_accum,
                             void* scratch, cudaStream_t stream);
/// norm_accum'dan sonraki adimin kirpmasini yazar, norm_out'a normu (istege bagli), biriktiriciyi sifirlar.
void delayed_clip_finish(float* clip, float* norm_out, double* norm_accum, double max_norm, cudaStream_t stream);
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
/// FAZ 7.B selective: depolama tipinde GELU (tanh) ileri, FP32 hesap.
void gelu_forward(void* out, const void* x, DType type, std::size_t n, cudaStream_t stream);
/// out = a + b.
void add(float* out, const float* a, const float* b, std::size_t n, cudaStream_t stream);
/// y += x.
void add_inplace(float* y, const float* x, std::size_t n, cudaStream_t stream);
void fill(float* out, float value, std::size_t n, cudaStream_t stream);
/// FAZ 9.7: x[0..n) icindeki sonlu olmayan eleman sayisini *out'a ekler (atomik; *out once sifirlanmali).
void count_nonfinite(unsigned long long* out, const float* x, std::size_t n, cudaStream_t stream);
/// qkv [B, T, 3, H, D] -> q, k, v [B, H, T, D].
void split_qkv_heads(float* q, float* k, float* v, const float* qkv, int batch, int time, int heads, int head_dim,
                     cudaStream_t stream);
/// Ters: dq, dk, dv [B,H,T,D] -> dqkv [B,T,3,H,D].
void merge_qkv_heads(float* dqkv, const float* dq, const float* dk, const float* dv, int batch, int time, int heads,
                     int head_dim, cudaStream_t stream);
/// y [B, H, T, D] -> out [B, T, H*D] (ve tersi: unpermute_backward).
void merge_heads(float* out, const float* y, int batch, int time, int heads, int head_dim, cudaStream_t stream);
void split_heads(float* y, const float* in, int batch, int time, int heads, int head_dim, cudaStream_t stream);
// ---- FAZ 4 tip parametreli yardimcilar -------------------------------------------------
void cast(void* dst, DType dst_type, const void* src, DType src_type, std::size_t n, cudaStream_t stream);
void split_qkv_heads(void* q, void* k, void* v, const void* qkv, DType type, int batch, int time, int heads,
                     int head_dim, cudaStream_t stream);
void merge_qkv_heads(void* dqkv, const void* dq, const void* dk, const void* dv, DType type, int batch, int time,
                     int heads, int head_dim, cudaStream_t stream);
void merge_heads(void* out, const void* y, DType type, int batch, int time, int heads, int head_dim,
                 cudaStream_t stream);
void split_heads(void* y, const void* in, DType type, int batch, int time, int heads, int head_dim,
                 cudaStream_t stream);

/// Belirlenimci tam toplam (tek blok son asama): out[0] = sum(x).
[[nodiscard]] std::size_t sum_scratch_bytes(std::size_t n);
void sum(float* out, const float* x, std::size_t n, void* scratch, cudaStream_t stream);

}  // namespace cuda
}  // namespace nexus
