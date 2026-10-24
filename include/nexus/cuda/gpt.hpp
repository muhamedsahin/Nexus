#pragma once

/**
 * @file gpt.hpp
 * @brief FAZ 3.8/3.I: tam GPU egitim yolu — GPT-2 mimarisi, FP32/TF32, tek kart.
 *
 * Parametre, gradyan, AdamW durumu, aktivasyon ve logit kartta yasar; ileri ve geri
 * gecisteki her islem kartta calisir. Host'a yalniz skaler kayip doner (asenkron,
 * pinned bellek, N adimda bir okunur). Tokenlar pinned host'tan cift tamponla
 * asenkron H2D kopyalanir.
 *
 * Mimari (GPT-2, Radford ve ark. 2019): wte[tok] + wpe[pos] -> L x [LN -> QKV ->
 * nedensel dikkat -> proj -> +res -> LN -> fc (GELU tanh) -> proj -> +res] -> LN ->
 * bagli (tied) wte^T kafa -> kaydirilmis CE.
 *
 * Kullanilan cekirdekler: cuBLASLt (bias, bias+GELU+aux, DGELU, BGRAD epiloglari,
 * strided batched dikkat GEMM'leri), Welford LayerNorm + add&norm, cekirdek ici
 * nedensel maske, belirlenimci gomme geri, yerinde fuzyonlu CE, foreach AdamW +
 * aygitta global norm kirpma.
 *
 * Karisik hassasiyet (FAZ 4): `compute_dtype` BF16/F16 iken matmul ve dikkat dar tipte
 * Tensor Core'da calisir (autocast tablosu, nexus/amp/autocast.hpp); softmax, CE ve
 * norm istatistikleri FP32 biriktirir; residual akisi, logit ve gradyanlar FP32'dir.
 * Ana kopya modlari (4.2):
 *   A CastPerForward: FP32 parametre, her ileri geciste dar kopyaya cevrilir (PyTorch AMP).
 *   B Fp32Master (varsayilan): dar model agirligi + optimizer'da FP32 ana kopya; AdamW
 *     guncellemeyi FP32'de yapip dar kopyayi ayni cekirdekte yazar (Megatron/DeepSpeed).
 *   C NoMaster (deneysel): matris agirliklari yalniz BF16, stokastik yuvarlamali guncelleme;
 *     kucuk parametreler (LN, bias) FP32 kalir.
 * F16'da dinamik kayip olcegi aygitta tutulur; unscale + isfinite AdamW'in norm gecisinde
 * tek bayrakla yapilir, inf/NaN adimi atlanir (host senkronu yok).
 *
 * Kapsam disi (sonraki fazlar): FlashAttention (FAZ 5; burada
 * dikkat skoru [B,H,T,T] yazilir), aktivasyon yeniden hesaplama (FAZ 7), gradyan
 * biriktirme (FAZ 9).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/cuda/runtime.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace cuda {

enum class MasterWeights : std::uint8_t { CastPerForward = 0, Fp32Master = 1, NoMaster = 2 };

[[nodiscard]] const char* to_string(MasterWeights mode) noexcept;

enum class AttentionBackend { Auto, Math, Flash };

[[nodiscard]] const char* to_string(AttentionBackend backend) noexcept;

struct GptConfig {
  int vocab = 50257;
  int max_seq = 1024;
  int layers = 12;
  int heads = 12;
  int dim = 768;
  float ln_eps = 1.0e-5f;
  bool deterministic = true;  // gomme geri: siralama+segment (atomiksiz)
  // ---- FAZ 4 ----
  DType compute_dtype = DType::F32;                 // F32 (TF32 politikasi globaldir), BF16, F16
  MasterWeights master = MasterWeights::Fp32Master;  // yalniz dar compute_dtype'ta anlamli
  bool dynamic_loss_scale = false;                  // F16'da zorunlu olarak acilir
  nexus_train::LossScalePolicy loss_scale{};
  // ---- FAZ 5 ----
  // Auto: dar compute_dtype + flash_supported ise FlashAttention-2, yoksa math (bmm+softmax,
  // [L,B,H,T,T] olasilik tamponu). Secim attention_backend() ile okunur; NEXUS_SDPA_DEBUG=1
  // math'e dusme nedenini stderr'e yazar. Flash istenip desteklenmezse kurucu hata atar.
  AttentionBackend attention = AttentionBackend::Auto;
  bool deterministic_attention = false;  // flash geri: atomiksiz dQ (T/64 kat dQ tamponu)
};

struct GptParamInfo {
  std::string name;
  std::vector<int> shape;  // satir-ana; matrisler [giris, cikis]
  std::size_t offset = 0;  // eleman ofseti (tek bitisik tampon)
  std::size_t numel = 0;
  bool decay = false;      // agirlik sonumu (2B matrisler)
};

class GptModel {
 public:
  /// batch x time mikro-batch icin tum tamponlari ayirir. Agirliklar GPT-2 baslatmasiyla
  /// (N(0, 0.02), artik projeksiyonlar 0.02/sqrt(2L), LN=1/0, bias=0) host'ta tohumla
  /// uretilip yuklenir.
  GptModel(const GptConfig& config, int batch, int time, std::uint64_t seed, const Stream& stream);
  ~GptModel();
  GptModel(const GptModel&) = delete;
  GptModel& operator=(const GptModel&) = delete;

  [[nodiscard]] const GptConfig& config() const noexcept { return config_; }
  [[nodiscard]] int batch() const noexcept { return batch_; }
  [[nodiscard]] int time() const noexcept { return time_; }
  [[nodiscard]] std::size_t num_parameters() const noexcept { return num_params_; }
  [[nodiscard]] const std::vector<GptParamInfo>& parameter_info() const noexcept { return params_info_; }
  [[nodiscard]] std::size_t activation_bytes() const noexcept { return activation_bytes_; }
  /// Parametre + gradyan + optimizer durumu baytlari (tum ana kopya/golge tamponlari dahil).
  [[nodiscard]] std::size_t state_bytes() const noexcept { return state_bytes_; }

  /// d_tokens: aygitta [batch*time]. Kayip aygitta yazilir (host senkronu yok).
  /// Gradyanlar sifirlanip yeniden hesaplanir.
  void forward_backward(const std::int32_t* d_tokens);
  /// Yalniz ileri + kayip (degerlendirme).
  void forward(const std::int32_t* d_tokens);
  /// Foreach AdamW; agirlik sonumu yalniz 2B matrislere. grad norm aygitta.
  void optimizer_step(const DeviceAdamWConfig& config);

  [[nodiscard]] const float* device_loss() const noexcept { return loss_.as<float>(); }
  [[nodiscard]] const float* device_grad_norm() const noexcept { return grad_norm_.as<float>(); }
  /// FP32 gradyan tamponu (test: tasma enjeksiyonu).
  [[nodiscard]] float* device_gradients() noexcept { return grads_.as<float>(); }
  /// Kayip olcegi durumu (dinamik olcek kapaliysa olcek 1, sayaclar 0). Senkron okuma.
  [[nodiscard]] DeviceLossScaleState read_loss_scale() const;
  [[nodiscard]] bool uses_loss_scale() const noexcept { return use_loss_scale_; }
  /// Gercekte kullanilan dikkat yolu (Flash ya da Math; Auto asla).
  [[nodiscard]] AttentionBackend attention_backend() const noexcept { return flash_ ? AttentionBackend::Flash : AttentionBackend::Math; }
  [[nodiscard]] std::int64_t step_count() const noexcept { return step_; }
  /// Modelin tum islerini kuyruga koydugu hesap akisi.
  [[nodiscard]] const Stream& stream() const noexcept { return stream_; }

  // Host kopyalari (parity/test icin; senkron).
  [[nodiscard]] std::vector<float> download_parameters() const;
  void upload_parameters(const std::vector<float>& values);
  [[nodiscard]] std::vector<float> download_gradients() const;
  [[nodiscard]] float read_loss() const;

 private:
  struct Buffers;
  void build_layout();
  void run_forward(const std::int32_t* d_tokens, bool for_backward);
  void run_backward(const std::int32_t* d_tokens);
  [[nodiscard]] bool narrow() const noexcept { return config_.compute_dtype != DType::F32; }
  [[nodiscard]] float* master(std::size_t offset) const;      // FP32 ana kopya (yoksa nullptr)
  [[nodiscard]] void* shadow(std::size_t offset) const;       // dar kopya (yoksa nullptr)
  [[nodiscard]] const void* weight(std::size_t offset) const;  // GEMM girisi (compute_dtype)
  void refresh_shadow_from_master();

  GptConfig config_;
  int batch_ = 0;
  int time_ = 0;
  Stream stream_;
  std::vector<GptParamInfo> params_info_;
  std::size_t num_params_ = 0;
  std::size_t activation_bytes_ = 0;
  std::size_t state_bytes_ = 0;
  std::size_t small_begin_ = 0;   // kucuk (bias/LN) parametrelerin basladigi ofset
  std::size_t master_begin_ = 0;  // ana kopyanin kapsadigi ilk ofset (C modunda small_begin_)
  bool use_loss_scale_ = false;
  bool flash_ = false;
  std::int64_t step_ = 0;
  DeviceBuffer params_, shadow_, grads_, exp_avg_, exp_avg_sq_, loss_, grad_norm_, loss_scale_state_;
  std::unique_ptr<Buffers> buffers_;
};

/// Egitmen: pinned cift tamponlu asenkron token kopyasi + N adimda bir kayip okuma.
class GptTrainer {
 public:
  GptTrainer(GptModel& model, DeviceAdamWConfig optimizer, int loss_read_interval = 1);
  ~GptTrainer();
  GptTrainer(const GptTrainer&) = delete;
  GptTrainer& operator=(const GptTrainer&) = delete;

  /// tokens: host [batch*time]. Adimi kuyruga koyar ve hemen doner. Kayip okuma
  /// araliginda bir senkron noktasi olusur; donus son okunan kayiptir (henuz yoksa NaN).
  float step(const std::int32_t* tokens);
  /// Kuyruktaki tum isin bitmesini bekler ve son kaybi dondurur.
  float synchronize();

 private:
  GptModel* model_;
  DeviceAdamWConfig optimizer_;
  int interval_;
  int slot_ = 0;
  std::int64_t steps_ = 0;
  float last_loss_ = 0.0f;
  bool has_loss_ = false;
  std::int32_t* pinned_tokens_[2] = {nullptr, nullptr};
  float* pinned_loss_ = nullptr;
  DeviceBuffer device_tokens_[2];
  Event copy_done_[2];
  Event loss_ready_;
  bool loss_pending_ = false;
};

}  // namespace cuda
}  // namespace nexus
