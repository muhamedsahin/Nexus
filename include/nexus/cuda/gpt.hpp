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
 * Kapsam disi (sonraki fazlar): BF16/FP16 (FAZ 4), FlashAttention (FAZ 5; burada
 * dikkat skoru [B,H,T,T] yazilir), aktivasyon yeniden hesaplama (FAZ 7), gradyan
 * biriktirme (FAZ 9).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/cuda/runtime.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace cuda {

struct GptConfig {
  int vocab = 50257;
  int max_seq = 1024;
  int layers = 12;
  int heads = 12;
  int dim = 768;
  float ln_eps = 1.0e-5f;
  bool deterministic = true;  // gomme geri: siralama+segment (atomiksiz)
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

  /// d_tokens: aygitta [batch*time]. Kayip aygitta yazilir (host senkronu yok).
  /// Gradyanlar sifirlanip yeniden hesaplanir.
  void forward_backward(const std::int32_t* d_tokens);
  /// Yalniz ileri + kayip (degerlendirme).
  void forward(const std::int32_t* d_tokens);
  /// Foreach AdamW; agirlik sonumu yalniz 2B matrislere. grad norm aygitta.
  void optimizer_step(const DeviceAdamWConfig& config);

  [[nodiscard]] const float* device_loss() const noexcept { return loss_.as<float>(); }
  [[nodiscard]] const float* device_grad_norm() const noexcept { return grad_norm_.as<float>(); }
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

  GptConfig config_;
  int batch_ = 0;
  int time_ = 0;
  Stream stream_;
  std::vector<GptParamInfo> params_info_;
  std::size_t num_params_ = 0;
  std::size_t activation_bytes_ = 0;
  std::int64_t step_ = 0;
  DeviceBuffer params_, grads_, exp_avg_, exp_avg_sq_, loss_, grad_norm_;
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
