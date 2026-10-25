#pragma once

/**
 * @file lm_backend.hpp
 * @brief FAZ 9: cuda::GptModel'i dil modeli egitmeni v2'ye (train::LanguageTrainerBuilder) baglar.
 *
 *   auto trainer = train::LanguageTrainerBuilder(cuda::gpt_backend(config))
 *       .execution_policy(ExecutionPolicy::cuda(0)).precision(train::Precision::BF16)
 *       .recompute(train::RecomputePolicy::Selective).micro_batch(8).seq_len(1024)
 *       .global_batch_tokens(8 * 1024 * 4) ... .build();
 *
 * Model istekteki micro_batch / seq_len / precision / master / recompute / seed / deterministic
 * ile kurulur (config'deki ayni alanlar ezilir; vocab, katman, dikkat yolu, CE parcasi ve
 * optimizer turu config'den gelir). Biriktirmede mikro adim 0 gradyani sifirlar, sonrakiler
 * forward_backward(..., accumulate = true) ile ekler (ek tampon yok). NaN korumasi (set_nonfinite_guard) AdamW / AdamW8bit'te acilir. AdamW8bitFused desteklenmez
 * (guncelleme geri geciste, mikro adim agirligi ve biriktirme ile uyumsuz); ZeroOffload yalniz
 * accum = 1 ve NaN korumasiz calisir.
 *
 * Mikro adim: host tokenlari iki pinned yuvadan (cift tampon) model akisinda H2D -> forward_backward
 * (dlogits agirligi w_i, i > 0 ise +=) -> kayip pinned halkaya asenkron. Adim sonu tek senkron:
 * mikro kayiplar + grad norm + kayip olcegi durumu (found_inf, skipped_steps) birlikte okunur.
 * Tepe FLOPS: CC 8.6/8.9 icin SM basina saat basina 512 (BF16/F16, FP32 biriktirme, dense) ve 256
 * (TF32); CC 8.0 icin 1024 / 512; saat NVML azami SM saati (yoksa cudaDevAttrClockRate). Diger mimariler 0.
 *
 * FAZ 10.3 DDP (attach_process_group): son mikro adimda GptModel geri gecis kancasi her asama
 * (lnf, katman L-1..0, gomme) icin hesap akisina olay kaydeder; tum asamalari hazir olan kova bir
 * iletisim is parcacigina verilir: ayri akista olayi bekler -> dilimleri pinned host kovaya D2H ->
 * ProcessGroup all-reduce (host, tcp) -> H2D geri -> bitis olayi. optimizer_step tum kovalari bekler,
 * hesap akisi bitis olaylarini bekler. Iletisim geri gecisin kalaniyla ortusur. Tek GPU'yu paylasan
 * cok surecte calisir (WDDM'de CUDA IPC / NCCL yok: aktarim host uzerinden).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gpt.hpp"
#include "nexus/train/lm_trainer.hpp"

#include <memory>

namespace nexus {
namespace cuda {

[[nodiscard]] train::BackendFactory gpt_backend(GptConfig config);

/// Tepe dense Tensor Core FLOPS (0: bu mimari icin tablo yok). Saat: NVML azami SM saati (nvidia-smi
/// clocks.max.sm), NVML yoksa cudaDevAttrClockRate; clock_mhz/clock_source kullanilani dondurur.
[[nodiscard]] double tensor_core_peak_flops(int device, DType compute, double* clock_mhz = nullptr,
                                            const char** clock_source = nullptr);

class GptLmBackend final : public train::LmBackend {
 public:
  GptLmBackend(GptConfig config, const train::LmBackendRequest& request);
  ~GptLmBackend() override;

  std::string device() const override;
  train::LmShape shape() const override;
  std::vector<train::LmParamInfo> parameters() const override;
  void set_param_scales(const std::vector<float>& weight_decay, const std::vector<float>& lr) override;
  int micro_batch() const override { return model_->batch(); }
  int seq_len() const override { return model_->time(); }
  void reshape(int micro_batch, int seq_len) override;
  void micro_step(const std::int32_t* tokens, float grad_weight, int index, int count) override;
  void optimizer_step(const train::AdamW& optimizer, double lr, double clip) override;
  train::LmStepResult finish_step() override;
  bool loss_scaled() const override { return model_->uses_loss_scale(); }
  double eval_loss(const std::int32_t* tokens) override;
  std::vector<float> token_losses(const std::int32_t* tokens) override;
  train::LmNonFinite diagnose(const std::int32_t* tokens) override;
  std::vector<unsigned char> save_state() const override;
  void load_state(const std::vector<unsigned char>& blob) override;
  std::vector<train::LmNormEntry> layer_norms() override;
  std::uint64_t peak_memory_bytes() const override;
  void reset_peak_memory() override;
  double peak_flops() const override { return peak_flops_; }

  void attach_process_group(std::shared_ptr<dist::ProcessGroup> group, const dist::DdpOptions& options) override;
  void broadcast_parameters() override;
  dist::DdpStepStats ddp_stats() const override { return stats_; }

  [[nodiscard]] GptModel& model() noexcept { return *model_; }

 private:
  const std::int32_t* upload(const std::int32_t* tokens);
  struct Ddp;
  void on_stage(int stage);
  void finish_sync();
  void plan_buckets();
  std::unique_ptr<Ddp> ddp_;
  dist::DdpStepStats stats_{};

  int device_ = 0;
  Stream stream_;
  std::unique_ptr<GptModel> model_;
  std::size_t tokens_per_micro_ = 0;
  int slot_ = 0;
  std::int32_t* pinned_tokens_[2] = {nullptr, nullptr};
  DeviceBuffer device_tokens_[2];
  Event copy_done_[2];
  bool copy_pending_[2] = {false, false};
  float* pinned_losses_ = nullptr;   // [kMaxAccum]
  struct StepTail {
    float grad_norm;
    DeviceLossScaleState scale;
  };
  StepTail* pinned_tail_ = nullptr;
  Event step_done_;
  int count_ = 0;
  double peak_flops_ = 0.0;
  static constexpr int kMaxAccum = 4096;
};

}  // namespace cuda
}  // namespace nexus
