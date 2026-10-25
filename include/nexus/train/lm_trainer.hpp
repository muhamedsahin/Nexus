#pragma once

/**
 * @file lm_trainer.hpp
 * @brief FAZ 9: dil modeli egitmeni v2 — gradyan biriktirme, LR zamanlayici, global norm kirpma,
 *        parametre gruplari, degerlendirme, olcum (token/s, MFU, tepe bellek), NaN/sicrama teshisi.
 *
 * Tek surec (tek kart ya da CPU). Model "arka uc" (LmBackend) arkasindadir:
 *   - CPU: nn::Module (LanguageTrainerBuilder(nn::Module&)), FP32, NexusOptim AdamW.
 *   - CUDA: cuda::GptModel (nexus/cuda/lm_backend.hpp, cuda::gpt_backend(config)).
 * Dongu (her optimizer adimi):
 *   1. accum = global_batch_tokens / (micro_batch * seq_len) mikro yigin kaynaktan alinir
 *      (host; satir basina ornek kimligi saklanir).
 *   2. Mikro adim i: ileri + geri, dlogits agirligi w_i = gecerli_i / toplam_gecerli (token
 *      agirlikli ortalama; esit uzunlukta 1/accum). Gradyan += birikir.
 *   3. Son mikro adimdan sonra: global norm kirpma + AdamW (LR = taban * zamanlayici(adim) *
 *      grup carpani). inf/NaN gradyanda adim aygitta atlanir (parametre/moment degismez).
 *   4. Sonraki adimin verisi aygit calisirken hazirlanir; sonra adim sonucu (mikro kayiplar,
 *      grad norm, atlandi mi) tek senkronla okunur.
 * "Adim" her zaman optimizer adimidir. Eski LanguageTrainer::fit(rows, epochs) bu donguye
 * yonlenir: "epoch" = ayni yiginda bir optimizer adimi (biriktirme yok).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/config.hpp"
#include "nexus/train/schedule.hpp"
#include "nexus_train/logging/logger_base.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nexusdata {
namespace text {
class TokenStream;
}  // namespace text
}  // namespace nexusdata

namespace nexus {
namespace nn {
class Module;
}  // namespace nn

namespace train {

// ------------------------------------------------------------------------------------------
// Yapilandirma
// ------------------------------------------------------------------------------------------

struct AdamW {
  double lr = 6.0e-4;
  double beta1 = 0.9;
  double beta2 = 0.95;
  double eps = 1.0e-8;
  double weight_decay = 0.1;
};

/// Ad onekiyle grup kurali (muP LR carpani vb.). weight_decay_scale < 0: politikadan.
struct ParamGroupRule {
  std::string prefix;
  float lr_scale = 1.0f;
  float weight_decay_scale = -1.0f;
};

struct ParamGroupPolicy {
  enum class Decay { MatricesOnly, All, None };
  /// MatricesOnly (varsayilan, nanoGPT): >= 2 boyutlu agirliklar sonumlu; bias ve norm 0.
  /// All: her parametre (eski LanguageTrainer davranisi). None: hic.
  Decay decay = Decay::MatricesOnly;
  bool decay_embeddings = true;  // false: gomme tablolari (wte/wpe, nn::Embedding) sonumsuz
  std::vector<ParamGroupRule> rules;
};

enum class Precision { F32, BF16, F16 };
enum class MasterWeightsPolicy { Fp32, None, CastPerForward };
enum class RecomputePolicy { None, Selective, Full };

enum class NanPolicy {
  Skip,      // adim atlanir, egitim surer (art arda max_consecutive_skips'ten fazlasi: durur)
  Stop,      // egitim durur (TrainSummary::stopped)
  Rollback,  // son anlik goruntuye donulur, LR carpilir; veri ileri devam eder (kotu yigin atlanir)
};

struct NanOptions {
  NanPolicy policy = NanPolicy::Skip;
  bool diagnose = true;                // anomali modunda adimi yeniden calistir, ilk NaN katmanini bul
  std::int64_t snapshot_every = 100;   // Rollback: host anlik goruntu araligi (adim)
  double rollback_lr_factor = 0.5;
  int max_rollbacks = 3;
  int max_consecutive_skips = 16;
};

struct SpikeOptions {
  bool enabled = true;
  int window = 64;          // hareketli medyan penceresi (adim)
  double k = 6.0;           // esik: medyan + k * 1.4826 * MAD
  std::int64_t min_steps = 16;
};

// ------------------------------------------------------------------------------------------
// Arka uc arayuzu
// ------------------------------------------------------------------------------------------

struct LmParamInfo {
  std::string name;
  std::size_t numel = 0;
  int rank = 1;
  bool embedding = false;  // gomme tablosu (bagli kafa dahil)
};

/// MFU icin model sekli (BOLUM D: 6 N + 12 L H d_head T). Bilinmiyorsa layers = 0.
struct LmShape {
  std::int64_t params = 0;          // toplam
  std::int64_t params_no_pos = 0;   // konum gommesi haric (nanoGPT get_num_params)
  int layers = 0;
  int heads = 0;
  int head_dim = 0;
};

struct LmBackendRequest {
  int micro_batch = 1;
  int seq_len = 2;
  Precision precision = Precision::F32;
  MasterWeightsPolicy master = MasterWeightsPolicy::Fp32;
  RecomputePolicy recompute = RecomputePolicy::None;
  ExecutionPolicy policy = ExecutionPolicy::cpu();
  std::uint64_t seed = 1337;
  bool deterministic = true;
  int accum_steps = 1;
};

struct LmStepResult {
  std::vector<float> losses;  // mikro adim basina ortalama kayip
  double grad_norm = 0.0;     // kirpma oncesi global norm
  bool skipped = false;       // inf/NaN (ya da F16 tasma) yuzunden atlandi
  std::int64_t skipped_total = 0;
  double loss_scale = 1.0;
};

struct LmNonFinite {
  std::string first_activation;  // ilk sonlu olmayan cikti ("h.3"); bos: ileri gecis sonlu
  std::vector<std::string> parameters;
  std::vector<std::string> gradients;
  float loss = 0.0f;
  int micro_batch = -1;          // tekrar ureten ilk mikro yigin (-1: hicbiri)
  bool reproduced = false;
};

struct LmNormEntry {
  std::string name;
  double weight = 0.0;
  double grad = 0.0;
};

class LmBackend {
 public:
  virtual ~LmBackend() = default;
  [[nodiscard]] virtual std::string device() const = 0;
  [[nodiscard]] virtual LmShape shape() const = 0;
  [[nodiscard]] virtual std::vector<LmParamInfo> parameters() const = 0;
  /// parameters() sirasiyla sonum ve LR carpanlari (egitim baslamadan cagrilir).
  virtual void set_param_scales(const std::vector<float>& weight_decay, const std::vector<float>& lr) = 0;
  [[nodiscard]] virtual int micro_batch() const = 0;
  [[nodiscard]] virtual int seq_len() const = 0;
  /// Eski API icin: CPU arka ucu yigin boyutunu degistirebilir; GPU sabit tamponludur.
  virtual void reshape(int micro_batch, int seq_len) = 0;
  /// tokens: host [micro_batch * seq_len]. index/count: biriktirme penceresindeki yer.
  virtual void micro_step(const std::int32_t* tokens, float grad_weight, int index, int count) = 0;
  /// Kirpma + AdamW (asenkron olabilir). lr: bu adimin taban LR'si (zamanlayici dahil).
  virtual void optimizer_step(const AdamW& optimizer, double lr, double clip) = 0;
  /// Adimi bekler ve sonucu dondurur.
  [[nodiscard]] virtual LmStepResult finish_step() = 0;
  [[nodiscard]] virtual bool loss_scaled() const { return false; }
  /// Yalniz ileri; gecerli hedefler uzerinde ortalama kayip (gradyan/aktivasyon saklama yok).
  [[nodiscard]] virtual double eval_loss(const std::int32_t* tokens) = 0;
  /// Konum basina kayip [micro_batch * seq_len] (hedef tokens[b, t+1]; son konum 0).
  [[nodiscard]] virtual std::vector<float> token_losses(const std::int32_t* tokens) = 0;
  [[nodiscard]] virtual LmNonFinite diagnose(const std::int32_t* tokens) = 0;
  [[nodiscard]] virtual std::vector<unsigned char> save_state() const = 0;
  virtual void load_state(const std::vector<unsigned char>& blob) = 0;
  [[nodiscard]] virtual std::vector<LmNormEntry> layer_norms() = 0;
  /// Tepe aygit bellegi (ayirici + modelin onceden ayirdigi; CPU: 0 = olculmez).
  [[nodiscard]] virtual std::uint64_t peak_memory_bytes() const = 0;
  virtual void reset_peak_memory() = 0;
  /// Kullanilan sayi bicimi icin tepe dense FLOPS (0: bilinmiyor -> MFU NaN).
  [[nodiscard]] virtual double peak_flops() const = 0;
};

using BackendFactory = std::function<std::unique_ptr<LmBackend>(const LmBackendRequest&)>;

/// CPU arka ucu: nn::Module, FP32, NexusOptim AdamW (grup basina sonum ve LR).
[[nodiscard]] BackendFactory cpu_backend(nn::Module& model);

// ------------------------------------------------------------------------------------------
// Veri
// ------------------------------------------------------------------------------------------

class TokenBatchSource {
 public:
  virtual ~TokenBatchSource() = default;
  /// dst: [rows * time]; ids: satir basina ornek kimligi (sicrama kaydi icin).
  virtual void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) = 0;
  [[nodiscard]] virtual std::string state() const = 0;
  virtual void restore(const std::string& state) = 0;
};

/// nexusdata TokenStream: satir = ornegin ilk `time` tokeni (pencere + 1 >= time olmali).
class StreamSource final : public TokenBatchSource {
 public:
  explicit StreamSource(nexusdata::text::TokenStream& stream);
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override;
  [[nodiscard]] std::string state() const override;
  void restore(const std::string& state) override;

 private:
  nexusdata::text::TokenStream* stream_;
};

/// Sabit satirlar (eski API ve testler): her cagri ayni yigini dondurur.
class FixedRowsSource final : public TokenBatchSource {
 public:
  FixedRowsSource(std::vector<std::int32_t> tokens, int rows, int time);
  void set(std::vector<std::int32_t> tokens, int rows, int time);
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override;
  [[nodiscard]] std::string state() const override { return std::to_string(calls_); }
  void restore(const std::string& state) override { calls_ = std::stoull(state); }

 private:
  std::vector<std::int32_t> tokens_;
  int rows_ = 0;
  int time_ = 0;
  std::uint64_t calls_ = 0;
};

// ------------------------------------------------------------------------------------------
// Kayitlar
// ------------------------------------------------------------------------------------------

struct StepRecord {
  std::int64_t step = 0;
  double loss = 0.0;          // token agirlikli adim kaybi
  double lr = 0.0;
  double grad_norm = 0.0;
  double tokens_per_s = 0.0;
  double mfu = 0.0;           // NaN: tepe FLOPS bilinmiyor
  double step_ms = 0.0;
  double data_ms = 0.0;       // host'ta yigin hazirlama (aygit calisirken)
  std::uint64_t peak_memory = 0;
  bool skipped = false;
  std::int64_t skipped_total = 0;
  double val_loss = 0.0;      // NaN: bu adimda degerlendirme yok
};

struct SpikeEvent {
  std::int64_t step = 0;
  double loss = 0.0;
  double median = 0.0;
  double mad = 0.0;
  std::vector<std::uint64_t> sample_ids;  // adimdaki tum satirlarin ornek kimlikleri
};

struct NanEvent {
  std::int64_t step = 0;
  double loss = 0.0;
  double grad_norm = 0.0;
  LmNonFinite report;
  std::string action;  // "skip" | "stop" | "rollback -> adim N, lr x0.5"
};

struct EvalResult {
  double loss = 0.0;
  double perplexity = 0.0;
  std::int64_t tokens = 0;
};

struct TrainSummary {
  std::int64_t steps = 0;       // son adim sayaci
  double last_loss = 0.0;
  bool stopped = false;
  std::string reason;
};

/// HellaSwag tarzi coktan secmeli: baglam + her secenek; puan = secenek tokenlarinin log-olasilik toplami.
struct MultipleChoiceItem {
  std::vector<std::int32_t> context;
  std::vector<std::vector<std::int32_t>> choices;
};

// ------------------------------------------------------------------------------------------
// Egitmen
// ------------------------------------------------------------------------------------------

class LanguageModelTrainer;

class LanguageTrainerBuilder {
 public:
  explicit LanguageTrainerBuilder(nn::Module& model);
  explicit LanguageTrainerBuilder(BackendFactory factory);

  LanguageTrainerBuilder& data(TokenBatchSource& train, TokenBatchSource* validation = nullptr);
  LanguageTrainerBuilder& data(nexusdata::text::TokenStream& train, nexusdata::text::TokenStream* validation = nullptr);
  LanguageTrainerBuilder& optimizer(AdamW config, ParamGroupPolicy groups = {});
  LanguageTrainerBuilder& schedule(LrSchedule schedule);
  LanguageTrainerBuilder& global_batch_tokens(std::int64_t tokens);
  LanguageTrainerBuilder& micro_batch(int rows);
  LanguageTrainerBuilder& seq_len(int time);
  LanguageTrainerBuilder& grad_clip(double max_norm);
  LanguageTrainerBuilder& precision(Precision precision, MasterWeightsPolicy master = MasterWeightsPolicy::Fp32);
  LanguageTrainerBuilder& recompute(RecomputePolicy policy);
  LanguageTrainerBuilder& eval_every(std::int64_t steps, std::int64_t eval_tokens);
  LanguageTrainerBuilder& log_every(std::int64_t steps);
  LanguageTrainerBuilder& execution_policy(ExecutionPolicy policy);
  LanguageTrainerBuilder& seed(std::uint64_t seed);
  LanguageTrainerBuilder& deterministic(bool enabled);
  LanguageTrainerBuilder& nan_policy(NanOptions options);
  LanguageTrainerBuilder& spike_detector(SpikeOptions options);
  LanguageTrainerBuilder& histogram_every(std::int64_t steps);
  LanguageTrainerBuilder& logger(std::shared_ptr<nexus_train::LoggerBase> sink);
  LanguageTrainerBuilder& log_console(bool enabled);
  LanguageTrainerBuilder& log_csv(const std::string& path);
  LanguageTrainerBuilder& log_jsonl(const std::string& path);
  LanguageTrainerBuilder& log_tensorboard(const std::string& directory);
  LanguageTrainerBuilder& log_wandb(const std::string& path);
  /// Test/teshis kancasi: tum mikro adimlardan sonra, optimizer adimindan once.
  LanguageTrainerBuilder& after_backward(std::function<void(LmBackend&, std::int64_t step)> hook);

  [[nodiscard]] std::unique_ptr<LanguageModelTrainer> build();

 private:
  friend class LanguageModelTrainer;
  BackendFactory factory_;
  bool cpu_module_ = false;
  TokenBatchSource* train_ = nullptr;
  TokenBatchSource* val_ = nullptr;
  std::unique_ptr<TokenBatchSource> owned_train_;
  std::unique_ptr<TokenBatchSource> owned_val_;
  AdamW adamw_{};
  ParamGroupPolicy groups_{};
  LrSchedule schedule_{};
  std::int64_t global_batch_tokens_ = 0;
  LmBackendRequest request_{};
  double clip_ = 0.0;
  std::int64_t eval_every_ = 0;
  std::int64_t eval_tokens_ = 0;
  std::int64_t log_every_ = 1;
  NanOptions nan_{};
  SpikeOptions spike_{};
  std::int64_t histogram_every_ = 0;
  bool console_ = true;
  std::vector<std::shared_ptr<nexus_train::LoggerBase>> loggers_;
  std::function<void(LmBackend&, std::int64_t)> after_backward_;
};

class LanguageModelTrainer {
 public:
  ~LanguageModelTrainer();
  LanguageModelTrainer(const LanguageModelTrainer&) = delete;
  LanguageModelTrainer& operator=(const LanguageModelTrainer&) = delete;

  /// Adim sayaci max_steps olana kadar egitir (devamda kaldigi yerden).
  TrainSummary train(std::int64_t max_steps);
  /// Sabit dogrulama kumesinde (her cagrida ayni tokenlar) kayip ve perplexity.
  [[nodiscard]] EvalResult evaluate();
  /// Secenek basina log-olasilik toplami (buyuk = daha olasi).
  [[nodiscard]] std::vector<double> score_choices(const MultipleChoiceItem& item);
  /// Model + optimizer + kaynak + adim durumu (FAZ 15 tam kayit sistemine kadar tek dosya).
  void save(const std::string& path) const;
  void resume(const std::string& path);
  /// Eski API: yigin boyutu degisirse (yalniz CPU arka ucu).
  void reshape(int micro_batch, int seq_len);

  [[nodiscard]] std::int64_t step() const noexcept { return step_; }
  [[nodiscard]] int accum_steps() const noexcept { return accum_; }
  [[nodiscard]] std::int64_t tokens_per_step() const noexcept;
  [[nodiscard]] LmBackend& backend() noexcept { return *backend_; }
  [[nodiscard]] const std::vector<StepRecord>& history() const noexcept { return history_; }
  [[nodiscard]] const std::vector<SpikeEvent>& spikes() const noexcept { return spikes_; }
  [[nodiscard]] const std::vector<NanEvent>& nan_events() const noexcept { return nan_events_; }
  [[nodiscard]] const std::vector<std::vector<LmNormEntry>>& norm_history() const noexcept { return norms_; }
  [[nodiscard]] double lr_factor() const noexcept { return lr_factor_; }
  [[nodiscard]] double flops_per_token() const;

 private:
  friend class LanguageTrainerBuilder;
  explicit LanguageModelTrainer(LanguageTrainerBuilder& builder);
  void fetch(std::vector<std::int32_t>& tokens, std::vector<std::uint64_t>& ids);
  void log(const StepRecord& record, const std::vector<LmNormEntry>* norms);
  void take_snapshot();
  bool detect_spike(double loss);

  struct Snapshot {
    std::vector<unsigned char> backend;
    std::int64_t step = -1;
  };

  std::unique_ptr<LmBackend> backend_;
  TokenBatchSource* train_ = nullptr;
  TokenBatchSource* val_ = nullptr;
  std::unique_ptr<TokenBatchSource> owned_train_;
  std::unique_ptr<TokenBatchSource> owned_val_;
  AdamW adamw_{};
  LrSchedule schedule_{};
  int micro_batch_ = 1;
  int seq_len_ = 2;
  int accum_ = 1;
  double clip_ = 0.0;
  std::int64_t eval_every_ = 0;
  std::int64_t eval_batches_ = 0;
  std::string val_origin_;  // sabit dogrulama kumesinin baslangic durumu
  std::int64_t log_every_ = 1;
  NanOptions nan_{};
  SpikeOptions spike_{};
  std::int64_t histogram_every_ = 0;
  bool console_ = true;
  std::vector<std::shared_ptr<nexus_train::LoggerBase>> loggers_;
  std::function<void(LmBackend&, std::int64_t)> after_backward_;

  std::int64_t step_ = 0;
  double lr_factor_ = 1.0;
  int rollbacks_ = 0;
  int consecutive_skips_ = 0;
  bool prefetched_ = false;
  std::vector<std::int32_t> tokens_, next_tokens_;
  std::vector<std::uint64_t> ids_, next_ids_;
  double next_data_ms_ = 0.0;
  std::string prefetch_state_;  // prefetch oncesi kaynak durumu (save icin)
  Snapshot snapshot_;
  std::deque<double> recent_;
  std::vector<StepRecord> history_;
  std::vector<SpikeEvent> spikes_;
  std::vector<NanEvent> nan_events_;
  std::vector<std::vector<LmNormEntry>> norms_;
};

}  // namespace train
}  // namespace nexus
