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
 * FAZ 5 FlashAttention, FAZ 7 yeniden hesaplama, FAZ 9 gradyan biriktirme / NaN korumasi /
 * parametre grubu carpanlari / durum kopyasi bu sinifa eklendi (asagidaki API).
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
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace cuda {

enum class MasterWeights : std::uint8_t { CastPerForward = 0, Fp32Master = 1, NoMaster = 2 };

[[nodiscard]] const char* to_string(MasterWeights mode) noexcept;

enum class AttentionBackend { Auto, Math, Flash };

/// FAZ 7.1: katman aktivasyonlarinin geri gecis icin saklanmasi.
///  None: hepsi saklanir. Full: yalniz katman girisi (res) + kucuk istatistikler; geri geciste
///  katman ileri gecisi yeniden calisir. EveryK: l % k == 0 olan katmanlar Full gibi, digerleri
///  None gibi. Selective: GEMM ciktilari (qkv, att, fc_pre) ve xmid saklanir; norm ciktilari
///  (ln1, ln2) ve GELU ciktisi (fc_act) yeniden hesaplanir (yalniz flash dikkatle).
enum class Recompute { None, Selective, Full, EveryK };
/// FAZ 7.3: AdamW (FP32 durum, aygit) | AdamW8bit (blok bazli 8-bit m/v, aygit) |
/// ZeroOffload (FP32 ana kopya + m + v host'ta, gradyan katman katman D2H, CPU AdamW, BF16 H2D).
enum class OptimizerKind { AdamW, AdamW8bit, ZeroOffload, AdamW8bitFused };
// AdamW8bitFused: 8-bit AdamW geri geciste katman katman uygulanir (LOMO / PyTorch
// post-accumulate-grad hook tarzi): katman matris gradyani tek katmanlik tampona yazilir ve hemen
// guncellenir, tam gradyan tamponu yoktur. Kirpma bir adim gecikmeli (onceki adimin global
// normu). BF16 + ana kopyasiz (C) zorunlu; set_fused_optimizer ile hiperparametre verilir,
// forward_backward adimi atar, optimizer_step yalniz sonraki kirpmayi sonlandirir.

[[nodiscard]] const char* to_string(AttentionBackend backend) noexcept;
[[nodiscard]] const char* to_string(Recompute mode) noexcept;
[[nodiscard]] const char* to_string(OptimizerKind kind) noexcept;

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
  // ---- FAZ 7 ----
  Recompute recompute = Recompute::None;
  int recompute_every = 2;
  int ce_chunk_rows = 0;
  // ---- FAZ 13 ----
  // GEMM boyut dolgusu (inductor pad_mm karsiligi): wte/lm_head satirlari ve logit adimi bu katlara
  // yukseltilir. Dolgu satirlari sifirla baslar ve gradyan almaz (CE dolgu sutunlarini yok sayar,
  // dlogits 0); kayip ve gercek parametrelerin matematigi ayni. 1: dolgu yok.
  int vocab_pad_multiple = 1;
  // FAZ 13: BF16'da CE dlogits'i dogrudan dar tipte yazar (ayri cast cekirdegi + FP32 yazimi yok; ayni RN
  // yuvarlama, sonuc bit bit ayni). Yalniz parcasiz CE (ce_chunk_rows = 0).
  bool fuse_ce_cast = false;
  bool fast_ce = false;  // FAZ 13: cevrimici softmax CE cekirdegi (toplama sirasi farkli; varsayilan FAZ 3 sirasi)
  [[nodiscard]] int padded_vocab() const noexcept {
    return vocab_pad_multiple > 1 ? (vocab + vocab_pad_multiple - 1) / vocab_pad_multiple * vocab_pad_multiple : vocab;
  }  // > 0: lm_head GEMM + CE B*T ekseninde bu satirlik parcalarla (logit tamponu sabit)
  OptimizerKind optimizer = OptimizerKind::AdamW;
  bool adam8_embedding_fp32 = true;        // 8-bit: wte/wpe durumu 32-bit (bitsandbytes StableEmbedding onerisi)
  DType offload_grad_dtype = DType::BF16;  // ZeroOffload: D2H gradyan tipi (F32: GPU AdamW ile birebir test)
  int offload_threads = 0;                 // ZeroOffload CPU AdamW is parcacigi (0: donanim)
};

struct GptParamInfo {
  std::string name;
  std::vector<int> shape;  // satir-ana; matrisler [giris, cikis]
  std::size_t offset = 0;  // eleman ofseti (tek bitisik tampon)
  std::size_t numel = 0;
  bool decay = false;      // agirlik sonumu (2B matrisler)
};

/// FAZ 9.3: parametre basina optimizer carpanlari (parameter_info sirasiyla).
struct GptParamScale {
  float weight_decay = 1.0f;  // DeviceAdamWConfig::weight_decay carpani (0: sonum yok)
  float lr = 1.0f;            // ogrenme hizi carpani (muP gruplari)
};

/// FAZ 9.7: sonlu olmayan deger teshisi (ileri: katman ciktilari, geri: gradyanlar).
struct GptNonFiniteReport {
  std::string first_activation;              // ilk sonlu olmayan cikti: "embedding", "h.3", "lm_head"; bos: hepsi sonlu
  std::vector<std::string> parameters;       // sonlu olmayan agirliklar ("h.3.fc_w", "wte" ...)
  std::vector<std::string> gradients;        // sonlu olmayan gradyanlar, geri gecis sirasiyla (son katman once)
  float loss = 0.0f;
};

/// FAZ 9.7: geri donus (rollback) icin aygit durumunun host kopyasi.
struct GptStateSnapshot {
  std::vector<std::vector<unsigned char>> blobs;
  std::int64_t step = 0;
};

class GptModel {
 public:
  /// batch x time mikro-batch icin tum tamponlari ayirir. Agirliklar GPT-2 baslatmasiyla
  /// (N(0, 0.02), artik projeksiyonlar 0.02/sqrt(2L), LN=1/0, bias=0) host'ta tohumla
  /// uretilip yuklenir.
  GptModel(const GptConfig& config, int batch, int time, std::uint64_t seed, const Stream& stream);
  ~GptModel();  // ZeroOffload pinned tamponlarini birakir
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
  /// Aygitta tutulan parametre + gradyan + optimizer durumu baytlari (ZeroOffload'da host disi).
  [[nodiscard]] std::size_t device_state_bytes() const noexcept { return device_state_bytes_; }
  /// ZeroOffload: host'ta tutulan bayt (sayfalanabilir + pinned).
  [[nodiscard]] std::size_t host_state_bytes() const noexcept { return host_state_bytes_; }
  /// Son optimizer_step'in CPU guncelleme suresi (ZeroOffload; digerlerinde 0).
  [[nodiscard]] double last_cpu_update_ms() const noexcept { return last_cpu_update_ms_; }

  /// d_tokens: aygitta [batch*time]. Kayip aygitta yazilir (host senkronu yok).
  /// `grad_scale` dlogits'e ek carpandir (FAZ 9.2: mikro adim agirligi, gecerli_token_i / toplam).
  /// accumulate = false: gradyanlar sifirlanip yeniden hesaplanir. true (FAZ 9.2 biriktirme): mevcut
  /// gradyana EKLENIR — agirlik GEMM'leri beta = 1, BGRAD epilogunun bias gradyani [4C] ara tampondan
  /// +=, LN ve gomme gradyanlari zaten +=. Ek gradyan tamponu yoktur. ZeroOffload/AdamW8bitFused'da yok.
  void forward_backward(const std::int32_t* d_tokens, float grad_scale = 1.0f, bool accumulate = false);
  /// FAZ 9.7: kayip olcegi yokken de inf/NaN gradyanli adimi atla (parametre/moment degismez,
  /// read_loss_scale().skipped_steps artar). Varsayilan kapali (eski davranis).
  void set_nonfinite_guard(bool enabled);
  /// FAZ 9.3: parametre basina sonum/LR carpanlari (bos: GptParamInfo::decay'dan).
  void set_param_scales(const std::vector<GptParamScale>& scales);
  /// FAZ 13.6 (CUDA Graph): CE gradyan olcegi aygittan okunur (nullptr: host grad_scale); AdamW lr/adim aygittan
  /// (nullptr: host). Yalniz AdamW (32-bit), kayip olceksiz, ana kopyali modlarda; aksi acik hata.
  void set_graph_parameters(const float* device_grad_scale, const DeviceAdamDynamic* device_adam);
  void set_step_count(std::int64_t step) noexcept { step_ = step; }
  [[nodiscard]] bool fuse_ce_cast() const noexcept {
    return config_.fuse_ce_cast && config_.compute_dtype == DType::BF16 && config_.ce_chunk_rows <= 0;
  }
  [[nodiscard]] const std::vector<GptParamScale>& param_scales() const noexcept { return scales_; }
  /// FAZ 9.7: parametre + optimizer + kayip olcegi durumunun host kopyasi / geri yuklenmesi.
  [[nodiscard]] GptStateSnapshot save_state() const;
  void load_state(const GptStateSnapshot& snapshot);
  /// FAZ 9.7: d_tokens ile ileri gecisi yeniden calistirip ilk sonlu olmayan katman ciktisini,
  /// sonra geri gecisi calistirip sonlu olmayan gradyanlari adlariyla bulur (senkron, yavas).
  /// Gradyan tamponunu degistirir; optimizer adimi yapmaz.
  [[nodiscard]] GptNonFiniteReport diagnose_nonfinite(const std::int32_t* d_tokens);
  /// FAZ 9.5: son ileri gecisin konum basina kaybi [batch*time] (hedef tokens[b, t+1]; son konum 0).
  [[nodiscard]] const float* device_token_losses() const noexcept;
  /// Parametre (katmanli tensorlerde "h.<l>.<ad>") basina gradyan ve agirlik L2 normu (senkron).
  struct NormEntry {
    std::string name;
    double weight = 0.0;
    double grad = 0.0;
  };
  [[nodiscard]] std::vector<NormEntry> layer_norms() const;
  /// FAZ 10.3 DDP: geri gecis asama kancasi (host'ta, isler kuyruga alinirken cagrilir; asamanin
  /// gradyanlari o ana kadar kuyruga alinmis islerle tamamlanir, cagiran akisa olay kaydetmeli).
  /// Sira: kStageHead (lnf_w/b), katman L-1 ... 0 (katmanin tum [l] dilimleri), kStageEmbedding
  /// (wte, wpe). Bos fonksiyon: kanca yok.
  static constexpr int kStageHead = -1;
  static constexpr int kStageEmbedding = -2;
  void set_backward_hook(std::function<void(int stage)> hook) { backward_hook_ = std::move(hook); }
  /// Yalniz ileri + kayip (degerlendirme).
  void forward(const std::int32_t* d_tokens);
  /// Foreach AdamW; agirlik sonumu yalniz 2B matrislere. grad norm aygitta.
  void optimizer_step(const DeviceAdamWConfig& config);
  /// AdamW8bitFused: geri geciste kullanilacak hiperparametreler (ilk forward_backward'dan once).
  void set_fused_optimizer(const DeviceAdamWConfig& config);

  [[nodiscard]] const float* device_loss() const noexcept { return loss_.as<float>(); }
  [[nodiscard]] const float* device_grad_norm() const noexcept { return grad_norm_.as<float>(); }
  /// FP32 gradyan tamponu (test: tasma enjeksiyonu). ZeroOffload'da nullptr (gradyan host'ta).
  [[nodiscard]] float* device_gradients() noexcept { return offload() ? nullptr : grads_.as<float>(); }
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
  struct Offload;
  void build_layout();
  void head_forward(const std::int32_t* d_tokens, bool for_backward);
  void cpu_optimizer_step(const DeviceAdamWConfig& config);
  void sync_device_from_host_master(bool rebuild_bf16);
  void layer_forward(int layer, bool recompute);
  [[nodiscard]] bool offload() const noexcept { return config_.optimizer == OptimizerKind::ZeroOffload; }
  /// wte/wpe + kucuk parametre bolgesinin sikistirilmis indeksi (katman matrisleri haric).
  [[nodiscard]] std::size_t rest_index(std::size_t offset) const noexcept;
  /// 8-bit modlarda 32-bit Adam durumunun sikistirilmis indeksi (gomme 8-bit ise yalniz kucukler).
  [[nodiscard]] std::size_t state32_index(std::size_t offset) const noexcept {
    return q8_begin_ == 0 ? offset - small_begin_ : rest_index(offset);
  }
  [[nodiscard]] bool in_layer_matrices(std::size_t offset) const noexcept {
    return offset >= matrices_begin_ && offset < small_begin_;
  }
  void run_forward(const std::int32_t* d_tokens, bool for_backward);
  void run_backward(const std::int32_t* d_tokens);
  [[nodiscard]] bool narrow() const noexcept { return config_.compute_dtype != DType::F32; }
  [[nodiscard]] const GptParamScale& scale_of(const GptParamInfo& p) const noexcept {
    return scales_[static_cast<std::size_t>(&p - params_info_.data())];
  }
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
  std::size_t matrices_begin_ = 0;  // ilk katman matrisi (qkv_w) ofseti
  std::size_t rest_count_ = 0;      // matrices_begin_ + (num_params_ - small_begin_)
  std::size_t device_state_bytes_ = 0;
  std::size_t host_state_bytes_ = 0;
  double last_cpu_update_ms_ = 0.0;
  bool use_loss_scale_ = false;
  bool flash_ = false;
  bool guard_ = false;       // FAZ 9.7: kayip olceksiz inf/NaN adim atlama
  float grad_scale_ = 1.0f;  // FAZ 9.2: forward_backward mikro adim agirligi
  bool accumulate_ = false;  // FAZ 9.2: bu geri geciste gradyana ekle
  std::function<void(int)> backward_hook_;  // FAZ 10.3
  std::vector<GptParamScale> scales_;
  std::int64_t step_ = 0;
  const float* graph_grad_scale_ = nullptr;           // FAZ 13.6
  const DeviceAdamDynamic* graph_adam_ = nullptr;     // FAZ 13.6
  bool adam_descriptors_ready_ = false;               // FAZ 13.6: scratch'te tanimlayicilar yuklu
  DeviceBuffer params_, shadow_, grads_, exp_avg_, exp_avg_sq_, loss_, grad_norm_, loss_scale_state_;
  // 8-bit AdamW durumu (katman matrisleri [+ gomme]): kodlar ve blok absmax'lari.
  DeviceBuffer q8_m_, q8_v_, q8_absmax_m_, q8_absmax_v_;
  std::size_t q8_begin_ = 0;  // 8-bit bolgenin ilk ofseti (bolge [q8_begin_, small_begin_))
  std::unique_ptr<Buffers> buffers_;
  std::unique_ptr<Offload> offload_;
  // AdamW8bitFused durumu
  bool fused_ready_ = false;
  DeviceAdamWConfig fused_config_{};
  DeviceBuffer fused_scratch_, fused_clip_, fused_norm_accum_;
  [[nodiscard]] bool fused() const noexcept { return config_.optimizer == OptimizerKind::AdamW8bitFused; }
  [[nodiscard]] bool layer_grad_stream() const noexcept { return offload() || fused(); }
  void fused_update_layer(int layer, const float* layer_grads);
  void fused_update_rest();
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
  /// d_tokens: aygitta [batch*time] (or. TokenLoader::next ile); kopya yapilmaz.
  float step_device(const std::int32_t* d_tokens);
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
