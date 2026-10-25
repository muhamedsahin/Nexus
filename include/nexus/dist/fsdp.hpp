#pragma once

/**
 * @file fsdp.hpp
 * @brief FAZ 11: ZeRO-0/1/2/3, FSDP (FULL_SHARD), HSDP; onceden getirme + sinirlayici; BF16 iletisim;
 *        NVMe takasi (ZeRO-Infinity); parcali kayit ve yeniden parcalama. CPU nn::Sequential modelleri.
 *
 * Duz birim (FlatParameter): sarmalama politikasina gore (PerBlock: parametreli her Sequential
 * katmani; Whole: tum model) bir birimin parametreleri tek duz uzayda ardisik [0, N). Uzay parca
 * grubu boyuna (dp; HSDP'de dugum basina rank) esit bolunur: S = ceil(N / dp), dolgu sifir. Rank r
 * [r S, (r + 1) S) parcasinin sahibidir. nn katmanlari parametreyi kendi Tensor'unda tuttugu icin
 * "gorunum" yerine paketleme/acma kopyasi kullanilir (kopyasiz gorunum yok — raporda yazili).
 *
 * Asamalar (hepsi ayni iletisim desenini kullanir; sonuc asamadan bagimsiz bit bit ayni):
 *   NoShard (ZeRO-0, DDP): tam parametre/gradyan/optimizer; gradyan RS + AG.
 *   ShardOptimizer (ZeRO-1): optimizer durumu + FP32 ana kopya parcali; gradyan RS + AG (all-reduce),
 *     parca guncellenir, parametre AG.
 *   ShardGradOp (ZeRO-2): + gradyan parcali: yalniz RS; tam gradyan RS bitince serbest. Parametre AG.
 *   FullShard (ZeRO-3/FSDP): + parametre parcali: birimden once AG -> hesap -> serbest; geri geciste
 *     tekrar AG -> backward -> RS -> serbest. Adimda parametre AG yok (sonraki ileri geciste).
 *   Hybrid (HSDP): FullShard dugum ici; RS sonrasi parca gradyani dugumler arasi all-reduce.
 * Ayni RS + AG deseni sayesinde tum asamalar ogede ayni toplama sirasini gorur; AdamW ogede
 * calistigi icin agirliklar asamalar arasi BIT BIT aynidir (test_zero_stages_equivalence).
 *
 * Onceden getirme (11.3): ilk adimda birimlerin ileri yurutme sirasi kaydedilir; sonraki ileri
 * gecislerde birim k hesaplanirken sonraki birimin AG'si ucustadir, geri geciste ters sirayla.
 * limit_all_gathers: ayni anda en fazla N AG ucusta (tepe bellek sinirlayicisi). RS'ler asenkron
 * baslar, adimda beklenir. trace() olay kaydi (gather/fwd/bwd/rs) siralama testleri icindir.
 *
 * Karisik hassasiyet (11.4): parca FP32 ana kopya; all-gather param_dtype'ta (BF16: iletisim yarim;
 * hesap BF16'ya yuvarlanmis agirlikla — FSDP param_dtype anlami), reduce-scatter reduce_dtype'ta.
 *
 * Offload (11.5): CPU modelinde parametreler zaten host'ta ("cpu" offload anlamsiz). nvme: optimizer
 * durumu (m, v) ve/veya FullShard'da FP32 parametre parcasi kullanim disinda NVMe dosyasinda; adimdan
 * once asenkron okunur, sonra yazilip RAM'den birakilir (yavas ama dogru).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/fsdp_options.hpp"
#include "nexus/dist/nvme.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus_model/core/parameter.hpp"
#include "nexus_model/core/tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace nn {
class Sequential;
}  // namespace nn

namespace dist {

/// Rank basina kalici bellek (adimlar arasinda tutulan; gecici AG/RS tamponlari haric).
struct FsdpMemory {
  std::uint64_t param_bytes = 0;      // parametre tensorleri + FP32 parca / ana kopya
  std::uint64_t grad_bytes = 0;       // gradyan tensorleri + parca gradyani
  std::uint64_t optimizer_bytes = 0;  // m, v (RAM'deki)
  std::uint64_t nvme_bytes = 0;       // NVMe'deki durum
  [[nodiscard]] std::uint64_t total() const noexcept { return param_bytes + grad_bytes + optimizer_bytes; }
};

struct FsdpUnit {
  std::string name;
  int layer = 0;                  // Sequential katman indeksi
  std::vector<std::size_t> params;  // model.parameters() indeksleri
  std::vector<std::size_t> begin;   // duz uzayda parametre baslangici
  std::size_t numel = 0;          // N
  std::size_t shard = 0;          // S
};

class FsdpEngine {
 public:
  FsdpEngine(nn::Sequential& model, GroupPtr group, FsdpOptions options);
  ~FsdpEngine();
  FsdpEngine(const FsdpEngine&) = delete;
  FsdpEngine& operator=(const FsdpEngine&) = delete;

  /// model.parameters() sirasiyla sonum / LR carpanlari.
  void set_param_scales(const std::vector<float>& weight_decay, const std::vector<float>& lr);
  /// Rank 0'in agirliklarini tum rank'lere dagitir ve parcalari kurar (egitimden once bir kez).
  void initialize();

  /// Ileri gecis (FullShard'da birim birim AG + serbest birakma).
  [[nodiscard]] nexus_model::Tensor forward(const nexus_model::Tensor& input);
  /// Geri gecis. sync = false: no_sync (tam gradyan yerelde birikir, iletisim yok).
  void backward(const nexus_model::Tensor& grad_output, bool sync);
  void zero_grad();

  struct StepResult {
    double grad_norm = 0.0;
    bool skipped = false;
  };
  StepResult step(const FsdpAdamW& optimizer, double lr, double clip);

  [[nodiscard]] FsdpMemory persistent_memory() const;
  [[nodiscard]] const std::vector<FsdpUnit>& units() const noexcept { return units_; }
  [[nodiscard]] const std::vector<std::string>& trace() const noexcept { return trace_; }
  void clear_trace() { trace_.clear(); }
  [[nodiscard]] const std::vector<int>& recorded_order() const noexcept { return order_; }
  /// Ayni anda ucusta olan en cok all-gather (sinirlayici kaniti).
  [[nodiscard]] int max_inflight() const noexcept { return max_inflight_; }
  [[nodiscard]] int shard_size() const noexcept { return shard_world_; }
  [[nodiscard]] const FsdpOptions& options() const noexcept { return options_; }

  /// Parcali durum (bu rank'in parcalari: FP32 parametre, m, v, adim). Tam durum: full_parameters().
  [[nodiscard]] std::vector<unsigned char> save_shard();
  void load_shard(const std::vector<unsigned char>& blob);
  /// Baska dp'de kaydedilmis TUM rank parcalarindan bu rank'in yeni parcalarini kurar (11.H).
  void load_resharded(const std::vector<std::vector<unsigned char>>& old_rank_blobs);
  /// Tum parametrelerin duz FP32 kopyasi (model.parameters() sirasi, birlesik) — her rank'te.
  [[nodiscard]] std::vector<float> full_parameters();
  /// Ayni, m ve v icin (karsilastirma ve kucuk modellerde tam durum).
  [[nodiscard]] std::vector<float> full_optimizer_state(int which);

  struct Unit;

 private:
  void gather(std::size_t u);
  void wait_gathered(std::size_t u);
  void release_params(std::size_t u);
  void ensure_grads(std::size_t u);
  void launch_reduce(std::size_t u);
  void finish_reduce(std::size_t u);
  void prefetch_after(std::size_t position, bool backward);
  void load_master(std::size_t u);
  void store_master(std::size_t u);
  void load_optimizer(std::size_t u);
  void store_optimizer(std::size_t u);
  [[nodiscard]] bool sharded_params() const noexcept;
  [[nodiscard]] std::size_t shard_offset() const noexcept;

  nn::Sequential* model_;
  GroupPtr group_;
  GroupPtr shard_group_;    // parca grubu (HSDP: dugum ici; aksi halde group_)
  GroupPtr replica_group_;  // HSDP: dugumler arasi
  FsdpOptions options_;
  int shard_world_ = 1;
  int shard_rank_ = 0;
  std::vector<FsdpUnit> units_;
  std::vector<std::unique_ptr<Unit>> state_;
  std::vector<nexus_model::Parameter*> params_;
  std::vector<std::vector<std::size_t>> shapes_;
  std::vector<float> wd_, lr_;
  std::vector<int> order_;
  bool order_recorded_ = false;
  int inflight_ = 0;
  int max_inflight_ = 0;
  std::vector<std::string> trace_;
  std::unique_ptr<NvmeStore> nvme_;
  std::int64_t step_ = 0;
};

}  // namespace dist
}  // namespace nexus
