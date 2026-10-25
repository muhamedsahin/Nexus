#pragma once

/**
 * @file parallel_lm.hpp
 * @brief FAZ 12.B-12.E + 12.H: lm::LmConfig modelinin 4B paralel (DP x PP x CP x TP) CPU surumu.
 *
 * Model kodu degismez: ayni lm::LanguageModel (FAZ 6) agirliklarindan rank'in parcalari dilimlenir;
 * yapilandirma paralel katman surumlerini secer. Destek: LayerNorm/RMSNorm, GELU/SwiGLU/GeGLU, MHA/GQA,
 * RoPE/ogrenilen konum/konumsuz, bagli/ayrik gomme, bias (+ yalniz QKV bias). Desteklenmeyen alanlar
 * (muP, son-norm, paralel blok, kayan pencere, softcap, dropout, qk_norm, ALiBi, sinuzoidal, yeniden
 * hesaplama) acik hata verir.
 *
 * Mesh (adlar sirasiyla, tp en hizli): {"dp", "pp", "cp", "tp"}.
 *
 * Tensor paralelligi (12.B, Megatron-LM):
 *   QKV column-parallel (q baslari tp'ye bolunur; GQA'da Hkv % tp == 0 ise KV de bolunur, tp % Hkv == 0
 *   ise her rank grubundaki KV basini cogaltir ve KV gradyani "kv grubu"nda toplanir), cikis row-parallel.
 *   MLP: up (SwiGLU'da W1|W3 eslesen dilimler) column, down row; gated aktivasyon yerel.
 *   Gomme VocabParallelEmbedding (sozluk parcasi disi satir 0 -> Partial -> all-reduce / reduce-scatter),
 *   lm_head sozluk-parcali; capraz entropi logit toplanmadan: yerel max -> all-reduce(Max), yerel
 *   sum-exp ve hedef logit -> all-reduce(Sum). Bagli gommede gomme ve kafa ayni parcayi kullanir.
 * Dizi paralelligi (12.C, Megatron SP): TP bolgesi disi (norm, artik) satir (token) ekseninde parcali
 *   [N / tp, C]; all-reduce yerine bolge girisinde all-gather, cikisinda reduce-scatter. Norm ve bias
 *   gibi cogaltilmis parametrelerin gradyani tp'de all-reduce edilir.
 * Async TP (12.D, collective matmul): SP'de all-gather + GEMM ve GEMM + reduce-scatter halka uzerinde
 *   parca parca; parca k hesaplanirken parca k+1 halkada ilerler (send/recv, ayni is parcacigi kuyrugu).
 * Ardisik paralellik (12.E): katmanlar pp * v parcaya maliyet dengeli bolunur (gomme parca 0'da, son norm
 *   + kafa son parcada); zamanlamalar pipeline.hpp. Asamalar arasi aktivasyon/gradyan yonlu kenar
 *   gruplariyla asenkron send/recv; alici tamponlari adim basinda ayrilir ve tuketim sirasiyla
 *   kuyruga alinir. Bagli gommede ilk ve son asamanin kopyalarinin gradyani {ilk, son} grubunda toplanir.
 * Baglam paralelligi (12.F): her dizi cp'ye (zigzag ya da ardisik) bolunur; dikkat Ring ya da Ulysses.
 * Veri paralelligi + ZeRO-1: gradyanlar dp x cp grubunda duz tampon uzerinde reduce-scatter (+ all-gather);
 *   zero = 1'de AdamW durumu dp x cp'ye parcali, guncel parametre all-gather ile dagitilir. Her iki yol
 *   ayni RS desenini kullanir (zero 0/1 bit bit ayni).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/context_parallel.hpp"
#include "nexus/dist/dtensor.hpp"
#include "nexus/dist/fsdp_options.hpp"
#include "nexus/dist/pipeline.hpp"
#include "nexus/lm/config.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace lm {
class LanguageModel;
}  // namespace lm

namespace dist {

struct ParallelConfig {
  int dp = 1;
  int tp = 1;
  int pp = 1;
  int cp = 1;
  int zero = 0;            // 0: tam optimizer durumu; 1: AdamW durumu dp x cp'ye parcali
  bool sp = false;         // dizi paralelligi (tp > 1 ister)
  bool async_tp = false;   // collective matmul (sp ister)
  PipeSchedule pp_schedule = PipeSchedule::OneF1B;
  int virtual_stages = 1;  // interleaved_1f1b icin v
  CpAlgorithm cp_algorithm = CpAlgorithm::Ring;
  CpLayout cp_layout = CpLayout::Zigzag;
  int attention_threads = 1;  // rank'ler is parcacigi oldugundan varsayilan seri Flash
  bool trace = false;         // pipeline eylem izi (test)
};

/// Son forward_backward'in olcumleri (bu rank).
struct ParallelStepStats {
  double step_ms = 0.0;     // eylemlerin toplam duvar suresi
  double idle_ms = 0.0;     // asama girisi beklenirken (recv) gecen sure
  double forward_ms = 0.0;  // F toplam
  double backward_ms = 0.0; // B toplam (ZB-H1'de yalniz giris gradyani)
  double weight_ms = 0.0;   // W toplam (ZB-H1)
  int forwards = 0, backwards = 0, weights = 0;
  int deferred_w_peak = 0;  // ZB-H1: ayni anda bekleyen en cok W
  std::uint64_t peak_activation_bytes = 0;
  std::vector<std::string> trace;  // "F3c1", "B3c1", "W3c1" ...
  [[nodiscard]] double bubble() const noexcept { return step_ms > 0.0 ? idle_ms / step_ms : 0.0; }
};

class ParallelLm {
 public:
  struct Param {
    std::string name;                     // lm::LanguageModel parametre adi
    std::vector<std::int64_t> full_shape;
    // grad: zero 0'da tamamen indirgenmis; zero 1'de dp x cp indirgemesi yalniz duz parcada (grad yerel kalir).
    std::vector<float> value, grad;  // AdamW m/v duz tamponda (zero 1: dp x cp parcasi)
    bool decay = false;
    float wd = 0.0f;        // set_param_scales (varsayilan: decay ? 1 : 0)
    float lr_scale = 1.0f;
    bool tp_replicated = false;  // tum tp rank'lerinde ayni deger
    int kv_offset = -1;          // >= 0: bu yerel sutundan itibaren KV tp'de cogaltilmis (kv grubu)
    bool tied_copy = false;      // pp > 1 bagli gomme: son asamadaki kopya
    // Dilim: Full | Rows [row_begin, row_begin + rows) | Cols (sutun / oge araliklari).
    enum class Slice : std::uint8_t { Full, Rows, Cols } slice = Slice::Full;
    std::int64_t row_begin = 0, rows = 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> cols;  // (baslangic, boy)
  };

  /// Kollektif (world grubunun tum rank'leri). Agirliklar `reference`tan dilimlenir.
  ParallelLm(const lm::LanguageModel& reference, GroupPtr world, ParallelConfig config);
  ~ParallelLm();
  ParallelLm(const ParallelLm&) = delete;
  ParallelLm& operator=(const ParallelLm&) = delete;

  [[nodiscard]] const ParallelConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const lm::LmConfig& lm_config() const noexcept { return lmc_; }
  [[nodiscard]] DeviceMesh& mesh() noexcept { return *mesh_; }
  [[nodiscard]] const GroupPtr& data_parallel_group() const;

  /// Bir adim: micro[i] = bu dp kopyasinin mikro yigini [batch * seq] (tp/pp/cp rank'lerinde ayni).
  /// weights[i] = mikro yigin token agirligi (gecerli_i / toplam). Gradyanlar biriktirilir ve dp x cp
  /// (+ tp/kv/bagli) indirgemeleri yapilir. Donus: mikro yigin basina ortalama kayip (her rank'te).
  std::vector<float> forward_backward(const std::vector<const std::int32_t*>& micro, const std::vector<float>& weights,
                                      int batch, int seq);
  struct StepResult {
    double grad_norm = 0.0;
    bool skipped = false;
  };
  /// Global norm kirpmasi (torch katsayisi) + AdamW (torch). Gradyanlari sifirlar.
  StepResult step(const FsdpAdamW& optimizer, double lr, double clip);
  /// Yalniz ileri (tek mikro yigin, gecerli hedefler uzerinde ortalama kayip; her rank'te).
  [[nodiscard]] double eval_loss(const std::int32_t* tokens, int batch, int seq);

  [[nodiscard]] std::vector<Param>& parameters() noexcept { return params_; }
  [[nodiscard]] const std::vector<Param>& parameters() const noexcept { return params_; }
  void set_param_scales(const std::vector<float>& weight_decay, const std::vector<float>& lr);
  /// Yerel parametrenin tam tensordeki duz indeksleri.
  [[nodiscard]] std::vector<std::int64_t> full_index(std::size_t i) const;
  [[nodiscard]] const ParallelStepStats& last_stats() const noexcept { return stats_; }
  /// Kalici bayt (parametre + gradyan + optimizer durumu).
  [[nodiscard]] std::uint64_t persistent_bytes() const;
  [[nodiscard]] std::int64_t step_count() const noexcept { return step_; }
  /// Bu rank'in durumu (yerel parametreler + AdamW m/v parcasi + adim). Ayni paralel yerlesimde yuklenir.
  [[nodiscard]] std::vector<unsigned char> save_state() const;
  void load_state(const std::vector<unsigned char>& blob);
  /// Rank'in parca araliklari (rapor): [ilk parca, ...] -> katman araligi.
  [[nodiscard]] std::string layout_summary() const;

  struct Impl;

 private:
  ParallelConfig cfg_;
  lm::LmConfig lmc_;
  std::unique_ptr<DeviceMesh> mesh_;
  std::vector<Param> params_;
  std::unique_ptr<Impl> impl_;
  ParallelStepStats stats_;
  std::int64_t step_ = 0;
};

/// Ayni surecteki rank nesnelerinden tam tensoru kurar. which: 0 deger, 1 gradyan.
/// replica_diff: cogaltilmis kopyalar arasindaki en buyuk |fark| (tp/dp/cp/bagli tutarlilik kontrolu).
[[nodiscard]] std::vector<float> assemble_full(const std::vector<const ParallelLm*>& ranks, const std::string& name,
                                               int which, double* replica_diff = nullptr);

}  // namespace dist
}  // namespace nexus
