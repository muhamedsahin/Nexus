#pragma once

/**
 * @file infer.hpp
 * @brief FAZ 16: GPT-2 GPU cikarim motoru — sayfali KV onbellegi (blok tahsis edici, copy-on-write), artimli
 *        cozme (prefill / decode, karisik yigin), decode CUDA Graph (kova basina), ornekleme (tek cekirdek),
 *        spekulatif cozme (ilk-k katman taslak), nicemleme (W8A8, W4A16, KV INT8).
 *
 * Sayi bicimi: agirliklar FP16 (ya da INT8/INT4), artik akis (residual) FP32, layernorm / GEMM girdileri FP16,
 * cuBLASLt FP32 biriktirme; proj/fc2 GEMM'leri residual'a beta = 1 ile FP32 ekler (bias epilogu).
 * Ince GEMM (m <= 16 satir: decode, spekulatif dogrulama): kendi cekirdegi — her cikis sabit sirali bolunmus-K
 * indirgemeyle hesaplanir, satir sonucu m'den BAGIMSIZDIR (spekulatif == greedy birebir); m > 16: cuBLASLt.
 * KV: katman basina [blok, blok_boyu, H, D] FP16 ya da INT8 (+ (yuva, bas) basina olcek FP32).
 * Bir adim (forward): istek parcalari (sira, yeni tokenlar) tek paketli yiginda; her token konumu = sira
 * uzunlugu + i. Yeni sira (onbellek bos) prefill parcalari FlashAttention-2 varlen (FP16) ile, digerleri
 * (decode, spekulatif dogrulama, onbellekli devam) sayfali dikkat cekirdegiyle (her sorgu kendi konumuna kadar
 * nedensel) hesaplanir. Tum tokenlarin K/V'si sayfalara yazilir (yuva eslemesi).
 * Logit: istenen satirlar (parca basina son `logits` token) toplanir -> ln_f -> lm_head (sozluk 64'e dolgulu).
 * CUDA Graph: yalniz decode (parca basina 1 token, tum sira onbellekli); kova boylari 1,2,4,..,max_seqs;
 * girisler sabit aygit tamponlarinda (blok tablosu, konum, yuva) -> her adim yalniz kopyala + oynat.
 * Taslak (spekulatif): ayni agirliklarin ilk `layers` katmani + ln_f + lm_head; KV'si hedefle ortak sayfalara
 * yazilir ve dogrulama ayni konumlari tum katmanlarda yeniden yazar.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace cuda {
namespace infer {

/// Host FP32 GPT-2 agirliklari (gpt2.nxg, benchmarks/runner/faz16_gpt2.py export).
struct Gpt2Weights {
  int layers = 0, heads = 0, d = 0, vocab = 0, ctx = 0;
  std::vector<float> wte, wpe, ln_f_w, ln_f_b;
  struct Layer {
    std::vector<float> ln1_w, ln1_b, attn_w, attn_b, proj_w, proj_b, ln2_w, ln2_b, fc_w, fc_b, fc2_w, fc2_b;
  };
  std::vector<Layer> h;
  [[nodiscard]] std::int64_t parameters() const;
};
[[nodiscard]] Gpt2Weights load_gpt2(const std::string& path);

enum class WeightQuant : std::uint8_t { F16, W8A8, W4A16 };
enum class KvType : std::uint8_t { F16, INT8 };
[[nodiscard]] const char* to_string(WeightQuant q) noexcept;

/// Kalibrasyon (W8A8 SmoothQuant alfa, W4A16 AWQ olcek aramasi): token dizisi (aktivasyon istatistigi).
struct QuantOptions {
  WeightQuant weights = WeightQuant::F16;
  KvType kv = KvType::F16;
  int w4_group = 128;
  float smooth_alpha = 0.5f;               // W8A8 SmoothQuant
  std::vector<std::int32_t> calibration;   // bos: olcek = 1 (yalniz yuvarlama)
};

/// F16: hiz kipi (agirlik/aktivasyon/KV FP16). F32: dogrulama kipi (her sey FP32, TF32 kapali; prefill de
/// sayfali cekirdekle) — test_kv_cache_matches_full <= 1e-4 bu kipte olculur.
enum class Precision : std::uint8_t { F16, F32 };

struct EngineConfig {
  Precision precision = Precision::F16;
  int max_seqs = 64;
  int max_tokens = 2048;       // adim basina en cok token (prefill dahil)
  int block_size = 16;
  std::size_t kv_bytes = std::size_t{1} << 30;  // KV havuzu (blok sayisi bundan)
  bool cuda_graphs = true;
  QuantOptions quant{};
};

/// Ornekleme (satir basina). temperature = 0: greedy (esitlikte en kucuk id).
struct SamplingParams {
  float temperature = 1.0f;
  int top_k = 0;             // 0: kapali
  float top_p = 1.0f;        // 1: kapali
  float min_p = 0.0f;        // 0: kapali
  float repetition_penalty = 1.0f;  // HF: istem + cikti tokenlari
  float frequency_penalty = 0.0f;   // OpenAI: yalniz cikti tokenlari
  float presence_penalty = 0.0f;
  std::uint64_t seed = 0;
};

struct Chunk {
  int seq = -1;
  std::vector<std::int32_t> tokens;
  int logits = 1;     // son kac tokenin logiti istenir (0: hic)
  bool output = false;  // tokenlar cikti mi (frekans/varlik cezasi sayaci); istem: false
};

struct StepStats {
  int tokens = 0;
  int prefill_tokens = 0;
  int decode_tokens = 0;
  bool graph = false;
  double ms = 0.0;
};

class Engine {
 public:
  Engine(const Gpt2Weights& weights, EngineConfig config, cudaStream_t stream = nullptr);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // ---- Sira ve KV bloklari -----------------------------------------------------------------
  [[nodiscard]] int add_sequence();
  /// Copy-on-write dal: bloklar paylasilir (ref sayaci); paylasilan son bloga yazilacaginda kopyalanir.
  [[nodiscard]] int fork_sequence(int src);
  void free_sequence(int seq);
  /// Uzunlugu kisaltir (spekulatif ret): fazla bloklar serbest, ceza sayaclari geri alinir.
  void truncate(int seq, int length);
  [[nodiscard]] int length(int seq) const;
  [[nodiscard]] const std::vector<std::int32_t>& tokens(int seq) const;
  [[nodiscard]] int free_blocks() const noexcept;
  [[nodiscard]] int total_blocks() const noexcept;
  [[nodiscard]] int blocks_needed(int seq, int new_tokens) const;
  [[nodiscard]] int block_size() const noexcept;

  // ---- Ileri gecis --------------------------------------------------------------------------
  /// Parcalari tek adimda isler; KV yazilir, uzunluklar artar. Doner: aygit logitleri [satir, vocab_padded]
  /// (satirlar parca sirasiyla, parca icinde token sirasiyla). layers < katman sayisi: taslak (ilk k katman).
  const float* forward(const std::vector<Chunk>& chunks, int layers = -1);
  [[nodiscard]] int logit_rows() const noexcept;
  [[nodiscard]] int vocab() const noexcept;
  [[nodiscard]] int vocab_padded() const noexcept;
  [[nodiscard]] std::vector<float> logits_host(int row) const;
  /// Son forward'un logit satirlari icin ornekler (params satir basina; seq ceza sayaci icin).
  [[nodiscard]] std::vector<std::int32_t> sample(const std::vector<SamplingParams>& params, const std::vector<int>& seqs,
                                                 std::uint64_t offset);
  /// Satir basina ofset (istek basina tekrar uretilebilir ornekleme: ofset = istegin token indeksi).
  [[nodiscard]] std::vector<std::int32_t> sample(const std::vector<SamplingParams>& params, const std::vector<int>& seqs,
                                                 const std::vector<std::uint64_t>& offsets);
  /// Son forward satirlari icin -log p(target) (perplexity).
  [[nodiscard]] std::vector<float> nll(const std::vector<std::int32_t>& targets);
  [[nodiscard]] const StepStats& last_step() const noexcept;

  // ---- Bellek / bilgi -------------------------------------------------------------------------
  [[nodiscard]] std::size_t weight_bytes() const noexcept;
  [[nodiscard]] std::size_t kv_bytes_per_token() const noexcept;
  [[nodiscard]] int layers() const noexcept;
  [[nodiscard]] int graphs_captured() const noexcept;
  [[nodiscard]] const EngineConfig& config() const noexcept;
  /// Dahili (kalibrasyon): forward sirasinda dogrusal katman girdilerini host'a kopyalar (FP32 motor).
  void impl_capture(std::vector<std::vector<float>>* capture);

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Yardimci: greedy / ornekli uretim (tek istek, decode CUDA Graph'li). Doner: uretilen tokenlar.
struct GenerateStats {
  double ttft_ms = 0.0;
  double tpot_ms = 0.0;
  int tokens = 0;
};
[[nodiscard]] std::vector<std::int32_t> generate(Engine& engine, const std::vector<std::int32_t>& prompt, int max_new,
                                                 const SamplingParams& sp, GenerateStats* stats = nullptr);

/// Spekulatif cozme (greedy dogrulama): taslak = ilk `draft_layers` katman, gamma aday.
struct SpecStats {
  int steps = 0;              // hedef dogrulama gecisi
  int drafted = 0;
  int accepted = 0;
  double tokens_per_step = 0.0;
  double ms = 0.0;
};
[[nodiscard]] std::vector<std::int32_t> speculative_generate(Engine& engine, const std::vector<std::int32_t>& prompt,
                                                             int max_new, int draft_layers, int gamma,
                                                             SpecStats* stats = nullptr);

}  // namespace infer
}  // namespace cuda
}  // namespace nexus
