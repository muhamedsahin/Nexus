#pragma once

/**
 * @file model.hpp
 * @brief FAZ 6.C-6.F ve FAZ 18.5: DecoderBlock (NXS tur 34), LanguageModel (tur 35), baslatma, muP,
 *        ve sifir-ayirmali (zero-allocation) FP32 / BF16 / INT8 / INT4 CPU cikarim motoru.
 *
 * On-normalizasyon (varsayilan):
 *   h = x + Attn(N1(x));  y = h + MLP(N2(h))
 * Paralel blok (GPT-J/PaLM): y = x + Attn(N(x)) + MLP(N(x)).
 * Son-normalizasyon (prenorm = false, tur 33 CausalTransformerBlock tarifi):
 *   h = N1(x + Attn(x));  y = N2(h + MLP(h))
 *
 * Parametre adlari (agirliklar giris-ana, y = x W):
 *   wte [Vp, d], wpe [P, d], h{l}.norm1.{weight,bias}, h{l}.attn.qkv.{weight [d,(H+2Hkv)D], bias},
 *   h{l}.attn.o.{weight [HD, d], bias}, h{l}.attn.{q,k}_norm.weight [D], h{l}.norm2.*,
 *   h{l}.mlp.up.{weight [d, 2F | F], bias} (kapili MLP'de [W1 | W3] tek GEMM),
 *   h{l}.mlp.down.{weight [F, d], bias}, norm_f.*, lm_head.weight [Vp, d] (bagli degilse).
 *
 * Baslatma (tohumdan, yalniz host'ta -> cihazdan bagimsiz):
 *   gpt2:  N(0, s); o ve down s / sqrt(2L); bias 0; norm 1.
 *   llama: N(0, s) tum matris ve gommeler (HF LlamaPreTrainedModel).
 *   mup:   m = d_model / mup_base_width; gizli matrisler N(0, s / sqrt(m)), o/down ayrica
 *          / sqrt(2L); dikkat olcegi 1/D; logit carpani 1/m; Adam LR carpani gizli
 *          matrislerde 1/m (Tensor Programs V, Tablo 3).
 *   init_truncated: +-2s disi yeniden ornekleme.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/fused/training.hpp"
#include "nexus/lm/config.hpp"
#include "nexus/lm/moe_layer.hpp"
#include "nexus/lm/ops.hpp"
#include "nexus/model/nxs_format.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace lm {

struct NamedParam {
  std::string name;
  autograd::Variable value;
  bool decay = false;    // AdamW agirlik sonumu (matrisler + gommeler)
  float lr_mult = 1.0f;  // muP
};

/// FAZ 18.5: CPU uzerinde cikarim (inference) icin agirlik niceleme / veri turu secimi.
enum class CpuWeightMode : std::uint8_t {
  Auto = 0,  // varsayilan (F32 agirliklar, KV-cache ve NoGrad yolunda sifir-ayirmali yurutme)
  F32 = 1,   // devriklenmis [N, K] FP32 GEMV/GEMM
  BF16 = 2,  // [N, K] BF16 paketli, FP32 biriktiricili AVX2 FMA
  INT8 = 3,  // kanal-simetrik INT8 (`cpu::PackedS8` + `gemm_s8`)
  INT4 = 4,  // kanal-simetrik 4-bit nibble-seritli INT4 (`cpu::PackedQ4` + `gemm_q4`)
};

/// FAZ 16: artimli cozme icin KV onbellegi (CPU, bitisik, batch 1). Katman basina [uzunluk, Hkv*D] K ve V
/// (RoPE ve qk_norm SONRASI). forward(options.cache) yeni tokenlari ekler; konum ofseti = onceki uzunluk.
struct KvCache {
  std::vector<std::vector<float>> k, v;
  int length = 0;
  void reset() {
    for (auto& x : k) x.clear();
    for (auto& x : v) x.clear();
    length = 0;
  }
  [[nodiscard]] std::size_t bytes() const noexcept {
    std::size_t n = 0;
    for (const auto& x : k) n += x.size() * 4;
    for (const auto& x : v) n += x.size() * 4;
    return n;
  }
};

struct ForwardOptions {
  bool training = false;
  std::uint64_t dropout_seed = 0;
  int position_offset = 0;
  KvCache* cache = nullptr;  // FAZ 16: verilirse batch 1; position_offset yok sayilir (cache->length)
};

/// Blok ileri gecisinin ortak baglami (tum katmanlar paylasir).
struct BlockContext {
  int batch = 1;
  int time = 1;
  bool training = false;
  std::uint64_t dropout_seed = 0;
  std::shared_ptr<const std::vector<int>> positions;
  std::shared_ptr<const RopeCache> rope;
  std::shared_ptr<const std::vector<float>> alibi;
  std::vector<autograd::Variable>* router_losses = nullptr;  // FAZ 14: MoE aux + z dugumleri
  std::vector<MoeLayerStats>* moe_stats = nullptr;
  KvCache* cache = nullptr;  // FAZ 16
};

class ParamFactory;
class LanguageModel;

class DecoderBlock {
 public:
  DecoderBlock(const LmConfig& config, int index, ParamFactory& factory);
  [[nodiscard]] autograd::Variable forward(const autograd::Variable& x, const BlockContext& ctx) const;
  [[nodiscard]] nxs::LayerSpec nxs_spec() const;
  [[nodiscard]] nxs::LayerSpec nxs_moe_spec() const;
  [[nodiscard]] bool is_moe() const noexcept { return moe_; }
  [[nodiscard]] int index() const noexcept { return index_; }

 private:
  friend class LanguageModel;
  [[nodiscard]] autograd::Variable norm(const autograd::Variable& x, const autograd::Variable& w,
                                        const autograd::Variable& b) const;
  [[nodiscard]] autograd::Variable linear(const autograd::Variable& x, const autograd::Variable& w,
                                          const autograd::Variable& b) const;
  [[nodiscard]] autograd::Variable attn(const autograd::Variable& h, const BlockContext& ctx) const;
  [[nodiscard]] autograd::Variable mlp(const autograd::Variable& h) const;
  [[nodiscard]] autograd::Variable ffn(const autograd::Variable& h, const BlockContext& ctx) const;  // yogun ya da MoE
  [[nodiscard]] autograd::Variable dropout(const autograd::Variable& x, const BlockContext& ctx, int slot) const;

  LmConfig c_;
  int index_ = 0;
  autograd::Variable n1w_, n1b_, n2w_, n2b_, qkvw_, qkvb_, ow_, ob_, qn_, kn_, upw_, upb_, downw_, downb_;
  bool moe_ = false;                              // FAZ 14
  autograd::Variable router_, noise_, eup_, edown_;
};

class LanguageModel {
 public:
  struct CpuInferenceState;

  explicit LanguageModel(const LmConfig& config);
  /// FAZ 17: zero_init = true: rastgele baslatma atlanir (agirliklar sonra yuklenecek: HF / GGUF ice aktarma).
  LanguageModel(const LmConfig& config, bool zero_init);

  [[nodiscard]] const LmConfig& config() const noexcept { return config_; }
  /// logits [B*T, padded_vocab]; dolgu sutunlari -inf.
  [[nodiscard]] autograd::Variable logits(const std::vector<std::int32_t>& tokens, int batch, int time,
                                          const ForwardOptions& options = {}) const;
  /// FAZ 16: bos KV onbellegi (katman sayisi kadar).
  [[nodiscard]] KvCache make_cache() const;
  /// Kaydirilmis CE (fused::cross_entropy_shifted).
  [[nodiscard]] autograd::Variable loss(const std::vector<std::int32_t>& tokens, int batch, int time,
                                        const ForwardOptions& options = {}) const;

  /// FAZ 18.5: Agirliklari secilen CPU cikarim bicimine (F32 / BF16 / INT8 / INT4) hazirlar ve paketler.
  void prepare_cpu_inference(CpuWeightMode mode = CpuWeightMode::F32) const;
  [[nodiscard]] CpuWeightMode cpu_weight_mode() const noexcept { return cpu_mode_; }
  [[nodiscard]] std::size_t cpu_inference_weight_bytes() const noexcept;

  [[nodiscard]] const std::vector<NamedParam>& parameters() const noexcept { return params_; }
  [[nodiscard]] std::int64_t num_parameters() const noexcept;
  /// Ada gore parametre (yoksa nullptr).
  [[nodiscard]] const NamedParam* find(const std::string& name) const noexcept;
  [[nodiscard]] const std::vector<DecoderBlock>& blocks() const noexcept { return blocks_; }
  /// FAZ 14: son ileri gecisteki MoE katmani uzman kullanimi (MoE yoksa bos).
  [[nodiscard]] const std::vector<MoeLayerStats>& moe_stats() const noexcept { return moe_stats_; }
  /// Katman basina gradyan L2 normu (h{l}.* parametreleri).
  [[nodiscard]] std::vector<double> layer_grad_norms() const;
  void zero_grad();

  /// NXS: tur 35 kok (config JSON gomulu) + tur 34 bloklar + adli agirliklar (bagli gomme tek kopya).
  void save_nxs(const std::string& path) const;
  [[nodiscard]] static LanguageModel load_nxs(const std::string& path);

 private:
  [[nodiscard]] autograd::Variable forward_logits(const std::vector<std::int32_t>& tokens, int batch, int time,
                                                  const ForwardOptions& options, std::vector<autograd::Variable>* aux) const;
  [[nodiscard]] autograd::Variable forward_cpu_fast(const std::vector<std::int32_t>& tokens, int batch, int time,
                                                    const ForwardOptions& options) const;
  [[nodiscard]] autograd::Variable embed(const std::vector<std::int32_t>& tokens, int batch, int time,
                                         const ForwardOptions& options) const;

  LmConfig config_;
  std::vector<NamedParam> params_;
  std::vector<DecoderBlock> blocks_;
  autograd::Variable wte_, wpe_, nfw_, nfb_, head_, pad_mask_;
  std::shared_ptr<RopeCache> rope_;
  std::shared_ptr<const std::vector<float>> alibi_;
  mutable std::vector<MoeLayerStats> moe_stats_;
  mutable CpuWeightMode cpu_mode_ = CpuWeightMode::Auto;
  mutable std::shared_ptr<CpuInferenceState> cpu_infer_;
};

[[nodiscard]] LanguageModel build(const LmConfig& config);
/// Ön tanimli + JSON overrides (orn. "{\"n_layer\":2}").
[[nodiscard]] LanguageModel build(const std::string& preset_name, const std::string& overrides_json = "{}");

/// AdamW: (sonum, LR carpani) gruplari; global norm kirpmasi tum gruplar uzerinde.
class AdamW {
 public:
  AdamW(const LanguageModel& model, fused::AdamWConfig config);
  fused::AdamWStepInfo step();

 private:
  const LanguageModel* model_;
  fused::AdamWConfig config_;
  std::vector<std::vector<float>> m_, v_;
  std::int64_t step_ = 0;
};

/// Baslatma yardimcisi (DecoderBlock ve LanguageModel parametreleri sirayla bundan uretir).
class ParamFactory {
 public:
  ParamFactory(const LmConfig& config, std::vector<NamedParam>* out);
  autograd::Variable normal(const std::string& name, std::vector<std::int64_t> dims, float std, bool decay,
                            float lr_mult = 1.0f);
  autograd::Variable constant(const std::string& name, std::vector<std::int64_t> dims, float value);
  [[nodiscard]] float width_mult() const noexcept { return width_mult_; }
  void set_zero_init(bool z) noexcept { zero_ = z; }

 private:
  bool zero_ = false;
  const LmConfig* c_;
  std::vector<NamedParam>* out_;
  std::uint64_t state_;
  float width_mult_ = 1.0f;
  [[nodiscard]] double gaussian();
};

}  // namespace lm
}  // namespace nexus
