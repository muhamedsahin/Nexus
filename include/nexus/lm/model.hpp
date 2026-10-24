#pragma once

/**
 * @file model.hpp
 * @brief FAZ 6.C-6.F: DecoderBlock (NXS tur 34), LanguageModel (tur 35), baslatma, muP, kurucu.
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

struct ForwardOptions {
  bool training = false;
  std::uint64_t dropout_seed = 0;
  int position_offset = 0;
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
};

class ParamFactory;

class DecoderBlock {
 public:
  DecoderBlock(const LmConfig& config, int index, ParamFactory& factory);
  [[nodiscard]] autograd::Variable forward(const autograd::Variable& x, const BlockContext& ctx) const;
  [[nodiscard]] nxs::LayerSpec nxs_spec() const;
  [[nodiscard]] int index() const noexcept { return index_; }

 private:
  [[nodiscard]] autograd::Variable norm(const autograd::Variable& x, const autograd::Variable& w,
                                        const autograd::Variable& b) const;
  [[nodiscard]] autograd::Variable linear(const autograd::Variable& x, const autograd::Variable& w,
                                          const autograd::Variable& b) const;
  [[nodiscard]] autograd::Variable attn(const autograd::Variable& h, const BlockContext& ctx) const;
  [[nodiscard]] autograd::Variable mlp(const autograd::Variable& h) const;
  [[nodiscard]] autograd::Variable dropout(const autograd::Variable& x, const BlockContext& ctx, int slot) const;

  LmConfig c_;
  int index_ = 0;
  autograd::Variable n1w_, n1b_, n2w_, n2b_, qkvw_, qkvb_, ow_, ob_, qn_, kn_, upw_, upb_, downw_, downb_;
};

class LanguageModel {
 public:
  explicit LanguageModel(const LmConfig& config);

  [[nodiscard]] const LmConfig& config() const noexcept { return config_; }
  /// logits [B*T, padded_vocab]; dolgu sutunlari -inf.
  [[nodiscard]] autograd::Variable logits(const std::vector<std::int32_t>& tokens, int batch, int time,
                                          const ForwardOptions& options = {}) const;
  /// Kaydirilmis CE (fused::cross_entropy_shifted).
  [[nodiscard]] autograd::Variable loss(const std::vector<std::int32_t>& tokens, int batch, int time,
                                        const ForwardOptions& options = {}) const;

  [[nodiscard]] const std::vector<NamedParam>& parameters() const noexcept { return params_; }
  [[nodiscard]] std::int64_t num_parameters() const noexcept;
  /// Ada gore parametre (yoksa nullptr).
  [[nodiscard]] const NamedParam* find(const std::string& name) const noexcept;
  [[nodiscard]] const std::vector<DecoderBlock>& blocks() const noexcept { return blocks_; }
  /// Katman basina gradyan L2 normu (h{l}.* parametreleri).
  [[nodiscard]] std::vector<double> layer_grad_norms() const;
  void zero_grad();

  /// NXS: tur 35 kok (config JSON gomulu) + tur 34 bloklar + adli agirliklar (bagli gomme tek kopya).
  void save_nxs(const std::string& path) const;
  [[nodiscard]] static LanguageModel load_nxs(const std::string& path);

 private:
  [[nodiscard]] autograd::Variable embed(const std::vector<std::int32_t>& tokens, int batch, int time,
                                         const ForwardOptions& options) const;

  LmConfig config_;
  std::vector<NamedParam> params_;
  std::vector<DecoderBlock> blocks_;
  autograd::Variable wte_, wpe_, nfw_, nfb_, head_, pad_mask_;
  std::shared_ptr<RopeCache> rope_;
  std::shared_ptr<const std::vector<float>> alibi_;
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

 private:
  const LmConfig* c_;
  std::vector<NamedParam>* out_;
  std::uint64_t state_;
  float width_mult_ = 1.0f;
  [[nodiscard]] double gaussian();
};

}  // namespace lm
}  // namespace nexus
