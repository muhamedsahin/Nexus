/**
 * @file model.cpp
 * @brief FAZ 6.C-6.F: DecoderBlock, LanguageModel, baslatma/muP, AdamW gruplari, NXS.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/model.hpp"

#include "nexus/core/error.hpp"
#include "nexus/fused/fused.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <span>
#include <utility>

namespace nexus {
namespace lm {

namespace ag = autograd;
using ag::Shape;
using ag::Variable;

namespace {
// Gradyani olmayan (kullanilmamis) parametre icin nullptr.
const Variable* grad_of(const Variable& v) {
  const auto& meta = v.meta();
  return meta && meta->grad && meta->grad->has_data() ? meta->grad.get() : nullptr;
}
}  // namespace

// ---- ParamFactory: platformdan bagimsiz RNG (splitmix64 + Box-Muller, FP64) -------------
ParamFactory::ParamFactory(const LmConfig& config, std::vector<NamedParam>* out)
    : c_(&config), out_(out), state_(config.seed ^ 0x9E3779B97F4A7C15ULL) {
  if (config.init == InitKind::MuP) {
    width_mult_ = static_cast<float>(config.d_model) / static_cast<float>(config.mup_base_width);
  }
}

double ParamFactory::gaussian() {
  auto next = [this]() {
    std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (static_cast<double>(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);  // (0, 1)
  };
  const double u1 = next();
  const double u2 = next();
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

Variable ParamFactory::normal(const std::string& name, std::vector<std::int64_t> dims, float std, bool decay,
                              float lr_mult) {
  Shape shape{std::move(dims)};
  std::vector<float> values(static_cast<std::size_t>(shape.numel()));
  for (float& v : values) {
    double z = gaussian();
    if (c_->init_truncated) {
      while (std::fabs(z) > 2.0) {
        z = gaussian();
      }
    }
    v = static_cast<float>(z * std);
  }
  Variable var = Variable::make(std::move(values), std::move(shape), true);
  out_->push_back({name, var, decay, lr_mult});
  return var;
}

Variable ParamFactory::constant(const std::string& name, std::vector<std::int64_t> dims, float value) {
  Shape shape{std::move(dims)};
  std::vector<float> values(static_cast<std::size_t>(shape.numel()), value);  // shape tasinmadan once
  Variable var = Variable::make(std::move(values), std::move(shape), true);
  out_->push_back({name, var, false, 1.0f});
  return var;
}

// ---- DecoderBlock -----------------------------------------------------------------------
DecoderBlock::DecoderBlock(const LmConfig& c, int index, ParamFactory& f) : c_(c), index_(index) {
  const std::string p = "h" + std::to_string(index) + ".";
  const std::int64_t d = c.d_model;
  const std::int64_t H = c.n_head;
  const std::int64_t Hk = c.kv_heads();
  const std::int64_t D = c.hdim();
  const std::int64_t F = c.d_ff;
  const float s = c.init_std;
  const float m = f.width_mult();
  const bool mup = c.init == InitKind::MuP;
  const float hidden = mup ? s / std::sqrt(m) : s;
  const float hidden_lr = mup ? 1.0f / m : 1.0f;
  const float resid = c.init == InitKind::Llama ? hidden : hidden / std::sqrt(2.0f * static_cast<float>(c.n_layer));
  const bool ln_bias = c.norm == NormKind::LayerNorm && c.bias;
  n1w_ = f.constant(p + "norm1.weight", {d}, 1.0f);
  if (ln_bias) {
    n1b_ = f.constant(p + "norm1.bias", {d}, 0.0f);
  }
  qkvw_ = f.normal(p + "attn.qkv.weight", {d, (H + 2 * Hk) * D}, hidden, true, hidden_lr);
  if (c.bias || c.qkv_bias) {
    qkvb_ = f.constant(p + "attn.qkv.bias", {(H + 2 * Hk) * D}, 0.0f);
  }
  if (c.qk_norm) {
    qn_ = f.constant(p + "attn.q_norm.weight", {D}, 1.0f);
    kn_ = f.constant(p + "attn.k_norm.weight", {D}, 1.0f);
  }
  ow_ = f.normal(p + "attn.o.weight", {H * D, d}, resid, true, hidden_lr);
  if (c.bias) {
    ob_ = f.constant(p + "attn.o.bias", {d}, 0.0f);
  }
  if (!c.parallel_block) {
    n2w_ = f.constant(p + "norm2.weight", {d}, 1.0f);
    if (ln_bias) {
      n2b_ = f.constant(p + "norm2.bias", {d}, 0.0f);
    }
  }
  moe_ = c.is_moe_layer(index);
  if (!moe_ || c.moe_shared_expert) {  // MoE'de yogun MLP yalniz paylasimli uzman olarak
    const std::int64_t up = c.gated_mlp() ? 2 * F : F;
    upw_ = f.normal(p + "mlp.up.weight", {d, up}, hidden, true, hidden_lr);
    if (c.bias) {
      upb_ = f.constant(p + "mlp.up.bias", {up}, 0.0f);
    }
    downw_ = f.normal(p + "mlp.down.weight", {F, d}, resid, true, hidden_lr);
    if (c.bias) {
      downb_ = f.constant(p + "mlp.down.bias", {d}, 0.0f);
    }
  }
  if (moe_) {
    const std::int64_t E = c.moe_experts, Fe = c.expert_ff();
    router_ = f.normal(p + "moe.router.weight", {d, E}, hidden, false, hidden_lr);  // FP32, sonumsuz
    if (c.moe_noisy) noise_ = f.constant(p + "moe.noise.weight", {d, E}, 0.0f);
    eup_ = f.normal(p + "moe.experts.up.weight", {E, d, 2 * Fe}, hidden, true, hidden_lr);
    edown_ = f.normal(p + "moe.experts.down.weight", {E, Fe, d}, resid, true, hidden_lr);
  }
}

Variable DecoderBlock::ffn(const Variable& h, const BlockContext& ctx) const {
  if (!moe_) return mlp(h);
  moe::Spec s;
  s.experts = c_.moe_experts;
  s.top_k = c_.moe_top_k;
  s.d_model = c_.d_model;
  s.d_ff = c_.expert_ff();
  s.capacity_factor = c_.moe_capacity_factor;
  s.min_capacity = c_.moe_min_capacity;
  s.overflow = c_.moe_reroute ? moe::Overflow::Reroute : moe::Overflow::Drop;
  s.groups = c_.moe_groups;
  s.noisy = c_.moe_noisy;
  s.aux_alpha = c_.moe_aux_alpha;
  s.z_alpha = c_.moe_z_alpha;
  const std::uint64_t seed = ctx.dropout_seed ^ (0x5EEDull + static_cast<std::uint64_t>(index_) * 0x9E3779B97F4A7C15ULL);
  Variable rl;
  MoeLayerStats st;
  st.layer = index_;
  Variable y = moe_ffn(h, router_, noise_, eup_, edown_, s, ctx.training, seed, ctx.router_losses ? &rl : nullptr,
                       ctx.moe_stats ? &st : nullptr);
  if (ctx.router_losses && (s.aux_alpha != 0.0f || s.z_alpha != 0.0f)) ctx.router_losses->push_back(rl);
  if (ctx.moe_stats) ctx.moe_stats->push_back(std::move(st));
  if (c_.moe_shared_expert) y = ag::add(y, mlp(h));
  return y;
}

Variable DecoderBlock::norm(const Variable& x, const Variable& w, const Variable& b) const {
  if (c_.norm == NormKind::RmsNorm) {
    return fused::rms_norm(x, w, c_.norm_eps);
  }
  if (b.has_data()) {
    return fused::fused_layernorm(x, w, b, c_.norm_eps);
  }
  const Variable zero = Variable::make(std::vector<float>(w.values().size(), 0.0f), w.shape(), false);
  return fused::fused_layernorm(x, w, zero, c_.norm_eps);
}

Variable DecoderBlock::linear(const Variable& x, const Variable& w, const Variable& b) const {
  return lm::linear(x, w, b);
}

Variable DecoderBlock::dropout(const Variable& x, const BlockContext& ctx, int slot) const {
  if (!ctx.training || c_.resid_dropout <= 0.0f) {
    return x;
  }
  const std::uint64_t offset = (static_cast<std::uint64_t>(index_) * 4 + static_cast<std::uint64_t>(slot)) << 40;
  return fused::dropout_philox(x, c_.resid_dropout, ctx.dropout_seed, offset, true);
}

Variable DecoderBlock::attn(const Variable& h, const BlockContext& ctx) const {
  const int H = c_.n_head;
  const int Hk = c_.kv_heads();
  const int D = c_.hdim();
  const Variable qkv = linear(h, qkvw_, qkvb_);
  Variable q = ag::narrow(qkv, 1, 0, static_cast<std::int64_t>(H) * D);
  Variable k = ag::narrow(qkv, 1, static_cast<std::int64_t>(H) * D, static_cast<std::int64_t>(Hk) * D);
  const Variable v = ag::narrow(qkv, 1, static_cast<std::int64_t>(H + Hk) * D, static_cast<std::int64_t>(Hk) * D);
  if (c_.qk_norm) {
    q = rms_norm_heads(q, qn_, H, D, c_.norm_eps);
    k = rms_norm_heads(k, kn_, Hk, D, c_.norm_eps);
  }
  if (c_.pos == PosKind::Rope) {
    q = rope(q, H, D, ctx.positions, ctx.rope);
    k = rope(k, Hk, D, ctx.positions, ctx.rope);
  }
  AttentionSpec spec;
  spec.batch = ctx.batch;
  spec.time = ctx.time;
  spec.heads = H;
  spec.kv_heads = Hk;
  spec.head_dim = D;
  spec.causal = true;
  spec.window = c_.window;
  spec.softcap = c_.attn_softcap;
  spec.scale = c_.init == InitKind::MuP ? 1.0f / static_cast<float>(D) : 0.0f;
  spec.alibi_slopes = ctx.alibi;
  if (ctx.training && c_.attn_dropout > 0.0f) {
    spec.dropout_p = c_.attn_dropout;
    spec.seed = ctx.dropout_seed;
    spec.offset = (static_cast<std::uint64_t>(index_) * 4 + 3) << 40;
  }
  return linear(attention(q, k, v, spec), ow_, ob_);
}

Variable DecoderBlock::mlp(const Variable& h) const {
  const Variable up = linear(h, upw_, upb_);
  Variable act;
  switch (c_.mlp) {
    case MlpKind::Gelu:
      act = ag::gelu(up);
      break;
    case MlpKind::SwiGlu:
      act = swiglu(up);
      break;
    case MlpKind::GeGlu:
      act = geglu(up);
      break;
  }
  return linear(act, downw_, downb_);
}

Variable DecoderBlock::forward(const Variable& x, const BlockContext& ctx) const {
  if (!c_.prenorm) {
    const Variable h = norm(ag::add(x, dropout(attn(x, ctx), ctx, 0)), n1w_, n1b_);
    return norm(ag::add(h, dropout(ffn(h, ctx), ctx, 1)), n2w_, n2b_);
  }
  const Variable h = norm(x, n1w_, n1b_);
  if (c_.parallel_block) {
    return ag::add(ag::add(x, dropout(attn(h, ctx), ctx, 0)), dropout(ffn(h, ctx), ctx, 1));
  }
  const Variable r = ag::add(x, dropout(attn(h, ctx), ctx, 0));
  return ag::add(r, dropout(ffn(norm(r, n2w_, n2b_), ctx), ctx, 1));
}

nxs::LayerSpec DecoderBlock::nxs_spec() const {
  return nxs::spec_of(nxs::LayerKind::DecoderBlock, "DecoderBlock",
                      {nxs::i32("index", index_), nxs::i32("d_model", c_.d_model), nxs::i32("n_head", c_.n_head),
                       nxs::i32("n_kv_head", c_.kv_heads()), nxs::i32("head_dim", c_.hdim()), nxs::i32("d_ff", c_.d_ff),
                       nxs::str("norm", to_string(c_.norm)), nxs::str("mlp", to_string(c_.mlp)),
                       nxs::boolean("prenorm", c_.prenorm), nxs::boolean("parallel", c_.parallel_block)});
}

nxs::LayerSpec DecoderBlock::nxs_moe_spec() const {
  return nxs::spec_of(nxs::LayerKind::MoEFFN, "MoEFFN",
                      {nxs::i32("experts", c_.moe_experts), nxs::i32("top_k", c_.moe_top_k), nxs::i32("d_ff", c_.expert_ff()),
                       nxs::boolean("shared_expert", c_.moe_shared_expert), nxs::boolean("noisy", c_.moe_noisy)});
}

// ---- LanguageModel ----------------------------------------------------------------------
LanguageModel::LanguageModel(const LmConfig& config) : config_(config) {
  validate(config_);
  const LmConfig& c = config_;
  ParamFactory f(config_, &params_);
  const std::int64_t d = c.d_model;
  const std::int64_t Vp = c.padded_vocab();
  wte_ = f.normal("wte", {Vp, d}, c.init_std, true);
  if (c.pos == PosKind::Learned) {
    wpe_ = f.normal("wpe", {c.max_position, d}, c.init_std, true);
  }
  blocks_.reserve(static_cast<std::size_t>(c.n_layer));
  for (int l = 0; l < c.n_layer; ++l) {
    blocks_.emplace_back(config_, l, f);
  }
  nfw_ = f.constant("norm_f.weight", {d}, 1.0f);
  if (c.norm == NormKind::LayerNorm && c.bias) {
    nfb_ = f.constant("norm_f.bias", {d}, 0.0f);
  }
  if (!c.tie_embeddings) {
    head_ = f.normal("lm_head.weight", {Vp, d}, c.init_std, true);
  }
  if (Vp > c.vocab_size) {
    std::vector<float> mask(static_cast<std::size_t>(Vp), 0.0f);
    std::fill(mask.begin() + c.vocab_size, mask.end(), -std::numeric_limits<float>::infinity());
    pad_mask_ = Variable::make(std::move(mask), Shape{{Vp}}, false);
  }
  if (c.pos == PosKind::Rope) {
    rope_ = std::make_shared<RopeCache>(c.rot_dim(), c.rope_theta, c.rope_scaling, c.max_position);
  }
  if (c.pos == PosKind::Alibi) {
    alibi_ = std::make_shared<const std::vector<float>>(alibi_slopes(c.n_head));
  }
}

std::int64_t LanguageModel::num_parameters() const noexcept {
  std::int64_t n = 0;
  for (const NamedParam& p : params_) {
    n += p.value.numel();
  }
  return n;
}

const NamedParam* LanguageModel::find(const std::string& name) const noexcept {
  for (const NamedParam& p : params_) {
    if (p.name == name) {
      return &p;
    }
  }
  return nullptr;
}

Variable LanguageModel::embed(const std::vector<std::int32_t>& tokens, int batch, int time,
                              const ForwardOptions& o) const {
  const LmConfig& c = config_;
  if (c.pos == PosKind::Learned) {
    if (o.position_offset != 0 || time > c.max_position) {
      throw TensorError("Nexus lm: ogrenilen konum kodu offset/T > max_position desteklemez");
    }
    return fused::embedding_pos(wte_, wpe_, tokens, batch, time);
  }
  std::vector<std::int64_t> idx(tokens.begin(), tokens.end());
  Variable x = ag::embedding(wte_, idx);
  if (c.pos == PosKind::Sinusoidal) {
    // Tembel tablo: max_len siniri yok (Vaswani ve ark., her konum icin yeniden hesaplanir).
    const int d = c.d_model;
    std::vector<float> table(static_cast<std::size_t>(batch) * time * d);
    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < time; ++t) {
        const double pos = t + o.position_offset;
        for (int i = 0; i < d; ++i) {
          const double angle = pos / std::pow(10000.0, static_cast<double>(i / 2 * 2) / d);
          table[(static_cast<std::size_t>(b) * time + t) * d + i] =
              static_cast<float>(i % 2 == 0 ? std::sin(angle) : std::cos(angle));
        }
      }
    }
    x = ag::add(x, Variable::make(std::move(table), x.shape(), false));
  }
  return x;
}

Variable LanguageModel::logits(const std::vector<std::int32_t>& tokens, int batch, int time,
                               const ForwardOptions& o) const {
  return forward_logits(tokens, batch, time, o, nullptr);
}

Variable LanguageModel::forward_logits(const std::vector<std::int32_t>& tokens, int batch, int time, const ForwardOptions& o,
                                       std::vector<Variable>* aux) const {
  if (static_cast<int>(tokens.size()) != batch * time) {
    throw TensorError("Nexus lm: tokens boyu batch*time olmali");
  }
  for (const std::int32_t t : tokens) {
    if (t < 0 || t >= config_.vocab_size) {
      throw TensorError("Nexus lm: token sozluk disinda");
    }
  }
  BlockContext ctx;
  ctx.batch = batch;
  ctx.time = time;
  ctx.training = o.training;
  ctx.dropout_seed = o.dropout_seed;
  ctx.alibi = alibi_;
  ctx.router_losses = aux;
  moe_stats_.clear();
  ctx.moe_stats = &moe_stats_;
  if (rope_) {
    rope_->ensure(time + o.position_offset);
    auto pos = std::make_shared<std::vector<int>>(static_cast<std::size_t>(batch) * time);
    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < time; ++t) {
        (*pos)[static_cast<std::size_t>(b) * time + t] = t + o.position_offset;
      }
    }
    ctx.positions = pos;
    // ensure tabloyu yeniden kurabilir: bu ileri gecis icin degismez kopya.
    ctx.rope = std::make_shared<const RopeCache>(*rope_);
  }
  Variable x = embed(tokens, batch, time, o);
  for (const DecoderBlock& block : blocks_) {
    const bool ckpt = config_.recompute == RecomputeKind::Full ||
                      (config_.recompute == RecomputeKind::EveryK && block.index() % config_.recompute_every == 0);
    if (ckpt) {
      // Yalniz blok girisi saklanir; geri geciste blok yeniden calisir (dropout ayni seed/offset).
      x = ag::checkpoint([&block, ctx](const std::vector<Variable>& in) { return block.forward(in[0], ctx); }, {x});
    } else {
      x = block.forward(x, ctx);
    }
  }
  if (config_.norm == NormKind::RmsNorm) {
    x = fused::rms_norm(x, nfw_, config_.norm_eps);
  } else {
    const Variable b = nfb_.has_data() ? nfb_
                                       : Variable::make(std::vector<float>(nfw_.values().size(), 0.0f), nfw_.shape(), false);
    x = fused::fused_layernorm(x, nfw_, b, config_.norm_eps);
  }
  Variable out = linear_t(x, config_.tie_embeddings ? wte_ : head_);
  if (config_.init == InitKind::MuP) {
    out = ag::mul_scalar(out, 1.0f / (static_cast<float>(config_.d_model) / config_.mup_base_width));
  }
  if (pad_mask_.has_data()) {
    out = ag::add_bias(out, pad_mask_);
  }
  return out;
}

Variable LanguageModel::loss(const std::vector<std::int32_t>& tokens, int batch, int time,
                             const ForwardOptions& o) const {
  std::vector<Variable> aux;
  Variable loss = fused::cross_entropy_shifted(forward_logits(tokens, batch, time, o, &aux), tokens, batch, time);
  for (const Variable& a : aux) loss = ag::add(loss, a);  // FAZ 14: MoE yonlendirici kayiplari (aux + z)
  return loss;
}

std::vector<double> LanguageModel::layer_grad_norms() const {
  std::vector<double> out(static_cast<std::size_t>(config_.n_layer), 0.0);
  for (const NamedParam& p : params_) {
    if (p.name.size() < 2 || p.name[0] != 'h' || !std::isdigit(static_cast<unsigned char>(p.name[1]))) {
      continue;
    }
    const int layer = std::stoi(p.name.substr(1));
    const Variable* g = grad_of(p.value);
    if (g == nullptr) {
      continue;
    }
    double acc = 0.0;
    for (const float x : g->values()) {
      acc += static_cast<double>(x) * x;
    }
    out[static_cast<std::size_t>(layer)] += acc;
  }
  for (double& x : out) {
    x = std::sqrt(x);
  }
  return out;
}

void LanguageModel::zero_grad() {
  for (NamedParam& p : params_) {
    p.value.zero_grad();
  }
}

void LanguageModel::save_nxs(const std::string& path) const {
  nxs::Archive archive;
  nxs::LayerSpec root = nxs::spec_of(nxs::LayerKind::LanguageModel, "LanguageModel",
                                     {nxs::str("config", to_json(config_)), nxs::i32("format", 1)});
  for (const DecoderBlock& b : blocks_) {
    nxs::LayerSpec bs = b.nxs_spec();
    if (b.is_moe()) bs.children.push_back(b.nxs_moe_spec());  // FAZ 14: NexusMoE cocugu (tur 36)
    root.children.push_back(std::move(bs));
  }
  archive.layers.push_back(std::move(root));
  for (const NamedParam& p : params_) {
    nxs::Weight w;
    w.name = p.name;
    for (const std::int64_t d : p.value.shape().dims) {
      w.dims.push_back(static_cast<std::uint64_t>(d));
    }
    w.values = p.value.values();
    archive.weights.push_back(std::move(w));
  }
  nxs::save_archive(path, archive);
}

LanguageModel LanguageModel::load_nxs(const std::string& path) {
  const nxs::Archive archive = nxs::load_archive(path);
  if (archive.layers.size() != 1 || archive.layers[0].type != static_cast<std::uint16_t>(nxs::LayerKind::LanguageModel)) {
    throw TensorError("Nexus lm: arsiv bir LanguageModel (tur 35) degil");
  }
  const nxs::Attr* cfg = nxs::find_attr(archive.layers[0], "config");
  if (cfg == nullptr || cfg->kind != nxs::AttrKind::String) {
    throw TensorError("Nexus lm: arsivde config JSON yok");
  }
  LanguageModel model(from_json(cfg->s));
  if (archive.weights.size() != model.params_.size()) {
    throw TensorError("Nexus lm: arsiv agirlik sayisi modelle uyusmuyor");
  }
  for (const nxs::Weight& w : archive.weights) {
    const NamedParam* p = model.find(w.name);
    if (p == nullptr || w.values.size() != p->value.values().size()) {
      throw TensorError("Nexus lm: arsiv agirligi modelle uyusmuyor: " + w.name);
    }
    Variable v = p->value;
    v.mutable_values() = w.values;
  }
  return model;
}

LanguageModel build(const LmConfig& config) { return LanguageModel(config); }

LanguageModel build(const std::string& preset_name, const std::string& overrides_json) {
  return LanguageModel(apply_overrides(preset(preset_name), overrides_json));
}

// ---- AdamW --------------------------------------------------------------------------------
AdamW::AdamW(const LanguageModel& model, fused::AdamWConfig config) : model_(&model), config_(config) {
  for (const NamedParam& p : model.parameters()) {
    m_.emplace_back(p.value.values().size(), 0.0f);
    v_.emplace_back(p.value.values().size(), 0.0f);
  }
}

fused::AdamWStepInfo AdamW::step() {
  ++step_;
  const auto& params = model_->parameters();
  std::vector<fused::ParamSlot> all;
  std::vector<std::vector<float>> zeros;  // gradyani olmayan parametre (kullanilmayan) icin
  zeros.reserve(params.size());
  for (std::size_t i = 0; i < params.size(); ++i) {
    Variable var = params[i].value;
    const Variable* g = grad_of(var);
    const float* gp = nullptr;
    if (g != nullptr) {
      gp = g->values().data();
    } else {
      zeros.emplace_back(var.values().size(), 0.0f);
      gp = zeros.back().data();
    }
    all.push_back({var.mutable_values().data(), gp, m_[i].data(), v_[i].data(), var.values().size()});
  }
  fused::AdamWStepInfo info;
  info.grad_norm = fused::global_grad_norm(all);
  if (config_.max_grad_norm > 0.0 && info.grad_norm > config_.max_grad_norm) {
    info.clip_coef = static_cast<float>(config_.max_grad_norm / (info.grad_norm + 1e-6));
    for (std::size_t i = 0; i < params.size(); ++i) {
      float* g = const_cast<float*>(all[i].grad);
      for (std::size_t j = 0; j < all[i].numel; ++j) {
        g[j] *= info.clip_coef;
      }
    }
  }
  std::map<std::pair<bool, float>, std::vector<fused::ParamSlot>> groups;
  for (std::size_t i = 0; i < params.size(); ++i) {
    groups[{params[i].decay, params[i].lr_mult}].push_back(all[i]);
  }
  for (const auto& [key, slots] : groups) {
    fused::AdamWConfig c = config_;
    c.max_grad_norm = 0.0;
    c.weight_decay = key.first ? config_.weight_decay : 0.0;
    c.lr = config_.lr * key.second;
    (void)fused::adamw_multi_tensor(std::span<const fused::ParamSlot>(slots), c, step_);
  }
  return info;
}

}  // namespace lm
}  // namespace nexus
