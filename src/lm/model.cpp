/**
 * @file model.cpp
 * @brief FAZ 6.C-6.F ve FAZ 18.5: DecoderBlock, LanguageModel, baslatma/muP, AdamW gruplari, NXS
 *        ve sifir-ayirmali (zero-allocation) FP32 / BF16 / INT8 / INT4 CPU cikarim motoru.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/model.hpp"

#include "nexus/attention/cpu.hpp"
#include "nexus/core/error.hpp"
#include "nexus/cpu/gemm.hpp"
#include "nexus/cpu/threads.hpp"
#include "nexus/fused/fused.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <span>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

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

#if defined(__AVX2__)
inline float hsum256_ps(__m256 v) noexcept {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

inline __m256 fast_exp256_ps(__m256 x) noexcept {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.33654f)), _mm256_set1_ps(88.37626f));
  __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
  const __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}
#endif

// Hizli AVX2 RMSNorm / LayerNorm (in-place veya ayri tampon).
void norm_rows_cpu(const float* x, const float* w, const float* b, float* y, int rows, int cols, float eps,
                   bool rms) noexcept {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * cols;
    float* yr = y + static_cast<std::size_t>(r) * cols;
    if (rms) {
      float sum_sq = 0.0f;
      int j = 0;
#if defined(__AVX2__)
      __m256 vsq = _mm256_setzero_ps();
      for (; j + 8 <= cols; j += 8) {
        const __m256 vx = _mm256_loadu_ps(xr + j);
        vsq = _mm256_fmadd_ps(vx, vx, vsq);
      }
      sum_sq = hsum256_ps(vsq);
#endif
      for (; j < cols; ++j) sum_sq += xr[j] * xr[j];
      const float inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
      j = 0;
#if defined(__AVX2__)
      const __m256 vinv = _mm256_set1_ps(inv);
      for (; j + 8 <= cols; j += 8) {
        _mm256_storeu_ps(yr + j, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + j), vinv), _mm256_loadu_ps(w + j)));
      }
#endif
      for (; j < cols; ++j) yr[j] = xr[j] * inv * w[j];
    } else {
      float sum = 0.0f;
      int j = 0;
#if defined(__AVX2__)
      __m256 vs = _mm256_setzero_ps();
      for (; j + 8 <= cols; j += 8) vs = _mm256_add_ps(vs, _mm256_loadu_ps(xr + j));
      sum = hsum256_ps(vs);
#endif
      for (; j < cols; ++j) sum += xr[j];
      const float mean = sum / static_cast<float>(cols);
      float var_sum = 0.0f;
      j = 0;
#if defined(__AVX2__)
      const __m256 vmean = _mm256_set1_ps(mean);
      __m256 vv = _mm256_setzero_ps();
      for (; j + 8 <= cols; j += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(xr + j), vmean);
        vv = _mm256_fmadd_ps(d, d, vv);
      }
      var_sum = hsum256_ps(vv);
#endif
      for (; j < cols; ++j) {
        const float d = xr[j] - mean;
        var_sum += d * d;
      }
      const float inv = 1.0f / std::sqrt(var_sum / static_cast<float>(cols) + eps);
      j = 0;
#if defined(__AVX2__)
      const __m256 vinv = _mm256_set1_ps(inv);
      for (; j + 8 <= cols; j += 8) {
        __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + j), vmean), vinv);
        v = _mm256_mul_ps(v, _mm256_loadu_ps(w + j));
        if (b) v = _mm256_add_ps(v, _mm256_loadu_ps(b + j));
        _mm256_storeu_ps(yr + j, v);
      }
#endif
      for (; j < cols; ++j) {
        yr[j] = (xr[j] - mean) * inv * w[j] + (b ? b[j] : 0.0f);
      }
    }
  }
}

void add_inplace_cpu(float* dst, const float* src, std::size_t n) noexcept {
  std::size_t i = 0;
#if defined(__AVX2__)
  for (; i + 16 <= n; i += 16) {
    _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    _mm256_storeu_ps(dst + i + 8, _mm256_add_ps(_mm256_loadu_ps(dst + i + 8), _mm256_loadu_ps(src + i + 8)));
  }
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
  }
#endif
  for (; i < n; ++i) dst[i] += src[i];
}

void mlp_act_cpu(const float* up, float* out, int rows, int F, MlpKind kind) noexcept {
  constexpr float kInvSqrt2 = 0.70710678118654752f;
  if (kind == MlpKind::Gelu) {
    const std::size_t total = static_cast<std::size_t>(rows) * F;
    for (std::size_t i = 0; i < total; ++i) {
      const float a = up[i];
      out[i] = a * 0.5f * (1.0f + std::erf(a * kInvSqrt2));
    }
    return;
  }
  for (int r = 0; r < rows; ++r) {
    const float* a = up + static_cast<std::size_t>(r) * 2 * F;
    const float* b = a + F;
    float* y = out + static_cast<std::size_t>(r) * F;
    int i = 0;
    if (kind == MlpKind::SwiGlu) {
#if defined(__AVX2__)
      const __m256 one = _mm256_set1_ps(1.0f);
      const __m256 zero = _mm256_setzero_ps();
      for (; i + 8 <= F; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 sg = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp256_ps(_mm256_sub_ps(zero, va))));
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(va, sg), vb));
      }
#endif
      for (; i < F; ++i) {
        const float sg = 1.0f / (1.0f + std::exp(-a[i]));
        y[i] = (a[i] * sg) * b[i];
      }
    } else {
      for (; i < F; ++i) {
        const float cdf = 0.5f * (1.0f + std::erf(a[i] * kInvSqrt2));
        y[i] = (a[i] * cdf) * b[i];
      }
    }
  }
}

// Tek-token (Tq == 1, batch == 1) GQA dikkat: KV onbellegini kopyalamadan dogrudan ck/cv uzerinde calisir.
void decode_attention_1tok_cpu(float* out, const float* q, const float* ck, const float* cv, int H, int Hk, int D,
                               int tk, float scale, int window, float softcap, const float* alibi_slopes) {
  const int group = H / Hk;
  const int qpos = tk - 1;
  const int j_lo = window > 0 ? std::max(0, qpos - window + 1) : 0;
  const int span = tk - j_lo;

  auto run_head = [&](int h) {
    const int hk = h / group;
    const float* qi = q + static_cast<std::size_t>(h) * D;
    float* oi = out + static_cast<std::size_t>(h) * D;
    if (span <= 0) {
      std::memset(oi, 0, static_cast<std::size_t>(D) * sizeof(float));
      return;
    }
    thread_local std::vector<float> scores;
    if (static_cast<int>(scores.size()) < span) scores.resize(static_cast<std::size_t>(span));
    const float slope = alibi_slopes ? alibi_slopes[h] : 0.0f;
    float m = -std::numeric_limits<float>::infinity();
    for (int j = j_lo; j < tk; ++j) {
      const float* kj = ck + (static_cast<std::size_t>(j) * Hk + hk) * D;
      float dot = 0.0f;
      int d = 0;
#if defined(__AVX2__)
      __m256 vdot0 = _mm256_setzero_ps(), vdot1 = _mm256_setzero_ps();
      for (; d + 16 <= D; d += 16) {
        vdot0 = _mm256_fmadd_ps(_mm256_loadu_ps(qi + d), _mm256_loadu_ps(kj + d), vdot0);
        vdot1 = _mm256_fmadd_ps(_mm256_loadu_ps(qi + d + 8), _mm256_loadu_ps(kj + d + 8), vdot1);
      }
      for (; d + 8 <= D; d += 8) {
        vdot0 = _mm256_fmadd_ps(_mm256_loadu_ps(qi + d), _mm256_loadu_ps(kj + d), vdot0);
      }
      dot = hsum256_ps(_mm256_add_ps(vdot0, vdot1));
#endif
      for (; d < D; ++d) dot += qi[d] * kj[d];
      float s = dot * scale;
      if (softcap > 0.0f) s = softcap * std::tanh(s / softcap);
      if (alibi_slopes) s += slope * static_cast<float>(j - qpos);
      scores[static_cast<std::size_t>(j - j_lo)] = s;
      m = std::max(m, s);
    }
    float l = 0.0f;
    for (int idx = 0; idx < span; ++idx) {
      const float p = std::exp(scores[static_cast<std::size_t>(idx)] - m);
      scores[static_cast<std::size_t>(idx)] = p;
      l += p;
    }
    const float inv_l = l > 0.0f ? (1.0f / l) : 0.0f;
    std::memset(oi, 0, static_cast<std::size_t>(D) * sizeof(float));
    for (int idx = 0; idx < span; ++idx) {
      const float p = scores[static_cast<std::size_t>(idx)] * inv_l;
      const float* vj = cv + (static_cast<std::size_t>(j_lo + idx) * Hk + hk) * D;
      int d = 0;
#if defined(__AVX2__)
      const __m256 vp = _mm256_set1_ps(p);
      for (; d + 16 <= D; d += 16) {
        _mm256_storeu_ps(oi + d, _mm256_fmadd_ps(vp, _mm256_loadu_ps(vj + d), _mm256_loadu_ps(oi + d)));
        _mm256_storeu_ps(oi + d + 8, _mm256_fmadd_ps(vp, _mm256_loadu_ps(vj + d + 8), _mm256_loadu_ps(oi + d + 8)));
      }
      for (; d + 8 <= D; d += 8) {
        _mm256_storeu_ps(oi + d, _mm256_fmadd_ps(vp, _mm256_loadu_ps(vj + d), _mm256_loadu_ps(oi + d)));
      }
#endif
      for (; d < D; ++d) oi[d] += p * vj[d];
    }
  };

  if (H >= 8 && span >= 32) {
    cpu::Pool::instance().run(static_cast<std::size_t>(H), [&](int, std::size_t b, std::size_t e) {
      for (std::size_t h = b; h < e; ++h) run_head(static_cast<int>(h));
    });
  } else {
    for (int h = 0; h < H; ++h) run_head(h);
  }
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
    if (zero_) break;  // FAZ 17: yuklenecek agirliklar (vector zaten sifir)
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
  Variable kk = k, vv = v;
  AttentionSpec spec;
  if (ctx.cache != nullptr) {  // FAZ 16: onceki K/V + yeni (batch 1)
    std::vector<float>& ck = ctx.cache->k[static_cast<std::size_t>(index_)];
    std::vector<float>& cv = ctx.cache->v[static_cast<std::size_t>(index_)];
    ck.insert(ck.end(), k.values().begin(), k.values().end());
    cv.insert(cv.end(), v.values().begin(), v.values().end());
    const std::int64_t tk = static_cast<std::int64_t>(ck.size()) / (static_cast<std::int64_t>(Hk) * D);
    kk = Variable::make(ck, Shape{{tk, static_cast<std::int64_t>(Hk) * D}}, false);
    vv = Variable::make(cv, Shape{{tk, static_cast<std::int64_t>(Hk) * D}}, false);
    spec.kv_time = static_cast<int>(tk);
  }
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
  return linear(attention(q, kk, vv, spec), ow_, ob_);
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

// ---- FAZ 18.5: Sifir-ayirmali CPU Cikarim Motoru (FP32 / BF16 / INT8 / INT4) -------------
struct PreparedWeight {
  int K = 0;  // in_features
  int N = 0;  // out_features
  CpuWeightMode mode = CpuWeightMode::Auto;
  bool b_trans = false;
  const float* raw_f32 = nullptr;         // Auto modunda dogrudan Variable belleği
  std::vector<float> f32_nt;              // F32 modunda [N, K] devriklenmis (M=1'de birim adimli akis)
  std::vector<std::uint16_t> bf16_nt;     // BF16 modunda [N, K] devriklenmis
  cpu::PackedS8 s8;                       // INT8 modunda [ceil(N/16)][Kp/4][16][4]
  cpu::PackedQ4 q4;                       // INT4 modunda [ceil(N/16)][Kp/8][16][4]

  [[nodiscard]] std::size_t bytes() const noexcept {
    switch (mode) {
      case CpuWeightMode::Auto:
        return static_cast<std::size_t>(K) * N * sizeof(float);
      case CpuWeightMode::F32:
        return f32_nt.size() * sizeof(float);
      case CpuWeightMode::BF16:
        return bf16_nt.size() * sizeof(std::uint16_t);
      case CpuWeightMode::INT8:
        return s8.data.size() * sizeof(std::int8_t) + s8.scale.size() * sizeof(float);
      case CpuWeightMode::INT4:
        return q4.data.size() * sizeof(std::uint8_t) + q4.scale.size() * sizeof(float);
    }
    return 0;
  }

  static PreparedWeight build(const float* src, int K, int N, bool src_is_nt, CpuWeightMode mode) {
    PreparedWeight pw;
    pw.K = K;
    pw.N = N;
    pw.mode = mode;
    if (mode == CpuWeightMode::Auto) {
      pw.raw_f32 = src;
      pw.b_trans = src_is_nt;
      return pw;
    }
    if (mode == CpuWeightMode::F32 || mode == CpuWeightMode::BF16) {
      pw.b_trans = true;
      std::vector<float> nt(static_cast<std::size_t>(N) * K);
      if (src_is_nt) {
        std::memcpy(nt.data(), src, nt.size() * sizeof(float));
      } else {
        cpu::Pool::instance().run(static_cast<std::size_t>(N), [&](int, std::size_t nb, std::size_t ne) {
          for (std::size_t n = nb; n < ne; ++n) {
            float* dst = nt.data() + n * static_cast<std::size_t>(K);
            for (int k = 0; k < K; ++k) {
              dst[k] = src[static_cast<std::size_t>(k) * N + n];
            }
          }
        });
      }
      if (mode == CpuWeightMode::F32) {
        pw.f32_nt = std::move(nt);
      } else {
        pw.bf16_nt = cpu::to_bf16(nt.data(), nt.size());
      }
      return pw;
    }
    if (mode == CpuWeightMode::INT8) {
      pw.s8 = cpu::pack_s8(src, N, K, src_is_nt);
      return pw;
    }
    pw.q4 = cpu::pack_q4(src, N, K, src_is_nt);
    return pw;
  }
};

struct LanguageModel::CpuInferenceState {
  struct LayerWeights {
    PreparedWeight qkv;
    PreparedWeight o;
    PreparedWeight up;
    PreparedWeight down;
  };
  CpuWeightMode mode = CpuWeightMode::Auto;
  std::vector<LayerWeights> layers;
  PreparedWeight lm_head;
  std::size_t total_bytes = 0;
};

namespace {

struct FastInferenceScratch {
  std::vector<float> x;
  std::vector<float> h;
  std::vector<float> qkv;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> attn_out;
  std::vector<float> lse;
  std::vector<float> up;
  std::vector<float> act;
  std::vector<float> tmp_d;
  std::vector<std::int8_t> aq;
  std::vector<float> sa;

  void ensure(int N, int d, int qkv_dim, int H, int Hk, int D, int up_dim, int F, int max_K) {
    const auto sz = [](int a, int b) { return static_cast<std::size_t>(a) * static_cast<std::size_t>(b); };
    if (x.size() < sz(N, d)) x.resize(sz(N, d));
    if (h.size() < sz(N, d)) h.resize(sz(N, d));
    if (qkv.size() < sz(N, qkv_dim)) qkv.resize(sz(N, qkv_dim));
    if (N > 1) {
      if (q.size() < sz(N, H * D)) q.resize(sz(N, H * D));
      if (k.size() < sz(N, Hk * D)) k.resize(sz(N, Hk * D));
      if (v.size() < sz(N, Hk * D)) v.resize(sz(N, Hk * D));
      if (lse.size() < sz(N, H)) lse.resize(sz(N, H));
    }
    if (attn_out.size() < sz(N, H * D)) attn_out.resize(sz(N, H * D));
    if (up.size() < sz(N, up_dim)) up.resize(sz(N, up_dim));
    if (act.size() < sz(N, F)) act.resize(sz(N, F));
    if (tmp_d.size() < sz(N, d)) tmp_d.resize(sz(N, d));
    const int kp8 = (max_K + 7) / 8 * 8;
    if (aq.size() < sz(N, kp8)) aq.resize(sz(N, kp8));
    if (sa.size() < static_cast<std::size_t>(N)) sa.resize(static_cast<std::size_t>(N));
  }
};

thread_local FastInferenceScratch t_infer_scratch;

void run_prepared_linear(const float* A, int M, const PreparedWeight& W, const float* bias, float* C,
                         FastInferenceScratch& sc) {
  if (W.mode == CpuWeightMode::INT8) {
    cpu::quantize_rows_s8(A, M, W.K, W.K, sc.aq.data(), W.s8.Kp, sc.sa.data());
    cpu::gemm_s8(sc.aq.data(), sc.sa.data(), M, W.s8, C, W.N, bias);
    return;
  }
  if (W.mode == CpuWeightMode::INT4) {
    cpu::quantize_rows_s8(A, M, W.K, W.K, sc.aq.data(), W.q4.Kp, sc.sa.data());
    cpu::gemm_q4(sc.aq.data(), sc.sa.data(), M, W.q4, C, W.N, bias);
    return;
  }
  cpu::GemmArgs g;
  g.M = M;
  g.N = W.N;
  g.K = W.K;
  g.A = A;
  g.lda = W.K;
  g.b_trans = W.b_trans;
  g.ldb = W.b_trans ? W.K : W.N;
  g.C = C;
  g.ldc = W.N;
  g.bias = bias;
  if (W.mode == CpuWeightMode::BF16) {
    g.B = W.bf16_nt.data();
    g.btype = cpu::BType::BF16;
  } else if (W.mode == CpuWeightMode::F32) {
    g.B = W.f32_nt.data();
    g.btype = cpu::BType::F32;
  } else {
    g.B = W.raw_f32;
    g.btype = cpu::BType::F32;
  }
  cpu::gemm(g);
}

}  // namespace

// ---- LanguageModel ----------------------------------------------------------------------
LanguageModel::LanguageModel(const LmConfig& config) : LanguageModel(config, false) {}

LanguageModel::LanguageModel(const LmConfig& config, bool zero_init) : config_(config) {
  validate(config_);
  const LmConfig& c = config_;
  ParamFactory f(config_, &params_);
  f.set_zero_init(zero_init);
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

void LanguageModel::prepare_cpu_inference(CpuWeightMode mode) const {
  cpu_mode_ = mode;
  auto st = std::make_shared<CpuInferenceState>();
  st->mode = mode;
  const int d = config_.d_model;
  const int H = config_.n_head;
  const int Hk = config_.kv_heads();
  const int D = config_.hdim();
  const int F = config_.d_ff;
  const int qkv_dim = (H + 2 * Hk) * D;
  const int up_dim = config_.gated_mlp() ? 2 * F : F;
  const int Vp = config_.padded_vocab();

  st->layers.resize(blocks_.size());
  std::size_t total = 0;
  for (std::size_t l = 0; l < blocks_.size(); ++l) {
    const DecoderBlock& blk = blocks_[l];
    if (blk.moe_) continue;
    auto& lw = st->layers[l];
    lw.qkv = PreparedWeight::build(blk.qkvw_.values().data(), d, qkv_dim, /*src_is_nt=*/false, mode);
    lw.o = PreparedWeight::build(blk.ow_.values().data(), H * D, d, /*src_is_nt=*/false, mode);
    lw.up = PreparedWeight::build(blk.upw_.values().data(), d, up_dim, /*src_is_nt=*/false, mode);
    lw.down = PreparedWeight::build(blk.downw_.values().data(), F, d, /*src_is_nt=*/false, mode);
    total += lw.qkv.bytes() + lw.o.bytes() + lw.up.bytes() + lw.down.bytes();
  }
  const Variable& hw = config_.tie_embeddings ? wte_ : head_;
  // Cikis basligi (lm_head) logit dagilimina dogrudan etki ettiginden INT4 modunda Q8_0 (INT8) tutulur (GGUF Q4_K standardi).
  const CpuWeightMode head_mode = (mode == CpuWeightMode::INT4) ? CpuWeightMode::INT8 : mode;
  st->lm_head = PreparedWeight::build(hw.values().data(), d, Vp, /*src_is_nt=*/true, head_mode);
  total += st->lm_head.bytes();
  st->total_bytes = total;
  cpu_infer_ = std::move(st);
}

std::size_t LanguageModel::cpu_inference_weight_bytes() const noexcept {
  if (cpu_infer_) return cpu_infer_->total_bytes;
  std::size_t n = 0;
  for (const NamedParam& p : params_) n += p.value.values().size() * sizeof(float);
  return n;
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
    if (o.position_offset + time > c.max_position) {
      throw TensorError("Nexus lm: ogrenilen konum kodu: offset + T > max_position");
    }
    if (o.position_offset == 0) return fused::embedding_pos(wte_, wpe_, tokens, batch, time);
    // FAZ 16 (KV onbellekli devam): konum satirlari [offset, offset + T)
    std::vector<std::int64_t> idx(tokens.begin(), tokens.end()), pidx(static_cast<std::size_t>(batch) * time);
    for (int b = 0; b < batch; ++b)
      for (int t = 0; t < time; ++t) pidx[static_cast<std::size_t>(b) * time + t] = t + o.position_offset;
    return ag::add(ag::embedding(wte_, idx), ag::embedding(wpe_, pidx));
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

Variable LanguageModel::forward_cpu_fast(const std::vector<std::int32_t>& tokens, int batch, int time,
                                         const ForwardOptions& o) const {
  const LmConfig& c = config_;
  const int N = batch * time;
  const int d = c.d_model;
  const int H = c.n_head;
  const int Hk = c.kv_heads();
  const int D = c.hdim();
  const int F = c.d_ff;
  const int qkv_dim = (H + 2 * Hk) * D;
  const int up_dim = c.gated_mlp() ? 2 * F : F;
  const int Vp = c.padded_vocab();
  const bool rms = (c.norm == NormKind::RmsNorm);
  const int pos_offset = o.cache ? o.cache->length : o.position_offset;

  if (c.pos == PosKind::Learned && pos_offset + time > c.max_position) {
    throw TensorError("Nexus lm: ogrenilen konum kodu: offset + T > max_position");
  }
  if (rope_) {
    rope_->ensure(time + pos_offset);
  }
  std::vector<int> pos(static_cast<std::size_t>(N));
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < time; ++t) {
      pos[static_cast<std::size_t>(b) * time + t] = t + pos_offset;
    }
  }

  FastInferenceScratch& sc = t_infer_scratch;
  sc.ensure(N, d, qkv_dim, H, Hk, D, up_dim, F, std::max({d, H * D, F}));

  // 1. Gomme (Embedding)
  const float* wte_ptr = wte_.values().data();
  const float* wpe_ptr = (c.pos == PosKind::Learned && wpe_.has_data()) ? wpe_.values().data() : nullptr;
  for (int r = 0; r < N; ++r) {
    float* xr = sc.x.data() + static_cast<std::size_t>(r) * d;
    std::memcpy(xr, wte_ptr + static_cast<std::size_t>(tokens[static_cast<std::size_t>(r)]) * d,
                static_cast<std::size_t>(d) * sizeof(float));
    if (wpe_ptr) {
      add_inplace_cpu(xr, wpe_ptr + static_cast<std::size_t>(pos[static_cast<std::size_t>(r)]) * d,
                      static_cast<std::size_t>(d));
    } else if (c.pos == PosKind::Sinusoidal) {
      const double pval = pos[static_cast<std::size_t>(r)];
      for (int i = 0; i < d; ++i) {
        const double angle = pval / std::pow(10000.0, static_cast<double>(i / 2 * 2) / d);
        xr[i] += static_cast<float>(i % 2 == 0 ? std::sin(angle) : std::cos(angle));
      }
    }
  }

  const CpuInferenceState* st = cpu_infer_.get();

  // 2. Kod cozucu bloklar (Decoder blocks)
  for (std::size_t l = 0; l < blocks_.size(); ++l) {
    const DecoderBlock& blk = blocks_[l];
    norm_rows_cpu(sc.x.data(), blk.n1w_.values().data(), blk.n1b_.has_data() ? blk.n1b_.values().data() : nullptr,
                  sc.h.data(), N, d, c.norm_eps, rms);

    const float* qkv_bias = blk.qkvb_.has_data() ? blk.qkvb_.values().data() : nullptr;
    if (st) {
      run_prepared_linear(sc.h.data(), N, st->layers[l].qkv, qkv_bias, sc.qkv.data(), sc);
    } else {
      PreparedWeight pw = PreparedWeight::build(blk.qkvw_.values().data(), d, qkv_dim, false, CpuWeightMode::Auto);
      run_prepared_linear(sc.h.data(), N, pw, qkv_bias, sc.qkv.data(), sc);
    }

    float* q_ptr = nullptr;
    float* k_ptr = nullptr;
    float* v_ptr = nullptr;
    if (N == 1) {
      q_ptr = sc.qkv.data();
      k_ptr = sc.qkv.data() + static_cast<std::size_t>(H) * D;
      v_ptr = sc.qkv.data() + static_cast<std::size_t>(H + Hk) * D;
    } else {
      q_ptr = sc.q.data();
      k_ptr = sc.k.data();
      v_ptr = sc.v.data();
      for (int r = 0; r < N; ++r) {
        const float* src = sc.qkv.data() + static_cast<std::size_t>(r) * qkv_dim;
        std::memcpy(q_ptr + static_cast<std::size_t>(r) * H * D, src, static_cast<std::size_t>(H * D) * sizeof(float));
        std::memcpy(k_ptr + static_cast<std::size_t>(r) * Hk * D, src + static_cast<std::size_t>(H) * D,
                    static_cast<std::size_t>(Hk * D) * sizeof(float));
        std::memcpy(v_ptr + static_cast<std::size_t>(r) * Hk * D, src + static_cast<std::size_t>(H + Hk) * D,
                    static_cast<std::size_t>(Hk * D) * sizeof(float));
      }
    }

    if (c.qk_norm) {
      norm_rows_cpu(q_ptr, blk.qn_.values().data(), nullptr, q_ptr, N * H, D, c.norm_eps, /*rms=*/true);
      norm_rows_cpu(k_ptr, blk.kn_.values().data(), nullptr, k_ptr, N * Hk, D, c.norm_eps, /*rms=*/true);
    }
    if (c.pos == PosKind::Rope) {
      rope_apply(q_ptr, N, H, D, pos.data(), *rope_, false);
      rope_apply(k_ptr, N, Hk, D, pos.data(), *rope_, false);
    }

    const float* k_attn = k_ptr;
    const float* v_attn = v_ptr;
    int tk = time;
    if (o.cache != nullptr) {
      std::vector<float>& ck = o.cache->k[l];
      std::vector<float>& cv = o.cache->v[l];
      const std::size_t add_len = static_cast<std::size_t>(N) * Hk * D;
      if (ck.capacity() < ck.size() + add_len) {
        ck.reserve(std::max(ck.size() + add_len, static_cast<std::size_t>(256) * Hk * D));
        cv.reserve(std::max(cv.size() + add_len, static_cast<std::size_t>(256) * Hk * D));
      }
      ck.insert(ck.end(), k_ptr, k_ptr + add_len);
      cv.insert(cv.end(), v_ptr, v_ptr + add_len);
      tk = static_cast<int>(ck.size() / (static_cast<std::size_t>(Hk) * D));
      k_attn = ck.data();
      v_attn = cv.data();
    }

    const float attn_scale = c.init == InitKind::MuP ? (1.0f / static_cast<float>(D))
                                                     : (1.0f / std::sqrt(static_cast<float>(D)));
    if (batch == 1 && time == 1) {
      decode_attention_1tok_cpu(sc.attn_out.data(), q_ptr, k_attn, v_attn, H, Hk, D, tk, attn_scale, c.window,
                                c.attn_softcap, alibi_ ? alibi_->data() : nullptr);
    } else {
      AttentionParams p;
      p.batch = batch;
      p.heads = H;
      p.kv_heads = Hk;
      p.tq = time;
      p.tk = tk;
      p.head_dim = D;
      p.dtype = DType::F32;
      p.scale = c.init == InitKind::MuP ? 1.0f / static_cast<float>(D) : 0.0f;
      p.causal = true;
      p.window = c.window;
      p.softcap = c.attn_softcap;
      p.alibi_slopes = alibi_ ? alibi_->data() : nullptr;
      p.set_contiguous_bthd();
      attention::flash_attention_forward_cpu(sc.attn_out.data(), sc.lse.data(), q_ptr, k_attn, v_attn, p);
    }

    const float* o_bias = blk.ob_.has_data() ? blk.ob_.values().data() : nullptr;
    if (st) {
      run_prepared_linear(sc.attn_out.data(), N, st->layers[l].o, o_bias, sc.tmp_d.data(), sc);
    } else {
      PreparedWeight pw = PreparedWeight::build(blk.ow_.values().data(), H * D, d, false, CpuWeightMode::Auto);
      run_prepared_linear(sc.attn_out.data(), N, pw, o_bias, sc.tmp_d.data(), sc);
    }
    add_inplace_cpu(sc.x.data(), sc.tmp_d.data(), static_cast<std::size_t>(N) * d);

    norm_rows_cpu(sc.x.data(), blk.n2w_.values().data(), blk.n2b_.has_data() ? blk.n2b_.values().data() : nullptr,
                  sc.h.data(), N, d, c.norm_eps, rms);

    const float* up_bias = blk.upb_.has_data() ? blk.upb_.values().data() : nullptr;
    if (st) {
      run_prepared_linear(sc.h.data(), N, st->layers[l].up, up_bias, sc.up.data(), sc);
    } else {
      PreparedWeight pw = PreparedWeight::build(blk.upw_.values().data(), d, up_dim, false, CpuWeightMode::Auto);
      run_prepared_linear(sc.h.data(), N, pw, up_bias, sc.up.data(), sc);
    }

    mlp_act_cpu(sc.up.data(), sc.act.data(), N, F, c.mlp);

    const float* down_bias = blk.downb_.has_data() ? blk.downb_.values().data() : nullptr;
    if (st) {
      run_prepared_linear(sc.act.data(), N, st->layers[l].down, down_bias, sc.tmp_d.data(), sc);
    } else {
      PreparedWeight pw = PreparedWeight::build(blk.downw_.values().data(), F, d, false, CpuWeightMode::Auto);
      run_prepared_linear(sc.act.data(), N, pw, down_bias, sc.tmp_d.data(), sc);
    }
    add_inplace_cpu(sc.x.data(), sc.tmp_d.data(), static_cast<std::size_t>(N) * d);
  }

  // 3. Son norm + LM Head
  norm_rows_cpu(sc.x.data(), nfw_.values().data(), nfb_.has_data() ? nfb_.values().data() : nullptr, sc.h.data(), N, d,
                c.norm_eps, rms);

  std::vector<float> out_logits(static_cast<std::size_t>(N) * Vp);
  if (st) {
    run_prepared_linear(sc.h.data(), N, st->lm_head, nullptr, out_logits.data(), sc);
  } else {
    const Variable& hw = c.tie_embeddings ? wte_ : head_;
    PreparedWeight pw = PreparedWeight::build(hw.values().data(), d, Vp, true, CpuWeightMode::Auto);
    run_prepared_linear(sc.h.data(), N, pw, nullptr, out_logits.data(), sc);
  }

  if (c.init == InitKind::MuP) {
    const float mup_scale = 1.0f / (static_cast<float>(c.d_model) / c.mup_base_width);
    for (float& v : out_logits) v *= mup_scale;
  }
  if (Vp > c.vocab_size) {
    for (int r = 0; r < N; ++r) {
      float* row = out_logits.data() + static_cast<std::size_t>(r) * Vp;
      std::fill(row + c.vocab_size, row + Vp, -std::numeric_limits<float>::infinity());
    }
  }
  if (o.cache != nullptr) {
    o.cache->length += time;
  }
  return Variable::make(std::move(out_logits), Shape{{N, Vp}}, false);
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
  ForwardOptions oc = o;
  if (o.cache != nullptr) {
    if (batch != 1) throw TensorError("Nexus lm: KV onbellekli ileri gecis batch 1 ister");
    if (static_cast<int>(o.cache->k.size()) != config_.n_layer) throw TensorError("Nexus lm: KV onbellegi baska bir modelden");
    oc.position_offset = o.cache->length;
  }

  // FAZ 18.5: Egitim disinda (KV onbellekli cozme, prepare_cpu_inference cagrildiysa veya NoGradGuard altinda)
  // yogun on-normalizasyonlu modeller icin sifir-ayirmali hizli CPU cikarim motoru.
  const bool can_fast_cpu = !o.training && aux == nullptr && config_.prenorm && !config_.parallel_block &&
                            config_.moe_experts == 0 &&
                            (o.cache != nullptr || cpu_infer_ != nullptr || !ag::grad_enabled());
  if (can_fast_cpu) {
    return forward_cpu_fast(tokens, batch, time, o);
  }

  BlockContext ctx;
  ctx.cache = o.cache;
  ctx.batch = batch;
  ctx.time = time;
  ctx.training = o.training;
  ctx.dropout_seed = o.dropout_seed;
  ctx.alibi = alibi_;
  ctx.router_losses = aux;
  moe_stats_.clear();
  ctx.moe_stats = &moe_stats_;
  if (rope_) {
    rope_->ensure(time + oc.position_offset);
    auto pos = std::make_shared<std::vector<int>>(static_cast<std::size_t>(batch) * time);
    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < time; ++t) {
        (*pos)[static_cast<std::size_t>(b) * time + t] = t + oc.position_offset;
      }
    }
    ctx.positions = pos;
    // ensure tabloyu yeniden kurabilir: bu ileri gecis icin degismez kopya.
    ctx.rope = std::make_shared<const RopeCache>(*rope_);
  }
  Variable x = embed(tokens, batch, time, oc);
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
  if (o.cache != nullptr) {
    o.cache->length += time;
  }
  return out;
}

KvCache LanguageModel::make_cache() const {
  KvCache c;
  c.k.resize(static_cast<std::size_t>(config_.n_layer));
  c.v.resize(static_cast<std::size_t>(config_.n_layer));
  return c;
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
