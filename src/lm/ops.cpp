/**
 * @file ops.cpp
 * @brief FAZ 6.A/6.B ve FAZ 18: RoPE, SwiGLU/GeGLU (AVX2 + paralel), GQA flash dikkat ve qk_norm autograd op'lari.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/ops.hpp"

#include "nexus/attention/cpu.hpp"
#include "nexus/core/error.hpp"
#include "nexus/cpu/gemm.hpp"
#include "nexus/cpu/threads.hpp"
#include "nexus/fused/training.hpp"
#include "nexus_model/core/kernels.hpp"

#include <cmath>
#include <cstring>
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

inline bool need_grad_any(std::initializer_list<const Variable*> vars) noexcept {
  if (!ag::grad_enabled()) return false;
  for (const Variable* v : vars) {
    if (v != nullptr && v->has_data() && v->requires_grad()) return true;
  }
  return false;
}

}  // namespace

Variable rope(const Variable& x, int heads, int head_dim, std::shared_ptr<const std::vector<int>> positions,
              std::shared_ptr<const RopeCache> cache) {
  const Shape& s = x.shape();
  if (s.rank() != 2 || s.dims[1] != static_cast<std::int64_t>(heads) * head_dim ||
      static_cast<std::int64_t>(positions->size()) != s.dims[0]) {
    throw TensorError("Nexus lm::rope: x [N, heads*head_dim] ve N konum bekler");
  }
  const int rows = static_cast<int>(s.dims[0]);
  std::vector<float> out = x.values();
  rope_apply(out.data(), rows, heads, head_dim, positions->data(), *cache, false);
  if (!need_grad_any({&x})) {
    return Variable::make(std::move(out), s, false);
  }
  return ag::custom_function("lm_rope", {x}, std::move(out), s,
                             [rows, heads, head_dim, positions, cache, s](const Variable& g) {
                               std::vector<float> dx = g.values();
                               rope_apply(dx.data(), rows, heads, head_dim, positions->data(), *cache, true);
                               return std::vector<Variable>{Variable::make(std::move(dx), s, false)};
                             });
}

Variable linear(const Variable& x, const Variable& w, const Variable& b) {
  if (x.shape().rank() != 2 || w.shape().rank() != 2 || x.shape().dims[1] != w.shape().dims[0]) {
    throw TensorError("Nexus lm::linear: x [N, in], w [in, out] bekler");
  }
  const int n = static_cast<int>(x.shape().dims[0]);
  const int in = static_cast<int>(w.shape().dims[0]);
  const int out = static_cast<int>(w.shape().dims[1]);
  const bool has_b = b.has_data();
  if (has_b && b.numel() != out) {
    throw TensorError("Nexus lm::linear: bias boyu out olmali");
  }
  std::vector<float> y(static_cast<std::size_t>(n) * out);
  nexus::cpu::GemmArgs g;
  g.M = n;
  g.N = out;
  g.K = in;
  g.A = x.values().data();
  g.lda = in;
  g.B = w.values().data();
  g.ldb = out;
  g.C = y.data();
  g.ldc = out;
  g.bias = has_b ? b.values().data() : nullptr;
  nexus::cpu::gemm(g);
  if (!need_grad_any({&x, &w, has_b ? &b : nullptr})) {
    return Variable::make(std::move(y), Shape{{n, out}}, false);
  }
  // Kayit saved-tensor kancasindan gecer: checkpoint/offload bu tensorleri yonetebilir.
  const ag::SavedTensor xs = ag::save_variable(x);
  const ag::SavedTensor ws = ag::save_variable(w);
  std::vector<Variable> inputs{x, w};
  if (has_b) {
    inputs.push_back(b);
  }
  return ag::custom_function("lm_linear", inputs, std::move(y), Shape{{n, out}}, [xs, ws, n, in, out, has_b](const Variable& grad) {
    const Variable xv = ag::restore_variable(xs, "lm_linear");
    const Variable wv = ag::restore_variable(ws, "lm_linear");
    const float* dy = grad.values().data();
    std::vector<float> dx(static_cast<std::size_t>(n) * in);
    std::vector<float> dw(static_cast<std::size_t>(in) * out);
    {
      nexus::cpu::GemmArgs gd;
      gd.M = n;
      gd.N = in;
      gd.K = out;
      gd.A = dy;
      gd.lda = out;
      gd.B = wv.values().data();
      gd.ldb = out;
      gd.b_trans = true;
      gd.C = dx.data();
      gd.ldc = in;
      nexus::cpu::gemm(gd);  // dx = dy W^T
    }
    {
      nexus::cpu::GemmArgs gw;
      gw.M = in;
      gw.N = out;
      gw.K = n;
      gw.A = xv.values().data();
      gw.lda = in;
      gw.a_trans = true;
      gw.B = dy;
      gw.ldb = out;
      gw.C = dw.data();
      gw.ldc = out;
      nexus::cpu::gemm(gw);  // dW = x^T dy
    }
    std::vector<Variable> grads{Variable::make(std::move(dx), xv.shape(), false),
                                Variable::make(std::move(dw), wv.shape(), false)};
    if (has_b) {
      std::vector<float> db(static_cast<std::size_t>(out), 0.0f);
      for (int r = 0; r < n; ++r) {
        const float* dyr = dy + static_cast<std::size_t>(r) * out;
        int j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= out; j += 8) {
          _mm256_storeu_ps(db.data() + j, _mm256_add_ps(_mm256_loadu_ps(db.data() + j), _mm256_loadu_ps(dyr + j)));
        }
#endif
        for (; j < out; ++j) {
          db[static_cast<std::size_t>(j)] += dyr[j];
        }
      }
      grads.push_back(Variable::make(std::move(db), Shape{{out}}, false));
    }
    return grads;
  });
}

Variable linear_t(const Variable& x, const Variable& w) {
  if (x.shape().rank() != 2 || w.shape().rank() != 2 || x.shape().dims[1] != w.shape().dims[1]) {
    throw TensorError("Nexus lm::linear_t: x [N, in], w [out, in] bekler");
  }
  const int n = static_cast<int>(x.shape().dims[0]);
  const int in = static_cast<int>(w.shape().dims[1]);
  const int out = static_cast<int>(w.shape().dims[0]);
  std::vector<float> y(static_cast<std::size_t>(n) * out);
  nexus::cpu::GemmArgs g;
  g.M = n;
  g.N = out;
  g.K = in;
  g.A = x.values().data();
  g.lda = in;
  g.B = w.values().data();
  g.ldb = in;
  g.b_trans = true;
  g.C = y.data();
  g.ldc = out;
  nexus::cpu::gemm(g);
  if (!need_grad_any({&x, &w})) {
    return Variable::make(std::move(y), Shape{{n, out}}, false);
  }
  const ag::SavedTensor xs = ag::save_variable(x);
  const ag::SavedTensor ws = ag::save_variable(w);
  return ag::custom_function("lm_linear_t", {x, w}, std::move(y), Shape{{n, out}}, [xs, ws, n, in, out](const Variable& grad) {
    const Variable xv = ag::restore_variable(xs, "lm_linear_t");
    const Variable wv = ag::restore_variable(ws, "lm_linear_t");
    const float* dy = grad.values().data();
    std::vector<float> dx(static_cast<std::size_t>(n) * in);
    std::vector<float> dw(static_cast<std::size_t>(out) * in);
    nexus::cpu::gemm_nn(dy, wv.values().data(), dx.data(), n, out, in);  // dx = dy W
    {
      nexus::cpu::GemmArgs gw;
      gw.M = out;
      gw.N = in;
      gw.K = n;
      gw.A = dy;
      gw.lda = out;
      gw.a_trans = true;
      gw.B = xv.values().data();
      gw.ldb = in;
      gw.C = dw.data();
      gw.ldc = in;
      nexus::cpu::gemm(gw);  // dW = dy^T x
    }
    return std::vector<Variable>{Variable::make(std::move(dx), xv.shape(), false),
                                 Variable::make(std::move(dw), wv.shape(), false)};
  });
}

namespace {

constexpr float kInvSqrt2 = 0.70710678118654752f;
constexpr float kInvSqrt2Pi = 0.39894228040143268f;

// act(a) ve act'(a). SiLU: a s(a); GELU (erf, ag::gelu ile ayni).
template <bool Silu>
void act(float a, float* f, float* df) {
  if constexpr (Silu) {
    const float sg = 1.0f / (1.0f + std::exp(-a));
    *f = a * sg;
    *df = sg * (1.0f + a * (1.0f - sg));
  } else {
    const float cdf = 0.5f * (1.0f + std::erf(a * kInvSqrt2));
    *f = a * cdf;
    *df = cdf + a * kInvSqrt2Pi * std::exp(-0.5f * a * a);
  }
}

template <bool Silu>
void gated_rows_fwd(const float* xv, float* out, std::int64_t r0, std::int64_t r1, std::int64_t f) {
  for (std::int64_t r = r0; r < r1; ++r) {
    const float* a = xv + r * 2 * f;
    const float* b = a + f;
    float* y = out + r * f;
    for (std::int64_t i = 0; i < f; ++i) {
      float fa = 0.0f;
      float dfa = 0.0f;
      act<Silu>(a[i], &fa, &dfa);
      y[i] = fa * b[i];
    }
  }
}

template <bool Silu>
void gated_rows_bwd(const float* xv, const float* gv, float* dx, std::int64_t r0, std::int64_t r1, std::int64_t f) {
  for (std::int64_t r = r0; r < r1; ++r) {
    const float* a = xv + r * 2 * f;
    const float* b = a + f;
    const float* gy = gv + r * f;
    float* da = dx + r * 2 * f;
    float* db = da + f;
    for (std::int64_t i = 0; i < f; ++i) {
      float fa = 0.0f;
      float dfa = 0.0f;
      act<Silu>(a[i], &fa, &dfa);
      da[i] = gy[i] * b[i] * dfa;
      db[i] = gy[i] * fa;
    }
  }
}

template <bool Silu>
Variable gated(const Variable& x, const char* name) {
  const Shape& s = x.shape();
  if (s.rank() != 2 || s.dims[1] % 2 != 0) {
    throw TensorError(std::string("Nexus lm::") + name + ": x [N, 2F] bekler");
  }
  const std::int64_t n = s.dims[0];
  const std::int64_t f = s.dims[1] / 2;
  const float* xv = x.values().data();
  std::vector<float> out(static_cast<std::size_t>(n * f));
  if (n * f >= 8192 && n >= 2) {
    cpu::Pool::instance().run(static_cast<std::size_t>(n), [&](int, std::size_t r0, std::size_t r1) {
      gated_rows_fwd<Silu>(xv, out.data(), static_cast<std::int64_t>(r0), static_cast<std::int64_t>(r1), f);
    });
  } else {
    gated_rows_fwd<Silu>(xv, out.data(), 0, n, f);
  }
  if (!need_grad_any({&x})) {
    return Variable::make(std::move(out), Shape{{n, f}}, false);
  }
  const ag::SavedTensor saved = ag::save_variable(x);
  return ag::custom_function(name, {x}, std::move(out), Shape{{n, f}}, [saved, n, f, s, name](const Variable& g) {
    const Variable xr = ag::restore_variable(saved, name);
    const float* xv2 = xr.values().data();
    const float* gv = g.values().data();
    std::vector<float> dx(static_cast<std::size_t>(n * 2 * f));
    if (n * f >= 8192 && n >= 2) {
      cpu::Pool::instance().run(static_cast<std::size_t>(n), [&](int, std::size_t r0, std::size_t r1) {
        gated_rows_bwd<Silu>(xv2, gv, dx.data(), static_cast<std::int64_t>(r0), static_cast<std::int64_t>(r1), f);
      });
    } else {
      gated_rows_bwd<Silu>(xv2, gv, dx.data(), 0, n, f);
    }
    return std::vector<Variable>{Variable::make(std::move(dx), s, false)};
  });
}

}  // namespace

Variable swiglu(const Variable& x) { return gated<true>(x, "lm_swiglu"); }
Variable geglu(const Variable& x) { return gated<false>(x, "lm_geglu"); }

Variable attention(const Variable& q, const Variable& k, const Variable& v, const AttentionSpec& a) {
  const std::int64_t rows = static_cast<std::int64_t>(a.batch) * a.time;
  const int tk = a.kv_time > 0 ? a.kv_time : a.time;
  const std::int64_t krows = static_cast<std::int64_t>(a.batch) * tk;
  if (q.shape().rank() != 2 || q.shape().dims[0] != rows ||
      q.shape().dims[1] != static_cast<std::int64_t>(a.heads) * a.head_dim ||
      k.shape().dims != std::vector<std::int64_t>{krows, static_cast<std::int64_t>(a.kv_heads) * a.head_dim} ||
      v.shape().dims != k.shape().dims || tk < a.time) {
    throw TensorError("Nexus lm::attention: q [B*T, H*D], k/v [B*Tk, Hkv*D] (Tk >= T) bekler");
  }
  AttentionParams p;
  p.batch = a.batch;
  p.heads = a.heads;
  p.kv_heads = a.kv_heads;
  p.tq = a.time;
  p.tk = tk;
  p.head_dim = a.head_dim;
  p.dtype = DType::F32;
  p.scale = a.scale;
  p.causal = a.causal;
  p.window = a.window;
  p.softcap = a.softcap;
  p.dropout_p = a.dropout_p;
  p.seed = a.seed;
  p.offset = a.offset;
  p.alibi_slopes = a.alibi_slopes ? a.alibi_slopes->data() : nullptr;
  p.set_contiguous_bthd();
  if (!need_grad_any({&q, &k, &v})) {
    std::vector<float> o(static_cast<std::size_t>(q.numel()));
    std::vector<float> lse(static_cast<std::size_t>(rows) * a.heads);
    attention::flash_attention_forward_cpu(o.data(), lse.data(), q.values().data(), k.values().data(),
                                           v.values().data(), p);
    return Variable::make(std::move(o), q.shape(), false);
  }
  auto o = std::make_shared<std::vector<float>>(static_cast<std::size_t>(q.numel()));
  auto lse = std::make_shared<std::vector<float>>(static_cast<std::size_t>(rows) * a.heads);
  attention::flash_attention_forward_cpu(o->data(), lse->data(), q.values().data(), k.values().data(),
                                         v.values().data(), p);
  const ag::SavedTensor qs = ag::save_variable(q);
  const ag::SavedTensor ks = ag::save_variable(k);
  const ag::SavedTensor vs = ag::save_variable(v);
  const ag::SavedTensor os = ag::save_variable(Variable::make(*o, q.shape(), false));
  const ag::SavedTensor ls = ag::save_variable(Variable::make(*lse, Shape{{rows * a.heads}}, false));
  auto slopes = a.alibi_slopes;  // isaretci p icinde: omru kapanista tutulur
  return ag::custom_function("lm_flash_attention", {q, k, v}, *o, q.shape(),
                             [p, qs, ks, vs, os, ls, slopes](const Variable& g) {
                               (void)slopes;
                               const Variable qv = ag::restore_variable(qs, "lm_flash_attention");
                               const Variable kv = ag::restore_variable(ks, "lm_flash_attention");
                               const Variable vv = ag::restore_variable(vs, "lm_flash_attention");
                               const Variable ov = ag::restore_variable(os, "lm_flash_attention");
                               const Variable lv = ag::restore_variable(ls, "lm_flash_attention");
                               std::vector<float> dq(qv.values().size()), dk(kv.values().size()), dv(vv.values().size());
                               attention::flash_attention_backward_cpu(dq.data(), dk.data(), dv.data(),
                                                                       g.values().data(), qv.values().data(),
                                                                       kv.values().data(), vv.values().data(),
                                                                       ov.values().data(), lv.values().data(), p);
                               return std::vector<Variable>{Variable::make(std::move(dq), qv.shape(), false),
                                                            Variable::make(std::move(dk), kv.shape(), false),
                                                            Variable::make(std::move(dv), vv.shape(), false)};
                             });
}

Variable rms_norm_heads(const Variable& x, const Variable& gamma, int heads, int head_dim, float eps) {
  const std::int64_t n = x.shape().dims[0];
  const Variable flat = ag::reshape(x, Shape{{n * heads, head_dim}});
  return ag::reshape(fused::rms_norm(flat, gamma, eps), x.shape());
}

std::vector<float> alibi_slopes(int heads) {
  int closest = 1;
  while (closest * 2 <= heads) {
    closest *= 2;
  }
  std::vector<float> out;
  const double base = std::pow(2.0, -8.0 / closest);
  for (int i = 1; i <= closest; ++i) {
    out.push_back(static_cast<float>(std::pow(base, i)));
  }
  if (closest != heads) {
    const double extra = std::pow(2.0, -4.0 / closest);
    for (int i = 0; i < heads - closest; ++i) {
      out.push_back(static_cast<float>(std::pow(extra, 2 * i + 1)));
    }
  }
  return out;
}

}  // namespace lm
}  // namespace nexus
