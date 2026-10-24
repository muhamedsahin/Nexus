/**
 * @file ops.cpp
 * @brief FAZ 6.A/6.B: RoPE, SwiGLU/GeGLU, GQA flash dikkat ve qk_norm autograd op'lari.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/ops.hpp"

#include "nexus/attention/cpu.hpp"
#include "nexus/core/error.hpp"
#include "nexus/fused/training.hpp"
#include "nexus_model/core/kernels.hpp"

#include <cmath>
#include <utility>

namespace nexus {
namespace lm {

namespace ag = autograd;
using ag::Shape;
using ag::Variable;

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
  nexus_model::kernels::gemm_nn(x.values().data(), w.values().data(), y.data(), n, in, out);
  if (has_b) {
    const float* bv = b.values().data();
    for (int r = 0; r < n; ++r) {
      float* yr = y.data() + static_cast<std::size_t>(r) * out;
      for (int j = 0; j < out; ++j) {
        yr[j] += bv[j];
      }
    }
  }
  const Variable xs = x.detach();
  const Variable ws = w.detach();
  std::vector<Variable> inputs{x, w};
  if (has_b) {
    inputs.push_back(b);
  }
  return ag::custom_function("lm_linear", inputs, std::move(y), Shape{{n, out}}, [xs, ws, n, in, out, has_b](const Variable& g) {
    const float* dy = g.values().data();
    std::vector<float> dx(static_cast<std::size_t>(n) * in, 0.0f);
    std::vector<float> dw(static_cast<std::size_t>(in) * out, 0.0f);
    nexus_model::kernels::gemm_nt_accum(dy, ws.values().data(), dx.data(), n, out, in);  // dx = dy W^T
    nexus_model::kernels::gemm_tn_accum(xs.values().data(), dy, dw.data(), in, n, out);  // dW = x^T dy
    std::vector<Variable> grads{Variable::make(std::move(dx), xs.shape(), false),
                                Variable::make(std::move(dw), ws.shape(), false)};
    if (has_b) {
      std::vector<float> db(static_cast<std::size_t>(out), 0.0f);
      for (int r = 0; r < n; ++r) {
        for (int j = 0; j < out; ++j) {
          db[static_cast<std::size_t>(j)] += dy[static_cast<std::size_t>(r) * out + j];
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
  std::vector<float> y(static_cast<std::size_t>(n) * out, 0.0f);
  nexus_model::kernels::gemm_nt_accum(x.values().data(), w.values().data(), y.data(), n, in, out);
  const Variable xs = x.detach();
  const Variable ws = w.detach();
  return ag::custom_function("lm_linear_t", {x, w}, std::move(y), Shape{{n, out}}, [xs, ws, n, in, out](const Variable& g) {
    const float* dy = g.values().data();
    std::vector<float> dx(static_cast<std::size_t>(n) * in);
    std::vector<float> dw(static_cast<std::size_t>(out) * in, 0.0f);
    nexus_model::kernels::gemm_nn(dy, ws.values().data(), dx.data(), n, out, in);        // dx = dy W
    nexus_model::kernels::gemm_tn_accum(dy, xs.values().data(), dw.data(), out, n, in);  // dW = dy^T x
    return std::vector<Variable>{Variable::make(std::move(dx), xs.shape(), false),
                                 Variable::make(std::move(dw), ws.shape(), false)};
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
Variable gated(const Variable& x, const char* name) {
  const Shape& s = x.shape();
  if (s.rank() != 2 || s.dims[1] % 2 != 0) {
    throw TensorError(std::string("Nexus lm::") + name + ": x [N, 2F] bekler");
  }
  const std::int64_t n = s.dims[0];
  const std::int64_t f = s.dims[1] / 2;
  const std::vector<float>& xv = x.values();
  std::vector<float> out(static_cast<std::size_t>(n * f));
  for (std::int64_t r = 0; r < n; ++r) {
    const float* a = xv.data() + r * 2 * f;
    const float* b = a + f;
    float* y = out.data() + r * f;
    for (std::int64_t i = 0; i < f; ++i) {
      float fa = 0.0f;
      float dfa = 0.0f;
      act<Silu>(a[i], &fa, &dfa);
      y[i] = fa * b[i];
    }
  }
  const Variable saved = x.detach();
  return ag::custom_function(name, {x}, std::move(out), Shape{{n, f}}, [saved, n, f, s](const Variable& g) {
    const std::vector<float>& xv2 = saved.values();
    const std::vector<float>& gv = g.values();
    std::vector<float> dx(static_cast<std::size_t>(n * 2 * f));
    for (std::int64_t r = 0; r < n; ++r) {
      const float* a = xv2.data() + r * 2 * f;
      const float* b = a + f;
      const float* gy = gv.data() + r * f;
      float* da = dx.data() + r * 2 * f;
      float* db = da + f;
      for (std::int64_t i = 0; i < f; ++i) {
        float fa = 0.0f;
        float dfa = 0.0f;
        act<Silu>(a[i], &fa, &dfa);
        da[i] = gy[i] * b[i] * dfa;
        db[i] = gy[i] * fa;
      }
    }
    return std::vector<Variable>{Variable::make(std::move(dx), s, false)};
  });
}

}  // namespace

Variable swiglu(const Variable& x) { return gated<true>(x, "lm_swiglu"); }
Variable geglu(const Variable& x) { return gated<false>(x, "lm_geglu"); }

Variable attention(const Variable& q, const Variable& k, const Variable& v, const AttentionSpec& a) {
  const std::int64_t rows = static_cast<std::int64_t>(a.batch) * a.time;
  if (q.shape().rank() != 2 || q.shape().dims[0] != rows ||
      q.shape().dims[1] != static_cast<std::int64_t>(a.heads) * a.head_dim ||
      k.shape().dims != std::vector<std::int64_t>{rows, static_cast<std::int64_t>(a.kv_heads) * a.head_dim} ||
      v.shape().dims != k.shape().dims) {
    throw TensorError("Nexus lm::attention: q [B*T, H*D], k/v [B*T, Hkv*D] bekler");
  }
  AttentionParams p;
  p.batch = a.batch;
  p.heads = a.heads;
  p.kv_heads = a.kv_heads;
  p.tq = a.time;
  p.tk = a.time;
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
  auto o = std::make_shared<std::vector<float>>(static_cast<std::size_t>(q.numel()));
  auto lse = std::make_shared<std::vector<float>>(static_cast<std::size_t>(rows) * a.heads);
  attention::flash_attention_forward_cpu(o->data(), lse->data(), q.values().data(), k.values().data(),
                                         v.values().data(), p);
  const Variable qs = q.detach();
  const Variable ks = k.detach();
  const Variable vs = v.detach();
  auto slopes = a.alibi_slopes;  // isaretci p icinde: omru kapanista tutulur
  return ag::custom_function("lm_flash_attention", {q, k, v}, *o, q.shape(),
                             [p, o, lse, qs, ks, vs, slopes](const Variable& g) {
                               (void)slopes;
                               std::vector<float> dq(qs.values().size()), dk(ks.values().size()), dv(vs.values().size());
                               attention::flash_attention_backward_cpu(dq.data(), dk.data(), dv.data(),
                                                                       g.values().data(), qs.values().data(),
                                                                       ks.values().data(), vs.values().data(),
                                                                       o->data(), lse->data(), p);
                               return std::vector<Variable>{Variable::make(std::move(dq), qs.shape(), false),
                                                            Variable::make(std::move(dk), ks.shape(), false),
                                                            Variable::make(std::move(dv), vs.shape(), false)};
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
