/**
 * @file fused.cpp
 * @brief FAZ 3 füzyonlu çekirdekler: ileri+geri tek dugumde.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/fused/fused.hpp"

#include <cmath>
#include <utility>

namespace nexus {
namespace fused {
namespace {

using autograd::AutogradMeta;
using autograd::Edge;
using autograd::Node;
using autograd::Shape;
using autograd::Variable;
using autograd::grad_enabled;

class FusedNode : public Node {
 public:
  std::function<std::vector<Variable>(std::vector<Variable>&&)> backward_fn;
  std::vector<Variable> apply(std::vector<Variable>&& grads) override { return backward_fn(std::move(grads)); }
};

[[nodiscard]] Edge edge_of(const Variable& v) { return Edge{v.grad_fn(), 0}; }

[[nodiscard]] Variable emit(std::vector<float> values, const Shape& shape, std::shared_ptr<FusedNode> node,
                            std::vector<Edge> edges) {
  if (!node) {
    return Variable::make(std::move(values), shape, false);
  }
  return autograd::make_op_output(std::move(values), shape, std::move(node), std::move(edges));
}

constexpr float kInvSqrt2 = 0.7071067811865475f;
constexpr float kInvSqrt2Pi = 0.3989422804014327f;

[[nodiscard]] float apply_act(Activation act, float pre) {
  switch (act) {
    case Activation::Identity:
      return pre;
    case Activation::Relu:
      return pre > 0.0f ? pre : 0.0f;
    case Activation::Gelu:
      return 0.5f * pre * (1.0f + std::erf(pre * kInvSqrt2));
    case Activation::Silu:
      return pre / (1.0f + std::exp(-pre));
  }
  return pre;
}

[[nodiscard]] float deriv_act(Activation act, float pre) {
  switch (act) {
    case Activation::Identity:
      return 1.0f;
    case Activation::Relu:
      return pre > 0.0f ? 1.0f : 0.0f;
    case Activation::Gelu:
      return 0.5f * (1.0f + std::erf(pre * kInvSqrt2)) +
             pre * kInvSqrt2Pi * std::exp(-0.5f * pre * pre);
    case Activation::Silu: {
      const float sig = 1.0f / (1.0f + std::exp(-pre));
      return sig * (1.0f + pre * (1.0f - sig));
    }
  }
  return 1.0f;
}

}  // namespace

// ---- fused_linear: y = act(x @ w + b) --------------------------------------------------
Variable fused_linear(const Variable& x, const Variable& w, const Variable& b, Activation act) {
  if (x.shape().rank() != 2 || w.shape().rank() != 2 || b.shape().rank() != 1) {
    throw std::invalid_argument("Nexus fused: fused_linear x[2D] w[2D] b[1D] ister");
  }
  const std::int64_t m = x.shape().dims[0];
  const std::int64_t k = x.shape().dims[1];
  const std::int64_t k2 = w.shape().dims[0];
  const std::int64_t n = w.shape().dims[1];
  if (k != k2 || b.numel() != n) {
    throw std::invalid_argument("Nexus fused: fused_linear boyut uyusmazligi");
  }
  const auto& xv = x.values();
  const auto& wv = w.values();
  const auto& bv = b.values();
  std::vector<float> pre(static_cast<std::size_t>(m * n));
  std::vector<float> out(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float acc = bv[static_cast<std::size_t>(j)];
      for (std::int64_t t = 0; t < k; ++t) {
        acc += xv[static_cast<std::size_t>(i * k + t)] * wv[static_cast<std::size_t>(t * n + j)];
      }
      pre[static_cast<std::size_t>(i * n + j)] = acc;
      out[static_cast<std::size_t>(i * n + j)] = apply_act(act, acc);
    }
  }
  const Shape shape{{m, n}};
  const bool needs = grad_enabled() && (x.requires_grad() || w.requires_grad() || b.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<FusedNode>();
  node->name = "FusedLinearBackward";
  const bool x_req = x.requires_grad();
  const bool w_req = w.requires_grad();
  const bool b_req = b.requires_grad();
  std::vector<Edge> edges;
  if (x_req) {
    edges.push_back(edge_of(x));
  }
  if (w_req) {
    edges.push_back(edge_of(w));
  }
  if (b_req) {
    edges.push_back(edge_of(b));
  }
  autograd::SavedTensor saved_x = autograd::save_variable(x);
  autograd::SavedTensor saved_w = autograd::save_variable(w);
  node->backward_fn = [x_req, w_req, b_req, saved_x, saved_w, pre, m, k, n, act](std::vector<Variable>&& grads) {
    const Variable& x_in = autograd::restore_variable(saved_x, "fused_linear");
    const Variable& w_in = autograd::restore_variable(saved_w, "fused_linear");
    const auto& gy = grads[0].values();
    std::vector<float> dact(static_cast<std::size_t>(m * n));
    for (std::int64_t i = 0; i < m * n; ++i) {
      dact[static_cast<std::size_t>(i)] = gy[static_cast<std::size_t>(i)] * deriv_act(act, pre[static_cast<std::size_t>(i)]);
    }
    std::vector<Variable> result;
    if (x_req) {
      std::vector<float> dx(static_cast<std::size_t>(m * k), 0.0f);
      for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t t = 0; t < k; ++t) {
          float acc = 0.0f;
          for (std::int64_t j = 0; j < n; ++j) {
            acc += dact[static_cast<std::size_t>(i * n + j)] * w_in.values()[static_cast<std::size_t>(t * n + j)];
          }
          dx[static_cast<std::size_t>(i * k + t)] = acc;
        }
      }
      result.push_back(Variable::make(std::move(dx), Shape{{m, k}}, false));
    }
    if (w_req) {
      std::vector<float> dw(static_cast<std::size_t>(k * n), 0.0f);
      for (std::int64_t t = 0; t < k; ++t) {
        for (std::int64_t j = 0; j < n; ++j) {
          float acc = 0.0f;
          for (std::int64_t i = 0; i < m; ++i) {
            acc += x_in.values()[static_cast<std::size_t>(i * k + t)] * dact[static_cast<std::size_t>(i * n + j)];
          }
          dw[static_cast<std::size_t>(t * n + j)] = acc;
        }
      }
      result.push_back(Variable::make(std::move(dw), Shape{{k, n}}, false));
    }
    if (b_req) {
      std::vector<float> db(static_cast<std::size_t>(n), 0.0f);
      for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
          db[static_cast<std::size_t>(j)] += dact[static_cast<std::size_t>(i * n + j)];
        }
      }
      result.push_back(Variable::make(std::move(db), Shape{{n}}, false));
    }
    return result;
  };
  return emit(std::move(out), shape, std::move(node), std::move(edges));
}

// ---- fused_layernorm ------------------------------------------------------------------------
Variable fused_layernorm(const Variable& x, const Variable& gamma, const Variable& beta, float eps) {
  if (x.shape().rank() != 2 || gamma.shape().rank() != 1 || beta.shape().rank() != 1) {
    throw std::invalid_argument("Nexus fused: fused_layernorm x[2D] gamma/beta[1D] ister");
  }
  const std::int64_t m = x.shape().dims[0];
  const std::int64_t n = x.shape().dims[1];
  if (gamma.numel() != n || beta.numel() != n) {
    throw std::invalid_argument("Nexus fused: fused_layernorm boyut uyusmazligi");
  }
  const auto& xv = x.values();
  const auto& gv = gamma.values();
  const auto& bv = beta.values();
  std::vector<float> mean(static_cast<std::size_t>(m));
  std::vector<float> rstd(static_cast<std::size_t>(m));
  std::vector<float> out(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    float mu = 0.0f;
    for (std::int64_t j = 0; j < n; ++j) {
      mu += xv[static_cast<std::size_t>(i * n + j)];
    }
    mu /= static_cast<float>(n);
    float var = 0.0f;
    for (std::int64_t j = 0; j < n; ++j) {
      const float d = xv[static_cast<std::size_t>(i * n + j)] - mu;
      var += d * d;
    }
    var /= static_cast<float>(n);
    const float r = 1.0f / std::sqrt(var + eps);
    mean[static_cast<std::size_t>(i)] = mu;
    rstd[static_cast<std::size_t>(i)] = r;
    for (std::int64_t j = 0; j < n; ++j) {
      out[static_cast<std::size_t>(i * n + j)] =
          (xv[static_cast<std::size_t>(i * n + j)] - mu) * r * gv[static_cast<std::size_t>(j)] +
          bv[static_cast<std::size_t>(j)];
    }
  }
  const Shape shape{{m, n}};
  const bool needs = grad_enabled() && (x.requires_grad() || gamma.requires_grad() || beta.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<FusedNode>();
  node->name = "FusedLayerNormBackward";
  const bool x_req = x.requires_grad();
  const bool g_req = gamma.requires_grad();
  const bool b_req = beta.requires_grad();
  std::vector<Edge> edges;
  if (x_req) {
    edges.push_back(edge_of(x));
  }
  if (g_req) {
    edges.push_back(edge_of(gamma));
  }
  if (b_req) {
    edges.push_back(edge_of(beta));
  }
  autograd::SavedTensor saved_x = autograd::save_variable(x);
  autograd::SavedTensor saved_gamma = autograd::save_variable(gamma);
  node->backward_fn = [x_req, g_req, b_req, saved_x, saved_gamma, mean, rstd, m, n](std::vector<Variable>&& grads) {
    const Variable& x_in = autograd::restore_variable(saved_x, "fused_layernorm");
    const Variable& gamma_in = autograd::restore_variable(saved_gamma, "fused_layernorm");
    const auto& gy = grads[0].values();
    const auto& xv2 = x_in.values();
    const auto& gv2 = gamma_in.values();
    std::vector<float> dxhat(static_cast<std::size_t>(m * n));
    for (std::int64_t i = 0; i < m * n; ++i) {
      dxhat[static_cast<std::size_t>(i)] = gy[static_cast<std::size_t>(i)] * gv2[static_cast<std::size_t>(i % n)];
    }
    std::vector<Variable> result;
    if (x_req) {
      std::vector<float> dx(static_cast<std::size_t>(m * n));
      for (std::int64_t i = 0; i < m; ++i) {
        const float mu = mean[static_cast<std::size_t>(i)];
        const float r = rstd[static_cast<std::size_t>(i)];
        float s1 = 0.0f;
        float s2 = 0.0f;
        for (std::int64_t j = 0; j < n; ++j) {
          const float d = xv2[static_cast<std::size_t>(i * n + j)] - mu;
          const float dh = dxhat[static_cast<std::size_t>(i * n + j)];
          s1 += dh;
          s2 += dh * d;
        }
        const float grad_mean = -r * s1;
        const float grad_var = -0.5f * r * r * r * s2;
        const float inv_n = 1.0f / static_cast<float>(n);
        for (std::int64_t j = 0; j < n; ++j) {
          const float d = xv2[static_cast<std::size_t>(i * n + j)] - mu;
          dx[static_cast<std::size_t>(i * n + j)] =
              dxhat[static_cast<std::size_t>(i * n + j)] * r + grad_var * 2.0f * d * inv_n + grad_mean * inv_n;
        }
      }
      result.push_back(Variable::make(std::move(dx), Shape{{m, n}}, false));
    }
    if (g_req) {
      std::vector<float> dgamma(static_cast<std::size_t>(n), 0.0f);
      for (std::int64_t i = 0; i < m; ++i) {
        const float mu = mean[static_cast<std::size_t>(i)];
        const float r = rstd[static_cast<std::size_t>(i)];
        for (std::int64_t j = 0; j < n; ++j) {
          const float xhat = (xv2[static_cast<std::size_t>(i * n + j)] - mu) * r;
          dgamma[static_cast<std::size_t>(j)] += gy[static_cast<std::size_t>(i * n + j)] * xhat;
        }
      }
      result.push_back(Variable::make(std::move(dgamma), Shape{{n}}, false));
    }
    if (b_req) {
      std::vector<float> dbeta(static_cast<std::size_t>(n), 0.0f);
      for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
          dbeta[static_cast<std::size_t>(j)] += gy[static_cast<std::size_t>(i * n + j)];
        }
      }
      result.push_back(Variable::make(std::move(dbeta), Shape{{n}}, false));
    }
    return result;
  };
  return emit(std::move(out), shape, std::move(node), std::move(edges));
}

// ---- fused_residual_layernorm: y = layernorm(x + residual) -----------------------------------
Variable fused_residual_layernorm(const Variable& x, const Variable& residual, const Variable& gamma,
                                  const Variable& beta, float eps) {
  if (x.shape().dims != residual.shape().dims) {
    throw std::invalid_argument("Nexus fused: fused_residual_layernorm sekil uyusmazligi");
  }
  if (x.shape().rank() != 2 || gamma.shape().rank() != 1 || beta.shape().rank() != 1) {
    throw std::invalid_argument("Nexus fused: fused_residual_layernorm x[2D] gamma/beta[1D] ister");
  }
  const std::int64_t m = x.shape().dims[0];
  const std::int64_t n = x.shape().dims[1];
  const auto& xv = x.values();
  const auto& rv = residual.values();
  const auto& gv = gamma.values();
  const auto& bv = beta.values();
  std::vector<float> t(static_cast<std::size_t>(m * n));  // x + residual (tek tampon)
  for (std::size_t i = 0; i < t.size(); ++i) {
    t[i] = xv[i] + rv[i];
  }
  std::vector<float> mean(static_cast<std::size_t>(m));
  std::vector<float> rstd(static_cast<std::size_t>(m));
  std::vector<float> out(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    float mu = 0.0f;
    for (std::int64_t j = 0; j < n; ++j) {
      mu += t[static_cast<std::size_t>(i * n + j)];
    }
    mu /= static_cast<float>(n);
    float var = 0.0f;
    for (std::int64_t j = 0; j < n; ++j) {
      const float d = t[static_cast<std::size_t>(i * n + j)] - mu;
      var += d * d;
    }
    var /= static_cast<float>(n);
    const float r = 1.0f / std::sqrt(var + eps);
    mean[static_cast<std::size_t>(i)] = mu;
    rstd[static_cast<std::size_t>(i)] = r;
    for (std::int64_t j = 0; j < n; ++j) {
      out[static_cast<std::size_t>(i * n + j)] =
          (t[static_cast<std::size_t>(i * n + j)] - mu) * r * gv[static_cast<std::size_t>(j)] +
          bv[static_cast<std::size_t>(j)];
    }
  }
  const Shape shape{{m, n}};
  const bool needs = grad_enabled() && (x.requires_grad() || residual.requires_grad() || gamma.requires_grad() ||
                                        beta.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<FusedNode>();
  node->name = "FusedResidualLayerNormBackward";
  const bool x_req = x.requires_grad();
  const bool res_req = residual.requires_grad();
  const bool g_req = gamma.requires_grad();
  const bool b_req = beta.requires_grad();
  std::vector<Edge> edges;
  if (x_req) {
    edges.push_back(edge_of(x));
  }
  if (res_req) {
    edges.push_back(edge_of(residual));
  }
  if (g_req) {
    edges.push_back(edge_of(gamma));
  }
  if (b_req) {
    edges.push_back(edge_of(beta));
  }
  node->backward_fn = [x_req, res_req, g_req, b_req, t, mean, rstd, m, n, gv](std::vector<Variable>&& grads) {
    const auto& gy = grads[0].values();
    std::vector<float> dxhat(static_cast<std::size_t>(m * n));
    for (std::int64_t i = 0; i < m * n; ++i) {
      dxhat[static_cast<std::size_t>(i)] = gy[static_cast<std::size_t>(i)] * gv[static_cast<std::size_t>(i % n)];
    }
    std::vector<float> grad_t(static_cast<std::size_t>(m * n));
    for (std::int64_t i = 0; i < m; ++i) {
      const float mu = mean[static_cast<std::size_t>(i)];
      const float r = rstd[static_cast<std::size_t>(i)];
      float s1 = 0.0f;
      float s2 = 0.0f;
      for (std::int64_t j = 0; j < n; ++j) {
        const float d = t[static_cast<std::size_t>(i * n + j)] - mu;
        const float dh = dxhat[static_cast<std::size_t>(i * n + j)];
        s1 += dh;
        s2 += dh * d;
      }
      const float grad_mean = -r * s1;
      const float grad_var = -0.5f * r * r * r * s2;
      const float inv_n = 1.0f / static_cast<float>(n);
      for (std::int64_t j = 0; j < n; ++j) {
        const float d = t[static_cast<std::size_t>(i * n + j)] - mu;
        grad_t[static_cast<std::size_t>(i * n + j)] =
            dxhat[static_cast<std::size_t>(i * n + j)] * r + grad_var * 2.0f * d * inv_n + grad_mean * inv_n;
      }
    }
    std::vector<Variable> result;
    if (x_req) {
      result.push_back(Variable::make(std::vector<float>(grad_t), Shape{{m, n}}, false));
    }
    if (res_req) {
      result.push_back(Variable::make(std::vector<float>(grad_t), Shape{{m, n}}, false));
    }
    if (g_req) {
      std::vector<float> dgamma(static_cast<std::size_t>(n), 0.0f);
      for (std::int64_t i = 0; i < m; ++i) {
        const float mu = mean[static_cast<std::size_t>(i)];
        const float r = rstd[static_cast<std::size_t>(i)];
        for (std::int64_t j = 0; j < n; ++j) {
          const float xhat = (t[static_cast<std::size_t>(i * n + j)] - mu) * r;
          dgamma[static_cast<std::size_t>(j)] += gy[static_cast<std::size_t>(i * n + j)] * xhat;
        }
      }
      result.push_back(Variable::make(std::move(dgamma), Shape{{n}}, false));
    }
    if (b_req) {
      std::vector<float> dbeta(static_cast<std::size_t>(n), 0.0f);
      for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
          dbeta[static_cast<std::size_t>(j)] += gy[static_cast<std::size_t>(i * n + j)];
        }
      }
      result.push_back(Variable::make(std::move(dbeta), Shape{{n}}, false));
    }
    return result;
  };
  return emit(std::move(out), shape, std::move(node), std::move(edges));
}

// ---- fused_softmax: cevrimici (online) kararli softmax ------------------------------------------
Variable fused_softmax(const Variable& x) {
  if (x.shape().rank() != 2) {
    throw std::invalid_argument("Nexus fused: fused_softmax yalnizca 2D");
  }
  const std::int64_t m = x.shape().dims[0];
  const std::int64_t n = x.shape().dims[1];
  const auto& xv = x.values();
  std::vector<float> probs(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    // Tek geciste calisan max + olcekli toplam (online softmax).
    float running_max = xv[static_cast<std::size_t>(i * n)];
    float running_sum = 1.0f;  // exp(x[0] - max) = 1
    for (std::int64_t j = 1; j < n; ++j) {
      const float v = xv[static_cast<std::size_t>(i * n + j)];
      if (v > running_max) {
        running_sum = running_sum * std::exp(running_max - v) + 1.0f;
        running_max = v;
      } else {
        running_sum += std::exp(v - running_max);
      }
    }
    for (std::int64_t j = 0; j < n; ++j) {
      probs[static_cast<std::size_t>(i * n + j)] =
          std::exp(xv[static_cast<std::size_t>(i * n + j)] - running_max) / running_sum;
    }
  }
  const Shape shape{{m, n}};
  const bool needs = grad_enabled() && x.requires_grad();
  if (!needs) {
    return Variable::make(std::move(probs), shape, false);
  }
  auto node = std::make_shared<FusedNode>();
  node->name = "FusedSoftmaxBackward";
  node->backward_fn = [probs, m, n](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(static_cast<std::size_t>(m * n));
    for (std::int64_t i = 0; i < m; ++i) {
      float dot = 0.0f;
      for (std::int64_t j = 0; j < n; ++j) {
        dot += g[static_cast<std::size_t>(i * n + j)] * probs[static_cast<std::size_t>(i * n + j)];
      }
      for (std::int64_t j = 0; j < n; ++j) {
        d[static_cast<std::size_t>(i * n + j)] =
            probs[static_cast<std::size_t>(i * n + j)] * (g[static_cast<std::size_t>(i * n + j)] - dot);
      }
    }
    return std::vector<Variable>{Variable::make(std::move(d), Shape{{m, n}}, false)};
  };
  return emit(std::move(probs), shape, std::move(node), {edge_of(x)});
}

}  // namespace fused
}  // namespace nexus
