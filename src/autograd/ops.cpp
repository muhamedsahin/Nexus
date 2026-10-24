/**
 * @file ops.cpp
 * @brief FAZ 2 autograd op kumesi: ileri + geri formulleri.
 *
 * Her op ileri geçiste ciktiyi hesaplar; gradyan gerekliyse bir `LambdaNode`
 * kurar, girisleri `save_variable` ile kaydeder (sürüm denetimi) ve geri
 * formülünü bu dugume baglar. Türevler standarttir; sayisal gradyan testi
 * (test_phase2_autograd) bunlari sonlu farklarla dogrular.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"

#include <algorithm>
#include <cmath>

namespace nexus {
namespace autograd {
namespace {

class LambdaNode : public Node {
 public:
  std::function<std::vector<Variable>(std::vector<Variable>&&)> backward_fn;
  std::vector<Variable> apply(std::vector<Variable>&& grads) override { return backward_fn(std::move(grads)); }
};

[[nodiscard]] Edge edge_of(const Variable& v) { return Edge{v.grad_fn(), 0}; }

void require_same_shape(const Variable& a, const Variable& b, const char* op) {
  if (a.shape().dims != b.shape().dims) {
    throw std::invalid_argument(std::string("Nexus autograd: ") + op + " sekil uyusmazligi");
  }
}

// Grad gerekliyse dugumle, gerekmiyorsa dugumsuz cikti uretir.
[[nodiscard]] Variable emit(std::vector<float> values, const Shape& shape, std::shared_ptr<LambdaNode> node,
                            std::vector<Edge> edges) {
  if (!node) {
    return Variable::make(std::move(values), shape, false);
  }
  return make_op_output(std::move(values), shape, std::move(node), std::move(edges));
}

}  // namespace

// ---- add -------------------------------------------------------------------------------
Variable add(const Variable& a, const Variable& b) {
  require_same_shape(a, b, "add");
  const auto& av = a.values();
  const auto& bv = b.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = av[i] + bv[i];
  }
  const bool needs = grad_enabled() && (a.requires_grad() || b.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "AddBackward";
  std::vector<Edge> edges;
  const bool a_req = a.requires_grad();
  const bool b_req = b.requires_grad();
  if (a_req) {
    edges.push_back(edge_of(a));
  }
  if (b_req) {
    edges.push_back(edge_of(b));
  }
  node->backward_fn = [a_req, b_req](std::vector<Variable>&& grads) {
    std::vector<Variable> result;
    if (a_req) {
      result.push_back(grads[0]);
    }
    if (b_req) {
      result.push_back(grads[0]);
    }
    return result;
  };
  return emit(std::move(out), a.shape(), std::move(node), std::move(edges));
}

// ---- sub -------------------------------------------------------------------------------
Variable sub(const Variable& a, const Variable& b) {
  require_same_shape(a, b, "sub");
  const auto& av = a.values();
  const auto& bv = b.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = av[i] - bv[i];
  }
  const bool needs = grad_enabled() && (a.requires_grad() || b.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "SubBackward";
  const bool a_req = a.requires_grad();
  const bool b_req = b.requires_grad();
  std::vector<Edge> edges;
  if (a_req) {
    edges.push_back(edge_of(a));
  }
  if (b_req) {
    edges.push_back(edge_of(b));
  }
  node->backward_fn = [a_req, b_req](std::vector<Variable>&& grads) {
    std::vector<Variable> result;
    if (a_req) {
      result.push_back(grads[0]);
    }
    if (b_req) {
      const auto& g = grads[0].values();
      std::vector<float> ng(g.size());
      for (std::size_t i = 0; i < g.size(); ++i) {
        ng[i] = -g[i];
      }
      result.push_back(Variable::make(std::move(ng), grads[0].shape(), false));
    }
    return result;
  };
  return emit(std::move(out), a.shape(), std::move(node), std::move(edges));
}

// ---- mul -------------------------------------------------------------------------------
Variable mul(const Variable& a, const Variable& b) {
  require_same_shape(a, b, "mul");
  const auto& av = a.values();
  const auto& bv = b.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = av[i] * bv[i];
  }
  const bool needs = grad_enabled() && (a.requires_grad() || b.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "MulBackward";
  const bool a_req = a.requires_grad();
  const bool b_req = b.requires_grad();
  std::vector<Edge> edges;
  if (a_req) {
    edges.push_back(edge_of(a));
  }
  if (b_req) {
    edges.push_back(edge_of(b));
  }
  SavedTensor saved_a = save_variable(a);
  SavedTensor saved_b = save_variable(b);
  node->backward_fn = [a_req, b_req, saved_a, saved_b](std::vector<Variable>&& grads) {
    const Variable& a_in = restore_variable(saved_a, "mul");
    const Variable& b_in = restore_variable(saved_b, "mul");
    std::vector<Variable> result;
    const auto& g = grads[0].values();
    if (a_req) {
      std::vector<float> d(g.size());
      for (std::size_t i = 0; i < g.size(); ++i) {
        d[i] = g[i] * b_in.values()[i];
      }
      result.push_back(Variable::make(std::move(d), a_in.shape(), false));
    }
    if (b_req) {
      std::vector<float> d(g.size());
      for (std::size_t i = 0; i < g.size(); ++i) {
        d[i] = g[i] * a_in.values()[i];
      }
      result.push_back(Variable::make(std::move(d), b_in.shape(), false));
    }
    return result;
  };
  return emit(std::move(out), a.shape(), std::move(node), std::move(edges));
}

// ---- div -------------------------------------------------------------------------------
Variable div(const Variable& a, const Variable& b) {
  require_same_shape(a, b, "div");
  const auto& av = a.values();
  const auto& bv = b.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = av[i] / bv[i];
  }
  const bool needs = grad_enabled() && (a.requires_grad() || b.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "DivBackward";
  const bool a_req = a.requires_grad();
  const bool b_req = b.requires_grad();
  std::vector<Edge> edges;
  if (a_req) {
    edges.push_back(edge_of(a));
  }
  if (b_req) {
    edges.push_back(edge_of(b));
  }
  SavedTensor saved_a = save_variable(a);
  SavedTensor saved_b = save_variable(b);
  node->backward_fn = [a_req, b_req, saved_a, saved_b](std::vector<Variable>&& grads) {
    const Variable& a_in = restore_variable(saved_a, "div");
    const Variable& b_in = restore_variable(saved_b, "div");
    std::vector<Variable> result;
    const auto& g = grads[0].values();
    if (a_req) {
      std::vector<float> d(g.size());
      for (std::size_t i = 0; i < g.size(); ++i) {
        d[i] = g[i] / b_in.values()[i];
      }
      result.push_back(Variable::make(std::move(d), a_in.shape(), false));
    }
    if (b_req) {
      std::vector<float> d(g.size());
      for (std::size_t i = 0; i < g.size(); ++i) {
        d[i] = -g[i] * a_in.values()[i] / (b_in.values()[i] * b_in.values()[i]);
      }
      result.push_back(Variable::make(std::move(d), b_in.shape(), false));
    }
    return result;
  };
  return emit(std::move(out), a.shape(), std::move(node), std::move(edges));
}

// ---- neg -------------------------------------------------------------------------------
Variable neg(const Variable& a) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = -av[i];
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "NegBackward";
  node->backward_fn = [](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = -g[i];
    }
    return std::vector<Variable>{Variable::make(std::move(d), grads[0].shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- add_scalar --------------------------------------------------------------------------
Variable add_scalar(const Variable& a, float scalar) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = av[i] + scalar;
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "AddScalarBackward";
  node->backward_fn = [](std::vector<Variable>&& grads) { return std::vector<Variable>{grads[0]}; };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- mul_scalar --------------------------------------------------------------------------
Variable mul_scalar(const Variable& a, float scalar) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = av[i] * scalar;
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "MulScalarBackward";
  node->backward_fn = [scalar](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = g[i] * scalar;
    }
    return std::vector<Variable>{Variable::make(std::move(d), grads[0].shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- pow -----------------------------------------------------------------------------------
Variable pow(const Variable& a, float exponent) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = std::pow(av[i], exponent);
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "PowBackward";
  SavedTensor saved = save_variable(a);
  node->backward_fn = [saved, exponent](std::vector<Variable>&& grads) {
    const Variable& in = restore_variable(saved, "pow");
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = g[i] * exponent * std::pow(in.values()[i], exponent - 1.0f);
    }
    return std::vector<Variable>{Variable::make(std::move(d), in.shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- relu -----------------------------------------------------------------------------------
Variable relu(const Variable& a) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = av[i] > 0.0f ? av[i] : 0.0f;
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "ReluBackward";
  SavedTensor saved = save_variable(a);
  node->backward_fn = [saved](std::vector<Variable>&& grads) {
    const Variable& in = restore_variable(saved, "relu");
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = in.values()[i] > 0.0f ? g[i] : 0.0f;
    }
    return std::vector<Variable>{Variable::make(std::move(d), in.shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- sigmoid --------------------------------------------------------------------------------
Variable sigmoid(const Variable& a) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = 1.0f / (1.0f + std::exp(-av[i]));
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "SigmoidBackward";
  const auto out_copy = out;
  node->backward_fn = [out_copy](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = g[i] * out_copy[i] * (1.0f - out_copy[i]);
    }
    return std::vector<Variable>{Variable::make(std::move(d), grads[0].shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- tanh ------------------------------------------------------------------------------------
Variable tanh(const Variable& a) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = std::tanh(av[i]);
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "TanhBackward";
  const auto out_copy = out;
  node->backward_fn = [out_copy](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = g[i] * (1.0f - out_copy[i] * out_copy[i]);
    }
    return std::vector<Variable>{Variable::make(std::move(d), grads[0].shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- exp --------------------------------------------------------------------------------------
Variable exp(const Variable& a) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = std::exp(av[i]);
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "ExpBackward";
  const auto out_copy = out;
  node->backward_fn = [out_copy](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = g[i] * out_copy[i];
    }
    return std::vector<Variable>{Variable::make(std::move(d), grads[0].shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- log --------------------------------------------------------------------------------------
Variable log(const Variable& a) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = std::log(av[i]);
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "LogBackward";
  SavedTensor saved = save_variable(a);
  node->backward_fn = [saved](std::vector<Variable>&& grads) {
    const Variable& in = restore_variable(saved, "log");
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = g[i] / in.values()[i];
    }
    return std::vector<Variable>{Variable::make(std::move(d), in.shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- gelu (exact erf) -----------------------------------------------------------------------
Variable gelu(const Variable& a) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  constexpr float kInvSqrt2 = 0.7071067811865475f;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = 0.5f * av[i] * (1.0f + std::erf(av[i] * kInvSqrt2));
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "GeluBackward";
  SavedTensor saved = save_variable(a);
  node->backward_fn = [saved](std::vector<Variable>&& grads) {
    const Variable& in = restore_variable(saved, "gelu");
    const auto& g = grads[0].values();
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    constexpr float kInvSqrt2Pi = 0.3989422804014327f;
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      const float x = in.values()[i];
      const float cdf = 0.5f * (1.0f + std::erf(x * kInvSqrt2));
      const float gaussian = std::exp(-0.5f * x * x);
      d[i] = g[i] * (cdf + x * kInvSqrt2Pi * gaussian);
    }
    return std::vector<Variable>{Variable::make(std::move(d), in.shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- silu (swish) ---------------------------------------------------------------------------
Variable silu(const Variable& a) {
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = av[i] / (1.0f + std::exp(-av[i]));
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "SiluBackward";
  SavedTensor saved = save_variable(a);
  node->backward_fn = [saved](std::vector<Variable>&& grads) {
    const Variable& in = restore_variable(saved, "silu");
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      const float x = in.values()[i];
      const float sig = 1.0f / (1.0f + std::exp(-x));
      d[i] = g[i] * sig * (1.0f + x * (1.0f - sig));
    }
    return std::vector<Variable>{Variable::make(std::move(d), in.shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- sum ---------------------------------------------------------------------------------------
Variable sum(const Variable& a) {
  const auto& av = a.values();
  float total = 0.0f;
  for (const float value : av) {
    total += value;
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::vector<float>{total}, Shape{}, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "SumBackward";
  const Shape shape = a.shape();
  node->backward_fn = [shape](std::vector<Variable>&& grads) {
    return std::vector<Variable>{Variable::full(shape, grads[0].values()[0], false)};
  };
  return emit(std::vector<float>{total}, Shape{}, std::move(node), {edge_of(a)});
}

// ---- mean --------------------------------------------------------------------------------------
Variable mean(const Variable& a) {
  const auto& av = a.values();
  float total = 0.0f;
  for (const float value : av) {
    total += value;
  }
  const float result = total / static_cast<float>(av.size());
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::vector<float>{result}, Shape{}, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "MeanBackward";
  const Shape shape = a.shape();
  const float scale = 1.0f / static_cast<float>(av.size());
  node->backward_fn = [shape, scale](std::vector<Variable>&& grads) {
    return std::vector<Variable>{Variable::full(shape, grads[0].values()[0] * scale, false)};
  };
  return emit(std::vector<float>{result}, Shape{}, std::move(node), {edge_of(a)});
}

// ---- reshape ------------------------------------------------------------------------------------
Variable reshape(const Variable& a, Shape shape) {
  if (a.numel() != shape.numel()) {
    throw std::invalid_argument("Nexus autograd: reshape eleman sayisini degistiremez");
  }
  std::vector<float> out = a.values();  // kopya (görünüm yerine basitlik)
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), std::move(shape), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "ReshapeBackward";
  const Shape original = a.shape();
  node->backward_fn = [original](std::vector<Variable>&& grads) {
    std::vector<float> d = grads[0].values();  // kopya, sekil eskiye doner
    return std::vector<Variable>{Variable::make(std::move(d), original, false)};
  };
  return emit(std::move(out), std::move(shape), std::move(node), {edge_of(a)});
}

// ---- transpose (2D) ------------------------------------------------------------------------------
Variable transpose(const Variable& a) {
  if (a.shape().rank() != 2) {
    throw std::invalid_argument("Nexus autograd: transpose yalnizca 2D");
  }
  const std::int64_t m = a.shape().dims[0];
  const std::int64_t n = a.shape().dims[1];
  const auto& av = a.values();
  std::vector<float> out(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      out[static_cast<std::size_t>(j * m + i)] = av[static_cast<std::size_t>(i * n + j)];
    }
  }
  const Shape shape{{n, m}};
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "TransposeBackward";
  const std::int64_t cm = m;
  const std::int64_t cn = n;
  node->backward_fn = [cm, cn](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();  // [n,m]
    std::vector<float> d(static_cast<std::size_t>(cm * cn));
    for (std::int64_t i = 0; i < cm; ++i) {
      for (std::int64_t j = 0; j < cn; ++j) {
        d[static_cast<std::size_t>(i * cn + j)] = g[static_cast<std::size_t>(j * cm + i)];
      }
    }
    return std::vector<Variable>{Variable::make(std::move(d), Shape{{cm, cn}}, false)};
  };
  return emit(std::move(out), shape, std::move(node), {edge_of(a)});
}

// ---- matmul (2D) ------------------------------------------------------------------------------
Variable matmul(const Variable& a, const Variable& b) {
  if (a.shape().rank() != 2 || b.shape().rank() != 2) {
    throw std::invalid_argument("Nexus autograd: matmul yalnizca 2D");
  }
  const std::int64_t m = a.shape().dims[0];
  const std::int64_t k = a.shape().dims[1];
  const std::int64_t k2 = b.shape().dims[0];
  const std::int64_t n = b.shape().dims[1];
  if (k != k2) {
    throw std::invalid_argument("Nexus autograd: matmul ic boyut uyusmazligi");
  }
  const auto& av = a.values();
  const auto& bv = b.values();
  std::vector<float> out(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::int64_t t = 0; t < k; ++t) {
        acc += av[static_cast<std::size_t>(i * k + t)] * bv[static_cast<std::size_t>(t * n + j)];
      }
      out[static_cast<std::size_t>(i * n + j)] = acc;
    }
  }
  const Shape shape{{m, n}};
  const bool needs = grad_enabled() && (a.requires_grad() || b.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "MatmulBackward";
  const bool a_req = a.requires_grad();
  const bool b_req = b.requires_grad();
  std::vector<Edge> edges;
  if (a_req) {
    edges.push_back(edge_of(a));
  }
  if (b_req) {
    edges.push_back(edge_of(b));
  }
  SavedTensor saved_a = save_variable(a);
  SavedTensor saved_b = save_variable(b);
  node->backward_fn = [a_req, b_req, saved_a, saved_b, m, k, n](std::vector<Variable>&& grads) {
    const Variable& a_in = restore_variable(saved_a, "matmul");
    const Variable& b_in = restore_variable(saved_b, "matmul");
    const auto& g = grads[0].values();
    std::vector<Variable> result;
    if (a_req) {
      std::vector<float> d(static_cast<std::size_t>(m * k));
      for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t t = 0; t < k; ++t) {
          float acc = 0.0f;
          for (std::int64_t j = 0; j < n; ++j) {
            acc += g[static_cast<std::size_t>(i * n + j)] * b_in.values()[static_cast<std::size_t>(t * n + j)];
          }
          d[static_cast<std::size_t>(i * k + t)] = acc;
        }
      }
      result.push_back(Variable::make(std::move(d), Shape{{m, k}}, false));
    }
    if (b_req) {
      std::vector<float> d(static_cast<std::size_t>(k * n));
      for (std::int64_t t = 0; t < k; ++t) {
        for (std::int64_t j = 0; j < n; ++j) {
          float acc = 0.0f;
          for (std::int64_t i = 0; i < m; ++i) {
            acc += a_in.values()[static_cast<std::size_t>(i * k + t)] * g[static_cast<std::size_t>(i * n + j)];
          }
          d[static_cast<std::size_t>(t * n + j)] = acc;
        }
      }
      result.push_back(Variable::make(std::move(d), Shape{{k, n}}, false));
    }
    return result;
  };
  return emit(std::move(out), shape, std::move(node), std::move(edges));
}

// ---- softmax (2D, son boyut) ---------------------------------------------------------------------
Variable softmax(const Variable& a) {
  if (a.shape().rank() != 2) {
    throw std::invalid_argument("Nexus autograd: softmax yalnizca 2D");
  }
  const std::int64_t n = a.shape().dims[0];
  const std::int64_t c = a.shape().dims[1];
  const auto& av = a.values();
  std::vector<float> out(static_cast<std::size_t>(n * c));
  for (std::int64_t i = 0; i < n; ++i) {
    float maxv = av[static_cast<std::size_t>(i * c)];
    for (std::int64_t j = 1; j < c; ++j) {
      maxv = std::fmax(maxv, av[static_cast<std::size_t>(i * c + j)]);
    }
    float total = 0.0f;
    for (std::int64_t j = 0; j < c; ++j) {
      const float e = std::exp(av[static_cast<std::size_t>(i * c + j)] - maxv);
      out[static_cast<std::size_t>(i * c + j)] = e;
      total += e;
    }
    for (std::int64_t j = 0; j < c; ++j) {
      out[static_cast<std::size_t>(i * c + j)] /= total;
    }
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "SoftmaxBackward";
  const auto out_copy = out;
  node->backward_fn = [out_copy, n, c](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(static_cast<std::size_t>(n * c));
    for (std::int64_t i = 0; i < n; ++i) {
      float dot = 0.0f;
      for (std::int64_t j = 0; j < c; ++j) {
        dot += g[static_cast<std::size_t>(i * c + j)] * out_copy[static_cast<std::size_t>(i * c + j)];
      }
      for (std::int64_t j = 0; j < c; ++j) {
        d[static_cast<std::size_t>(i * c + j)] =
            out_copy[static_cast<std::size_t>(i * c + j)] * (g[static_cast<std::size_t>(i * c + j)] - dot);
      }
    }
    return std::vector<Variable>{Variable::make(std::move(d), Shape{{n, c}}, false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- cross_entropy -------------------------------------------------------------------------------
Variable cross_entropy(const Variable& logits, const std::vector<std::int64_t>& targets) {
  if (logits.shape().rank() != 2) {
    throw std::invalid_argument("Nexus autograd: cross_entropy logits 2D olmali");
  }
  const std::int64_t n = logits.shape().dims[0];
  const std::int64_t c = logits.shape().dims[1];
  if (static_cast<std::int64_t>(targets.size()) != n) {
    throw std::invalid_argument("Nexus autograd: cross_entropy hedef sayisi uyusmazligi");
  }
  const auto& lv = logits.values();
  std::vector<float> probs(static_cast<std::size_t>(n * c));
  float loss = 0.0f;
  for (std::int64_t i = 0; i < n; ++i) {
    float maxv = lv[static_cast<std::size_t>(i * c)];
    for (std::int64_t j = 1; j < c; ++j) {
      maxv = std::fmax(maxv, lv[static_cast<std::size_t>(i * c + j)]);
    }
    float total = 0.0f;
    for (std::int64_t j = 0; j < c; ++j) {
      const float e = std::exp(lv[static_cast<std::size_t>(i * c + j)] - maxv);
      probs[static_cast<std::size_t>(i * c + j)] = e;
      total += e;
    }
    for (std::int64_t j = 0; j < c; ++j) {
      probs[static_cast<std::size_t>(i * c + j)] /= total;
    }
    loss += -std::log(std::fmax(probs[static_cast<std::size_t>(i * c + targets[static_cast<std::size_t>(i)])],
                                1.0e-12f));
  }
  loss /= static_cast<float>(n);
  const bool needs = grad_enabled() && logits.requires_grad();
  if (!needs) {
    return Variable::make(std::vector<float>{loss}, Shape{}, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "CrossEntropyBackward";
  node->backward_fn = [probs, targets, n, c](std::vector<Variable>&& grads) {
    std::vector<float> d(static_cast<std::size_t>(n * c));
    const float scale = grads[0].values()[0] / static_cast<float>(n);
    for (std::int64_t i = 0; i < n; ++i) {
      for (std::int64_t j = 0; j < c; ++j) {
        const float onehot = (j == targets[static_cast<std::size_t>(i)]) ? 1.0f : 0.0f;
        d[static_cast<std::size_t>(i * c + j)] =
            (probs[static_cast<std::size_t>(i * c + j)] - onehot) * scale;
      }
    }
    return std::vector<Variable>{Variable::make(std::move(d), Shape{{n, c}}, false)};
  };
  return emit(std::vector<float>{loss}, Shape{}, std::move(node), {edge_of(logits)});
}

// ---- add_bias (son boyutta yayin) -----------------------------------------------------------------
Variable add_bias(const Variable& x, const Variable& bias) {
  const auto& xs = x.shape().dims;
  if (xs.empty()) {
    throw std::invalid_argument("Nexus autograd: add_bias giris boyutsuz");
  }
  const std::int64_t d = xs.back();
  if (bias.numel() != d || bias.shape().rank() != 1) {
    throw std::invalid_argument("Nexus autograd: add_bias bias [D] olmali");
  }
  const auto& xv = x.values();
  const auto& bv = bias.values();
  std::vector<float> out(xv.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = xv[i] + bv[i % static_cast<std::size_t>(d)];
  }
  const bool needs = grad_enabled() && (x.requires_grad() || bias.requires_grad());
  if (!needs) {
    return Variable::make(std::move(out), x.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "AddBiasBackward";
  const bool x_req = x.requires_grad();
  const bool bias_req = bias.requires_grad();
  std::vector<Edge> edges;
  if (x_req) {
    edges.push_back(edge_of(x));
  }
  if (bias_req) {
    edges.push_back(edge_of(bias));
  }
  node->backward_fn = [x_req, bias_req, d](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<Variable> result;
    if (x_req) {
      result.push_back(grads[0]);
    }
    if (bias_req) {
      std::vector<float> db(static_cast<std::size_t>(d), 0.0f);
      for (std::size_t i = 0; i < g.size(); ++i) {
        db[i % static_cast<std::size_t>(d)] += g[i];
      }
      result.push_back(Variable::make(std::move(db), Shape{{d}}, false));
    }
    return result;
  };
  return emit(std::move(out), x.shape(), std::move(node), std::move(edges));
}

// ---- dropout ---------------------------------------------------------------------------------------
Variable dropout(const Variable& a, float p, bool training, std::uint64_t seed) {
  if (!(p >= 0.0f && p < 1.0f)) {
    throw std::invalid_argument("Nexus autograd: dropout olasiligi [0,1) araliginda olmali");
  }
  if (!training) {
    return a;  // eval: ozdeslik
  }
  const auto& av = a.values();
  std::vector<float> out(av.size());
  std::vector<float> mask(av.size());
  std::uint64_t state = seed == 0 ? 0x9E3779B97F4A7C15ULL : seed;
  auto next_uniform = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state >> 11) & 0x1FFFFFFFFFFFFFULL) / static_cast<float>(1ULL << 53);
  };
  const float scale = 1.0f / (1.0f - p);
  for (std::size_t i = 0; i < out.size(); ++i) {
    mask[i] = next_uniform() < p ? 0.0f : scale;
    out[i] = av[i] * mask[i];
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "DropoutBackward";
  node->backward_fn = [mask](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = g[i] * mask[i];
    }
    return std::vector<Variable>{Variable::make(std::move(d), grads[0].shape(), false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- log_softmax (2D, son boyut) -------------------------------------------------------------------
Variable log_softmax(const Variable& a) {
  if (a.shape().rank() != 2) {
    throw std::invalid_argument("Nexus autograd: log_softmax yalnizca 2D");
  }
  const std::int64_t n = a.shape().dims[0];
  const std::int64_t c = a.shape().dims[1];
  const auto& av = a.values();
  std::vector<float> out(av.size());
  for (std::int64_t i = 0; i < n; ++i) {
    const float* row = av.data() + i * c;
    float maxv = row[0];
    for (std::int64_t j = 1; j < c; ++j) {
      maxv = std::fmax(maxv, row[j]);
    }
    double total = 0.0;
    for (std::int64_t j = 0; j < c; ++j) {
      total += std::exp(static_cast<double>(row[j] - maxv));
    }
    const float lse = maxv + static_cast<float>(std::log(total));
    for (std::int64_t j = 0; j < c; ++j) {
      out[static_cast<std::size_t>(i * c + j)] = row[j] - lse;
    }
  }
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), a.shape(), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "LogSoftmaxBackward";
  const auto out_copy = out;
  node->backward_fn = [out_copy, n, c](std::vector<Variable>&& grads) {
    // dx = g - softmax * sum(g)
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::int64_t i = 0; i < n; ++i) {
      float gsum = 0.0f;
      for (std::int64_t j = 0; j < c; ++j) {
        gsum += g[static_cast<std::size_t>(i * c + j)];
      }
      for (std::int64_t j = 0; j < c; ++j) {
        const std::size_t k = static_cast<std::size_t>(i * c + j);
        d[k] = g[k] - std::exp(out_copy[k]) * gsum;
      }
    }
    return std::vector<Variable>{Variable::make(std::move(d), Shape{{n, c}}, false)};
  };
  return emit(std::move(out), a.shape(), std::move(node), {edge_of(a)});
}

// ---- embedding (satir toplama) ----------------------------------------------------------------------
Variable embedding(const Variable& weight, const std::vector<std::int64_t>& indices, std::int64_t padding_idx) {
  if (weight.shape().rank() != 2) {
    throw std::invalid_argument("Nexus autograd: embedding agirligi [V,D] olmali");
  }
  const std::int64_t vocab = weight.shape().dims[0];
  const std::int64_t dim = weight.shape().dims[1];
  const auto count = static_cast<std::int64_t>(indices.size());
  const auto& wv = weight.values();
  std::vector<float> out(static_cast<std::size_t>(count * dim));
  for (std::int64_t i = 0; i < count; ++i) {
    const std::int64_t row = indices[static_cast<std::size_t>(i)];
    if (row < 0 || row >= vocab) {
      throw std::out_of_range("Nexus autograd: embedding indeksi sozluk disinda");
    }
    std::copy(wv.begin() + row * dim, wv.begin() + (row + 1) * dim, out.begin() + i * dim);
  }
  const Shape shape{{count, dim}};
  const bool needs = grad_enabled() && weight.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "EmbeddingBackward";
  node->backward_fn = [indices, vocab, dim, padding_idx](std::vector<Variable>&& grads) {
    // Deterministik segment toplami: indeks sirasiyla biriktirilir (atomiksiz).
    const auto& g = grads[0].values();
    std::vector<float> d(static_cast<std::size_t>(vocab * dim), 0.0f);
    for (std::size_t i = 0; i < indices.size(); ++i) {
      const std::int64_t row = indices[i];
      if (row == padding_idx) {
        continue;
      }
      for (std::int64_t j = 0; j < dim; ++j) {
        d[static_cast<std::size_t>(row * dim + j)] += g[i * static_cast<std::size_t>(dim) + static_cast<std::size_t>(j)];
      }
    }
    return std::vector<Variable>{Variable::make(std::move(d), Shape{{vocab, dim}}, false)};
  };
  return emit(std::move(out), shape, std::move(node), {edge_of(weight)});
}

// ---- narrow (2D dilim, kopyali) ---------------------------------------------------------------------
Variable narrow(const Variable& a, int dim, std::int64_t start, std::int64_t length) {
  if (a.shape().rank() != 2 || (dim != 0 && dim != 1)) {
    throw std::invalid_argument("Nexus autograd: narrow yalnizca 2D, dim 0/1");
  }
  const std::int64_t rows = a.shape().dims[0];
  const std::int64_t cols = a.shape().dims[1];
  const std::int64_t extent = dim == 0 ? rows : cols;
  if (start < 0 || length < 0 || start + length > extent) {
    throw std::out_of_range("Nexus autograd: narrow araligi sinir disinda");
  }
  const std::int64_t out_rows = dim == 0 ? length : rows;
  const std::int64_t out_cols = dim == 1 ? length : cols;
  const auto& av = a.values();
  std::vector<float> out(static_cast<std::size_t>(out_rows * out_cols));
  for (std::int64_t i = 0; i < out_rows; ++i) {
    for (std::int64_t j = 0; j < out_cols; ++j) {
      const std::int64_t si = dim == 0 ? i + start : i;
      const std::int64_t sj = dim == 1 ? j + start : j;
      out[static_cast<std::size_t>(i * out_cols + j)] = av[static_cast<std::size_t>(si * cols + sj)];
    }
  }
  const Shape shape{{out_rows, out_cols}};
  const bool needs = grad_enabled() && a.requires_grad();
  if (!needs) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "NarrowBackward";
  node->backward_fn = [rows, cols, out_rows, out_cols, dim, start](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(static_cast<std::size_t>(rows * cols), 0.0f);
    for (std::int64_t i = 0; i < out_rows; ++i) {
      for (std::int64_t j = 0; j < out_cols; ++j) {
        const std::int64_t si = dim == 0 ? i + start : i;
        const std::int64_t sj = dim == 1 ? j + start : j;
        d[static_cast<std::size_t>(si * cols + sj)] = g[static_cast<std::size_t>(i * out_cols + j)];
      }
    }
    return std::vector<Variable>{Variable::make(std::move(d), Shape{{rows, cols}}, false)};
  };
  return emit(std::move(out), shape, std::move(node), {edge_of(a)});
}

// ---- cat (2D, dim 0/1) -------------------------------------------------------------------------------
Variable cat(const std::vector<Variable>& parts, int dim) {
  if (parts.empty() || (dim != 0 && dim != 1)) {
    throw std::invalid_argument("Nexus autograd: cat bos olmayan liste ve dim 0/1 bekler");
  }
  const int other = dim == 0 ? 1 : 0;
  if (parts[0].shape().rank() != 2) {
    throw std::invalid_argument("Nexus autograd: cat parcalari 2D olmali");
  }
  const std::int64_t fixed = parts[0].shape().dims[static_cast<std::size_t>(other)];
  std::vector<std::int64_t> extents;
  std::int64_t total = 0;
  for (const Variable& part : parts) {
    if (part.shape().rank() != 2 || part.shape().dims[static_cast<std::size_t>(other)] != fixed) {
      throw std::invalid_argument("Nexus autograd: cat parcalari 2D ve diger boyutta esit olmali");
    }
    extents.push_back(part.shape().dims[static_cast<std::size_t>(dim)]);
    total += extents.back();
  }
  const std::int64_t rows = dim == 0 ? total : fixed;
  const std::int64_t cols = dim == 1 ? total : fixed;
  std::vector<float> out(static_cast<std::size_t>(rows * cols));
  std::int64_t offset = 0;
  for (std::size_t p = 0; p < parts.size(); ++p) {
    const auto& pv = parts[p].values();
    const std::int64_t pr = parts[p].shape().dims[0];
    const std::int64_t pc = parts[p].shape().dims[1];
    for (std::int64_t i = 0; i < pr; ++i) {
      for (std::int64_t j = 0; j < pc; ++j) {
        const std::int64_t oi = dim == 0 ? i + offset : i;
        const std::int64_t oj = dim == 1 ? j + offset : j;
        out[static_cast<std::size_t>(oi * cols + oj)] = pv[static_cast<std::size_t>(i * pc + j)];
      }
    }
    offset += extents[p];
  }
  const Shape shape{{rows, cols}};
  std::vector<bool> req;
  bool any = false;
  for (const Variable& part : parts) {
    req.push_back(part.requires_grad());
    any = any || part.requires_grad();
  }
  if (!(grad_enabled() && any)) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "CatBackward";
  std::vector<Edge> edges;
  for (const Variable& part : parts) {
    if (part.requires_grad()) {
      edges.push_back(edge_of(part));
    }
  }
  node->backward_fn = [req, extents, dim, cols, fixed](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<Variable> result;
    std::int64_t offset = 0;
    for (std::size_t p = 0; p < extents.size(); ++p) {
      if (req[p]) {
        const std::int64_t pr = dim == 0 ? extents[p] : fixed;
        const std::int64_t pc = dim == 1 ? extents[p] : fixed;
        std::vector<float> d(static_cast<std::size_t>(pr * pc));
        for (std::int64_t i = 0; i < pr; ++i) {
          for (std::int64_t j = 0; j < pc; ++j) {
            const std::int64_t oi = dim == 0 ? i + offset : i;
            const std::int64_t oj = dim == 1 ? j + offset : j;
            d[static_cast<std::size_t>(i * pc + j)] = g[static_cast<std::size_t>(oi * cols + oj)];
          }
        }
        result.push_back(Variable::make(std::move(d), Shape{{pr, pc}}, false));
      }
      offset += extents[p];
    }
    return result;
  };
  return emit(std::move(out), shape, std::move(node), std::move(edges));
}

// ---- CustomFunction ------------------------------------------------------------------------------------
Variable custom_function(std::string name, const std::vector<Variable>& inputs, std::vector<float> output,
                         Shape output_shape, CustomBackwardFn backward_fn) {
  if (static_cast<std::int64_t>(output.size()) != output_shape.numel()) {
    throw std::invalid_argument("Nexus autograd: custom_function cikti boyutu sekille uyusmuyor");
  }
  if (!backward_fn) {
    throw std::invalid_argument("Nexus autograd: custom_function geri fonksiyonu bos");
  }
  std::vector<bool> req;
  bool any = false;
  for (const Variable& input : inputs) {
    req.push_back(input.requires_grad());
    any = any || input.requires_grad();
  }
  if (!(grad_enabled() && any)) {
    return Variable::make(std::move(output), output_shape, false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = std::move(name);
  std::vector<Edge> edges;
  std::vector<Shape> shapes;
  for (const Variable& input : inputs) {
    shapes.push_back(input.shape());
    if (input.requires_grad()) {
      edges.push_back(edge_of(input));
    }
  }
  const std::string label = node->name;
  node->backward_fn = [req, shapes, label, fn = std::move(backward_fn)](std::vector<Variable>&& grads) {
    std::vector<Variable> all = fn(grads[0]);
    if (all.size() != req.size()) {
      throw std::runtime_error("Nexus autograd: " + label + " her giris icin bir gradyan dondurmeli");
    }
    std::vector<Variable> result;
    for (std::size_t i = 0; i < all.size(); ++i) {
      if (!req[i]) {
        continue;
      }
      if (!all[i].has_data() || all[i].shape().dims != shapes[i].dims) {
        throw std::runtime_error("Nexus autograd: " + label + " gradyan sekli girisle uyusmuyor");
      }
      result.push_back(std::move(all[i]));
    }
    return result;
  };
  return emit(std::move(output), output_shape, std::move(node), std::move(edges));
}

}  // namespace autograd
}  // namespace nexus
