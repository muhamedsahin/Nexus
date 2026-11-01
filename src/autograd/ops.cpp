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

#include "nexus/core/tensor.hpp"
#include "nexus/core/tensor_ops.hpp"

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

// ---- Dalga 2: Tensor yolu ---------------------------------------------------------------
// Girdilerden biri Tensor depolamaliysa (Variable::wrap / Tensor op ciktisi) islem nexus::ops ile yapilir:
// her dtype (dagiticidaki cekirdekler), CUDA dahil. Gradyan her girdiye KENDI depolama turunde doner
// (FAZ 2 vector girdisine F32 host vector), boylece karisik grafiklerde FAZ 2 dugumleri values() ile calisir.
[[nodiscard]] bool tensor_path(const Variable& a) { return a.is_tensor_backed(); }

[[nodiscard]] Variable grad_as(bool vector_input, nexus::Tensor g) {
  return vector_input ? Variable::from_tensor(g, false) : Variable::wrap(std::move(g), false);
}

[[nodiscard]] Variable tensor_binary(const Variable& a, const Variable& b, bool is_mul) {
  require_same_shape(a, b, is_mul ? "mul" : "add");  // Tensor yolunda yayin yok (eksen indirgemesi yok)
  const nexus::Tensor ta = a.tensor(), tb = b.tensor();
  nexus::Tensor out = is_mul ? nexus::ops::mul(ta, tb) : nexus::ops::add(ta, tb);
  const bool needs = grad_enabled() && (a.requires_grad() || b.requires_grad());
  if (!needs) {
    return Variable::wrap(std::move(out), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = is_mul ? "MulBackward(Tensor)" : "AddBackward(Tensor)";
  std::vector<Edge> edges;
  const bool a_req = a.requires_grad(), b_req = b.requires_grad();
  const bool a_vec = !a.is_tensor_backed(), b_vec = !b.is_tensor_backed();
  if (a_req) edges.push_back(edge_of(a));
  if (b_req) edges.push_back(edge_of(b));
  SavedTensor sa, sb;
  if (is_mul) {
    if (b_req) sa = save_variable(a);
    if (a_req) sb = save_variable(b);
  }
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    const nexus::Tensor g = grads[0].tensor();
    std::vector<Variable> result;
    if (a_req) result.push_back(grad_as(a_vec, is_mul ? nexus::ops::mul(g, sb.unpack("mul").tensor()) : g.shallow()));
    if (b_req) result.push_back(grad_as(b_vec, is_mul ? nexus::ops::mul(g, sa.unpack("mul").tensor()) : g.shallow()));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::move(edges));
}

[[nodiscard]] Variable tensor_matmul(const Variable& a, const Variable& b) {
  const nexus::Tensor ta = a.tensor(), tb = b.tensor();
  nexus::Tensor out = nexus::ops::matmul(ta, tb);
  const bool needs = grad_enabled() && (a.requires_grad() || b.requires_grad());
  if (!needs) {
    return Variable::wrap(std::move(out), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "MatmulBackward(Tensor)";
  std::vector<Edge> edges;
  const bool a_req = a.requires_grad(), b_req = b.requires_grad();
  const bool a_vec = !a.is_tensor_backed(), b_vec = !b.is_tensor_backed();
  if (a_req) edges.push_back(edge_of(a));
  if (b_req) edges.push_back(edge_of(b));
  const SavedTensor sa = b_req ? save_variable(a) : SavedTensor{};
  const SavedTensor sb = a_req ? save_variable(b) : SavedTensor{};
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    const nexus::Tensor g = grads[0].tensor();
    std::vector<Variable> result;
    // dA = g B^T, dB = A^T g; devrik gorunumler kopyasiz (gemm trans bayragi)
    if (a_req) result.push_back(grad_as(a_vec, nexus::ops::matmul(g, sb.unpack("matmul").tensor().transpose(0, 1))));
    if (b_req) result.push_back(grad_as(b_vec, nexus::ops::matmul(sa.unpack("matmul").tensor().transpose(0, 1), g)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::move(edges));
}


// Kaydedilmis cikti (sigmoid / tanh / exp / softmax): kopya yerine surum damgali sig gorunum.
struct SavedOut {
  nexus::Tensor t;
  std::uint64_t version = 0;
  [[nodiscard]] const nexus::Tensor& get(const char* op) const {
    if (t.version() != version)
      throw std::runtime_error(std::string("Nexus autograd: ") + op + " ciktisi geri gecisten once yerinde degisti (surum)");
    return t;
  }
};

[[nodiscard]] Variable tensor_unary(const Variable& a, nexus::ops::UnaryFn fn, double scalar, const char* name) {
  using F = nexus::ops::UnaryFn;
  const nexus::Tensor ta = a.tensor();
  nexus::Tensor out = nexus::ops::unary(fn, ta, scalar);
  if (!(grad_enabled() && a.requires_grad())) return Variable::wrap(std::move(out), false);
  auto node = std::make_shared<LambdaNode>();
  node->name = std::string(name) + "Backward(Tensor)";
  const bool a_vec = !a.is_tensor_backed();
  const bool uses_out = fn == F::Sigmoid || fn == F::Tanh || fn == F::Exp;
  const bool uses_in = fn == F::Pow || fn == F::Relu || fn == F::Log || fn == F::Gelu || fn == F::Silu;
  const SavedTensor saved_in = uses_in ? save_variable(a) : SavedTensor{};
  const SavedOut saved_out = uses_out ? SavedOut{out.shallow(), out.version()} : SavedOut{};
  const std::string op = name;
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    const nexus::Tensor g = grads[0].tensor();
    const nexus::Tensor sv = uses_out ? saved_out.get(op.c_str()) : uses_in ? saved_in.unpack(op.c_str()).tensor() : g;
    std::vector<Variable> result;
    result.push_back(grad_as(a_vec, nexus::ops::unary_grad(fn, g, sv.dtype() == g.dtype() ? sv : sv.to(g.dtype()), scalar)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(a)});
}

[[nodiscard]] Variable tensor_sub_div(const Variable& a, const Variable& b, bool is_div) {
  require_same_shape(a, b, is_div ? "div" : "sub");
  const nexus::Tensor ta = a.tensor(), tb = b.tensor();
  nexus::Tensor out = is_div ? nexus::ops::div(ta, tb) : nexus::ops::sub(ta, tb);
  const bool needs = grad_enabled() && (a.requires_grad() || b.requires_grad());
  if (!needs) return Variable::wrap(std::move(out), false);
  auto node = std::make_shared<LambdaNode>();
  node->name = is_div ? "DivBackward(Tensor)" : "SubBackward(Tensor)";
  const bool a_req = a.requires_grad(), b_req = b.requires_grad();
  const bool a_vec = !a.is_tensor_backed(), b_vec = !b.is_tensor_backed();
  std::vector<Edge> edges;
  if (a_req) edges.push_back(edge_of(a));
  if (b_req) edges.push_back(edge_of(b));
  const SavedTensor sa = is_div ? save_variable(a) : SavedTensor{};
  const SavedTensor sb = is_div ? save_variable(b) : SavedTensor{};
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    using F = nexus::ops::UnaryFn;
    const nexus::Tensor g = grads[0].tensor();
    std::vector<Variable> result;
    if (!is_div) {
      if (a_req) result.push_back(grad_as(a_vec, g.shallow()));
      if (b_req) result.push_back(grad_as(b_vec, nexus::ops::unary(F::Neg, g)));
      return result;
    }
    const nexus::Tensor av = sa.unpack("div").tensor(), bv = sb.unpack("div").tensor();
    if (a_req) result.push_back(grad_as(a_vec, nexus::ops::div(g, bv)));
    if (b_req) result.push_back(grad_as(b_vec, nexus::ops::unary(F::Neg, nexus::ops::div(nexus::ops::mul(g, av), nexus::ops::mul(bv, bv)))));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::move(edges));
}

// Gorunum op'lari (kopyasiz): reshape / transpose / narrow
template <typename Fwd, typename Bwd>
[[nodiscard]] Variable tensor_view_op(const Variable& a, const char* name, Fwd fwd, Bwd bwd) {
  nexus::Tensor out = fwd(a.tensor());
  if (!(grad_enabled() && a.requires_grad())) return Variable::wrap(std::move(out), false);
  auto node = std::make_shared<LambdaNode>();
  node->name = std::string(name) + "Backward(Tensor)";
  const bool a_vec = !a.is_tensor_backed();
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    std::vector<Variable> result;
    result.push_back(grad_as(a_vec, bwd(grads[0].tensor())));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(a)});
}

[[nodiscard]] Variable tensor_softmax(const Variable& a, bool log) {
  if (a.shape().rank() != 2) throw std::invalid_argument(log ? "Nexus autograd: log_softmax yalnizca 2D" : "Nexus autograd: softmax yalnizca 2D");
  nexus::Tensor out = nexus::ops::row_softmax(a.tensor(), log);
  if (!(grad_enabled() && a.requires_grad())) return Variable::wrap(std::move(out), false);
  auto node = std::make_shared<LambdaNode>();
  node->name = log ? "LogSoftmaxBackward(Tensor)" : "SoftmaxBackward(Tensor)";
  const bool a_vec = !a.is_tensor_backed();
  const SavedOut y{out.shallow(), out.version()};
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    std::vector<Variable> result;
    result.push_back(grad_as(a_vec, nexus::ops::row_softmax_grad(grads[0].tensor(), y.get(log ? "log_softmax" : "softmax"), log)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(a)});
}

[[nodiscard]] nexus::Tensor index_tensor(const std::vector<std::int64_t>& v, const nexus::Tensor& like) {
  nexus::Tensor h = nexus::Tensor::empty(nexus::Shape{v.size()}, nexus::DType::I64);
  if (!v.empty()) std::copy(v.begin(), v.end(), h.data_as<std::int64_t>());
  return like.space() == nexus::MemorySpace::Device ? h.to(like.space(), like.device()) : h;
}

[[nodiscard]] Variable tensor_cross_entropy(const Variable& logits, const std::vector<std::int64_t>& targets) {
  if (logits.shape().rank() != 2) throw std::invalid_argument("Nexus autograd: cross_entropy logits 2D olmali");
  if (static_cast<std::int64_t>(targets.size()) != logits.shape().dims[0])
    throw std::invalid_argument("Nexus autograd: cross_entropy hedef sayisi uyusmazligi");
  const nexus::Tensor tl = logits.tensor();
  const nexus::Tensor tt = index_tensor(targets, tl);
  nexus::Tensor out = nexus::ops::cross_entropy(tl, tt);  // F32 {1} (F64 logitte F64)
  if (!(grad_enabled() && logits.requires_grad())) return Variable::wrap(std::move(out), false);
  auto node = std::make_shared<LambdaNode>();
  node->name = "CrossEntropyBackward(Tensor)";
  const bool l_vec = !logits.is_tensor_backed();
  const SavedTensor sl = save_variable(logits);
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    nexus::Tensor g = grads[0].tensor();
    const nexus::DType loss_dt = tl.dtype() == nexus::DType::F64 ? nexus::DType::F64 : nexus::DType::F32;
    if (g.dtype() != loss_dt) g = g.to(loss_dt);
    std::vector<Variable> result;
    result.push_back(grad_as(l_vec, nexus::ops::cross_entropy_grad(sl.unpack("cross_entropy").tensor(), tt, g)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(logits)});
}

[[nodiscard]] Variable tensor_add_bias(const Variable& x, const Variable& bias) {
  const auto& xs = x.shape().dims;
  if (xs.empty()) throw std::invalid_argument("Nexus autograd: add_bias giris boyutsuz");
  if (bias.numel() != xs.back() || bias.shape().rank() != 1) throw std::invalid_argument("Nexus autograd: add_bias bias [D] olmali");
  nexus::Tensor out = nexus::ops::add(x.tensor(), bias.tensor());  // son eksende yayin
  const bool needs = grad_enabled() && (x.requires_grad() || bias.requires_grad());
  if (!needs) return Variable::wrap(std::move(out), false);
  auto node = std::make_shared<LambdaNode>();
  node->name = "AddBiasBackward(Tensor)";
  const bool x_req = x.requires_grad(), b_req = bias.requires_grad();
  const bool x_vec = !x.is_tensor_backed(), b_vec = !bias.is_tensor_backed();
  std::vector<Edge> edges;
  if (x_req) edges.push_back(edge_of(x));
  if (b_req) edges.push_back(edge_of(bias));
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    const nexus::Tensor g = grads[0].tensor();
    std::vector<Variable> result;
    if (x_req) result.push_back(grad_as(x_vec, g.shallow()));
    if (b_req) result.push_back(grad_as(b_vec, nexus::ops::reduce_rows(g)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::move(edges));
}

[[nodiscard]] Variable tensor_dropout(const Variable& a, float p, std::uint64_t seed) {
  // FAZ 2 ile ayni LCG maskesi (host'ta uretilir, girdinin dtype / cihazina tasinir): F32 CPU'da bit bit ayni.
  const nexus::Tensor ta = a.tensor();
  nexus::Tensor mask = nexus::Tensor::empty(ta.shape(), nexus::DType::F32);
  std::uint64_t state = seed == 0 ? 0x9E3779B97F4A7C15ULL : seed;
  auto next_uniform = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state >> 11) & 0x1FFFFFFFFFFFFFULL) / static_cast<float>(1ULL << 53);
  };
  const float scale = 1.0f / (1.0f - p);
  float* mp = mask.data();
  for (std::size_t i = 0; i < mask.numel(); ++i) mp[i] = next_uniform() < p ? 0.0f : scale;
  if (ta.dtype() != nexus::DType::F32) mask = mask.to(ta.dtype());
  if (ta.space() == nexus::MemorySpace::Device) mask = mask.to(ta.space(), ta.device());
  nexus::Tensor out = nexus::ops::mul(ta.is_contiguous() ? ta : ta.contiguous(), mask);
  if (!(grad_enabled() && a.requires_grad())) return Variable::wrap(std::move(out), false);
  auto node = std::make_shared<LambdaNode>();
  node->name = "DropoutBackward(Tensor)";
  const bool a_vec = !a.is_tensor_backed();
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    std::vector<Variable> result;
    result.push_back(grad_as(a_vec, nexus::ops::mul(grads[0].tensor().contiguous(), mask)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(a)});
}

[[nodiscard]] Variable tensor_embedding(const Variable& weight, const std::vector<std::int64_t>& indices, std::int64_t padding_idx) {
  if (weight.shape().rank() != 2) throw std::invalid_argument("Nexus autograd: embedding agirligi [V,D] olmali");
  const std::int64_t vocab = weight.shape().dims[0];
  for (const std::int64_t r : indices)
    if (r < 0 || r >= vocab) throw std::out_of_range("Nexus autograd: embedding indeksi sozluk disinda");
  const nexus::Tensor tw = weight.tensor();
  const nexus::Tensor idx = index_tensor(indices, tw);
  nexus::Tensor out = nexus::ops::gather_rows(tw, idx);
  if (!(grad_enabled() && weight.requires_grad())) return Variable::wrap(std::move(out), false);
  // Belirlenimci plan (host): kararli siralama -> perm, segment satiri, segment ofseti
  const auto L = static_cast<std::int64_t>(indices.size());
  std::vector<std::int64_t> perm(indices.size());
  for (std::size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<std::int64_t>(i);
  std::stable_sort(perm.begin(), perm.end(), [&](std::int64_t x, std::int64_t y) { return indices[static_cast<std::size_t>(x)] < indices[static_cast<std::size_t>(y)]; });
  std::vector<std::int64_t> seg_row, seg_off;
  for (std::size_t q = 0; q < perm.size(); ++q) {
    const std::int64_t r = indices[static_cast<std::size_t>(perm[q])];
    if (seg_row.empty() || seg_row.back() != r) {
      seg_row.push_back(r);
      seg_off.push_back(static_cast<std::int64_t>(q));
    }
  }
  seg_off.push_back(L);
  std::vector<std::int64_t> plan_v(perm);
  plan_v.insert(plan_v.end(), seg_row.begin(), seg_row.end());
  plan_v.insert(plan_v.end(), seg_off.begin(), seg_off.end());
  const nexus::Tensor plan = index_tensor(plan_v, tw);
  const auto U = static_cast<std::int64_t>(seg_row.size());
  auto node = std::make_shared<LambdaNode>();
  node->name = "EmbeddingBackward(Tensor)";
  const bool w_vec = !weight.is_tensor_backed();
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    std::vector<Variable> result;
    result.push_back(grad_as(w_vec, nexus::ops::embedding_grad(grads[0].tensor(), plan, vocab, L, U, padding_idx)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(weight)});
}

[[nodiscard]] Variable tensor_cat(const std::vector<Variable>& parts, int dim) {
  if (parts.empty() || (dim != 0 && dim != 1)) throw std::invalid_argument("Nexus autograd: cat bos olmayan liste ve dim 0/1 bekler");
  std::vector<nexus::Tensor> ts;
  for (const Variable& p : parts) ts.push_back(p.tensor());
  const int other = dim == 0 ? 1 : 0;
  std::size_t total = 0;
  for (const nexus::Tensor& t : ts) {
    if (t.rank() != 2 || t.shape()[static_cast<std::size_t>(other)] != ts[0].shape()[static_cast<std::size_t>(other)])
      throw std::invalid_argument("Nexus autograd: cat parcalari 2D ve diger boyutta esit olmali");
    if (t.dtype() != ts[0].dtype() || t.space() != ts[0].space() || t.device() != ts[0].device())
      throw std::invalid_argument("Nexus autograd: cat parcalari ayni dtype ve cihazda olmali (otomatik donusum yok)");
    total += t.shape()[static_cast<std::size_t>(dim)];
  }
  const std::size_t rows = dim == 0 ? total : ts[0].shape()[0], cols = dim == 1 ? total : ts[0].shape()[1];
  nexus::Tensor out = nexus::Tensor::empty(nexus::Shape{rows, cols}, ts[0].dtype(), ts[0].space(), ts[0].device());
  std::vector<std::size_t> offs;
  std::size_t off = 0;
  for (const nexus::Tensor& t : ts) {
    const std::size_t len = t.shape()[static_cast<std::size_t>(dim)];
    nexus::Tensor dst = out.narrow(static_cast<std::size_t>(dim), off, len);
    dst.copy_from(t);
    offs.push_back(off);
    off += len;
  }
  bool any = false;
  std::vector<bool> req, vec;
  for (const Variable& p : parts) {
    req.push_back(p.requires_grad());
    vec.push_back(!p.is_tensor_backed());
    any = any || p.requires_grad();
  }
  if (!(grad_enabled() && any)) return Variable::wrap(std::move(out), false);
  auto node = std::make_shared<LambdaNode>();
  node->name = "CatBackward(Tensor)";
  std::vector<Edge> edges;
  std::vector<std::size_t> lens;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    lens.push_back(ts[i].shape()[static_cast<std::size_t>(dim)]);
    if (req[i]) edges.push_back(edge_of(parts[i]));
  }
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    const nexus::Tensor g = grads[0].tensor();
    std::vector<Variable> result;
    for (std::size_t i = 0; i < req.size(); ++i)
      if (req[i]) result.push_back(grad_as(vec[i], g.narrow(static_cast<std::size_t>(dim), offs[i], lens[i]).contiguous()));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::move(edges));
}

[[nodiscard]] Variable tensor_sum(const Variable& a) {
  const nexus::Tensor ta = a.tensor();
  nexus::Tensor out = nexus::ops::sum(ta);  // sekil {1}
  if (!(grad_enabled() && a.requires_grad())) {
    return Variable::wrap(std::move(out), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "SumBackward(Tensor)";
  const nexus::Shape in_shape = ta.shape();
  const bool a_vec = !a.is_tensor_backed();
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    // {1} -> girdi sekli: adim 0 yayin, sonra bitisik (cihazda kopya cekirdegi)
    std::vector<Variable> result;
    result.push_back(grad_as(a_vec, grads[0].tensor().expand(in_shape).contiguous()));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(a)});
}

}  // namespace

// ---- Dalga 2: to (dtype / cihaz) -----------------------------------------------------------
Variable to(const Variable& a, nexus::DType dtype) {
  const nexus::Tensor ta = a.tensor();
  nexus::Tensor out = ta.to(dtype);
  if (!(grad_enabled() && a.requires_grad())) {
    return Variable::wrap(std::move(out), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "ToDtypeBackward(Tensor)";
  const nexus::DType src = ta.dtype();
  const bool a_vec = !a.is_tensor_backed();
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    std::vector<Variable> result;
    result.push_back(grad_as(a_vec, grads[0].tensor().to(src)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(a)});
}

Variable to(const Variable& a, nexus::MemorySpace space, nexus::DeviceRef device) {
  const nexus::Tensor ta = a.tensor();
  nexus::Tensor out = ta.to(space, device);
  if (!(grad_enabled() && a.requires_grad())) {
    return Variable::wrap(std::move(out), false);
  }
  auto node = std::make_shared<LambdaNode>();
  node->name = "ToDeviceBackward(Tensor)";
  const nexus::MemorySpace src_space = ta.space();
  const nexus::DeviceRef src_device = ta.device();
  const bool a_vec = !a.is_tensor_backed();
  node->backward_fn = [=](std::vector<Variable>&& grads) {
    std::vector<Variable> result;
    result.push_back(grad_as(a_vec, grads[0].tensor().to(src_space, src_device)));
    return result;
  };
  return make_op_output_tensor(std::move(out), std::move(node), std::vector<Edge>{edge_of(a)});
}

namespace {

}  // namespace

// ---- add -------------------------------------------------------------------------------
Variable add(const Variable& a, const Variable& b) {
  if (tensor_path(a) || tensor_path(b)) return tensor_binary(a, b, false);
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
  if (tensor_path(a) || tensor_path(b)) return tensor_sub_div(a, b, false);
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
  if (tensor_path(a) || tensor_path(b)) return tensor_binary(a, b, true);
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
  if (tensor_path(a) || tensor_path(b)) return tensor_sub_div(a, b, true);
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Neg, 0.0, "Neg");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::AddScalar, scalar, "AddScalar");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::MulScalar, scalar, "MulScalar");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Pow, exponent, "Pow");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Relu, 0.0, "Relu");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Sigmoid, 0.0, "Sigmoid");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Tanh, 0.0, "Tanh");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Exp, 0.0, "Exp");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Log, 0.0, "Log");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Gelu, 0.0, "Gelu");
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
  if (tensor_path(a)) return tensor_unary(a, nexus::ops::UnaryFn::Silu, 0.0, "Silu");
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
  if (tensor_path(a)) return tensor_sum(a);
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
  if (tensor_path(a)) return tensor_unary(sum(a), nexus::ops::UnaryFn::MulScalar, 1.0 / static_cast<double>(a.numel()), "Mean");
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
  if (tensor_path(a)) {
    std::vector<std::size_t> nd;
    for (const std::int64_t d : shape.dims) nd.push_back(static_cast<std::size_t>(d));
    if (nd.empty()) nd.push_back(1);
    const nexus::Shape target(nd);
    const nexus::Shape original = a.tensor().shape();
    return tensor_view_op(a, "Reshape", [target](const nexus::Tensor& t) { return t.reshape(target); },
                          [original](const nexus::Tensor& g) { return g.reshape(original); });
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
  if (tensor_path(a)) {
    return tensor_view_op(a, "Transpose", [](const nexus::Tensor& t) { return t.transpose(0, 1); },
                          [](const nexus::Tensor& g) { return g.transpose(0, 1); });
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
  if (tensor_path(a) || tensor_path(b)) return tensor_matmul(a, b);
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
  if (tensor_path(a)) return tensor_softmax(a, false);
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
  if (tensor_path(logits)) return tensor_cross_entropy(logits, targets);
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
  if (tensor_path(x) || tensor_path(bias)) return tensor_add_bias(x, bias);
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
  if (tensor_path(a)) return tensor_dropout(a, p, seed);
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
  if (tensor_path(a)) return tensor_softmax(a, true);
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
  if (tensor_path(weight)) return tensor_embedding(weight, indices, padding_idx);
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
  if (tensor_path(a)) {
    const nexus::Tensor ta = a.tensor();
    const nexus::Shape full = ta.shape();
    const nexus::DType dt = ta.dtype();
    const nexus::MemorySpace sp = ta.space();
    const nexus::DeviceRef dv = ta.device();
    const auto d = static_cast<std::size_t>(dim), st = static_cast<std::size_t>(start), len = static_cast<std::size_t>(length);
    return tensor_view_op(a, "Narrow", [=](const nexus::Tensor& t) { return t.narrow(d, st, len); },
                          [=](const nexus::Tensor& g) {
                            nexus::Tensor z = nexus::Tensor::zeros(full, dt, sp, dv);
                            nexus::Tensor view = z.narrow(d, st, len);
                            view.copy_from(g);
                            return z;
                          });
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
  if (std::any_of(parts.begin(), parts.end(), [](const Variable& p) { return p.is_tensor_backed(); })) return tensor_cat(parts, dim);
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
