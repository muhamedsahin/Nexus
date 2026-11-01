#pragma once

/**
 * @file tensor_autograd_cases.hpp
 * @brief Dalga 2: autograd Tensor yolu test vakalari (CPU: test_tensor_autograd_ops, GPU: test_tensor_cuda ortak).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/autograd/tensor_optim.hpp"
#include "nexus/core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace tac {

namespace ag = nexus::autograd;
using nexus::DType;
using nexus::Shape;
using nexus::Tensor;

inline std::vector<double> randv(std::size_t n, unsigned seed, double lo = 0.0, double scale = 1.0) {
  std::mt19937 g(seed);
  std::normal_distribution<double> d(0.0, scale);
  std::vector<double> v(n);
  for (auto& x : v) x = lo > 0 ? lo + std::fabs(d(g)) : d(g);
  return v;
}

struct Input {
  std::vector<double> v;
  std::vector<std::int64_t> dims;
};

inline const nexus::DeviceRef kCuda0{nexus::DeviceKind::Cuda, 0};

inline ag::Variable make_input(const Input& in, bool tensor, DType t, bool gpu = false) {
  if (!tensor) return ag::Variable::make(std::vector<float>(in.v.begin(), in.v.end()), ag::Shape{in.dims}, true);
  std::vector<std::size_t> d(in.dims.begin(), in.dims.end());
  Tensor h = Tensor::from_values(Shape(d), in.v, t);
  return ag::Variable::wrap(gpu ? h.to(nexus::MemorySpace::Device, kCuda0) : h, true);
}

inline std::vector<double> to_vec(const ag::Variable& v) {
  if (v.is_tensor_backed()) return v.tensor().to(nexus::MemorySpace::Host, nexus::DeviceRef::cpu()).to(DType::F64).to_double_vector();
  return std::vector<double>(v.values().begin(), v.values().end());
}

struct OpCase {
  std::string name;
  std::vector<Input> inputs;
  std::function<ag::Variable(const std::vector<ag::Variable>&)> fn;
  bool scalar_out = false;
  bool parity_only = false;  // tasarim geregi sonlu farkla uyusmayan (padding_idx)
};

inline std::vector<OpCase> cases() {
  const Input a{randv(24, 1), {4, 6}}, b{randv(24, 2), {4, 6}}, pos{randv(24, 3, 0.5), {4, 6}}, bias{randv(6, 4), {6}};
  const Input w{randv(60, 5), {10, 6}};
  std::vector<OpCase> c;
  auto u = [&](std::string n, std::function<ag::Variable(const ag::Variable&)> f, Input in = {}) {
    c.push_back({std::move(n), {in.v.empty() ? a : in}, [f](const std::vector<ag::Variable>& x) { return f(x[0]); }});
  };
  u("neg", [](auto& x) { return ag::neg(x); });
  u("add_scalar", [](auto& x) { return ag::add_scalar(x, 0.7f); });
  u("mul_scalar", [](auto& x) { return ag::mul_scalar(x, -1.3f); });
  u("pow", [](auto& x) { return ag::pow(x, 2.5f); }, pos);
  u("relu", [](auto& x) { return ag::relu(x); });
  u("sigmoid", [](auto& x) { return ag::sigmoid(x); });
  u("tanh", [](auto& x) { return ag::tanh(x); });
  u("exp", [](auto& x) { return ag::exp(x); });
  u("log", [](auto& x) { return ag::log(x); }, pos);
  u("gelu", [](auto& x) { return ag::gelu(x); });
  u("silu", [](auto& x) { return ag::silu(x); });
  u("softmax", [](auto& x) { return ag::softmax(x); });
  u("log_softmax", [](auto& x) { return ag::log_softmax(x); });
  u("transpose", [](auto& x) { return ag::transpose(x); });
  u("reshape", [](auto& x) { return ag::reshape(x, ag::Shape{{3, 8}}); });
  u("narrow_d1", [](auto& x) { return ag::narrow(x, 1, 1, 3); });
  u("narrow_d0", [](auto& x) { return ag::narrow(x, 0, 2, 2); });
  u("dropout", [](auto& x) { return ag::dropout(x, 0.3f, true, 7); });
  c.push_back({"mean", {a}, [](auto& x) { return ag::mean(x[0]); }, true});
  c.push_back({"cross_entropy", {a}, [](auto& x) { return ag::cross_entropy(x[0], {5, 0, 3, 3}); }, true});
  c.push_back({"sub", {a, b}, [](auto& x) { return ag::sub(x[0], x[1]); }});
  c.push_back({"div", {a, pos}, [](auto& x) { return ag::div(x[0], x[1]); }});
  c.push_back({"add_bias", {a, bias}, [](auto& x) { return ag::add_bias(x[0], x[1]); }});
  c.push_back({"embedding", {w}, [](auto& x) { return ag::embedding(x[0], {3, 7, 3, 0, 9, 7, 7, 2}, -1); }});
  // padding satiri ileri geciste kullanilir ama gradyan almaz (FAZ 2 / PyTorch anlami): yalniz parity
  c.push_back({"embedding_pad", {w}, [](auto& x) { return ag::embedding(x[0], {3, 7, 3, 0, 9, 7, 7, 2}, 7); }, false, true});
  c.push_back({"cat_d0", {a, b}, [](auto& x) { return ag::cat({x[0], x[1]}, 0); }});
  c.push_back({"cat_d1", {a, b}, [](auto& x) { return ag::cat({x[0], x[1]}, 1); }});
  return c;
}

// Skaler kayip: ciktiyi sabit rastgele R ile carpip topla (gradyanlar bire esit olmasin)
inline ag::Variable loss_of(const ag::Variable& out, bool scalar_out, bool tensor, DType t) {
  if (scalar_out) return out;
  const auto n = static_cast<std::size_t>(out.numel());
  Input r{randv(n, 99), out.shape().dims};
  ag::Variable R = make_input(r, tensor, t, tensor && out.on_device());
  R = R.detach();
  return ag::sum(ag::mul(out, R));
}

inline double max_rel(const std::vector<double>& x, const std::vector<double>& y) {
  double e = 0, s = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    e = std::max(e, std::fabs(x[i] - y[i]));
    s = std::max(s, std::fabs(y[i]));
  }
  return s < 1e-12 ? e : e / s;
}

// ---- uctan uca: kucuk dil modeli -----------------------------------------------------------------
struct Lm {
  ag::Variable emb, w1, b1, w2, b2;
};
constexpr std::int64_t kV = 32, kD = 24, kH = 48, kT = 64;

inline std::vector<std::int64_t> markov_tokens(int step) {
  std::vector<std::int64_t> t(kT + 1);
  std::uint64_t st = 0x243F6A8885A308D3ULL + static_cast<std::uint64_t>(step) * 7919;
  auto nx = [&] {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return st >> 33;
  };
  t[0] = static_cast<std::int64_t>(nx() % kV);
  for (std::size_t i = 1; i < t.size(); ++i) t[i] = (t[i - 1] * 5 + 3 + static_cast<std::int64_t>(nx() % 3)) % kV;
  return t;
}

inline Lm init_lm(bool tensor, bool gpu = false) {
  auto mk = [&](std::int64_t r, std::int64_t c, unsigned seed, double scale) {
    Input in{randv(static_cast<std::size_t>(r * c), seed, 0.0, scale), c == 0 ? std::vector<std::int64_t>{r} : std::vector<std::int64_t>{r, c}};
    if (c == 0) in.v = std::vector<double>(static_cast<std::size_t>(r), 0.0);
    return make_input(in, tensor, DType::F32, gpu);
  };
  // bias: sifir (olcek 1 yalniz bos randv icin; normal_distribution sigma > 0 ister, Debug STL dogrular)
  return {mk(kV, kD, 41, 0.3), mk(kD, kH, 42, 0.2), mk(kH, 0, 0, 1), mk(kH, kV, 43, 0.2), mk(kV, 0, 0, 1)};
}

// amp: F32 ana agirliklar -> to(BF16) hesap; kayip F32
inline ag::Variable lm_loss(const Lm& m, const std::vector<std::int64_t>& tok, bool amp) {
  const std::vector<std::int64_t> in(tok.begin(), tok.end() - 1), tgt(tok.begin() + 1, tok.end());
  auto c = [&](const ag::Variable& v) { return amp ? ag::to(v, DType::BF16) : v; };
  ag::Variable x = ag::embedding(c(m.emb), in);
  ag::Variable h = ag::gelu(ag::add_bias(ag::matmul(x, c(m.w1)), c(m.b1)));
  ag::Variable logits = ag::add_bias(ag::matmul(h, c(m.w2)), c(m.b2));
  return ag::cross_entropy(logits, tgt);
}

inline std::vector<float> train_tensor(int steps, bool amp, bool gpu = false) {
  Lm m = init_lm(true, gpu);
  ag::TensorAdamW::Config cfg;
  cfg.lr = 3e-3;
  cfg.beta2 = 0.95;
  cfg.max_grad_norm = 1.0;
  ag::TensorAdamW opt({{m.emb, false}, {m.w1, false}, {m.b1, false}, {m.w2, false}, {m.b2, false}}, cfg);
  std::vector<float> losses;
  for (int s = 1; s <= steps; ++s) {
    opt.zero_grad();
    ag::Variable loss = lm_loss(m, markov_tokens(s), amp);
    loss.backward();
    losses.push_back(loss.tensor().to(nexus::MemorySpace::Host, nexus::DeviceRef::cpu()).to(DType::F32)[0]);
    opt.step();
  }
  return losses;
}


}  // namespace tac
