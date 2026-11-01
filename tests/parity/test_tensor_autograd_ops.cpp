/**
 * @file test_tensor_autograd_ops.cpp
 * @brief Dalga 2: tum autograd op'larinin Tensor yolu + TensorAdamW + uctan uca egitim.
 *
 *   parity    : her op icin FAZ 2 vector yolu == Tensor yolu (F32): ileri + her girdinin gradyani
 *   gradcheck : ayni op'lar F64 Tensor yolunda sonlu farkla
 *   adamw     : TensorAdamW == fused::adamw_multi_tensor (F32 bit bit, kirpmali); BF16 parametre
 *   train     : embedding + matmul + add_bias + gelu + matmul + add_bias + cross_entropy dil modeli:
 *               FAZ 2 (vector + fused AdamW) vs Tensor F32 (TensorAdamW) vs BF16 AMP (F32 ana agirlik, BF16 hesap)
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/autograd/tensor_optim.hpp"
#include "nexus/core/tensor.hpp"
#include "nexus/core/tensor_ops.hpp"
#include "nexus/fused/training.hpp"
#include "nexus_test.hpp"
#include "tensor_autograd_cases.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace ag = nexus::autograd;
using nexus::DType;
using nexus::Shape;
using nexus::Tensor;

namespace {

using namespace tac;

void test_parity() {
  nexus_test::section("parity: FAZ 2 vector yolu == Tensor yolu (F32), ileri + gradyanlar");
  int bitwise = 0, total = 0;
  double worst = 0;
  std::string worst_op;
  for (const OpCase& oc : cases()) {
    std::vector<ag::Variable> xv, xt;
    for (const Input& in : oc.inputs) {
      xv.push_back(make_input(in, false, DType::F32));
      xt.push_back(make_input(in, true, DType::F32));
    }
    const ag::Variable ov = oc.fn(xv), ot = oc.fn(xt);
    const bool tensor_out = ot.is_tensor_backed();
    std::vector<double> fv = to_vec(ov), ft = to_vec(ot);
    double e = max_rel(ft, fv);
    loss_of(ov, oc.scalar_out, false, DType::F32).backward();
    loss_of(ot, oc.scalar_out, true, DType::F32).backward();
    bool bits = ft == fv;
    for (std::size_t i = 0; i < xv.size(); ++i) {
      const auto gv = to_vec(xv[i].grad()), gt = to_vec(xt[i].grad());
      e = std::max(e, max_rel(gt, gv));
      bits = bits && gv == gt;
      nexus_test::expect(xt[i].grad().is_tensor_backed(), oc.name + ": Tensor girdisinin gradyani Tensor");
    }
    ++total;
    bitwise += bits ? 1 : 0;
    if (e > worst) {
      worst = e;
      worst_op = oc.name;
    }
    nexus_test::expect(tensor_out && e <= 1e-6, oc.name + ": Tensor yolu == FAZ 2 (<= 1e-6)");
  }
  std::printf("  info  %d op: %d bit bit ayni; en kotu goreli fark %.2e (%s)\n", total, bitwise, worst, worst_op.c_str());
}

void test_gradcheck() {
  nexus_test::section("gradcheck: F64 Tensor yolu, sonlu fark");
  double worst = 0;
  std::string worst_op;
  for (const OpCase& oc : cases()) {
    if (oc.parity_only) continue;
    std::vector<ag::Variable> x;
    for (const Input& in : oc.inputs) x.push_back(make_input(in, true, DType::F64));
    loss_of(oc.fn(x), oc.scalar_out, true, DType::F64).backward();
    double e = 0;
    for (std::size_t k = 0; k < x.size(); ++k) {
      const auto g = to_vec(x[k].grad());
      Tensor base = x[k].tensor();
      for (std::size_t i = 0; i < base.numel(); ++i) {
        double* p = base.data_as<double>() + i;
        const double keep = *p, h = 1e-6;
        auto f = [&] {
          std::vector<ag::Variable> d;
          for (const auto& v : x) d.push_back(v.detach());
          return to_vec(loss_of(oc.fn(d), oc.scalar_out, true, DType::F64))[0];
        };
        *p = keep + h;
        const double fp = f();
        *p = keep - h;
        const double fm = f();
        *p = keep;
        e = std::max(e, std::fabs((fp - fm) / (2 * h) - g[i]) / std::max(1.0, std::fabs(g[i])));
      }
    }
    if (e > worst) {
      worst = e;
      worst_op = oc.name;
    }
    nexus_test::expect(e <= 1e-6, oc.name + ": F64 gradcheck <= 1e-6");
  }
  std::printf("  info  F64 sonlu fark en kotu %.2e (%s)\n", worst, worst_op.c_str());
}

void test_adamw() {
  nexus_test::section("adamw: TensorAdamW == fused::adamw_multi_tensor");
  const std::size_t n = 1000;
  const auto p0 = randv(n, 31), p1 = randv(97, 32);
  ag::TensorAdamW::Config cfg;
  cfg.lr = 3e-3;
  cfg.beta1 = 0.9;
  cfg.beta2 = 0.95;
  cfg.weight_decay = 0.1;
  cfg.max_grad_norm = 1.0;
  ag::Variable a = ag::Variable::wrap(Tensor::from_values(Shape{n}, p0, DType::F32), true);
  ag::Variable b = ag::Variable::wrap(Tensor::from_values(Shape{97}, p1, DType::F32), true);
  ag::TensorAdamW opt({{a, true}, {b, false}}, cfg);
  std::vector<float> ra(p0.begin(), p0.end()), rb(p1.begin(), p1.end()), ma(n), va(n), mb(97), vb(97);
  bool same = true;
  for (int step = 1; step <= 5; ++step) {
    const auto ga = randv(n, 100 + step, 0.0, 3.0), gb = randv(97, 200 + step, 0.0, 3.0);
    a.set_grad(ag::Variable::wrap(Tensor::from_values(Shape{n}, ga, DType::F32)));
    b.set_grad(ag::Variable::wrap(Tensor::from_values(Shape{97}, gb, DType::F32)));
    opt.step();
    std::vector<float> fa(ga.begin(), ga.end()), fb(gb.begin(), gb.end());
    nexus::fused::AdamWConfig fc;
    fc.lr = cfg.lr;
    fc.beta1 = cfg.beta1;
    fc.beta2 = cfg.beta2;
    fc.eps = cfg.eps;
    fc.max_grad_norm = cfg.max_grad_norm;
    // iki grup: decay'li a, decay'siz b; kirpma global (iki grubu birlikte) -> once norm, sonra iki cagri
    const nexus::fused::ParamSlot both[2] = {{ra.data(), fa.data(), ma.data(), va.data(), n}, {rb.data(), fb.data(), mb.data(), vb.data(), 97}};
    const double norm = nexus::fused::global_grad_norm(std::span<const nexus::fused::ParamSlot>(both, 2));
    const float clip = static_cast<float>(std::min(1.0, cfg.max_grad_norm / (norm + 1.0e-6)));
    for (auto& x : fa) x *= clip;
    for (auto& x : fb) x *= clip;
    fc.max_grad_norm = 0.0;
    fc.weight_decay = cfg.weight_decay;
    (void)nexus::fused::adamw_multi_tensor(std::span<const nexus::fused::ParamSlot>(&both[0], 1), fc, step);
    fc.weight_decay = 0.0;
    (void)nexus::fused::adamw_multi_tensor(std::span<const nexus::fused::ParamSlot>(&both[1], 1), fc, step);
    same = same && std::memcmp(a.tensor().data(), ra.data(), n * 4) == 0 && std::memcmp(b.tensor().data(), rb.data(), 97 * 4) == 0;
  }
  nexus_test::expect(same, "TensorAdamW F32 == fused::adamw_multi_tensor (5 adim, kirpma + decay gruplari) bit bit");
  // BF16 parametre: FP32 m/v, sonuc RNE; F32 yola yakin
  ag::Variable c = ag::Variable::wrap(Tensor::from_values(Shape{n}, p0, DType::BF16), true);
  ag::Variable cf = ag::Variable::wrap(Tensor::from_values(Shape{n}, p0, DType::F32), true);
  ag::TensorAdamW o1({{c, true}}, cfg), o2({{cf, true}}, cfg);
  for (int step = 1; step <= 5; ++step) {
    const auto g = randv(n, 300 + step);
    c.set_grad(ag::Variable::wrap(Tensor::from_values(Shape{n}, g, DType::BF16)));
    cf.set_grad(ag::Variable::wrap(Tensor::from_values(Shape{n}, g, DType::BF16).to(DType::F32)));
    o1.step();
    o2.step();
  }
  const double e = max_rel(c.tensor().to(DType::F64).to_double_vector(), cf.tensor().to(DType::F64).to_double_vector());
  std::printf("  info  BF16 parametreli AdamW F32'ye gore %.2e\n", e);
  nexus_test::expect(c.tensor().dtype() == DType::BF16 && e <= 1e-2, "BF16 parametre: dtype korunur, F32'ye <= 1e-2");
  nexus_test::expect([] {
    try {
      ag::Variable v = ag::Variable::make({1, 2}, ag::Shape{{2}}, true);
      ag::TensorAdamW bad({{v, true}}, {});
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  }(), "FAZ 2 vector parametresi acik hata");
}

std::vector<float> train_faz2(int steps) {
  Lm m = init_lm(false);
  std::vector<ag::Variable*> ps = {&m.emb, &m.w1, &m.b1, &m.w2, &m.b2};
  std::vector<std::vector<float>> mm, vv;
  for (auto* p : ps) {
    mm.emplace_back(p->values().size(), 0.0f);
    vv.emplace_back(p->values().size(), 0.0f);
  }
  std::vector<float> losses;
  for (int s = 1; s <= steps; ++s) {
    for (auto* p : ps) p->zero_grad();
    ag::Variable loss = lm_loss(m, markov_tokens(s), false);
    loss.backward();
    losses.push_back(loss.values()[0]);
    std::vector<std::vector<float>> grads;
    std::vector<nexus::fused::ParamSlot> slots;
    for (std::size_t i = 0; i < ps.size(); ++i) grads.push_back(ps[i]->grad().values());
    for (std::size_t i = 0; i < ps.size(); ++i)
      slots.push_back({ps[i]->mutable_values().data(), grads[i].data(), mm[i].data(), vv[i].data(), grads[i].size()});
    nexus::fused::AdamWConfig fc;
    fc.lr = 3e-3;
    fc.beta2 = 0.95;
    fc.max_grad_norm = 1.0;
    (void)nexus::fused::adamw_multi_tensor(slots, fc, s);
  }
  return losses;
}

void test_train() {
  nexus_test::section("train: uctan uca kucuk dil modeli (CPU)");
  const int steps = 60;
  const auto f2 = train_faz2(steps), tf = train_tensor(steps, false), tb = train_tensor(steps, true);
  double e32 = 0, e16 = 0;
  for (int s = 0; s < steps; ++s) {
    e32 = std::max(e32, std::fabs(static_cast<double>(tf[s]) - f2[s]) / f2[s]);
    e16 = std::max(e16, std::fabs(static_cast<double>(tb[s]) - f2[s]) / f2[s]);
  }
  std::printf("  info  kayip %.4f -> %.4f (FAZ 2) | Tensor F32 %.4f | BF16 AMP %.4f; en kotu adim farki F32 %.2e, BF16 %.2e\n", f2[0],
              f2.back(), tf.back(), tb.back(), e32, e16);
  nexus_test::expect(e32 <= 1e-5, "Tensor yolu F32 egitim egrisi == FAZ 2 (vector + fused AdamW), <= 1e-5");
  nexus_test::expect(e16 <= 3e-2 && tb.back() < 0.7f * tb[0], "BF16 AMP egitimi ogreniyor ve FAZ 2 egrisine <= 3e-2");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "parity";
  try {
    if (mode == "parity") test_parity();
    else if (mode == "gradcheck") test_gradcheck();
    else if (mode == "adamw") test_adamw();
    else if (mode == "train") test_train();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_tensor_autograd_ops");
}
