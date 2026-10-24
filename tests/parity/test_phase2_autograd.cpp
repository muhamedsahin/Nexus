/**
 * @file test_phase2_autograd.cpp
 * @brief FAZ 2: genel tensör autograd motoru sozlesmeleri.
 *
 * Dogrulananlar:
 *  - Grad modu: set_grad_enabled / NoGradGuard / EnableGradGuard.
 *  - Temel backward: d/dx x^2 = 2x.
 *  - Sayisal gradyan: sonlu farklar, op turevlerini dogrular (mul, relu,
 *    sigmoid, tanh, exp, log, gelu, silu, pow, mean, matmul, cross_entropy,
 *    add_bias, add).
 *  - Yaprak gradyan birikimi (AccumulateGrad +=) iki backward'ta.
 *  - SavedTensor sürüm denetimi: yerinde degisiklik geri geçiste hata verir.
 *  - Grad kancasi (register_hook) gradyani donusturur.
 *  - MLP: kucuk cok katmanli model loss dusurur (motor uctan uca calisir).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <exception>
#include <cstdio>
#include <functional>
#include <vector>

namespace {

using nexus::autograd::Shape;
using nexus::autograd::Variable;
using nexus::autograd::add;
using nexus::autograd::add_bias;
using nexus::autograd::cross_entropy;
using nexus::autograd::exp;
using nexus::autograd::gelu;
using nexus::autograd::log;
using nexus::autograd::matmul;
using nexus::autograd::mean;
using nexus::autograd::mul;
using nexus::autograd::mul_scalar;
using nexus::autograd::pow;
using nexus::autograd::relu;
using nexus::autograd::reshape;
using nexus::autograd::sigmoid;
using nexus::autograd::silu;
using nexus::autograd::softmax;
using nexus::autograd::sum;
using nexus::autograd::tanh;

// fn: [n] yaprak -> skaler kayip. Sonlu farklarla analitik gradyani karsilastirir.
bool gradcheck(const std::function<Variable(const Variable&)>& fn, std::vector<float> x0, float step = 1.0e-3f) {
  const std::int64_t n = static_cast<std::int64_t>(x0.size());
  Variable leaf = Variable::make(x0, Shape{{n}}, true);
  Variable y = fn(leaf);
  if (y.numel() != 1) {
    std::printf("  info  gradcheck skaler kayip bekleniyordu\n");
    return false;
  }
  y.backward();
  const auto& analytic = leaf.grad().values();
  for (std::int64_t i = 0; i < n; ++i) {
    const float original = x0[static_cast<std::size_t>(i)];
    x0[static_cast<std::size_t>(i)] = original + step;
    const float yp = fn(Variable::make(x0, Shape{{n}}, false)).values()[0];
    x0[static_cast<std::size_t>(i)] = original - step;
    const float ym = fn(Variable::make(x0, Shape{{n}}, false)).values()[0];
    x0[static_cast<std::size_t>(i)] = original;
    const float numeric = (yp - ym) / (2.0f * step);
    const float tolerance = 2.0e-2f * (1.0f + std::fabs(numeric));
    if (std::fabs(numeric - analytic[static_cast<std::size_t>(i)]) > tolerance) {
      std::printf("  info  gradcheck hata i=%lld num=%.6f ana=%.6f\n", static_cast<long long>(i), numeric,
                  analytic[static_cast<std::size_t>(i)]);
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  nexus_test::section("grad modu");
  {
    Variable x = Variable::make({3.0f}, Shape{{1}}, true);
    nexus::autograd::set_grad_enabled(false);
    Variable off = mul_scalar(x, 2.0f);
    nexus_test::expect(!off.requires_grad(), "set_grad_enabled(false) gradyan uretmez");
    nexus::autograd::set_grad_enabled(true);
    {
      nexus::autograd::NoGradGuard guard;
      Variable inside = mul_scalar(x, 2.0f);
      nexus_test::expect(!inside.requires_grad(), "NoGradGuard gradyan uretmez");
    }
    {
      nexus::autograd::EnableGradGuard guard;
      Variable inside = mul_scalar(x, 2.0f);
      nexus_test::expect(inside.requires_grad(), "EnableGradGuard gradyan uretir");
    }
  }

  nexus_test::section("temel backward");
  {
    Variable x = Variable::make({2.0f}, Shape{{1}}, true);
    Variable y = mul(x, x);
    nexus_test::expect(std::fabs(y.values()[0] - 4.0f) < 1.0e-6f, "y = x^2 = 4");
    y.backward();
    nexus_test::expect(std::fabs(x.grad().values()[0] - 4.0f) < 1.0e-6f, "dy/dx = 2x = 4");
  }

  nexus_test::section("yaprak gradyan birikimi");
  {
    Variable x = Variable::make({3.0f}, Shape{{1}}, true);
    Variable y = mul_scalar(x, 2.0f);
    const Variable ones = Variable::ones(Shape{{1}}, false);
    y.backward(ones, /*retain_graph=*/true);
    y.backward(ones, /*retain_graph=*/true);
    nexus_test::expect(std::fabs(x.grad().values()[0] - 4.0f) < 1.0e-6f, "grad += birikir (2 + 2)");
  }

  nexus_test::section("sayisal gradyan (sonlu farklar)");
  {
    bool ok = true;
    auto check = [&ok](const char* name, const std::function<Variable(const Variable&)>& fn, std::vector<float> x0) {
      const bool pass = gradcheck(fn, x0);
      nexus_test::expect(pass, name);
      ok = ok && pass;
    };
    check("mul gradyani", [](const Variable& x) { return sum(mul(x, x)); }, {0.5f, 1.5f, -1.0f, 2.0f});
    check("relu gradyani", [](const Variable& x) { return sum(relu(x)); }, {-1.0f, 0.5f, 2.0f, -0.5f});
    check("sigmoid gradyani", [](const Variable& x) { return sum(sigmoid(x)); }, {0.0f, 1.0f, -1.0f, 0.5f});
    check("tanh gradyani", [](const Variable& x) { return sum(tanh(x)); }, {0.0f, 1.0f, -1.0f, 0.5f});
    check("exp gradyani", [](const Variable& x) { return sum(exp(x)); }, {-0.5f, 0.0f, 0.5f, 1.0f});
    check("log gradyani", [](const Variable& x) { return sum(log(x)); }, {0.5f, 1.0f, 1.5f, 2.0f});
    check("gelu gradyani", [](const Variable& x) { return sum(gelu(x)); }, {-1.0f, 0.0f, 0.5f, 1.0f});
    check("silu gradyani", [](const Variable& x) { return sum(silu(x)); }, {-1.0f, 0.0f, 0.5f, 1.0f});
    check("pow gradyani", [](const Variable& x) { return sum(pow(x, 3.0f)); }, {0.5f, 1.0f, 1.5f, 2.0f});
    check("mean gradyani", [](const Variable& x) { return mean(x); }, {1.0f, 2.0f, 3.0f, 4.0f});
    check(
        "add gradyani", [](const Variable& x) { return sum(add(x, Variable::full(x.shape(), 2.0f, false))); },
        {1.0f, 2.0f, 3.0f, 4.0f});

    const Variable b = Variable::make({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, Shape{{3, 2}}, false);
    check("matmul gradyani", [b](const Variable& x) { return sum(matmul(reshape(x, Shape{{2, 3}}), b)); },
          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    const std::vector<std::int64_t> targets = {1};
    check("cross_entropy gradyani",
          [targets](const Variable& x) { return cross_entropy(reshape(x, Shape{{1, 3}}), targets); },
          {0.5f, -0.2f, 0.1f});

    const Variable bias = Variable::make({0.1f, 0.2f, 0.3f}, Shape{{3}}, false);
    check("add_bias gradyani", [bias](const Variable& x) { return sum(add_bias(reshape(x, Shape{{2, 3}}), bias)); },
          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  }

  nexus_test::section("kaydedilmis sürüm denetimi");
  {
    Variable x = Variable::make({2.0f, 3.0f}, Shape{{2}}, true);
    Variable y = mul(x, x);
    x.fill_(5.0f);  // yerinde degisiklik sürümü artirir
    bool threw = false;
    try {
      y.backward();
    } catch (const std::exception&) {
      threw = true;
    }
    nexus_test::expect(threw, "yerinde degistirilen kaydedilmis tensör hata verir");
  }

  nexus_test::section("grad kancasi");
  {
    Variable x = Variable::make({3.0f}, Shape{{1}}, true);
    Variable y = mul_scalar(x, 2.0f);
    x.register_hook([](const Variable& g) { return mul_scalar(g, 10.0f); });
    y.backward();
    nexus_test::expect(std::fabs(x.grad().values()[0] - 20.0f) < 1.0e-6f, "hook gradyani olcekler (2*10)");
  }

  nexus_test::section("yerinde kisit");
  {
    Variable x = Variable::make({1.0f}, Shape{{1}}, true);
    Variable y = mul_scalar(x, 2.0f);
    bool threw = false;
    try {
      y.add_(Variable::ones(Shape{{1}}, false));
    } catch (const std::exception&) {
      threw = true;
    }
    nexus_test::expect(threw, "yaprak olmayan tensör yerinde degistirilemez");
  }

  nexus_test::section("mlp egitimi (motor uctan uca)");
  {
    const std::int64_t in = 4;
    const std::int64_t hid = 8;
    const std::int64_t out = 3;
    Variable w1 = Variable::randn(Shape{{in, hid}}, 11, true);
    Variable b1 = Variable::zeros(Shape{{hid}}, true);
    Variable w2 = Variable::randn(Shape{{hid, out}}, 22, true);
    Variable b2 = Variable::zeros(Shape{{out}}, true);

    constexpr int kSamples = 16;
    std::vector<Variable> inputs;
    std::vector<std::int64_t> labels;
    for (int s = 0; s < kSamples; ++s) {
      inputs.push_back(Variable::randn(Shape{{1, in}}, 100 + s, false));
      labels.push_back(s % out);
    }

    const float lr = 0.05f;
    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 60; ++epoch) {
      float epoch_loss = 0.0f;
      for (int s = 0; s < kSamples; ++s) {
        const Variable h = relu(add_bias(matmul(inputs[s], w1), b1));
        const Variable logits = add_bias(matmul(h, w2), b2);
        const Variable loss = cross_entropy(logits, {labels[s]});
        loss.backward();
        epoch_loss += loss.values()[0];
        w1.add_(mul_scalar(w1.grad(), -lr));
        w1.zero_grad();
        b1.add_(mul_scalar(b1.grad(), -lr));
        b1.zero_grad();
        w2.add_(mul_scalar(w2.grad(), -lr));
        w2.zero_grad();
        b2.add_(mul_scalar(b2.grad(), -lr));
        b2.zero_grad();
      }
      if (epoch == 0) {
        first_loss = epoch_loss / static_cast<float>(kSamples);
      }
      last_loss = epoch_loss / static_cast<float>(kSamples);
    }
    std::printf("  info  mlp loss %.4f -> %.4f\n", first_loss, last_loss);
    nexus_test::expect(last_loss < first_loss, "mlp kaybi dusuyor");
  }

  return nexus_test::finish("faz2-autograd");
}
