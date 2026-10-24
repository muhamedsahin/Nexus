/**
 * @file test_phase3_fused.cpp
 * @brief FAZ 3: füzyonlu çekirdeklerin paritesi ve sayisal gradyani.
 *
 * Her füzyonlu op, füzyonsuz bilesimle (autograd op'lari) birebir ayni ciktiyi
 * ve ayni gradyani uretmeli; ayrica sonlu farklarla dogrulanir. Füzyon yalniz
 * bellek trafigini azaltir, sayisal sonuc degistirmez.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/fused/fused.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace {

using nexus::autograd::Shape;
using nexus::autograd::Variable;
using namespace nexus::fused;

// fn: [n] yaprak -> skaler kayip. Sonlu farklarla analitik gradyani karsilastirir.
bool gradcheck(const std::function<Variable(const Variable&)>& fn, std::vector<float> x0, float step = 1.0e-3f) {
  const std::int64_t n = static_cast<std::int64_t>(x0.size());
  Variable leaf = Variable::make(x0, Shape{{n}}, true);
  Variable y = fn(leaf);
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

[[nodiscard]] float max_abs_diff(const Variable& a, const Variable& b) {
  const auto& av = a.values();
  const auto& bv = b.values();
  float worst = 0.0f;
  for (std::size_t i = 0; i < av.size(); ++i) {
    worst = std::fmax(worst, std::fabs(av[i] - bv[i]));
  }
  return worst;
}

[[nodiscard]] float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::fmax(worst, std::fabs(a[i] - b[i]));
  }
  return worst;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  nexus_test::section("fused_linear: parite ve gradyan");
  {
    const std::int64_t m = 2;
    const std::int64_t k = 3;
    const std::int64_t n = 2;
    const std::vector<float> xv = {0.5f, -1.0f, 0.2f, 1.5f, 0.0f, -0.7f};
    const std::vector<float> wv = {0.3f, -0.2f, 0.1f, 0.4f, 0.5f, -0.1f};
    const std::vector<float> bv = {0.1f, -0.3f};

    for (Activation act : {Activation::Identity, Activation::Relu, Activation::Gelu, Activation::Silu}) {
      // Füzyonlu
      Variable xf = Variable::make(xv, Shape{{m, k}}, true);
      Variable wf = Variable::make(wv, Shape{{k, n}}, true);
      Variable bf = Variable::make(bv, Shape{{n}}, true);
      Variable yf = fused_linear(xf, wf, bf, act);

      // Füzyonsuz
      Variable xu = Variable::make(xv, Shape{{m, k}}, true);
      Variable wu = Variable::make(wv, Shape{{k, n}}, true);
      Variable bu = Variable::make(bv, Shape{{n}}, true);
      Variable base = nexus::autograd::add_bias(nexus::autograd::matmul(xu, wu), bu);
      Variable yu = (act == Activation::Identity)   ? base
                    : (act == Activation::Relu)     ? nexus::autograd::relu(base)
                    : (act == Activation::Gelu)     ? nexus::autograd::gelu(base)
                                                    : nexus::autograd::silu(base);

      nexus_test::expect(max_abs_diff(yf, yu) < 1.0e-5f, "fused_linear cikti paritesi");

      nexus::autograd::sum(yf).backward();
      nexus::autograd::sum(yu).backward();
      nexus_test::expect(max_abs_diff(xf.grad().values(), xu.grad().values()) < 1.0e-4f, "fused_linear grad_x paritesi");
      nexus_test::expect(max_abs_diff(wf.grad().values(), wu.grad().values()) < 1.0e-4f, "fused_linear grad_w paritesi");
      nexus_test::expect(max_abs_diff(bf.grad().values(), bu.grad().values()) < 1.0e-4f, "fused_linear grad_b paritesi");
    }

    // Sayisal gradyan: x, w, b ayri ayri (digerleri sabit).
    const Variable wc = Variable::make(wv, Shape{{k, n}}, false);
    const Variable bc = Variable::make(bv, Shape{{n}}, false);
    bool ok = gradcheck(
        [wc, bc, m, k, n](const Variable& x) {
          return nexus::autograd::sum(fused_linear(nexus::autograd::reshape(x, Shape{{m, k}}), wc, bc, Activation::Gelu));
        },
        xv);
    nexus_test::expect(ok, "fused_linear grad_x sayisal");

    const Variable xc = Variable::make(xv, Shape{{m, k}}, false);
    ok = gradcheck(
        [xc, bc, k, n](const Variable& w) {
          return nexus::autograd::sum(fused_linear(xc, nexus::autograd::reshape(w, Shape{{k, n}}), bc, Activation::Gelu));
        },
        wv);
    nexus_test::expect(ok, "fused_linear grad_w sayisal");

    ok = gradcheck(
        [xc, wc, n](const Variable& b) { return nexus::autograd::sum(fused_linear(xc, wc, b, Activation::Gelu)); }, bv);
    nexus_test::expect(ok, "fused_linear grad_b sayisal");
  }

  nexus_test::section("fused_layernorm: parite ve gradyan");
  {
    const std::int64_t m = 2;
    const std::int64_t n = 4;
    const float eps = 1.0e-5f;
    const std::vector<float> xv = {1.0f, -2.0f, 3.0f, 0.5f, -1.5f, 0.0f, 2.5f, -0.5f};
    const std::vector<float> gv = {1.0f, 0.5f, 2.0f, 0.8f};
    const std::vector<float> bv = {0.0f, 0.1f, -0.2f, 0.3f};

    auto ref_ln = [&](const std::vector<float>& t) {
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
        for (std::int64_t j = 0; j < n; ++j) {
          out[static_cast<std::size_t>(i * n + j)] =
              (t[static_cast<std::size_t>(i * n + j)] - mu) * r * gv[static_cast<std::size_t>(j)] +
              bv[static_cast<std::size_t>(j)];
        }
      }
      return out;
    };

    Variable x = Variable::make(xv, Shape{{m, n}}, true);
    Variable g = Variable::make(gv, Shape{{n}}, true);
    Variable b = Variable::make(bv, Shape{{n}}, true);
    Variable y = fused_layernorm(x, g, b, eps);
    nexus_test::expect(max_abs_diff(y.values(), ref_ln(xv)) < 1.0e-5f, "fused_layernorm cikti paritesi");

    const Variable gc = Variable::make(gv, Shape{{n}}, false);
    const Variable bc = Variable::make(bv, Shape{{n}}, false);
    bool ok = gradcheck(
        [gc, bc, m, n, eps](const Variable& xx) {
          return nexus::autograd::sum(fused_layernorm(nexus::autograd::reshape(xx, Shape{{m, n}}), gc, bc, eps));
        },
        xv);
    nexus_test::expect(ok, "fused_layernorm grad_x sayisal");

    const Variable xc = Variable::make(xv, Shape{{m, n}}, false);
    ok = gradcheck([xc, bc, eps](const Variable& gg) { return nexus::autograd::sum(fused_layernorm(xc, gg, bc, eps)); },
                   gv);
    nexus_test::expect(ok, "fused_layernorm grad_gamma sayisal");

    ok = gradcheck([xc, gc, eps](const Variable& bb) { return nexus::autograd::sum(fused_layernorm(xc, gc, bb, eps)); },
                   bv);
    nexus_test::expect(ok, "fused_layernorm grad_beta sayisal");
  }

  nexus_test::section("fused_residual_layernorm: parite ve gradyan");
  {
    const std::int64_t m = 2;
    const std::int64_t n = 4;
    const float eps = 1.0e-5f;
    const std::vector<float> xv = {0.5f, -1.0f, 2.0f, 0.2f, -0.8f, 1.5f, -0.3f, 0.9f};
    const std::vector<float> rv = {0.1f, 0.2f, -0.1f, 0.3f, -0.2f, 0.0f, 0.4f, -0.1f};
    const std::vector<float> gv = {1.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<float> bv = {0.0f, 0.0f, 0.0f, 0.0f};

    // Referans: t = x + residual, sonra layernorm.
    std::vector<float> t(static_cast<std::size_t>(m * n));
    for (std::size_t i = 0; i < t.size(); ++i) {
      t[i] = xv[i] + rv[i];
    }
    // layernorm referansini basitce yeniden hesapla.
    auto ref_ln = [&](const std::vector<float>& inp) {
      std::vector<float> out(static_cast<std::size_t>(m * n));
      for (std::int64_t i = 0; i < m; ++i) {
        float mu = 0.0f;
        for (std::int64_t j = 0; j < n; ++j) {
          mu += inp[static_cast<std::size_t>(i * n + j)];
        }
        mu /= static_cast<float>(n);
        float var = 0.0f;
        for (std::int64_t j = 0; j < n; ++j) {
          const float d = inp[static_cast<std::size_t>(i * n + j)] - mu;
          var += d * d;
        }
        var /= static_cast<float>(n);
        const float r = 1.0f / std::sqrt(var + eps);
        for (std::int64_t j = 0; j < n; ++j) {
          out[static_cast<std::size_t>(i * n + j)] =
              (inp[static_cast<std::size_t>(i * n + j)] - mu) * r * gv[static_cast<std::size_t>(j)] +
              bv[static_cast<std::size_t>(j)];
        }
      }
      return out;
    };

    Variable x = Variable::make(xv, Shape{{m, n}}, true);
    Variable r = Variable::make(rv, Shape{{m, n}}, true);
    Variable g = Variable::make(gv, Shape{{n}}, true);
    Variable b = Variable::make(bv, Shape{{n}}, true);
    Variable y = fused_residual_layernorm(x, r, g, b, eps);
    nexus_test::expect(max_abs_diff(y.values(), ref_ln(t)) < 1.0e-5f, "fused_residual_layernorm cikti paritesi");

    const Variable rc = Variable::make(rv, Shape{{m, n}}, false);
    const Variable gc = Variable::make(gv, Shape{{n}}, false);
    const Variable bc = Variable::make(bv, Shape{{n}}, false);
    const bool ok = gradcheck(
        [rc, gc, bc, m, n, eps](const Variable& xx) {
          return nexus::autograd::sum(
              fused_residual_layernorm(nexus::autograd::reshape(xx, Shape{{m, n}}), rc, gc, bc, eps));
        },
        xv);
    nexus_test::expect(ok, "fused_residual_layernorm grad_x sayisal");
  }

  nexus_test::section("fused_softmax: parite ve gradyan");
  {
    const std::int64_t m = 2;
    const std::int64_t n = 3;
    const std::vector<float> xv = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
    const std::vector<float> wv = {0.5f, -0.2f, 1.0f, 0.3f, 0.8f, -0.4f};
    const Variable wc = Variable::make(wv, Shape{{m, n}}, false);

    Variable xf = Variable::make(xv, Shape{{m, n}}, true);
    Variable yf = fused_softmax(xf);
    Variable xu = Variable::make(xv, Shape{{m, n}}, true);
    Variable yu = nexus::autograd::softmax(xu);

    nexus_test::expect(max_abs_diff(yf, yu) < 1.0e-5f, "fused_softmax cikti paritesi");

    // Agirlikli toplam ile anlamli gradyan karsilastirmasi.
    nexus::autograd::sum(nexus::autograd::mul(yf, wc)).backward();
    nexus::autograd::sum(nexus::autograd::mul(yu, wc)).backward();
    nexus_test::expect(max_abs_diff(xf.grad().values(), xu.grad().values()) < 1.0e-4f, "fused_softmax grad paritesi");

    const bool ok = gradcheck(
        [wc, m, n](const Variable& x) {
          return nexus::autograd::sum(
              nexus::autograd::mul(fused_softmax(nexus::autograd::reshape(x, Shape{{m, n}})), wc));
        },
        xv);
    nexus_test::expect(ok, "fused_softmax sayisal gradyan");
  }

  return nexus_test::finish("faz3-füzyonlu-cekirdekler");
}
