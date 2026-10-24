/**
 * @file test_phase2_engine.cpp
 * @brief FAZ 2 spesifikasyon testleri: tum op'larda gradcheck, saved-tensor
 *        kancalariyla offload, anomali modu, ara tensör kancalari, cok kok,
 *        ikinci backward denetimi ve CustomFunction ile klasik katman esitligi.
 *
 * promt.txt FAZ 2 TESTLER listesi:
 *  - test_gradcheck_all_ops            -> "gradcheck: tum op'lar"
 *  - test_inplace_detection            -> "yerinde degisiklik tespiti" (+ kanca yolunda)
 *  - test_saved_hooks_offload          -> "saved_tensors_hooks offload"
 *  - test_custom_function_matches_classic -> "CustomFunction = klasik Linear"
 *  - test_anomaly_reports_nan_node     -> "anomali modu"
 *  - test_multi_stream_backward (CUDA) -> bu makinede CPU derlemesi; kapsam disi, raporda yazili.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/autograd/module_function.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ag = nexus::autograd;
using ag::Shape;
using ag::Variable;

namespace {

Variable leaf(std::vector<float> values, Shape shape) { return Variable::make(std::move(values), std::move(shape), true); }

Variable constant(std::vector<float> values, Shape shape) {
  return Variable::make(std::move(values), std::move(shape), false);
}

void expect_gradcheck(const char* name, const std::function<Variable(const std::vector<Variable>&)>& fn,
                      const std::vector<Variable>& inputs) {
  const ag::GradcheckResult result = ag::gradcheck(fn, inputs);
  if (!result.ok) {
    std::printf("  info  %s max_abs=%.3g giris=%zu indeks=%zu\n", name, result.max_abs_error, result.worst_input,
                result.worst_index);
  }
  nexus_test::expect(result.ok, std::string("gradcheck ") + name);
}

template <typename Fn>
bool throws_with(Fn&& fn, const std::string& needle, std::string* message = nullptr) {
  try {
    fn();
  } catch (const std::exception& error) {
    if (message != nullptr) {
      *message = error.what();
    }
    return needle.empty() || std::string(error.what()).find(needle) != std::string::npos;
  }
  return false;
}

}  // namespace

int run() {

  nexus_test::section("gradcheck: tum op'lar");
  {
    const Variable a = leaf({0.6f, -1.2f, 0.9f, 1.4f, -0.3f, 0.8f}, Shape{{2, 3}});
    const Variable b = leaf({1.1f, 0.4f, -0.7f, 0.5f, 1.3f, -0.9f}, Shape{{2, 3}});
    const Variable pos = leaf({0.6f, 1.2f, 0.9f, 1.4f, 0.3f, 0.8f}, Shape{{2, 3}});
    const Variable w32 = leaf({0.2f, -0.5f, 0.7f, 0.1f, -0.3f, 0.4f}, Shape{{3, 2}});
    const Variable bias = leaf({0.1f, -0.2f, 0.3f}, Shape{{3}});
    const std::vector<std::int64_t> targets = {2, 0};

    expect_gradcheck("add", [](const auto& v) { return ag::sum(ag::mul(ag::add(v[0], v[1]), v[0])); }, {a, b});
    expect_gradcheck("sub", [](const auto& v) { return ag::sum(ag::mul(ag::sub(v[0], v[1]), v[1])); }, {a, b});
    expect_gradcheck("mul", [](const auto& v) { return ag::sum(ag::mul(v[0], v[1])); }, {a, b});
    expect_gradcheck("div", [](const auto& v) { return ag::sum(ag::div(v[0], v[1])); }, {a, pos});
    expect_gradcheck("neg", [](const auto& v) { return ag::sum(ag::mul(ag::neg(v[0]), v[0])); }, {a});
    expect_gradcheck("add_scalar/mul_scalar",
                     [](const auto& v) { return ag::sum(ag::mul(ag::add_scalar(v[0], 0.5f), ag::mul_scalar(v[0], 3.0f))); },
                     {a});
    expect_gradcheck("add_bias", [](const auto& v) { return ag::sum(ag::mul(ag::add_bias(v[0], v[1]), v[0])); },
                     {a, bias});
    expect_gradcheck("relu", [](const auto& v) { return ag::sum(ag::mul(ag::relu(v[0]), v[0])); }, {a});
    expect_gradcheck("gelu", [](const auto& v) { return ag::sum(ag::gelu(v[0])); }, {a});
    expect_gradcheck("silu", [](const auto& v) { return ag::sum(ag::silu(v[0])); }, {a});
    expect_gradcheck("sigmoid", [](const auto& v) { return ag::sum(ag::sigmoid(v[0])); }, {a});
    expect_gradcheck("tanh", [](const auto& v) { return ag::sum(ag::tanh(v[0])); }, {a});
    expect_gradcheck("exp", [](const auto& v) { return ag::sum(ag::exp(v[0])); }, {a});
    expect_gradcheck("log", [](const auto& v) { return ag::sum(ag::log(v[0])); }, {pos});
    expect_gradcheck("pow", [](const auto& v) { return ag::sum(ag::pow(v[0], 2.5f)); }, {pos});
    expect_gradcheck("mean", [](const auto& v) { return ag::mean(ag::mul(v[0], v[0])); }, {a});
    expect_gradcheck("reshape", [](const auto& v) { return ag::sum(ag::mul(ag::reshape(v[0], Shape{{3, 2}}), ag::reshape(v[1], Shape{{3, 2}}))); },
                     {a, b});
    expect_gradcheck("transpose", [](const auto& v) { return ag::sum(ag::matmul(ag::transpose(v[0]), v[1])); }, {a, b});
    expect_gradcheck("matmul", [](const auto& v) { return ag::sum(ag::tanh(ag::matmul(v[0], v[1]))); }, {a, w32});
    expect_gradcheck("softmax", [](const auto& v) { return ag::sum(ag::mul(ag::softmax(v[0]), v[1])); }, {a, b});
    expect_gradcheck("log_softmax", [](const auto& v) { return ag::sum(ag::mul(ag::log_softmax(v[0]), v[1])); },
                     {a, b});
    expect_gradcheck("cross_entropy", [targets](const auto& v) { return ag::cross_entropy(v[0], targets); }, {a});
    expect_gradcheck("dropout (sabit tohum)",
                     [](const auto& v) { return ag::sum(ag::mul(ag::dropout(v[0], 0.3f, true, 77), v[0])); }, {a});
    expect_gradcheck("embedding", [](const auto& v) { return ag::sum(ag::tanh(ag::embedding(v[0], {1, 0, 1, 2}))); },
                     {Variable::make({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}, Shape{{4, 2}}, true)});
    expect_gradcheck("narrow dim0/dim1",
                     [](const auto& v) {
                       return ag::sum(ag::mul(ag::narrow(v[0], 1, 1, 2), ag::narrow(ag::narrow(v[1], 0, 0, 2), 1, 0, 2)));
                     },
                     {a, b});
    expect_gradcheck("cat dim0/dim1",
                     [](const auto& v) {
                       const Variable rows = ag::cat({v[0], v[1]}, 0);   // [4,3]
                       const Variable cols = ag::cat({v[1], v[0]}, 1);   // [2,6]
                       return ag::add(ag::sum(ag::tanh(rows)), ag::sum(ag::mul(cols, cols)));
                     },
                     {a, b});
  }

  nexus_test::section("embedding padding_idx");
  {
    Variable table = Variable::make({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{{3, 2}}, true);
    const Variable out = ag::embedding(table, {0, 2, 2}, /*padding_idx=*/2);
    ag::sum(out).backward();
    const auto& g = table.grad().values();
    nexus_test::expect(g[0] == 1.0f && g[1] == 1.0f, "satir 0 gradyan alir");
    nexus_test::expect(g[4] == 0.0f && g[5] == 0.0f, "padding_idx satiri gradyan almaz");
    nexus_test::expect(throws_with([&] { (void)ag::embedding(table, {3}); }, "sozluk"), "sinir disi indeks hata");
  }

  nexus_test::section("saved_tensors_hooks offload");
  {
    const auto run = [](bool hooked, int* packs, int* unpacks) {
      Variable x = Variable::make({0.5f, -1.0f, 2.0f, 0.25f}, Shape{{2, 2}}, true);
      Variable w = Variable::make({1.5f, -0.5f, 0.25f, 2.0f}, Shape{{2, 2}}, true);
      std::unique_ptr<ag::SavedTensorsHooksGuard> guard;
      if (hooked) {
        guard = std::make_unique<ag::SavedTensorsHooksGuard>(
            [packs](const Variable& v) {
              // "CPU'ya bosalt": bagimsiz bir kopyaya paketle (ozgun tampona referans yok).
              ++*packs;
              auto store = std::make_shared<std::pair<std::vector<float>, Shape>>(v.values(), v.shape());
              return std::static_pointer_cast<void>(store);
            },
            [unpacks](std::shared_ptr<void>&& packed) {
              ++*unpacks;
              auto store = std::static_pointer_cast<std::pair<std::vector<float>, Shape>>(packed);
              return Variable::make(store->first, store->second, false);
            });
      }
      const Variable y = ag::sum(ag::tanh(ag::matmul(ag::mul(x, x), w)));
      guard.reset();  // kancalar ileri geçiste etkili; geri geçiste paketler okunur
      y.backward();
      std::vector<float> grads = x.grad().values();
      grads.insert(grads.end(), w.grad().values().begin(), w.grad().values().end());
      return grads;
    };
    int packs = 0;
    int unpacks = 0;
    const std::vector<float> plain = run(false, &packs, &unpacks);
    nexus_test::expect(packs == 0 && unpacks == 0, "kancasiz yol paketlemez");
    const std::vector<float> hooked = run(true, &packs, &unpacks);
    nexus_test::expect(packs == 4, "mul(2) + matmul(2) kaydi pack'ten gecti");
    nexus_test::expect(unpacks == 4, "geri geçis her kaydi unpack etti");
    nexus_test::expect(nexus_test::identical<float>(plain, hooked), "offload gradyanlari kancasizla bit-bit ayni");

    // Kanca yolunda da yerinde yazma yakalanir (ozgun ust-veri uzerinden).
    Variable x = Variable::make({1.0f, 2.0f}, Shape{{2}}, true);
    Variable y;
    {
      ag::SavedTensorsHooksGuard guard(
          [](const Variable& v) { return std::static_pointer_cast<void>(std::make_shared<Variable>(v.detach())); },
          [](std::shared_ptr<void>&& packed) { return *std::static_pointer_cast<Variable>(packed); });
      y = ag::sum(ag::mul(x, x));
    }
    x.fill_(3.0f);
    nexus_test::expect(throws_with([&] { y.backward(); }, "yerinde"), "kanca yolunda yerinde degisiklik tespiti");
    nexus_test::expect(throws_with([] { ag::set_saved_tensors_hooks([](const Variable&) { return std::shared_ptr<void>(); }, {}); },
                                   "birlikte"),
                       "yalniz pack verilirse hata");
  }

  nexus_test::section("anomali modu");
  {
    // d/dx x^0.5 = 0.5 x^-0.5 -> x=0'da Inf. Anomali kapaliyken sessiz, acikken dugum adiyla hata.
    {
      Variable x = Variable::make({0.0f, 1.0f}, Shape{{2}}, true);
      const Variable y = ag::sum(ag::pow(x, 0.5f));
      y.backward();
      nexus_test::expect(!std::isfinite(x.grad().values()[0]), "anomali kapali: Inf gradyan sessizce akar");
    }
    {
      ag::AnomalyModeGuard anomaly(true);
      Variable x = Variable::make({0.0f, 1.0f}, Shape{{2}}, true);
      const Variable y = ag::sum(ag::pow(x, 0.5f));
      std::string message;
      const bool reported = throws_with([&] { y.backward(); }, "PowBackward", &message);
      nexus_test::expect(reported, "anomali NaN/Inf ureten dugumu adiyla raporlar");
      nexus_test::expect(message.find("ileri geçis") != std::string::npos, "anomali ileri geçis baglamini verir");
    }
    nexus_test::expect(!ag::anomaly_mode(), "AnomalyModeGuard onceki durumu geri koyar");
  }

  nexus_test::section("ara tensör kancasi ve retain_grad");
  {
    Variable x = Variable::make({1.0f, 2.0f}, Shape{{2}}, true);
    Variable h = ag::mul_scalar(x, 3.0f);  // ara tensör
    h.retain_grad();
    int fired = 0;
    h.register_hook([&fired](const Variable& g) {
      ++fired;
      return ag::mul_scalar(g, 2.0f);
    });
    const Variable y = ag::sum(ag::mul(h, h));  // dy/dh = 2h = [6, 12]
    y.backward();
    nexus_test::expect(fired == 1, "ara tensör kancasi bir kez ateslenir");
    nexus_test::expect(std::fabs(h.grad().values()[0] - 12.0f) < 1.0e-5f, "retain_grad kanca sonrasi gradyani tutar");
    nexus_test::expect(std::fabs(x.grad().values()[1] - 72.0f) < 1.0e-4f, "kanca asagi akisi olcekler (12*2*3)");
  }

  nexus_test::section("cok kok ve ikinci backward");
  {
    Variable x = Variable::make({2.0f}, Shape{{1}}, true);
    const Variable y1 = ag::mul(x, x);           // x^2
    const Variable y2 = ag::mul_scalar(y1, 3.0f);  // 3x^2, y1'e bagli
    ag::backward({y1, y2}, {Variable::ones(Shape{{1}}, false), Variable::ones(Shape{{1}}, false)}, false);
    nexus_test::expect(std::fabs(x.grad().values()[0] - 16.0f) < 1.0e-5f, "kokler birlikte: 2x + 6x = 16 (kok bir kez)");

    Variable z = Variable::make({1.0f}, Shape{{1}}, true);
    const Variable out = ag::mul_scalar(ag::mul(z, z), 2.0f);
    out.backward();
    nexus_test::expect(throws_with([&] { out.backward(); }, "ikinci kez"), "retain_graph olmadan ikinci backward hata");
  }

  nexus_test::section("grad modu ve detach");
  {
    Variable x = Variable::make({1.0f, 2.0f}, Shape{{2}}, true);
    {
      ag::NoGradGuard no_grad;
      const Variable y = ag::exp(x);
      nexus_test::expect(!y.requires_grad() && !y.grad_fn(), "NoGrad: dugum yok");
    }
    const Variable d = x.detach();
    nexus_test::expect(!d.requires_grad() && d.values().data() == x.values().data(), "detach veriyi paylasir, grafiksiz");
  }

  nexus_test::section("CustomFunction = klasik Linear");
  {
    // Ayni agirlikli iki klasik Linear: biri dogrudan forward/backward, digeri autograd icinde.
    nexus_model::Linear classic(3, 2, true);
    nexus_model::Linear wrapped(3, 2, true);
    auto cp = classic.parameters();
    auto wp = wrapped.parameters();
    for (std::size_t i = 0; i < cp.size(); ++i) {
      for (std::size_t k = 0; k < cp[i]->data.numel(); ++k) {
        cp[i]->data[k] = 0.1f * static_cast<float>(k + 1) - 0.25f * static_cast<float>(i);
        wp[i]->data[k] = cp[i]->data[k];
      }
      cp[i]->zero_grad();
      wp[i]->zero_grad();
    }
    const std::vector<float> xs = {0.5f, -1.0f, 2.0f, 1.5f, 0.25f, -0.75f};

    // Klasik: L = sum(tanh(Linear(x))) -> dL/dy = 1 - tanh^2.
    nexus_model::Tensor xt = nexus_model::Tensor::zeros({2, 3});
    std::copy(xs.begin(), xs.end(), xt.data());
    const nexus_model::Tensor yt = classic.forward(xt);
    nexus_model::Tensor gy = nexus_model::Tensor::zeros({2, 2});
    for (std::size_t k = 0; k < yt.numel(); ++k) {
      const float t = std::tanh(yt.data()[k]);
      gy.data()[k] = 1.0f - t * t;
    }
    const nexus_model::Tensor dx_classic = classic.backward(gy);

    // Autograd: ayni zincir, Linear bir CustomFunction dugumu.
    Variable x = Variable::make(xs, Shape{{2, 3}}, true);
    const Variable y = ag::module_function(wrapped, x, "LinearFunction");
    nexus_test::expect(y.grad_fn() && y.grad_fn()->name == "LinearFunction", "CustomFunction dugumu teybe girdi");
    ag::sum(ag::tanh(y)).backward();

    bool same_dx = true;
    for (std::size_t k = 0; k < xs.size(); ++k) {
      same_dx = same_dx && x.grad().values()[k] == dx_classic.data()[k];
    }
    nexus_test::expect(same_dx, "dx klasik backward ile bit-bit ayni");
    bool same_params = true;
    for (std::size_t i = 0; i < cp.size(); ++i) {
      for (std::size_t k = 0; k < cp[i]->grad.numel(); ++k) {
        same_params = same_params && cp[i]->grad[k] == wp[i]->grad[k];
      }
    }
    nexus_test::expect(same_params, "parametre gradyanlari klasik yolla bit-bit ayni");

    // Autograd matmul formulasyonuyla capraz kontrol (W [out,in] -> x W^T + b).
    Variable xa = Variable::make(xs, Shape{{2, 3}}, true);
    std::vector<float> wt(6);
    for (int o = 0; o < 2; ++o) {
      for (int in = 0; in < 3; ++in) {
        wt[static_cast<std::size_t>(in * 2 + o)] = cp[0]->data[static_cast<std::size_t>(o * 3 + in)];
      }
    }
    const Variable wv = Variable::make(wt, Shape{{3, 2}}, false);
    const Variable bv = Variable::make({cp[1]->data[0], cp[1]->data[1]}, Shape{{2}}, false);
    ag::sum(ag::tanh(ag::add_bias(ag::matmul(xa, wv), bv))).backward();
    float worst = 0.0f;
    for (std::size_t k = 0; k < xs.size(); ++k) {
      worst = std::fmax(worst, std::fabs(xa.grad().values()[k] - x.grad().values()[k]));
    }
    nexus_test::info("custom_vs_autograd_max_abs", worst);
    nexus_test::expect(worst < 1.0e-5f, "CustomFunction autograd op bilesimiyle ayni dx (<1e-5)");
  }

  nexus_test::section("custom_function sozlesmesi");
  {
    Variable x = Variable::make({1.0f, 2.0f}, Shape{{2}}, true);
    const Variable bad = ag::custom_function("BadFunction", {x}, {1.0f, 2.0f}, Shape{{2}},
                                             [](const Variable&) { return std::vector<Variable>{}; });
    nexus_test::expect(throws_with([&] { ag::sum(bad).backward(); }, "her giris"), "eksik gradyan listesi hata");
    nexus_test::expect(throws_with([&] { (void)ag::custom_function("X", {x}, {1.0f}, Shape{{2}}, [](const Variable& g) {
                                       return std::vector<Variable>{g};
                                     }); },
                                   "boyutu"),
                       "cikti/sekil uyusmazligi hata");
  }

  return nexus_test::finish("faz2-motor-spesifikasyon");
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    return run();
  } catch (const std::exception& error) {
    std::printf("  FAIL  beklenmeyen istisna: %s\n", error.what());
    return 1;
  }
}
