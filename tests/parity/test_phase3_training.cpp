/**
 * @file test_phase3_training.cpp
 * @brief FAZ 3 egitim yolu testleri (CPU): kaydirilmis CE, RMSNorm, gomme+konum,
 *        Philox dropout ve foreach AdamW.
 *
 * promt.txt FAZ 3 TESTLER listesi:
 *  - Her cekirdek FP64 referansina karsi fwd/bwd; sekil taramasi (V tek, V=50257
 *    hizasiz, B=1, T=2). T=1 kaydirilmis kayipta hedef birakmaz -> acik hata.
 *  - test_ce_shift_no_copy           -> ledger + heap sayaci: logit boyutunda ikinci tampon yok
 *  - test_adamw_fused_vs_reference   -> 1000 adim, NexusOptim CPU AdamW ile <= 1e-6
 *  - test_dropout_reproducible       -> ayni seed/offset ayni maske; fwd/bwd tutarli
 *  - test_gpt2s_parity_fp32          -> GPU egitim yolu gerektirir; bu CPU testinin kapsami disi
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/memory.hpp"
#include "nexus/fused/training.hpp"
#include "nexus/prof/memory_ledger.hpp"
#include "nexus_optim/algorithms/adamw.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace ag = nexus::autograd;
namespace fu = nexus::fused;
using ag::Shape;
using ag::Variable;

namespace {

std::vector<float> random_vector(std::size_t n, std::uint64_t seed, float scale = 1.0f) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<float> out(n);
  for (float& value : out) {
    value = dist(rng);
  }
  return out;
}

struct Reference {
  double loss = 0.0;
  std::vector<double> grad;
  std::int64_t valid = 0;
};

// Kopyalarla kurulan acik FP64 referansi: logits[:, :-1] ile tokens[:, 1:].
Reference reference_ce(const std::vector<float>& logits, const std::vector<std::int32_t>& tokens, int batch, int time,
                       int vocab, const fu::CrossEntropyOptions& options) {
  Reference ref;
  ref.grad.assign(logits.size(), 0.0);
  std::vector<std::size_t> rows;
  std::vector<std::int32_t> targets;
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t + 1 < time; ++t) {
      const std::int32_t target = tokens[static_cast<std::size_t>(b * time + t + 1)];
      if (target == options.ignore_index) {
        continue;
      }
      rows.push_back(static_cast<std::size_t>(b * time + t));
      targets.push_back(target);
    }
  }
  ref.valid = static_cast<std::int64_t>(rows.size());
  const double eps = options.label_smoothing;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const float* row = logits.data() + rows[r] * static_cast<std::size_t>(vocab);
    double maxv = row[0];
    for (int j = 1; j < vocab; ++j) {
      maxv = std::max(maxv, static_cast<double>(row[j]));
    }
    double total = 0.0;
    for (int j = 0; j < vocab; ++j) {
      total += std::exp(static_cast<double>(row[j]) - maxv);
    }
    const double lse = maxv + std::log(total);
    double row_loss = 0.0;
    for (int j = 0; j < vocab; ++j) {
      const double q = (j == targets[r] ? 1.0 - eps : 0.0) + eps / vocab;
      row_loss += -q * (static_cast<double>(row[j]) - lse);
    }
    row_loss += options.z_loss * lse * lse;
    ref.loss += row_loss;
    for (int j = 0; j < vocab; ++j) {
      const double p = std::exp(static_cast<double>(row[j]) - lse);
      const double q = (j == targets[r] ? 1.0 - eps : 0.0) + eps / vocab;
      ref.grad[rows[r] * static_cast<std::size_t>(vocab) + static_cast<std::size_t>(j)] =
          (p - q + 2.0 * options.z_loss * lse * p) / static_cast<double>(rows.size());
    }
  }
  ref.loss /= static_cast<double>(rows.size());
  return ref;
}

void check_ce_case(const char* name, int batch, int time, int vocab, const fu::CrossEntropyOptions& options,
                   bool with_ignore) {
  const auto logits = random_vector(static_cast<std::size_t>(batch * time * vocab), 1000u + static_cast<unsigned>(vocab),
                                    2.0f);
  std::vector<std::int32_t> tokens(static_cast<std::size_t>(batch * time));
  std::mt19937 rng(static_cast<unsigned>(vocab));
  for (auto& token : tokens) {
    token = static_cast<std::int32_t>(rng() % static_cast<unsigned>(vocab));
  }
  if (with_ignore && tokens.size() > 2) {
    tokens[1] = options.ignore_index;
  }
  const Reference ref = reference_ce(logits, tokens, batch, time, vocab, options);
  std::vector<float> grad(logits.size(), -1.0f);
  const fu::CrossEntropyResult got =
      fu::cross_entropy_shifted(logits.data(), tokens.data(), batch, time, vocab, options, grad.data());
  double worst = 0.0;
  for (std::size_t i = 0; i < grad.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(grad[i]) - ref.grad[i]));
  }
  const double loss_rel = std::fabs(static_cast<double>(got.loss) - ref.loss) / std::max(1.0, std::fabs(ref.loss));
  if (worst >= 1.0e-6 || loss_rel >= 1.0e-6) {
    std::printf("  info  %s loss_rel=%.3g grad_max_abs=%.3g\n", name, loss_rel, worst);
  }
  nexus_test::expect(got.valid == ref.valid, std::string(name) + ": gecerli token sayisi");
  nexus_test::expect(loss_rel < 1.0e-6, std::string(name) + ": kayip FP64 referansla (<1e-6 bagil)");
  nexus_test::expect(worst < 1.0e-6, std::string(name) + ": dlogits FP64 referansla (<1e-6)");
}

template <typename Fn>
bool throws(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

int run() {
  nexus_test::section("cross_entropy_shifted: FP64 referans ve sekil taramasi");
  {
    check_ce_case("B=2 T=5 V=7 (tek V)", 2, 5, 7, {}, false);
    check_ce_case("B=1 T=2 V=50257 (hizasiz)", 1, 2, 50257, {}, false);
    check_ce_case("B=3 T=4 V=33 ignore_index", 3, 4, 33, {}, true);
    fu::CrossEntropyOptions smooth;
    smooth.label_smoothing = 0.1f;
    check_ce_case("label_smoothing=0.1", 2, 6, 19, smooth, false);
    fu::CrossEntropyOptions z;
    z.z_loss = 1.0e-4f;
    check_ce_case("z_loss=1e-4", 2, 6, 19, z, true);
    nexus_test::expect(throws([] {
                         std::vector<float> l(4, 0.0f);
                         std::vector<std::int32_t> t(2, 0);
                         (void)fu::cross_entropy_shifted(l.data(), t.data(), 2, 1, 2, {}, nullptr);
                       }),
                       "T=1: kaydirilmis kayipta hedef yok -> acik hata");
    nexus_test::expect(throws([] {
                         std::vector<float> l(4, 0.0f);
                         std::vector<std::int32_t> t = {0, 5};
                         (void)fu::cross_entropy_shifted(l.data(), t.data(), 1, 2, 2, {}, nullptr);
                       }),
                       "sozluk disi hedef -> acik hata");
  }

  nexus_test::section("test_ce_shift_no_copy (yerinde, ikinci logit tamponu yok)");
  {
    const int batch = 4;
    const int time = 16;
    const int vocab = 4099;
    const std::size_t count = static_cast<std::size_t>(batch * time * vocab);
    const auto source = random_vector(count, 77, 1.5f);
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(batch * time));
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = static_cast<std::int32_t>((i * 131u) % static_cast<std::size_t>(vocab));
    }
    // Ayri tamponlu sonuc (karsilastirma icin).
    std::vector<float> separate(count);
    const auto expected = fu::cross_entropy_shifted(source.data(), tokens.data(), batch, time, vocab, {}, separate.data());

    auto& ledger = nexus::prof::ledger();
    ledger.reset_all();
    float* logits = nullptr;
    {
      nexus::prof::AllocationLabelScope scope(nexus::prof::MemoryLabel::Activation);
      logits = static_cast<float*>(nexus::allocate_aligned(count * sizeof(float)));
    }
    std::copy(source.begin(), source.end(), logits);
    const std::int64_t peak_before = ledger.peak_bytes();
    fu::CrossEntropyResult inplace;
    std::uint64_t heap_allocs = 0;
    {
      nexus_test::AllocationScope heap;
      inplace = fu::cross_entropy_shifted(logits, tokens.data(), batch, time, vocab, {}, logits);
      heap_allocs = heap.count();
    }
    const std::int64_t peak_after = ledger.peak_bytes();
    nexus_test::info("ce_logit_bytes", static_cast<double>(count * sizeof(float)));
    nexus_test::info("ce_ledger_peak_delta", static_cast<double>(peak_after - peak_before));
    nexus_test::expect(peak_after == peak_before, "ledger: CE sirasinda tepe bellek artmadi");
    nexus_test::expect(heap_allocs == 0, "CE sicak yolunda heap ayirma yok (kaydirma kopya degil ofset)");
    nexus_test::expect(inplace.loss == expected.loss, "yerinde kayip ayri tamponla bit-bit ayni");
    nexus_test::expect(std::equal(separate.begin(), separate.end(), logits), "yerinde dlogits ayri tamponla bit-bit ayni");
    nexus::release_owned(logits);
  }

  nexus_test::section("cross_entropy_shifted autograd");
  {
    const int batch = 2;
    const int time = 4;
    const int vocab = 5;
    std::vector<std::int32_t> tokens = {1, 3, 0, 4, 2, 2, 1, 0};
    const Variable logits = Variable::make(random_vector(static_cast<std::size_t>(batch * time * vocab), 5),
                                           Shape{{batch * time, vocab}}, true);
    fu::CrossEntropyOptions options;
    options.label_smoothing = 0.05f;
    options.z_loss = 1.0e-3f;
    const auto result = ag::gradcheck(
        [&](const std::vector<Variable>& v) { return fu::cross_entropy_shifted(v[0], tokens, batch, time, options); },
        {logits});
    nexus_test::expect(result.ok, "gradcheck cross_entropy_shifted (smoothing + z-loss)");

    // Duz CE (secenek yok) = autograd::cross_entropy(kaydirilmis satirlar).
    Variable a = Variable::make(std::vector<float>(logits.values()), logits.shape(), true);
    const Variable fused_loss = fu::cross_entropy_shifted(a, tokens, batch, time);
    fused_loss.backward();
    Variable b = Variable::make(std::vector<float>(logits.values()), logits.shape(), true);
    std::vector<Variable> rows;
    std::vector<std::int64_t> targets;
    for (int bb = 0; bb < batch; ++bb) {
      rows.push_back(ag::narrow(b, 0, bb * time, time - 1));
      for (int t = 0; t + 1 < time; ++t) {
        targets.push_back(tokens[static_cast<std::size_t>(bb * time + t + 1)]);
      }
    }
    const Variable plain_loss = ag::cross_entropy(ag::cat(rows, 0), targets);
    plain_loss.backward();
    float worst = std::fabs(fused_loss.values()[0] - plain_loss.values()[0]);
    for (std::size_t i = 0; i < a.grad().values().size(); ++i) {
      worst = std::max(worst, std::fabs(a.grad().values()[i] - b.grad().values()[i]));
    }
    nexus_test::expect(worst < 1.0e-6f, "fuzyonlu CE = narrow+cat+cross_entropy bilesimi (<1e-6)");
  }

  nexus_test::section("rms_norm");
  {
    const Variable x = Variable::make(random_vector(12, 9), Shape{{3, 4}}, true);
    const Variable g = Variable::make({1.0f, 0.5f, -1.5f, 2.0f}, Shape{{4}}, true);
    const Variable y = fu::rms_norm(x, g, 1.0e-6f);
    double worst = 0.0;
    for (int i = 0; i < 3; ++i) {
      double sq = 0.0;
      for (int j = 0; j < 4; ++j) {
        sq += static_cast<double>(x.values()[static_cast<std::size_t>(i * 4 + j)]) * x.values()[static_cast<std::size_t>(i * 4 + j)];
      }
      const double r = 1.0 / std::sqrt(sq / 4.0 + 1.0e-6);
      for (int j = 0; j < 4; ++j) {
        const std::size_t k = static_cast<std::size_t>(i * 4 + j);
        worst = std::max(worst, std::fabs(y.values()[k] - x.values()[k] * r * g.values()[static_cast<std::size_t>(j)]));
      }
    }
    nexus_test::expect(worst < 1.0e-6, "rms_norm ileri FP64 referansla");
    const auto check = ag::gradcheck(
        [](const std::vector<Variable>& v) { return ag::sum(ag::tanh(fu::rms_norm(v[0], v[1], 1.0e-6f))); }, {x, g});
    nexus_test::expect(check.ok, "gradcheck rms_norm (x ve gamma)");
  }

  nexus_test::section("embedding_pos (wte[tok] + wpe[pos])");
  {
    const Variable wte = Variable::make(random_vector(5 * 3, 11), Shape{{5, 3}}, true);
    const Variable wpe = Variable::make(random_vector(4 * 3, 12), Shape{{4, 3}}, true);
    const std::vector<std::int32_t> tokens = {4, 0, 4, 1, 2, 2};
    const Variable out = fu::embedding_pos(wte, wpe, tokens, 2, 3);
    bool same = true;
    for (int r = 0; r < 6; ++r) {
      for (int j = 0; j < 3; ++j) {
        const float expected = wte.values()[static_cast<std::size_t>(tokens[static_cast<std::size_t>(r)] * 3 + j)] +
                               wpe.values()[static_cast<std::size_t>((r % 3) * 3 + j)];
        same = same && out.values()[static_cast<std::size_t>(r * 3 + j)] == expected;
      }
    }
    nexus_test::expect(same, "ileri: wte[tok] + wpe[pos] bit-bit");
    const auto check = ag::gradcheck(
        [&tokens](const std::vector<Variable>& v) { return ag::sum(ag::tanh(fu::embedding_pos(v[0], v[1], tokens, 2, 3))); },
        {wte, wpe});
    nexus_test::expect(check.ok, "gradcheck embedding_pos (iki tablo)");
  }

  nexus_test::section("test_dropout_reproducible (Philox4x32-10)");
  {
    // Random123 kat_vectors bilinen cevaplari.
    const auto k0 = fu::philox4x32_10({0u, 0u, 0u, 0u}, {0u, 0u});
    nexus_test::expect(k0[0] == 0x6627e8d5u && k0[1] == 0xe169c58du && k0[2] == 0xbc57ac4cu && k0[3] == 0x9b00dbd8u,
                       "Philox KAT: sifir sayac/anahtar");
    const auto k1 = fu::philox4x32_10({0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu});
    nexus_test::expect(k1[0] == 0x408f276du && k1[1] == 0x41c83b0eu && k1[2] == 0xa20bc7c6u && k1[3] == 0x6d5451fdu,
                       "Philox KAT: tum bitler 1");
    const auto k2 = fu::philox4x32_10({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u});
    nexus_test::expect(k2[0] == 0xd16cfe09u && k2[1] == 0x94fdccebu && k2[2] == 0x5001e420u && k2[3] == 0x24126ea1u,
                       "Philox KAT: pi rakamlari");

    const std::size_t n = 100000;
    const Variable x = Variable::make(std::vector<float>(n, 1.0f), Shape{{static_cast<std::int64_t>(n)}}, true);
    const Variable a = fu::dropout_philox(x, 0.3f, 1234, 0);
    const Variable b = fu::dropout_philox(x, 0.3f, 1234, 0);
    const Variable c = fu::dropout_philox(x, 0.3f, 1234, 1u << 20);
    nexus_test::expect(a.values() == b.values(), "ayni (seed, offset) -> ayni maske");
    nexus_test::expect(a.values() != c.values(), "farkli offset -> farkli maske");
    const auto dropped = static_cast<double>(std::count(a.values().begin(), a.values().end(), 0.0f));
    nexus_test::info("dropout_fraction", dropped / static_cast<double>(n));
    nexus_test::expect(std::fabs(dropped / static_cast<double>(n) - 0.3) < 0.01, "dusurulen oran ~p (|fark| < 0.01)");
    ag::sum(a).backward();
    nexus_test::expect(x.grad().values() == a.values(), "geri geçis maskeyi yeniden uretir (grad == cikti, x=1)");
    nexus_test::expect(a.grad_fn() && a.grad_fn()->name == "DropoutPhiloxBackward", "maske saklanmaz: Philox dugumu");
    const Variable eval = fu::dropout_philox(x, 0.3f, 1234, 0, false);
    nexus_test::expect(eval.values() == x.values(), "eval modunda ozdeslik");
  }

  nexus_test::section("test_adamw_fused_vs_reference (1000 adim)");
  {
    const std::vector<std::size_t> sizes = {7, 64, 129, 1};
    nexus_optim::ParamGroupOptions group_options;
    group_options.learning_rate = 1.0e-3;
    // Referans iki yolda kosulur: NexusOptim skaler (IEEE FP32 bolme) yolu ve bu
    // makinedeki varsayilan AVX2 yolu (bolme yerine _mm256_rcp_ps + bir Newton adimi).
    for (const bool scalar_reference : {true, false}) {
    nexus_optim::force_simd_level(scalar_reference ? std::optional<nexus_optim::SimdLevel>(nexus_optim::SimdLevel::Scalar)
                                                   : std::nullopt);
    for (const float max_norm : {0.0f, 0.5f}) {
      std::vector<std::vector<float>> ref_p;
      std::vector<std::vector<float>> ref_g;
      std::vector<std::vector<float>> fus_p;
      std::vector<std::vector<float>> fus_g;
      std::vector<std::vector<float>> m;
      std::vector<std::vector<float>> v;
      for (std::size_t t = 0; t < sizes.size(); ++t) {
        ref_p.push_back(random_vector(sizes[t], 300 + t));
        fus_p.push_back(ref_p.back());
        ref_g.emplace_back(sizes[t], 0.0f);
        fus_g.emplace_back(sizes[t], 0.0f);
        m.emplace_back(sizes[t], 0.0f);
        v.emplace_back(sizes[t], 0.0f);
      }
      std::vector<float*> pp;
      std::vector<float*> gp;
      for (std::size_t t = 0; t < sizes.size(); ++t) {
        pp.push_back(ref_p[t].data());
        gp.push_back(ref_g[t].data());
      }
      nexus_optim::AdamW<float>::Options native;
      native.lr = 1.0e-3;
      native.weight_decay = 0.01;
      nexus_optim::AdamW<float> reference(
          std::vector<nexus_optim::ParamGroup<float>>{nexus_optim::make_group<float>(pp, gp, sizes, group_options)},
          native);
      std::vector<fu::ParamSlot> slots;
      for (std::size_t t = 0; t < sizes.size(); ++t) {
        slots.push_back({fus_p[t].data(), fus_g[t].data(), m[t].data(), v[t].data(), sizes[t]});
      }
      fu::AdamWConfig config;
      config.lr = 1.0e-3;
      config.weight_decay = 0.01;
      config.max_grad_norm = max_norm;
      double worst = 0.0;
      bool clipped_seen = false;
      for (int step = 1; step <= 1000; ++step) {
        for (std::size_t t = 0; t < sizes.size(); ++t) {
          for (std::size_t i = 0; i < sizes[t]; ++i) {
            // Deterministik, parametreye bagli gradyan (kuadratik kayip + sinus gurultu).
            const float g = 0.3f * fus_p[t][i] + 0.05f * std::sin(static_cast<float>(step * 7 + i * 3 + t));
            fus_g[t][i] = g;
            ref_g[t][i] = 0.3f * ref_p[t][i] + 0.05f * std::sin(static_cast<float>(step * 7 + i * 3 + t));
          }
        }
        if (max_norm > 0.0f) {
          // Referans: once global norm kirpma (PyTorch clip_grad_norm_), sonra NexusOptim adimi.
          double total = 0.0;
          for (const auto& g : ref_g) {
            for (const float value : g) {
              total += static_cast<double>(value) * value;
            }
          }
          const float coef = static_cast<float>(std::min(1.0, max_norm / (std::sqrt(total) + 1.0e-6)));
          clipped_seen = clipped_seen || coef < 1.0f;
          for (auto& g : ref_g) {
            for (float& value : g) {
              value *= coef;
            }
          }
        }
        reference.step();
        (void)fu::adamw_multi_tensor(slots, config, step);
      }
      for (std::size_t t = 0; t < sizes.size(); ++t) {
        for (std::size_t i = 0; i < sizes[t]; ++i) {
          // Bagil hata (|ref| >= 1'de bagil, altinda mutlak): NexusOptim AVX2/FMA yolu ile
          // skaler FP32 yolu arasinda ULP duzeyi farklar 1000 adimda birikir.
          worst = std::max(worst, std::fabs(static_cast<double>(fus_p[t][i]) - ref_p[t][i]) /
                                      std::max(1.0, std::fabs(static_cast<double>(ref_p[t][i]))));
        }
      }
      const std::string tag = std::string(scalar_reference ? " [skaler ref]" : " [AVX2 ref]") +
                              (max_norm > 0.0f ? " + global norm kirpma" : "");
      nexus_test::info(std::string("adamw_max_rel") + (scalar_reference ? "_scalar" : "_avx2") +
                           (max_norm > 0.0f ? "_clip" : ""),
                       worst);
      if (scalar_reference) {
        nexus_test::expect(worst <= 1.0e-6, "foreach AdamW NexusOptim ile 1000 adim <= 1e-6" + tag);
      } else {
        // rcp_ps+NR ~1 ULP bolme hatasi gradyan geri beslemesiyle birikir; yine de <= 1e-6.
        nexus_test::expect(worst <= 1.0e-6, "foreach AdamW NexusOptim AVX2 yaklasik bolmesiyle <= 1e-6" + tag);
      }
      if (max_norm > 0.0f) {
        nexus_test::expect(clipped_seen, "kirpma gercekten devreye girdi" + tag);
      }
    }
    }
    nexus_optim::force_simd_level(std::nullopt);
    nexus_test::expect(throws([] { (void)fu::adamw_multi_tensor({}, {}, 0); }), "step=0 acik hata");
  }

  return nexus_test::finish("faz3-egitim-yolu-cpu");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    return run();
  } catch (const std::exception& error) {
    std::printf("  FAIL  beklenmeyen istisna: %s\n", error.what());
    return 1;
  }
}
