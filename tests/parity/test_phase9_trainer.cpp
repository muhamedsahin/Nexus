/**
 * @file test_phase9_trainer.cpp
 * @brief FAZ 9: dil modeli egitmeni v2 (CPU yolu, nn::Module arka ucu).
 *
 *   test_phase9_trainer old_api   : eski LanguageTrainer(model, lr, wd).fit kayiplari FAZ 9 oncesi
 *                                   koddan alinan altin degerlerle bit bit ayni
 *   test_phase9_trainer accum     : micro_batch=2 x accum=4 == micro_batch=8 x accum=1 (FP32) gradyan
 *   test_phase9_trainer schedule  : WarmupCosine = nanoGPT get_lr; isinmasiz kosinus = NexusOptim
 *                                   CosineAnnealingLR; WSD/Linear/InverseSqrt sinir degerleri
 *   test_phase9_trainer groups    : sifir gradyanla tek adim: yalniz sonumlu gruplar (1 - lr*wd*s) kuculur
 *   test_phase9_trainer nan       : atla / durdur / geri don; bozuk agirlikta ilk NaN katmani adiyla
 *   test_phase9_trainer spike     : hareketli medyan + k*MAD sicramasi ornek kimlikleriyle kaydedilir
 *   test_phase9_trainer resume    : 3 adim + kayit + yeni surec(model) + 3 adim == 6 adim (bit bit)
 *   test_phase9_trainer eval      : sabit dogrulama kumesi; coktan secmeli puan = -sum(token kaybi)
 *   test_phase9_trainer loggers   : CSV, JSONL, W&B JSONL, TensorBoard (CRC denetimli geri okuma)
 *   test_phase9_trainer classifier: siniflandirma Trainer'i ortak altyapida (biriktirme, adim zamanlayici)
 *   test_phase9_trainer golden    : (arac) eski API kayiplarini yazdirir
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/train/language.hpp"
#include "nexus/train/lm_trainer.hpp"
#include "nexus/train/loggers.hpp"
#include "nexus/train/trainer.hpp"
#include "nexus_optim/scheduler/cosine_annealing_lr.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace nt = nexus::train;

namespace {

constexpr int kVocab = 11;
constexpr int kEmbed = 16;

nexus::nn::Sequential nano_model() {
  nexus::nn::Sequential model;
  model.add(std::make_shared<nexus::nn::Embedding>(kVocab, kEmbed, -1));
  model.add(std::make_shared<nexus::nn::PositionalEncoding>(kEmbed, 32, false));
  model.add(std::make_shared<nexus::nn::CausalTransformerBlock>(kEmbed, 2, 32));
  model.add(std::make_shared<nexus::nn::Linear>(static_cast<std::size_t>(kEmbed), static_cast<std::size_t>(kVocab), true));
  return model;
}

std::vector<std::int32_t> lcg_tokens(std::size_t n, std::uint32_t seed) {
  std::vector<std::int32_t> v(n);
  std::uint32_t s = seed;
  for (std::int32_t& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<std::int32_t>((s >> 8) % kVocab);
  }
  return v;
}

nexus::TokenRows nano_rows(int batch, int time, std::uint32_t seed) {
  nexus::TokenRows rows;
  rows.batch = batch;
  rows.time = time;
  const std::vector<std::int32_t> v = lcg_tokens(static_cast<std::size_t>(batch * time), seed);
  rows.values.assign(v.begin(), v.end());
  return rows;
}

/// Adim basina farkli (durumlu, devam ettirilebilir) yigin: satir r'nin tokenlari (cagri, r)'den.
class CountingSource final : public nt::TokenBatchSource {
 public:
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
    for (int r = 0; r < rows; ++r) {
      const std::uint64_t id = index_++;
      const std::vector<std::int32_t> row = lcg_tokens(static_cast<std::size_t>(time), static_cast<std::uint32_t>(id * 2654435761u + 17u));
      std::memcpy(dst + static_cast<std::size_t>(r) * time, row.data(), row.size() * 4);
      if (ids != nullptr) {
        ids[r] = id;
      }
    }
  }
  std::string state() const override { return std::to_string(index_); }
  void restore(const std::string& s) override { index_ = std::stoull(s); }

 private:
  std::uint64_t index_ = 0;
};

std::vector<std::vector<float>> grads_of(const nexus::nn::Module& model) {
  std::vector<std::vector<float>> out;
  for (nexus_model::Parameter* p : model.parameters()) {
    out.emplace_back(p->grad.data(), p->grad.data() + p->grad.numel());
  }
  return out;
}

std::vector<std::vector<float>> weights_of(const nexus::nn::Module& model) {
  std::vector<std::vector<float>> out;
  for (nexus_model::Parameter* p : model.parameters()) {
    out.emplace_back(p->data.data(), p->data.data() + p->data.numel());
  }
  return out;
}

/// nano_model() her cagrida kuresel RNG'den farkli baslar: karsilastirmali testler ayni agirlikla baslar.
void copy_weights(const nexus::nn::Module& from, nexus::nn::Module& to) {
  const auto src = from.parameters();
  const auto dst = to.parameters();
  for (std::size_t i = 0; i < src.size(); ++i) {
    std::memcpy(dst[i]->data.data(), src[i]->data.data(), src[i]->data.numel() * sizeof(float));
  }
}

double max_rel_diff(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t k = 0; k < a[i].size(); ++k) {
      const double d = std::fabs(static_cast<double>(a[i][k]) - b[i][k]);
      worst = std::max(worst, d / (1.0e-6 + std::fabs(static_cast<double>(b[i][k]))));
    }
  }
  return worst;
}

nt::SpikeOptions no_spike() {
  nt::SpikeOptions s;
  s.enabled = false;
  return s;
}

// ------------------------------------------------------------------------------------------

int test_old_api(bool print) {
  nexus_test::section("test_old_api_still_works: nano LanguageTrainer altin kayiplar");
  nexus::nn::Sequential model = nano_model();
  nexus::LanguageTrainer trainer(model, 0.02f, 0.01f);
  const nexus::TokenRows rows = nano_rows(3, 12, 7u);
  std::vector<float> losses;
  for (int i = 0; i < 5; ++i) {
    losses.push_back(trainer.fit(rows, 1));
  }
  losses.push_back(trainer.fit(rows, 10));
  if (print) {
    for (float v : losses) {
      std::printf("%.9g\n", static_cast<double>(v));
    }
    return 0;
  }
  // FAZ 9 oncesi language.hpp (7a63578) ile ayni model/veri: build-phase2-cpu, "golden" modu.
  const float golden[] = {3.57453966f, 2.56375265f, 2.07644224f, 1.81453419f, 1.65796411f, 0.693540752f};
  bool same = true;
  for (std::size_t i = 0; i < losses.size(); ++i) {
    same = same && losses[i] == golden[i];
    nexus_test::info("loss", losses[i]);
  }
  nexus_test::expect(same, "eski API kayiplari FAZ 9 oncesiyle bit bit ayni");
  nexus_test::expect(trainer.v2() != nullptr && trainer.v2()->step() == 15, "15 fit epoch'u = 15 optimizer adimi");
  return 0;
}

int test_accum() {
  nexus_test::section("test_accum_equivalence: micro 2 x accum 4 == micro 8 x accum 1 (FP32, CPU)");
  const int T = 12;
  const std::vector<std::int32_t> rows = lcg_tokens(8 * T, 99u);
  struct Run {
    std::vector<std::vector<float>> grads, weights;
    double loss = 0.0;
  };
  const nexus::nn::Sequential reference = nano_model();
  auto run = [&](int micro, int accum) {
    Run out;
    nexus::nn::Sequential model = nano_model();
    copy_weights(reference, model);
    nt::FixedRowsSource source(rows, 8, T);
    auto trainer = nt::LanguageTrainerBuilder(model)
                       .data(source)
                       .optimizer(nt::AdamW{1.0e-2, 0.9, 0.95, 1.0e-8, 0.1})
                       .micro_batch(micro)
                       .seq_len(T)
                       .global_batch_tokens(8 * T)
                       .grad_clip(1.0)
                       .spike_detector(no_spike())
                       .log_console(false)
                       .after_backward([&](nt::LmBackend&, std::int64_t) { out.grads = grads_of(model); })
                       .build();
    nexus_test::expect(trainer->accum_steps() == accum, "accum_steps = global / (micro * T)");
    out.loss = trainer->train(1).last_loss;
    out.weights = weights_of(model);
    return out;
  };
  const Run a = run(2, 4);
  const Run b = run(8, 1);
  const double g = max_rel_diff(a.grads, b.grads);
  const double w = max_rel_diff(a.weights, b.weights);
  nexus_test::info("grad_max_rel_diff", g);
  nexus_test::info("weight_max_rel_diff_after_step", w);
  nexus_test::info("loss_a", a.loss);
  nexus_test::info("loss_b", b.loss);
  nexus_test::expect(g < 1.0e-4, "biriktirilmis gradyan == tek buyuk yigin gradyani (rel < 1e-4)");
  nexus_test::expect(w < 1.0e-5, "adim sonrasi agirliklar ayni (rel < 1e-5)");
  nexus_test::expect(std::fabs(a.loss - b.loss) < 1.0e-5, "token agirlikli adim kaybi ayni");
  return 0;
}

int test_schedule() {
  nexus_test::section("test_schedule_values");
  // nanoGPT train.py get_lr (birebir kopya, carpan olarak).
  auto nanogpt = [](std::int64_t it, std::int64_t warmup, std::int64_t decay_iters, double min_ratio) {
    if (it < warmup) {
      return static_cast<double>(it + 1) / static_cast<double>(warmup + 1);
    }
    if (it > decay_iters) {
      return min_ratio;
    }
    const double r = static_cast<double>(it - warmup) / static_cast<double>(decay_iters - warmup);
    const double coeff = 0.5 * (1.0 + std::cos(3.14159265358979323846 * r));
    return min_ratio + coeff * (1.0 - min_ratio);
  };
  const nt::LrSchedule cos = nt::WarmupCosine{100, 1000, 0.1};
  double worst = 0.0;
  for (std::int64_t s = 0; s <= 1200; ++s) {
    worst = std::max(worst, std::fabs(cos.factor(s) - nanogpt(s, 100, 1000, 0.1)));
  }
  nexus_test::info("warmup_cosine_vs_nanogpt_max_abs", worst);
  nexus_test::expect(worst == 0.0, "WarmupCosine == nanoGPT get_lr (0..1200)");
  nexus_test::expect(cos.factor(0) == 1.0 / 101.0 && cos.factor(100) == 1.0 && cos.factor(1000) == 0.1, "kosinus sinirlari");

  struct DummyGroup {
    struct {
      double learning_rate = 1.0;
    } options;
  };
  struct DummyOpt {
    std::vector<DummyGroup> groups{DummyGroup{}};
    std::vector<DummyGroup>& param_groups() { return groups; }
  } dummy;
  nexus_optim::CosineAnnealingLR ref(dummy, 50, 0.2);
  const nt::LrSchedule cos0 = nt::WarmupCosine{0, 50, 0.2};
  double worst_ref = 0.0;
  for (std::int64_t s = 0; s <= 50; ++s) {
    ref.step();  // k. cagri epoch k-1 degerini verir
    worst_ref = std::max(worst_ref, std::fabs(ref.get_last_lr() - cos0.factor(s)));
  }
  nexus_test::info("cosine_vs_nexusoptim_max_abs", worst_ref);
  nexus_test::expect(worst_ref < 1.0e-15, "isinmasiz kosinus == NexusOptim CosineAnnealingLR");

  const nt::LrSchedule wsd = nt::WSD{10, 80, 10, 0.0, false};
  nexus_test::expect(wsd.factor(9) == 10.0 / 11.0 && wsd.factor(10) == 1.0 && wsd.factor(89) == 1.0, "WSD isinma/sabit");
  nexus_test::expect(std::fabs(wsd.factor(90) - 0.9) < 1e-12 && wsd.factor(99) == 0.0 && wsd.factor(500) == 0.0, "WSD lineer inis");
  const nt::LrSchedule wsd_sqrt = nt::WSD{0, 0, 4, 0.0, true};
  nexus_test::expect(std::fabs(wsd_sqrt.factor(0) - (1.0 - std::sqrt(0.25))) < 1e-12, "WSD 1-sqrt inis");
  const nt::LrSchedule lin = nt::LinearLr{0, 10, 0.0};
  nexus_test::expect(lin.factor(0) == 1.0 && std::fabs(lin.factor(5) - 0.5) < 1e-12 && lin.factor(10) == 0.0, "lineer");
  const nt::LrSchedule inv = nt::InverseSqrt{3};
  nexus_test::expect(inv.factor(2) == 0.75 && inv.factor(3) == 1.0 && std::fabs(inv.factor(15) - 0.5) < 1e-12, "ters karekok");
  const nt::LrSchedule cst = nt::ConstantLr{4};
  nexus_test::expect(cst.factor(0) == 0.2 && cst.factor(4) == 1.0 && cst.factor(10000) == 1.0, "sabit + isinma");
  bool threw = false;
  try {
    const nt::LrSchedule bad = nt::WarmupCosine{10, 10, 0.1};
    (void)bad;
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  nexus_test::expect(threw, "max_steps <= warmup reddedilir");
  return 0;
}

int test_groups() {
  nexus_test::section("test_param_groups_wd: sifir gradyanla tek adim");
  const double lr = 0.1;
  const double wd = 0.5;
  nexus::nn::Sequential model = nano_model();
  nt::FixedRowsSource source(lcg_tokens(2 * 8, 3u), 2, 8);
  nt::ParamGroupPolicy groups;
  groups.decay_embeddings = false;
  groups.rules.push_back(nt::ParamGroupRule{"3.", 2.0f, -1.0f});  // kafa: LR x2 (muP tarzi)
  auto trainer = nt::LanguageTrainerBuilder(model)
                     .data(source)
                     .optimizer(nt::AdamW{lr, 0.9, 0.95, 1.0e-8, wd}, groups)
                     .micro_batch(2)
                     .seq_len(8)
                     .spike_detector(no_spike())
                     .log_console(false)
                     .after_backward([&](nt::LmBackend&, std::int64_t) {
                       for (nexus_model::Parameter* p : model.parameters()) {
                         p->grad.zero();  // AdamW guncellemesi 0; yalniz sonum kalir
                       }
                     })
                     .build();
  const std::vector<nt::LmParamInfo> info = trainer->backend().parameters();
  const auto before = weights_of(model);
  trainer->train(1);
  const auto after = weights_of(model);
  bool ok = true;
  int decayed = 0;
  int kept = 0;
  for (std::size_t i = 0; i < info.size(); ++i) {
    const bool decay = info[i].rank >= 2 && !info[i].embedding;
    const double s = info[i].name.rfind("3.", 0) == 0 ? 2.0 : 1.0;
    const float factor = decay ? static_cast<float>(1.0 - lr * s * wd) : 1.0f;
    double worst = 0.0;
    for (std::size_t k = 0; k < before[i].size(); ++k) {
      worst = std::max(worst, std::fabs(static_cast<double>(after[i][k]) - static_cast<double>(before[i][k]) * factor));
    }
    ok = ok && worst < 1.0e-6;
    (decay ? decayed : kept) += 1;
    std::printf("  %-28s rank=%d emb=%d decay=%d lr_scale=%.0f max_err=%.2e\n", info[i].name.c_str(), info[i].rank,
                info[i].embedding ? 1 : 0, decay ? 1 : 0, s, worst);
  }
  nexus_test::expect(decayed > 0 && kept > 0, "iki grup da dolu");
  nexus_test::expect(ok, "sonum yalniz >=2B, gomme disi agirliklarda; grup LR carpani uygulanir");
  return 0;
}

int test_nan() {
  nexus_test::section("test_nan_policy: atla / durdur / geri don");
  auto make = [](nexus::nn::Sequential& model, nt::TokenBatchSource& src, nt::NanOptions nan,
                 std::function<void(nt::LmBackend&, std::int64_t)> hook) {
    return nt::LanguageTrainerBuilder(model)
        .data(src)
        .optimizer(nt::AdamW{1.0e-2, 0.9, 0.95, 1.0e-8, 0.0})
        .micro_batch(2)
        .seq_len(10)
        .nan_policy(nan)
        .spike_detector(no_spike())
        .log_console(false)
        .after_backward(std::move(hook))
        .build();
  };
  // 1) Atla: adim 3'te gradyana NaN -> adim uygulanmaz, egitim surer.
  {
    nexus::nn::Sequential model = nano_model();
    CountingSource src;
    std::vector<std::vector<float>> before_bad;
    nt::NanOptions opt;
    opt.policy = nt::NanPolicy::Skip;
    auto trainer = make(model, src, opt, [&](nt::LmBackend&, std::int64_t step) {
      if (step == 3) {
        before_bad = weights_of(model);
        model.parameters()[2]->grad[0] = std::nanf("");
      }
    });
    trainer->train(3);
    trainer->train(4);
    const auto after_bad = weights_of(model);
    const nt::TrainSummary s = trainer->train(8);
    nexus_test::expect(max_rel_diff(after_bad, before_bad) == 0.0, "atla: NaN adiminda agirlik degismez");
    nexus_test::expect(!s.stopped && s.steps == 8 && std::isfinite(s.last_loss), "atla: egitim surer, kayip sonlu");
    nexus_test::expect(trainer->nan_events().size() == 1 && trainer->nan_events()[0].step == 3 &&
                           trainer->nan_events()[0].action == "skip",
                       "atla: tek olay, adim 3");
    nexus_test::expect(!trainer->nan_events()[0].report.reproduced, "gecici gradyan NaN'i anomali modunda tekrar etmez");
  }
  // 2) Durdur.
  {
    nexus::nn::Sequential model = nano_model();
    CountingSource src;
    nt::NanOptions opt;
    opt.policy = nt::NanPolicy::Stop;
    auto trainer = make(model, src, opt, [&](nt::LmBackend&, std::int64_t step) {
      if (step == 2) {
        model.parameters()[0]->grad[5] = INFINITY;
      }
    });
    const nt::TrainSummary s = trainer->train(10);
    nexus_test::expect(s.stopped && s.steps == 3, "durdur: adim 2'de durur");
    std::printf("  neden: %s\n", s.reason.c_str());
  }
  // 3) Geri don: adim 4'te (anlik goruntu 4'ten sonra) katman 2'nin agirligi bozulur; adim 5'in ileri
  //    gecisi NaN. Anomali modu ilk NaN katmanini ve bozuk parametreyi adiyla bulur; adim 4'e donulur.
  {
    nexus::nn::Sequential model = nano_model();
    CountingSource src;
    nt::NanOptions opt;
    opt.policy = nt::NanPolicy::Rollback;
    opt.snapshot_every = 2;
    bool corrupted = false;
    auto trainer = make(model, src, opt, [&](nt::LmBackend&, std::int64_t step) {
      if (step == 4 && !corrupted) {
        corrupted = true;
        model.parameters()[2]->data[0] = std::nanf("");  // CausalTransformerBlock'un ilk tensoru
      }
    });
    const nt::TrainSummary s = trainer->train(12);
    const auto& ev = trainer->nan_events();
    nexus_test::expect(!s.stopped && s.steps == 12 && std::isfinite(s.last_loss), "geri don: egitim tamamlanir, kayip sonlu");
    nexus_test::expect(ev.size() == 1 && ev[0].step == 5, "geri don: bozulmadan sonraki ilk ileri geciste (adim 5) yakalandi");
    if (!ev.empty()) {
      const nt::LmNonFinite& r = ev[0].report;
      std::printf("  ilk NaN: %s | bozuk parametre: %s | eylem: %s\n", r.first_activation.c_str(),
                  r.parameters.empty() ? "-" : r.parameters[0].c_str(), ev[0].action.c_str());
      nexus_test::expect(r.reproduced && r.first_activation == "2.CausalTransformerBlock", "ilk NaN katmani adiyla: 2.CausalTransformerBlock");
      nexus_test::expect(r.parameters.size() == 1 && r.parameters[0].rfind("2.CausalTransformerBlock.", 0) == 0,
                         "bozuk parametre adiyla raporlanir");
      nexus_test::expect(ev[0].action.find("adim 4") != std::string::npos && trainer->lr_factor() == 0.5,
                         "son anlik goruntuye (adim 4) donuldu, LR x0.5");
    }
    bool finite = true;
    for (const auto& w : weights_of(model)) {
      for (float v : w) {
        finite = finite && std::isfinite(v);
      }
    }
    nexus_test::expect(finite, "geri donusten sonra agirliklar sonlu");
  }
  return 0;
}

int test_spike() {
  nexus_test::section("test_loss_spike: hareketli medyan + k*MAD");
  class SpikySource final : public nt::TokenBatchSource {
   public:
    void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
      // Sabit 0,1,2,... deseni (hizla ogrenilir); 40. cagrida rastgele tokenlar (kayip sicrar).
      const std::vector<std::int32_t> noise = lcg_tokens(static_cast<std::size_t>(rows) * time, 5u);
      for (int r = 0; r < rows; ++r) {
        for (int t = 0; t < time; ++t) {
          dst[r * time + t] = calls_ == 40 ? noise[static_cast<std::size_t>(r * time + t)] : (t + r) % kVocab;
        }
        ids[r] = 1000 * calls_ + static_cast<std::uint64_t>(r);
      }
      ++calls_;
    }
    std::string state() const override { return std::to_string(calls_); }
    void restore(const std::string& s) override { calls_ = std::stoull(s); }

   private:
    std::uint64_t calls_ = 0;
  } src;
  nexus::nn::Sequential model = nano_model();
  nt::SpikeOptions spike;
  spike.window = 16;
  spike.k = 6.0;
  spike.min_steps = 8;
  auto trainer = nt::LanguageTrainerBuilder(model)
                     .data(src)
                     .optimizer(nt::AdamW{5.0e-3, 0.9, 0.95, 1.0e-8, 0.0})
                     .micro_batch(2)
                     .seq_len(12)
                     .spike_detector(spike)
                     .log_console(false)
                     .build();
  trainer->train(50);
  const auto& sp = trainer->spikes();
  for (const nt::SpikeEvent& e : sp) {
    std::printf("  sicrama adim %lld kayip %.3f medyan %.3f mad %.4f ilk ornek %llu\n", static_cast<long long>(e.step), e.loss,
                e.median, e.mad, static_cast<unsigned long long>(e.sample_ids.empty() ? 0 : e.sample_ids[0]));
  }
  bool found = false;
  for (const nt::SpikeEvent& e : sp) {
    found = found || (e.step == 40 && e.sample_ids.size() == 2 && e.sample_ids[0] == 40000 && e.sample_ids[1] == 40001);
  }
  nexus_test::expect(found, "adim 40 sicramasi ornek kimlikleriyle (40000, 40001) kaydedildi");
  nexus_test::expect(sp.size() <= 2, "yanlis alarm yok (en fazla bir komsu adim)");
  return 0;
}

int test_resume() {
  nexus_test::section("test_resume: 3 + kayit + devam == 6 adim");
  const std::string path = (std::filesystem::temp_directory_path() / "nexus_faz9_resume.nxlm").string();
  auto make = [](nexus::nn::Sequential& model, nt::TokenBatchSource& src) {
    return nt::LanguageTrainerBuilder(model)
        .data(src)
        .optimizer(nt::AdamW{1.0e-2, 0.9, 0.95, 1.0e-8, 0.1})
        .schedule(nt::WarmupCosine{2, 6, 0.1})
        .micro_batch(2)
        .seq_len(10)
        .global_batch_tokens(40)
        .grad_clip(1.0)
        .spike_detector(no_spike())
        .log_console(false)
        .build();
  };
  const nexus::nn::Sequential reference = nano_model();
  std::vector<double> straight;
  {
    nexus::nn::Sequential model = nano_model();
    copy_weights(reference, model);
    CountingSource src;
    auto t = make(model, src);
    t->train(6);
    for (const auto& r : t->history()) {
      straight.push_back(r.loss);
    }
  }
  std::vector<double> resumed;
  {
    nexus::nn::Sequential model = nano_model();
    copy_weights(reference, model);
    CountingSource src;
    auto t = make(model, src);
    t->train(3);
    for (const auto& r : t->history()) {
      resumed.push_back(r.loss);
    }
    t->save(path);
  }
  {
    nexus::nn::Sequential model = nano_model();  // ayni baslatma, sonra kayittan yukleme
    for (nexus_model::Parameter* p : model.parameters()) {
      p->data.zero();  // kaydin gercekten yuklendigini gostermek icin bozulur
    }
    CountingSource src;
    auto t = make(model, src);
    t->resume(path);
    nexus_test::expect(t->step() == 3, "devam adim sayaci 3");
    t->train(6);
    for (const auto& r : t->history()) {
      resumed.push_back(r.loss);
    }
  }
  bool same = straight.size() == 6 && resumed.size() == 6;
  for (std::size_t i = 0; same && i < 6; ++i) {
    same = straight[i] == resumed[i];
    std::printf("  adim %zu: %.9g / %.9g\n", i, straight[i], resumed[i]);
  }
  nexus_test::expect(same, "devam eden egitim kesintisizle bit bit ayni");
  std::filesystem::remove(path);
  return 0;
}

int test_eval() {
  nexus_test::section("test_eval: sabit dogrulama kumesi + coktan secmeli puan");
  nexus::nn::Sequential model = nano_model();
  CountingSource train;
  CountingSource val;
  auto trainer = nt::LanguageTrainerBuilder(model)
                     .data(train, &val)
                     .optimizer(nt::AdamW{5.0e-3, 0.9, 0.95, 1.0e-8, 0.0})
                     .micro_batch(2)
                     .seq_len(10)
                     .eval_every(4, 60)
                     .spike_detector(no_spike())
                     .log_console(false)
                     .build();
  const nt::EvalResult e0 = trainer->evaluate();
  const nt::EvalResult e1 = trainer->evaluate();
  nexus_test::expect(e0.loss == e1.loss && e0.tokens == 3 * 2 * 9, "ayni tokenlar (3 yigin), ayni kayip; kaynak ilerlemez");
  trainer->train(8);
  int evals = 0;
  for (const auto& r : trainer->history()) {
    evals += std::isfinite(r.val_loss) ? 1 : 0;
  }
  nexus_test::expect(evals == 2, "eval_every(4): adim 4 ve 8");
  const nt::EvalResult e2 = trainer->evaluate();
  nexus_test::info("val_loss_before", e0.loss);
  nexus_test::info("val_loss_after", e2.loss);
  nexus_test::info("val_ppl_after", e2.perplexity);
  nexus_test::expect(std::fabs(e2.perplexity - std::exp(e2.loss)) < 1e-9, "ppl = exp(kayip)");

  // Coktan secmeli: puan = -sum(secenek token kaybi). Ayni satiri tek basina ve toplu puanlayinca ayni.
  nt::MultipleChoiceItem item;
  item.context = {1, 2, 3, 4};
  item.choices = {{5, 6}, {7}, {1, 1, 1}};
  const std::vector<double> scores = trainer->score_choices(item);
  nexus_test::expect(scores.size() == 3, "secenek basina bir puan");
  std::vector<std::int32_t> row(2 * 10, 0);
  std::copy(item.context.begin(), item.context.end(), row.begin());
  row[4] = 5;
  row[5] = 6;
  const std::vector<float> losses = trainer->backend().token_losses(row.data());
  const double manual = -(static_cast<double>(losses[3]) + losses[4]);
  nexus_test::info("score0", scores[0]);
  nexus_test::info("manual0", manual);
  nexus_test::expect(std::fabs(scores[0] - manual) < 1e-6, "puan = -(kayip[ctx-1] + kayip[ctx]) elle hesapla ayni");
  return 0;
}

int test_loggers() {
  nexus_test::section("test_loggers: CSV, JSONL, W&B JSONL, TensorBoard");
  const char check[] = "123456789";
  nexus_test::expect(nt::crc32c(check, 9) == 0xE3069283u, "crc32c test vektoru");
  const auto dir = std::filesystem::temp_directory_path() / "nexus_faz9_logs";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  auto tb = std::make_shared<nt::TensorBoardLogger>((dir / "tb").string());
  {
    nexus::nn::Sequential model = nano_model();
    CountingSource train;
    CountingSource val;
    auto trainer = nt::LanguageTrainerBuilder(model)
                       .data(train, &val)
                       .optimizer(nt::AdamW{5.0e-3, 0.9, 0.95, 1.0e-8, 0.1})
                       .schedule(nt::WarmupCosine{2, 10, 0.1})
                       .micro_batch(2)
                       .seq_len(10)
                       .eval_every(5, 20)
                       .log_every(1)
                       .histogram_every(5)
                       .log_csv((dir / "log.csv").string())
                       .log_jsonl((dir / "log.jsonl").string())
                       .log_wandb((dir / "wandb-history.jsonl").string())
                       .logger(tb)
                       .spike_detector(no_spike())
                       .log_console(true)
                       .build();
    trainer->train(10);
    nexus_test::expect(trainer->norm_history().size() == 2, "katman normlari her 5 adimda");
  }
  std::string version;
  const std::vector<nt::TensorBoardScalar> scalars = nt::read_tensorboard_scalars(tb->path(), &version);
  int loss_tags = 0;
  int val_tags = 0;
  for (const auto& s : scalars) {
    loss_tags += s.tag == "train/loss" ? 1 : 0;
    val_tags += s.tag == "val/loss" ? 1 : 0;
  }
  nexus_test::info("tensorboard_scalars", static_cast<double>(scalars.size()));
  nexus_test::expect(version == "brain.Event:2", "TensorBoard dosya surumu kaydi");
  nexus_test::expect(loss_tags == 10 && val_tags == 2, "TensorBoard: 10 train/loss, 2 val/loss (CRC'ler gecerli)");
  auto lines = [](const std::filesystem::path& p) {
    std::ifstream in(p);
    std::vector<std::string> out;
    for (std::string l; std::getline(in, l);) {
      out.push_back(l);
    }
    return out;
  };
  const auto wandb = lines(dir / "wandb-history.jsonl");
  nexus_test::expect(wandb.size() == 10 && wandb[0].rfind("{\"_step\":1,\"_runtime\":", 0) == 0 &&
                         wandb[4].find("\"val/loss\":") != std::string::npos,
                     "W&B history satirlari (_step, _runtime, val/loss)");
  const auto csv = lines(dir / "log.csv");
  nexus_test::expect(!csv.empty() && csv[0] == "kind,epoch,step,name,value", "NexusTrain CsvLogger basligi");
  const auto jsonl = lines(dir / "log.jsonl");
  nexus_test::expect(jsonl.size() == 10 && jsonl[0].find("\"perf/tokens_per_s\"") != std::string::npos, "NexusTrain JsonLogger");
  tb.reset();  // dosya kapanmadan dizin silinemez (Windows)
  std::filesystem::remove_all(dir);
  return 0;
}

int test_classifier() {
  nexus_test::section("test_classifier_shared_infra: Trainer biriktirme + adim zamanlayicisi (FAZ 9.8)");
  constexpr int kIn = 4;
  constexpr int kClasses = 3;
  std::vector<float> features(16 * kIn);
  std::vector<int> labels(16);
  std::uint32_t seed = 11u;
  for (float& f : features) {
    seed = seed * 1664525u + 1013904223u;
    f = static_cast<float>((seed >> 9) % 1000) / 500.0f - 1.0f;
  }
  for (std::size_t i = 0; i < labels.size(); ++i) {
    labels[i] = static_cast<int>(i % kClasses);
  }
  auto make_model = [] {
    return nexus::nn::Sequential().add<nexus::nn::Linear>(kIn, 8).add<nexus::nn::ReLU>().add<nexus::nn::Linear>(8, kClasses);
  };
  nexus::nn::Sequential reference = make_model();
  auto run = [&](int batch, int accum, nt::LrSchedule schedule) {
    nexus::nn::Sequential model = make_model();
    copy_weights(reference, model);
    nexus::data::DataLoader train(features, labels, kIn, batch, false);
    nexus::data::DataLoader val(features, labels, kIn, 16, false);
    auto trainer = nexus::TrainerBuilder(model)
                       .optimizer(nexus::optim::AdamW(model->parameters(), {.lr = 0.05f, .weight_decay = 0.01f}))
                       .loss(nexus::loss::CrossEntropy())
                       .gradient_clip_norm(0.5f)
                       .gradient_accumulation(accum)
                       .step_schedule(schedule)
                       .build();
    trainer.fit(train, val, 3);
    return std::make_pair(weights_of(trainer.model()), trainer.optimizer_steps());
  };
  const nt::LrSchedule sched = nt::WarmupCosine{1, 6, 0.1};
  const auto a = run(2, 4, sched);  // 16 ornek / (2 x 4) = 2 adim / epoch
  const auto b = run(8, 1, sched);
  const double diff = max_rel_diff(a.first, b.first);
  nexus_test::info("param_max_rel_diff", diff);
  nexus_test::expect(a.second == 6 && b.second == 6, "3 epoch x 2 adim, iki yapilandirmada da");
  nexus_test::expect(diff < 1.0e-4, "micro 2 x accum 4 == batch 8 (kirpma + zamanlayici dahil)");
  const auto c = run(3, 2, sched);  // 16 = 3+3 | 3+3 | 3+1 -> eksik son pencere de adim atar
  nexus_test::expect(c.second == 9, "eksik son pencere adim atar (3 epoch x 3 adim)");
  return 0;
}

}  // namespace

int run(const std::string& mode);

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // cokme halinde de cikti kaybolmasin
  const std::string mode = argc > 1 ? argv[1] : "old_api";
  try {
    return run(mode);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "istisna: %s\n", e.what());
    return 1;
  }
}

int run(const std::string& mode) {
  if (mode == "golden") {
    return test_old_api(true);
  }
  if (mode == "old_api") {
    test_old_api(false);
  } else if (mode == "accum") {
    test_accum();
  } else if (mode == "schedule") {
    test_schedule();
  } else if (mode == "groups") {
    test_groups();
  } else if (mode == "nan") {
    test_nan();
  } else if (mode == "spike") {
    test_spike();
  } else if (mode == "resume") {
    test_resume();
  } else if (mode == "eval") {
    test_eval();
  } else if (mode == "loggers") {
    test_loggers();
  } else if (mode == "classifier") {
    test_classifier();
  } else {
    std::fprintf(stderr, "bilinmeyen mod: %s\n", mode.c_str());
    return 2;
  }
  return nexus_test::finish("test_phase9_trainer");
}
