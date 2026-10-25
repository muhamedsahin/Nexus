/**
 * @file test_phase9_trainer_cuda.cpp
 * @brief FAZ 9: dil modeli egitmeni v2, GPU arka ucu (cuda::gpt_backend) testleri, uctan uca
 *        egitim ve benchmark.
 *
 *   accum    : FP32 (TF32 kapali), micro 2 x accum 4 == micro 8 x accum 1: gradyan ve adim sonrasi agirlik
 *   groups   : sifir gradyanla tek adim; yalniz sonumlu gruplar kuculur, grup LR carpani uygulanir
 *   nan      : atla (gradyan NaN) ve geri don (bozuk agirlik: ilk NaN katmani "h.1", parametre "h.1.fc_w")
 *   eval_mem : degerlendirme sirasinda yeni aygit ayirmasi / tepe artisi yok
 *   resume   : 3 + kayit + yeni model + 3 == 6 adim (BF16, bit bit)
 *   train    : uctan uca (NXTK manifest'leri) — bkz. kullanim
 *   bench    : gpt2s BF16 + flash + fused AdamW + selective: GptTrainer ve v2 egitmen token/s, MFU, bellek
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/lm_backend.hpp"
#include "nexus/train/lm_trainer.hpp"
#include "nexus_test.hpp"
#include "nexusdata/text/token_shard.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace nt = nexus::train;
namespace nd = nexusdata::text;

namespace {

constexpr int kSkip = 77;

nc::GptConfig tiny_config() {
  nc::GptConfig c;
  c.vocab = 512;
  c.max_seq = 64;
  c.layers = 2;
  c.heads = 4;
  c.dim = 64;
  return c;
}

std::vector<std::int32_t> lcg_tokens(std::size_t n, std::uint32_t seed, int vocab) {
  std::vector<std::int32_t> v(n);
  std::uint32_t s = seed;
  for (std::int32_t& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<std::int32_t>((s >> 8) % static_cast<std::uint32_t>(vocab));
  }
  return v;
}

class CountingSource final : public nt::TokenBatchSource {
 public:
  explicit CountingSource(int vocab) : vocab_(vocab) {}
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
    for (int r = 0; r < rows; ++r) {
      const std::uint64_t id = index_++;
      const auto row = lcg_tokens(static_cast<std::size_t>(time), static_cast<std::uint32_t>(id * 2654435761u + 17u), vocab_);
      std::memcpy(dst + static_cast<std::size_t>(r) * time, row.data(), row.size() * 4);
      if (ids != nullptr) {
        ids[r] = id;
      }
    }
  }
  std::string state() const override { return std::to_string(index_); }
  void restore(const std::string& s) override { index_ = std::stoull(s); }

 private:
  int vocab_;
  std::uint64_t index_ = 0;
};

nc::GptModel& gpt(nt::LmBackend& b) { return dynamic_cast<nc::GptLmBackend&>(b).model(); }

nt::SpikeOptions no_spike() {
  nt::SpikeOptions s;
  s.enabled = false;
  return s;
}

double max_rel(const std::vector<float>& a, const std::vector<float>& b, double floor) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]) / (floor + std::fabs(static_cast<double>(b[i]))));
  }
  return worst;
}

// ------------------------------------------------------------------------------------------

int test_accum() {
  nexus_test::section("test_accum_equivalence_cuda: FP32 (highest), micro 2 x 4 == micro 8 x 1");
  const nc::MatmulPrecision prev = nc::float32_matmul_precision();
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  const int T = 32;
  const auto rows = lcg_tokens(8 * T, 5u, 512);
  struct Run {
    std::vector<float> grads, params;
    double loss = 0.0;
  };
  auto run = [&](int micro) {
    Run out;
    nt::FixedRowsSource source(rows, 8, T);
    auto trainer = nt::LanguageTrainerBuilder(nc::gpt_backend(tiny_config()))
                       .execution_policy(nexus::ExecutionPolicy::cuda(0))
                       .precision(nt::Precision::F32)
                       .deterministic(true)
                       .seed(7)
                       .data(source)
                       .optimizer(nt::AdamW{1.0e-3, 0.9, 0.95, 1.0e-8, 0.1})
                       .micro_batch(micro)
                       .seq_len(T)
                       .global_batch_tokens(8 * T)
                       .grad_clip(1.0)
                       .spike_detector(no_spike())
                       .log_console(false)
                       .after_backward([&](nt::LmBackend& b, std::int64_t) { out.grads = gpt(b).download_gradients(); })
                       .build();
    out.loss = trainer->train(1).last_loss;
    out.params = gpt(trainer->backend()).download_parameters();
    return out;
  };
  const Run a = run(2);
  const Run b = run(8);
  double gmax = 0.0;
  for (float g : b.grads) {
    gmax = std::max(gmax, std::fabs(static_cast<double>(g)));
  }
  const double g = max_rel(a.grads, b.grads, 1.0e-3 * gmax);
  // Parametre farki tam bir Adam adiminin (lr) kesri olarak: sifirdan baslayan bias'larda |g| ~ 1e-11
  // yuvarlama gurultusu ilk Adam adiminda g / (|g| + eps) uzerinden lr * 1e-3 mertebesinde fark verir.
  double w = 0.0;
  for (std::size_t i = 0; i < a.params.size(); ++i) {
    w = std::max(w, std::fabs(static_cast<double>(a.params[i]) - b.params[i]) / 1.0e-3);
  }
  nexus_test::info("grad_max_rel_diff (taban 1e-3*max|g|)", g);
  nexus_test::info("param_max_abs_diff_over_lr", w);
  nexus_test::info("loss_accum", a.loss);
  nexus_test::info("loss_full", b.loss);
  nexus_test::expect(g < 1.0e-4, "biriktirilmis gradyan == buyuk yigin (rel < 1e-4)");
  nexus_test::expect(w < 1.0e-2, "adim sonrasi parametre farki < lr'nin %1'i");
  nexus_test::expect(std::fabs(a.loss - b.loss) < 1.0e-5, "token agirlikli adim kaybi ayni");
  nc::set_float32_matmul_precision(prev);
  return 0;
}

int test_groups() {
  nexus_test::section("test_param_groups_wd_cuda: sifir gradyanla tek adim (FP32)");
  const nc::MatmulPrecision prev = nc::float32_matmul_precision();
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  const double lr = 0.1;
  const double wd = 0.5;
  CountingSource src(512);
  nt::ParamGroupPolicy groups;
  groups.decay_embeddings = false;
  groups.rules.push_back(nt::ParamGroupRule{"fc_w", 3.0f, -1.0f});
  auto trainer = nt::LanguageTrainerBuilder(nc::gpt_backend(tiny_config()))
                     .execution_policy(nexus::ExecutionPolicy::cuda(0))
                     .data(src)
                     .optimizer(nt::AdamW{lr, 0.9, 0.95, 1.0e-8, wd}, groups)
                     .micro_batch(2)
                     .seq_len(16)
                     .spike_detector(no_spike())
                     .log_console(false)
                     .after_backward([](nt::LmBackend& b, std::int64_t) {
                       nc::GptModel& m = gpt(b);
                       NEXUS_CUDA_CHECK(cudaMemsetAsync(m.device_gradients(), 0, m.num_parameters() * 4, m.stream().handle()));
                     })
                     .build();
  nc::GptModel& m = gpt(trainer->backend());
  const auto before = m.download_parameters();
  trainer->train(1);
  const auto after = m.download_parameters();
  bool ok = true;
  for (const nc::GptParamInfo& p : m.parameter_info()) {
    const bool decay = p.decay && p.name != "wte" && p.name != "wpe";
    const double s = p.name == "fc_w" ? 3.0 : 1.0;
    const float f = decay ? static_cast<float>(1.0 - lr * s * wd) : 1.0f;
    double worst = 0.0;
    for (std::size_t i = p.offset; i < p.offset + p.numel; ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(after[i]) - static_cast<double>(before[i]) * f));
    }
    std::printf("  %-10s decay=%d lr_scale=%.0f max_err=%.2e\n", p.name.c_str(), decay ? 1 : 0, s, worst);
    ok = ok && worst < 1.0e-6;
  }
  nexus_test::expect(ok, "sonum yalniz katman matrislerinde (gomme kapali), fc_w LR x3");
  nc::set_float32_matmul_precision(prev);
  return 0;
}

int test_nan() {
  nexus_test::section("test_nan_policy_cuda: atla + geri don (BF16)");
  auto make = [](nt::TokenBatchSource& src, nt::NanOptions nan, std::function<void(nt::LmBackend&, std::int64_t)> hook) {
    return nt::LanguageTrainerBuilder(nc::gpt_backend(tiny_config()))
        .execution_policy(nexus::ExecutionPolicy::cuda(0))
        .precision(nt::Precision::BF16)
        .data(src)
        .optimizer(nt::AdamW{3.0e-3, 0.9, 0.95, 1.0e-8, 0.1})
        .micro_batch(2)
        .seq_len(32)
        .global_batch_tokens(128)
        .grad_clip(1.0)
        .nan_policy(nan)
        .spike_detector(no_spike())
        .log_console(false)
        .after_backward(std::move(hook))
        .build();
  };
  {
    CountingSource src(512);
    nt::NanOptions opt;
    std::vector<float> before;
    auto t = make(src, opt, [&](nt::LmBackend& b, std::int64_t step) {
      if (step == 3) {
        nc::GptModel& m = gpt(b);
        before = m.download_parameters();
        const float nan = std::nanf("");
        NEXUS_CUDA_CHECK(cudaMemcpy(m.device_gradients() + 100, &nan, 4, cudaMemcpyHostToDevice));
      }
    });
    t->train(4);
    const auto after = gpt(t->backend()).download_parameters();
    const nt::TrainSummary s = t->train(8);
    nexus_test::expect(std::memcmp(before.data(), after.data(), before.size() * 4) == 0, "atla: NaN adiminda parametre degismez (aygitta)");
    nexus_test::expect(!s.stopped && std::isfinite(s.last_loss) && t->nan_events().size() == 1, "atla: egitim surer");
    nexus_test::expect(t->history()[3].skipped_total == 1, "skipped_steps sayaci 1");
  }
  {
    CountingSource src(512);
    nt::NanOptions opt;
    opt.policy = nt::NanPolicy::Rollback;
    opt.snapshot_every = 2;
    bool corrupted = false;
    auto t = make(src, opt, [&](nt::LmBackend& b, std::int64_t step) {
      if (step == 4 && !corrupted) {
        corrupted = true;
        nc::GptModel& m = gpt(b);
        std::vector<float> p = m.download_parameters();
        for (const nc::GptParamInfo& info : m.parameter_info()) {
          if (info.name == "fc_w") {
            p[info.offset + info.numel / 2 + 7] = std::nanf("");  // katman 1'in fc_w dilimi (L = 2)
          }
        }
        m.upload_parameters(p);
      }
    });
    const nt::TrainSummary s = t->train(12);
    const auto& ev = t->nan_events();
    nexus_test::expect(!s.stopped && s.steps == 12 && std::isfinite(s.last_loss), "geri don: egitim tamamlanir");
    if (!ev.empty()) {
      const nt::LmNonFinite& r = ev[0].report;
      std::printf("  adim %lld | ilk NaN: %s | parametre: %s | ilk NaN gradyan: %s | eylem: %s\n",
                  static_cast<long long>(ev[0].step), r.first_activation.c_str(),
                  r.parameters.empty() ? "-" : r.parameters[0].c_str(), r.gradients.empty() ? "-" : r.gradients[0].c_str(),
                  ev[0].action.c_str());
      nexus_test::expect(ev.size() == 1 && ev[0].step == 5, "adim 5'te yakalandi");
      nexus_test::expect(r.reproduced && r.first_activation == "h.1", "ilk NaN katmani: h.1");
      nexus_test::expect(r.parameters.size() == 1 && r.parameters[0] == "h.1.fc_w", "bozuk parametre: h.1.fc_w");
      nexus_test::expect(ev[0].action.find("adim 4") != std::string::npos, "adim 4 anlik goruntusune donuldu");
    } else {
      nexus_test::expect(false, "NaN olayi kaydedilmedi");
    }
  }
  return 0;
}

int test_eval_mem() {
  nexus_test::section("test_eval_no_grad_memory: degerlendirmede ayirma/tepe artisi yok");
  CountingSource train(512);
  CountingSource val(512);
  auto t = nt::LanguageTrainerBuilder(nc::gpt_backend(tiny_config()))
               .execution_policy(nexus::ExecutionPolicy::cuda(0))
               .precision(nt::Precision::BF16)
               .data(train, &val)
               .optimizer(nt::AdamW{})
               .micro_batch(4)
               .seq_len(64)
               .eval_every(0, 0)
               .spike_detector(no_spike())
               .log_console(false)
               .build();
  t->train(2);  // isinma (cuBLASLt calisma alani, plan onbellegi)
  (void)t->evaluate();
  nexus::cuda::device_allocator().reset_peak_stats();
  const nc::DeviceAllocatorStats s0 = nexus::cuda::device_allocator().stats(0);
  std::size_t free0 = 0;
  std::size_t total = 0;
  NEXUS_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  const nt::EvalResult e = t->evaluate();
  const nc::DeviceAllocatorStats s1 = nexus::cuda::device_allocator().stats(0);
  std::size_t free1 = 0;
  NEXUS_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  nexus_test::info("val_loss", e.loss);
  nexus_test::info("allocs_during_eval", static_cast<double>(s1.num_allocs - s0.num_allocs));
  nexus_test::info("peak_allocated_increase", static_cast<double>(s1.peak_allocated_bytes) - static_cast<double>(s0.allocated_bytes));
  nexus_test::expect(s1.num_allocs == s0.num_allocs, "degerlendirmede yeni aygit ayirmasi yok");
  nexus_test::expect(s1.peak_allocated_bytes == s0.allocated_bytes, "tepe ayrilmis bellek artmadi");
  nexus_test::expect(free1 == free0, "cudaMemGetInfo bos bellek ayni");
  return 0;
}

int test_resume() {
  nexus_test::section("test_trainer_resume_cuda: BF16, 3 + kayit + 3 == 6");
  const std::string path = (std::filesystem::temp_directory_path() / "nexus_faz9_cuda.nxlm").string();
  auto make = [](nt::TokenBatchSource& src) {
    return nt::LanguageTrainerBuilder(nc::gpt_backend(tiny_config()))
        .execution_policy(nexus::ExecutionPolicy::cuda(0))
        .precision(nt::Precision::BF16)
        .deterministic(true)
        .data(src)
        .optimizer(nt::AdamW{3.0e-3, 0.9, 0.95, 1.0e-8, 0.1})
        .schedule(nt::WarmupCosine{2, 6, 0.1})
        .micro_batch(2)
        .seq_len(32)
        .global_batch_tokens(128)
        .grad_clip(1.0)
        .spike_detector(no_spike())
        .log_console(false)
        .build();
  };
  std::vector<double> a;
  std::vector<double> b;
  {
    CountingSource src(512);
    auto t = make(src);
    t->train(6);
    for (const auto& r : t->history()) a.push_back(r.loss);
  }
  {
    CountingSource src(512);
    auto t = make(src);
    t->train(3);
    for (const auto& r : t->history()) b.push_back(r.loss);
    t->save(path);
  }
  {
    CountingSource src(512);
    auto t = make(src);
    t->resume(path);
    t->train(6);
    for (const auto& r : t->history()) b.push_back(r.loss);
  }
  bool same = a.size() == 6 && b.size() == 6;
  for (std::size_t i = 0; same && i < 6; ++i) {
    std::printf("  adim %zu: %.9g / %.9g\n", i, a[i], b[i]);
    same = a[i] == b[i];
  }
  nexus_test::expect(same, "GPU devam: kesintisizle bit bit ayni");
  std::filesystem::remove(path);
  return 0;
}

// ------------------------------------------------------------------------------------------
// Uctan uca egitim ve benchmark
// ------------------------------------------------------------------------------------------

struct Args {
  std::map<std::string, std::string> kv;
  [[nodiscard]] std::string str(const std::string& k, const std::string& d) const {
    const auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  [[nodiscard]] double num(const std::string& k, double d) const {
    const auto it = kv.find(k);
    return it == kv.end() ? d : std::stod(it->second);
  }
};

Args parse(int argc, char** argv, int first) {
  Args a;
  for (int i = first; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) {
      throw std::invalid_argument("bayrak bekleniyor: " + k);
    }
    a.kv[k.substr(2)] = argv[i + 1];
  }
  return a;
}

nc::GptConfig model_config(const std::string& name, int seq) {
  nc::GptConfig c;  // gpt2s: 124M (L12 H12 C768, V50257)
  c.max_seq = std::max(1024, seq);
  if (name == "tiny") {  // ~10M: L4 H4 C256
    c.layers = 4;
    c.heads = 4;
    c.dim = 256;
  } else if (name != "gpt2s") {
    throw std::invalid_argument("model: tiny | gpt2s");
  }
  return c;
}

nt::Precision precision_of(const std::string& p) {
  if (p == "bf16") return nt::Precision::BF16;
  if (p == "f16") return nt::Precision::F16;
  if (p == "f32") return nt::Precision::F32;
  throw std::invalid_argument("precision: bf16 | f16 | f32");
}

nt::RecomputePolicy recompute_of(const std::string& r) {
  if (r == "none") return nt::RecomputePolicy::None;
  if (r == "selective") return nt::RecomputePolicy::Selective;
  if (r == "full") return nt::RecomputePolicy::Full;
  throw std::invalid_argument("recompute: none | selective | full");
}

/// Kaynaktan alinan her yigini dosyaya da yazar (nanoGPT referansi ayni tokenlarla beslensin).
class TeeSource final : public nt::TokenBatchSource {
 public:
  TeeSource(nt::TokenBatchSource& inner, const std::string& path) : inner_(&inner) {
    if (!path.empty()) {
      out_.open(path, std::ios::binary | std::ios::trunc);
      if (!out_) throw std::runtime_error("yazilamadi: " + path);
    }
  }
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
    inner_->next(dst, rows, time, ids);
    if (out_.is_open()) {
      out_.write(reinterpret_cast<const char*>(dst), static_cast<std::streamsize>(rows) * time * 4);
      out_.flush();
    }
  }
  std::string state() const override { return inner_->state(); }
  void restore(const std::string& s) override { inner_->restore(s); }

 private:
  nt::TokenBatchSource* inner_;
  std::ofstream out_;
};

void dump_init(nc::GptModel& m, const std::string& path) {
  const std::vector<float> p = m.download_parameters();
  std::ofstream bin(path, std::ios::binary | std::ios::trunc);
  bin.write(reinterpret_cast<const char*>(p.data()), static_cast<std::streamsize>(p.size() * 4));
  std::ofstream js(path + ".json", std::ios::trunc);
  js << "{\"layers\":" << m.config().layers << ",\"heads\":" << m.config().heads << ",\"dim\":" << m.config().dim
     << ",\"vocab\":" << m.config().vocab << ",\"max_seq\":" << m.config().max_seq << ",\"params\":[";
  bool first = true;
  for (const nc::GptParamInfo& info : m.parameter_info()) {
    js << (first ? "" : ",") << "{\"name\":\"" << info.name << "\",\"offset\":" << info.offset << ",\"shape\":[";
    for (std::size_t i = 0; i < info.shape.size(); ++i) js << (i ? "," : "") << info.shape[i];
    js << "]}";
    first = false;
  }
  js << "]}\n";
}

int run_train(const Args& a) {
  const std::string model = a.str("model", "tiny");
  const int B = static_cast<int>(a.num("micro", 16));
  const int T = static_cast<int>(a.num("seq", 256));
  const int accum = static_cast<int>(a.num("accum", 1));
  const auto steps = static_cast<std::int64_t>(a.num("steps", 200));
  const auto warmup = static_cast<std::int64_t>(a.num("warmup", 20));
  const std::string log_dir = a.str("log-dir", "");
  const std::uint64_t seed = static_cast<std::uint64_t>(a.num("seed", 1337));
  const nd::Manifest train_m = nd::Manifest::load(a.str("train", ""));
  const nd::Manifest val_m = nd::Manifest::load(a.str("val", ""));
  nd::TokenStream train_stream(train_m, T - 1, seed);
  nd::TokenStream val_stream(val_m, T - 1, seed + 1);
  nt::StreamSource train_src(train_stream);
  nt::StreamSource val_src(val_stream);
  TeeSource tee(train_src, a.str("dump-batches", ""));
  nc::GptConfig config = model_config(model, T);
  config.ce_chunk_rows = static_cast<int>(a.num("ce-chunk", 0));
  nt::LanguageTrainerBuilder builder(nc::gpt_backend(config));
  builder.execution_policy(nexus::ExecutionPolicy::cuda(0))
      .precision(precision_of(a.str("precision", "bf16")))
      .recompute(recompute_of(a.str("recompute", "none")))
      .deterministic(a.num("deterministic", 0) != 0)
      .seed(seed)
      .data(tee, &val_src)
      .optimizer(nt::AdamW{a.num("lr", 6.0e-4), 0.9, 0.95, 1.0e-8, a.num("wd", 0.1)})
      .schedule(nt::WarmupCosine{warmup, steps, a.num("min-lr-ratio", 0.1)})
      .micro_batch(B)
      .seq_len(T)
      .global_batch_tokens(static_cast<std::int64_t>(B) * T * accum)
      .grad_clip(a.num("clip", 1.0))
      .eval_every(static_cast<std::int64_t>(a.num("eval-every", 50)), static_cast<std::int64_t>(a.num("eval-tokens", 65536)))
      .log_every(static_cast<std::int64_t>(a.num("log-every", 10)))
      .histogram_every(static_cast<std::int64_t>(a.num("hist-every", 0)));
  if (!log_dir.empty()) {
    std::filesystem::create_directories(log_dir);
    builder.log_jsonl(log_dir + "/train.jsonl").log_csv(log_dir + "/train.csv").log_wandb(log_dir + "/wandb-history.jsonl")
        .log_tensorboard(log_dir + "/tb");
  }
  auto trainer = builder.build();
  if (!a.str("dump-init", "").empty()) {
    dump_init(gpt(trainer->backend()), a.str("dump-init", ""));
  }
  const nt::EvalResult e0 = trainer->evaluate();
  std::printf("baslangic dogrulama kaybi %.4f (ppl %.1f), %lld token/adim, accum %d\n", e0.loss, e0.perplexity,
              static_cast<long long>(trainer->tokens_per_step()), trainer->accum_steps());
  const auto t0 = std::chrono::steady_clock::now();
  const nt::TrainSummary s = trainer->train(steps);
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const nt::EvalResult e1 = trainer->evaluate();
  if (!log_dir.empty()) {
    // Tum adimlar (gunlukculer log_every'de bir yazar; egri karsilastirmasi adim adim yapilir).
    std::ofstream h(log_dir + "/history.csv", std::ios::trunc);
    h << "step,loss,lr,grad_norm,step_ms,tokens_per_s,mfu,val_loss,skipped\n";
    for (const nt::StepRecord& r : trainer->history()) {
      h << r.step << ',' << r.loss << ',' << r.lr << ',' << r.grad_norm << ',' << r.step_ms << ',' << r.tokens_per_s << ','
        << r.mfu << ',' << r.val_loss << ',' << (r.skipped ? 1 : 0) << '\n';
    }
  }
  // Ozet: isinma sonrasi (ilk 5 adim haric) ortalama hiz.
  double ms = 0.0;
  int n = 0;
  for (const nt::StepRecord& r : trainer->history()) {
    if (r.step >= 5) {
      ms += r.step_ms;
      ++n;
    }
  }
  const double mean_ms = n > 0 ? ms / n : 0.0;
  const double tps = mean_ms > 0 ? trainer->tokens_per_step() / (mean_ms / 1e3) : 0.0;
  const double peak = trainer->backend().peak_flops();
  std::printf(
      "{\"model\":\"%s\",\"steps\":%lld,\"stopped\":%s,\"tokens_per_step\":%lld,\"accum\":%d,\"wall_s\":%.1f,"
      "\"mean_step_ms\":%.2f,\"tokens_per_s\":%.0f,\"mfu\":%.4f,\"peak_flops\":%.3e,\"peak_reserved_gb\":%.3f,"
      "\"val_loss_start\":%.4f,\"val_loss_end\":%.4f,\"val_ppl_end\":%.3f,\"last_train_loss\":%.4f,\"nan_events\":%zu,"
      "\"spikes\":%zu}\n",
      model.c_str(), static_cast<long long>(s.steps), s.stopped ? "true" : "false",
      static_cast<long long>(trainer->tokens_per_step()), trainer->accum_steps(), wall, mean_ms, tps,
      peak > 0 ? trainer->flops_per_token() * tps / peak : 0.0, peak,
      trainer->backend().peak_memory_bytes() / 1.0e9, e0.loss, e1.loss, e1.perplexity, s.last_loss,
      trainer->nan_events().size(), trainer->spikes().size());
  return 0;
}

int run_bench(const Args& a) {
  const int B = static_cast<int>(a.num("micro", 8));
  const int T = static_cast<int>(a.num("seq", 1024));
  const int accum = static_cast<int>(a.num("accum", 4));
  const int steps = static_cast<int>(a.num("steps", 10));
  nc::GptConfig config = model_config("gpt2s", T);
  config.compute_dtype = nexus::DType::BF16;
  config.recompute = nc::Recompute::Selective;
  config.attention = nc::AttentionBackend::Flash;
  double clock = 0.0;
  const char* clock_source = "";
  const double peak = nc::tensor_core_peak_flops(0, nexus::DType::BF16, &clock, &clock_source);
  std::printf("  info  tepe BF16 dense %.1f TFLOPS (SM saati %.0f MHz, %s)\n", peak / 1e12, clock, clock_source);
  // (1) Ham GptTrainer (FAZ 3-8 yolu): biriktirme yok, N adimda bir kayip okuma.
  double raw_ms = 0.0;
  double raw_peak_gb = 0.0;
  {
    nc::device_allocator().reset_peak_stats();
    nc::Stream stream = nc::Stream::create(0);
    nc::GptModel model(config, B, T, 1337, stream);
    nc::DeviceAdamWConfig adam;
    adam.lr = 6e-4;
    adam.beta2 = 0.95;
    adam.weight_decay = 0.1;
    adam.max_grad_norm = 1.0;
    nc::GptTrainer trainer(model, adam, 10);
    const auto tokens = lcg_tokens(static_cast<std::size_t>(B) * T, 3u, 50257);
    for (int i = 0; i < 3; ++i) (void)trainer.step(tokens.data());
    trainer.synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps * accum; ++i) (void)trainer.step(tokens.data());
    trainer.synchronize();
    raw_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / (steps * accum);
    raw_peak_gb = nc::device_allocator().stats(0).peak_reserved_bytes / 1e9;
    stream.synchronize();
  }
  nc::device_allocator().empty_cache();
  // (2) v2 egitmen, accum = 1 ve accum = A (her adimda tek senkron).
  struct V2 {
    double ms = 0.0;
    double tps = 0.0;
    double mfu = 0.0;
    double peak_gb = 0.0;
    double fpt = 0.0;
  };
  auto v2 = [&](int acc) {
    V2 r;
    nc::device_allocator().reset_peak_stats();
    CountingSource src(50257);
    auto t = nt::LanguageTrainerBuilder(nc::gpt_backend(config))
                 .execution_policy(nexus::ExecutionPolicy::cuda(0))
                 .precision(nt::Precision::BF16)
                 .recompute(nt::RecomputePolicy::Selective)
                 .deterministic(false)
                 .data(src)
                 .optimizer(nt::AdamW{6.0e-4, 0.9, 0.95, 1.0e-8, 0.1})
                 .schedule(nt::WarmupCosine{10, 1000, 0.1})
                 .micro_batch(B)
                 .seq_len(T)
                 .global_batch_tokens(static_cast<std::int64_t>(B) * T * acc)
                 .grad_clip(1.0)
                 .log_console(false)
                 .build();
    t->train(3);
    const int n = acc == 1 ? steps * accum : steps;
    t->train(3 + n);
    double ms = 0.0;
    for (std::size_t i = 3; i < t->history().size(); ++i) ms += t->history()[i].step_ms;
    r.ms = ms / n;
    r.tps = t->tokens_per_step() / (r.ms / 1e3);
    r.fpt = t->flops_per_token();
    r.mfu = r.fpt * r.tps / peak;
    r.peak_gb = t->backend().peak_memory_bytes() / 1e9;
    return r;
  };
  const V2 a1 = v2(1);
  nc::device_allocator().empty_cache();
  const V2 aN = v2(accum);
  const double raw_tps = static_cast<double>(B) * T / (raw_ms / 1e3);
  std::printf("  info  B=%d T=%d, FLOP/token %.3e (6N + 12 L H d T)\n", B, T, a1.fpt);
  std::printf("  info  GptTrainer (ham)      : %8.1f ms/mikro  %7.0f tok/s  MFU %5.1f%%  tepe %.2f GB\n", raw_ms, raw_tps,
              100.0 * a1.fpt * raw_tps / peak, raw_peak_gb);
  std::printf("  info  v2 accum=1            : %8.1f ms/adim   %7.0f tok/s  MFU %5.1f%%  tepe %.2f GB\n", a1.ms, a1.tps,
              100.0 * a1.mfu, a1.peak_gb);
  std::printf("  info  v2 accum=%-2d           : %8.1f ms/adim   %7.0f tok/s  MFU %5.1f%%  tepe %.2f GB\n", accum, aN.ms,
              aN.tps, 100.0 * aN.mfu, aN.peak_gb);
  std::printf(
      "{\"B\":%d,\"T\":%d,\"accum\":%d,\"peak_tflops\":%.2f,\"clock_mhz\":%.0f,\"raw_ms_per_micro\":%.2f,\"raw_tps\":%.0f,"
      "\"raw_peak_gb\":%.3f,\"v2_a1_ms\":%.2f,\"v2_a1_tps\":%.0f,\"v2_a1_mfu\":%.4f,\"v2_a1_peak_gb\":%.3f,\"v2_aN_ms\":%.2f,"
      "\"v2_aN_tps\":%.0f,\"v2_aN_mfu\":%.4f,\"v2_aN_peak_gb\":%.3f}\n",
      B, T, accum, peak / 1e12, clock, raw_ms, raw_tps, raw_peak_gb, a1.ms, a1.tps, a1.mfu, a1.peak_gb, aN.ms, aN.tps,
      aN.mfu, aN.peak_gb);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return kSkip;
  }
  const std::string mode = argc > 1 ? argv[1] : "accum";
  try {
    if (mode == "accum") test_accum();
    else if (mode == "groups") test_groups();
    else if (mode == "nan") test_nan();
    else if (mode == "eval_mem") test_eval_mem();
    else if (mode == "resume") test_resume();
    else if (mode == "train") return run_train(parse(argc, argv, 2));
    else if (mode == "bench") return run_bench(parse(argc, argv, 2));
    else {
      std::printf(
          "kullanim: test_phase9_trainer_cuda <accum|groups|nan|eval_mem|resume>\n"
          "          test_phase9_trainer_cuda train --train M --val M [--model tiny|gpt2s --micro B --seq T --accum A\n"
          "              --steps N --warmup W --lr LR --precision bf16 --recompute none --eval-every K --eval-tokens N\n"
          "              --log-dir D --dump-init F --dump-batches F --seed S]\n"
          "          test_phase9_trainer_cuda bench [--micro 8 --seq 1024 --accum 4 --steps 10]\n");
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("faz9-trainer-cuda");
}
