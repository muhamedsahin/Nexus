/**
 * @file bench_phase3_training.cpp
 * @brief FAZ 3 egitim yolu benchmarklari (CPU), BOLUM G protokoluyle:
 *        isinma + 7 tekrar, medyan/min/max/std, makine kimligi JSON'a gomulu.
 *
 * Olculenler:
 *  1. Kaydirilmis CE, B*T=2048, V=50304: fuzyonlu (yerinde) yol vs eski
 *     LanguageTrainer yolu (duzlestirilmis logit kopyasi + gradyan kopyasi +
 *     tam tensöre geri kopya). Sure ve OLCULEN ek heap baytlari (global
 *     operator new sayaci; tahmin degil).
 *  2. AdamW, 124M parametre: foreach (fuzyonlu, skaler C++) vs NexusOptim AdamW
 *     (bu makinede AVX2 yolu). ms/adim.
 *  3. RMSNorm vs LayerNorm ileri+geri, [4096, 768]: etkin GB/s.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/fused/fused.hpp"
#include "nexus/fused/training.hpp"
#include "nexus/loss/losses.hpp"
#include "nexus_optim/algorithms/adamw.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <new>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// ---- Olculen heap: global operator new/delete bayt sayaci -------------------------------
namespace {
std::atomic<std::int64_t> g_live_bytes{0};
std::atomic<std::int64_t> g_peak_bytes{0};

void note_alloc(std::size_t bytes) noexcept {
  const std::int64_t now = g_live_bytes.fetch_add(static_cast<std::int64_t>(bytes)) + static_cast<std::int64_t>(bytes);
  std::int64_t peak = g_peak_bytes.load();
  while (now > peak && !g_peak_bytes.compare_exchange_weak(peak, now)) {
  }
}
}  // namespace

void* operator new(std::size_t bytes) {
  void* p = std::malloc(bytes + 16);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  *static_cast<std::size_t*>(p) = bytes;
  note_alloc(bytes);
  return static_cast<char*>(p) + 16;
}
void operator delete(void* p) noexcept {
  if (p == nullptr) {
    return;
  }
  char* base = static_cast<char*>(p) - 16;
  g_live_bytes.fetch_sub(static_cast<std::int64_t>(*reinterpret_cast<std::size_t*>(base)));
  std::free(base);
}
void operator delete(void* p, std::size_t) noexcept { operator delete(p); }

namespace {

using Clock = std::chrono::steady_clock;
namespace fu = nexus::fused;

struct Stats {
  double median = 0, min = 0, max = 0, stddev = 0;
  int repeats = 0;
};

Stats measure(const std::function<void()>& fn, int warmup = 2, int repeats = 7) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  std::vector<double> ms;
  for (int i = 0; i < repeats; ++i) {
    const auto start = Clock::now();
    fn();
    ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  }
  std::vector<double> sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  Stats s;
  s.repeats = repeats;
  s.median = sorted[sorted.size() / 2];
  s.min = sorted.front();
  s.max = sorted.back();
  double mean = 0;
  for (const double v : ms) {
    mean += v;
  }
  mean /= static_cast<double>(ms.size());
  for (const double v : ms) {
    s.stddev += (v - mean) * (v - mean);
  }
  s.stddev = std::sqrt(s.stddev / static_cast<double>(ms.size()));
  return s;
}

std::string cpu_name() {
#if defined(_MSC_VER)
  int regs[4];
  char brand[49] = {};
  __cpuid(regs, 0x80000000);
  if (static_cast<unsigned>(regs[0]) >= 0x80000004u) {
    for (int i = 0; i < 3; ++i) {
      __cpuid(regs, 0x80000002 + i);
      std::memcpy(brand + i * 16, regs, 16);
    }
  }
  std::string name(brand);
  while (!name.empty() && name.back() == ' ') {
    name.pop_back();
  }
  return name;
#else
  return "unknown";
#endif
}

std::string stats_json(const Stats& s) {
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer),
                "{\"median_ms\": %.6f, \"min_ms\": %.6f, \"max_ms\": %.6f, \"std_ms\": %.6f, \"repeats\": %d}", s.median,
                s.min, s.max, s.stddev, s.repeats);
  return buffer;
}

void print(const char* name, const Stats& s) {
  std::printf("  info  %-34s medyan=%.3f ms  min=%.3f  max=%.3f  std=%.3f (n=%d)\n", name, s.median, s.min, s.max,
              s.stddev, s.repeats);
}

}  // namespace

int main() {
  std::printf("== faz3 egitim yolu benchmarki (CPU) ==\n");
  std::mt19937_64 rng(20261024);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failed = 0;

  // ---- 1. Kaydirilmis CE --------------------------------------------------------------------
  const int batch = 2;
  const int time = 1024;
  const int vocab = 50304;
  const std::size_t count = static_cast<std::size_t>(batch) * time * vocab;
  std::vector<float> logits(count);
  for (float& v : logits) {
    v = normal(rng);
  }
  std::vector<int> tokens(static_cast<std::size_t>(batch * time));
  for (int& t : tokens) {
    t = static_cast<int>(rng() % static_cast<std::uint64_t>(vocab));
  }
  std::vector<float> full(count, 0.0f);  // her iki yolun ortak ciktisi: [B,T,V] gradyan

  float fused_loss = 0.0f;
  std::int64_t fused_extra = 0;
  const Stats ce_fused = measure([&] {
    const std::int64_t before = g_live_bytes.load();
    g_peak_bytes.store(before);
    fused_loss = fu::cross_entropy_shifted(logits.data(), reinterpret_cast<const std::int32_t*>(tokens.data()), batch,
                                           time, vocab, {}, full.data())
                     .loss;
    fused_extra = g_peak_bytes.load() - before;
  });

  float legacy_loss = 0.0f;
  std::int64_t legacy_extra = 0;
  const nexus::loss::CrossEntropy legacy_ce{};
  const Stats ce_legacy = measure([&] {
    const std::int64_t before = g_live_bytes.load();
    g_peak_bytes.store(before);
    // Eski LanguageTrainer::step govdesi (FAZ 3.E oncesi), birebir.
    const int steps = time - 1;
    const int rows = batch * steps;
    std::vector<float> flat(static_cast<std::size_t>(rows) * vocab);
    std::vector<int> target(static_cast<std::size_t>(rows));
    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < steps; ++t) {
        const float* src = logits.data() + static_cast<std::size_t>(b * time + t) * vocab;
        std::copy(src, src + vocab, flat.data() + static_cast<std::size_t>(b * steps + t) * vocab);
        target[static_cast<std::size_t>(b * steps + t)] = tokens[static_cast<std::size_t>(b * time + t + 1)];
      }
    }
    std::vector<float> grad(flat.size(), 0.0f);
    legacy_loss = legacy_ce.forward_backward(flat.data(), target.data(), rows, vocab, grad.data());
    std::vector<float> legacy_full(count, 0.0f);
    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < steps; ++t) {
        const float* src = grad.data() + static_cast<std::size_t>(b * steps + t) * vocab;
        std::copy(src, src + vocab, legacy_full.data() + static_cast<std::size_t>(b * time + t) * vocab);
      }
    }
    legacy_extra = g_peak_bytes.load() - before;
  });
  print("ce_shifted_fused (B*T=2048,V=50304)", ce_fused);
  print("ce_legacy_copies", ce_legacy);
  const double logit_mb = static_cast<double>(count * sizeof(float)) / 1048576.0;
  std::printf("  info  logit tamponu=%.1f MiB  ek heap: fuzyonlu=%.1f MiB  eski=%.1f MiB\n", logit_mb,
              static_cast<double>(fused_extra) / 1048576.0, static_cast<double>(legacy_extra) / 1048576.0);
  std::printf("  info  kayip: fuzyonlu=%.7f eski=%.7f\n", fused_loss, legacy_loss);
  if (std::fabs(fused_loss - legacy_loss) > 1.0e-4f * std::fabs(legacy_loss)) {
    std::printf("  FAIL  iki CE yolu ayni kaybi vermeli\n");
    ++failed;
  }

  // ---- 2. AdamW 124M ------------------------------------------------------------------------
  const std::size_t params = 124439808;  // GPT-2 124M (nexus_estimate ile ayni sayim)
  std::vector<float> p(params);
  std::vector<float> g(params);
  for (std::size_t i = 0; i < params; ++i) {
    p[i] = 0.02f * static_cast<float>(static_cast<std::int64_t>(i % 1999) - 999) / 999.0f;
    g[i] = 1.0e-3f * static_cast<float>(static_cast<std::int64_t>(i % 997) - 498) / 498.0f;
  }
  std::vector<float> m(params, 0.0f);
  std::vector<float> v(params, 0.0f);
  const fu::ParamSlot slot{p.data(), g.data(), m.data(), v.data(), params};
  fu::AdamWConfig config;
  config.lr = 6.0e-4;
  config.weight_decay = 0.1;
  config.max_grad_norm = 1.0;
  std::int64_t step = 0;
  const Stats adam_fused = measure([&] { (void)fu::adamw_multi_tensor({&slot, 1}, config, ++step); }, 1, 5);
  nexus_optim::AdamW<float>::Options native;
  native.lr = 6.0e-4;
  native.weight_decay = 0.1;
  nexus_optim::ParamGroupOptions group_options;
  group_options.learning_rate = 6.0e-4;
  nexus_optim::AdamW<float> reference(
      std::vector<nexus_optim::ParamGroup<float>>{
          nexus_optim::make_group<float>({p.data()}, {g.data()}, {params}, group_options)},
      native);
  const Stats adam_optim = measure([&] { reference.step(); }, 1, 5);
  print("adamw_foreach_fused (124M, +clip)", adam_fused);
  print("adamw_nexus_optim (124M, AVX2)", adam_optim);
  m.clear();
  m.shrink_to_fit();
  v.clear();
  v.shrink_to_fit();

  // ---- 3. RMSNorm vs LayerNorm --------------------------------------------------------------
  namespace ag = nexus::autograd;
  const std::int64_t rows = 4096;
  const std::int64_t width = 768;
  std::vector<float> xs(static_cast<std::size_t>(rows * width));
  for (float& value : xs) {
    value = normal(rng);
  }
  const ag::Variable x = ag::Variable::make(xs, ag::Shape{{rows, width}}, true);
  const ag::Variable gamma = ag::Variable::ones(ag::Shape{{width}}, true);
  const ag::Variable beta = ag::Variable::zeros(ag::Shape{{width}}, true);
  const ag::Variable ones = ag::Variable::ones(ag::Shape{{rows, width}}, false);
  const Stats rms = measure([&] {
    const ag::Variable y = fu::rms_norm(x, gamma, 1.0e-6f);
    y.backward(ones);
  });
  const Stats ln = measure([&] {
    const ag::Variable y = nexus::fused::fused_layernorm(x, gamma, beta, 1.0e-5f);
    y.backward(ones);
  });
  // Etkin bant: ileri x okuma + y yazma, geri g+x okuma + dx yazma = 5 x M x N x 4 bayt.
  const double bytes = 5.0 * static_cast<double>(rows * width) * 4.0;
  print("rms_norm fwd+bwd [4096,768]", rms);
  print("layer_norm fwd+bwd [4096,768]", ln);
  std::printf("  info  etkin bant: rms=%.2f GB/s  ln=%.2f GB/s\n", bytes / (rms.median * 1.0e6),
              bytes / (ln.median * 1.0e6));

  // ---- JSON ---------------------------------------------------------------------------------
  const std::filesystem::path dir(NEXUS_FAZ3_DOCS_DIR);
  std::filesystem::create_directories(dir);
  std::ofstream out(dir / "bench_phase3_training.json", std::ios::binary | std::ios::trunc);
  out << "{\n  \"bench\": \"bench_phase3_training\",\n  \"phase\": 3,\n";
  out << "  \"machine\": {\"cpu\": \"" << cpu_name() << "\", \"compiler\": \"MSVC " << _MSC_VER
      << "\", \"git_commit\": \"" << NEXUS_GIT_COMMIT << "\", \"threads\": 1},\n";
  out << "  \"protocol\": \"isinma + tekrar; medyan/min/max/std (BOLUM G)\",\n";
  out << "  \"results\": {\n";
  out << "    \"ce_shifted_fused\": " << stats_json(ce_fused) << ",\n";
  out << "    \"ce_legacy_copies\": " << stats_json(ce_legacy) << ",\n";
  out << "    \"ce_logit_bytes\": " << count * sizeof(float) << ",\n";
  out << "    \"ce_fused_extra_heap_bytes\": " << fused_extra << ",\n";
  out << "    \"ce_legacy_extra_heap_bytes\": " << legacy_extra << ",\n";
  out << "    \"adamw_124m_foreach_fused\": " << stats_json(adam_fused) << ",\n";
  out << "    \"adamw_124m_nexus_optim_avx2\": " << stats_json(adam_optim) << ",\n";
  out << "    \"rms_norm_4096x768_fwd_bwd\": " << stats_json(rms) << ",\n";
  out << "    \"layer_norm_4096x768_fwd_bwd\": " << stats_json(ln) << "\n  },\n";
  out << "  \"notes\": \"CPU tek is parcacigi. GPU (cuBLASLt/CUDA cekirdekleri) olculmedi; FAZ 3 GPU yolu acik is.\"\n}\n";
  std::printf("  info  json: %s\n", (dir / "bench_phase3_training.json").string().c_str());
  std::printf(failed == 0 ? "  OK    bench-faz3-egitim tamamlandi\n" : "  FAIL  bench-faz3-egitim\n");
  return failed == 0 ? 0 : 1;
}
