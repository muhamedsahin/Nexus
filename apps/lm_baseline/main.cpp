/**
 * @file main.cpp
 * @brief lm_baseline (FAZ 0): kucuk dil modeli taban cizgisi + durust olcum.
 *
 * Kullanim:
 *   lm_baseline --config docs/benchmarks/2026-10-23-faz0/lm_baseline_config.json
 *               [--json cikti.json] [--trace iz.json]
 *
 * Ne olcur, neyi uydurmaz:
 *  - Adim sureleri ScopedTimer ile gercek steady_clock olcumudur.
 *  - Parametre sayisi model.parameters() uzerinden GERCEK sayimdir;
 *    estimate.hpp formulu yalnizca karsilastirma icin ayrica raporlanir.
 *  - MFU yalnizca yapilandirmada aygit zirve FLOP/s (device.peak_tflops)
 *    verildiyse hesaplanir; verilmezse null yazilir (uydurulmaz).
 *  - MFU ust siniri roofline'dan: min(zirve, yogunluk*bant) / zirve.
 *  - Surec doruk bellegi Windows'ta PeakWorkingSetSize, POSIX'te VmHWM'dir;
 *    platform disinda null. Muhasebe defteri yalnizca nexus bellek kapisindan
 *    gecen ayrimlari sayar (nexus_model tensorlari kendi ayiricisini kullanir;
 *    birlesik sayim FAZ 1'de k64 tek ayiriciyla gelir).
 *  - Parcalanma orani FAZ 1 k64 havuzundan (reserved/current) olculur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/nexus.hpp"
#include "nexus/prof/counters.hpp"
#include "nexus/prof/memory_ledger.hpp"
#include "nexus/prof/timer.hpp"
#include "nexus/prof/trace.hpp"
#include "nexus/train/language.hpp"
#include "estimate.hpp"
#include "json_mini.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h psapi.h'dan once gelmeli (BOOL/DWORD tanimlari).
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using nexus::estimate::DTypePlan;
using nexus::estimate::Plan;

int g_failed = 0;

void expect(bool condition, const char* name) {
  if (condition) {
    std::printf("  PASS  %s\n", name);
    return;
  }
  std::printf("  FAIL  %s\n", name);
  ++g_failed;
}

// Surec doruk calisma kumesi (bayt). Platform destegi yoksa -1 (null rapor).
[[nodiscard]] long long peak_working_set_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    return static_cast<long long>(counters.PeakWorkingSetSize);
  }
  return -1;
#elif defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      long long kb = 0;
      std::istringstream rest(line.substr(6));
      rest >> kb;
      return kb * 1024;
    }
  }
  return -1;
#else
  return -1;
#endif
}

// Deterministik LCG token ureteci: ayni tohum -> ayni veri (tekrarlanabilirlik).
class TokenGenerator {
 public:
  explicit TokenGenerator(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return state_ >> 33;
  }

 private:
  std::uint64_t state_;
};

void usage() {
  std::printf("lm_baseline (FAZ 0)\n"
              "  --config <dosya.json>   zorunlu yapilandirma\n"
              "  --json <dosya>          sonuc JSON yolu (verilmezse stdout)\n"
              "  --trace <dosya>         chrome iz yolu (ops.)\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path;
  std::string json_path;
  std::string trace_path;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--config" && i + 1 < argc) {
        config_path = argv[++i];
      } else if (argument == "--json" && i + 1 < argc) {
        json_path = argv[++i];
      } else if (argument == "--trace" && i + 1 < argc) {
        trace_path = argv[++i];
      } else {
        usage();
        return 2;
      }
    }
    if (config_path.empty()) {
      usage();
      return 2;
    }

    const nexus::jsonmini::Value config = nexus::jsonmini::parse_file(config_path);
    const nexus::jsonmini::Value& model_config = config.at("model");
    const nexus::jsonmini::Value& data_config = config.at("data");
    const nexus::jsonmini::Value& train_config = config.at("train");

    const std::int64_t vocab = nexus::jsonmini::require_int(model_config, "vocab");
    const std::int64_t dim = nexus::jsonmini::require_int(model_config, "dim");
    const std::int64_t heads = nexus::jsonmini::require_int(model_config, "heads");
    const std::int64_t ff = nexus::jsonmini::require_int(model_config, "ff");
    const std::int64_t layers = nexus::jsonmini::require_int(model_config, "layers");
    const std::int64_t batch = nexus::jsonmini::require_int(data_config, "batch");
    const std::int64_t time = nexus::jsonmini::require_int(data_config, "time");
    const std::int64_t seed = nexus::jsonmini::get_int(data_config, "seed", 12345);
    const float lr = static_cast<float>(nexus::jsonmini::require_number(train_config, "lr"));
    const float weight_decay = static_cast<float>(nexus::jsonmini::get_number(train_config, "weight_decay", 0.0));
    const std::int64_t steps = nexus::jsonmini::require_int(train_config, "steps");

    double peak_tflops = 0.0;
    double mem_gbps = 0.0;
    if (config.has("device")) {
      const nexus::jsonmini::Value& device = config.at("device");
      peak_tflops = nexus::jsonmini::get_number(device, "peak_tflops", 0.0);
      mem_gbps = nexus::jsonmini::get_number(device, "mem_gbps", 0.0);
    }

    if (vocab <= 0 || dim <= 0 || heads <= 0 || ff <= 0 || layers <= 0 || batch <= 0 || time < 2 || steps <= 0) {
      throw std::invalid_argument("lm_baseline: yapilandirma alanlari pozitif olmali (time >= 2)");
    }
    if (dim % heads != 0) {
      throw std::invalid_argument("lm_baseline: dim % heads == 0 olmali");
    }

    // --- Model: Embedding + sinus konum + layers x CausalTransformerBlock + Linear head
    // (bugunku Nexus dil modeli tarifi; tests/test_nxs.cpp regresyon testiyle ayni yapi).
    nexus::nn::Sequential model;
    model.add<nexus::nn::Embedding>(static_cast<int>(vocab), static_cast<int>(dim), -1);
    model.add<nexus::nn::PositionalEncoding>(static_cast<int>(dim), static_cast<int>(time), false);
    for (std::int64_t layer = 0; layer < layers; ++layer) {
      model.add<nexus::nn::CausalTransformerBlock>(static_cast<int>(dim), static_cast<int>(heads), static_cast<int>(ff));
    }
    model.add<nexus::nn::Linear>(static_cast<std::size_t>(dim), static_cast<std::size_t>(vocab), true);

    // GERCEK parametre sayimi (tahmin degil).
    std::int64_t counted_params = 0;
    for (const nexus_model::Parameter* parameter : model.parameters()) {
      counted_params += static_cast<std::int64_t>(parameter->data.numel());
    }
    expect(counted_params > 0, "parametre sayimi pozitif");

    // Formulle karsilastirma (tahmin; bagli head yok, bias decoder'da var kabul).
    nexus::estimate::ModelShape shape;
    shape.n_layer = layers;
    shape.d_model = dim;
    shape.n_head = heads;
    shape.n_kv_head = heads;
    shape.d_ff = ff;
    shape.vocab = vocab;
    shape.seq_len = time;
    shape.micro_batch = batch;
    shape.learned_pos = 0;
    shape.tie_head = false;
    shape.ffn_swiglu = false;
    shape.bias = true;
    shape.rmsnorm = false;
    const auto param_estimate = nexus::estimate::count_parameters(shape);

    // --- Veri: deterministik LCG (tekrarlanabilir taban cizgisi) ------------------
    nexus::TokenRows rows;
    rows.batch = static_cast<int>(batch);
    rows.time = static_cast<int>(time);
    rows.values.resize(static_cast<std::size_t>(batch * time));
    TokenGenerator generator(static_cast<std::uint64_t>(seed));
    for (std::int64_t i = 0; i < batch * time; ++i) {
      rows.values[static_cast<std::size_t>(i)] = static_cast<int>(generator.next() % static_cast<std::uint64_t>(vocab));
    }

    // --- Egitim + olcum -------------------------------------------------------------
    nexus::LanguageTrainer trainer(model, lr, weight_decay);
    auto& profiler = nexus::prof::profiler();
    auto& ledger = nexus::prof::ledger();
    auto& counters = nexus::prof::counters();
    profiler.reset();
    counters.reset();
    ledger.reset_peak();
    profiler.set_enabled(true);

    std::vector<float> losses;
    losses.reserve(static_cast<std::size_t>(steps));
    for (std::int64_t step = 0; step < steps; ++step) {
      const nexus::prof::ScopedTimer step_timer("lm_baseline.step", "train");
      losses.push_back(trainer.fit(rows, 1));
    }
    profiler.set_enabled(false);

    // --- Metrikler -------------------------------------------------------------------
    const auto totals = profiler.totals();
    const auto step_total = totals.find("lm_baseline.step");
    if (step_total == totals.end() || step_total->second.calls == 0) {
      throw std::runtime_error("lm_baseline: adim zamanlamasi kaydedilmedi");
    }
    const double mean_step_ns =
        static_cast<double>(step_total->second.total_ns) / static_cast<double>(step_total->second.calls);
    const double min_step_ns = static_cast<double>(step_total->second.min_ns);
    const double max_step_ns = static_cast<double>(step_total->second.max_ns);
    const double tokens_per_step = static_cast<double>(batch * time);
    // Ortalama adim suresi tek adimdir: token/s = adim basina token / ortalama adim saniyesi.
    const double tokens_per_second = tokens_per_step / (mean_step_ns * 1.0e-9);

    const Plan plan;
    const auto flop_estimate = nexus::estimate::estimate_flops(shape, plan);
    counters.add_flops("lm_baseline.estimated_flops",
                       static_cast<std::int64_t>(flop_estimate.flops_per_step * static_cast<double>(steps)));

    // MFU: yalnizca aygit zirvesi verildiyse. Verilmediyse null (uydurulmaz).
    bool has_mfu = peak_tflops > 0.0;
    double mfu = 0.0;
    double mfu_ceiling = 0.0;
    if (has_mfu) {
      const double peak_flops = peak_tflops * 1.0e12;
      mfu = flop_estimate.flops_per_step / (mean_step_ns * 1.0e-9 * peak_flops);
      if (mem_gbps > 0.0) {
        const auto memory = nexus::estimate::estimate_memory(shape, DTypePlan{}, plan);
        const double bytes_per_step = static_cast<double>(memory.activation_bytes + memory.logits_bytes) +
                                      3.0 * static_cast<double>(flop_estimate.matmul_params) * 4.0;
        const auto roof = nexus::estimate::roofline_bound(flop_estimate.flops_per_step, bytes_per_step, peak_flops,
                                                          mem_gbps * 1.0e9);
        mfu_ceiling = roof.ceiling_flops_per_s / peak_flops;
      } else {
        mfu_ceiling = 1.0;  // bant verisi yoksa ust sinir hesap zirvesidir
      }
    }

    const long long peak_ws = peak_working_set_bytes();
    const auto ledger_snap = ledger.snapshot();

    bool losses_finite = true;
    for (const float value : losses) {
      if (!std::isfinite(value)) {
        losses_finite = false;
      }
    }
    expect(losses_finite, "kayip egrisi sonlu");
    expect(!losses.empty() && losses.back() < losses.front(), "taban cizgisi kaybi dusuyor");
    std::printf("  info  kayip %.4f -> %.4f  adim=%.2fms  token/s=%.1f  param=%lld\n", losses.front(), losses.back(),
                mean_step_ns / 1.0e6, tokens_per_second, static_cast<long long>(counted_params));

    // --- JSON cikti --------------------------------------------------------------------
    std::ostringstream out;
    out.precision(17);
    out << "{\n  \"app\": \"lm_baseline\",\n  \"phase\": 0,\n";
    out << "  \"config\": " << "{\"model\":{\"vocab\":" << vocab << ",\"dim\":" << dim << ",\"heads\":" << heads
        << ",\"ff\":" << ff << ",\"layers\":" << layers << "},\"data\":{\"batch\":" << batch << ",\"time\":" << time
        << ",\"seed\":" << seed << "},\"train\":{\"lr\":" << lr << ",\"weight_decay\":" << weight_decay
        << ",\"steps\":" << steps << "}},\n";
    out << "  \"params\": {\"counted\": " << counted_params << ", \"estimated\": " << param_estimate.total
        << ", \"note\": \"counted gercek sayim; estimated formule gore (decoder bias farki olabilir)\"},\n";
    out << "  \"loss_curve\": [";
    for (std::size_t i = 0; i < losses.size(); ++i) {
      if (i != 0) {
        out << ", ";
      }
      out << losses[i];
    }
    out << "],\n";
    out << "  \"timing\": {\"steps\": " << steps << ", \"mean_step_ns\": " << mean_step_ns << ", \"min_step_ns\": "
        << min_step_ns << ", \"max_step_ns\": " << max_step_ns << ", \"tokens_per_sec\": " << tokens_per_second
        << "},\n";
    out << "  \"flops\": {\"estimated_per_step\": " << flop_estimate.flops_per_step
        << ", \"note\": \"Kaplan/Chinchilla 6N + PaLM dikkat terimi; tahmindir\"},\n";
    if (has_mfu) {
      out << "  \"mfu\": {\"measured\": " << mfu << ", \"ceiling\": " << mfu_ceiling << "},\n";
    } else {
      out << "  \"mfu\": null,\n";
    }
    out << "  \"memory\": {\"process_peak_working_set_bytes\": " << peak_ws
        << ", \"ledger_peak_bytes\": " << ledger_snap.peak_bytes << ", \"ledger_current_bytes\": "
        << ledger_snap.current_bytes
        << ", \"note\": \"defter yalnizca nexus kapi ayrimlarini sayar; birlesik sayim FAZ 1\"},\n";
    // FAZ 1 k64 havuzu: reserved/current. Yalniz nexus kapisindan gecen host ayirmalarini kapsar;
    // nexus_model tensorleri kendi ayiricisini kullanir (birlesik tensor FAZ 1.B/1.G acik is).
    out << "  \"fragmentation\": {\"ratio\": " << nexus::memory_fragmentation_ratio()
        << ", \"note\": \"k64 havuzu reserved/current; -1: canli ayirma yok, yalniz onbellek\"}\n";
    out << "}\n";

    if (!json_path.empty()) {
      std::ofstream file(json_path, std::ios::binary | std::ios::trunc);
      if (!file) {
        throw std::runtime_error("lm_baseline: json dosyasi acilamadi: " + json_path);
      }
      file << out.str();
      std::printf("  info  json yazildi: %s\n", json_path.c_str());
    } else {
      std::cout << out.str();
    }

    if (!trace_path.empty()) {
      const bool dumped = nexus::prof::dump_chrome_trace(trace_path);
      expect(dumped, "chrome izi yazildi");
    }
  } catch (const std::exception& error) {
    std::printf("  FAIL  istisna: %s\n", error.what());
    return 1;
  }

  if (g_failed != 0) {
    std::printf("  FAIL  toplam %d basarisiz denetim\n", g_failed);
    return 1;
  }
  std::printf("  OK    lm-taban-cizgisi tamamlandi\n");
  return 0;
}
