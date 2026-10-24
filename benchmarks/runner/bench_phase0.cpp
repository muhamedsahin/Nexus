/**
 * @file bench_phase0.cpp
 * @brief FAZ 0: olcum altyapisinin kendi maliyeti ve referans cekirdek hizi.
 *
 * Olculenler (hepsi gercek zamanla, uydurma metrik yok):
 *  - Bellek kapisi ayirma+serbest birakma: ns/islem (defter kaydi dahil).
 *  - Defter anlik goruntusu: ns/cagri.
 *  - ScopedTimer maliyeti: profiler kapali ve acikken ns/kurulus-yikilis.
 *  - Sayac add_flops: ns/cagri.
 *  - Referans naif FP32 matmul (256^3): olculen GFLOP/s. Aygit zirvesi
 *    verilmedigi icin MFU iddiasi uretilmez; yalnizca olculen hiz raporlanir.
 *  - Chrome iz dokumu: yazma suresi ve dosya dogrulamasi.
 * Sonuclar docs/benchmarks/2026-10-23-faz0/bench_phase0.json'a yazilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/memory.hpp"
#include "nexus/prof/counters.hpp"
#include "nexus/prof/memory_ledger.hpp"
#include "nexus/prof/timer.hpp"
#include "nexus/prof/trace.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

// bench_phase0 kendi kendine yeterlidir: tests/support'a baglanmaz, boylece
// NEXUS_BUILD_TESTS=OFF iken de derlenir. Rapor bicimi nexus_test ile aynidir.
namespace nexus_test {
namespace {
int g_failed = 0;
}

void expect(bool condition, std::string_view name) {
  if (condition) {
    std::printf("  PASS  %.*s\n", static_cast<int>(name.size()), name.data());
    return;
  }
  std::printf("  FAIL  %.*s\n", static_cast<int>(name.size()), name.data());
  ++g_failed;
}

void info(std::string_view key, double value) {
  std::printf("  info  %.*s=%.6g\n", static_cast<int>(key.size()), key.data(), value);
}

void section(std::string_view title) {
  std::printf("== %.*s ==\n", static_cast<int>(title.size()), title.data());
}

[[nodiscard]] int finish(std::string_view suite) {
  if (g_failed != 0) {
    std::printf("  FAIL  %.*s: %d basarisiz denetim\n", static_cast<int>(suite.size()), suite.data(), g_failed);
    return 1;
  }
  std::printf("  OK    %.*s tamamlandi\n", static_cast<int>(suite.size()), suite.data());
  return 0;
}
}  // namespace nexus_test

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double ns_since(const Clock::time_point& start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count() * 1.0;
}

// Referans naif matmul: C[m,n] = sum_k A[m,k]*B[k,n]. Bilincli olarak sade;
// FAZ 1+ cekirdekleri bu taban cizgisine karsi kiyaslanir.
void naive_matmul(const float* a, const float* b, float* c, int n) {
  for (int m = 0; m < n; ++m) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < n; ++k) {
        sum += a[static_cast<std::size_t>(m * n + k)] * b[static_cast<std::size_t>(k * n + col)];
      }
      c[static_cast<std::size_t>(m * n + col)] = sum;
    }
  }
}

}  // namespace

int main() {
  nexus_test::section("faz0 olcum altyapisi benchmarki");
  auto& ledger = nexus::prof::ledger();
  auto& profiler = nexus::prof::profiler();
  auto& counters = nexus::prof::counters();
  profiler.reset();
  counters.reset();
  ledger.reset_all();

  // --- 1. Bellek kapisi: ayirma + serbest birakma -----------------------------
  double alloc_ns_per_op = 0.0;
  {
    constexpr int kIterations = 100000;
    const auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      void* block = nexus::allocate_aligned(4096);
      nexus::release_owned(block);
    }
    alloc_ns_per_op = ns_since(start) / static_cast<double>(kIterations);
    nexus_test::info("kapi_ns_islem", alloc_ns_per_op);
    nexus_test::expect(ledger.snapshot().live_blocks == 0, "benchmark sonrasi defter dengede");
  }

  // --- 2. Defter anlik goruntusu ----------------------------------------------
  double snapshot_ns = 0.0;
  {
    constexpr int kIterations = 20000;
    const auto start = Clock::now();
    std::int64_t accumulator = 0;
    for (int i = 0; i < kIterations; ++i) {
      accumulator += ledger.snapshot().current_bytes;
    }
    snapshot_ns = ns_since(start) / static_cast<double>(kIterations);
    nexus_test::info("defter_snapshot_ns", snapshot_ns);
    nexus_test::expect(accumulator == 0, "snapshot degeri tutarli");
  }

  // --- 3. ScopedTimer maliyeti: kapali ve acik ---------------------------------
  double timer_disabled_ns = 0.0;
  double timer_enabled_ns = 0.0;
  {
    constexpr int kIterations = 50000;
    profiler.set_enabled(false);
    auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      nexus::prof::ScopedTimer timer("bench.disabled");
    }
    timer_disabled_ns = ns_since(start) / static_cast<double>(kIterations);

    profiler.set_enabled(true);
    start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      nexus::prof::ScopedTimer timer("bench.enabled");
    }
    timer_enabled_ns = ns_since(start) / static_cast<double>(kIterations);
    profiler.set_enabled(false);
    nexus_test::info("zamanlayici_kapali_ns", timer_disabled_ns);
    nexus_test::info("zamanlayici_acik_ns", timer_enabled_ns);
    nexus_test::expect(profiler.totals().count("bench.disabled") == 0, "kapali zamanlayici kayit yazmadi");
    nexus_test::expect(profiler.totals().count("bench.enabled") == 1, "acik zamanlayici toplandi");
    profiler.reset();
  }

  // --- 4. Sayac verimi -----------------------------------------------------------
  double counter_add_ns = 0.0;
  {
    constexpr int kIterations = 200000;
    const auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      counters.add_flops("bench.counter", 64);
    }
    counter_add_ns = ns_since(start) / static_cast<double>(kIterations);
    nexus_test::info("sayac_add_ns", counter_add_ns);
    nexus_test::expect(counters.total_flops() == 64LL * kIterations, "sayac toplami dogru");
  }

  // --- 5. Referans naif FP32 matmul ---------------------------------------------
  double matmul_gflops = 0.0;
  {
    constexpr int kN = 256;
    constexpr int kReps = 3;
    const std::size_t count = static_cast<std::size_t>(kN) * static_cast<std::size_t>(kN);
    std::vector<float> a(count, 1.0f);
    std::vector<float> b(count, 1.0f);
    std::vector<float> c(count, 0.0f);
    const auto start = Clock::now();
    for (int rep = 0; rep < kReps; ++rep) {
      naive_matmul(a.data(), b.data(), c.data(), kN);
    }
    const double elapsed_ns = ns_since(start);
    const double flops = 2.0 * static_cast<double>(kN) * static_cast<double>(kN) * static_cast<double>(kN) *
                         static_cast<double>(kReps);
    matmul_gflops = flops / elapsed_ns;  // flop/ns == GFLOP/s
    counters.add_flops("bench.matmul", static_cast<std::int64_t>(flops));
    nexus_test::info("referans_matmul_gflops", matmul_gflops);
    bool correct = true;
    for (std::size_t i = 0; i < count && correct; ++i) {
      correct = c[i] == static_cast<float>(kN);
    }
    nexus_test::expect(correct, "referans matmul sonucu dogru");
  }

  // --- 6. Chrome iz dokumu --------------------------------------------------------
  double trace_dump_us = 0.0;
  bool trace_written = false;
  {
    profiler.reset();
    profiler.set_enabled(true);
    {
      nexus::prof::ScopedTimer step("bench.simulated_step", "train");
      {
        nexus::prof::ScopedTimer forward("bench.forward", "cpu_op");
        (void)forward.elapsed_ns();
      }
      {
        nexus::prof::ScopedTimer backward("bench.backward", "cpu_op");
        (void)backward.elapsed_ns();
      }
    }
    profiler.set_enabled(false);

    const std::filesystem::path docs_dir(NEXUS_FAZ0_DOCS_DIR);
    std::error_code ec;
    std::filesystem::create_directories(docs_dir, ec);
    const std::filesystem::path trace_path = docs_dir / "bench_phase0_trace.json";
    const auto start = Clock::now();
    trace_written = nexus::prof::dump_chrome_trace(trace_path.string());
    trace_dump_us = ns_since(start) / 1000.0;
    nexus_test::info("iz_dokumu_us", trace_dump_us);
    nexus_test::expect(trace_written, "iz dosyasi yazildi");
    if (trace_written) {
      std::ifstream back(trace_path, std::ios::binary);
      const std::string content((std::istreambuf_iterator<char>(back)), std::istreambuf_iterator<char>());
      nexus_test::expect(content.find("bench.simulated_step") != std::string::npos, "iz adim araligini icerir");
    }
  }

  // --- 7. Makinayla okunur JSON ------------------------------------------------------
  {
    const std::filesystem::path docs_dir(NEXUS_FAZ0_DOCS_DIR);
    std::error_code ec;
    std::filesystem::create_directories(docs_dir, ec);
    const std::filesystem::path json_path = docs_dir / "bench_phase0.json";
    std::ofstream out(json_path, std::ios::binary | std::ios::trunc);
    if (out) {
      out.precision(17);
      out << "{\n";
      out << "  \"bench\": \"bench_phase0\",\n  \"phase\": 0,\n";
      out << "  \"results\": {\n";
      out << "    \"gate_alloc_free_ns_per_op\": " << alloc_ns_per_op << ",\n";
      out << "    \"ledger_snapshot_ns\": " << snapshot_ns << ",\n";
      out << "    \"timer_disabled_ns\": " << timer_disabled_ns << ",\n";
      out << "    \"timer_enabled_ns\": " << timer_enabled_ns << ",\n";
      out << "    \"counter_add_flops_ns\": " << counter_add_ns << ",\n";
      out << "    \"reference_matmul_gflops\": " << matmul_gflops << ",\n";
      out << "    \"trace_dump_us\": " << trace_dump_us << "\n  },\n";
      out << "  \"notes\": {\n";
      out << "    \"mfu\": \"olculmedi: aygit zirve FLOP/s verilmeden MFU iddiasi uretilmez\",\n";
      out << "    \"fragmentation\": \"FAZ 1 k64 havuzu reserved/current ile olcecek\",\n";
      out << "    \"matmul\": \"naif FP32 256^3 referans; cekirdek kiyaslari FAZ 1+\"\n  }\n";
      out << "}\n";
      out.flush();
      nexus_test::expect(static_cast<bool>(out), "bench JSON yazildi");
      nexus_test::info("json_yazildi", 1);
    } else {
      nexus_test::expect(false, "bench JSON yazilamadi");
    }
  }

  return nexus_test::finish("bench-faz0-olcum-altyapisi");
}
