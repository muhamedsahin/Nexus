/**
 * @file bench_phase1.cpp
 * @brief FAZ 1: k64 ayiricinin isabet/isabet-kaci maliyeti, parcalanma ve empty_cache.
 *
 * Olculenler (hepsi gercek steady_clock):
 *  - Onbellek isabeti: isinmis havuza allocate+deallocate ns/islem.
 *  - Onbellek isabet kaci: her seferinde empty_cache ile taze OS ayirma.
 *  - empty_cache maliyeti: dolu onbellegi OS'a geri verme suresi.
 *  - Karisik is yuku altinda reserved/current ve parcalanma orani.
 *  - nexus bellek kapisinin (allocate_aligned/release_owned) islem/sn.
 * Sonuclar docs/benchmarks/2026-10-23-faz1/bench_phase1.json'a yazilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/allocator.hpp"
#include "nexus/core/memory.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int g_failed = 0;

void expect(bool condition, std::string_view name) {
  if (condition) {
    std::printf("  PASS  %.*s\n", static_cast<int>(name.size()), name.data());
  } else {
    std::printf("  FAIL  %.*s\n", static_cast<int>(name.size()), name.data());
    ++g_failed;
  }
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

[[nodiscard]] double ns_since(const Clock::time_point& start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count() * 1.0;
}

}  // namespace

int main() {
  auto& pool = nexus::host_allocator();
  section("faz1 k64 ayirici benchmarki");

  // --- Onbellek isabeti (isinmis havuz) --------------------------------------
  double hit_ns = 0.0;
  {
    constexpr int kIterations = 300000;
    pool.reset();
    void* warm = pool.allocate(1024);  // 1024 sinifini isit
    pool.deallocate(warm, 1024);
    const auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      void* block = pool.allocate(1024);
      pool.deallocate(block, 1024);
    }
    hit_ns = ns_since(start) / static_cast<double>(kIterations);
    info("onbellek_isabet_ns_islem", hit_ns);
    expect(pool.stats().current_bytes == 0, "isabet kosusu dengede");
  }

  // --- Onbellek isabet kaci (her seferinde taze OS) ---------------------------
  double miss_ns = 0.0;
  {
    constexpr int kIterations = 50000;
    const auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      void* block = pool.allocate(1024);
      pool.deallocate(block, 1024);
      pool.empty_cache();  // bir sonraki ayirmayi OS'a zorlar
    }
    miss_ns = ns_since(start) / static_cast<double>(kIterations);
    info("onbellek_kaci_ns_islem", miss_ns);
  }

  // --- empty_cache maliyeti ------------------------------------------------------
  double empty_cache_ns = 0.0;
  {
    pool.reset();
    constexpr int kBlocks = 20000;
    std::vector<void*> blocks;
    blocks.reserve(kBlocks);
    for (int i = 0; i < kBlocks; ++i) {
      blocks.push_back(pool.allocate(256));
    }
    for (void* block : blocks) {
      pool.deallocate(block, 256);
    }
    const auto start = Clock::now();
    pool.empty_cache();
    empty_cache_ns = ns_since(start);
    info("empty_cache_ns", empty_cache_ns);
    expect(pool.stats().reserved_bytes == 0, "empty_cache her seyi geri verdi");
  }

  // --- Karisik is yuku altinda parcalanma ---------------------------------------
  double frag_peak = 0.0;
  std::uint64_t reserved_peak = 0;
  std::uint64_t current_peak = 0;
  {
    pool.reset();
    constexpr std::size_t kSizes[] = {64, 128, 256, 512, 1024, 4096};
    constexpr std::size_t kSizeCount = sizeof(kSizes) / sizeof(kSizes[0]);
    std::vector<void*> blocks;
    for (std::size_t round = 0; round < 64; ++round) {
      for (std::size_t size : kSizes) {
        blocks.push_back(pool.allocate(size));
      }
    }
    // block[i], kSizes[i % kSizeCount] ile ayrildi; yarisini deterministik serbest birak.
    for (std::size_t i = 0; i < blocks.size(); i += 2) {
      pool.deallocate(blocks[i], kSizes[i % kSizeCount]);
    }
    auto s = pool.stats();
    reserved_peak = s.reserved_bytes;
    current_peak = s.current_bytes;
    frag_peak = pool.fragmentation_ratio();
    info("parcalanma_orani", frag_peak);
    info("reserved_bayt", static_cast<double>(reserved_peak));
    info("current_bayt", static_cast<double>(current_peak));
    expect(s.current_bytes > 0 && s.reserved_bytes >= s.current_bytes, "reserved >= current");
    expect(frag_peak >= 1.0, "parcalanma orani >= 1.0");
    // Temizlik: kalan bloklari serbest birak ve havuzu OS'a geri ver.
    for (std::size_t i = 1; i < blocks.size(); i += 2) {
      pool.deallocate(blocks[i], kSizes[i % kSizeCount]);
    }
    pool.empty_cache();
  }

  // --- Kapidan islem/sn -------------------------------------------------------------
  double gate_ops_per_sec = 0.0;
  {
    constexpr int kIterations = 1000000;
    pool.reset();
    const auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      void* block = nexus::allocate_aligned(64);
      nexus::release_owned(block);
    }
    const double elapsed_s = ns_since(start) / 1.0e9;
    gate_ops_per_sec = static_cast<double>(kIterations) / elapsed_s;
    info("kapi_islem_sn", gate_ops_per_sec);
    expect(pool.stats().current_bytes == 0, "kapi kosusu dengede");
    pool.reset();
  }

  // --- Makinayla okunur JSON ---------------------------------------------------------
  {
    const std::filesystem::path docs_dir(NEXUS_FAZ1_DOCS_DIR);
    std::error_code ec;
    std::filesystem::create_directories(docs_dir, ec);
    const std::filesystem::path json_path = docs_dir / "bench_phase1.json";
    std::ofstream out(json_path, std::ios::binary | std::ios::trunc);
    if (out) {
      out.precision(17);
      out << "{\n  \"bench\": \"bench_phase1\",\n  \"phase\": 1,\n";
      out << "  \"results\": {\n";
      out << "    \"cache_hit_ns_per_op\": " << hit_ns << ",\n";
      out << "    \"cache_miss_ns_per_op\": " << miss_ns << ",\n";
      out << "    \"empty_cache_ns\": " << empty_cache_ns << ",\n";
      out << "    \"fragmentation_ratio_peak\": " << frag_peak << ",\n";
      out << "    \"reserved_peak_bytes\": " << reserved_peak << ",\n";
      out << "    \"current_peak_bytes\": " << current_peak << ",\n";
      out << "    \"gate_ops_per_sec\": " << gate_ops_per_sec << "\n  },\n";
      out << "  \"notes\": {\n";
      out << "    \"fragmentation\": \"reserved/current; FAZ 1 k64 havuzundan olculur\",\n";
      out << "    \"bins\": \"power-of-two boyut siniflari, 64 bayt taban\",\n";
      out << "    \"device\": \"FAZ 1 yalnizca AlignedHost havuzu; cihaz havuzu sonraki faz\"\n  }\n";
      out << "}\n";
      out.flush();
      expect(static_cast<bool>(out), "bench JSON yazildi");
    } else {
      expect(false, "bench JSON yazilamadi");
    }
  }

  return finish("bench-faz1-k64-ayirici");
}
