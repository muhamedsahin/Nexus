/**
 * @file test_phase0_prof.cpp
 * @brief FAZ 0: bellek muhasebe defteri, sayaclari, kapsamli zamanlayici ve Chrome izi.
 *
 * Dogrulanan sozlesmeler:
 *  - Ayrimlar nexus bellek kapisindan gecer ve deftere etiketleriyle yazilir.
 *  - AllocationLabelScope ic ice kurulur, cikista eski etiketi geri koyar.
 *  - Serbest birakma ayni etiket/uzaydan dusurur; doruklar korunur.
 *  - Cok is parcacikli ayirma-serbest birakma defteri bozmaz (kilit dogrulugu).
 *  - Profiler kapaliyken ScopedTimer kayit yazmaz; acikken ice ice araliklar
 *    dogru toplanir (dis >= ic).
 *  - Chrome iz JSON'u makinayla okunur; dosyaya dokum basarili/basarisiz
 *    durumlarinda dogru bool dondurur.
 *  - NVTX: yalnizca derleme zamaninda gercek durum raporlanir; bayrak
 *    kapaliyken uydurma isaret uretilmez.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/memory.hpp"
#include "nexus/prof/counters.hpp"
#include "nexus/prof/memory_ledger.hpp"
#include "nexus/prof/nvtx.hpp"
#include "nexus/prof/timer.hpp"
#include "nexus/prof/trace.hpp"
#include "nexus_test.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {

using nexus::prof::AllocationLabelScope;
using nexus::prof::MemoryLabel;

[[nodiscard]] std::size_t idx(MemoryLabel label) noexcept { return static_cast<std::size_t>(label); }

void busy_wait(std::chrono::microseconds duration) {
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < duration) {
  }
}

}  // namespace

int main() {
  nexus_test::section("bellek muhasebe defteri");
  auto& ledger = nexus::prof::ledger();
  ledger.reset_all();
  nexus_test::expect(ledger.current_bytes() == 0 && ledger.peak_bytes() == 0, "sifirlama defteri temizler");

  {
    AllocationLabelScope scope(MemoryLabel::Parameter);
    void* block = nexus::allocate_aligned(4096);
    nexus_test::expect(block != nullptr, "kapi 4096 bayt ayirdi");
    const auto snap = ledger.snapshot();
    nexus_test::expect(snap.by_label[idx(MemoryLabel::Parameter)].current_bytes == 4096, "parametre etiketi artti");
    nexus_test::expect(snap.by_space[0].current_bytes == 4096, "AlignedHost uzayi artti");
    nexus_test::expect(snap.current_bytes == 4096 && snap.live_blocks == 1, "genel toplam ve blok sayisi");

    {
      AllocationLabelScope inner(MemoryLabel::Gradient);
      void* grad = nexus::allocate_aligned(1024);
      const auto nested = ledger.snapshot();
      nexus_test::expect(nested.by_label[idx(MemoryLabel::Gradient)].current_bytes == 1024, "ic kapsam etiketi");
      nexus_test::expect(nested.by_label[idx(MemoryLabel::Parameter)].current_bytes == 4096, "dis kapsam korunur");
      nexus_test::expect(nested.peak_bytes == 5120, "es zamanli doruk");
      nexus::release_owned(grad);
    }
    nexus_test::expect(nexus::prof::current_allocation_label() == MemoryLabel::Parameter,
                       "kapsam cikisi etiketi geri koyar");
    nexus::release_owned(block);
  }

  {
    const auto snap = ledger.snapshot();
    nexus_test::expect(snap.current_bytes == 0 && snap.live_blocks == 0, "serbest birakma dusurur");
    nexus_test::expect(snap.peak_bytes == 5120, "genel doruk hatirlanir");
    nexus_test::expect(snap.by_label[idx(MemoryLabel::Parameter)].peak_bytes == 4096, "etiket dorugu parametre");
    nexus_test::expect(snap.by_label[idx(MemoryLabel::Gradient)].peak_bytes == 1024, "etiket dorugu gradyan");
    nexus_test::expect(snap.total_allocs == 2, "toplam ayirma sayisi");
  }

  {
    void* temp = nexus::allocate_aligned(2048);
    const auto snap = ledger.snapshot();
    nexus_test::expect(snap.by_label[idx(MemoryLabel::Temporary)].current_bytes == 2048, "kapsamsiz etiket Temporary");
    nexus::release_owned(temp);
  }

  ledger.reset_peak();
  nexus_test::expect(ledger.peak_bytes() == ledger.current_bytes(), "reset_peak dorugu anliga ceker");


  {
    ledger.reset_all();
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
      workers.emplace_back([]() {
        for (int i = 0; i < 250; ++i) {
          AllocationLabelScope scope(MemoryLabel::OptimizerState);
          void* block = nexus::allocate_aligned(64);
          nexus::release_owned(block);
        }
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
    const auto snap = ledger.snapshot();
    nexus_test::expect(snap.current_bytes == 0 && snap.live_blocks == 0, "cok is parcacigi: kalan yok");
    nexus_test::expect(snap.by_label[idx(MemoryLabel::OptimizerState)].total_allocs == 1000,
                       "cok is parcacigi: 1000 ayirma");
    nexus_test::expect(snap.peak_bytes >= 64, "cok is parcacigi: doruk en az bir blok");
  }

  try {
    AllocationLabelScope scope(MemoryLabel::CommBuffer);
    void* pinned = nexus::allocate_pinned(4096);
    const auto snap = ledger.snapshot();
    nexus_test::expect(snap.by_label[idx(MemoryLabel::CommBuffer)].current_bytes == 4096, "pinned etiketli ayirma");
    nexus_test::expect(snap.by_space[1].current_bytes == 4096, "PinnedHost uzayi");
    nexus::release_owned(pinned);
  } catch (const std::exception&) {
    // VirtualLock ayricaligi olmayan sistemlerde kapi istisna atabilir; bu bir hata degil.
    nexus_test::info("pinned_atlandi", 1);
  }

  nexus_test::section("sayaclar");
  auto& counters = nexus::prof::counters();
  counters.reset();
  counters.add_flops("gemm", 1000);
  counters.add_flops("gemm", 500);
  counters.add_bytes("gemm", 64);
  counters.add_calls("gemm");
  counters.add_calls("gemm");
  counters.add_calls("attention", 3);
  {
    const auto snap = counters.snapshot();
    const auto gemm = snap.find("gemm");
    nexus_test::expect(gemm != snap.end() && gemm->second.flops == 1500, "flop toplami");
    nexus_test::expect(gemm != snap.end() && gemm->second.bytes == 64, "bayt toplami");
    nexus_test::expect(gemm != snap.end() && gemm->second.calls == 2, "cagri sayisi");
    nexus_test::expect(counters.total_flops() == 1500, "genel flop toplami");
    nexus_test::expect(counters.total_bytes() == 64, "genel bayt toplami");
  }
  counters.reset();
  nexus_test::expect(counters.total_flops() == 0 && counters.snapshot().empty(), "sayac sifirlama");

  nexus_test::section("kapsamli zamanlayici");
  auto& profiler = nexus::prof::profiler();
  profiler.reset();
  profiler.set_enabled(false);
  {
    nexus::prof::ScopedTimer off("faz0.disabled");
    busy_wait(std::chrono::microseconds(50));
  }
  nexus_test::expect(profiler.intervals().empty(), "kapali profiler kayit yazmaz");

  profiler.set_enabled(true);
  {
    nexus::prof::ScopedTimer outer("faz0.outer");
    {
      nexus::prof::ScopedTimer inner("faz0.inner");
      busy_wait(std::chrono::microseconds(300));
    }
    busy_wait(std::chrono::microseconds(100));
    nexus_test::expect(outer.elapsed_ns() > 0, "elapsed_ns canli okunur");
  }
  {
    const auto totals = profiler.totals();
    const auto outer = totals.find("faz0.outer");
    const auto inner = totals.find("faz0.inner");
    nexus_test::expect(outer != totals.end() && outer->second.calls == 1, "dis aralik kaydi");
    nexus_test::expect(inner != totals.end() && inner->second.calls == 1, "ic aralik kaydi");
    nexus_test::expect(outer != totals.end() && inner != totals.end() &&
                           outer->second.total_ns >= inner->second.total_ns,
                       "ice ice toplamlar: dis >= ic");
    const auto intervals = profiler.intervals();
    nexus_test::expect(intervals.size() == 2, "iki aralik kaydedildi");
    nexus_test::expect(!intervals.front().name.empty() && !intervals.front().category.empty(), "kayit alanlari dolu");
  }
  profiler.set_enabled(false);
  {
    nexus::prof::ScopedTimer after("faz0.after");
    busy_wait(std::chrono::microseconds(50));
  }
  {
    // totals() kopya dondurur: find ve end ayni kopya uzerinde olmali.
    const auto after_totals = profiler.totals();
    nexus_test::expect(after_totals.find("faz0.after") == after_totals.end(), "kapatilinca yeni kayit yok");
  }

  nexus_test::section("chrome izi");
  {
    const std::string json = nexus::prof::chrome_trace_json();
    nexus_test::expect(json.find("\"traceEvents\"") != std::string::npos, "iz JSON traceEvents alani");
    nexus_test::expect(json.find("faz0.outer") != std::string::npos, "iz aralik adlarini icerir");
    nexus_test::expect(json.find("\"ph\":\"X\"") != std::string::npos, "iz complete olay bicimi");
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "nexus_faz0_trace.json";
    nexus_test::expect(nexus::prof::dump_chrome_trace(path.string()), "iz dosyaya yazildi");
    std::ifstream back(path, std::ios::binary);
    const std::string content((std::istreambuf_iterator<char>(back)), std::istreambuf_iterator<char>());
    nexus_test::expect(!content.empty() && content.find("traceEvents") != std::string::npos, "iz dosyasi okunur");
    nexus_test::expect(!nexus::prof::dump_chrome_trace(std::filesystem::temp_directory_path().string()),
                       "gecersiz yolda dokum false");
  }

  nexus_test::section("kapi ve defter tutarliligi");
  {
    ledger.reset_all();
    const std::uint64_t before = nexus::live_allocation_count();
    const std::uint64_t total_before = nexus::allocation_count();
    std::vector<void*> blocks;
    {
      AllocationLabelScope scope(MemoryLabel::KvCache);
      for (int i = 0; i < 8; ++i) {
        blocks.push_back(nexus::allocate_aligned(128));
      }
    }
    for (int i = 0; i < 4; ++i) {
      nexus::release_owned(blocks[static_cast<std::size_t>(i)]);
    }
    const auto snap = ledger.snapshot();
    nexus_test::expect(snap.live_blocks == 4 && snap.current_bytes == 512, "yarisi serbest: 4 blok 512 bayt");
    nexus_test::expect(nexus::live_allocation_count() == before + 4, "canli kapi sayaci defterle uyumlu");
    nexus_test::expect(nexus::allocation_count() == total_before + 8, "toplam kapi sayaci monoton (8 ayirma)");
    for (int i = 4; i < 8; ++i) {
      nexus::release_owned(blocks[static_cast<std::size_t>(i)]);
    }
    nexus_test::expect(ledger.snapshot().live_blocks == 0, "hepsi serbest: blok yok");
    nexus_test::expect(nexus::live_allocation_count() == before, "canli kapi sayaci basa dondu");
  }

  nexus_test::section("nvtx");
  // Derleme zamaninda gercek durum; bayrak kapaliyken cagri noktalari silinir,
  // uydurma isaret uretilmez. Test yalnizca durumu raporlar.
  nexus_test::info("nvtx_derlenmis", nexus::prof::nvtx_active() ? 1 : 0);

  return nexus_test::finish("faz0-profil-altyapisi");
}
