/**
 * @file test_phase1_allocator.cpp
 * @brief FAZ 1: k64 onbellekli ayirici (CachingAllocator) sozlesmeleri.
 *
 * Dogrulananlar:
 *  - round_size 64 bayt tabanli, iki kati siniflar uretir; her sinif 64'e tam bolunur.
 *  - Ayirma 64 bayt hizalidir; serbest birakilan blok OS'a donmez, ayni boyut
 *    sinifinda onbelleklenir ve bir sonraki istekte yeniden kullanilir (cache_hits).
 *  - reserved (OS'tan tutulan) >= current (canli); parcalanma = reserved/current.
 *  - empty_cache reserved'i current'a ceker; current==0 ve reserved==0 ise oran 1.0,
 *    current==0 ve reserved>0 ise -1.0 (tanimsiz/tek basina onbellek).
 *  - Cok is parcacikli ayirma/serbest birakma havuzu bozmaz.
 *  - nexus bellek kapisi (allocate_aligned/release_owned) havuzdan beslenir;
 *    Tensor da bu yoldan gecer. allocation_count mantiksal ayirmayi sayar.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/allocator.hpp"
#include "nexus/core/memory.hpp"
#include "nexus/core/tensor.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace {
using nexus::CachingAllocator;

[[nodiscard]] std::uint64_t address(void* pointer) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer);
}

}  // namespace

int main() {
  auto& pool = nexus::host_allocator();

  nexus_test::section("k64 yuvarlama");
  nexus_test::expect(CachingAllocator::round_size(0) == 64, "0 -> 64");
  nexus_test::expect(CachingAllocator::round_size(1) == 64, "1 -> 64");
  nexus_test::expect(CachingAllocator::round_size(64) == 64, "64 -> 64");
  nexus_test::expect(CachingAllocator::round_size(65) == 128, "65 -> 128");
  nexus_test::expect(CachingAllocator::round_size(127) == 128, "127 -> 128");
  nexus_test::expect(CachingAllocator::round_size(128) == 128, "128 -> 128");
  nexus_test::expect(CachingAllocator::round_size(129) == 256, "129 -> 256");
  nexus_test::expect(CachingAllocator::round_size(4096) == 4096, "4096 -> 4096");
  nexus_test::expect(CachingAllocator::round_size(4097) == 8192, "4097 -> 8192");
  {
    bool aligned = true;
    for (std::size_t size : {1u, 64u, 65u, 100u, 4096u, 4097u, 1000000u}) {
      aligned = aligned && (CachingAllocator::round_size(size) % CachingAllocator::alignment() == 0);
    }
    nexus_test::expect(aligned, "siniflar 64'e tam bolunur");
  }

  nexus_test::section("ayirma, hizalama, yeniden kullanim");
  pool.reset();
  {
    auto s = pool.stats();
    nexus_test::expect(s.reserved_bytes == 0 && s.current_bytes == 0 && s.live_blocks == 0, "reset temiz baslar");
  }
  void* p = pool.allocate(100);
  nexus_test::expect(p != nullptr, "ayirma basarili");
  nexus_test::expect(address(p) % 64 == 0, "isaretci 64 bayt hizali");
  {
    auto s = pool.stats();
    nexus_test::expect(s.current_bytes == 128 && s.reserved_bytes == 128, "100 bayt -> 128 sinifi");
    nexus_test::expect(s.live_blocks == 1 && s.os_allocations == 1, "tek canli blok, tek OS blogu");
    nexus_test::expect(s.cache_misses == 1 && s.cache_hits == 0, "ilk ayirma OS'a gider");
  }
  pool.deallocate(p, 100);
  {
    auto s = pool.stats();
    nexus_test::expect(s.current_bytes == 0 && s.live_blocks == 0, "serbest: canli kalmadi");
    nexus_test::expect(s.reserved_bytes == 128, "blok OS'a donmedi (onbellek)");
    nexus_test::expect(s.total_allocs == 1 && s.total_frees == 1, "sayaçlar");
  }
  void* q = pool.allocate(100);
  {
    auto s = pool.stats();
    nexus_test::expect(q == p, "aynı adres yeniden kullanildi");
    nexus_test::expect(s.cache_hits == 1 && s.cache_misses == 1, "ikinci ayirma onbellekten");
    nexus_test::expect(s.reserved_bytes == 128 && s.os_allocations == 1, "yeni OS blogu acilmadi");
  }
  pool.deallocate(q, 100);

  nexus_test::section("parcalanma orani");
  pool.reset();
  void* a = pool.allocate(64);
  void* b = pool.allocate(64);
  void* c = pool.allocate(64);
  pool.deallocate(a, 64);  // onbelleklenir
  {
    auto s = pool.stats();
    nexus_test::expect(s.current_bytes == 128 && s.reserved_bytes == 192, "2 canli + 1 onbellekli");
    const double ratio = pool.fragmentation_ratio();
    nexus_test::expect(std::fabs(ratio - 1.5) < 1.0e-9, "oran = 192/128 = 1.5");
    nexus_test::expect(nexus::memory_fragmentation_ratio() == ratio, "kapi erisimi ayni degeri dondurur");
  }
  pool.empty_cache();
  {
    auto s = pool.stats();
    nexus_test::expect(s.reserved_bytes == 128 && s.current_bytes == 128, "empty_cache reserved'i current'a ceker");
    nexus_test::expect(std::fabs(pool.fragmentation_ratio() - 1.0) < 1.0e-9, "parcalanma 1.0");
  }
  pool.deallocate(b, 64);
  pool.deallocate(c, 64);
  {
    auto s = pool.stats();
    nexus_test::expect(s.current_bytes == 0 && s.reserved_bytes == 128, "yalnizca onbellek kaldi");
    nexus_test::expect(pool.fragmentation_ratio() == -1.0, "canli yok + onbellek var -> -1.0");
  }
  pool.empty_cache();
  nexus_test::expect(pool.stats().reserved_bytes == 0, "hepsi OS'a dondu");
  nexus_test::expect(pool.fragmentation_ratio() == 1.0, "bos havuz -> 1.0");

  nexus_test::section("buyuk blok");
  pool.reset();
  const std::size_t big = 3u * 1024u * 1024u;  // 3145728 bayt
  void* bigp = pool.allocate(big);
  nexus_test::expect(pool.stats().reserved_bytes == 4194304, "3 MiB -> 2^22 sinifi");
  pool.deallocate(bigp, big);
  pool.empty_cache();
  nexus_test::expect(pool.stats().reserved_bytes == 0, "buyuk blok serbest birakildi");

  nexus_test::section("cok is parcacigi");
  pool.reset();
  {
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
      workers.emplace_back([&pool]() {
        for (int i = 0; i < 500; ++i) {
          void* block = pool.allocate(256);
          pool.deallocate(block, 256);
        }
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
    auto s = pool.stats();
    nexus_test::expect(s.current_bytes == 0 && s.live_blocks == 0, "cok is parcacigi: canli kalmadi");
    nexus_test::expect(s.total_allocs == 2000 && s.total_frees == 2000, "2000 ayirma + 2000 serbest");
    nexus_test::expect(s.cache_hits > 0, "onbellek yeniden kullanildi");
    nexus_test::expect(s.reserved_bytes > 0, "onbellekli bloklar tutuldu");
  }
  pool.reset();

  nexus_test::section("kapi entegrasyonu");
  pool.reset();
  {
    const std::uint64_t before = nexus::allocation_count();
    const std::uint64_t live_before = nexus::live_allocation_count();
    void* k = nexus::allocate_aligned(100);
    nexus_test::expect(nexus::allocation_count() == before + 1, "kapi sayaci artti");
    auto s = pool.stats();
    nexus_test::expect(s.current_bytes == 128, "kapi ayirmasi havuzdan");
    nexus::release_owned(k);
    nexus_test::expect(nexus::live_allocation_count() == live_before, "canli kapi sayaci dondu");
    nexus_test::expect(nexus::allocation_count() == before + 1, "toplam sayac monoton kalir");
    auto s2 = pool.stats();
    nexus_test::expect(s2.current_bytes == 0 && s2.reserved_bytes == 128, "kapi serbesti havuzda");
  }
  pool.reset();
  {
    // Tensor kapidan beslenir; 40 float = 160 bayt -> 256 sinifi.
    nexus::Tensor tensor = nexus::Tensor::zeros({40});
    auto s = pool.stats();
    nexus_test::expect(s.current_bytes == 256 && s.live_blocks == 1, "Tensor havuzdan ayirdi");
  }
  {
    auto s = pool.stats();
    nexus_test::expect(s.current_bytes == 0 && s.reserved_bytes == 256, "Tensor yikildi, blok onbellekte");
  }
  pool.reset();

  return nexus_test::finish("faz1-k64-ayirici");
}
