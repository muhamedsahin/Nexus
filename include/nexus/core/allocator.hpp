#pragma once

/**
 * @file allocator.hpp
 * @brief k64 boyut-sinifli onbellekli ayirici (CachingAllocator) — FAZ 1.
 *
 * Amac:
 *   - Kucuk tensor/aktivasyon ayirmalarinda OS cagrisi (malloc/VirtualAlloc)
 *     sayisini dusurmek: serbest birakilan bloklar yok edilmez, boyut sinifina
 *     gore havuzda tutulur ve bir sonraki uygun istekte yeniden kullanilir.
 *   - Tek ayirici uzerinden "reserved" (OS'tan tutulan toplam) ve "current"
 *     (kullanicida canli olan toplam) tek yerde sayilir; FAZ 0 defteri ise
 *     etiket bazli kirilimi korur. Parcalanma orani reserved/current'tir.
 *
 * Boyut siniflari 64 bayttan baslar ve iki katina cikar (64, 128, 256, ...).
 * `round_size(bytes)` her istegi bu sinifa yuvarlar; boylece ayirma her zaman
 * 64 bayt hizalidir. Sinif sayisi sinirlidir, serbest liste std::map ile
 * boyut anahtarina gore tutulur.
 *
 * Cok is parcacikli kullanim icin tum islemler mutex altindadir. FAZ 1
 * olceginde bu yeterlidir; FAZ 13'te is parcacigi yerel havuz + periyodik
 * birlestirme bu sinifa eklenir (ayni arayuz korunur).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace nexus {

/// k64 onbellekli ayirici. Host (AlignedHost) yolu icin surec genelinde tek ornek.
class CachingAllocator {
 public:
  struct Stats {
    std::uint64_t reserved_bytes = 0;  // OS'tan tutulan toplam (canli + onbellekli)
    std::uint64_t current_bytes = 0;   // kullanicida canli olan toplam (yuvarlanmis)
    std::uint64_t live_blocks = 0;
    std::uint64_t total_allocs = 0;
    std::uint64_t total_frees = 0;
    std::uint64_t os_allocations = 0;  // acilan OS blogu sayisi
    std::uint64_t cache_hits = 0;      // serbest listeden karsilanan
    std::uint64_t cache_misses = 0;    // yeni OS blogu gereken
  };

  CachingAllocator() = default;
  ~CachingAllocator();

  CachingAllocator(const CachingAllocator&) = delete;
  CachingAllocator& operator=(const CachingAllocator&) = delete;

  /// 64 bayt hizali, boyut-sinifli ayirma. `bytes == 0` ise nullptr.
  [[nodiscard]] void* allocate(std::size_t bytes);

  /// `allocate` ile alinani havuza geri verir (OS'a degil). `bytes`,
  /// `allocate` cagrisinda istenen gercek boyuttur (sinif bulmak icin).
  void deallocate(void* ptr, std::size_t bytes) noexcept;

  [[nodiscard]] Stats stats() const;

  /// reserved/current. Canli ayirma yoksa: reserved>0 icin -1.0, ikisi de 0
  /// icin 1.0. Orne 1.0 ise parcalanma yok, >1.0 ic parcalanma/onbellek var.
  [[nodiscard]] double fragmentation_ratio() const;

  /// Onbelleklenmis tum serbest bloklari OS'a geri verir; reserved, current'a
  /// cekilir (canli bloklar yerinde kalir).
  void empty_cache() noexcept;

  /// `empty_cache()` + butun sayaclari sifirlar. Testler icin.
  void reset() noexcept;

  /// k64 sinifina yuvarlar: >=64 ve iki kati (power of two).
  [[nodiscard]] static std::size_t round_size(std::size_t bytes) noexcept;

  [[nodiscard]] static constexpr std::size_t alignment() noexcept { return 64; }

 private:
  void empty_cache_locked() noexcept;

  [[nodiscard]] void* os_alloc(std::size_t bytes);
  void os_free(void* ptr) noexcept;

  mutable std::mutex mutex_;
  std::map<std::size_t, std::vector<void*>> free_{};
  std::uint64_t reserved_bytes_ = 0;
  std::uint64_t current_bytes_ = 0;
  std::uint64_t live_blocks_ = 0;
  std::uint64_t total_allocs_ = 0;
  std::uint64_t total_frees_ = 0;
  std::uint64_t os_allocations_ = 0;
  std::uint64_t cache_hits_ = 0;
  std::uint64_t cache_misses_ = 0;
};

/// Surec genelindeki tek host havuzu. nexus bellek kapisi ve Tensor bunu kullanir.
[[nodiscard]] CachingAllocator& host_allocator() noexcept;

}  // namespace nexus
