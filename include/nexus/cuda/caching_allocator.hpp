#pragma once

/**
 * @file caching_allocator.hpp
 * @brief Akis farkindalikli, onbellekli CUDA aygit ayiricisi + pinned host havuzu — FAZ 1.4.
 *
 * PyTorch CUDACachingAllocator tasarimi referans alinmistir:
 *  - Kucuk (<= 1 MiB) ve buyuk havuz; istekler 512 bayta yuvarlanir.
 *  - Kucuk istekler 2 MiB, 1-10 MiB arasi istekler 20 MiB segmentten; daha buyukleri
 *    2 MiB'a yuvarlanmis kendi segmentinden gelir. Bloklar bolunur (split) ve serbest
 *    birakilinca ayni segmentteki serbest komsularla birlestirilir (merge).
 *  - Serbest liste akis basinadir: bir akista serbest kalan blok baska akisa verilmez.
 *  - `record_stream(ptr, s)`: tampon baska akista da kullanildiysa serbest birakma
 *    o akislara olay kaydeder ve olaylar tamamlanana kadar ertelenir (erken yeniden
 *    kullanim yarisi olmaz).
 *  - OOM: once onbellek bosaltilir ve yeniden denenir (`num_alloc_retries`); yine
 *    olmazsa etiket basina bayt, en buyuk serbest blok ve parcalanma iceren ayrintili
 *    `TensorError`.
 *  - Istatistik: allocated / reserved / active / inactive_split, tepe degerler,
 *    cache_hits, num_alloc_retries, num_ooms, segment sayisi.
 *
 * CUDA VMM "genisletilebilir segment" (cuMemCreate/cuMemMap) bu surumde YOKTUR;
 * yalniz klasik cudaMalloc segment havuzu vardir (rapor bunu acik is olarak yazar).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nexus {
namespace cuda {

struct DeviceAllocatorStats {
  std::uint64_t allocated_bytes = 0;       // kullanicida canli (yuvarlanmis)
  std::uint64_t peak_allocated_bytes = 0;
  std::uint64_t requested_bytes = 0;       // kullanicinin istedigi (yuvarlanmamis)
  std::uint64_t reserved_bytes = 0;        // cudaMalloc ile tutulan segmentler
  std::uint64_t peak_reserved_bytes = 0;
  std::uint64_t active_bytes = 0;          // canli + erteleme bekleyen
  std::uint64_t inactive_split_bytes = 0;  // bolunmus segmentlerdeki serbest bayt
  std::uint64_t num_allocs = 0;
  std::uint64_t num_frees = 0;
  std::uint64_t cache_hits = 0;            // cudaMalloc cagirmadan karsilanan
  std::uint64_t num_cuda_mallocs = 0;
  std::uint64_t num_alloc_retries = 0;
  std::uint64_t num_ooms = 0;
  std::uint64_t num_segments = 0;
  std::uint64_t largest_free_block = 0;
};

/// Degistirilebilir aygit ayirici arayuzu (test icin sayacli ayirici, ileride
/// cudaMallocAsync/RMM arka ucu).
class DeviceAllocator {
 public:
  virtual ~DeviceAllocator() = default;
  [[nodiscard]] virtual void* allocate(std::size_t bytes, const Stream& stream) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual void record_stream(void* ptr, const Stream& stream) = 0;
  virtual void empty_cache() = 0;
  [[nodiscard]] virtual DeviceAllocatorStats stats(int device = -1) const = 0;
  virtual void reset_peak_stats() = 0;
};

class CachingDeviceAllocator final : public DeviceAllocator {
 public:
  CachingDeviceAllocator();
  ~CachingDeviceAllocator() override;
  CachingDeviceAllocator(const CachingDeviceAllocator&) = delete;
  CachingDeviceAllocator& operator=(const CachingDeviceAllocator&) = delete;

  [[nodiscard]] void* allocate(std::size_t bytes, const Stream& stream) override;
  void deallocate(void* ptr) override;
  void record_stream(void* ptr, const Stream& stream) override;
  /// Erteleme olaylarini isler; tamamen serbest segmentleri cudaFree ile birakir.
  void empty_cache() override;
  [[nodiscard]] DeviceAllocatorStats stats(int device = -1) const override;
  void reset_peak_stats() override;
  /// Bu ayiricidan alinmis isaretcinin istenen boyutu (yoksa 0).
  [[nodiscard]] std::size_t allocation_size(const void* ptr) const;
  /// Test/hata ayiklama: ayrintili bellek ozeti metni.
  [[nodiscard]] std::string summary(int device = -1) const;

  static constexpr std::size_t kMinBlock = 512;
  static constexpr std::size_t kSmallSize = 1u << 20;      // 1 MiB
  static constexpr std::size_t kSmallBuffer = 2u << 20;    // 2 MiB
  static constexpr std::size_t kLargeBuffer = 20u << 20;   // 20 MiB
  static constexpr std::size_t kMinLargeAlloc = 10u << 20; // 10 MiB
  static constexpr std::size_t kRoundLarge = 2u << 20;     // 2 MiB
  [[nodiscard]] static std::size_t round_size(std::size_t bytes) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Surec genelinde aygit ayiricisi (varsayilan: CachingDeviceAllocator).
[[nodiscard]] DeviceAllocator& device_allocator();
/// Ayiriciyi degistirir; nullptr varsayilana doner. Ayirici, kendisinden alinan
/// tum bloklar serbest kalana kadar yasamalidir.
void set_device_allocator(DeviceAllocator* allocator) noexcept;
/// Varsayilan onbellekli ayiricinin kendisi (istatistik ve test icin).
[[nodiscard]] CachingDeviceAllocator& caching_device_allocator();

// ---- Pinned host havuzu ---------------------------------------------------------
struct HostAllocatorStats {
  std::uint64_t allocated_bytes = 0;
  std::uint64_t reserved_bytes = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t num_host_allocs = 0;
  std::uint64_t deferred_frees = 0;
};

/// cudaHostAlloc havuzu: iki kati boyut siniflari; `record_stream` ile isaretli blok
/// o akistaki asenkron kopya bitene kadar yeniden verilmez.
class CachingHostAllocator {
 public:
  CachingHostAllocator();
  ~CachingHostAllocator();
  CachingHostAllocator(const CachingHostAllocator&) = delete;
  CachingHostAllocator& operator=(const CachingHostAllocator&) = delete;

  [[nodiscard]] void* allocate(std::size_t bytes);
  void deallocate(void* ptr);
  void record_stream(void* ptr, const Stream& stream);
  void empty_cache();
  [[nodiscard]] HostAllocatorStats stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] CachingHostAllocator& pinned_allocator();

/// RAII aygit tamponu (gecerli akista ayrilir, yikimda ayiriciya doner).
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes, const Stream& stream = current_stream());
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* data() const noexcept { return ptr_; }
  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(ptr_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  void reset();

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
  std::uint8_t label_ = 4;  // prof::MemoryLabel; ayirma anindaki etiket serbestte de kullanilir
};

}  // namespace cuda
}  // namespace nexus
