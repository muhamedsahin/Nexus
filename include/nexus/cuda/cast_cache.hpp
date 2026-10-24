#pragma once

/**
 * @file cast_cache.hpp
 * @brief Autocast agirlik onbellegi — FAZ 4.1.
 *
 * Ayni FP32 agirlik ayni adimda birden fazla kez dar tipe cevrilmez: anahtar (kaynak
 * isaretcisi, hedef tip, eleman sayisi, surum). Surum degistiginde (optimizer agirligi
 * guncelledi) kayit yeniden cevrilir. `clear()` adim sonunda cagrilir ve tamponlari
 * onbellekli ayiriciya geri verir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"
#include "nexus/cuda/caching_allocator.hpp"

#include <cstdint>
#include <map>
#include <tuple>

namespace nexus {
namespace cuda {

class CastCache {
 public:
  /// FP32 kaynagin `type` kopyasi (aygit). Ayni (src, type, n, stream, version) icin bir kez.
  [[nodiscard]] const void* get(const float* src, std::size_t n, DType type, std::uint64_t version,
                                const Stream& stream);
  void clear();

  [[nodiscard]] std::uint64_t hits() const noexcept { return hits_; }
  [[nodiscard]] std::uint64_t casts() const noexcept { return casts_; }
  [[nodiscard]] std::size_t entries() const noexcept { return entries_.size(); }

 private:
  struct Entry {
    DeviceBuffer buffer;
    std::uint64_t version = 0;
  };
  std::map<std::tuple<const float*, int, std::size_t, cudaStream_t>, Entry> entries_;
  std::uint64_t hits_ = 0;
  std::uint64_t casts_ = 0;
};

}  // namespace cuda
}  // namespace nexus
