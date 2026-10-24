#pragma once

/**
 * @file nvtx.hpp
 * @brief NVTX araliklari. Yalnizca NEXUS_ENABLE_NVTX tanimli ve NVTX basligi
 *        bulunabilirken gercek cagri uretir; aksi halde derleyicinin yok ettigi
 *        bos govdedir (bayrak kapaliyken uydurma isaret uretmez, hic uretmez).
 *
 * Kural: NVTX bayragi kapaliyken cagri noktalari silinir; bayrak acikken
 * Nsight Systems zaman cizelgesinde Nexus araliklari gorunur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <string>

#if defined(NEXUS_ENABLE_NVTX)
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NEXUS_NVTX_ACTIVE 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define NEXUS_NVTX_ACTIVE 1
#endif
#endif
#endif

namespace nexus {
namespace prof {

/// NVTX araligi: kurulus push, yikilis pop. Bayrak kapaliysa tamamen bos.
class ScopedNvtx {
 public:
#if defined(NEXUS_NVTX_ACTIVE)
  explicit ScopedNvtx(const char* name) noexcept { nvtxRangePushA(name != nullptr ? name : ""); }
  ~ScopedNvtx() { nvtxRangePop(); }
#else
  explicit ScopedNvtx(const char* name) noexcept { (void)name; }
  ~ScopedNvtx() = default;
#endif
  ScopedNvtx(const ScopedNvtx&) = delete;
  ScopedNvtx& operator=(const ScopedNvtx&) = delete;
};

/// Anlik NVTX isareti (instant marker). Bayrak kapaliysa bos.
inline void nvtx_mark(const char* message) noexcept {
#if defined(NEXUS_NVTX_ACTIVE)
  nvtxMarkA(message != nullptr ? message : "");
#else
  (void)message;
#endif
}

/// NVTX derlenmis mi? Testler bunu raporlamak icin kullanir.
[[nodiscard]] constexpr bool nvtx_active() noexcept {
#if defined(NEXUS_NVTX_ACTIVE)
  return true;
#else
  return false;
#endif
}

}  // namespace prof
}  // namespace nexus
