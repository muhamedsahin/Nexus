#pragma once

/**
 * @file hip_compat.hpp
 * @brief FAZ 19.B: ROCm / HIP uyumluluk basligi, hipify sembol eslemesi, wavefront=32/64
 *        ve 64 KB LDS sinirli FlashAttention karo secimi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace nexus::backend::hip {

/// AMD CDNA (MI100/MI200/MI300) wavefront=64, RDNA (RX 6000/7000/8000, Radeon 680M/780M) wave32 veya wave64.
inline constexpr std::uint32_t kCdnaWavefrontSize = 64;
inline constexpr std::uint32_t kRdnaWave32Size = 32;
inline constexpr std::size_t kRocmLdsLimitBytes = 64u * 1024u;  // 64 KB LDS / CU siniri

struct HipSymbolMap {
  std::string_view cuda_symbol;
  std::string_view hip_symbol;
  std::string_view category;  // "runtime" | "blaslt" | "comm"
};

/// CUDA -> HIP (hipify) dogrudan esleme tablosu (FAZ 19.B).
[[nodiscard]] const std::vector<HipSymbolMap>& symbol_table();

/// Verilen CUDA kaynak metnini HIP/ROCm kaynak metnine donusturur ve sabit 32 warp varsayimini denetler.
[[nodiscard]] std::string hipify_source(std::string_view cuda_source);

struct FlashTileConfig {
  int br = 64;
  int bc = 32;
  int head_dim = 64;
  int warps = 4;
  int wavefront_size = 64;
  std::size_t lds_bytes = 0;
  bool fits_lds = false;
};

/// ROCm 64 KB LDS sinirina uygun FlashAttention karo boyu secimi.
[[nodiscard]] FlashTileConfig select_rocm_flash_tiles(int head_dim, int wavefront_size = 64,
                                                      std::size_t lds_limit = kRocmLdsLimitBytes) noexcept;

struct KernelAuditReport {
  int files_scanned = 0;
  int wave_adaptive_reductions = 0;
  int hardcoded_warp_violations = 0;
  bool all_clean = false;
  std::vector<std::string> notes;
};

/// Monorepodaki CUDA/HIP cekirdeklerinde sabit 32 yerine kWarp / warpSize kullanimini denetler.
[[nodiscard]] KernelAuditReport audit_cuda_kernels_for_hip(const std::string& repo_root);

/// gomulu HIP cekirdek kaynak metni (hiprtc ile derlenmeye hazir).
[[nodiscard]] std::string_view embedded_hip_kernels_source() noexcept;

}  // namespace nexus::backend::hip
