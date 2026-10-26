#pragma once

/**
 * @file codegen.hpp
 * @brief FAZ 13.4: fuzyon kumeleri icin kod uretimi — CPU (C++ + AVX2/SVML, cl.exe ile calisma aninda DLL) ve
 *        CUDA kaynagi (NVRTC derlemesi nexus_cuda'da: nexus/cuda/nvrtc_jit.hpp). Disk onbellegi: kaynak
 *        hash'i -> .dll / .cubin; ikinci calistirmada derleyici cagrisi yok.
 *
 * CPU cekirdek imzasi (extern "C"):
 *   void nxk_<hash>(const float* const* in, float* const* out, long long rows, long long cols)
 * Satir girdileri (BiasAdd b [C]) b + c ile, digerleri x + r * cols + c ile okunur; ic dongu 8'li AVX2
 * (SVML _mm256_exp_ps / _mm256_erf_ps / _mm256_tanh_ps / _mm256_log_ps), kuyruk skaler (ayni formul).
 * Derleyici: NEXUS_VCVARS ortam degiskeni ya da vswhere ile bulunan vcvars64.bat + cl /O2 /arch:AVX2 /LD.
 * Derleme basarisizsa acik hata (yorumlayici yedegi yok).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/ir.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nexus {
namespace ir {

using CpuKernelFn = void (*)(const float* const* in, float* const* out, long long rows, long long cols);

/// Bir kume icin cekirdek adi ("nxk_" + 16 hex).
[[nodiscard]] std::string kernel_name(const FusedOp& f);
/// Kumelerin CPU C++ kaynagi (tek derleme birimi).
[[nodiscard]] std::string cpu_source(const std::vector<const FusedOp*>& kernels);
/// CUDA C++ kaynagi (NVRTC): __global__ void nxk_<hash>(const float* const* in, float* const* out,
/// long long rows, long long cols) — izgara adimli dongu, cols % 4 == 0 ise float4.
[[nodiscard]] std::string cuda_source(const std::vector<const FusedOp*>& kernels);

struct JitStats {
  int compiler_invocations = 0;
  int cache_hits = 0;      // disk ya da bellek onbellegi
  double compile_ms = 0.0;  // derleyici suresi (soguk)
};

/// Kaynak -> yuklu DLL; ayni hash bellekte ve diskte onbellekli.
class CpuJit {
 public:
  explicit CpuJit(std::string cache_dir = "");
  ~CpuJit();
  CpuJit(const CpuJit&) = delete;
  CpuJit& operator=(const CpuJit&) = delete;
  /// Kaynagi derler (ya da onbellekten yukler) ve adlari cozer.
  std::map<std::string, CpuKernelFn> load(const std::string& source, const std::vector<std::string>& names);
  [[nodiscard]] const JitStats& stats() const noexcept { return stats_; }
  [[nodiscard]] const std::string& cache_dir() const noexcept { return dir_; }
  /// Kaynak dokumu (hata ayiklama, 13.8).
  void set_dump_dir(std::string dir) { dump_ = std::move(dir); }

 private:
  std::string dir_;
  std::string dump_;
  JitStats stats_;
  std::map<std::string, void*> modules_;  // hash -> HMODULE
};

[[nodiscard]] std::string source_hash(const std::string& text);
[[nodiscard]] std::string default_cache_dir();

}  // namespace ir
}  // namespace nexus
