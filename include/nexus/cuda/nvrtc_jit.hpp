#pragma once

/**
 * @file nvrtc_jit.hpp
 * @brief FAZ 13.4: CUDA kod uretimi icin NVRTC JIT — ir::cuda_source kaynagi -> cubin (aygitin sm_XY'si) ->
 *        cuModuleLoadData; disk onbellegi (kaynak + mimari hash -> .cubin): ikinci yuklemede NVRTC cagrisi yok.
 *        Triton bagimliligi yok. Derleme hatasi NVRTC gunluguyle acik hata.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/codegen.hpp"

#include <cuda_runtime_api.h>

#include <map>
#include <string>
#include <vector>

namespace nexus {
namespace cuda {

class NvrtcJit {
 public:
  explicit NvrtcJit(std::string cache_dir = "");
  ~NvrtcJit();
  NvrtcJit(const NvrtcJit&) = delete;
  NvrtcJit& operator=(const NvrtcJit&) = delete;

  /// Kaynagi derler/yukler, adlari cozer (onceki yuklemeler korunur).
  void load(const std::string& source, const std::vector<std::string>& names);
  /// Aygit isaretci dizileriyle (d_in / d_out aygitta) baslatma; izgara adimli, blok 256.
  void launch(const std::string& name, const float* const* d_in, float* const* d_out, long long rows, long long cols,
              cudaStream_t stream);
  /// Test yardimcisi: host girdileri -> aygit -> cekirdek -> host ciktilari (senkron).
  void run(const std::string& name, const std::vector<std::vector<float>>& in, std::vector<std::vector<float>>& out,
           long long rows, long long cols);
  [[nodiscard]] const ir::JitStats& stats() const noexcept { return stats_; }

 private:
  std::string dir_;
  ir::JitStats stats_;
  std::vector<void*> modules_;            // CUmodule
  std::map<std::string, void*> funcs_;    // CUfunction
};

}  // namespace cuda
}  // namespace nexus
