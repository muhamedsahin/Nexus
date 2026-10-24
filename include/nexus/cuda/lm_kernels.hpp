#pragma once

/**
 * @file lm_kernels.hpp
 * @brief FAZ 6 GPU cekirdekleri: RoPE (yerinde, ileri/transpoz) ve SwiGLU/GeGLU (ileri/geri).
 *
 * RoPE: x satirlari [rows, row_stride] (eleman), her satirda col0'dan baslayan `heads` adet
 * head_dim'lik bas. GPT/LLaMA qkv tamponunda Q ve K bitisik oldugundan (Q H*D | K Hkv*D | V)
 * tek cagri heads = H + Hkv ile ikisini birden dondurur. Konum = (satir % time) + offset.
 * cos/sin tablolari [P][rot/2] FP32 (`DeviceRopeTable`, nexus::lm::RopeCache'ten yuklenir),
 * duzen ve formul CPU `nexus::lm::rope_apply` ile ayni.
 *
 * SwiGLU: in [rows, 2F] = [a | b] -> out [rows, F] = act(a) * b; geri gecis din'i yazar.
 * Hesap FP32, depolama F32/BF16/F16.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/lm/rope.hpp"

#include <cuda_runtime.h>

namespace nexus {
namespace cuda {

/// Aygitta cos/sin [length][rot/2]; `ensure` ile tembel buyur (host RopeCache'ten).
class DeviceRopeTable {
 public:
  DeviceRopeTable(int rot_dim, float theta, lm::RopeScaling scaling, int max_position, const Stream& stream);
  void ensure(int length);
  [[nodiscard]] const float* cos() const noexcept { return cos_.as<float>(); }
  [[nodiscard]] const float* sin() const noexcept { return sin_.as<float>(); }
  [[nodiscard]] int rot_dim() const noexcept { return host_.rot_dim(); }
  [[nodiscard]] int length() const noexcept { return host_.length(); }

 private:
  lm::RopeCache host_;
  const Stream* stream_;
  DeviceBuffer cos_, sin_;
  int uploaded_ = 0;
};

struct RopeArgs {
  void* x = nullptr;
  DType dtype = DType::BF16;
  int rows = 0;
  long long row_stride = 0;  // eleman
  int col0 = 0;              // ilk basin sutunu
  int heads = 0;
  int head_dim = 0;
  int time = 1;  // konum = satir % time + offset
  int offset = 0;
  bool transpose = false;  // geri gecis
};

void rope_inplace(const RopeArgs& args, const DeviceRopeTable& table, cudaStream_t stream);

enum class GatedAct { SiLU, GeLU };

void gated_forward(void* out, const void* in, DType dtype, int rows, int f, GatedAct act, cudaStream_t stream);
/// din [rows, 2F] = (dout * b * act'(a) | dout * act(a)).
void gated_backward(void* din, const void* dout, const void* in, DType dtype, int rows, int f, GatedAct act,
                    cudaStream_t stream);

}  // namespace cuda
}  // namespace nexus
