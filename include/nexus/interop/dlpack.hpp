#pragma once

/**
 * @file dlpack.hpp
 * @brief FAZ 17.5: DLPack (v0.8 ABI) <-> nexus::Tensor, sifir kopya.
 *
 * from_dlpack(DLManagedTensor*): yonetilen tensoru TUKETIR; nexus::Tensor ayni isaretciyi goruntuler
 * (Tensor::adopt_shared) ve son kopyasi yok olunca ureticinin deleter'i cagrilir. Kabul edilen: float32 (kod 2,
 * 32 bit, 1 serit), bitisik (strides bos ya da C-sirali), cihaz CPU (1), CUDA (2) ya da CUDA host (3). Diger
 * dtype / adim / cihaz ACIK HATA (kopyaya dusme yok).
 * to_dlpack(shared_ptr<Tensor>): tensor yasadikca gecerli DLManagedTensor; deleter kaynagi birakir.
 * Python tarafi (`nexus.from_dlpack`, `Tensor.__dlpack__`) kapsul adlarini ("dltensor" -> "used_dltensor") yonetir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor.hpp"

#include <cstdint>
#include <memory>

extern "C" {
// dlpack.h v0.8 ile ayni yerlesim (https://github.com/dmlc/dlpack)
typedef enum { kDLCPU = 1, kDLCUDA = 2, kDLCUDAHost = 3 } DLDeviceType;
typedef struct {
  int32_t device_type;
  int32_t device_id;
} DLDevice;
typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;
typedef struct {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLTensor;
typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;
}

namespace nexus {
namespace interop {

[[nodiscard]] Tensor from_dlpack(DLManagedTensor* managed);
[[nodiscard]] DLManagedTensor* to_dlpack(std::shared_ptr<Tensor> tensor);

}  // namespace interop
}  // namespace nexus
