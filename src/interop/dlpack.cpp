/**
 * @file dlpack.cpp
 * @brief FAZ 17.5: DLPack sifir kopya alisverisi (bkz. nexus/interop/dlpack.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/dlpack.hpp"

#include "nexus/interop/json.hpp"

#include <vector>

namespace nexus {
namespace interop {

Tensor from_dlpack(DLManagedTensor* m) {
  if (m == nullptr) throw InteropError("DLPack: bos yonetilen tensor");
  const DLTensor& t = m->dl_tensor;
  // tuketilen tensoru hata yolunda da birak (sahiplik bize gecti)
  std::shared_ptr<void> keep(m, [](void* p) {
    auto* mt = static_cast<DLManagedTensor*>(p);
    if (mt->deleter) mt->deleter(mt);
  });
  if (t.dtype.code != 2 || t.dtype.bits != 32 || t.dtype.lanes != 1) {
    throw InteropError("DLPack: yalniz float32 (kod " + std::to_string(t.dtype.code) + ", " + std::to_string(t.dtype.bits) +
                       " bit gelmis); donusturme yapilmaz");
  }
  if (t.ndim < 0 || t.ndim > 8) throw InteropError("DLPack: ndim 0..8");
  std::vector<std::size_t> dims(static_cast<std::size_t>(t.ndim));
  for (int i = 0; i < t.ndim; ++i) {
    if (t.shape[i] < 0) throw InteropError("DLPack: negatif boyut");
    dims[static_cast<std::size_t>(i)] = static_cast<std::size_t>(t.shape[i]);
  }
  if (t.strides != nullptr) {  // C-sirali bitisik olmali (boyutu 1 olan eksenlerin adimi onemsiz)
    std::int64_t expect = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
      if (t.shape[i] != 1 && t.strides[i] != expect)
        throw InteropError("DLPack: bitisik olmayan adimlar (eksen " + std::to_string(i) + "); once .contiguous() cagirin");
      expect *= t.shape[i];
    }
  }
  MemorySpace space = MemorySpace::Host;
  DeviceRef dev = DeviceRef::cpu();
  if (t.device.device_type == kDLCUDA) {
    space = MemorySpace::Device;
    dev = DeviceRef::cuda(t.device.device_id);
  } else if (t.device.device_type == kDLCUDAHost) {
    space = MemorySpace::PinnedHost;
  } else if (t.device.device_type != kDLCPU) {
    throw InteropError("DLPack: desteklenmeyen cihaz tipi " + std::to_string(t.device.device_type) + " (CPU, CUDA, CUDA host)");
  }
  float* data = reinterpret_cast<float*>(static_cast<char*>(t.data) + t.byte_offset);
  return Tensor::adopt_shared(data, Shape(dims), space, dev, std::move(keep));
}

namespace {
struct Ctx {
  std::shared_ptr<Tensor> tensor;
  std::vector<std::int64_t> shape;
};
}  // namespace

DLManagedTensor* to_dlpack(std::shared_ptr<Tensor> tensor) {
  if (!tensor) throw InteropError("DLPack: bos tensor");
  auto* ctx = new Ctx{std::move(tensor), {}};
  const Tensor& t = *ctx->tensor;
  for (std::uint8_t i = 0; i < t.shape().rank(); ++i) ctx->shape.push_back(static_cast<std::int64_t>(t.shape()[i]));
  auto* m = new DLManagedTensor{};
  m->dl_tensor.data = const_cast<float*>(t.data());
  if (t.space() == MemorySpace::Device) {
    m->dl_tensor.device = DLDevice{kDLCUDA, t.device().index < 0 ? 0 : t.device().index};
  } else if (t.space() == MemorySpace::PinnedHost) {
    m->dl_tensor.device = DLDevice{kDLCUDAHost, 0};
  } else {
    m->dl_tensor.device = DLDevice{kDLCPU, 0};
  }
  m->dl_tensor.ndim = static_cast<std::int32_t>(ctx->shape.size());
  m->dl_tensor.dtype = DLDataType{2, 32, 1};
  m->dl_tensor.shape = ctx->shape.data();
  m->dl_tensor.strides = nullptr;  // bitisik
  m->dl_tensor.byte_offset = 0;
  m->manager_ctx = ctx;
  m->deleter = [](DLManagedTensor* self) {
    delete static_cast<Ctx*>(self->manager_ctx);
    delete self;
  };
  return m;
}

}  // namespace interop
}  // namespace nexus
