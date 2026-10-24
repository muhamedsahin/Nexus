/**
 * @file cast_cache.cpp
 * @brief FAZ 4.1 autocast agirlik onbellegi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/cast_cache.hpp"

#include "nexus/core/error.hpp"
#include "nexus/cuda/kernels.hpp"

namespace nexus {
namespace cuda {

const void* CastCache::get(const float* src, std::size_t n, DType type, std::uint64_t version, const Stream& stream) {
  if (src == nullptr || n == 0) {
    throw TensorError("Nexus CastCache: bos kaynak");
  }
  if (type != DType::BF16 && type != DType::F16) {
    throw TensorError("Nexus CastCache: hedef tip BF16 veya F16 olmali");
  }
  const auto key = std::make_tuple(src, static_cast<int>(type), n, stream.handle());
  auto it = entries_.find(key);
  if (it != entries_.end() && it->second.version == version) {
    hits_ += 1;
    return it->second.buffer.data();
  }
  if (it == entries_.end()) {
    it = entries_.emplace(key, Entry{DeviceBuffer(n * size_of(type), stream), version}).first;
  }
  it->second.version = version;
  cast(it->second.buffer.data(), type, src, DType::F32, n, stream.handle());
  casts_ += 1;
  return it->second.buffer.data();
}

void CastCache::clear() { entries_.clear(); }

}  // namespace cuda
}  // namespace nexus
