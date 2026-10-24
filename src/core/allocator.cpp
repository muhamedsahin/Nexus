/**
 * @file allocator.cpp
 * @brief k64 CachingAllocator gerceklemesi (ayrintilar allocator.hpp basliginda).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/allocator.hpp"

#include "nexus/core/error.hpp"

#include <cstdlib>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace nexus {
namespace {

[[nodiscard]] std::size_t align_up(std::size_t bytes, std::size_t alignment) noexcept {
  return (bytes + alignment - 1u) & ~(alignment - 1u);
}

}  // namespace

CachingAllocator::~CachingAllocator() {
  // Dogru kullanimda live_blocks == 0 olur; kalan canli bloklar cagiranindir.
  // Yalnizca onbellekli serbest bloklar OS'a geri verilir.
  empty_cache();
}

std::size_t CachingAllocator::round_size(std::size_t bytes) noexcept {
  constexpr std::size_t kMin = alignment();
  if (bytes <= kMin) {
    return kMin;
  }
  std::size_t size = kMin;
  while (size < bytes) {
    const std::size_t next = size << 1u;
    if (next <= size) {  // taskma: cok buyuk istek; oldugu gibi birak.
      return bytes;
    }
    size = next;
  }
  return size;
}

void* CachingAllocator::os_alloc(std::size_t bytes) {
  const std::size_t aligned = align_up(bytes, alignment());
  void* pointer = nullptr;
#if defined(_WIN32)
  pointer = _aligned_malloc(aligned, alignment());
#else
  pointer = std::aligned_alloc(alignment(), aligned);
#endif
  if (pointer == nullptr) {
    throw TensorError("Nexus: k64 havuzu OS bellegi ayiramadi");
  }
  return pointer;
}

void CachingAllocator::os_free(void* ptr) noexcept {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

void* CachingAllocator::allocate(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  const std::size_t size = round_size(bytes);
  std::lock_guard<std::mutex> guard(mutex_);
  total_allocs_ += 1;

  auto& bin = free_[size];
  if (!bin.empty()) {
    void* pointer = bin.back();
    bin.pop_back();
    cache_hits_ += 1;
    current_bytes_ += size;
    live_blocks_ += 1;
    return pointer;
  }

  cache_misses_ += 1;
  void* pointer = os_alloc(size);
  reserved_bytes_ += size;
  os_allocations_ += 1;
  current_bytes_ += size;
  live_blocks_ += 1;
  return pointer;
}

void CachingAllocator::deallocate(void* ptr, std::size_t bytes) noexcept {
  if (ptr == nullptr) {
    return;
  }
  const std::size_t size = round_size(bytes);
  std::lock_guard<std::mutex> guard(mutex_);
  total_frees_ += 1;
  if (current_bytes_ >= size) {
    current_bytes_ -= size;
  } else {
    current_bytes_ = 0;
  }
  if (live_blocks_ > 0) {
    live_blocks_ -= 1;
  }
  free_[size].push_back(ptr);
}

CachingAllocator::Stats CachingAllocator::stats() const {
  std::lock_guard<std::mutex> guard(mutex_);
  Stats result;
  result.reserved_bytes = reserved_bytes_;
  result.current_bytes = current_bytes_;
  result.live_blocks = live_blocks_;
  result.total_allocs = total_allocs_;
  result.total_frees = total_frees_;
  result.os_allocations = os_allocations_;
  result.cache_hits = cache_hits_;
  result.cache_misses = cache_misses_;
  return result;
}

double CachingAllocator::fragmentation_ratio() const {
  std::lock_guard<std::mutex> guard(mutex_);
  if (current_bytes_ == 0) {
    return reserved_bytes_ == 0 ? 1.0 : -1.0;
  }
  return static_cast<double>(reserved_bytes_) / static_cast<double>(current_bytes_);
}

void CachingAllocator::empty_cache_locked() noexcept {
  for (auto& entry : free_) {
    const std::size_t size = entry.first;
    for (void* pointer : entry.second) {
      os_free(pointer);
      if (reserved_bytes_ >= size) {
        reserved_bytes_ -= size;
      } else {
        reserved_bytes_ = 0;
      }
      if (os_allocations_ > 0) {
        os_allocations_ -= 1;
      }
    }
    entry.second.clear();
  }
  free_.clear();
}

void CachingAllocator::empty_cache() noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  empty_cache_locked();
}

void CachingAllocator::reset() noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  empty_cache_locked();
  reserved_bytes_ = 0;
  current_bytes_ = 0;
  live_blocks_ = 0;
  total_allocs_ = 0;
  total_frees_ = 0;
  os_allocations_ = 0;
  cache_hits_ = 0;
  cache_misses_ = 0;
}

CachingAllocator& host_allocator() noexcept {
  // Yorumlayici kapanis sirasi belirsiz; havuz kasitli olarak sizdirilir.
  static auto* instance = new CachingAllocator();
  return *instance;
}

}  // namespace nexus
