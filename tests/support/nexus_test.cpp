/**
 * @file nexus_test.cpp
 * @brief Test denetimleri ve global operator new/delete degistirmesi.
 *
 * Bu ceviri birimi OBJECT kutuphanesi olarak her teste dogrudan baglanir.
 * Statik kutuphane kullanilmaz: baglayici degistirilmis operator new'u
 * CRT'ninkinden once gormeyebilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus_test.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace {

std::atomic<std::uint64_t> g_new_calls{0};
std::atomic<int> g_scope_depth{0};
int g_failed = 0;
int g_passed = 0;

void count_allocation() noexcept {
  if (g_scope_depth.load(std::memory_order_relaxed) > 0) {
    g_new_calls.fetch_add(1, std::memory_order_relaxed);
  }
}

void* allocate(std::size_t bytes) {
  count_allocation();
  if (void* pointer = std::malloc(bytes == 0 ? 1 : bytes)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* allocate_aligned(std::size_t bytes, std::align_val_t alignment) {
  count_allocation();
  const auto align = static_cast<std::size_t>(alignment);
#if defined(_MSC_VER)
  void* pointer = _aligned_malloc(bytes == 0 ? 1 : bytes, align);
#else
  const std::size_t rounded = ((bytes == 0 ? 1 : bytes) + align - 1) / align * align;
  void* pointer = std::aligned_alloc(align, rounded);
#endif
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void release_aligned(void* pointer) noexcept {
#if defined(_MSC_VER)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

}  // namespace

namespace nexus_test {

void expect(bool condition, std::string_view name) {
  if (condition) {
    ++g_passed;
    std::cout << "  PASS  " << name << "\n";
    return;
  }
  ++g_failed;
  std::cout << "  FAIL  " << name << "\n";
}

void info(std::string_view key, double value) { std::cout << "  info  " << key << "=" << value << "\n"; }

void section(std::string_view title) { std::cout << title << "\n"; }

int failures() noexcept { return g_failed; }

int finish(std::string_view suite) {
  std::cout << suite << ": " << g_passed << " gecti, " << g_failed << " basarisiz\n";
  return g_failed == 0 ? 0 : 1;
}

AllocationScope::AllocationScope() noexcept : begin_(g_new_calls.load(std::memory_order_relaxed)) {
  g_scope_depth.fetch_add(1, std::memory_order_relaxed);
}

AllocationScope::~AllocationScope() { g_scope_depth.fetch_sub(1, std::memory_order_relaxed); }

std::uint64_t AllocationScope::count() const noexcept {
  return g_new_calls.load(std::memory_order_relaxed) - begin_;
}

double tolerance_ratio(std::span<const double> actual, std::span<const double> expected, double atol,
                       double rtol) noexcept {
  if (actual.size() != expected.size()) {
    return HUGE_VAL;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double allowed = atol + rtol * std::fabs(expected[i]);
    const double error = std::fabs(actual[i] - expected[i]);
    if (!std::isfinite(error)) {
      return HUGE_VAL;
    }
    worst = std::max(worst, error / allowed);
  }
  return worst;
}

}  // namespace nexus_test

void* operator new(std::size_t bytes) { return allocate(bytes); }
void* operator new[](std::size_t bytes) { return allocate(bytes); }
void* operator new(std::size_t bytes, std::align_val_t alignment) { return allocate_aligned(bytes, alignment); }
void* operator new[](std::size_t bytes, std::align_val_t alignment) { return allocate_aligned(bytes, alignment); }

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { release_aligned(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { release_aligned(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { release_aligned(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { release_aligned(pointer); }
