/**
 * @file threads.cpp
 * @brief FAZ 18.D: sabitli havuz (statik is bolme, belirlenimci), NUMA / buyuk sayfa ayirici.
 *
 * Havuz uyandirmasi: ardisik cagrilarda (orn. 22 katmanli LLM cozme veya derlenmis grafik) isciler kisa bir
 * `_mm_pause()` spin penceresinde bekler (< 0.3 us uyanma gecikmesi); bosta kaldiginda muteks altinda
 * `sleeping` sayacini artirip kosul degiskeninde uyur (0% CPU, kayip uyandirma yok). Ana is parcacigi
 * cekirdek 0'a yalniz ilk cagrida sabitlenir; her cagrida iki kez `SetThreadAffinityMask` IPI olusturmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cpu/threads.hpp"

#include "nexus/cpu/features.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

namespace nexus {
namespace cpu {

namespace {
thread_local bool t_in_pool = false;
thread_local int t_pinned_cpu = -2;

int env_threads() {
#if defined(_MSC_VER)
  char* v = nullptr;
  std::size_t n = 0;
  if (_dupenv_s(&v, &n, "NEXUS_CPU_THREADS") == 0 && v != nullptr) {
    const int r = std::atoi(v);
    std::free(v);
    return r;
  }
  return 0;
#else
  const char* v = std::getenv("NEXUS_CPU_THREADS");
  return v ? std::atoi(v) : 0;
#endif
}

/// Fiziksel cekirdek basina ilk mantiksal islemci, sonra (gerekirse) SMT esleri.
std::vector<int> pin_order() {
  const Features& f = features();
  std::vector<int> first, second;
  std::vector<bool> seen(static_cast<std::size_t>(f.physical) + 1, false);
  for (int l = 0; l < f.logical; ++l) {
    const int c = l < static_cast<int>(f.core_of.size()) ? f.core_of[static_cast<std::size_t>(l)] : l;
    if (c >= 0 && c < static_cast<int>(seen.size()) && !seen[static_cast<std::size_t>(c)]) {
      seen[static_cast<std::size_t>(c)] = true;
      first.push_back(l);
    } else {
      second.push_back(l);
    }
  }
  first.insert(first.end(), second.begin(), second.end());
  return first;
}

bool pin_self(int cpu) {
#if defined(_WIN32)
  return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu) != 0;
#else
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  return pthread_setaffinity_np(pthread_self(), sizeof(s), &s) == 0;
#endif
}

int current_cpu() {
#if defined(_WIN32)
  return static_cast<int>(GetCurrentProcessorNumber());
#else
  return sched_getcpu();
#endif
}
}  // namespace

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
struct Pool::Impl {
  int n = 1;
  std::vector<int> cpus;
  std::vector<std::thread> threads;
  std::mutex mu;
  std::mutex submit;  // disaridan esanli each() cagrilari sirayla (is gostergesi tek)
  std::condition_variable wake, done;
  alignas(64) std::atomic<std::uint64_t> generation{0};
  alignas(64) std::atomic<int> remaining{0};
  alignas(64) std::atomic<int> sleeping{0};
  alignas(64) std::atomic<bool> stop{false};
  const std::function<void(int)>* job = nullptr;

  static constexpr int kSpinIters = 8000;

  void loop(int w) {
    t_in_pool = true;
    std::uint64_t seen = 0;
    for (;;) {
      std::uint64_t g = generation.load(std::memory_order_acquire);
      if (g == seen) {
        for (int spin = 0; spin < kSpinIters; ++spin) {
          _mm_pause();
          g = generation.load(std::memory_order_acquire);
          if (g != seen || stop.load(std::memory_order_relaxed)) break;
        }
      }
      if (g == seen) {
        std::unique_lock<std::mutex> lock(mu);
        sleeping.fetch_add(1, std::memory_order_seq_cst);
        wake.wait(lock, [&] {
          return stop.load(std::memory_order_relaxed) || generation.load(std::memory_order_acquire) != seen;
        });
        sleeping.fetch_sub(1, std::memory_order_relaxed);
        if (stop.load(std::memory_order_relaxed)) return;
        g = generation.load(std::memory_order_acquire);
      }
      if (stop.load(std::memory_order_relaxed)) return;
      seen = g;
      const std::function<void(int)>* j = job;
      (*j)(w);
      if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mu);
        done.notify_one();
      }
    }
  }
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

Pool& Pool::instance() {
  static Pool p(env_threads() > 0 ? env_threads() : features().physical, true);
  return p;
}

Pool::Pool(int workers, bool pin) : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.n = std::max(1, workers);
  const std::vector<int> order = pin_order();
  m.cpus.assign(static_cast<std::size_t>(m.n), -1);
  if (pin)
    for (int w = 0; w < m.n; ++w) m.cpus[static_cast<std::size_t>(w)] = order[static_cast<std::size_t>(w) % order.size()];
  for (int w = 1; w < m.n; ++w) {
    m.threads.emplace_back([this, w] {
      if (impl_->cpus[static_cast<std::size_t>(w)] >= 0) pin_self(impl_->cpus[static_cast<std::size_t>(w)]);
      impl_->loop(w);
    });
  }
}

Pool::~Pool() {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->stop.store(true, std::memory_order_release);
  }
  impl_->wake.notify_all();
  for (auto& t : impl_->threads) t.join();
}

int Pool::size() const noexcept { return impl_->n; }
int Pool::pinned_cpu(int w) const { return impl_->cpus.at(static_cast<std::size_t>(w)); }

void Pool::each(const std::function<void(int)>& fn) {
  Impl& m = *impl_;
  if (t_in_pool || m.n == 1) {  // ic ice: seri (kilitlenme yok)
    for (int w = 0; w < m.n; ++w) fn(w);
    return;
  }
  std::lock_guard<std::mutex> serial(m.submit);
  if (m.cpus[0] >= 0 && t_pinned_cpu != m.cpus[0]) {
    pin_self(m.cpus[0]);
    t_pinned_cpu = m.cpus[0];
  }
  m.job = &fn;
  m.remaining.store(m.n - 1, std::memory_order_relaxed);
  m.generation.fetch_add(1, std::memory_order_seq_cst);
  if (m.sleeping.load(std::memory_order_seq_cst) > 0) {
    std::lock_guard<std::mutex> lock(m.mu);
    m.wake.notify_all();
  }
  t_in_pool = true;
  fn(0);
  t_in_pool = false;
  int spin = 0;
  while (m.remaining.load(std::memory_order_acquire) > 0) {
    if (++spin < Impl::kSpinIters) {
      _mm_pause();
    } else {
      std::unique_lock<std::mutex> lock(m.mu);
      m.done.wait(lock, [&] { return m.remaining.load(std::memory_order_acquire) == 0; });
      break;
    }
  }
  m.job = nullptr;
}

void Pool::run(std::size_t n, const std::function<void(int, std::size_t, std::size_t)>& fn) {
  const int W = size();
  const std::function<void(int)> body = [&](int w) {
    const std::size_t b = n * static_cast<std::size_t>(w) / static_cast<std::size_t>(W);
    const std::size_t e = n * static_cast<std::size_t>(w + 1) / static_cast<std::size_t>(W);
    if (b < e) fn(w, b, e);
  };
  each(body);
}

std::vector<int> Pool::observed_cpus() {
  std::vector<int> out(static_cast<std::size_t>(size()), -1);
  each([&](int w) {
    // kisa bir is: zamanlayicinin tasima sansini olcmek icin birkac ornek, cogunluk
    int votes[256] = {};
    for (int i = 0; i < 64; ++i) {
      const int c = current_cpu();
      if (c >= 0 && c < 256) ++votes[c];
      volatile double x = 0;
      for (int k = 0; k < 2000; ++k) x = x + k;
    }
    out[static_cast<std::size_t>(w)] = static_cast<int>(std::max_element(votes, votes + 256) - votes);
  });
  return out;
}

// ---- bellek ------------------------------------------------------------------------------
Buffer::Buffer(Buffer&& o) noexcept { *this = std::move(o); }
Buffer& Buffer::operator=(Buffer&& o) noexcept {
  if (this != &o) {
    this->~Buffer();
    data = o.data;
    bytes = o.bytes;
    large_pages = o.large_pages;
    numa_node = o.numa_node;
    reason = std::move(o.reason);
    o.data = nullptr;
  }
  return *this;
}
Buffer::~Buffer() {
  if (data == nullptr) return;
#if defined(_WIN32)
  VirtualFree(data, 0, MEM_RELEASE);
#else
  munmap(data, bytes);
#endif
  data = nullptr;
}

Buffer alloc(std::size_t bytes, const AllocOptions& o) {
  Buffer b;
  b.bytes = bytes;
#if defined(_WIN32)
  if (o.large_pages) {
    const SIZE_T lp = GetLargePageMinimum();
    HANDLE tok = nullptr;
    bool priv = false;
    if (lp != 0 && OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
      TOKEN_PRIVILEGES tp{};
      tp.PrivilegeCount = 1;
      if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
        priv = GetLastError() == ERROR_SUCCESS;
      }
      CloseHandle(tok);
    }
    if (priv) {
      const std::size_t rounded = (bytes + lp - 1) / lp * lp;
      b.data = o.numa_node >= 0 ? VirtualAllocExNuma(GetCurrentProcess(), nullptr, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                                     PAGE_READWRITE, static_cast<DWORD>(o.numa_node))
                                : VirtualAlloc(nullptr, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
      if (b.data) {
        b.large_pages = true;
        b.bytes = rounded;
      } else {
        b.reason = "MEM_LARGE_PAGES ayirma basarisiz (hata " + std::to_string(GetLastError()) + ")";
      }
    } else {
      b.reason = lp == 0 ? "isletim sistemi buyuk sayfa desteklemiyor"
                         : "SeLockMemoryPrivilege yok (yonetici 'Bellekte sayfalari kilitle' hakkini vermeli); 4 KB sayfa";
    }
  }
  if (!b.data) {
    b.data = o.numa_node >= 0 ? VirtualAllocExNuma(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE,
                                                   static_cast<DWORD>(o.numa_node))
                              : VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  }
#else
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | (o.large_pages ? MAP_HUGETLB : 0), -1, 0);
  if (p == MAP_FAILED && o.large_pages) {
    b.reason = "MAP_HUGETLB basarisiz; 4 KB sayfa";
    p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  } else if (o.large_pages) {
    b.large_pages = true;
  }
  b.data = p == MAP_FAILED ? nullptr : p;
#endif
  if (!b.data) throw std::bad_alloc();
  b.numa_node = o.numa_node;
  if (o.first_touch) {
    auto* c = static_cast<char*>(b.data);
    Pool::instance().run(b.bytes, [&](int, std::size_t s, std::size_t e) { std::memset(c + s, 0, e - s); });
  }
  return b;
}

}  // namespace cpu
}  // namespace nexus
