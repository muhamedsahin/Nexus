#pragma once

/**
 * @file threads.hpp
 * @brief FAZ 18.D: sabitlenmis (pinned) is parcacigi havuzu ve NUMA / buyuk sayfa / first-touch bellek.
 *
 * Havuz: varsayilan olarak fiziksel cekirdek basina bir isci (SMT esinin ikincisi kullanilmaz: FMA birimleri
 * paylasilir), isci i cekirdek i'nin ilk mantiksal islemcisine sabitlenir (Windows SetThreadAffinityMask /
 * Linux pthread_setaffinity_np). NEXUS_CPU_THREADS ortam degiskeni sayiyi degistirir. run(n, fn): isci w is
 * parcalarini statik olarak alir (belirlenimci) — fn(w, begin, end).
 * Bellek: alloc(bytes, {numa_node, large_pages, first_touch}): VirtualAllocExNuma (Windows) / mmap + mbind yok
 * (Linux: yalniz first-touch). Buyuk sayfa: MEM_LARGE_PAGES SeLockMemoryPrivilege ister; yoksa normal sayfaya
 * duser ve Buffer::large_pages = false + reason doldurulur (sessiz degil: cagiran gorur).
 * first_touch: havuz iscileri kendi dilimlerini sifirlar -> sayfalar iscinin dugumune yerlesir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace cpu {

class Pool {
 public:
  [[nodiscard]] static Pool& instance();
  explicit Pool(int workers, bool pin = true);
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  [[nodiscard]] int size() const noexcept;
  /// [0, n) araligini isciler arasinda esit (bitisik) boler; cagiran da isci 0 olarak calisir. Ic ice cagri seri.
  void run(std::size_t n, const std::function<void(int worker, std::size_t begin, std::size_t end)>& fn);
  /// Her isciyi bir kez calistirir: fn(worker).
  void each(const std::function<void(int worker)>& fn);
  /// Iscinin sabitlendigi mantiksal islemci (-1: sabitlenmedi).
  [[nodiscard]] int pinned_cpu(int worker) const;
  /// Isci icinden gozlemlenen calistigi mantiksal islemci (test: sabitleme dogrulamasi).
  [[nodiscard]] std::vector<int> observed_cpus();

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

struct AllocOptions {
  int numa_node = -1;       // -1: herhangi
  bool large_pages = false;
  bool first_touch = true;  // havuz iscileriyle sifirla
};

struct Buffer {
  void* data = nullptr;
  std::size_t bytes = 0;
  bool large_pages = false;
  int numa_node = -1;
  std::string reason;  // istenen ozellik saglanamadiysa neden
  Buffer() = default;
  Buffer(Buffer&&) noexcept;
  Buffer& operator=(Buffer&&) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer();
};

[[nodiscard]] Buffer alloc(std::size_t bytes, const AllocOptions& options = {});

}  // namespace cpu
}  // namespace nexus
