#pragma once

/**
 * @file token_loader.hpp
 * @brief FAZ 8.F: asenkron GPU token yukleyici — arka plan is parcacigi -> pinned halka ->
 *        kopya akisinda H2D -> olayla hesap akisina devir.
 *
 * Akis: uretici geri cagrisi (or. nexusdata::text::TokenStream) is parcacigi icinde pinned
 * yuvayi doldurur, ayri kopya akisinda cudaMemcpyAsync baslatilir ve `h2d_done` olayi kaydedilir.
 * `next(compute)` hazir yuvayi alir; hesap akisi olayi bekler (host engellenmez). Yuva bir
 * sonraki `next` cagrisinda birakilir: hesap akisina `consumed` olayi kaydedilir, is parcacigi
 * yuvayi yeniden kullanmadan once bu olayi bekler (aygit tamponu okunurken uzerine yazilmaz).
 * `wait_ms()`: `next`'in hazir yuva icin bekledigi toplam host suresi (veri bekleme olcusu).
 * Ureticideki istisna yakalanir ve `next`'te yeniden firlatilir (sessiz bos yigin yok).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/runtime.hpp"

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace nexus {
namespace cuda {

class TokenLoader {
 public:
  /// dst'ye tam `count` token yazar (is parcacigi icinde cagrilir).
  using Producer = std::function<void(std::int32_t* dst, std::size_t count)>;

  TokenLoader(Producer producer, std::size_t tokens_per_batch, int slots = 4, int device = 0);
  ~TokenLoader();
  TokenLoader(const TokenLoader&) = delete;
  TokenLoader& operator=(const TokenLoader&) = delete;

  /// Bir sonraki yigini dondurur (aygit isaretcisi). `compute` H2D bitene kadar bekletilir;
  /// onceki yigin birakilir. Isaretci bir sonraki `next` cagrisina kadar gecerlidir.
  [[nodiscard]] const std::int32_t* next(const Stream& compute);
  [[nodiscard]] double wait_ms() const noexcept { return wait_ms_; }
  [[nodiscard]] std::int64_t batches() const noexcept { return taken_; }
  [[nodiscard]] std::size_t tokens_per_batch() const noexcept { return count_; }

 private:
  struct Slot {
    std::int32_t* host = nullptr;
    DeviceBuffer device;
    Event h2d_done{false};
    Event consumed{false};
    bool consumed_pending = false;
    bool ready = false;
  };
  void run();

  Producer producer_;
  std::size_t count_;
  int device_;
  Stream copy_stream_;
  std::vector<Slot> slots_;
  std::mutex mu_;
  std::condition_variable cv_;
  int fill_ = 0;    // is parcaciginin dolduracagi yuva
  int take_ = 0;    // tuketicinin alacagi yuva
  int held_ = -1;   // tuketicinin elindeki yuva
  int free_ = 0;    // bos yuva sayisi
  bool stop_ = false;
  std::exception_ptr error_;
  double wait_ms_ = 0.0;
  std::int64_t taken_ = 0;
  std::thread worker_;
};

}  // namespace cuda
}  // namespace nexus
