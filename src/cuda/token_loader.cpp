/**
 * @file token_loader.cpp
 * @brief FAZ 8.F: pinned halkali asenkron token yukleyici (bkz. token_loader.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/token_loader.hpp"

#include <chrono>
#include <stdexcept>

namespace nexus {
namespace cuda {

TokenLoader::TokenLoader(Producer producer, std::size_t tokens_per_batch, int slots, int device)
    : producer_(std::move(producer)), count_(tokens_per_batch), device_(device) {
  if (!producer_ || count_ == 0 || slots < 2) {
    throw std::invalid_argument("TokenLoader: uretici, sifirdan buyuk yigin ve en az 2 yuva gerekir");
  }
  DeviceGuard guard(device_);
  copy_stream_ = Stream::create(device_);
  slots_.resize(static_cast<std::size_t>(slots));
  for (Slot& s : slots_) {
    s.host = static_cast<std::int32_t*>(pinned_allocator().allocate(count_ * sizeof(std::int32_t)));
    s.device = DeviceBuffer(count_ * sizeof(std::int32_t), copy_stream_);
  }
  free_ = slots;
  worker_ = std::thread([this] { run(); });
}

TokenLoader::~TokenLoader() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
  // Hesap akisi son yigini hala okuyor olabilir: tamponlar birakilmadan once aygit bosaltilir.
  cudaDeviceSynchronize();
  for (Slot& s : slots_) {
    s.device.reset();
    pinned_allocator().deallocate(s.host);
  }
  copy_stream_.destroy();
}

void TokenLoader::run() {
  try {
    NEXUS_CUDA_CHECK(cudaSetDevice(device_));
    const int n = static_cast<int>(slots_.size());
    for (;;) {
      int k = 0;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return stop_ || free_ > 0; });
        if (stop_) return;
        k = fill_;
      }
      Slot& s = slots_[static_cast<std::size_t>(k)];
      // Hesap akisi bu yuvanin aygit tamponunu bitirmeden pinned/aygit tamponu yeniden yazilmaz.
      // consumed, hesap akisi h2d_done'u bekledikten sonra kaydedildigi icin onceki H2D de bitmistir.
      if (s.consumed_pending) s.consumed.synchronize();
      producer_(s.host, count_);
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(s.device.data(), s.host, count_ * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       copy_stream_.handle()));
      s.h2d_done.record(copy_stream_);
      {
        std::lock_guard<std::mutex> lock(mu_);
        s.ready = true;
        fill_ = (k + 1) % n;
        --free_;
      }
      cv_.notify_all();
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      error_ = std::current_exception();
    }
    cv_.notify_all();
  }
}

const std::int32_t* TokenLoader::next(const Stream& compute) {
  const int n = static_cast<int>(slots_.size());
  if (held_ >= 0) {
    Slot& h = slots_[static_cast<std::size_t>(held_)];
    h.consumed.record(compute);
    {
      std::lock_guard<std::mutex> lock(mu_);
      h.consumed_pending = true;
      h.ready = false;
      ++free_;
      held_ = -1;
    }
    cv_.notify_all();
  }
  int k = 0;
  const auto t0 = std::chrono::steady_clock::now();
  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return slots_[static_cast<std::size_t>(take_)].ready || error_ != nullptr; });
    if (!slots_[static_cast<std::size_t>(take_)].ready) std::rethrow_exception(error_);
    k = take_;
    take_ = (k + 1) % n;
  }
  wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  Slot& s = slots_[static_cast<std::size_t>(k)];
  s.h2d_done.wait(compute);
  held_ = k;
  ++taken_;
  return s.device.as<std::int32_t>();
}

}  // namespace cuda
}  // namespace nexus
