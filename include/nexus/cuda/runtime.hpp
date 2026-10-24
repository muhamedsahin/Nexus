#pragma once

/**
 * @file runtime.hpp
 * @brief CUDA akis/olay semantigi — FAZ 1.3: Stream, Event, StreamGuard, DeviceGuard.
 *
 * - `Stream`: sahipli (cudaStreamNonBlocking) veya varsayilan (legacy, nullptr) akis.
 * - `current_stream(device)`: is parcacigi basina gecerli akis; `StreamGuard` kapsam
 *   boyunca degistirir. Tum Nexus CUDA cekirdekleri gecerli akista calisir.
 * - `Event`: record/wait/synchronize/query/elapsed_ms.
 *
 * Bu baslik yalniz `NEXUS_ENABLE_CUDA` ile derlenen kodda kullanilir; CUDA'siz
 * derlemede GPU yolu yoktur ve bu baslik dahil edilmez (sessiz CPU geri donusu yok).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace nexus {
namespace cuda {

/// cudaError_t'yi aciklayici TensorError'a cevirir.
void check(cudaError_t status, const char* what);

#define NEXUS_CUDA_CHECK(expr) ::nexus::cuda::check((expr), #expr)

[[nodiscard]] int device_count() noexcept;
[[nodiscard]] int current_device();

/// Kapsam boyunca aygiti degistirir.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device);
  ~DeviceGuard();
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = 0;
};

/// Hafif akis tanitici. Sahiplik `owned` ile tasinir; kopyalar tanitici paylasir.
class Stream {
 public:
  Stream() = default;  // varsayilan (legacy) akis, aygit 0
  Stream(cudaStream_t handle, int device) noexcept : handle_(handle), device_(device) {}

  /// Yeni, engellemeyen akis olusturur (sahibi cagiran: destroy() ile birakir).
  [[nodiscard]] static Stream create(int device = 0, int priority = 0);
  void destroy();

  [[nodiscard]] cudaStream_t handle() const noexcept { return handle_; }
  [[nodiscard]] int device() const noexcept { return device_; }
  [[nodiscard]] bool is_default() const noexcept { return handle_ == nullptr; }
  void synchronize() const;
  [[nodiscard]] bool query() const;  // tum is bitti mi

  friend bool operator==(const Stream& a, const Stream& b) noexcept {
    return a.handle_ == b.handle_ && a.device_ == b.device_;
  }

 private:
  cudaStream_t handle_ = nullptr;
  int device_ = 0;
};

/// Is parcacigi basina gecerli akis (atanmamissa varsayilan akis).
[[nodiscard]] Stream current_stream(int device = -1);
void set_current_stream(const Stream& stream);

class StreamGuard {
 public:
  explicit StreamGuard(const Stream& stream);
  ~StreamGuard();
  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;

 private:
  Stream previous_;
};

/// CUDA olayi; varsayilan olarak zamanlama acik.
class Event {
 public:
  explicit Event(bool timing = true);
  ~Event();
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;

  void record(const Stream& stream);
  /// `stream`, bu olay tamamlanana kadar bekler (host engellenmez).
  void wait(const Stream& stream) const;
  void synchronize() const;
  [[nodiscard]] bool query() const;
  /// this -> end arasi milisaniye (ikisi de kayitli ve tamamlanmis olmali).
  [[nodiscard]] float elapsed_ms(const Event& end) const;
  [[nodiscard]] cudaEvent_t handle() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
  bool recorded_ = false;
};

}  // namespace cuda
}  // namespace nexus
