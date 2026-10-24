/**
 * @file runtime.cpp
 * @brief FAZ 1.3 CUDA akis/olay uygulamasi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/runtime.hpp"

#include "nexus/core/error.hpp"

#include <array>
#include <utility>

namespace nexus {
namespace cuda {
namespace {

constexpr int kMaxDevices = 16;
thread_local std::array<Stream, kMaxDevices> t_current{};
thread_local std::array<bool, kMaxDevices> t_has_current{};

}  // namespace

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw TensorError(std::string("Nexus CUDA: ") + what + " basarisiz: " + cudaGetErrorName(status) + " (" +
                      cudaGetErrorString(status) + ")");
  }
}

int device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // yapiskan olmayan hatayi temizle
    return 0;
  }
  return count;
}

int current_device() {
  int device = 0;
  NEXUS_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

DeviceGuard::DeviceGuard(int device) : previous_(current_device()) {
  if (device != previous_) {
    NEXUS_CUDA_CHECK(cudaSetDevice(device));
  }
}

DeviceGuard::~DeviceGuard() { cudaSetDevice(previous_); }

Stream Stream::create(int device, int priority) {
  DeviceGuard guard(device);
  cudaStream_t handle = nullptr;
  NEXUS_CUDA_CHECK(cudaStreamCreateWithPriority(&handle, cudaStreamNonBlocking, priority));
  return Stream(handle, device);
}

void Stream::destroy() {
  if (handle_ != nullptr) {
    cudaStreamDestroy(handle_);
    handle_ = nullptr;
  }
}

void Stream::synchronize() const { NEXUS_CUDA_CHECK(cudaStreamSynchronize(handle_)); }

bool Stream::query() const {
  const cudaError_t status = cudaStreamQuery(handle_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  NEXUS_CUDA_CHECK(status);
  return true;
}

Stream current_stream(int device) {
  if (device < 0) {
    device = current_device();
  }
  if (device >= kMaxDevices) {
    throw TensorError("Nexus CUDA: aygit indeksi desteklenen sinirin ustunde");
  }
  return t_has_current[static_cast<std::size_t>(device)] ? t_current[static_cast<std::size_t>(device)]
                                                         : Stream(nullptr, device);
}

void set_current_stream(const Stream& stream) {
  if (stream.device() < 0 || stream.device() >= kMaxDevices) {
    throw TensorError("Nexus CUDA: akis aygiti gecersiz");
  }
  t_current[static_cast<std::size_t>(stream.device())] = stream;
  t_has_current[static_cast<std::size_t>(stream.device())] = true;
}

StreamGuard::StreamGuard(const Stream& stream) : previous_(current_stream(stream.device())) {
  set_current_stream(stream);
}

StreamGuard::~StreamGuard() { set_current_stream(previous_); }

Event::Event(bool timing) {
  NEXUS_CUDA_CHECK(cudaEventCreateWithFlags(&event_, timing ? cudaEventDefault : cudaEventDisableTiming));
}

Event::~Event() {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
  }
}

Event::Event(Event&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)), recorded_(std::exchange(other.recorded_, false)) {}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
    event_ = std::exchange(other.event_, nullptr);
    recorded_ = std::exchange(other.recorded_, false);
  }
  return *this;
}

void Event::record(const Stream& stream) {
  NEXUS_CUDA_CHECK(cudaEventRecord(event_, stream.handle()));
  recorded_ = true;
}

void Event::wait(const Stream& stream) const {
  if (!recorded_) {
    return;  // kaydedilmemis olay: beklenecek is yok (PyTorch semantigi)
  }
  NEXUS_CUDA_CHECK(cudaStreamWaitEvent(stream.handle(), event_, 0));
}

void Event::synchronize() const {
  if (recorded_) {
    NEXUS_CUDA_CHECK(cudaEventSynchronize(event_));
  }
}

bool Event::query() const {
  if (!recorded_) {
    return true;
  }
  const cudaError_t status = cudaEventQuery(event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  NEXUS_CUDA_CHECK(status);
  return true;
}

float Event::elapsed_ms(const Event& end) const {
  if (!recorded_ || !end.recorded_) {
    throw TensorError("Nexus CUDA: elapsed_ms icin iki olay da kaydedilmeli");
  }
  float ms = 0.0f;
  NEXUS_CUDA_CHECK(cudaEventElapsedTime(&ms, event_, end.event_));
  return ms;
}

}  // namespace cuda
}  // namespace nexus
