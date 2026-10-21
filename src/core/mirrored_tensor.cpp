/**
 * @file mirrored_tensor.cpp
 * @brief MirroredTensor icin ayirma, durum gecisleri ve akis uzerinde kopya.
 *
 * Kopyalar nesnenin kendi engellemeyen akisinda kuyruga girer; boylece
 * varsayilan akistaki baska bir is (ornegin MatrixFlash GEMM'i) ile
 * ortusebilir. `sync_host` her zaman bekler cunku cagiran hemen host'u okur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/mirrored_tensor.hpp"

#include <cstring>
#include <utility>

namespace nexus {

MirroredTensor::MirroredTensor(Shape shape, DeviceRef device) : shape_(std::move(shape)), device_(device) {
  if (device_.is_hybrid()) {
    throw TensorError("Nexus: hybrid bir yurutme politikasidir, MirroredTensor icin somut cihaz verin");
  }
  if (bytes() == 0) {
    return;
  }
  host_space_ = unified() ? MemorySpace::Host : MemorySpace::PinnedHost;
  host_ = static_cast<float*>(unified() ? allocate_aligned(bytes()) : allocate_pinned(bytes()));
  owns_host_ = true;
  std::memset(host_, 0, bytes());
  state_ = unified() ? MirrorState::Synced : MirrorState::HostAhead;
}

MirroredTensor MirroredTensor::wrap_host(float* host, Shape shape, DeviceRef device) {
  if (shape.numel() > 0 && host == nullptr) {
    throw TensorError("Nexus: bos olmayan wrap_host icin host isaretcisi gerekli");
  }
  if (device.is_hybrid()) {
    throw TensorError("Nexus: hybrid bir yurutme politikasidir, MirroredTensor icin somut cihaz verin");
  }
  MirroredTensor mirror;
  mirror.shape_ = std::move(shape);
  mirror.device_ = device;
  mirror.host_ = host;
  mirror.host_space_ = MemorySpace::Host;
  mirror.state_ = mirror.unified() ? MirrorState::Synced : MirrorState::HostAhead;
  return mirror;
}

MirroredTensor MirroredTensor::wrap(float* host, float* device_data, Shape shape, DeviceRef device,
                                    MirrorState state) {
  if (shape.numel() > 0 && (host == nullptr || device_data == nullptr)) {
    throw TensorError("Nexus: wrap icin host ve cihaz isaretcisi gerekli");
  }
  if (device.is_host() && host != device_data) {
    throw TensorError("Nexus: CPU cihazinda host ve cihaz tamponu ayni adres olmali");
  }
  MirroredTensor mirror;
  mirror.shape_ = std::move(shape);
  mirror.device_ = device;
  mirror.host_ = host;
  mirror.device_data_ = device.is_host() ? nullptr : device_data;
  mirror.host_space_ = MemorySpace::Host;
  mirror.state_ = device.is_host() ? MirrorState::Synced : state;
  return mirror;
}

MirroredTensor::MirroredTensor(MirroredTensor&& other) noexcept { swap(other); }

MirroredTensor& MirroredTensor::operator=(MirroredTensor&& other) noexcept {
  if (this != &other) {
    release();
    swap(other);
  }
  return *this;
}

MirroredTensor::~MirroredTensor() { release(); }

void MirroredTensor::swap(MirroredTensor& other) noexcept {
  using std::swap;
  swap(shape_, other.shape_);
  swap(device_, other.device_);
  swap(host_, other.host_);
  swap(device_data_, other.device_data_);
  swap(host_space_, other.host_space_);
  swap(state_, other.state_);
  swap(stream_, other.stream_);
  swap(device_stream_, other.device_stream_);
  swap(event_, other.event_);
  swap(owns_host_, other.owns_host_);
  swap(owns_device_, other.owns_device_);
  swap(registered_host_, other.registered_host_);
  swap(pending_, other.pending_);
  swap(bytes_to_device_, other.bytes_to_device_);
  swap(bytes_to_host_, other.bytes_to_host_);
}

void MirroredTensor::release() noexcept {
  if (stream_ != nullptr) {
    destroy_stream(device_, stream_);
    stream_ = nullptr;
  }
  if (event_ != nullptr) {
    destroy_event(device_, event_);
    event_ = nullptr;
  }
  if (registered_host_) {
    unregister_host_memory(host_);
    registered_host_ = false;
  }
  if (owns_device_) {
    release_owned(device_data_);
  }
  if (owns_host_) {
    release_owned(host_);
  }
  host_ = nullptr;
  device_data_ = nullptr;
  owns_host_ = false;
  owns_device_ = false;
  pending_ = false;
  state_ = MirrorState::Synced;
}

void MirroredTensor::ensure_device() {
  if (unified() || device_data_ != nullptr || bytes() == 0) {
    return;
  }
  device_data_ = static_cast<float*>(allocate_device(device_, bytes()));
  owns_device_ = true;
}

void MirroredTensor::ensure_stream() {
  if (unified() || stream_ != nullptr) {
    return;
  }
  stream_ = create_stream(device_);
  try {
    event_ = create_event(device_);
  } catch (...) {
    destroy_stream(device_, stream_);
    stream_ = nullptr;
    throw;
  }
}

float* MirroredTensor::device_data() {
  if (unified()) {
    return host_;
  }
  ensure_device();
  return device_data_;
}

TensorView<float> MirroredTensor::host_view() noexcept {
  return TensorView<float>(host_, shape_, shape_.contiguous_strides(), true, host_space_, DeviceRef::cpu());
}

TensorView<const float> MirroredTensor::host_view() const noexcept {
  return TensorView<const float>(host_, shape_, shape_.contiguous_strides(), true, host_space_, DeviceRef::cpu());
}

TensorView<float> MirroredTensor::device_view() {
  float* data = device_data();
  const MemorySpace space = unified() ? host_space_ : MemorySpace::Device;
  return TensorView<float>(data, shape_, shape_.contiguous_strides(), true, space, device_);
}

void MirroredTensor::modify_host() {
  if (unified()) {
    return;
  }
  if (state_ == MirrorState::DeviceAhead) {
    throw TensorError("Nexus: cihazdaki yazma senkronlanmadan host degistirilemez; once sync_host() cagirin");
  }
  synchronize();
  state_ = MirrorState::HostAhead;
}

void MirroredTensor::modify_device() {
  if (unified()) {
    return;
  }
  if (state_ == MirrorState::HostAhead) {
    throw TensorError("Nexus: hosttaki yazma senkronlanmadan cihaz degistirilemez; once sync_device() cagirin");
  }
  synchronize();
  ensure_device();
  state_ = MirrorState::DeviceAhead;
}

void MirroredTensor::sync_device(bool blocking) {
  if (unified() || state_ != MirrorState::HostAhead || bytes() == 0) {
    return;
  }
  ensure_device();
  ensure_stream();
  // Cihaz akisinda hala bu tamponu okuyan is varsa uzerine yazilmasin (WAR).
  stream_wait(device_, stream_, device_stream_, event_);
  copy_bytes_async(device_data_, MemorySpace::Device, device_, host_, host_space_, DeviceRef::cpu(), bytes(),
                   stream_);
  // Cihaz akisina sonradan giren cekirdekler kopyayi bekler (RAW); host beklemez.
  stream_wait(device_, device_stream_, stream_, event_);
  bytes_to_device_ += bytes();
  state_ = MirrorState::Synced;
  pending_ = true;
  if (blocking) {
    synchronize();
  }
}

void MirroredTensor::sync_host() {
  if (unified() || state_ != MirrorState::DeviceAhead || bytes() == 0) {
    return;
  }
  ensure_stream();
  // Cihaz akisindaki yazmalar (cekirdekler, cudaMemcpy) kopyadan once bitmeli.
  stream_wait(device_, stream_, device_stream_, event_);
  copy_bytes_async(host_, host_space_, DeviceRef::cpu(), device_data_, MemorySpace::Device, device_, bytes(),
                   stream_);
  pending_ = true;
  synchronize();
  bytes_to_host_ += bytes();
  state_ = MirrorState::Synced;
}

void MirroredTensor::synchronize() {
  if (pending_ && stream_ != nullptr) {
    synchronize_stream(device_, stream_);
  }
  pending_ = false;
}

bool MirroredTensor::pin_host() {
  if (unified() || host_ == nullptr) {
    return false;
  }
  if (host_space_ == MemorySpace::PinnedHost) {
    return true;
  }
  if (!register_host_memory(host_, bytes())) {
    return false;
  }
  registered_host_ = true;
  host_space_ = MemorySpace::PinnedHost;
  return true;
}

}  // namespace nexus
