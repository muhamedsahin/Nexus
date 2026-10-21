#pragma once

/**
 * @file mirrored_tensor.hpp
 * @brief Host ve hizlandirici kopyasini tutarli tutan cift tamponlu tensör.
 *
 * Bir egitim adiminda ayni parametre CPU'da (veri hazirlama, optimizer)
 * ve GPU'da (GEMM) kullanilir. `MirroredTensor` iki tamponu ve hangisinin
 * guncel oldugunu birlikte tasir; kopya yalnizca gerektiginde yapilir.
 *
 * Sozlesme (Kokkos DualView ile ayni):
 *   - Bir tarafa yazan kod once `modify_host()` / `modify_device()` cagirir.
 *   - Diger tarafi okuyacak kod once `sync_device()` / `sync_host()` cagirir.
 *   - Her iki taraf da senkronlanmadan degistirilirse `TensorError` atilir;
 *     sessiz veri kaybi olmaz.
 *
 * Akis siralamasi: kopyalar nesnenin kendi engellemeyen kopya akisinda
 * kosar. Cihaz tamponunu okuyan/yazan cekirdekler `device_stream()` uzerinde
 * kabul edilir (varsayilan: CUDA'nin eski varsayilan akisi, MatrixFlash
 * cekirdeklerinin akisi). `sync_host` kopyadan once bu akistaki yazmalari,
 * `sync_device` ise bu akistaki okumalari olaylarla bekler; ardindan cihaz
 * akisi H2D kopyasini bekler. Host hicbir siralama icin engellenmez.
 *
 * Cihaz CPU ise tek tampon vardir (`unified()`): senkronlar no-op'tur ve
 * hicbir kopya yapilmaz. `wrap_host` baska bir kutuphanenin host bellegini
 * (ornegin nexus_model::Tensor) kopyalamadan kullanir; yalnizca cihaz aynasi
 * ayrilir. `wrap` iki tarafi da dis bellek olarak alir ve hicbir sey ayirmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/error.hpp"
#include "nexus/core/memory.hpp"
#include "nexus/core/tensor_view.hpp"

#include <cstddef>
#include <cstdint>

namespace nexus {

/// Iki tampondan hangisinin son yazmayi tasidigi.
enum class MirrorState : std::uint8_t {
  Synced = 0,   ///< Iki taraf ayni bayti tasir.
  HostAhead,    ///< Host yazildi, cihaz eski.
  DeviceAhead   ///< Cihaz yazildi, host eski.
};

[[nodiscard]] inline constexpr const char* to_string(MirrorState state) noexcept {
  switch (state) {
    case MirrorState::Synced: return "synced";
    case MirrorState::HostAhead: return "host_ahead";
    case MirrorState::DeviceAhead: return "device_ahead";
  }
  return "unknown";
}

class MirroredTensor {
 public:
  MirroredTensor() = default;

  /// Sayfa kilitli host (cihaz hizlandiriciysa) ve tembel ayrilan cihaz aynasi.
  /// Host sifirla doldurulur; durum HostAhead'dir.
  explicit MirroredTensor(Shape shape, DeviceRef device = DeviceRef::cuda(0));

  /// Dis host bellegini kopyalamadan sarar. Sahibi bu nesneden uzun yasamalidir.
  [[nodiscard]] static MirroredTensor wrap_host(float* host, Shape shape, DeviceRef device);

  /// Iki tarafi da dis bellek olarak sarar; hicbir ayirma yapmaz.
  [[nodiscard]] static MirroredTensor wrap(float* host, float* device_data, Shape shape, DeviceRef device,
                                           MirrorState state);

  MirroredTensor(const MirroredTensor&) = delete;
  MirroredTensor& operator=(const MirroredTensor&) = delete;
  MirroredTensor(MirroredTensor&& other) noexcept;
  MirroredTensor& operator=(MirroredTensor&& other) noexcept;
  ~MirroredTensor();

  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] std::size_t numel() const noexcept { return shape_.numel(); }
  [[nodiscard]] std::size_t bytes() const noexcept { return shape_.numel() * sizeof(float); }
  [[nodiscard]] DeviceRef device() const noexcept { return device_; }
  [[nodiscard]] MirrorState state() const noexcept { return state_; }
  [[nodiscard]] bool unified() const noexcept { return device_.is_host(); }
  [[nodiscard]] bool device_allocated() const noexcept { return device_data_ != nullptr; }
  [[nodiscard]] bool host_pinned() const noexcept { return host_space_ == MemorySpace::PinnedHost; }

  [[nodiscard]] float* host_data() noexcept { return host_; }
  [[nodiscard]] const float* host_data() const noexcept { return host_; }
  /// Cihaz aynasi yoksa burada ayrilir. Unified modda host isaretcisini dondurur.
  [[nodiscard]] float* device_data();

  [[nodiscard]] TensorView<float> host_view() noexcept;
  [[nodiscard]] TensorView<const float> host_view() const noexcept;
  [[nodiscard]] TensorView<float> device_view();

  /// Host'a yazmadan once cagrilir. Kuyrukta H2D kopyasi varsa once onu bekler;
  /// aksi halde DMA host'u okurken yazma yarisir.
  void modify_host();
  /// Cihaza yazmadan once cagrilir. Bekleyen kopyayi tamamlar, sonra DeviceAhead olur.
  void modify_device();
  [[nodiscard]] bool needs_sync_device() const noexcept { return state_ == MirrorState::HostAhead; }
  [[nodiscard]] bool needs_sync_host() const noexcept { return state_ == MirrorState::DeviceAhead; }
  /// `sync_device(false)` ile kuyruga girmis ve henuz beklenmemis kopya var mi.
  [[nodiscard]] bool copy_pending() const noexcept { return pending_; }

  /// Host guncelse H2D kopyasini kuyruga koyar. Kopya `device_stream()`'deki
  /// onceki isten sonra baslar; sonraki cihaz isi kopyayi otomatik bekler.
  /// `blocking` ise host da kopyanin bitmesini bekler.
  void sync_device(bool blocking = true);
  /// Cihaz guncelse `device_stream()`'deki yazmalardan sonra D2H kopyasini
  /// yapar ve her zaman bekler; host hemen okunabilir.
  void sync_host();
  /// Bekleyen asenkron kopyalari tamamlar.
  void synchronize();

  /// Aliased host bellegini CUDA'ya sayfa kilitli kaydeder. Zaten kilitliyse true.
  bool pin_host();

  [[nodiscard]] std::uint64_t bytes_to_device() const noexcept { return bytes_to_device_; }
  [[nodiscard]] std::uint64_t bytes_to_host() const noexcept { return bytes_to_host_; }
  /// Nesnenin kendi kopya akisi; ilk senkronda acilir.
  [[nodiscard]] StreamHandle stream() const noexcept { return stream_; }
  /// Cihaz tamponunu kullanan cekirdeklerin akisi. nullptr: eski varsayilan akis.
  void set_device_stream(StreamHandle stream) noexcept { device_stream_ = stream; }
  [[nodiscard]] StreamHandle device_stream() const noexcept { return device_stream_; }

  void swap(MirroredTensor& other) noexcept;

 private:
  void ensure_device();
  void ensure_stream();
  void release() noexcept;

  Shape shape_{};
  DeviceRef device_{};
  float* host_ = nullptr;
  float* device_data_ = nullptr;
  MemorySpace host_space_ = MemorySpace::Host;
  MirrorState state_ = MirrorState::Synced;
  StreamHandle stream_ = nullptr;
  StreamHandle device_stream_ = nullptr;
  EventHandle event_ = nullptr;
  bool owns_host_ = false;
  bool owns_device_ = false;
  bool registered_host_ = false;
  bool pending_ = false;
  std::uint64_t bytes_to_device_ = 0;
  std::uint64_t bytes_to_host_ = 0;
};

inline void swap(MirroredTensor& lhs, MirroredTensor& rhs) noexcept { lhs.swap(rhs); }

}  // namespace nexus
