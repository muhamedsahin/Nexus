#pragma once

/**
 * @file memory.hpp
 * @brief 64 bayt hizali host, sayfa kilitli host ve CUDA cihaz bellegi.
 *
 * Sahip olunan her blok bir kayda yazilir. `Tensor` yalnizca kendisinin
 * ayirdigi isaretciyi geri verir. `alias` ve `share` kayda girmez; bunlar
 * baska bir kutuphanenin tamponunu goruntuler, serbest birakmaz.
 *
 * Ayirma sayaci `allocation_count()` yalnizca bu kapidan gecen ayirmalari
 * sayar (monoton toplam); canli sayi icin `live_allocation_count()`. Sifir kopyali gorunum bu sayaci artirmaz. Testler bunu dogrular.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/device.hpp"

#include <cstddef>
#include <cstdint>

namespace nexus {

enum class AllocKind : std::uint8_t { AlignedHost = 1, PinnedHost = 2, Device = 3 };

struct AllocInfo {
  AllocKind kind = AllocKind::AlignedHost;
  DeviceRef device{};
  std::size_t bytes = 0;
};

/// 64 bayt hizali sayfalanabilir host blogu. `bytes == 0` ise nullptr.
[[nodiscard]] void* allocate_aligned(std::size_t bytes);

/// Sayfa kilitli host. CUDA varsa once cudaMallocHost denenir.
/// Toolkit bir aygit acamazsa isletim sisteminin sayfa kilidi kullanilir.
[[nodiscard]] void* allocate_pinned(std::size_t bytes);

/// Hizlandirici bellegi. Faz 1'de yalnizca CUDA gercekten ayirir.
/// ROCm, TPU ve diger aileler acik bir TensorError firlatir.
[[nodiscard]] void* allocate_device(DeviceRef device, std::size_t bytes);

/// `allocate_*` ile alinan isaretciyi turune gore geri verir. nullptr ve
/// kayitta olmayan isaretci no-op'tur (yikici icinden cagrilabilsin diye).
void release_owned(void* ptr) noexcept;

[[nodiscard]] bool owns_allocation(const void* ptr) noexcept;
[[nodiscard]] AllocInfo allocation_info(const void* ptr);
[[nodiscard]] std::uint64_t allocation_count() noexcept;
/// Su an canli (serbest birakilmamis) kapi ayirmasi sayisi; sizinti testleri icin.
/// `allocation_count()` ise toplam (monoton) sayactir ve serbest birakmada azalmaz.
[[nodiscard]] std::uint64_t live_allocation_count() noexcept;

/// k64 havuzunun reserved/current parcalanma orani (>=1.0). Canli ayirma yoksa
/// reserved>0 icin -1.0, ikisi de 0 icin 1.0 doner. FAZ 1 metriği.
[[nodiscard]] double memory_fragmentation_ratio();

/// Host, pinned ve cihaz arasinda bayt kopyasi. Bos uzunluk no-op'tur.
/// Cihaz kopyasi CUDA toolkit bagli degilse TensorError firlatir.
void copy_bytes(void* dst, MemorySpace dst_space, DeviceRef dst_device, const void* src,
                MemorySpace src_space, DeviceRef src_device, std::size_t bytes);

void zero_bytes(void* dst, MemorySpace space, DeviceRef device, std::size_t bytes);

/// Cihaz uzerinde sifirdan farkli doldurma, host'ta hazirlanip kopyalanir.
/// Bu yol `backward_into` tarafindan kullanilmaz.
void fill_float(float* dst, MemorySpace space, DeviceRef device, std::size_t count, float value);

/// Asenkron kopya kuyrugu. CUDA'da `cudaStream_t`, CPU'da nullptr'dir.
/// nullptr akis CUDA'nin eski (legacy) varsayilan akisina gider.
using StreamHandle = void*;

/// Engellemeyen (cudaStreamNonBlocking) bir akis acar. CPU cihazi icin nullptr doner.
[[nodiscard]] StreamHandle create_stream(DeviceRef device);
/// Bekleyen kopyalar bitene kadar bekler, sonra akisi kapatir. Hata yutulur.
void destroy_stream(DeviceRef device, StreamHandle stream) noexcept;
/// Akistaki tum kopyalar bitene kadar bekler. CPU'da no-op.
void synchronize_stream(DeviceRef device, StreamHandle stream);

/// `copy_bytes` ile ayni kurallar, fakat CUDA kopyasi `stream` uzerinde kuyruga girer.
/// Host tarafi sayfa kilitli degilse CUDA kopyayi kendi icinde senkronlar; sonuc yine dogrudur.
/// Host-host kopyasi hemen yapilir.
void copy_bytes_async(void* dst, MemorySpace dst_space, DeviceRef dst_device, const void* src,
                      MemorySpace src_space, DeviceRef src_device, std::size_t bytes, StreamHandle stream);

/// Akislar arasi siralama olayi. CUDA'da `cudaEvent_t` (zamanlamasiz), CPU'da nullptr'dir.
using EventHandle = void*;

/// Zamanlama kaydi tutmayan bir olay acar. CPU cihazi icin nullptr doner.
[[nodiscard]] EventHandle create_event(DeviceRef device);
void destroy_event(DeviceRef device, EventHandle event) noexcept;

/**
 * @brief `producer` akisina su ana kadar girmis isi `consumer` akisinin sonraki
 *        isinden once siralar. Host beklemez.
 *
 * Olay `producer` uzerine kaydedilir, `consumer` o kaydi bekler. Ayni olay
 * sirali cagrilarda tekrar kullanilabilir; bekleme cagri anindaki kaydi yakalar.
 * nullptr akis CUDA'nin eski (legacy) varsayilan akisidir. CPU'da no-op.
 */
void stream_wait(DeviceRef device, StreamHandle consumer, StreamHandle producer, EventHandle event);

/// Var olan sayfalanabilir host blogunu CUDA'ya sayfa kilitli olarak kaydeder
/// (`cudaHostRegister`). Boylece asenkron kopya gercekten ortusur. CUDA yoksa
/// veya kayit reddedilirse false doner; bellek yine kullanilabilir kalir.
[[nodiscard]] bool register_host_memory(void* ptr, std::size_t bytes) noexcept;
void unregister_host_memory(void* ptr) noexcept;

}  // namespace nexus
