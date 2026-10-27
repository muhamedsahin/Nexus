#pragma once

/**
 * @file device.hpp
 * @brief Tensörun yasadigi bellek alani ve cihaz etiketi.
 *
 * Bu baslik bir calistirma motoru degildir. Faz 2'deki zamanlayici burada
 * kurulan etiketin uzerine oturur. Faz 1'de etiket, bir isaretcinin CPU'da mi,
 * sayfa kilitli hostta mi, yoksa bir hizlandiricida mi durdugunu soyler.
 * Ayirma yalnizca CPU, sayfa kilitli host ve (toolkit bagliysa) CUDA icin
 * gercekten yapilir. Diger cihazlarin mevcut bir isaretcisi `Tensor::alias`
 * ile goruntulenir; olmayan bir ayirici cagrilirsa hata atilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/error.hpp"

#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>

namespace nexus {

/// Hizlandiricinin ailesi. Indeks `DeviceRef::index` icindedir.
enum class DeviceKind : std::uint8_t {
  Cpu = 0,
  Cuda,
  Rocm,
  Tpu,
  Npu,
  Vulkan,
  DirectMl,
  Metal
};

/// Isaretcinin fiziksel bellek turu. Cihaz ailesinden ayri tutulur:
/// CUDA uzerindeki bir tampon da, CPU uzerindeki siradan bir tampon da
/// ayni tensör arabirimine girer; kopya karari bu alana bakar.
enum class MemorySpace : std::uint8_t {
  Host = 0,        ///< 64 bayt hizali, sayfalanabilir CPU bellegi.
  PinnedHost = 1,  ///< Sayfa kilitli host. CUDA varsa cudaMallocHost, yoksa VirtualLock / mlock.
  Device = 2       ///< Hizlandirici bellegi (CUDA, ya da alias edilmis dis isaretci).
};

/// Cihaz kimligi. CPU icin index -1'dir. `cuda(0)` birinci NVIDIA aygitidir.
struct DeviceRef {
  DeviceKind kind = DeviceKind::Cpu;
  int index = -1;

  static constexpr DeviceRef cpu() noexcept { return {}; }
  static constexpr DeviceRef cuda(int device_index = 0) noexcept {
    return DeviceRef{DeviceKind::Cuda, device_index};
  }
  static constexpr DeviceRef rocm(int device_index = 0) noexcept {
    return DeviceRef{DeviceKind::Rocm, device_index};
  }
  static constexpr DeviceRef tpu(int device_index = 0) noexcept {
    return DeviceRef{DeviceKind::Tpu, device_index};
  }
  static constexpr DeviceRef npu(int device_index = 0) noexcept {
    return DeviceRef{DeviceKind::Npu, device_index};
  }
  static constexpr DeviceRef vulkan(int device_index = 0) noexcept {
    return DeviceRef{DeviceKind::Vulkan, device_index};
  }
  static constexpr DeviceRef directml(int device_index = 0) noexcept {
    return DeviceRef{DeviceKind::DirectMl, device_index};
  }
  static constexpr DeviceRef metal(int device_index = 0) noexcept {
    return DeviceRef{DeviceKind::Metal, device_index};
  }
  /// CPU ve hizlandiricinin ayni adimda kullanildigi politika etiketi. Bellek yine hosttur.
  static constexpr DeviceRef hybrid() noexcept { return DeviceRef{DeviceKind::Cpu, -2}; }

  [[nodiscard]] constexpr bool is_host() const noexcept { return kind == DeviceKind::Cpu; }
  [[nodiscard]] constexpr bool is_unified() const noexcept {
    return kind == DeviceKind::Cpu || kind == DeviceKind::Metal;
  }
  [[nodiscard]] constexpr bool is_accelerator() const noexcept { return kind != DeviceKind::Cpu; }
  [[nodiscard]] constexpr bool is_hybrid() const noexcept { return kind == DeviceKind::Cpu && index == -2; }

  /// "cuda:0", "rocm:0", "hip:0", "vulkan:0", "mps:0", "metal:0", "dml:0", "tpu:0", "pjrt:0", "cpu" cozumler.
  [[nodiscard]] static DeviceRef parse(std::string_view spec);

  friend constexpr bool operator==(DeviceRef lhs, DeviceRef rhs) noexcept {
    return lhs.kind == rhs.kind && lhs.index == rhs.index;
  }
  friend constexpr bool operator!=(DeviceRef lhs, DeviceRef rhs) noexcept { return !(lhs == rhs); }

  [[nodiscard]] std::string to_string() const;
};

[[nodiscard]] inline constexpr const char* to_string(DeviceKind kind) noexcept {
  switch (kind) {
    case DeviceKind::Cpu: return "cpu";
    case DeviceKind::Cuda: return "cuda";
    case DeviceKind::Rocm: return "rocm";
    case DeviceKind::Tpu: return "tpu";
    case DeviceKind::Npu: return "npu";
    case DeviceKind::Vulkan: return "vulkan";
    case DeviceKind::DirectMl: return "directml";
    case DeviceKind::Metal: return "metal";
  }
  return "unknown";
}

[[nodiscard]] inline constexpr const char* to_string(MemorySpace space) noexcept {
  switch (space) {
    case MemorySpace::Host: return "host";
    case MemorySpace::PinnedHost: return "pinned_host";
    case MemorySpace::Device: return "device";
  }
  return "unknown";
}

inline std::string DeviceRef::to_string() const {
  if (is_hybrid()) {
    return "hybrid";
  }
  std::string text(nexus::to_string(kind));
  if (index >= 0) {
    text.push_back(':');
    text += std::to_string(index);
  }
  return text;
}

inline DeviceRef DeviceRef::parse(std::string_view spec) {
  while (!spec.empty() && std::isspace(static_cast<unsigned char>(spec.front()))) spec.remove_prefix(1);
  while (!spec.empty() && std::isspace(static_cast<unsigned char>(spec.back()))) spec.remove_suffix(1);
  if (spec.empty()) {
    throw TensorError("Nexus: bos cihaz dizgesi cozumlenemez");
  }
  const std::size_t colon = spec.find(':');
  const std::string_view prefix = colon == std::string_view::npos ? spec : spec.substr(0, colon);
  std::string lower;
  lower.reserve(prefix.size());
  for (char ch : prefix) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  if (lower == "hybrid") {
    if (colon != std::string_view::npos) {
      throw TensorError("Nexus: hybrid cihazi indeks almaz");
    }
    return DeviceRef::hybrid();
  }
  int idx = 0;
  if (colon != std::string_view::npos) {
    const std::string_view idx_str = spec.substr(colon + 1);
    if (idx_str.empty()) {
      throw TensorError("Nexus: cihaz indeks kismi bos: " + std::string(spec));
    }
    int parsed = 0;
    for (char ch : idx_str) {
      if (!std::isdigit(static_cast<unsigned char>(ch))) {
        throw TensorError("Nexus: gecersiz cihaz indeksi: " + std::string(spec));
      }
      parsed = parsed * 10 + (ch - '0');
    }
    idx = parsed;
  }
  if (lower == "cpu" || lower == "host") {
    return DeviceRef::cpu();
  }
  if (lower == "cuda" || lower == "gpu") {
    return DeviceRef::cuda(idx);
  }
  if (lower == "rocm" || lower == "hip") {
    return DeviceRef::rocm(idx);
  }
  if (lower == "vulkan" || lower == "vk" || lower == "spirv") {
    return DeviceRef::vulkan(idx);
  }
  if (lower == "metal" || lower == "mps") {
    return DeviceRef::metal(idx);
  }
  if (lower == "directml" || lower == "dml" || lower == "dx12") {
    return DeviceRef::directml(idx);
  }
  if (lower == "tpu" || lower == "pjrt" || lower == "xla" || lower == "openxla") {
    return DeviceRef::tpu(idx);
  }
  if (lower == "npu") {
    return DeviceRef::npu(idx);
  }
  throw TensorError("Nexus: bilinmeyen cihaz turu: " + std::string(spec));
}

using Device = DeviceRef;

}  // namespace nexus
