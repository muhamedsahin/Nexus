#pragma once

/**
 * @file planner.hpp
 * @brief FAZ 7.4: yasam suresi tabanli statik ofset planlayici (tek arena).
 *
 * Girdi: her tensör icin [begin, end) adim araligi ve bayt. Cikti: arena icinde ofsetler;
 * yasam suresi cakisan iki tensör bellekte ortusmez. Algoritma "boyuta gore acgozlu en iyi
 * uyum" (Pisarchyk & Lee 2020, TFLite): tensörler buyukten kucuge yerlestirilir, her biri
 * zamanda cakistigi yerlesmislerin arasindaki en kucuk yeterli bosluga (yoksa sona) konur.
 * Alt sinir tepe canli bayttir; `Plan::arena_bytes / peak_live_bytes - 1` parcalanmadir.
 *
 * Kayit tabanli kullanim: `LifetimeRecorder` bir adimin ayir/birak olaylarini mantiksal
 * saatle toplar; sonraki adimlar plana gore tek arenadan calisir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nexus {
namespace memory {

struct Lifetime {
  int begin = 0;  // ilk kullanim adimi (dahil)
  int end = 0;    // son kullanimdan sonraki adim (haric)
  std::uint64_t bytes = 0;
};

struct Plan {
  std::vector<std::uint64_t> offsets;  // girdiyle ayni sira
  std::uint64_t arena_bytes = 0;
  std::uint64_t peak_live_bytes = 0;
};

[[nodiscard]] Plan plan_offsets(const std::vector<Lifetime>& items, std::uint64_t alignment = 256);

/// Ayir/birak olaylarindan yasam sureleri (her olay saati bir artirir).
class LifetimeRecorder {
 public:
  void on_alloc(std::uint64_t id, std::uint64_t bytes);
  void on_free(std::uint64_t id);
  /// Birakilmamislar adimin sonuna kadar canli sayilir.
  [[nodiscard]] std::vector<Lifetime> lifetimes() const;

 private:
  int clock_ = 0;
  std::vector<Lifetime> items_;
  std::unordered_map<std::uint64_t, std::size_t> open_;
};

}  // namespace memory
}  // namespace nexus
