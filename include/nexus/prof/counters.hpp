#pragma once

/**
 * @file counters.hpp
 * @brief FLOP, bayt ve cagri sayaclari. Is parcacigi guvenli, ad bazinda toplama.
 *
 * Cekirdekler ve katmanlar islerini buraya bildirir; benchmark JSON'u ve
 * MFU/MBU hesapleri bu toplamlardan turer. Sayac kaydi yoksa kilit altinda
 * olusturulur; sicak yolda yalnizca kilit + toplama maliyeti vardir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace nexus {
namespace prof {

/// Bir adin uc sayaci.
struct CounterValues {
  std::int64_t flops = 0;
  std::int64_t bytes = 0;
  std::int64_t calls = 0;
};

/**
 * @brief Surec genelindeki tek sayac kaydi.
 *
 * `add_*` cagrilari mutex altinda yazar; FAZ 0 olceklerinde bu yeterli,
 * FAZ 13'te is parcacigi yerel biriktirme + periyodik birlestirme eklenecek.
 */
class CounterRegistry {
 public:
  CounterRegistry() = default;
  CounterRegistry(const CounterRegistry&) = delete;
  CounterRegistry& operator=(const CounterRegistry&) = delete;

  void add_flops(const std::string& name, std::int64_t count);
  void add_bytes(const std::string& name, std::int64_t count);
  void add_calls(const std::string& name, std::int64_t count = 1);

  [[nodiscard]] std::map<std::string, CounterValues> snapshot() const;
  [[nodiscard]] std::int64_t total_flops() const;
  [[nodiscard]] std::int64_t total_bytes() const;
  void reset();

 private:
  mutable std::mutex mutex_;
  std::map<std::string, CounterValues> counters_;
};

/// Surec genelindeki tek kayit.
[[nodiscard]] CounterRegistry& counters() noexcept;

}  // namespace prof
}  // namespace nexus
