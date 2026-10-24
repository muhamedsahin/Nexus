#pragma once

/**
 * @file timer.hpp
 * @brief Kapsamli zamanlayici ve aralik kaydi. Ice ice araliklar dogru toplanir.
 *
 * `ScopedTimer` kurulus saatinde baslar, yikilis saatinde durur; sureyi
 * `Profiler` kaydina ad bazinda yazar ve Chrome izine olay olarak dusurur.
 * Ice ice kurulus serbesttir: dis aralik kendi toplamini korur, ic araliklar
 * kendi adlariyla ayrica toplanir (PyTorch profiler'in self/total ayrimi:
 * burada her kayit toplam suredir; ic ice toplamlarin karsilastirilmasi
 * cagirana kalir, test bunu dogrular).
 *
 * Olcum `std::chrono::steady_clock` iledir. CUDA olay tabanli aygit
 * zamanlamasi FAZ 13'te grafik yakalamayla birlikte gelir; bu baslik
 * simdilik yalnizca host saatini kullanir ve uydurma aygit suresi uretmez.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace nexus {
namespace prof {

/// Tek bir olcum olayi; Chrome izine "X" (complete) olayi olarak yazilir.
struct IntervalRecord {
  std::string name;
  std::string category;
  std::uint64_t thread_id = 0;
  std::int64_t start_ns = 0;   // surec zamanlayicisinin baslangicina gore
  std::int64_t duration_ns = 0;
};

/// Ad basina toplam; benchmark tablosunun "toplam sure" sutunu buradan gelir.
struct IntervalTotal {
  std::uint64_t calls = 0;
  std::int64_t total_ns = 0;
  std::int64_t min_ns = 0;
  std::int64_t max_ns = 0;
};

/**
 * @brief Surec genelindeki tek aralik kaydi. Kapaliyken hicbir maliyet yoktur.
 */
class Profiler {
 public:
  Profiler() = default;
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  /// Kayit acik mi? Kapaliyken ScopedTimer yalnizca saat okur, kayit yazmaz.
  [[nodiscard]] bool enabled() const noexcept;
  void set_enabled(bool enabled) noexcept;

  /// Bir olcum olayini kayda alir. ScopedTimer bunu cagirir.
  void record(const IntervalRecord& interval);

  [[nodiscard]] std::vector<IntervalRecord> intervals() const;
  [[nodiscard]] std::map<std::string, IntervalTotal> totals() const;
  void reset();

  /// Profiler saatinin sifir noktasi (steady_clock epoch'undan ns).
  [[nodiscard]] static std::int64_t now_ns() noexcept;

 private:
  mutable std::mutex mutex_;
  bool enabled_ = false;
  std::vector<IntervalRecord> intervals_;
  std::map<std::string, IntervalTotal> totals_;
};

/// Surec genelindeki tek profiler.
[[nodiscard]] Profiler& profiler() noexcept;

/**
 * @brief RAII aralik zamanlayicisi.
 *
 * Profiler kapaliysa yalnizca iki saat okumasi yapar, kayit yazmaz. Aciksa
 * yikilis saatinde araligi kaydeder ve Chrome izine (trace.hpp) dusurur.
 */
class ScopedTimer {
 public:
  explicit ScopedTimer(std::string_view name, std::string_view category = "cpu_op") noexcept;
  ~ScopedTimer();
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

  /// Gecen sure (ns). Yikilimdan once de okunabilir.
  [[nodiscard]] std::int64_t elapsed_ns() const noexcept;

 private:
  std::string name_;
  std::string category_;
  std::chrono::steady_clock::time_point start_;
  bool record_ = false;
};

}  // namespace prof
}  // namespace nexus
