#pragma once

/**
 * @file memory_ledger.hpp
 * @brief Etiketli bellek muhasebe defteri: anlik ve doruk bayt, etiket bazinda kirilim.
 *
 * Her ayirma bir etiket tasir: parameter, gradient, optimizer_state, activation,
 * temporary, comm_buffer, kv_cache. Etiket, ayirmayi yapan is parcaciginda
 * `AllocationLabelScope` ile secilir; kapsamsiz ayirmalar `Temporary` sayilir.
 *
 * Defter yalnizca `nexus/core/memory.hpp` kapisindan gecen bloklari sayar
 * (aligned host, pinned host, CUDA device). Alt kutuphanelerin kendi
 * ayiricilari bu kapiyi kullanmadigi surece deftere girmez; FAZ 1'deki
 * CachingAllocator ve k64 tek ayirici bu muhasebeyi devralir.
 *
 * Parcalanma orani bu defterin isi degildir: parcalanma, havuzlu ayiricinin
 * (FAZ 1) reserved/current oranindan olculur. Bu defter yalnizca gercek
 * ayirma-serbest birakma akisini kaydeder; uydurma metrik uretmez.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace nexus {
namespace prof {

/// Bellek kullaniminin amac etiketi. FAZ 0 planindaki yedi sinif.
enum class MemoryLabel : std::uint8_t {
  Parameter = 0,
  Gradient = 1,
  OptimizerState = 2,
  Activation = 3,
  Temporary = 4,
  CommBuffer = 5,
  KvCache = 6,
  kCount = 7,
};

/// Etiket adi; JSON ve tablo ciktilarinda kullanilir.
[[nodiscard]] const char* label_name(MemoryLabel label) noexcept;

/// Ayirmanin bulundugu adres uzayi. memory.hpp'deki AllocKind ile birebir eslenir.
enum class LedgerSpace : std::uint8_t { AlignedHost = 1, PinnedHost = 2, Device = 3 };

/// Bir etiketin ve adres uzayinin anlik/doruk toplami.
struct LedgerEntry {
  std::int64_t current_bytes = 0;
  std::int64_t peak_bytes = 0;
  std::uint64_t live_blocks = 0;
  std::uint64_t total_allocs = 0;
};

/// Defterin tam kopyasi; benchmark JSON'una ve test dogrulamalarina gider.
struct LedgerSnapshot {
  std::array<LedgerEntry, static_cast<std::size_t>(MemoryLabel::kCount)> by_label{};
  std::array<LedgerEntry, 3> by_space{};  // [AlignedHost, PinnedHost, Device]
  std::int64_t current_bytes = 0;
  std::int64_t peak_bytes = 0;
  std::uint64_t live_blocks = 0;
  std::uint64_t total_allocs = 0;
};

/**
 * @brief Surec genelinde tek bellek muhasebe defteri.
 *
 * Tum islemler kilit altindadir; cok is parcacikli egitimde dogru toplar.
 * `record_free` eslesmeyen serbest birakmada atilir (cagiran zaten yalnizca
 * kayitli bloklari serbest birakir).
 */
class MemoryLedger {
 public:
  MemoryLedger() = default;
  MemoryLedger(const MemoryLedger&) = delete;
  MemoryLedger& operator=(const MemoryLedger&) = delete;

  void record_alloc(MemoryLabel label, LedgerSpace space, std::size_t bytes) noexcept;
  void record_free(MemoryLabel label, LedgerSpace space, std::size_t bytes) noexcept;

  [[nodiscard]] LedgerSnapshot snapshot() const noexcept;
  [[nodiscard]] std::int64_t current_bytes() const noexcept;
  [[nodiscard]] std::int64_t peak_bytes() const noexcept;

  /// Doruklari sifirlar; anlik degerler korunur. Benchmark arasinda cagrilir.
  void reset_peak() noexcept;
  /// Test icin tam sifirlama.
  void reset_all() noexcept;

 private:
  mutable std::atomic_flag flag_{};  // C++20: constexpr deger baslatma
  std::array<LedgerEntry, static_cast<std::size_t>(MemoryLabel::kCount)> by_label_{};
  std::array<LedgerEntry, 3> by_space_{};
  std::int64_t current_bytes_ = 0;
  std::int64_t peak_bytes_ = 0;
  std::uint64_t live_blocks_ = 0;
  std::uint64_t total_allocs_ = 0;
};

/// Surec genelindeki tek defter.
[[nodiscard]] MemoryLedger& ledger() noexcept;

/**
 * @brief Cagri yapan is parcaciginin su anki ayirma etiketi.
 *
 * Varsayilan `Temporary`. Kapsam nesnesi kurulusunda etiketi degistirir,
 * yikilisinda eski etiketi geri koyar; ic ice kurulabilir.
 */
[[nodiscard]] MemoryLabel current_allocation_label() noexcept;

/// Kapsam boyunca `nexus::allocate_*` cagrilari bu etiketle deftere yazilir.
class AllocationLabelScope {
 public:
  explicit AllocationLabelScope(MemoryLabel label) noexcept;
  ~AllocationLabelScope() noexcept;
  AllocationLabelScope(const AllocationLabelScope&) = delete;
  AllocationLabelScope& operator=(const AllocationLabelScope&) = delete;

 private:
  MemoryLabel previous_;
};

}  // namespace prof
}  // namespace nexus
