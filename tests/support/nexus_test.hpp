#pragma once

/**
 * @file nexus_test.hpp
 * @brief Nexus kok testleri icin ortak denetim, heap ayirma sayaci ve sayisal turev.
 *
 * `AllocationScope` kapsam boyunca global `operator new` cagrilarini sayar;
 * "sicak yolda sifir ayirma" iddiasi bu sayacla dogrulanir. Sayac tum
 * is parcaciklarini sayar, bu yuzden olcum sirasinda arka plan isi
 * calistirilmamalidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace nexus_test {

/// Kosulu yazdirir; basarisizsa hata sayacini artirir.
void expect(bool condition, std::string_view name);
/// Bilgi satiri: `  info  <key>=<value>`.
void info(std::string_view key, double value);
void section(std::string_view title);
[[nodiscard]] int failures() noexcept;
/// Ozet satirini yazar ve surec cikis kodunu dondurur.
[[nodiscard]] int finish(std::string_view suite);

/// Kapsam icindeki global operator new cagrilarini sayar. Ic ice kullanilabilir.
class AllocationScope {
 public:
  AllocationScope() noexcept;
  ~AllocationScope();
  AllocationScope(const AllocationScope&) = delete;
  AllocationScope& operator=(const AllocationScope&) = delete;
  [[nodiscard]] std::uint64_t count() const noexcept;

 private:
  std::uint64_t begin_;
};

template <typename T>
[[nodiscard]] bool finite(std::span<const T> values) noexcept {
  for (const T value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

/// Bit duzeyinde esitlik; backward ile backward_into ayni yolu kosmalidir.
template <typename T>
[[nodiscard]] bool identical(std::span<const T> lhs, std::span<const T> rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

/// max_i |a_i - b_i| / (atol + rtol * |b_i|). Sonuc <= 1 ise tolerans icindedir.
[[nodiscard]] double tolerance_ratio(std::span<const double> actual, std::span<const double> expected, double atol,
                                     double rtol) noexcept;

/**
 * @brief Merkezi farkla sayisal turev: (f(x + h e_i) - f(x - h e_i)) / 2h.
 * @param loss `double(std::span<const double>)` imzali cagrilabilir; x'i degistirmez.
 */
template <typename Loss>
[[nodiscard]] std::vector<double> numeric_gradient(std::span<const double> x, Loss&& loss, double step = 1.0e-6) {
  std::vector<double> point(x.begin(), x.end());
  std::vector<double> gradient(x.size());
  for (std::size_t i = 0; i < point.size(); ++i) {
    const double original = point[i];
    point[i] = original + step;
    const double plus = loss(std::span<const double>(point));
    point[i] = original - step;
    const double minus = loss(std::span<const double>(point));
    point[i] = original;
    gradient[i] = (plus - minus) / (2.0 * step);
  }
  return gradient;
}

}  // namespace nexus_test
