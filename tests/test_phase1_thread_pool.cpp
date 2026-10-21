/**
 * @file test_phase1_thread_pool.cpp
 * @brief MatrixFlash-Pro paralel dongu havuzunun dogruluk ve omur testi.
 *
 * Eski Chase-Lev uygulamasi cagiran thread'den calisan kuyruklarina itiyordu
 * (yalniz sahibin yapabilecegi is). Gorev kaybi kilitlenme, cift kosum ise
 * `parallel_for`'un erken donmesi ve ASan'in yakaladigi use-after-free
 * demekti. Bu test her parcanin tam bir kez kostugunu ve cagri donunce hicbir
 * calisanin cagiranin bellegine yazmadigini dogrular: her denemenin tamponu
 * hemen serbest birakilir, gec yazma ASan derlemesinde rapor edilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/nexus.hpp"
#include "nexus_test.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(NEXUS_HAS_MATRIX_PRO)
#include "matrix_pro/core/thread_pool.hpp"
#include "matrix_pro/cpu/gemm.hpp"
#endif

namespace {

using nexus_test::expect;

#if defined(NEXUS_HAS_MATRIX_PRO)

struct Coverage {
  std::atomic<int>* hits = nullptr;
  int throw_at = -1;
};

void mark(int lo, int hi, void* user) {
  auto* coverage = static_cast<Coverage*>(user);
  if (coverage->throw_at >= lo && coverage->throw_at < hi) {
    throw std::runtime_error("parca hatasi");
  }
  for (int i = lo; i < hi; ++i) {
    coverage->hits[i].fetch_add(1, std::memory_order_relaxed);
  }
}

/// Her indeks tam bir kez isaretlendiyse true. Tampon donuste serbest kalir.
bool covered_once(int count, int grain) {
  std::unique_ptr<std::atomic<int>[]> hits(new std::atomic<int>[static_cast<std::size_t>(count)]);
  for (int i = 0; i < count; ++i) {
    hits[static_cast<std::size_t>(i)].store(0, std::memory_order_relaxed);
  }
  Coverage coverage{hits.get(), -1};
  matrix_pro::WorkStealingPool::instance().parallel_for(0, count, grain, &mark, &coverage);
  for (int i = 0; i < count; ++i) {
    if (hits[static_cast<std::size_t>(i)].load(std::memory_order_relaxed) != 1) {
      return false;
    }
  }
  return true;
}

int run_trials(int trials, int seed) {
  int failures = 0;
  for (int trial = 0; trial < trials; ++trial) {
    const int count = 1 + (trial * 7919 + seed * 104729) % 5000;
    const int grain = 1 + (trial + seed) % 37;
    if (!covered_once(count, grain)) {
      ++failures;
    }
  }
  return failures;
}

struct Nested {
  std::atomic<int>* total = nullptr;
};

void inner(int lo, int hi, void* user) {
  static_cast<Nested*>(user)->total->fetch_add(hi - lo, std::memory_order_relaxed);
}

void outer(int lo, int hi, void* user) {
  for (int i = lo; i < hi; ++i) {
    matrix_pro::WorkStealingPool::instance().parallel_for(0, 100, 7, &inner, user);
  }
}

void test_pool() {
  nexus_test::section("MatrixFlash paralel dongu havuzu");
  auto& pool = matrix_pro::WorkStealingPool::instance();
  nexus_test::info("katilimci thread", pool.worker_count());
  expect(pool.worker_count() >= 1, "en az bir katilimci");

  expect(run_trials(1500, 0) == 0, "1500 rastgele boyutta her parca tam bir kez kosar");

  std::unique_ptr<std::atomic<int>[]> hits(new std::atomic<int>[1000]);
  for (int i = 0; i < 1000; ++i) {
    hits[static_cast<std::size_t>(i)].store(0);
  }
  Coverage failing{hits.get(), 517};
  bool threw = false;
  try {
    pool.parallel_for(0, 1000, 10, &mark, &failing);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  expect(threw, "parca istisnasi cagirana firlatilir");
  expect(run_trials(50, 1) == 0, "istisnadan sonra havuz kullanilabilir");

  std::atomic<int> total{0};
  Nested nested{&total};
  pool.parallel_for(0, 64, 4, &outer, &nested);
  expect(total.load() == 64 * 100, "ic ice parallel_for satir ici kosar, kilitlenmez");

  std::atomic<int> concurrent_failures{0};
  std::vector<std::thread> callers;
  for (int caller = 0; caller < 4; ++caller) {
    callers.emplace_back([&concurrent_failures, caller] {
      concurrent_failures.fetch_add(run_trials(200, caller + 2), std::memory_order_relaxed);
    });
  }
  for (std::thread& caller : callers) {
    caller.join();
  }
  expect(concurrent_failures.load() == 0, "4 thread ayni anda cagirdiginda sonuc dogru");
}

void test_gemm_lifetime() {
  nexus_test::section("cok thread'li GEMM ve tampon omru");
  int mismatches = 0;
  double worst = 0.0;
  for (int trial = 0; trial < 60; ++trial) {
    const int m = 64 + (trial * 37) % 240;
    const int n = 16 + (trial * 53) % 200;
    const int k = 8 + (trial * 29) % 150;
    std::vector<float> a(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
    std::vector<float> b(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < a.size(); ++i) {
      a[i] = static_cast<float>(std::sin(0.37 * static_cast<double>(i + static_cast<std::size_t>(trial))));
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = static_cast<float>(std::cos(0.23 * static_cast<double>(i)));
    }
    auto fast = std::make_unique<float[]>(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    std::vector<float> reference(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    matrix_pro::gemm_f32(m, n, k, a.data(), k, b.data(), n, fast.get(), n, 1.0f, 0.0f);
    matrix_pro::gemm_reference(m, n, k, a.data(), k, b.data(), n, reference.data(), n, 1.0f, 0.0f);
    for (std::size_t i = 0; i < reference.size(); ++i) {
      const double error = std::fabs(static_cast<double>(fast[i]) - static_cast<double>(reference[i]));
      worst = error > worst ? error : worst;
      if (error > 1.0e-3 * static_cast<double>(k)) {
        ++mismatches;
      }
    }
  }
  nexus_test::info("en buyuk |hizli - referans|", worst);
  expect(mismatches == 0, "60 GEMM referansla ayni; cikis tamponu donuste serbest");
}

#endif

}  // namespace

int main() {
  std::cout << "Nexus Faz 1 is parcacigi havuzu testi\n";
  try {
#if defined(NEXUS_HAS_MATRIX_PRO)
    test_pool();
    test_gemm_lifetime();
#else
    nexus_test::section("MatrixFlash-Pro bagli degil, havuz testi atlandi");
    expect(true, "MatrixFlash olmadan derleme gecerli");
#endif
  } catch (const std::exception& error) {
    std::cout << "  FAIL  istisna: " << error.what() << "\n";
    return 1;
  }
  return nexus_test::finish("test_phase1_thread_pool");
}
