#pragma once

/**
 * @file async_prefetcher.hpp
 * @brief DataLoader'ı arka planda bir thread ile besler.
 *
 * Üretici yalnızca producer thread'inde `DataLoader::begin/end` çağırır
 * (DataLoader çoklu tüketiciye güvenli değildir). Tüketici `next()` ile hazır
 * batch alır. Kuyruk `queue_depth` ile sınırlıdır: doluyken üretici, boşken
 * tüketici `condition_variable` üzerinde uyur (busy-wait yok).
 *
 * Kritik bölüm yalnızca batch başlığını (NDArray paylaşımlı işaretçileri) taşır.
 * Tensör baytları kilit altında kopyalanmaz. Asıl örtüşme, üretici `Dataset::get`
 * içindeyken tüketicinin forward/backward çalışmasıdır.
 *
 * Thread-safety:
 * - `next()` tek tüketici thread'den (eğitim thread'i).
 * - Yıkıcı, üretici kuyruk doluyken blokluysa `stop` ile uyandırır ve join eder.
 *   Yıkıcıyı `next()` içinden çağırma.
 * - deterministic_mode Trainer'da bu sınıfı hiç kurmaz.
 */

#include "nexusdata/loading/dataloader.hpp"

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

namespace nexus_train {

class AsyncPrefetcher {
 public:
  explicit AsyncPrefetcher(nexusdata::DataLoader& loader, std::size_t queue_depth = 2);
  ~AsyncPrefetcher();

  AsyncPrefetcher(const AsyncPrefetcher&) = delete;
  AsyncPrefetcher& operator=(const AsyncPrefetcher&) = delete;

  [[nodiscard]] std::optional<nexusdata::Batch> next();

 private:
  void producer_loop();

  nexusdata::DataLoader& loader_;
  std::size_t queue_depth_;
  std::queue<nexusdata::Batch> queue_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  bool stop_ = false;
  bool done_ = false;
  std::exception_ptr error_;
  std::thread worker_;
};

}  // namespace nexus_train
