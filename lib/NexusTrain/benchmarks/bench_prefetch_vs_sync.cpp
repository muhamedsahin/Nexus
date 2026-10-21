/**
 * @file bench_prefetch_vs_sync.cpp
 * @brief Yavaş Dataset::get ile senkron çekim ve async prefetch duvar sürelerini karşılaştırır.
 *
 * Üretici her örnekte uyur, tüketici her batch'te aynı süre kadar "hesap" uyur.
 * Örtüşme çalışıyorsa prefetch duvar süresi, iki sürenin toplamından belirgin kısadır.
 */

#include "nexus_train/prefetch/async_prefetcher.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/loading/dataloader.hpp"

#include <chrono>
#include <iostream>
#include <thread>

namespace {

class SlowDataset final : public nexusdata::Dataset {
 public:
  SlowDataset(std::size_t n, int delay_ms) : n_(n), delay_ms_(delay_ms) {}
  std::size_t size() const override { return n_; }
  nexusdata::Sample get(std::size_t index) const override {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
    nexusdata::Sample sample;
    sample.input = nexusdata::NDArray(nexusdata::Shape{1}, nexusdata::DType::Float32);
    sample.label = nexusdata::NDArray(nexusdata::Shape{1}, nexusdata::DType::Float32);
    sample.input.data<float>()[0] = static_cast<float>(index);
    sample.label.data<float>()[0] = 1.f;
    return sample;
  }

 private:
  std::size_t n_;
  int delay_ms_;
};

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main() {
  constexpr int kDelayMs = 5;
  constexpr int kBatches = 40;
  SlowDataset dataset(kBatches, kDelayMs);
  nexusdata::DataLoaderOptions options;
  options.batch_size = 1;
  options.shuffle = false;
  options.num_workers = 0;

  nexusdata::DataLoader sync_loader(dataset, options);
  const auto sync_start = std::chrono::steady_clock::now();
  int sync_count = 0;
  for (nexusdata::Batch batch : sync_loader) {
    (void)batch;
    std::this_thread::sleep_for(std::chrono::milliseconds(kDelayMs));
    ++sync_count;
  }
  const double sync_s = seconds_since(sync_start);

  nexusdata::DataLoader async_loader(dataset, options);
  const auto async_start = std::chrono::steady_clock::now();
  int async_count = 0;
  {
    nexus_train::AsyncPrefetcher prefetcher(async_loader, 2);
    while (auto batch = prefetcher.next()) {
      (void)batch;
      std::this_thread::sleep_for(std::chrono::milliseconds(kDelayMs));
      ++async_count;
    }
  }
  const double async_s = seconds_since(async_start);
  const double speedup = async_s > 0.0 ? sync_s / async_s : 0.0;
  std::cout << "batches " << sync_count << " / " << async_count << " sync_s " << sync_s << " prefetch_s " << async_s
            << " speedup " << speedup << "\n";
  return async_count == sync_count && speedup > 1.05 ? 0 : 1;
}
