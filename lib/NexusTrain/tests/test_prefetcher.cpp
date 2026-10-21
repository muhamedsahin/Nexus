#include "check.hpp"

#include "nexus_train/prefetch/async_prefetcher.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/loading/dataloader.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

class IndexedDataset final : public nexusdata::Dataset {
 public:
  IndexedDataset(std::size_t n, int delay_ms) : n_(n), delay_ms_(delay_ms) {}
  std::size_t size() const override { return n_; }
  nexusdata::Sample get(std::size_t index) const override {
    if (delay_ms_ > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
    }
    nexusdata::Sample sample;
    sample.input = nexusdata::NDArray(nexusdata::Shape{1}, nexusdata::DType::Float32);
    sample.label = nexusdata::NDArray(nexusdata::Shape{1}, nexusdata::DType::Float32);
    sample.input.data<float>()[0] = static_cast<float>(index);
    sample.label.data<float>()[0] = static_cast<float>(index);
    return sample;
  }

 private:
  std::size_t n_;
  int delay_ms_;
};

std::vector<float> pull_sync(nexusdata::DataLoader& loader) {
  std::vector<float> values;
  for (nexusdata::Batch batch : loader) {
    values.push_back(batch.labels.data<float>()[0]);
  }
  return values;
}

std::vector<float> pull_prefetch(nexusdata::DataLoader& loader, std::size_t depth) {
  nexus_train::AsyncPrefetcher prefetcher(loader, depth);
  std::vector<float> values;
  while (auto batch = prefetcher.next()) {
    values.push_back(batch->labels.data<float>()[0]);
  }
  return values;
}

}  // namespace

int test_prefetcher() {
  int failed = 0;
  {
    IndexedDataset dataset(12, 0);
    nexusdata::DataLoaderOptions options;
    options.batch_size = 1;
    options.shuffle = false;
    options.num_workers = 0;
    nexusdata::DataLoader sync_loader(dataset, options);
    nexusdata::DataLoader async_loader(dataset, options);
    const auto expected = pull_sync(sync_loader);
    const auto got = pull_prefetch(async_loader, 2);
    CHECK(got.size() == expected.size());
    for (std::size_t i = 0; i < got.size() && i < expected.size(); ++i) CHECK(got[i] == expected[i]);
  }
  {
    IndexedDataset dataset(6, 0);
    nexusdata::DataLoaderOptions options;
    options.batch_size = 1;
    options.shuffle = false;
    options.num_workers = 0;
    nexusdata::DataLoader loader(dataset, options);
    const auto got = pull_prefetch(loader, 1);
    CHECK(got.size() == 6);
    CHECK(got.front() == 0.f);
    CHECK(got.back() == 5.f);
  }
  {
    class EmptyDataset final : public nexusdata::Dataset {
     public:
      std::size_t size() const override { return 0; }
      nexusdata::Sample get(std::size_t) const override { throw std::runtime_error("empty"); }
    } dataset;
    nexusdata::DataLoaderOptions options;
    options.batch_size = 1;
    options.num_workers = 0;
    nexusdata::DataLoader loader(dataset, options);
    nexus_train::AsyncPrefetcher prefetcher(loader, 2);
    CHECK(!prefetcher.next().has_value());
  }
  {
    std::atomic<bool> finished{false};
    std::thread watchdog([&] {
      for (int i = 0; i < 50 && !finished.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (!finished.load()) {
        std::cerr << "prefetcher destructor deadlocked\n";
        std::abort();
      }
    });
    IndexedDataset dataset(8, 20);
    nexusdata::DataLoaderOptions options;
    options.batch_size = 1;
    options.shuffle = false;
    options.num_workers = 0;
    nexusdata::DataLoader loader(dataset, options);
    {
      nexus_train::AsyncPrefetcher prefetcher(loader, 2);
      (void)prefetcher.next();
    }
    finished.store(true);
    watchdog.join();
  }
  return failed;
}
