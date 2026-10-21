/**
 * @file async_prefetcher.cpp
 * @brief Sınırlı SPSC kuyruk. Yayın ve bekleme aynı mutex altındadır;
 * aksi halde predicate ile wait arasındaki kaçan uyanma kuyruğu kilitler.
 */

#include "nexus_train/prefetch/async_prefetcher.hpp"

#include <stdexcept>
#include <utility>

namespace nexus_train {

AsyncPrefetcher::AsyncPrefetcher(nexusdata::DataLoader& loader, std::size_t queue_depth)
    : loader_(loader), queue_depth_(queue_depth) {
  if (queue_depth_ == 0) {
    throw std::invalid_argument("NexusTrain: prefetch queue depth must be >= 1");
  }
  worker_ = std::thread([this] { producer_loop(); });
}

AsyncPrefetcher::~AsyncPrefetcher() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  not_full_.notify_all();
  not_empty_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

std::optional<nexusdata::Batch> AsyncPrefetcher::next() {
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait(lock, [&] { return stop_ || done_ || !queue_.empty(); });
  if (!queue_.empty()) {
    nexusdata::Batch batch = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return batch;
  }
  if (error_) {
    const std::exception_ptr error = error_;
    error_ = nullptr;
    std::rethrow_exception(error);
  }
  return std::nullopt;
}

void AsyncPrefetcher::producer_loop() {
  try {
    for (auto it = loader_.begin(); it != loader_.end(); ++it) {
      nexusdata::Batch batch = *it;
      std::unique_lock<std::mutex> lock(mutex_);
      not_full_.wait(lock, [&] { return stop_ || queue_.size() < queue_depth_; });
      if (stop_) {
        break;
      }
      queue_.push(std::move(batch));
      lock.unlock();
      not_empty_.notify_one();
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = std::current_exception();
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    done_ = true;
  }
  not_empty_.notify_all();
}

}  // namespace nexus_train
