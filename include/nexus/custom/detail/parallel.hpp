#pragma once

/**
 * @file parallel.hpp
 * @brief Kisa aralik bolme. Esik altinda tek cagri yapar.
 *
 * Esik asilinca surec omrunde en fazla 7 isci bekler. Cagri basina thread
 * acilmaz. WorkStealingPool kullanilmaz: o havuzun tek isi vardir ve ozel
 * aktivasyon oraya baglaninca kilitlenir. Isci icinden gelen cagri satir ici
 * calisir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace nexus {
namespace detail {
namespace {

thread_local bool t_inside_range_pool = false;

class RangePool {
 public:
    static RangePool& instance() {
        static RangePool pool;
        return pool;
    }

    void run(std::size_t n, const std::function<void(std::size_t, std::size_t)>& body) {
        constexpr unsigned kWorkers = 7;
        const unsigned threads = n < kWorkers ? static_cast<unsigned>(n) : kWorkers;
        const std::size_t chunk = (n + threads - 1) / threads;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            body_ = body;
            count_ = n;
            chunk_ = chunk;
            next_.store(0, std::memory_order_relaxed);
            done_.store(0, std::memory_order_relaxed);
            error_ = nullptr;
            generation_.fetch_add(1, std::memory_order_acq_rel);
        }
        wake_.notify_all();
        execute_local(0, n);
        while (done_.load(std::memory_order_acquire) < kWorkers + 1) {
            std::this_thread::yield();
        }
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

 private:
    RangePool() {
        for (int i = 0; i < 7; ++i) {
            threads_.emplace_back([this] { loop(); });
        }
    }

    ~RangePool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            generation_.fetch_add(1, std::memory_order_acq_rel);
        }
        wake_.notify_all();
        for (std::thread& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void execute_local(std::size_t, std::size_t) {
        for (;;) {
            const std::size_t begin = next_.fetch_add(chunk_, std::memory_order_relaxed);
            if (begin >= count_) {
                break;
            }
            const std::size_t end = begin + chunk_ < count_ ? begin + chunk_ : count_;
            try {
                body_(begin, end);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) {
                    error_ = std::current_exception();
                }
            }
        }
        done_.fetch_add(1, std::memory_order_acq_rel);
    }

    void loop() {
        t_inside_range_pool = true;
        std::uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || generation_.load(std::memory_order_acquire) != seen; });
                if (stop_) {
                    return;
                }
                seen = generation_.load(std::memory_order_acquire);
            }
            execute_local(0, 0);
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::function<void(std::size_t, std::size_t)> body_;
    std::size_t count_ = 0;
    std::size_t chunk_ = 1;
    std::atomic<std::size_t> next_{0};
    std::atomic<unsigned> done_{0};
    std::atomic<std::uint64_t> generation_{0};
    std::exception_ptr error_;
    bool stop_ = false;
    std::vector<std::thread> threads_;
};

}  // namespace

template <typename Fn>
void parallel_for_range(std::size_t n, std::size_t threshold, Fn&& body) {
    if (n == 0) {
        return;
    }
    if (n < threshold || t_inside_range_pool) {
        body(std::size_t{0}, n);
        return;
    }
    t_inside_range_pool = true;
    try {
        RangePool::instance().run(n, std::function<void(std::size_t, std::size_t)>(std::forward<Fn>(body)));
    } catch (...) {
        t_inside_range_pool = false;
        throw;
    }
    t_inside_range_pool = false;
}

}  // namespace detail
}  // namespace nexus
