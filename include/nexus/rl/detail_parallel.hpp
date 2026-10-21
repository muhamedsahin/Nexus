#pragma once

/**
 * @file detail_parallel.hpp
 * @brief Vektor ortamlar icin kalici indeks havuzu.
 *
 * Sekizden az kopya cagiran thread'de kalir. Sekiz ve uzerinde is, surec
 * boyunca yasayan iscilere dagitilir; her adim yeni thread acmaz. Cagiran
 * da bir dilim alir. Ilk istisna, tum indeksler bitince yeniden firlatilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace nexus {
namespace rl {
namespace detail {

inline void parallel_indexed_serial(std::size_t n, const auto& body) {
    for (std::size_t i = 0; i < n; ++i) {
        body(i);
    }
}

struct IndexTask {
    void (*call)(std::size_t, void*) = nullptr;
    void* user = nullptr;
    std::size_t count = 0;
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
};

inline void run_index_task(IndexTask* task) {
    for (;;) {
        const std::size_t index = task->next.fetch_add(1, std::memory_order_relaxed);
        if (index >= task->count) {
            return;
        }
        if (!task->failed.load(std::memory_order_relaxed)) {
            try {
                task->call(index, task->user);
            } catch (...) {
                std::lock_guard<std::mutex> lock(task->error_mutex);
                if (!task->failed.exchange(true, std::memory_order_acq_rel)) {
                    task->error = std::current_exception();
                }
            }
        }
        task->done.fetch_add(1, std::memory_order_release);
    }
}

class IndexPool {
 public:
    static IndexPool& instance() {
        static IndexPool pool;
        return pool;
    }

    template <typename Fn>
    void run(std::size_t n, Fn&& body) {
        thread_local int depth = 0;
        if (depth > 0 || current_.load(std::memory_order_seq_cst) != nullptr) {
            parallel_indexed_serial(n, body);
            return;
        }
        ensure_started();
        IndexTask task;
        task.call = &invoke<std::decay_t<Fn>>;
        std::decay_t<Fn> fn(std::forward<Fn>(body));
        task.user = &fn;
        task.count = n;
        IndexTask* vacant = nullptr;
        if (!current_.compare_exchange_strong(vacant, &task, std::memory_order_seq_cst)) {
            parallel_indexed_serial(n, body);
            return;
        }
        struct DepthGuard {
            int& depth;
            explicit DepthGuard(int& value) : depth(value) { ++depth; }
            ~DepthGuard() { --depth; }
        } guard(depth);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            generation_.fetch_add(1, std::memory_order_acq_rel);
        }
        wake_.notify_all();
        run_index_task(&task);
        while (task.done.load(std::memory_order_acquire) < n) {
            std::this_thread::yield();
        }
        current_.store(nullptr, std::memory_order_seq_cst);
        while (busy_.load(std::memory_order_seq_cst) != 0) {
            std::this_thread::yield();
        }
        if (task.error) {
            std::rethrow_exception(task.error);
        }
    }

 private:
    IndexPool() = default;

    ~IndexPool() {
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

    IndexPool(const IndexPool&) = delete;
    IndexPool& operator=(const IndexPool&) = delete;

    template <typename Fn>
    static void invoke(std::size_t index, void* user) {
        (*static_cast<Fn*>(user))(index);
    }

    void ensure_started() {
        if (started_.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_.load(std::memory_order_relaxed)) {
            return;
        }
        unsigned hardware = std::thread::hardware_concurrency();
        if (hardware < 2) {
            hardware = 2;
        }
        if (hardware > 8) {
            hardware = 8;
        }
        worker_count_ = hardware - 1;
        started_.store(true, std::memory_order_release);
        for (unsigned i = 0; i < worker_count_; ++i) {
            threads_.emplace_back([this] { loop(); });
        }
    }

    void loop() {
        std::uint64_t seen = generation_.load(std::memory_order_acquire);
        for (;;) {
            std::uint64_t now = generation_.load(std::memory_order_acquire);
            if (now == seen) {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
                continue;
            }
            seen = now;
            if (stop_) {
                return;
            }
            busy_.fetch_add(1, std::memory_order_seq_cst);
            IndexTask* task = current_.load(std::memory_order_seq_cst);
            if (task != nullptr) {
                run_index_task(task);
            }
            busy_.fetch_sub(1, std::memory_order_seq_cst);
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<std::thread> threads_;
    std::atomic<IndexTask*> current_{nullptr};
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<int> busy_{0};
    bool stop_ = false;
    std::atomic<bool> started_{false};
    unsigned worker_count_ = 0;
};

template <typename Fn>
void parallel_indexed(std::size_t n, Fn&& body) {
    if (n == 0) {
        return;
    }
    if (n < 8) {
        parallel_indexed_serial(n, body);
        return;
    }
    IndexPool::instance().run(n, std::forward<Fn>(body));
}

}  // namespace detail
}  // namespace rl
}  // namespace nexus
