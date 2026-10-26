#pragma once

/**
 * @file serve.hpp
 * @brief FAZ 16.F: nexus::serve — surekli batch'leme (continuous batching) sunucusu (surec ici).
 *
 * Istekler kuyruga girer; arka plan is parcacigi her adimda:
 *   1. calisan isteklerin decode tokenlarini (istek basina 1) toplar,
 *   2. token butcesi (max_batch_tokens), sira yuvasi (max_running) ve bos KV blogu yettikce bekleyenlerin
 *      prefill'ini AYNI adima ekler (prefill + decode tek forward),
 *   3. ornekler, biten istegin (max_new ya da stop token) KV bloklarini geri verir.
 * Bloklar yetmezse en son kabul edilen calisan istek onceden bosaltilir (preempt: bloklari birakilir, istem +
 * uretilen tokenlar yeniden prefill edilerek devam eder — vLLM "recompute").
 * Ornekleme ofseti = istegin token indeksi: sonuc yigin bilesiminden bagimsiz tekrar uretilebilir.
 * Metrikler: istek basina TTFT, TPOT, gecikme; toplam token/s, esanli istek, adim basina token.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/infer.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace nexus {
namespace cuda {
namespace serve {

struct Request {
  std::vector<std::int32_t> prompt;
  int max_new = 64;
  infer::SamplingParams sampling{};
  std::int32_t stop_token = -1;
  std::function<void(std::int32_t)> on_token;  // akis (sunucu is parcacigindan cagrilir)
};

struct Result {
  std::vector<std::int32_t> tokens;
  double ttft_ms = 0.0;      // kuyruga giris -> ilk token
  double tpot_ms = 0.0;      // sonraki tokenlar ortalamasi
  double latency_ms = 0.0;   // kuyruga giris -> bitis
  int preemptions = 0;
  bool stopped = false;      // stop token
};

struct Config {
  int max_batch_tokens = 2048;
  int max_running = 64;
};

struct Metrics {
  std::int64_t steps = 0;
  std::int64_t generated = 0;
  std::int64_t prefill_tokens = 0;
  std::int64_t preemptions = 0;
  int max_running = 0;
  double busy_ms = 0.0;
  double mean_batch = 0.0;   // adim basina decode satiri
};

class Server {
 public:
  Server(infer::Engine& engine, Config config = {});
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  std::future<Result> submit(Request request);
  /// Kuyruk ve calisan istekler bitene kadar bekler.
  void drain();
  [[nodiscard]] Metrics metrics() const;

 private:
  struct Live;
  void loop();
  bool step();

  infer::Engine* engine_;
  Config cfg_;
  mutable std::mutex mu_;
  std::condition_variable cv_, idle_;
  std::deque<std::shared_ptr<Live>> waiting_;
  std::vector<std::shared_ptr<Live>> running_;
  bool stop_ = false;
  Metrics metrics_;
  std::thread thread_;
};

}  // namespace serve
}  // namespace cuda
}  // namespace nexus
