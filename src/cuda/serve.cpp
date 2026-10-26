/**
 * @file serve.cpp
 * @brief FAZ 16.F: surekli batch'leme sunucusu (bkz. nexus/cuda/serve.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/serve.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace nexus {
namespace cuda {
namespace serve {

namespace {
using Clock = std::chrono::steady_clock;
double ms_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
}  // namespace

struct Server::Live {
  Request req;
  std::promise<Result> promise;
  Result result;
  int seq = -1;
  Clock::time_point arrival, first;
  bool has_first = false;
  std::int32_t pending = -1;  // sonraki adimda beslenecek token (son ornek)
};

Server::Server(infer::Engine& engine, Config config) : engine_(&engine), cfg_(config) {
  if (cfg_.max_running > engine.config().max_seqs) cfg_.max_running = engine.config().max_seqs;
  if (cfg_.max_batch_tokens > engine.config().max_tokens) cfg_.max_batch_tokens = engine.config().max_tokens;
  thread_ = std::thread([this] { loop(); });
}

Server::~Server() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

std::future<Result> Server::submit(Request request) {
  if (request.prompt.empty() || request.max_new < 1) throw std::invalid_argument("nexus::serve: bos istem ya da max_new < 1");
  if (static_cast<int>(request.prompt.size()) > cfg_.max_batch_tokens)
    throw std::invalid_argument("nexus::serve: istem max_batch_tokens'tan uzun");
  auto live = std::make_shared<Live>();
  live->req = std::move(request);
  live->arrival = Clock::now();
  std::future<Result> f = live->promise.get_future();
  {
    std::lock_guard<std::mutex> lock(mu_);
    waiting_.push_back(std::move(live));
  }
  cv_.notify_all();
  return f;
}

void Server::drain() {
  std::unique_lock<std::mutex> lock(mu_);
  idle_.wait(lock, [&] { return waiting_.empty() && running_.empty(); });
}

Metrics Server::metrics() const {
  std::lock_guard<std::mutex> lock(mu_);
  return metrics_;
}

void Server::loop() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&] { return stop_ || !waiting_.empty() || !running_.empty(); });
      if (stop_) return;
    }
    bool progressed = false;
    try {
      progressed = step();
    } catch (const std::exception& e) {
      // motor hatasi: tum istekler hatayla biter (sessiz yutma yok)
      std::lock_guard<std::mutex> lock(mu_);
      for (auto& l : running_) {
        l->promise.set_exception(std::make_exception_ptr(std::runtime_error(std::string("nexus::serve: ") + e.what())));
        if (l->seq >= 0) engine_->free_sequence(l->seq);
      }
      for (auto& l : waiting_) l->promise.set_exception(std::make_exception_ptr(std::runtime_error(std::string("nexus::serve: ") + e.what())));
      running_.clear();
      waiting_.clear();
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (waiting_.empty() && running_.empty()) idle_.notify_all();
    if (!progressed && !waiting_.empty() && running_.empty()) {
      // hicbir istek sigmiyor (bos motorda bile): hata
      auto l = waiting_.front();
      waiting_.pop_front();
      l->promise.set_exception(std::make_exception_ptr(std::runtime_error("nexus::serve: istek KV havuzuna sigmiyor")));
    }
  }
}

bool Server::step() {
  const auto t0 = Clock::now();
  infer::Engine& e = *engine_;
  std::vector<std::shared_ptr<Live>> admit;
  {
    std::lock_guard<std::mutex> lock(mu_);
    // 1. decode icin blok yeterli mi; degilse en son kabul edileni bosalt (recompute)
    for (;;) {
      int need = 0;
      for (auto& l : running_) need += e.blocks_needed(l->seq, 1);
      if (need <= e.free_blocks() || running_.empty()) break;
      auto victim = running_.back();
      running_.pop_back();
      e.free_sequence(victim->seq);
      victim->seq = -1;
      victim->result.preemptions += 1;
      metrics_.preemptions += 1;
      waiting_.push_front(victim);
    }
    // 2. bekleyenleri butce ve bloklar yettikce kabul et
    int budget = cfg_.max_batch_tokens - static_cast<int>(running_.size());
    int blocks = e.free_blocks();
    for (auto& l : running_) blocks -= e.blocks_needed(l->seq, 1);
    while (!waiting_.empty() && static_cast<int>(running_.size() + admit.size()) < cfg_.max_running) {
      auto& l = waiting_.front();
      const int len = static_cast<int>(l->req.prompt.size() + l->result.tokens.size());
      const int nb = (len + 1 + e.block_size() - 1) / e.block_size();
      if (len > budget || nb > blocks) break;
      budget -= len;
      blocks -= nb;
      admit.push_back(l);
      waiting_.pop_front();
    }
  }
  if (admit.empty() && running_.empty()) return false;
  // 3. tek forward: decode + prefill
  std::vector<infer::Chunk> chunks;
  std::vector<std::shared_ptr<Live>> rows;
  for (auto& l : running_) {
    infer::Chunk c;
    c.seq = l->seq;
    c.tokens = {l->pending};
    c.output = true;
    chunks.push_back(std::move(c));
    rows.push_back(l);
  }
  int prefill = 0;
  for (auto& l : admit) {
    l->seq = e.add_sequence();
    infer::Chunk c;
    c.seq = l->seq;
    c.tokens = l->req.prompt;
    c.tokens.insert(c.tokens.end(), l->result.tokens.begin(), l->result.tokens.end());  // preempt sonrasi devam
    prefill += static_cast<int>(c.tokens.size());
    chunks.push_back(std::move(c));
    rows.push_back(l);
  }
  (void)e.forward(chunks);
  std::vector<infer::SamplingParams> sp;
  std::vector<int> seqs;
  std::vector<std::uint64_t> offs;
  for (auto& l : rows) {
    sp.push_back(l->req.sampling);
    seqs.push_back(l->seq);
    offs.push_back(static_cast<std::uint64_t>(l->result.tokens.size()));
  }
  const std::vector<std::int32_t> next = e.sample(sp, seqs, offs);
  const auto now = Clock::now();
  std::vector<std::shared_ptr<Live>> still;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    auto& l = rows[i];
    const std::int32_t t = next[i];
    l->result.tokens.push_back(t);
    if (!l->has_first) {
      l->has_first = true;
      l->first = now;
      l->result.ttft_ms = ms_between(l->arrival, now);
    }
    if (l->req.on_token) l->req.on_token(t);
    const bool stop = t == l->req.stop_token;
    if (stop || static_cast<int>(l->result.tokens.size()) >= l->req.max_new) {
      l->result.stopped = stop;
      l->result.latency_ms = ms_between(l->arrival, now);
      const int n = static_cast<int>(l->result.tokens.size());
      l->result.tpot_ms = n > 1 ? ms_between(l->first, now) / (n - 1) : 0.0;
      e.free_sequence(l->seq);
      l->seq = -1;
      l->promise.set_value(l->result);
    } else {
      l->pending = t;
      still.push_back(l);
    }
  }
  std::lock_guard<std::mutex> lock(mu_);
  running_ = std::move(still);
  metrics_.steps += 1;
  metrics_.generated += static_cast<std::int64_t>(rows.size());
  metrics_.prefill_tokens += prefill;
  metrics_.max_running = std::max(metrics_.max_running, static_cast<int>(rows.size()));
  metrics_.busy_ms += ms_between(t0, Clock::now());
  metrics_.mean_batch += (static_cast<double>(rows.size()) - metrics_.mean_batch) / static_cast<double>(metrics_.steps);
  return true;
}

}  // namespace serve
}  // namespace cuda
}  // namespace nexus
