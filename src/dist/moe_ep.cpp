/**
 * @file moe_ep.cpp
 * @brief FAZ 14.D: uzman paralelligi — yonlendirme, all-to-all(v) dagitim, yerel gruplu GEMM, geri toplama.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/moe_ep.hpp"

#include <chrono>
#include <cstring>
#include <numeric>

namespace nexus {
namespace dist {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

MoeExpertParallel::MoeExpertParallel(GroupPtr ep, moe::Spec spec) : ep_(std::move(ep)), spec_(spec) {
  spec_.validate();
  ranks_ = ep_->size();
  if (spec_.experts % ranks_ != 0) {
    throw DistError("Nexus MoE EP: uzman sayisi (" + std::to_string(spec_.experts) + ") ep'ye (" + std::to_string(ranks_) +
                    ") bolunmeli");
  }
  local_ = spec_.experts / ranks_;
  first_ = ep_->rank() * local_;
}

void MoeExpertParallel::exchange(const float* send, const std::vector<std::int64_t>& send_rows, float* recv,
                                 const std::vector<std::int64_t>& recv_rows) {
  const std::size_t d = static_cast<std::size_t>(spec_.d_model);
  std::vector<std::size_t> sc(static_cast<std::size_t>(ranks_)), rc(static_cast<std::size_t>(ranks_));
  for (int r = 0; r < ranks_; ++r) {
    sc[static_cast<std::size_t>(r)] = static_cast<std::size_t>(send_rows[static_cast<std::size_t>(r)]) * d;
    rc[static_cast<std::size_t>(r)] = static_cast<std::size_t>(recv_rows[static_cast<std::size_t>(r)]) * d;
    if (r != ep_->rank()) stats_.bytes_sent += sc[static_cast<std::size_t>(r)] * 4;
  }
  const auto t0 = Clock::now();
  ep_->all_to_all_v(send, sc, recv, rc, DType::F32)->wait();
  stats_.a2a_ms += ms_since(t0);
}

void MoeExpertParallel::forward(const float* x, std::int64_t N, const float* router_w, const float* up, const float* down,
                                bool training, std::uint64_t seed, float* y) {
  const int d = spec_.d_model, k = spec_.top_k, E = spec_.experts, F = spec_.d_ff;
  stats_ = MoeEpStats{};
  cache_ = moe::LayerCache{};
  cache_.routing = moe::route(x, router_w, nullptr, N, spec_, training, seed);
  moe::Routing& r = cache_.routing;
  // Kuresel yuk dengeleme istatistikleri (aux/z tek rank ile ayni)
  {
    std::vector<std::int64_t> routed = r.routed;
    ep_->all_reduce(routed.data(), routed.size(), DType::I64, ReduceOp::Sum)->wait();
    std::vector<double> stat(r.prob_sum);
    stat.push_back(r.lse_sq_sum);
    ep_->all_reduce(stat.data(), stat.size(), DType::F64, ReduceOp::Sum)->wait();
    std::int64_t total = N;
    ep_->all_reduce(&total, 1, DType::I64, ReduceOp::Sum)->wait();
    r.routed = routed;
    r.prob_sum.assign(stat.begin(), stat.end() - 1);
    r.lse_sq_sum = stat.back();
    r.loss_tokens = total;
    r.recompute_losses(spec_);
  }
  const std::int64_t M = r.rows();
  cache_.x.assign(x, x + N * d);
  cache_.experts.xp.resize(static_cast<std::size_t>(M) * d);
  for (std::int64_t row = 0; row < M; ++row) {
    std::memcpy(cache_.experts.xp.data() + row * d, x + static_cast<std::int64_t>(r.perm_token[static_cast<std::size_t>(row)]) * d,
                static_cast<std::size_t>(d) * 4);
  }
  // Uzman basina sayimlar -> sahip rank'lere (I64 all-to-all)
  send_rows_.assign(static_cast<std::size_t>(ranks_), 0);
  for (int q = 0; q < ranks_; ++q) {
    send_rows_[static_cast<std::size_t>(q)] =
        r.offsets[static_cast<std::size_t>((q + 1) * local_)] - r.offsets[static_cast<std::size_t>(q * local_)];
  }
  std::vector<std::int64_t> my_counts(static_cast<std::size_t>(E));
  for (int e = 0; e < E; ++e) my_counts[static_cast<std::size_t>(e)] = r.offsets[static_cast<std::size_t>(e) + 1] - r.offsets[static_cast<std::size_t>(e)];
  recv_counts_.assign(static_cast<std::size_t>(ranks_) * local_, 0);
  {
    const std::vector<std::size_t> c(static_cast<std::size_t>(ranks_), static_cast<std::size_t>(local_));
    const auto t0 = Clock::now();
    ep_->all_to_all_v(my_counts.data(), c, recv_counts_.data(), c, DType::I64)->wait();
    stats_.a2a_ms += ms_since(t0);
  }
  recv_rows_.assign(static_cast<std::size_t>(ranks_), 0);
  for (int s = 0; s < ranks_; ++s) {
    for (int e = 0; e < local_; ++e) recv_rows_[static_cast<std::size_t>(s)] += recv_counts_[static_cast<std::size_t>(s * local_ + e)];
  }
  const std::int64_t R = std::accumulate(recv_rows_.begin(), recv_rows_.end(), std::int64_t{0});
  std::vector<float> recv(static_cast<std::size_t>(R) * d);
  exchange(cache_.experts.xp.data(), send_rows_, recv.data(), recv_rows_);
  stats_.rows_per_rank = send_rows_;
  // Alinan tampon kaynak-ana (kaynak icinde uzman-ana) -> uzman-ana (uzman icinde kaynak sirasi)
  std::vector<std::int64_t> src_base(static_cast<std::size_t>(ranks_) + 1, 0);
  for (int s = 0; s < ranks_; ++s) src_base[static_cast<std::size_t>(s) + 1] = src_base[static_cast<std::size_t>(s)] + recv_rows_[static_cast<std::size_t>(s)];
  local_offsets_.assign(static_cast<std::size_t>(local_) + 1, 0);
  order_.clear();
  order_.reserve(static_cast<std::size_t>(R));
  for (int e = 0; e < local_; ++e) {
    for (int s = 0; s < ranks_; ++s) {
      std::int64_t start = src_base[static_cast<std::size_t>(s)];
      for (int e2 = 0; e2 < e; ++e2) start += recv_counts_[static_cast<std::size_t>(s * local_ + e2)];
      const std::int64_t n = recv_counts_[static_cast<std::size_t>(s * local_ + e)];
      for (std::int64_t i = 0; i < n; ++i) order_.push_back(start + i);
    }
    local_offsets_[static_cast<std::size_t>(e) + 1] = static_cast<std::int64_t>(order_.size());
  }
  remote_ = moe::ExpertCache{};
  remote_.xp.resize(static_cast<std::size_t>(R) * d);
  for (std::int64_t i = 0; i < R; ++i) {
    std::memcpy(remote_.xp.data() + i * d, recv.data() + order_[static_cast<std::size_t>(i)] * d, static_cast<std::size_t>(d) * 4);
  }
  std::vector<float> yr(static_cast<std::size_t>(R) * d);
  const auto tc = Clock::now();
  moe::experts_forward(remote_.xp.data(), local_offsets_.data(), local_, up, down, d, F, yr.data(), &remote_, spec_.threads);
  stats_.compute_ms += ms_since(tc);
  // Geri: kaynak-ana duzene, all-to-all geri (alinan sayimlarla)
  std::vector<float> back(static_cast<std::size_t>(R) * d);
  for (std::int64_t i = 0; i < R; ++i) {
    std::memcpy(back.data() + order_[static_cast<std::size_t>(i)] * d, yr.data() + i * d, static_cast<std::size_t>(d) * 4);
  }
  cache_.experts.yp.resize(static_cast<std::size_t>(M) * d);
  exchange(back.data(), recv_rows_, cache_.experts.yp.data(), send_rows_);
  std::fill(y, y + N * d, 0.0f);
  for (std::int64_t t = 0; t < N; ++t) {
    float* yt = y + t * d;
    for (int j = 0; j < k; ++j) {
      const std::int64_t row = r.row[static_cast<std::size_t>(t * k + j)];
      if (row < 0) continue;
      const float g = r.gate[static_cast<std::size_t>(t * k + j)];
      const float* src = cache_.experts.yp.data() + row * d;
      for (int i = 0; i < d; ++i) yt[i] += g * src[i];
    }
  }
}

void MoeExpertParallel::backward(const float* dy, const float* router_w, const float* up, const float* down, float rlg,
                                 float* dx, float* drouter, float* dup, float* ddown) {
  const int d = spec_.d_model, k = spec_.top_k, F = spec_.d_ff;
  const moe::Routing& r = cache_.routing;
  const std::int64_t N = r.N, M = r.rows();
  const std::int64_t R = static_cast<std::int64_t>(order_.size());
  std::vector<float> dyp(static_cast<std::size_t>(M) * d), dgate(static_cast<std::size_t>(N) * k, 0.0f);
  for (std::int64_t t = 0; t < N; ++t) {
    for (int j = 0; j < k; ++j) {
      const std::int64_t row = r.row[static_cast<std::size_t>(t * k + j)];
      if (row < 0) continue;
      const float g = r.gate[static_cast<std::size_t>(t * k + j)];
      const float* yr = cache_.experts.yp.data() + row * d;
      double dot = 0.0;
      for (int i = 0; i < d; ++i) {
        dyp[static_cast<std::size_t>(row * d + i)] = g * dy[t * d + i];
        dot += static_cast<double>(dy[t * d + i]) * yr[i];
      }
      dgate[static_cast<std::size_t>(t * k + j)] = static_cast<float>(dot);
    }
  }
  std::vector<float> recv(static_cast<std::size_t>(R) * d), dyr(static_cast<std::size_t>(R) * d), dxr(static_cast<std::size_t>(R) * d);
  exchange(dyp.data(), send_rows_, recv.data(), recv_rows_);
  for (std::int64_t i = 0; i < R; ++i) {
    std::memcpy(dyr.data() + i * d, recv.data() + order_[static_cast<std::size_t>(i)] * d, static_cast<std::size_t>(d) * 4);
  }
  const auto tc = Clock::now();
  moe::experts_backward(dyr.data(), local_offsets_.data(), local_, up, down, d, F, remote_, dxr.data(), dup, ddown, spec_.threads);
  stats_.compute_ms += ms_since(tc);
  for (std::int64_t i = 0; i < R; ++i) {
    std::memcpy(recv.data() + order_[static_cast<std::size_t>(i)] * d, dxr.data() + i * d, static_cast<std::size_t>(d) * 4);
  }
  std::vector<float> dxp(static_cast<std::size_t>(M) * d);
  exchange(recv.data(), recv_rows_, dxp.data(), send_rows_);
  std::fill(dx, dx + N * d, 0.0f);
  for (std::int64_t t = 0; t < N; ++t) {
    for (int j = 0; j < k; ++j) {
      const std::int64_t row = r.row[static_cast<std::size_t>(t * k + j)];
      if (row < 0) continue;
      for (int i = 0; i < d; ++i) dx[t * d + i] += dxp[static_cast<std::size_t>(row * d + i)];
    }
  }
  moe::router_backward(r, cache_.x.data(), router_w, nullptr, dgate.data(), spec_, rlg, dx, drouter, nullptr);
}

}  // namespace dist
}  // namespace nexus
