/**
 * @file ddp.cpp
 * @brief FAZ 10.3: kova planlayici ve host kova indirgeyici.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/ddp.hpp"

#include "nexus/core/half.hpp"

namespace nexus {
namespace dist {

std::vector<Bucket> plan_buckets(const std::vector<Stage>& stages, std::size_t bucket_bytes, std::size_t element_bytes) {
  std::vector<Bucket> out;
  Bucket cur;
  for (const Stage& st : stages) {
    cur.stages.push_back(st.id);
    for (const GradSlice& s : st.slices) {
      cur.slices.push_back(s);
      cur.numel += s.numel;
    }
    if (cur.numel * element_bytes >= bucket_bytes) {
      out.push_back(std::move(cur));
      cur = Bucket{};
    }
  }
  if (!cur.stages.empty()) {
    out.push_back(std::move(cur));
  }
  return out;
}

void HostBucket::resize(std::size_t numel, int pad_to) {
  numel_ = numel;
  std::size_t padded = numel;
  if (pad_to > 1) {
    const auto k = static_cast<std::size_t>(pad_to);
    padded = (numel + k - 1) / k * k;  // hiyerarsik all-reduce: dugum basina rank sayisina bolunur
  }
  data_.assign(padded, 0.0f);
  external_ = nullptr;
}

void HostBucket::bind(float* external, std::size_t numel, std::size_t capacity) {
  external_ = external;
  numel_ = numel;
  capacity_ = capacity;
}

void HostBucket::launch(ProcessGroup& pg, const DdpOptions& options) {
  compressed_ = options.bf16_compress;
  float* f = data();
  std::size_t n = numel_;
  if (options.ranks_per_node > 1) {
    const auto k = static_cast<std::size_t>(options.ranks_per_node);
    n = (numel_ + k - 1) / k * k;
    if (n > capacity()) {
      throw DistError("Nexus DDP: kova tamponu hiyerarsik dolgu icin kucuk");
    }
  }
  void* buf = f;
  DType dt = DType::F32;
  if (compressed_) {
    half_.resize(n);
    for (std::size_t i = 0; i < n; ++i) half_[i] = float_to_bfloat16_bits(f[i]);
    buf = half_.data();
    dt = DType::BF16;
  }
  work_ = options.ranks_per_node > 0 ? pg.all_reduce_hierarchical(buf, n, dt, ReduceOp::Sum, options.ranks_per_node)
                                     : pg.all_reduce(buf, n, dt, ReduceOp::Sum);
}

void HostBucket::finish() {
  if (!work_) {
    return;
  }
  work_->wait();
  if (compressed_) {
    float* f = data();
    for (std::size_t i = 0; i < numel_; ++i) f[i] = bfloat16_bits_to_float(half_[i]);
  }
}

}  // namespace dist
}  // namespace nexus
