#pragma once

/**
 * @file transpose.hpp
 * @brief FAZ 17 ic yardimci: paralel karolu devrik kopya (HF / GGUF agirlik duzeni donusumleri).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace nexus {
namespace interop {
namespace detail {

/// dst[r * dst_cols + dst_col0 + c] = src[c * src_cols + src_col0 + r], c < out, r < in (64x64 karolar, is parcaciklari).
inline void transpose_block(float* dst, std::int64_t dst_cols, std::int64_t dst_col0, const float* src, std::int64_t src_cols,
                            std::int64_t src_col0, std::int64_t out, std::int64_t in) {
  constexpr std::int64_t B = 64;
  const std::int64_t tiles = (out + B - 1) / B;
  const int nt = std::max(1, std::min(static_cast<int>(std::thread::hardware_concurrency()), static_cast<int>(tiles)));
  auto work = [&](int t) {
    for (std::int64_t tc = t; tc < tiles; tc += nt) {
      const std::int64_t c0 = tc * B, c1 = std::min(out, c0 + B);
      for (std::int64_t r0 = 0; r0 < in; r0 += B) {
        const std::int64_t r1 = std::min(in, r0 + B);
        for (std::int64_t c = c0; c < c1; ++c)
          for (std::int64_t r = r0; r < r1; ++r) dst[r * dst_cols + dst_col0 + c] = src[c * src_cols + src_col0 + r];
      }
    }
  };
  if (nt == 1 || out * in < (1 << 16)) {
    for (int t = 0; t < nt; ++t) work(t);
    return;
  }
  std::vector<std::thread> ts;
  for (int t = 0; t < nt; ++t) ts.emplace_back(work, t);
  for (auto& th : ts) th.join();
}

}  // namespace detail
}  // namespace interop
}  // namespace nexus
