/**
 * @file pipeline.cpp
 * @brief FAZ 12.E: pipeline zamanlama uretimi, teorik kabarcik, maliyet dengeli bolumleme.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/pipeline.hpp"

#include "nexus/dist/process_group.hpp"

#include <algorithm>
#include <limits>

namespace nexus {
namespace dist {

const char* to_string(PipeSchedule s) noexcept {
  switch (s) {
    case PipeSchedule::GPipe:
      return "gpipe";
    case PipeSchedule::OneF1B:
      return "1f1b";
    case PipeSchedule::Interleaved1F1B:
      return "interleaved_1f1b";
    case PipeSchedule::ZeroBubbleH1:
      return "zb_h1";
  }
  return "?";
}

PipeSchedule parse_pipe_schedule(const std::string& name) {
  if (name == "gpipe") return PipeSchedule::GPipe;
  if (name == "1f1b") return PipeSchedule::OneF1B;
  if (name == "interleaved_1f1b" || name == "interleaved") return PipeSchedule::Interleaved1F1B;
  if (name == "zb_h1" || name == "zero_bubble" || name == "zbh1") return PipeSchedule::ZeroBubbleH1;
  throw DistError("Nexus PP: bilinmeyen zamanlama '" + name + "' (gpipe | 1f1b | interleaved_1f1b | zb_h1)");
}

std::string PipeAction::str() const {
  const char* k = kind == Kind::F ? "F" : kind == Kind::B ? "B" : "W";
  return std::string(k) + std::to_string(mb) + "c" + std::to_string(chunk);
}

std::vector<PipeAction> make_schedule(PipeSchedule s, int pp, int rank, int m, int v) {
  if (pp < 1 || rank < 0 || rank >= pp || m < 1 || v < 1) throw DistError("Nexus PP: gecersiz zamanlama parametresi");
  using K = PipeAction::Kind;
  std::vector<PipeAction> out;
  if (s != PipeSchedule::Interleaved1F1B && v != 1) {
    throw DistError(std::string("Nexus PP: ") + to_string(s) + " sanal asama (v > 1) desteklemez; interleaved_1f1b kullanin");
  }
  if (s == PipeSchedule::GPipe) {
    for (int mb = 0; mb < m; ++mb) out.push_back({K::F, mb, rank});
    for (int mb = 0; mb < m; ++mb) out.push_back({K::B, mb, rank});
    return out;
  }
  if (s == PipeSchedule::OneF1B || s == PipeSchedule::ZeroBubbleH1) {
    const int warm = std::min(pp - rank - 1, m);
    int f = 0, b = 0;
    for (; f < warm; ++f) out.push_back({K::F, f, rank});
    while (f < m) {
      out.push_back({K::F, f++, rank});
      out.push_back({K::B, b++, rank});
    }
    while (b < m) out.push_back({K::B, b++, rank});
    return out;
  }
  // Interleaved 1F1B
  if (m % pp != 0) {
    throw DistError("Nexus PP: interleaved_1f1b mikro yigin sayisinin (" + std::to_string(m) + ") pp'ye (" +
                    std::to_string(pp) + ") bolunmesini ister");
  }
  const int total = m * v;
  auto fwd = [&](int k) {
    const int chunk_local = (k / pp) % v;
    const int mb = (k / (pp * v)) * pp + k % pp;
    return PipeAction{K::F, mb, chunk_local * pp + rank};
  };
  auto bwd = [&](int k) {
    const int chunk_local = v - 1 - (k / pp) % v;
    const int mb = (k / (pp * v)) * pp + k % pp;
    return PipeAction{K::B, mb, chunk_local * pp + rank};
  };
  const int warm = std::min((pp - rank - 1) * 2 + (v - 1) * pp, total);
  int f = 0, b = 0;
  for (; f < warm; ++f) out.push_back(fwd(f));
  while (f < total) {
    out.push_back(fwd(f++));
    out.push_back(bwd(b++));
  }
  while (b < total) out.push_back(bwd(b++));
  return out;
}

double theoretical_bubble(PipeSchedule s, int pp, int m, int v, double tf, double tb, double tw) {
  const double p1 = pp - 1;
  switch (s) {
    case PipeSchedule::GPipe:
    case PipeSchedule::OneF1B:
      return p1 / (m + p1);
    case PipeSchedule::Interleaved1F1B:
      return p1 / (static_cast<double>(v) * m + p1);
    case PipeSchedule::ZeroBubbleH1: {
      const double idle = p1 * std::max(0.0, tf + tb - tw);
      return idle / (m * (tf + tb + tw) + idle);
    }
  }
  return 0.0;
}

std::vector<std::pair<int, int>> balance_partition(const std::vector<double>& costs, int parts) {
  const int n = static_cast<int>(costs.size());
  if (parts < 1 || parts > n) {
    throw DistError("Nexus PP: " + std::to_string(n) + " katman " + std::to_string(parts) + " parcaya bolunemez");
  }
  std::vector<double> pre(static_cast<std::size_t>(n) + 1, 0.0);
  for (int i = 0; i < n; ++i) pre[static_cast<std::size_t>(i) + 1] = pre[static_cast<std::size_t>(i)] + costs[static_cast<std::size_t>(i)];
  const double inf = std::numeric_limits<double>::infinity();
  // best[p][i]: ilk i katmani p parcaya bolmenin en kucuk en buyuk maliyeti.
  std::vector<std::vector<double>> best(static_cast<std::size_t>(parts) + 1, std::vector<double>(static_cast<std::size_t>(n) + 1, inf));
  std::vector<std::vector<int>> cut(static_cast<std::size_t>(parts) + 1, std::vector<int>(static_cast<std::size_t>(n) + 1, 0));
  best[0][0] = 0.0;
  for (int p = 1; p <= parts; ++p) {
    for (int i = p; i <= n; ++i) {
      for (int j = p - 1; j < i; ++j) {
        const double c = std::max(best[static_cast<std::size_t>(p) - 1][static_cast<std::size_t>(j)],
                                  pre[static_cast<std::size_t>(i)] - pre[static_cast<std::size_t>(j)]);
        if (c < best[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)] - 1e-12) {
          best[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)] = c;
          cut[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)] = j;
        }
      }
    }
  }
  std::vector<std::pair<int, int>> out(static_cast<std::size_t>(parts));
  int i = n;
  for (int p = parts; p >= 1; --p) {
    const int j = cut[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)];
    out[static_cast<std::size_t>(p) - 1] = {j, i};
    i = j;
  }
  return out;
}

}  // namespace dist
}  // namespace nexus
