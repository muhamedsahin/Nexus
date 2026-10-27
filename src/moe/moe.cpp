/**
 * @file moe.cpp
 * @brief FAZ 14: MoE cekirdegi — yonlendirme, kapasite, permutasyon, gruplu GEMM, birlestirme, kayiplar, geri gecis,
 *        dar uzman agirliklari ve double hassasiyetli bagimsiz referans.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cpu/gemm.hpp"
#include "nexus/moe/moe.hpp"

#include "nexus/core/fp8.hpp"
#include "nexus/core/half.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <mutex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace nexus {
namespace moe {

namespace K = nexus_model::kernels;

namespace {

[[noreturn]] void bad(const std::string& w) { throw std::runtime_error("Nexus MoE: " + w); }

void parallel_tasks(int count, int threads, const std::function<void(int)>& fn) {
  if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  threads = std::max(1, std::min(threads, count));
  if (threads == 1) {
    for (int i = 0; i < count; ++i) fn(i);
    return;
  }
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  std::exception_ptr err;
  std::mutex mu;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      try {
        for (int i = next++; i < count; i = next++) fn(i);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mu);
        err = std::current_exception();
      }
    });
  }
  for (auto& th : pool) th.join();
  if (err) std::rethrow_exception(err);
}

inline void silu(float a, float* f, float* df) {
  const float sg = 1.0f / (1.0f + std::exp(-a));
  *f = a * sg;
  *df = sg * (1.0f + a * (1.0f - sg));
}

/// splitmix64 + Box-Muller (platformdan bagimsiz, belirlenimci).
double gaussian(std::uint64_t seed, std::uint64_t idx) {
  auto mix = [](std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  };
  const std::uint64_t a = mix(seed ^ (idx * 2 + 1)), b = mix(seed ^ (idx * 2 + 2) ^ 0xA5A5A5A5ULL);
  const double u1 = (static_cast<double>(a >> 11) + 0.5) / 9007199254740992.0;
  const double u2 = (static_cast<double>(b >> 11) + 0.5) / 9007199254740992.0;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

float softplus(float v) { return v > 20.0f ? v : std::log1p(std::exp(v)); }

}  // namespace

void Spec::validate() const {
  if (experts < 1 || top_k < 1 || top_k > experts || d_model < 1 || d_ff < 1 || groups < 1 || capacity_factor <= 0.0f ||
      min_capacity < 0) {
    bad("Spec gecersiz (E >= k >= 1, d, F, groups >= 1, capacity_factor > 0)");
  }
}

void Routing::recompute_losses(const Spec& s) {
  const int E = s.experts, k = s.top_k;
  const double Nl = static_cast<double>(loss_tokens > 0 ? loss_tokens : N);
  double a = 0.0;
  for (int e = 0; e < E; ++e) {
    a += (static_cast<double>(routed[static_cast<std::size_t>(e)]) / (Nl * k)) * (prob_sum[static_cast<std::size_t>(e)] / Nl);
  }
  aux = static_cast<float>(s.aux_alpha * E * a);
  z = static_cast<float>(s.z_alpha * lse_sq_sum / Nl);
}

int Routing::idle_experts() const {
  int idle = 0;
  for (std::size_t e = 0; e + 1 < offsets.size(); ++e) idle += offsets[e + 1] == offsets[e] ? 1 : 0;
  return idle;
}

// ==========================================================================================
// Yonlendirme
// ==========================================================================================

Routing route(const float* x, const float* router_w, const float* noise_w, std::int64_t N, const Spec& s, bool training,
              std::uint64_t seed) {
  s.validate();
  const int E = s.experts, k = s.top_k, d = s.d_model;
  if (N % s.groups != 0) bad("token sayisi (" + std::to_string(N) + ") groups'a (" + std::to_string(s.groups) + ") bolunmeli");
  Routing r;
  r.N = N;
  const std::size_t NE = static_cast<std::size_t>(N) * E;
  r.logits.resize(NE);
  nexus::cpu::gemm_nn(x, router_w, r.logits.data(), static_cast<int>(N), d, E);
  r.probs.resize(NE);
  double zsum = 0.0;
  std::vector<double> P(static_cast<std::size_t>(E), 0.0);
  for (std::int64_t t = 0; t < N; ++t) {
    const float* l = r.logits.data() + t * E;
    float m = -std::numeric_limits<float>::infinity();
    for (int e = 0; e < E; ++e) m = std::max(m, l[e]);
    double sum = 0.0;
    for (int e = 0; e < E; ++e) sum += std::exp(static_cast<double>(l[e] - m));
    const double lse = m + std::log(sum);
    zsum += lse * lse;
    for (int e = 0; e < E; ++e) {
      const float p = static_cast<float>(std::exp(static_cast<double>(l[e]) - lse));
      r.probs[static_cast<std::size_t>(t * E + e)] = p;
      P[static_cast<std::size_t>(e)] += p;
    }
  }
  const bool noisy = training && s.noisy && noise_w != nullptr;
  r.select = r.logits;
  if (noisy) {
    r.noise_pre.resize(NE);
    r.noise_eps.resize(NE);
    nexus::cpu::gemm_nn(x, noise_w, r.noise_pre.data(), static_cast<int>(N), d, E);
    for (std::size_t i = 0; i < NE; ++i) {
      r.noise_eps[i] = static_cast<float>(gaussian(seed, i));
      r.select[i] += r.noise_eps[i] * softplus(r.noise_pre[i]);
    }
  }
  // top-k (esitlikte kucuk indeks) ve secilenler uzerinde softmax kapi
  r.expert.resize(static_cast<std::size_t>(N) * k);
  r.gate.resize(static_cast<std::size_t>(N) * k);
  r.row.assign(static_cast<std::size_t>(N) * k, -1);
  r.routed.assign(static_cast<std::size_t>(E), 0);
  std::vector<int> order(static_cast<std::size_t>(E));
  std::vector<std::vector<int>> rank_of(static_cast<std::size_t>(N));  // reroute icin tum siralama
  for (std::int64_t t = 0; t < N; ++t) {
    const float* h = r.select.data() + t * E;
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return h[a] > h[b]; });
    if (s.overflow == Overflow::Reroute) rank_of[static_cast<std::size_t>(t)] = order;
    float m = h[order[0]];
    double sum = 0.0;
    for (int j = 0; j < k; ++j) sum += std::exp(static_cast<double>(h[order[static_cast<std::size_t>(j)]] - m));
    for (int j = 0; j < k; ++j) {
      const int e = order[static_cast<std::size_t>(j)];
      r.expert[static_cast<std::size_t>(t * k + j)] = e;
      r.gate[static_cast<std::size_t>(t * k + j)] = static_cast<float>(std::exp(static_cast<double>(h[e] - m)) / sum);
      r.routed[static_cast<std::size_t>(e)] += 1;
    }
  }
  // aux + z (temiz logitler; f kapasite oncesi)
  r.prob_sum = P;
  r.lse_sq_sum = zsum;
  r.loss_tokens = N;
  r.recompute_losses(s);
  // kapasite ve kabul (grup grup; once tum 1. secimler)
  const std::int64_t Ng = N / s.groups;
  r.capacity = std::max<std::int64_t>(s.min_capacity, std::llround(static_cast<double>(k) * Ng / E * s.capacity_factor));
  std::vector<std::vector<std::vector<std::int64_t>>> lists(static_cast<std::size_t>(E),
                                                           std::vector<std::vector<std::int64_t>>(static_cast<std::size_t>(s.groups)));
  for (int g = 0; g < s.groups; ++g) {
    std::vector<std::int64_t> fill(static_cast<std::size_t>(E), 0);
    for (int j = 0; j < k; ++j) {
      for (std::int64_t t = g * Ng; t < (g + 1) * Ng; ++t) {
        const std::size_t slot = static_cast<std::size_t>(t * k + j);
        int e = r.expert[slot];
        if (fill[static_cast<std::size_t>(e)] >= r.capacity) {
          if (s.overflow == Overflow::Drop) {
            r.dropped += 1;
            continue;
          }
          int alt = -1;
          for (int e2 : rank_of[static_cast<std::size_t>(t)]) {
            bool used = false;
            for (int m = 0; m < k; ++m) used = used || r.expert[static_cast<std::size_t>(t * k + m)] == e2;
            if (!used && fill[static_cast<std::size_t>(e2)] < r.capacity) {
              alt = e2;
              break;
            }
          }
          if (alt < 0) {
            r.dropped += 1;
            continue;
          }
          r.rerouted += 1;
          r.expert[slot] = alt;  // kapi agirligi korunur; geri gecis ozgun secime (logit) akar
          e = alt;
        }
        fill[static_cast<std::size_t>(e)] += 1;
        lists[static_cast<std::size_t>(e)][static_cast<std::size_t>(g)].push_back(static_cast<std::int64_t>(slot));
      }
    }
  }
  r.offsets.assign(static_cast<std::size_t>(E) + 1, 0);
  for (int e = 0; e < E; ++e) {
    std::int64_t c = 0;
    for (const auto& l : lists[static_cast<std::size_t>(e)]) c += static_cast<std::int64_t>(l.size());
    r.offsets[static_cast<std::size_t>(e) + 1] = r.offsets[static_cast<std::size_t>(e)] + c;
  }
  r.perm_token.resize(static_cast<std::size_t>(r.offsets.back()));
  std::int64_t next = 0;
  for (int e = 0; e < E; ++e) {
    for (const auto& l : lists[static_cast<std::size_t>(e)]) {
      for (std::int64_t slot : l) {
        r.row[static_cast<std::size_t>(slot)] = next;
        r.perm_token[static_cast<std::size_t>(next)] = static_cast<std::int32_t>(slot / k);
        ++next;
      }
    }
  }
  return r;
}

// ==========================================================================================
// Gruplu uzman hesabi
// ==========================================================================================

namespace {

void expert_forward_one(const float* xp, std::int64_t n, const float* up, const float* down, int d, int F, float* h, float* a,
                        float* yp) {
  nexus::cpu::gemm_nn(xp, up, h, static_cast<int>(n), d, 2 * F);
  for (std::int64_t r = 0; r < n; ++r) {
    const float* hr = h + r * 2 * F;
    float* ar = a + r * F;
    for (int i = 0; i < F; ++i) {
      float f = 0.0f, df = 0.0f;
      silu(hr[i], &f, &df);
      ar[i] = f * hr[F + i];
    }
  }
  nexus::cpu::gemm_nn(a, down, yp, static_cast<int>(n), F, d);
}

}  // namespace

void experts_forward(const float* xp, const std::int64_t* off, int E, const float* up, const float* down, int d, int F, float* yp,
                     ExpertCache* cache, int threads) {
  const std::int64_t M = off[E];
  std::vector<float> h_local, a_local;
  float* h = nullptr;
  float* a = nullptr;
  if (cache) {
    cache->h.resize(static_cast<std::size_t>(M) * 2 * F);
    cache->a.resize(static_cast<std::size_t>(M) * F);
    h = cache->h.data();
    a = cache->a.data();
  } else {
    h_local.resize(static_cast<std::size_t>(M) * 2 * F);
    a_local.resize(static_cast<std::size_t>(M) * F);
    h = h_local.data();
    a = a_local.data();
  }
  const std::size_t wu = static_cast<std::size_t>(d) * 2 * F, wd = static_cast<std::size_t>(F) * d;
  parallel_tasks(E, threads, [&](int e) {
    const std::int64_t o = off[e], n = off[e + 1] - off[e];
    if (n == 0) return;
    expert_forward_one(xp + o * d, n, up + e * wu, down + e * wd, d, F, h + o * 2 * F, a + o * F, yp + o * d);
  });
}

void experts_backward(const float* dyp, const std::int64_t* off, int E, const float* up, const float* down, int d, int F,
                      const ExpertCache& c, float* dxp, float* dup, float* ddown, int threads) {
  const std::size_t wu = static_cast<std::size_t>(d) * 2 * F, wd = static_cast<std::size_t>(F) * d;
  parallel_tasks(E, threads, [&](int e) {
    const std::int64_t o = off[e], n = off[e + 1] - off[e];
    if (n == 0) return;
    const float* dy = dyp + o * d;
    const float* h = c.h.data() + o * 2 * F;
    const float* a = c.a.data() + o * F;
    const float* x = c.xp.data() + o * d;
    std::vector<float> da(static_cast<std::size_t>(n) * F, 0.0f), dh(static_cast<std::size_t>(n) * 2 * F);
    nexus::cpu::gemm_nt_accum(dy, down + e * wd, da.data(), static_cast<int>(n), d, F);
    nexus::cpu::gemm_tn_accum(a, dy, ddown + e * wd, F, static_cast<int>(n), d);
    for (std::int64_t r = 0; r < n; ++r) {
      const float* hr = h + r * 2 * F;
      for (int i = 0; i < F; ++i) {
        float f = 0.0f, df = 0.0f;
        silu(hr[i], &f, &df);
        const float g = da[static_cast<std::size_t>(r * F + i)];
        dh[static_cast<std::size_t>(r * 2 * F + i)] = g * hr[F + i] * df;
        dh[static_cast<std::size_t>(r * 2 * F + F + i)] = g * f;
      }
    }
    nexus::cpu::gemm_tn_accum(x, dh.data(), dup + e * wu, d, static_cast<int>(n), 2 * F);
    float* dx = dxp + o * d;
    std::fill(dx, dx + n * d, 0.0f);
    nexus::cpu::gemm_nt_accum(dh.data(), up + e * wu, dx, static_cast<int>(n), 2 * F, d);
  });
}

// ==========================================================================================
// Dar agirliklar (14.F)
// ==========================================================================================

std::uint64_t QuantizedExperts::bytes() const {
  return bf16_up.size() * 2 + bf16_down.size() * 2 + fp8_up.size() + fp8_down.size() + (scale_up.size() + scale_down.size()) * 4;
}

QuantizedExperts quantize_experts(const float* up, const float* down, int E, int d, int F, WeightType t) {
  if (t == WeightType::F32) bad("quantize_experts: BF16 ya da FP8 secin");
  QuantizedExperts q;
  q.type = t;
  q.experts = E;
  q.d = d;
  q.F = F;
  const std::size_t wu = static_cast<std::size_t>(d) * 2 * F, wd = static_cast<std::size_t>(F) * d;
  if (t == WeightType::BF16) {
    q.bf16_up.resize(wu * E);
    q.bf16_down.resize(wd * E);
    for (std::size_t i = 0; i < q.bf16_up.size(); ++i) q.bf16_up[i] = float_to_bfloat16_bits(up[i]);
    for (std::size_t i = 0; i < q.bf16_down.size(); ++i) q.bf16_down[i] = float_to_bfloat16_bits(down[i]);
    return q;
  }
  q.fp8_up.resize(wu * E);
  q.fp8_down.resize(wd * E);
  q.scale_up.resize(static_cast<std::size_t>(E));
  q.scale_down.resize(static_cast<std::size_t>(E));
  auto enc = [](const float* src, std::size_t n, std::uint8_t* dst, float* scale) {
    float amax = 0.0f;
    for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(src[i]));
    *scale = amax > 0.0f ? amax / 448.0f : 1.0f;  // E4M3 azami 448
    for (std::size_t i = 0; i < n; ++i) dst[i] = float_to_fp8(src[i] / *scale, Fp8Format::E4M3);
  };
  for (int e = 0; e < E; ++e) {
    enc(up + e * wu, wu, q.fp8_up.data() + e * wu, &q.scale_up[static_cast<std::size_t>(e)]);
    enc(down + e * wd, wd, q.fp8_down.data() + e * wd, &q.scale_down[static_cast<std::size_t>(e)]);
  }
  return q;
}

void experts_forward_quantized(const float* xp, const std::int64_t* off, const QuantizedExperts& q, float* yp, int threads) {
  const int d = q.d, F = q.F;
  const std::size_t wu = static_cast<std::size_t>(d) * 2 * F, wd = static_cast<std::size_t>(F) * d;
  parallel_tasks(q.experts, threads, [&](int e) {
    const std::int64_t o = off[e], n = off[e + 1] - off[e];
    if (n == 0) return;
    std::vector<float> u(wu), dn(wd), h(static_cast<std::size_t>(n) * 2 * F), a(static_cast<std::size_t>(n) * F);
    for (std::size_t i = 0; i < wu; ++i) {
      u[i] = q.type == WeightType::BF16 ? bfloat16_bits_to_float(q.bf16_up[e * wu + i])
                                         : fp8_to_float(q.fp8_up[e * wu + i], Fp8Format::E4M3) * q.scale_up[static_cast<std::size_t>(e)];
    }
    for (std::size_t i = 0; i < wd; ++i) {
      dn[i] = q.type == WeightType::BF16 ? bfloat16_bits_to_float(q.bf16_down[e * wd + i])
                                          : fp8_to_float(q.fp8_down[e * wd + i], Fp8Format::E4M3) * q.scale_down[static_cast<std::size_t>(e)];
    }
    expert_forward_one(xp + o * d, n, u.data(), dn.data(), d, F, h.data(), a.data(), yp + o * d);
  });
}

// ==========================================================================================
// Katman
// ==========================================================================================

void layer_forward(const float* x, std::int64_t N, const float* router_w, const float* noise_w, const float* up,
                   const float* down, const Spec& s, bool training, std::uint64_t seed, float* y, LayerCache* cache) {
  const int d = s.d_model, k = s.top_k;
  LayerCache local;
  LayerCache& c = cache ? *cache : local;
  c.routing = route(x, router_w, noise_w, N, s, training, seed);
  const Routing& r = c.routing;
  const std::int64_t M = r.rows();
  c.x.assign(x, x + N * d);
  c.experts.xp.resize(static_cast<std::size_t>(M) * d);
  for (std::int64_t row = 0; row < M; ++row) {
    std::memcpy(c.experts.xp.data() + row * d, x + static_cast<std::int64_t>(r.perm_token[static_cast<std::size_t>(row)]) * d,
                static_cast<std::size_t>(d) * 4);
  }
  c.experts.yp.resize(static_cast<std::size_t>(M) * d);
  experts_forward(c.experts.xp.data(), r.offsets.data(), s.experts, up, down, d, s.d_ff, c.experts.yp.data(), &c.experts, s.threads);
  std::fill(y, y + N * d, 0.0f);
  for (std::int64_t t = 0; t < N; ++t) {
    float* yt = y + t * d;
    for (int j = 0; j < k; ++j) {
      const std::int64_t row = r.row[static_cast<std::size_t>(t * k + j)];
      if (row < 0) continue;
      const float g = r.gate[static_cast<std::size_t>(t * k + j)];
      const float* yr = c.experts.yp.data() + row * d;
      for (int i = 0; i < d; ++i) yt[i] += g * yr[i];
    }
  }
}

void router_backward(const Routing& r, const float* x, const float* router_w, const float* noise_w, const float* dgate,
                     const Spec& s, float rlg, float* dx, float* drouter, float* dnoise) {
  const int E = s.experts, k = s.top_k, d = s.d_model;
  const std::int64_t N = r.N;
  // Kapi gradyani secim logitlerine (ozgun top-k sirasi: kapi ozgun secimden hesaplandi).
  std::vector<float> dsel(static_cast<std::size_t>(N) * E, 0.0f);
  std::vector<int> order(static_cast<std::size_t>(E));
  for (std::int64_t t = 0; t < N; ++t) {
    const float* h = r.select.data() + t * E;
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return h[a] > h[b]; });
    double sdot = 0.0;
    for (int j = 0; j < k; ++j) sdot += static_cast<double>(r.gate[static_cast<std::size_t>(t * k + j)]) * dgate[t * k + j];
    for (int j = 0; j < k; ++j) {
      const float g = r.gate[static_cast<std::size_t>(t * k + j)];
      dsel[static_cast<std::size_t>(t * E + order[static_cast<std::size_t>(j)])] +=
          static_cast<float>(g * (dgate[t * k + j] - sdot));
    }
  }
  std::vector<float> dl = dsel;  // temiz logitlere (H = logits + gurultu)
  if (rlg != 0.0f) {
    const double Nl = static_cast<double>(r.loss_tokens > 0 ? r.loss_tokens : N);  // EP: kuresel token sayisi
    std::vector<double> f(static_cast<std::size_t>(E));
    for (int e = 0; e < E; ++e) f[static_cast<std::size_t>(e)] = static_cast<double>(r.routed[static_cast<std::size_t>(e)]) / (Nl * k);
    for (std::int64_t t = 0; t < N; ++t) {
      const float* p = r.probs.data() + t * E;
      const float* l = r.logits.data() + t * E;
      double pf = 0.0;
      for (int e = 0; e < E; ++e) pf += p[e] * f[static_cast<std::size_t>(e)];
      float m = -std::numeric_limits<float>::infinity();
      for (int e = 0; e < E; ++e) m = std::max(m, l[e]);
      double se = 0.0;
      for (int e = 0; e < E; ++e) se += std::exp(static_cast<double>(l[e] - m));
      const double lse = m + std::log(se);
      for (int e = 0; e < E; ++e) {
        const double aux = s.aux_alpha * E / Nl * p[e] * (f[static_cast<std::size_t>(e)] - pf);
        const double z = s.z_alpha * 2.0 * lse / Nl * p[e];
        dl[static_cast<std::size_t>(t * E + e)] += static_cast<float>(rlg * (aux + z));
      }
    }
  }
  if (!r.noise_eps.empty() && noise_w != nullptr) {
    std::vector<float> dpre(dsel.size());
    for (std::size_t i = 0; i < dsel.size(); ++i) {
      const float sg = 1.0f / (1.0f + std::exp(-r.noise_pre[i]));
      dpre[i] = dsel[i] * r.noise_eps[i] * sg;
    }
    nexus::cpu::gemm_tn_accum(x, dpre.data(), dnoise, d, static_cast<int>(N), E);
    nexus::cpu::gemm_nt_accum(dpre.data(), noise_w, dx, static_cast<int>(N), E, d);
  }
  nexus::cpu::gemm_tn_accum(x, dl.data(), drouter, d, static_cast<int>(N), E);
  nexus::cpu::gemm_nt_accum(dl.data(), router_w, dx, static_cast<int>(N), E, d);
}

void layer_backward(const float* dy, const LayerCache& c, const float* router_w, const float* noise_w, const float* up,
                    const float* down, const Spec& s, float rlg, float* dx, float* drouter, float* dnoise, float* dup,
                    float* ddown) {
  const int d = s.d_model, k = s.top_k;
  const Routing& r = c.routing;
  const std::int64_t N = r.N, M = r.rows();
  std::vector<float> dyp(static_cast<std::size_t>(M) * d), dgate(static_cast<std::size_t>(N) * k, 0.0f);
  for (std::int64_t t = 0; t < N; ++t) {
    const float* dyt = dy + t * d;
    for (int j = 0; j < k; ++j) {
      const std::int64_t row = r.row[static_cast<std::size_t>(t * k + j)];
      if (row < 0) continue;
      const float g = r.gate[static_cast<std::size_t>(t * k + j)];
      const float* yr = c.experts.yp.data() + row * d;
      float* dr = dyp.data() + row * d;
      double dot = 0.0;
      for (int i = 0; i < d; ++i) {
        dr[i] = g * dyt[i];
        dot += static_cast<double>(dyt[i]) * yr[i];
      }
      dgate[static_cast<std::size_t>(t * k + j)] = static_cast<float>(dot);
    }
  }
  std::vector<float> dxp(static_cast<std::size_t>(M) * d);
  experts_backward(dyp.data(), r.offsets.data(), s.experts, up, down, d, s.d_ff, c.experts, dxp.data(), dup, ddown, s.threads);
  std::fill(dx, dx + N * d, 0.0f);
  for (std::int64_t t = 0; t < N; ++t) {
    float* dxt = dx + t * d;
    for (int j = 0; j < k; ++j) {
      const std::int64_t row = r.row[static_cast<std::size_t>(t * k + j)];
      if (row < 0) continue;
      const float* src = dxp.data() + row * d;
      for (int i = 0; i < d; ++i) dxt[i] += src[i];
    }
  }
  router_backward(r, c.x.data(), router_w, noise_w, dgate.data(), s, rlg, dx, drouter, dnoise);
}

// ==========================================================================================
// Bagimsiz referans (double, token basina)
// ==========================================================================================

Reference reference(const float* x, std::int64_t N, const float* Wg, const float* up, const float* down, const Routing& routing,
                    const Spec& s, const float* dy, float rlg) {
  const int E = s.experts, k = s.top_k, d = s.d_model, F = s.d_ff;
  Reference R;
  R.y.assign(static_cast<std::size_t>(N) * d, 0.0);
  R.dx.assign(static_cast<std::size_t>(N) * d, 0.0);
  R.drouter.assign(static_cast<std::size_t>(d) * E, 0.0);
  R.dup.assign(static_cast<std::size_t>(E) * d * 2 * F, 0.0);
  R.ddown.assign(static_cast<std::size_t>(E) * F * d, 0.0);
  std::vector<double> P(static_cast<std::size_t>(E), 0.0), f(static_cast<std::size_t>(E));
  for (int e = 0; e < E; ++e) f[static_cast<std::size_t>(e)] = static_cast<double>(routing.routed[static_cast<std::size_t>(e)]) / (static_cast<double>(N) * k);
  double zs = 0.0;
  for (std::int64_t t = 0; t < N; ++t) {
    const float* xt = x + t * d;
    std::vector<double> l(static_cast<std::size_t>(E), 0.0);
    for (int e = 0; e < E; ++e) {
      for (int i = 0; i < d; ++i) l[static_cast<std::size_t>(e)] += static_cast<double>(xt[i]) * Wg[i * E + e];
    }
    double m = *std::max_element(l.begin(), l.end()), se = 0.0;
    for (double v : l) se += std::exp(v - m);
    const double lse = m + std::log(se);
    std::vector<double> p(static_cast<std::size_t>(E));
    for (int e = 0; e < E; ++e) {
      p[static_cast<std::size_t>(e)] = std::exp(l[static_cast<std::size_t>(e)] - lse);
      P[static_cast<std::size_t>(e)] += p[static_cast<std::size_t>(e)] / N;
    }
    zs += lse * lse;
    // secilenlerin (rotalamadan, ayrik karar) kapisi double'da
    std::vector<int> sel(static_cast<std::size_t>(k));
    {
      std::vector<int> ord(static_cast<std::size_t>(E));
      std::iota(ord.begin(), ord.end(), 0);
      std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return l[static_cast<std::size_t>(a)] > l[static_cast<std::size_t>(b)]; });
      for (int j = 0; j < k; ++j) sel[static_cast<std::size_t>(j)] = ord[static_cast<std::size_t>(j)];
    }
    double gm = l[static_cast<std::size_t>(sel[0])], gs = 0.0;
    for (int j = 0; j < k; ++j) gs += std::exp(l[static_cast<std::size_t>(sel[static_cast<std::size_t>(j)])] - gm);
    std::vector<double> g(static_cast<std::size_t>(k)), dg(static_cast<std::size_t>(k), 0.0);
    for (int j = 0; j < k; ++j) g[static_cast<std::size_t>(j)] = std::exp(l[static_cast<std::size_t>(sel[static_cast<std::size_t>(j)])] - gm) / gs;
    std::vector<double> dxt(static_cast<std::size_t>(d), 0.0);
    for (int j = 0; j < k; ++j) {
      if (routing.row[static_cast<std::size_t>(t * k + j)] < 0) continue;
      const int e = routing.expert[static_cast<std::size_t>(t * k + j)];
      const float* U = up + static_cast<std::size_t>(e) * d * 2 * F;
      const float* Dn = down + static_cast<std::size_t>(e) * F * d;
      std::vector<double> h(static_cast<std::size_t>(2 * F), 0.0), a(static_cast<std::size_t>(F)), o(static_cast<std::size_t>(d), 0.0);
      for (int c = 0; c < 2 * F; ++c) {
        for (int i = 0; i < d; ++i) h[static_cast<std::size_t>(c)] += static_cast<double>(xt[i]) * U[i * 2 * F + c];
      }
      for (int c = 0; c < F; ++c) {
        const double sg = 1.0 / (1.0 + std::exp(-h[static_cast<std::size_t>(c)]));
        a[static_cast<std::size_t>(c)] = h[static_cast<std::size_t>(c)] * sg * h[static_cast<std::size_t>(F + c)];
      }
      for (int i = 0; i < d; ++i) {
        for (int c = 0; c < F; ++c) o[static_cast<std::size_t>(i)] += a[static_cast<std::size_t>(c)] * Dn[c * d + i];
      }
      for (int i = 0; i < d; ++i) R.y[static_cast<std::size_t>(t * d + i)] += g[static_cast<std::size_t>(j)] * o[static_cast<std::size_t>(i)];
      if (dy == nullptr) continue;
      const float* dyt = dy + t * d;
      for (int i = 0; i < d; ++i) dg[static_cast<std::size_t>(j)] += dyt[i] * o[static_cast<std::size_t>(i)];
      std::vector<double> da(static_cast<std::size_t>(F), 0.0), dh(static_cast<std::size_t>(2 * F));
      for (int c = 0; c < F; ++c) {
        for (int i = 0; i < d; ++i) {
          const double go = g[static_cast<std::size_t>(j)] * dyt[i];
          da[static_cast<std::size_t>(c)] += go * Dn[c * d + i];
          R.ddown[static_cast<std::size_t>(e) * F * d + static_cast<std::size_t>(c) * d + i] += a[static_cast<std::size_t>(c)] * go;
        }
      }
      for (int c = 0; c < F; ++c) {
        const double hv = h[static_cast<std::size_t>(c)], sg = 1.0 / (1.0 + std::exp(-hv));
        const double fv = hv * sg, df = sg * (1.0 + hv * (1.0 - sg));
        dh[static_cast<std::size_t>(c)] = da[static_cast<std::size_t>(c)] * h[static_cast<std::size_t>(F + c)] * df;
        dh[static_cast<std::size_t>(F + c)] = da[static_cast<std::size_t>(c)] * fv;
      }
      for (int i = 0; i < d; ++i) {
        for (int c = 0; c < 2 * F; ++c) {
          R.dup[static_cast<std::size_t>(e) * d * 2 * F + static_cast<std::size_t>(i) * 2 * F + c] += static_cast<double>(xt[i]) * dh[static_cast<std::size_t>(c)];
          dxt[static_cast<std::size_t>(i)] += dh[static_cast<std::size_t>(c)] * U[i * 2 * F + c];
        }
      }
    }
    if (dy == nullptr) continue;
    // kapi -> logit
    std::vector<double> dl(static_cast<std::size_t>(E), 0.0);
    double sd = 0.0;
    for (int j = 0; j < k; ++j) sd += g[static_cast<std::size_t>(j)] * dg[static_cast<std::size_t>(j)];
    for (int j = 0; j < k; ++j) dl[static_cast<std::size_t>(sel[static_cast<std::size_t>(j)])] += g[static_cast<std::size_t>(j)] * (dg[static_cast<std::size_t>(j)] - sd);
    double pf = 0.0;
    for (int e = 0; e < E; ++e) pf += p[static_cast<std::size_t>(e)] * f[static_cast<std::size_t>(e)];
    for (int e = 0; e < E; ++e) {
      dl[static_cast<std::size_t>(e)] += rlg * (s.aux_alpha * E / static_cast<double>(N) * p[static_cast<std::size_t>(e)] * (f[static_cast<std::size_t>(e)] - pf) +
                                                s.z_alpha * 2.0 * lse / static_cast<double>(N) * p[static_cast<std::size_t>(e)]);
    }
    for (int i = 0; i < d; ++i) {
      for (int e = 0; e < E; ++e) {
        R.drouter[static_cast<std::size_t>(i) * E + e] += static_cast<double>(xt[i]) * dl[static_cast<std::size_t>(e)];
        dxt[static_cast<std::size_t>(i)] += dl[static_cast<std::size_t>(e)] * Wg[i * E + e];
      }
    }
    for (int i = 0; i < d; ++i) R.dx[static_cast<std::size_t>(t * d + i)] = dxt[static_cast<std::size_t>(i)];
  }
  double aux = 0.0;
  for (int e = 0; e < E; ++e) aux += f[static_cast<std::size_t>(e)] * P[static_cast<std::size_t>(e)];
  R.aux = s.aux_alpha * E * aux;
  R.z = s.z_alpha * zs / N;
  return R;
}

double active_mlp_flops_per_token(const Spec& s) {
  return 2.0 * s.d_model * s.experts + static_cast<double>(s.top_k) * (2.0 * s.d_model * 2 * s.d_ff + 2.0 * s.d_ff * s.d_model);
}

}  // namespace moe
}  // namespace nexus
