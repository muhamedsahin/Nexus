/**
 * @file test_phase14_moe.cpp
 * @brief FAZ 14: MoE — yonlendirici, kapasite, gruplu GEMM, lm entegrasyonu, yuk dengeleme, EP, dar uzman agirliklari.
 *
 *   test_phase14_moe router   : test_moe_router_deterministic — ayni giris -> ayni top-k, kapi, kapasite atamasi
 *                               (gurultulu dahil); kapasite / dusurme / reroute kurallari
 *   test_phase14_moe k1       : test_moe_k1_equals_dense — E=1, k=1 lm modeli yogun SwiGLU ile kayip + gradyan birebir
 *   test_phase14_moe grad     : test_moe_grad_numerics — optimize yol vs bagimsiz double referans (permutasyonsuz) <= 1e-5
 *   test_phase14_moe grouped  : test_grouped_gemm_vs_loop — paralel gruplu = sirali uzman dongusu (bit bit), daha hizli;
 *                               token basina dongu ile ayni sonuc
 *   test_phase14_moe balance  : test_load_balance_in_range — bayt duzeyi TinyStories, 300 adim: aux kayipla her uzman
 *                               >= 0.5 / E (aux'suz karsilastirma raporlanir)
 *   test_phase14_moe ep       : test_ep_equivalence — 1 rank vs 2/4/8 rank EP: y, dx, uzman gradyanlari bit bit;
 *                               yonlendirici gradyani <= 1e-6, aux ayni
 *   test_phase14_moe quant    : BF16 / FP8 uzman agirligi: bellek 1/2, 1/4; cikis hatasi sinirli
 *   test_phase14_moe config   : moe-124m on tanimli (~124M toplam), JSON gidis-donus, NXS kayit/yukleme
 *   test_phase14_moe bench ...: MoE-124M vs ayni aktif FLOP'lu yogun egitim egrisi, gruplu GEMM olcumu, EP iletisimi
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/dist/moe_ep.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus/lm/model.hpp"
#include "nexus/moe/moe.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ag = nexus::autograd;
namespace lm = nexus::lm;
namespace moe = nexus::moe;
namespace nd = nexus::dist;
namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::vector<float> randn(std::size_t n, std::uint32_t seed, float scale) {
  std::vector<float> v(n);
  std::uint32_t s = seed * 2654435761u + 1u;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    const float u1 = (static_cast<float>(s >> 8) + 0.5f) / 16777216.0f;
    s = s * 1664525u + 1013904223u;
    const float u2 = (static_cast<float>(s >> 8) + 0.5f) / 16777216.0f;
    x = scale * std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
  }
  return v;
}

template <typename A, typename B>
double rel(const A& a, const B& b) {
  double m = 0.0, r = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    r = std::max(r, std::fabs(static_cast<double>(b[i])));
  }
  return m / std::max(r, 1e-30);
}

moe::Spec small_spec(int E, int k, int d, int F) {
  moe::Spec s;
  s.experts = E;
  s.top_k = k;
  s.d_model = d;
  s.d_ff = F;
  return s;
}

// ---- bayt duzeyi TinyStories -------------------------------------------------------------
std::string tiny_root() { return "C:/Users/muham/nexus-data/tinystories/"; }

std::vector<std::uint8_t> load_bytes(const std::string& path, std::size_t max_bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::vector<std::uint8_t> v(max_bytes);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(max_bytes));
  v.resize(static_cast<std::size_t>(f.gcount()));
  return v;
}

std::vector<std::int32_t> sample(const std::vector<std::uint8_t>& text, int B, int T, std::uint64_t step, std::uint64_t seed) {
  std::vector<std::int32_t> t(static_cast<std::size_t>(B) * T);
  std::uint64_t s = seed ^ (step * 0x9E3779B97F4A7C15ULL);
  for (int b = 0; b < B; ++b) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const std::size_t off = static_cast<std::size_t>((s >> 17) % (text.size() - static_cast<std::size_t>(T) - 1));
    for (int i = 0; i < T; ++i) t[static_cast<std::size_t>(b) * T + i] = text[off + static_cast<std::size_t>(i)];
  }
  return t;
}

// ==========================================================================================

void test_router() {
  const int N = 96, d = 16, E = 8, k = 2;
  const auto x = randn(static_cast<std::size_t>(N) * d, 1, 1.0f);
  const auto w = randn(static_cast<std::size_t>(d) * E, 2, 0.5f);
  const auto wn = randn(static_cast<std::size_t>(d) * E, 3, 0.3f);
  moe::Spec s = small_spec(E, k, d, 8);
  s.capacity_factor = 0.75f;  // dusurme olsun
  s.min_capacity = 1;
  s.groups = 2;
  s.noisy = true;
  for (bool noisy : {false, true}) {
    const moe::Routing a = moe::route(x.data(), w.data(), noisy ? wn.data() : nullptr, N, s, true, 42);
    const moe::Routing b = moe::route(x.data(), w.data(), noisy ? wn.data() : nullptr, N, s, true, 42);
    const bool same = a.expert == b.expert && a.gate == b.gate && a.row == b.row && a.offsets == b.offsets && a.aux == b.aux;
    nexus_test::expect(same, std::string("ayni giris -> ayni top-k, kapi ve kapasite atamasi") + (noisy ? " (gurultulu, ayni tohum)" : ""));
    // kurallar: grup basina uzman <= C; dusen sayisi = toplam atama - kabul
    bool cap_ok = true;
    for (int g = 0; g < s.groups; ++g) {
      std::vector<int> per(static_cast<std::size_t>(E), 0);
      for (int t = g * N / 2; t < (g + 1) * N / 2; ++t)
        for (int j = 0; j < k; ++j)
          if (a.row[static_cast<std::size_t>(t * k + j)] >= 0) per[static_cast<std::size_t>(a.expert[static_cast<std::size_t>(t * k + j)])] += 1;
      for (int c : per) cap_ok = cap_ok && c <= a.capacity;
    }
    std::printf("  info  %s: kapasite %lld (grup basina), dusen %lld / %d, bos uzman %d, aux %.4f, z %.4f\n",
                noisy ? "gurultulu" : "temiz", static_cast<long long>(a.capacity), static_cast<long long>(a.dropped), N * k,
                a.idle_experts(), a.aux, a.z);
    nexus_test::expect(cap_ok && a.dropped == N * k - a.rows() && a.dropped > 0, "kapasite siniri ve dusurme sayimi tutarli");
  }
  const moe::Routing c = moe::route(x.data(), w.data(), wn.data(), N, s, true, 43);
  nexus_test::expect(c.expert != moe::route(x.data(), w.data(), wn.data(), N, s, true, 42).expert,
                     "gurultu tohumu degisince secim degisir (kesif); egitim disinda gurultu yok");
  const moe::Routing ev1 = moe::route(x.data(), w.data(), wn.data(), N, s, false, 42);
  const moe::Routing ev2 = moe::route(x.data(), w.data(), wn.data(), N, s, false, 7);
  nexus_test::expect(ev1.expert == ev2.expert, "cikarimda (training = false) gurultu kapali: tohumdan bagimsiz");
  s.noisy = false;
  s.overflow = moe::Overflow::Reroute;
  const moe::Routing rr = moe::route(x.data(), w.data(), nullptr, N, s, true, 0);
  s.overflow = moe::Overflow::Drop;
  const moe::Routing dd = moe::route(x.data(), w.data(), nullptr, N, s, true, 0);
  std::printf("  info  reroute: %lld yeniden yonlendirildi, %lld dustu (drop modunda %lld)\n", static_cast<long long>(rr.rerouted),
              static_cast<long long>(rr.dropped), static_cast<long long>(dd.dropped));
  nexus_test::expect(rr.rerouted > 0 && rr.dropped < dd.dropped, "taşma (reroute) dusurmeyi azaltir");
}

void test_k1() {
  lm::LmConfig dense = lm::preset("llama2-7b");
  dense.n_layer = 2;
  dense.d_model = 32;
  dense.n_head = 4;
  dense.n_kv_head = 4;
  dense.d_ff = 48;
  dense.vocab_size = 97;
  dense.max_position = 64;
  lm::LmConfig mc = dense;
  mc.moe_experts = 1;
  mc.moe_top_k = 1;
  mc.moe_capacity_factor = 4.0f;
  mc.moe_aux_alpha = 0.0f;
  mc.moe_z_alpha = 0.0f;
  const lm::LanguageModel md(dense), mm(mc);
  // yogun MLP agirliklarini tek uzmana kopyala, diger parametreler ayni ada sahip
  for (const auto& p : mm.parameters()) {
    std::string src = p.name;
    if (src.find("moe.experts.up.weight") != std::string::npos) src.replace(src.find("moe.experts.up.weight"), 21, "mlp.up.weight");
    if (src.find("moe.experts.down.weight") != std::string::npos) src.replace(src.find("moe.experts.down.weight"), 23, "mlp.down.weight");
    const lm::NamedParam* q = md.find(src);
    if (q) const_cast<ag::Variable&>(p.value).mutable_values() = q->value.values();
  }
  const auto tok = randn(64, 5, 1.0f);
  std::vector<std::int32_t> t(64);
  for (std::size_t i = 0; i < t.size(); ++i) t[i] = static_cast<std::int32_t>(std::fabs(tok[i]) * 30) % 97;
  const ag::Variable ld = md.loss(t, 2, 32);
  ld.backward();
  const ag::Variable lmo = mm.loss(t, 2, 32);
  lmo.backward();
  double worst = 0.0;
  std::string wn;
  for (const auto& p : mm.parameters()) {
    if (p.name.find("router") != std::string::npos) continue;  // asagida ayrica: tam 0 olmali
    std::string src = p.name;
    if (src.find("moe.experts.up.weight") != std::string::npos) src.replace(src.find("moe.experts.up.weight"), 21, "mlp.up.weight");
    if (src.find("moe.experts.down.weight") != std::string::npos) src.replace(src.find("moe.experts.down.weight"), 23, "mlp.down.weight");
    const double r = rel(p.value.grad().values(), md.find(src)->value.grad().values());
    if (r > worst) {
      worst = r;
      wn = p.name;
    }
  }
  double router_abs = 0.0;
  for (const auto& p : mm.parameters()) {
    if (p.name.find("router") == std::string::npos) continue;
    for (float g : p.value.grad().values()) router_abs = std::max(router_abs, static_cast<double>(std::fabs(g)));
  }
  std::printf("  info  k=1, E=1: kayip yogun %.8f MoE %.8f; en kotu gradyan goreli %.2e (%s); yonlendirici |grad| %.1e\n",
              ld.values()[0], lmo.values()[0], worst, wn.c_str(), router_abs);
  nexus_test::expect(ld.values()[0] == lmo.values()[0] && worst == 0.0 && router_abs == 0.0,
                     "k=1/E=1 MoE == yogun SwiGLU MLP (kayip ve tum gradyanlar bit bit; kapi = 1, yonlendirici gradyani 0)");
}

void test_grad() {
  for (int mode = 0; mode < 2; ++mode) {
    const int N = 64, d = 24, E = 8, k = 2, F = 20;
    moe::Spec s = small_spec(E, k, d, F);
    s.groups = 2;
    s.capacity_factor = 0.8f;
    s.min_capacity = 1;
    s.overflow = mode == 0 ? moe::Overflow::Drop : moe::Overflow::Reroute;
    s.aux_alpha = 0.05f;
    s.z_alpha = 0.01f;
    const auto x = randn(static_cast<std::size_t>(N) * d, 11, 1.0f);
    const auto w = randn(static_cast<std::size_t>(d) * E, 12, 0.3f);
    const auto up = randn(static_cast<std::size_t>(E) * d * 2 * F, 13, 0.2f);
    const auto dn = randn(static_cast<std::size_t>(E) * F * d, 14, 0.2f);
    const auto dy = randn(static_cast<std::size_t>(N) * d, 15, 1.0f);
    moe::LayerCache c;
    std::vector<float> y(static_cast<std::size_t>(N) * d), dx(y.size()), dr(w.size(), 0.0f), du(up.size(), 0.0f), dd(dn.size(), 0.0f);
    moe::layer_forward(x.data(), N, w.data(), nullptr, up.data(), dn.data(), s, true, 0, y.data(), &c);
    moe::layer_backward(dy.data(), c, w.data(), nullptr, up.data(), dn.data(), s, 1.0f, dx.data(), dr.data(), nullptr, du.data(), dd.data());
    const moe::Reference R = moe::reference(x.data(), N, w.data(), up.data(), dn.data(), c.routing, s, dy.data(), 1.0f);
    const double ey = rel(y, R.y), ex = rel(dx, R.dx), er = rel(dr, R.drouter), eu = rel(du, R.dup), ed = rel(dd, R.ddown);
    const double ea = std::fabs(c.routing.aux - R.aux) / std::max(1e-12, std::fabs(R.aux));
    const double ez = std::fabs(c.routing.z - R.z) / std::max(1e-12, std::fabs(R.z));
    std::printf("  info  %s (dusen %lld, yeniden %lld): goreli y %.1e dx %.1e drouter %.1e dup %.1e ddown %.1e aux %.1e z %.1e\n",
                mode == 0 ? "drop" : "reroute", static_cast<long long>(c.routing.dropped), static_cast<long long>(c.routing.rerouted),
                ey, ex, er, eu, ed, ea, ez);
    nexus_test::expect(std::max({ey, ex, er, eu, ed, ea, ez}) <= 1e-5,
                       std::string("MoE ileri + geri == bagimsiz double referans (<= 1e-5), ") + (mode == 0 ? "drop" : "reroute"));
  }
}

void test_grouped() {
  const int N = 4096, d = 256, E = 8, k = 2, F = 512;
  moe::Spec s = small_spec(E, k, d, F);
  s.capacity_factor = 1.25f;
  const auto x = randn(static_cast<std::size_t>(N) * d, 21, 1.0f);
  const auto w = randn(static_cast<std::size_t>(d) * E, 22, 0.1f);
  const auto up = randn(static_cast<std::size_t>(E) * d * 2 * F, 23, 0.05f);
  const auto dn = randn(static_cast<std::size_t>(E) * F * d, 24, 0.05f);
  const moe::Routing r = moe::route(x.data(), w.data(), nullptr, N, s, true, 0);
  const std::int64_t M = r.rows();
  std::vector<float> xp(static_cast<std::size_t>(M) * d);
  for (std::int64_t row = 0; row < M; ++row) std::copy_n(x.data() + static_cast<std::int64_t>(r.perm_token[static_cast<std::size_t>(row)]) * d, d, xp.data() + row * d);
  std::vector<float> y1(static_cast<std::size_t>(M) * d), yn(y1.size());
  double tl = 1e30, tg = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    auto t0 = Clock::now();
    moe::experts_forward(xp.data(), r.offsets.data(), E, up.data(), dn.data(), d, F, y1.data(), nullptr, 1);
    tl = std::min(tl, ms_since(t0));
    t0 = Clock::now();
    moe::experts_forward(xp.data(), r.offsets.data(), E, up.data(), dn.data(), d, F, yn.data(), nullptr, 0);
    tg = std::min(tg, ms_since(t0));
  }
  // token basina dongu (gemv; tek tek): ayni sonuc, olcum icin ilk 256 satir
  const std::int64_t probe = std::min<std::int64_t>(256, M);
  std::vector<float> yt(static_cast<std::size_t>(probe) * d);
  auto t0 = Clock::now();
  std::vector<std::int64_t> one(static_cast<std::size_t>(E) + 1, 0);
  for (std::int64_t row = 0; row < probe; ++row) {
    int e = 0;
    while (r.offsets[static_cast<std::size_t>(e) + 1] <= row) ++e;
    std::fill(one.begin(), one.end(), 0);
    for (int q = e + 1; q <= E; ++q) one[static_cast<std::size_t>(q)] = 1;
    moe::experts_forward(xp.data() + row * d, one.data(), E, up.data(), dn.data(), d, F, yt.data() + row * d, nullptr, 1);
  }
  const double tt = ms_since(t0) * static_cast<double>(M) / static_cast<double>(probe);
  double diff = 0.0;
  for (std::size_t i = 0; i < yt.size(); ++i) diff = std::max(diff, static_cast<double>(std::fabs(yt[i] - y1[i])));
  const double flops = 2.0 * static_cast<double>(M) * (d * 2.0 * F + F * static_cast<double>(d));
  std::printf("  info  gruplu GEMM (E=%d, %lld satir, d=%d, F=%d): sirali uzman dongusu %.1f ms (%.1f GFLOP/s), paralel gruplu %.1f ms "
              "(%.1f GFLOP/s, %.2fx), token basina dongu ~%.0f ms (tahmini, %lld satirdan); token dongusu farki %.1e\n",
              E, static_cast<long long>(M), d, F, tl, flops / tl / 1e6, tg, flops / tg / 1e6, tl / tg, tt,
              static_cast<long long>(probe), diff);
  nexus_test::expect(y1 == yn, "paralel gruplu GEMM == sirali uzman dongusu (bit bit)");
#if defined(NEXUS_TEST_WITH_ASAN)
  // Instrumentation changes thread and memory costs. Keep all numerical checks,
  // but do not turn an ASan timing into a production performance acceptance gate.
  std::printf("  info  performance_status=no_data (ASan instrumentation; use a stable Release benchmark)\n");
#else
  nexus_test::expect(tg < tl, "paralel gruplu GEMM sirali donguden hizli");
#endif
  nexus_test::expect(diff == 0.0, "token basina dongu ile ayni sonuc (satir bagimsiz)");
}

struct TrainResult {
  std::vector<double> losses;
  std::vector<std::vector<double>> usage;  // katman -> uzman payi (son 20 adim ortalamasi)
  double ms_per_step = 0.0;
};

TrainResult train_bytes(const lm::LmConfig& c, int steps, int B, int T, double lr, const std::vector<std::uint8_t>& text,
                        bool quiet = true) {
  lm::LanguageModel model(c);
  nexus::fused::AdamWConfig ac;
  ac.lr = lr;
  ac.beta2 = 0.95;
  ac.weight_decay = 0.1;
  ac.max_grad_norm = 1.0;
  lm::AdamW opt(model, ac);
  TrainResult r;
  std::vector<std::vector<double>> acc;
  int nacc = 0;
  const auto t0 = Clock::now();
  for (int s = 0; s < steps; ++s) {
    const auto tok = sample(text, B, T, static_cast<std::uint64_t>(s), 1234);
    lm::ForwardOptions fo;
    fo.training = true;
    fo.dropout_seed = static_cast<std::uint64_t>(s);
    model.zero_grad();
    const ag::Variable loss = model.loss(tok, B, T, fo);
    loss.backward();
    (void)opt.step();
    r.losses.push_back(loss.values()[0]);
    if (s >= steps - 20) {
      const auto& st = model.moe_stats();
      if (acc.empty()) acc.assign(st.size(), std::vector<double>(static_cast<std::size_t>(c.moe_experts), 0.0));
      for (std::size_t l = 0; l < st.size(); ++l) {
        double tot = 0.0;
        for (auto v : st[l].kept) tot += static_cast<double>(v);
        for (int e = 0; e < c.moe_experts; ++e) acc[l][static_cast<std::size_t>(e)] += static_cast<double>(st[l].kept[static_cast<std::size_t>(e)]) / tot;
      }
      nacc += 1;
    }
    if (!quiet && (s % 25 == 0 || s + 1 == steps)) std::printf("    adim %4d kayip %.4f\n", s, r.losses.back());
  }
  r.ms_per_step = ms_since(t0) / steps;
  for (auto& l : acc)
    for (auto& v : l) v /= std::max(1, nacc);
  r.usage = acc;
  return r;
}

void test_balance() {
  const auto text = load_bytes(tiny_root() + "TinyStoriesV2-GPT4-valid.txt", 8u << 20);
  if (text.size() < 1000000) {
    std::printf("  SKIP  TinyStories yok (%s)\n", tiny_root().c_str());
    std::exit(77);
  }
  lm::LmConfig c = lm::preset("moe-124m");
  c.vocab_size = 256;
  c.n_layer = 2;
  c.d_model = 64;
  c.n_head = 4;
  c.n_kv_head = 4;
  c.d_ff = 96;
  c.max_position = 128;
  for (float alpha : {0.0f, 0.01f}) {
    c.moe_aux_alpha = alpha;
    const TrainResult r = train_bytes(c, 300, 8, 64, 3e-3, text);
    double lo = 1.0, hi = 0.0;
    for (const auto& l : r.usage)
      for (double v : l) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
    std::printf("  info  aux_alpha=%.2f: kayip %.3f -> %.3f; uzman payi min %.3f max %.3f (esit pay %.3f); %.0f ms/adim\n", alpha,
                r.losses.front(), r.losses.back(), lo, hi, 1.0 / c.moe_experts, r.ms_per_step);
    if (alpha > 0.0f) {
      nexus_test::expect(lo >= 0.5 / c.moe_experts, "aux kayipla her uzmanin payi >= 0.5 / E (son 20 adim)");
      nexus_test::expect(r.losses.back() < r.losses.front() * 0.6, "MoE modeli ogreniyor (kayip %40+ duser)");
    }
  }
}

void test_ep() {
  for (int ep : {2, 4, 8}) {
    const int Nr = 48, d = 32, E = 8, k = 2, F = 40;
    moe::Spec s = small_spec(E, k, d, F);
    s.capacity_factor = 0.9f;
    s.min_capacity = 1;
    s.aux_alpha = 0.02f;
    s.z_alpha = 0.002f;
    const std::int64_t N = static_cast<std::int64_t>(Nr) * ep;
    const auto x = randn(static_cast<std::size_t>(N) * d, 31, 1.0f);
    const auto w = randn(static_cast<std::size_t>(d) * E, 32, 0.3f);
    const auto up = randn(static_cast<std::size_t>(E) * d * 2 * F, 33, 0.2f);
    const auto dn = randn(static_cast<std::size_t>(E) * F * d, 34, 0.2f);
    const auto dy = randn(static_cast<std::size_t>(N) * d, 35, 1.0f);
    // tek rank: rank basina bir grup
    moe::Spec s1 = s;
    s1.groups = ep;
    moe::LayerCache c1;
    std::vector<float> y1(static_cast<std::size_t>(N) * d), dx1(y1.size()), dr1(w.size(), 0.0f), du1(up.size(), 0.0f), dd1(dn.size(), 0.0f);
    moe::layer_forward(x.data(), N, w.data(), nullptr, up.data(), dn.data(), s1, true, 0, y1.data(), &c1);
    moe::layer_backward(dy.data(), c1, w.data(), nullptr, up.data(), dn.data(), s1, 1.0f, dx1.data(), dr1.data(), nullptr, du1.data(), dd1.data());
    std::vector<float> y(y1.size()), dx(y1.size()), dr(w.size(), 0.0f), du(up.size(), 0.0f), dd(dn.size(), 0.0f);
    float aux = 0.0f;
    double a2a = 0.0, comp = 0.0;
    std::uint64_t bytes = 0;
    std::mutex mu;
    nd::ThreadWorld world(ep);
    std::vector<std::thread> th;
    for (int r = 0; r < ep; ++r) {
      th.emplace_back([&, r] {
        nd::GroupPtr g = world.create_group(r);
        nd::MoeExpertParallel m(g, s);
        const int L = m.local_experts();
        const std::size_t wu = static_cast<std::size_t>(d) * 2 * F, wd = static_cast<std::size_t>(F) * d;
        std::vector<float> ul(up.begin() + static_cast<std::ptrdiff_t>(m.first_expert() * wu), up.begin() + static_cast<std::ptrdiff_t>((m.first_expert() + L) * wu));
        std::vector<float> dl(dn.begin() + static_cast<std::ptrdiff_t>(m.first_expert() * wd), dn.begin() + static_cast<std::ptrdiff_t>((m.first_expert() + L) * wd));
        std::vector<float> yl(static_cast<std::size_t>(Nr) * d), dxl(yl.size()), drl(w.size(), 0.0f), dul(ul.size(), 0.0f), ddl(dl.size(), 0.0f);
        m.forward(x.data() + static_cast<std::size_t>(r) * Nr * d, Nr, w.data(), ul.data(), dl.data(), true, 0, yl.data());
        m.backward(dy.data() + static_cast<std::size_t>(r) * Nr * d, w.data(), ul.data(), dl.data(), 1.0f, dxl.data(), drl.data(), dul.data(), ddl.data());
        g->all_reduce(drl.data(), drl.size(), nexus::DType::F32, nd::ReduceOp::Sum)->wait();
        std::lock_guard<std::mutex> lock(mu);
        std::copy(yl.begin(), yl.end(), y.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * Nr * d));
        std::copy(dxl.begin(), dxl.end(), dx.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * Nr * d));
        std::copy(dul.begin(), dul.end(), du.begin() + static_cast<std::ptrdiff_t>(m.first_expert() * wu));
        std::copy(ddl.begin(), ddl.end(), dd.begin() + static_cast<std::ptrdiff_t>(m.first_expert() * wd));
        if (r == 0) {
          dr = drl;
          aux = m.routing().aux + m.routing().z;
        }
        a2a = std::max(a2a, m.stats().a2a_ms);
        comp = std::max(comp, m.stats().compute_ms);
        bytes += m.stats().bytes_sent;
      });
    }
    for (auto& t : th) t.join();
    const double er = rel(dr, dr1);
    std::printf("  info  EP %d rank: y %s, dx %s, dup %s, ddown %s; drouter goreli %.1e; aux+z %.6f vs %.6f; a2a %.2f ms, hesap %.2f ms, "
                "gonderilen %.1f KB\n",
                ep, y == y1 ? "bit bit" : "FARKLI", dx == dx1 ? "bit bit" : "FARKLI", du == du1 ? "bit bit" : "FARKLI",
                dd == dd1 ? "bit bit" : "FARKLI", er, aux, c1.routing.aux + c1.routing.z, a2a, comp, bytes / 1024.0);
    nexus_test::expect(y == y1 && dx == dx1 && du == du1 && dd == dd1 && er <= 1e-6 &&
                           std::fabs(aux - (c1.routing.aux + c1.routing.z)) <= 1e-6f * std::max(1.0f, std::fabs(aux)),
                       "test_ep_equivalence: 1 rank == " + std::to_string(ep) + " rank EP (cikti, gradyanlar, aux)");
  }
}

void test_quant() {
  const int N = 512, d = 128, E = 8, k = 2, F = 256;
  moe::Spec s = small_spec(E, k, d, F);
  const auto x = randn(static_cast<std::size_t>(N) * d, 41, 1.0f);
  const auto w = randn(static_cast<std::size_t>(d) * E, 42, 0.2f);
  const auto up = randn(static_cast<std::size_t>(E) * d * 2 * F, 43, 0.05f);
  const auto dn = randn(static_cast<std::size_t>(E) * F * d, 44, 0.05f);
  const moe::Routing r = moe::route(x.data(), w.data(), nullptr, N, s, false, 0);
  const std::int64_t M = r.rows();
  std::vector<float> xp(static_cast<std::size_t>(M) * d), y32(xp.size()), yq(xp.size());
  for (std::int64_t row = 0; row < M; ++row) std::copy_n(x.data() + static_cast<std::int64_t>(r.perm_token[static_cast<std::size_t>(row)]) * d, d, xp.data() + row * d);
  moe::experts_forward(xp.data(), r.offsets.data(), E, up.data(), dn.data(), d, F, y32.data(), nullptr, 0);
  const double fbytes = static_cast<double>(up.size() + dn.size()) * 4;
  for (moe::WeightType t : {moe::WeightType::BF16, moe::WeightType::FP8}) {
    const moe::QuantizedExperts q = moe::quantize_experts(up.data(), dn.data(), E, d, F, t);
    moe::experts_forward_quantized(xp.data(), r.offsets.data(), q, yq.data(), 0);
    const double e = rel(yq, y32);
    std::printf("  info  %s uzman agirligi: bellek %.2f MB (FP32 %.2f MB, %.2fx), cikis goreli hatasi %.2e\n",
                t == moe::WeightType::BF16 ? "BF16" : "FP8-E4M3", q.bytes() / 1e6, fbytes / 1e6, fbytes / q.bytes(), e);
    nexus_test::expect(t == moe::WeightType::BF16 ? (e < 1e-2 && fbytes / q.bytes() > 1.99) : (e < 8e-2 && fbytes / q.bytes() > 3.9),
                       std::string(t == moe::WeightType::BF16 ? "BF16" : "FP8") + " uzman agirligi: bellek kazanci ve sinirli hata");
  }
}

void test_config() {
  const lm::LmConfig c = lm::preset("moe-124m");
  const std::int64_t total = lm::count_parameters(c), active = lm::count_active_parameters(c);
  std::printf("  info  moe-124m: toplam %.1fM, aktif %.1fM (%.0f%%); E=%d k=%d, %d katman, d %d, uzman F %d, sozluk %d\n", total / 1e6,
              active / 1e6, 100.0 * active / total, c.moe_experts, c.moe_top_k, c.n_layer, c.d_model, c.expert_ff(), c.vocab_size);
  nexus_test::expect(total > 115e6 && total < 130e6 && active < total / 2, "moe-124m: ~124M toplam, aktif < toplamin yarisi");
  const lm::LmConfig back = lm::from_json(lm::to_json(c));
  nexus_test::expect(lm::to_json(back) == lm::to_json(c) && back.moe_experts == 8, "MoE alanlari JSON gidis-donus");
  lm::LmConfig sm = c;
  sm.n_layer = 2;
  sm.d_model = 32;
  sm.n_head = 4;
  sm.n_kv_head = 4;
  sm.d_ff = 32;
  sm.vocab_size = 64;
  sm.moe_shared_expert = true;
  sm.moe_noisy = true;
  sm.moe_every = 2;
  const lm::LanguageModel m(sm);
  std::int64_t counted = 0;
  for (const auto& p : m.parameters()) counted += p.value.numel();
  const fs::path path = fs::temp_directory_path() / "nexus_faz14_moe.nxs";
  m.save_nxs(path.string());
  const lm::LanguageModel m2 = lm::LanguageModel::load_nxs(path.string());
  std::vector<std::int32_t> t(32);
  for (int i = 0; i < 32; ++i) t[static_cast<std::size_t>(i)] = (i * 7) % 64;
  const float a = m.loss(t, 1, 32).values()[0], b = m2.loss(t, 1, 32).values()[0];
  std::printf("  info  paylasimli uzman + gurultu + moe_every=2: %lld parametre (count_parameters %lld); NXS kayip %.6f / %.6f\n",
              static_cast<long long>(counted), static_cast<long long>(lm::count_parameters(sm)), a, b);
  nexus_test::expect(counted == lm::count_parameters(sm) && a == b, "count_parameters dogru; NXS kayit/yukleme ayni kayip (tur 36 MoEFFN)");
}

// EP iletisim olcumu (thread ProcessGroup, tek makine): rank basina N token, moe-124m katman boyutlari.
int ep_bench(int Nr) {
  const int d = 512, E = 8, k = 2, F = 1856;
  for (int ep : {2, 4, 8}) {
    moe::Spec s = small_spec(E, k, d, F);
    s.threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / ep);
    const auto w = randn(static_cast<std::size_t>(d) * E, 52, 0.05f);
    const std::size_t wu = static_cast<std::size_t>(d) * 2 * F, wd = static_cast<std::size_t>(F) * d;
    const int L = E / ep;
    const auto up = randn(L * wu, 53, 0.02f);
    const auto dn = randn(L * wd, 54, 0.02f);
    std::vector<double> a2a(static_cast<std::size_t>(ep)), comp(static_cast<std::size_t>(ep)), total(static_cast<std::size_t>(ep));
    std::vector<std::uint64_t> bytes(static_cast<std::size_t>(ep));
    nd::ThreadWorld world(ep);
    std::vector<std::thread> th;
    for (int r = 0; r < ep; ++r) {
      th.emplace_back([&, r] {
        nd::GroupPtr g = world.create_group(r);
        nd::MoeExpertParallel m(g, s);
        const auto x = randn(static_cast<std::size_t>(Nr) * d, 60 + static_cast<std::uint32_t>(r), 1.0f);
        const auto dy = randn(static_cast<std::size_t>(Nr) * d, 70 + static_cast<std::uint32_t>(r), 1.0f);
        std::vector<float> y(x.size()), dx(x.size()), dr(w.size(), 0.0f), du(up.size(), 0.0f), dd(dn.size(), 0.0f);
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
          g->barrier()->wait();
          const auto t0 = Clock::now();
          m.forward(x.data(), Nr, w.data(), up.data(), dn.data(), true, 0, y.data());
          m.backward(dy.data(), w.data(), up.data(), dn.data(), 1.0f, dx.data(), dr.data(), du.data(), dd.data());
          const double t = ms_since(t0);
          if (t < best) {
            best = t;
            a2a[static_cast<std::size_t>(r)] = m.stats().a2a_ms;
            comp[static_cast<std::size_t>(r)] = m.stats().compute_ms;
            bytes[static_cast<std::size_t>(r)] = m.stats().bytes_sent;
          }
        }
        total[static_cast<std::size_t>(r)] = best;
      });
    }
    for (auto& t : th) t.join();
    double A = 0, C = 0, T = 0;
    std::uint64_t Bt = 0;
    for (int r = 0; r < ep; ++r) {
      A = std::max(A, a2a[static_cast<std::size_t>(r)]);
      C = std::max(C, comp[static_cast<std::size_t>(r)]);
      T = std::max(T, total[static_cast<std::size_t>(r)]);
      Bt += bytes[static_cast<std::size_t>(r)];
    }
    std::printf("{\"mode\": \"bench_moe_ep\", \"ep\": %d, \"tokens_per_rank\": %d, \"step_ms\": %.1f, \"a2a_ms\": %.1f, "
                "\"compute_ms\": %.1f, \"a2a_share\": %.3f, \"bytes_total\": %llu, \"a2a_gb_s\": %.2f, \"overlap\": 0}\n",
                ep, Nr, T, A, C, A / T, static_cast<unsigned long long>(Bt), Bt / (A * 1e-3) / 1e9);
  }
  return 0;
}

int bench(int steps, int B, int T) {
  const auto train = load_bytes(tiny_root() + "TinyStoriesV2-GPT4-train.txt", 64u << 20);
  if (train.empty()) return 77;
  lm::LmConfig moec = lm::preset("moe-124m");
  moec.vocab_size = 256;
  moec.max_position = std::max(moec.max_position, T);
  lm::LmConfig dense = moec;  // ayni aktif FLOP: yogun d_ff = k * uzman F
  dense.moe_experts = 0;
  dense.d_ff = moec.moe_top_k * moec.expert_ff();
  for (const lm::LmConfig* c : {&moec, &dense}) {
    std::printf("== %s: toplam %.1fM, aktif %.1fM\n", c->moe_experts ? "MoE-124M (E=8, k=2)" : "yogun (ayni aktif)",
                lm::count_parameters(*c) / 1e6, lm::count_active_parameters(*c) / 1e6);
    const TrainResult r = train_bytes(*c, steps, B, T, 1e-3, train, false);
    double last = 0.0;
    for (int i = std::max(0, steps - 20); i < steps; ++i) last += r.losses[static_cast<std::size_t>(i)];
    last /= std::min(20, steps);
    std::printf("{\"mode\": \"bench_moe_train\", \"model\": \"%s\", \"total_params\": %lld, \"active_params\": %lld, \"steps\": %d, "
                "\"tokens_per_step\": %d, \"loss_first\": %.4f, \"loss_last20\": %.4f, \"ms_per_step\": %.0f, \"tokens_per_s\": %.0f",
                c->moe_experts ? "moe-124m" : "dense-same-active", static_cast<long long>(lm::count_parameters(*c)),
                static_cast<long long>(lm::count_active_parameters(*c)), steps, B * T, r.losses.front(), last, r.ms_per_step,
                B * T / (r.ms_per_step * 1e-3));
    if (!r.usage.empty()) {
      std::printf(", \"usage_min\": %.4f, \"usage_max\": %.4f", [&] { double m = 1; for (auto& l : r.usage) for (double v : l) m = std::min(m, v); return m; }(),
                  [&] { double m = 0; for (auto& l : r.usage) for (double v : l) m = std::max(m, v); return m; }());
    }
    std::printf(", \"curve\": [");
    for (int i = 0; i < steps; i += std::max(1, steps / 20)) std::printf("%s%.4f", i ? ", " : "", r.losses[static_cast<std::size_t>(i)]);
    std::printf("]}\n");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "router";
  try {
    if (mode == "router") test_router();
    else if (mode == "k1") test_k1();
    else if (mode == "grad") test_grad();
    else if (mode == "grouped") test_grouped();
    else if (mode == "balance") test_balance();
    else if (mode == "ep") test_ep();
    else if (mode == "quant") test_quant();
    else if (mode == "config") test_config();
    else if (mode == "epbench" && argc > 2) return ep_bench(std::stoi(argv[2]));
    else if (mode == "bench" && argc > 4) return bench(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]));
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase14_moe");
}
