/**
 * @file test_phase16_infer.cpp
 * @brief FAZ 16 (CPU): lm::LanguageModel artimli cozme (KV onbellegi) — tum mimari varyantlari.
 *
 *   test_phase16_infer kvcache  : test_kv_cache_matches_full_cpu — GPT-2 tipi (ogrenilen konum, LayerNorm, GELU,
 *                                 bias, bagli), LLaMA tipi (RoPE, RMSNorm, SwiGLU, GQA), ALiBi + kayan pencere,
 *                                 qk_norm + paralel blok, sinuzoidal, kismi RoPE + softcap, MoE: artimli (5 + parca +
 *                                 tek tek) logitler == tam ileri (<= 1e-4)
 *   test_phase16_infer generate : onbellekli greedy == her adimda tam yeniden hesap; token basina sure
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/lm/model.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace ag = nexus::autograd;
namespace lm = nexus::lm;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

lm::LmConfig small(lm::LmConfig c) {
  c.n_layer = 2;
  c.d_model = 48;
  c.n_head = 4;
  c.n_kv_head = c.n_kv_head == c.n_head || c.n_kv_head == 0 ? 4 : 2;
  c.head_dim = 0;
  c.d_ff = 96;
  c.vocab_size = 101;
  c.max_position = 64;
  c.init_std = 0.2f;  // belirgin logitler
  return c;
}

std::vector<std::int32_t> tokens(int n, std::uint32_t seed, int vocab) {
  std::vector<std::int32_t> t(static_cast<std::size_t>(n));
  std::uint32_t s = seed;
  for (auto& v : t) {
    s = s * 1664525u + 1013904223u;
    v = static_cast<std::int32_t>((s >> 8) % static_cast<std::uint32_t>(vocab));
  }
  return t;
}

std::vector<float> run(const lm::LanguageModel& m, const std::vector<std::int32_t>& t, lm::KvCache* cache) {
  lm::ForwardOptions o;
  o.cache = cache;
  return m.logits(t, 1, static_cast<int>(t.size()), o).values();
}

void test_kvcache() {
  struct Case {
    std::string name;
    lm::LmConfig c;
  };
  std::vector<Case> cases;
  cases.push_back({"gpt2 tipi", small(lm::preset("gpt2-124m"))});
  {
    lm::LmConfig c = small(lm::preset("llama2-7b"));
    c.n_kv_head = 2;
    cases.push_back({"llama tipi (GQA 4/2)", c});
  }
  {
    lm::LmConfig c = small(lm::preset("llama2-7b"));
    c.pos = lm::PosKind::Alibi;
    c.window = 9;
    cases.push_back({"ALiBi + pencere 9", c});
  }
  {
    lm::LmConfig c = small(lm::preset("llama2-7b"));
    c.qk_norm = true;
    c.parallel_block = true;
    cases.push_back({"qk_norm + paralel blok", c});
  }
  {
    lm::LmConfig c = small(lm::preset("gpt2-124m"));
    c.pos = lm::PosKind::Sinusoidal;
    cases.push_back({"sinuzoidal", c});
  }
  {
    lm::LmConfig c = small(lm::preset("llama2-7b"));
    c.rotary_dim = 6;
    c.attn_softcap = 5.0f;
    cases.push_back({"kismi RoPE + softcap", c});
  }
  {
    lm::LmConfig c = small(lm::preset("llama2-7b"));
    c.moe_experts = 4;
    c.moe_top_k = 2;
    c.moe_capacity_factor = 64.0f;  // dusurme yok: tam ileri ile ayni yonlendirme
    cases.push_back({"MoE (E=4, k=2)", c});
  }
  const int T = 24;
  for (const Case& cs : cases) {
    const lm::LanguageModel m(cs.c);
    const auto t = tokens(T, 7, cs.c.vocab_size);
    const std::vector<float> full = run(m, t, nullptr);
    const std::size_t V = full.size() / T;
    lm::KvCache cache = m.make_cache();
    std::vector<float> inc = run(m, std::vector<std::int32_t>(t.begin(), t.begin() + 5), &cache);
    const auto mid = run(m, std::vector<std::int32_t>(t.begin() + 5, t.begin() + 12), &cache);  // parca (Tq = 7)
    inc.insert(inc.end(), mid.begin(), mid.end());
    for (int i = 12; i < T; ++i) {
      const auto one = run(m, {t[static_cast<std::size_t>(i)]}, &cache);
      inc.insert(inc.end(), one.begin(), one.end());
    }
    double mx = 0.0, sc = 0.0;
    for (std::size_t i = 0; i < full.size(); ++i) {
      if (!std::isfinite(full[i])) continue;  // sozluk dolgusu
      mx = std::max(mx, std::fabs(static_cast<double>(full[i]) - inc[i]));
      sc = std::max(sc, std::fabs(static_cast<double>(full[i])));
    }
    std::printf("  info  %-24s artimli (5 + 7 + %d x 1) vs tam: goreli %.1e; onbellek %d token, %zu bayt (V=%zu)\n", cs.name.c_str(),
                T - 12, mx / sc, cache.length, cache.bytes(), V);
    nexus_test::expect(mx / sc <= 1e-4 && cache.length == T,
                       "test_kv_cache_matches_full_cpu: " + cs.name + " (<= 1e-4)");
  }
}

void test_generate() {
  lm::LmConfig c = small(lm::preset("llama2-7b"));
  c.n_layer = 4;
  c.d_model = 128;
  c.d_ff = 256;
  c.max_position = 256;
  const lm::LanguageModel m(c);
  const auto prompt = tokens(16, 3, c.vocab_size);
  const int gen = 48;
  auto argmax_last = [&](const std::vector<float>& lg, std::size_t rows) {
    const std::size_t V = lg.size() / rows;
    const float* r = lg.data() + (rows - 1) * V;
    return static_cast<std::int32_t>(std::max_element(r, r + c.vocab_size) - r);
  };
  std::vector<std::int32_t> a = prompt, b = prompt;
  ag::NoGradGuard ng;
  const auto t0 = Clock::now();
  lm::KvCache cache = m.make_cache();
  std::vector<std::int32_t> feed = prompt;
  for (int i = 0; i < gen; ++i) {
    const auto lg = run(m, feed, &cache);
    const std::int32_t nx = argmax_last(lg, feed.size());
    a.push_back(nx);
    feed = {nx};
  }
  const double cached = ms_since(t0);
  const auto t1 = Clock::now();
  for (int i = 0; i < gen; ++i) {
    const auto lg = run(m, b, nullptr);
    b.push_back(argmax_last(lg, b.size()));
  }
  const double full = ms_since(t1);
  std::printf("  info  %d token greedy: onbellekli %.2f ms/token, her adimda tam yeniden hesap %.2f ms/token (%.1fx)\n", gen,
              cached / gen, full / gen, full / cached);
  nexus_test::expect(a == b, "onbellekli greedy == tam yeniden hesapli greedy (48 token)");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "kvcache";
  try {
    if (mode == "kvcache") test_kvcache();
    else if (mode == "generate") test_generate();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase16_infer");
}
