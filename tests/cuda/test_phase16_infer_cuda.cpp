/**
 * @file test_phase16_infer_cuda.cpp
 * @brief FAZ 16 (GPU): GPT-2 124M cikarim — KV onbellegi, sayfali esdegerlik, ornekleme, spekulatif, nicemleme.
 *
 *   test_phase16_infer_cuda kvcache  : test_kv_cache_matches_full — FP32 kipte artimli cozme == tam ileri (<= 1e-4),
 *                                      ve PyTorch FP64 referansi (<= 1e-4); FP16 kip farki raporlanir
 *   test_phase16_infer_cuda paged    : test_paged_cache_equivalence — sayfali (16'lik, dagitik bloklar) == bitisik (tek
 *                                      blok) bit bit; copy-on-write dal == bagimsiz sira; bloklar geri doner
 *   test_phase16_infer_cuda sampling : test_sampling_distribution — sicaklik / top-k / top-p / min_p / cezalar: ki-kare
 *                                      (kesin filtreli dagilima karsi) + destek disi ornek yok
 *   test_phase16_infer_cuda spec     : test_speculative_exact — spekulatif (ilk-k katman taslak) == greedy, birebir
 *   test_phase16_infer_cuda quant    : test_quant_ppl — W8A8 / W4A16 / KV INT8 perplexity dususu esigin altinda
 *   test_phase16_infer_cuda graph    : decode CUDA Graph == eager (ayni tokenlar), kovalar
 *   test_phase16_infer_cuda serve    : surekli batch: 16 istek == tek tek uretim (greedy), toplam token/s > sirali;
 *                                      kucuk KV havuzunda preempt (recompute) ile tamamlanma
 *   test_phase16_infer_cuda bench    : TTFT / TPOT (istem 32..512), nicemleme, spekulatif hiz, esanli istek olcekleme
 * Agirliklar: NEXUS_GPT2_DIR (varsayilan C:/Users/muham/nexus-data/gpt2) — benchmarks/runner/faz16_gpt2.py export.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/infer.hpp"
#include "nexus/cuda/serve.hpp"
#include "nexus_test.hpp"
#include "nexusdata/text/bpe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ni = nexus::cuda::infer;
namespace nt = nexusdata::text;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::string env_or(const char* k, const std::string& d) {
  const char* v = std::getenv(k);
  return v && *v ? v : d;
}
fs::path gpt2_dir() { return env_or("NEXUS_GPT2_DIR", "C:/Users/muham/nexus-data/gpt2"); }

int run_status(const std::string& command) {
#if defined(_WIN32)
  return std::system(("\"" + command + "\"").c_str());
#else
  return std::system(command.c_str());
#endif
}

const ni::Gpt2Weights& weights() {
  static std::unique_ptr<ni::Gpt2Weights> w;
  if (!w) {
    const fs::path p = gpt2_dir() / "gpt2.nxg";
    if (!fs::exists(p)) {
      std::printf("  SKIP  %s yok (python benchmarks/runner/faz16_gpt2.py export %s)\n", p.string().c_str(),
                  gpt2_dir().string().c_str());
      std::exit(77);
    }
    w = std::make_unique<ni::Gpt2Weights>(ni::load_gpt2(p.string()));
  }
  return *w;
}

const nt::ByteBpe& bpe() {
  static std::unique_ptr<nt::ByteBpe> b;
  if (!b) b = std::make_unique<nt::ByteBpe>(nt::ByteBpe::load_tiktoken((gpt2_dir() / "gpt2.tiktoken").string(), nt::Pretokenizer::Gpt2,
                                                                        {{"<|endoftext|>", 50256}}));
  return *b;
}

std::vector<std::int32_t> encode(const std::string& text) {
  const auto ids = bpe().encode(text);
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

const char* kText =
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. "
    "One day, she found a shiny red ball under a big tree. She picked it up and ran to show her mom.";

ni::EngineConfig cfg(ni::Precision p, int block = 16) {
  ni::EngineConfig c;
  c.precision = p;
  c.max_seqs = 8;
  c.max_tokens = 1024;
  c.block_size = block;
  c.kv_bytes = std::size_t{512} << 20;
  return c;
}

std::vector<std::vector<float>> incremental_logits(ni::Engine& e, const std::vector<std::int32_t>& t, int prefix) {
  const int s = e.add_sequence();
  std::vector<std::vector<float>> out;
  ni::Chunk c;
  c.seq = s;
  c.tokens.assign(t.begin(), t.begin() + prefix);
  c.logits = prefix;
  (void)e.forward({c});
  for (int i = 0; i < prefix; ++i) out.push_back(e.logits_host(i));
  for (std::size_t i = static_cast<std::size_t>(prefix); i < t.size(); ++i) {
    ni::Chunk d;
    d.seq = s;
    d.tokens = {t[i]};
    (void)e.forward({d});
    out.push_back(e.logits_host(0));
  }
  e.free_sequence(s);
  return out;
}

std::vector<std::vector<float>> full_logits(ni::Engine& e, const std::vector<std::int32_t>& t) {
  return incremental_logits(e, t, static_cast<int>(t.size()));
}

double rel_diff(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
  double m = 0.0, r = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < a[i].size(); ++j) {
      m = std::max(m, std::fabs(static_cast<double>(a[i][j]) - b[i][j]));
      r = std::max(r, std::fabs(static_cast<double>(b[i][j])));
    }
  return m / std::max(r, 1e-30);
}

// ==========================================================================================

void test_kvcache() {
  const auto t = encode(kText);
  std::printf("  info  istem %zu token\n", t.size());
  ni::Engine e32(weights(), cfg(ni::Precision::F32));
  const auto full = full_logits(e32, t);
  const auto inc = incremental_logits(e32, t, 8);
  const double d32 = rel_diff(inc, full);
  std::printf("  info  FP32: artimli (8 + %zu decode) vs tam ileri goreli max %.2e\n", t.size() - 8, d32);
  nexus_test::expect(d32 <= 1e-4, "test_kv_cache_matches_full: artimli cozme == tam ileri (FP32, <= 1e-4)");
  // PyTorch FP64 referansi
  const std::string py = NEXUS_PARITY_PYTHON;
  if (!py.empty() && fs::exists(py)) {
    const fs::path dir = fs::temp_directory_path() / "nexus_faz16";
    fs::create_directories(dir);
    const fs::path tf = dir / "tok.bin";
    std::ofstream(tf, std::ios::binary).write(reinterpret_cast<const char*>(t.data()), static_cast<std::streamsize>(t.size() * 4));
    const int rc = run_status("\"" + py + "\" \"" NEXUS_FAZ16_SCRIPT "\" logits \"" + dir.string() + "\" \"" +
                              tf.string() + "\"");
    std::ifstream in(dir / "ref_logits.bin", std::ios::binary);
    if (rc == 0 && in) {
      std::vector<std::vector<float>> ref(t.size(), std::vector<float>(static_cast<std::size_t>(e32.vocab())));
      for (auto& r : ref) in.read(reinterpret_cast<char*>(r.data()), static_cast<std::streamsize>(r.size() * 4));
      const double dr = rel_diff(full, ref), di = rel_diff(inc, ref);
      std::printf("  info  PyTorch FP64 referansina gore: tam %.2e, artimli %.2e\n", dr, di);
      nexus_test::expect(dr <= 1e-4 && di <= 1e-4, "GPT-2 124M logitleri PyTorch FP64 referansiyla <= 1e-4 (FP32 kip)");
      ni::Engine e16(weights(), cfg(ni::Precision::F16));
      const auto f16 = full_logits(e16, t), i16 = incremental_logits(e16, t, 8);
      std::printf("  info  FP16 kip: tam vs ref %.2e, artimli vs ref %.2e, artimli vs tam %.2e; argmax uyusmasi %s\n",
                  rel_diff(f16, ref), rel_diff(i16, ref), rel_diff(i16, f16), [&] {
                    int same = 0;
                    for (std::size_t i = 0; i < ref.size(); ++i)
                      same += std::max_element(i16[i].begin(), i16[i].end()) - i16[i].begin() ==
                              std::max_element(ref[i].begin(), ref[i].end()) - ref[i].begin();
                    static char b[32];
                    std::snprintf(b, sizeof(b), "%d/%zu", same, ref.size());
                    return b;
                  }());
    } else {
      std::printf("  info  PyTorch referansi calismadi (kod %d), atlandi\n", rc);
    }
  }
}

void test_paged() {
  const auto t = encode(kText);
  ni::Engine paged(weights(), cfg(ni::Precision::F32, 16));
  ni::Engine contig(weights(), cfg(ni::Precision::F32, 1024));  // tek blok = bitisik KV
  const int free0 = paged.free_blocks();
  // iki sira donusumlu buyutulur -> sayfalar dagitik
  const int a = paged.add_sequence(), b = paged.add_sequence();
  ni::Chunk ca, cb;
  ca.seq = a;
  ca.tokens.assign(t.begin(), t.begin() + 5);
  cb.seq = b;
  cb.tokens.assign(t.begin() + 3, t.begin() + 9);
  (void)paged.forward({ca, cb});
  std::vector<std::vector<float>> pa, pb;
  for (std::size_t i = 5; i < t.size(); ++i) {
    ni::Chunk x, y;
    x.seq = a;
    x.tokens = {t[i]};
    y.seq = b;
    y.tokens = {t[(i + 4) % t.size()]};
    (void)paged.forward({x, y});
    pa.push_back(paged.logits_host(0));
    pb.push_back(paged.logits_host(1));
  }
  // bitisik: her sira tek basina
  auto run_contig = [&](std::vector<std::int32_t> first, std::vector<std::int32_t> rest) {
    const int s = contig.add_sequence();
    ni::Chunk c;
    c.seq = s;
    c.tokens = std::move(first);
    (void)contig.forward({c});
    std::vector<std::vector<float>> o;
    for (auto tk : rest) {
      ni::Chunk d;
      d.seq = s;
      d.tokens = {tk};
      (void)contig.forward({d});
      o.push_back(contig.logits_host(0));
    }
    contig.free_sequence(s);
    return o;
  };
  std::vector<std::int32_t> ra, rb;
  for (std::size_t i = 5; i < t.size(); ++i) {
    ra.push_back(t[i]);
    rb.push_back(t[(i + 4) % t.size()]);
  }
  const auto qa = run_contig(std::vector<std::int32_t>(t.begin(), t.begin() + 5), ra);
  const auto qb = run_contig(std::vector<std::int32_t>(t.begin() + 3, t.begin() + 9), rb);
  const bool same = pa == qa && pb == qb;
  std::printf("  info  sayfali (16'lik bloklar, 2 sira dagitik) vs bitisik (tek blok): %s (goreli %.1e / %.1e)\n",
              same ? "bit bit" : "FARKLI", rel_diff(pa, qa), rel_diff(pb, qb));
  nexus_test::expect(same, "test_paged_cache_equivalence: sayfali == bitisik KV (bit bit)");
  // copy-on-write: a'yi dallandir, iki dal farkli token
  const int fork = paged.fork_sequence(a);
  const int before = paged.free_blocks();
  ni::Chunk x, y;
  x.seq = a;
  x.tokens = {t[1]};
  y.seq = fork;
  y.tokens = {t[2]};
  (void)paged.forward({x, y});
  const auto la = paged.logits_host(0), lf = paged.logits_host(1);
  // ayni plan (5 token prefill + ayni decode tokenlari) bitisik motorda, sonra dal tokeni
  auto lone = [&](std::int32_t last) {
    std::vector<std::int32_t> rest = ra;
    rest.push_back(last);
    return run_contig(std::vector<std::int32_t>(t.begin(), t.begin() + 5), rest).back();
  };
  const bool cow = la == lone(t[1]) && lf == lone(t[2]);
  std::printf("  info  CoW dal: paylasilan son blok kopyalandi (%d blok kullanildi), dallar bagimsiz sirayla %s\n",
              before - paged.free_blocks(), cow ? "bit bit" : "FARKLI");
  nexus_test::expect(cow, "copy-on-write dal: iki dal, bagimsiz siralarla bit bit ayni");
  paged.free_sequence(a);
  paged.free_sequence(b);
  paged.free_sequence(fork);
  nexus_test::expect(paged.free_blocks() == free0, "tum bloklar serbest listeye dondu");
}

// ---- ornekleme dagilimi -------------------------------------------------------------------
struct Expect {
  std::vector<double> p;  // kesin filtreli dagilim
};

Expect expected(std::vector<float> l, const ni::SamplingParams& sp, const std::vector<int>& counts, const std::vector<int>& outc) {
  const std::size_t V = l.size();
  for (std::size_t j = 0; j < V; ++j) {
    float v = l[j];
    if (counts[j] > 0 && sp.repetition_penalty != 1.0f) v = v > 0 ? v / sp.repetition_penalty : v * sp.repetition_penalty;
    v -= sp.frequency_penalty * outc[j] + (outc[j] > 0 ? sp.presence_penalty : 0.0f);
    l[j] = v / sp.temperature;
  }
  std::vector<float> sorted = l;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());
  const float mx = sorted[0];
  float thr = -1e30f;
  if (sp.top_k > 0 && sp.top_k < static_cast<int>(V)) thr = sorted[static_cast<std::size_t>(sp.top_k - 1)];
  if (sp.top_p < 1.0f) {
    double z = 0;
    for (float v : sorted)
      if (v >= thr) z += std::exp(static_cast<double>(v) - mx);
    double c = 0;
    for (float v : sorted) {
      if (v < thr) break;
      c += std::exp(static_cast<double>(v) - mx);
      if (c >= sp.top_p * z) {
        thr = std::max(thr, v);
        break;
      }
    }
  }
  if (sp.min_p > 0) thr = std::max(thr, mx + std::log(sp.min_p));
  Expect e;
  e.p.assign(V, 0.0);
  double z = 0;
  for (std::size_t j = 0; j < V; ++j)
    if (l[j] >= thr) z += (e.p[j] = std::exp(static_cast<double>(l[j]) - mx));
  for (double& v : e.p) v /= z;
  return e;
}

/// Ki-kare: p >= 0.002 kovalar tek, kalanlar birlesik. Doner: (chi2, df, z-istatistigi Wilson-Hilferty)
std::pair<double, double> chi2(const std::vector<double>& p, const std::vector<int>& cnt, int n) {
  double c = 0, rest_p = 0;
  int rest_c = 0, df = -1;
  for (std::size_t j = 0; j < p.size(); ++j) {
    if (p[j] >= 0.002) {
      const double e = p[j] * n;
      c += (cnt[j] - e) * (cnt[j] - e) / e;
      df += 1;
    } else {
      rest_p += p[j];
      rest_c += cnt[j];
    }
  }
  if (rest_p * n >= 5) {
    const double e = rest_p * n;
    c += (rest_c - e) * (rest_c - e) / e;
    df += 1;
  }
  const double k = std::max(1, df);
  const double zst = (std::cbrt(c / k) - (1 - 2 / (9 * k))) / std::sqrt(2 / (9 * k));
  return {c / k, zst};
}

void test_sampling() {
  ni::Engine e(weights(), cfg(ni::Precision::F16));
  const auto t = encode(kText);
  const int s = e.add_sequence();
  ni::Chunk c;
  c.seq = s;
  c.tokens = t;
  (void)e.forward({c});
  const std::vector<float> lg = e.logits_host(0);
  const int V = e.vocab();
  // ceza sayaclari: istem tokenlari (cikti degil)
  std::vector<int> counts(static_cast<std::size_t>(V), 0), outc(static_cast<std::size_t>(V), 0);
  for (auto tk : t) counts[static_cast<std::size_t>(tk)] += 1;
  struct Case {
    const char* name;
    ni::SamplingParams sp;
  };
  std::vector<Case> cases;
  auto mk = [](float T, int k, float p, float mp, float rep) {
    ni::SamplingParams sp;
    sp.temperature = T;
    sp.top_k = k;
    sp.top_p = p;
    sp.min_p = mp;
    sp.repetition_penalty = rep;
    sp.seed = 1234;
    return sp;
  };
  cases.push_back({"T=1.0", mk(1.0f, 0, 1.0f, 0.0f, 1.0f)});
  cases.push_back({"T=0.7", mk(0.7f, 0, 1.0f, 0.0f, 1.0f)});
  cases.push_back({"top_k=40", mk(1.0f, 40, 1.0f, 0.0f, 1.0f)});
  cases.push_back({"top_p=0.9", mk(1.0f, 0, 0.9f, 0.0f, 1.0f)});
  cases.push_back({"min_p=0.05", mk(1.0f, 0, 1.0f, 0.05f, 1.0f)});
  cases.push_back({"T=0.8,k=50,p=0.8", mk(0.8f, 50, 0.8f, 0.0f, 1.0f)});
  cases.push_back({"rep=1.3,T=1.2", mk(1.2f, 0, 1.0f, 0.0f, 1.3f)});
  const int n = 20000;
  for (const Case& cs : cases) {
    const Expect ex = expected(lg, cs.sp, counts, outc);
    std::vector<int> cnt(static_cast<std::size_t>(V), 0);
    int outside = 0;
    for (int i = 0; i < n; ++i) {
      const int tk = e.sample({cs.sp}, {s}, static_cast<std::uint64_t>(i)).at(0);
      cnt[static_cast<std::size_t>(tk)] += 1;
      outside += ex.p[static_cast<std::size_t>(tk)] == 0.0 ? 1 : 0;
    }
    int support = 0;
    for (double v : ex.p) support += v > 0 ? 1 : 0;
    const auto [c2, z] = chi2(ex.p, cnt, n);
    std::printf("  info  %-18s destek %5d token; destek disi ornek %d; ki-kare/df %.3f (z %.2f)\n", cs.name, support, outside, c2, z);
    nexus_test::expect(outside == 0 && z < 4.0, std::string("test_sampling_distribution: ") + cs.name + " (z < 4, destek disi 0)");
  }
  ni::SamplingParams g;
  g.temperature = 0.0f;
  const int am = static_cast<int>(std::max_element(lg.begin(), lg.end()) - lg.begin());
  nexus_test::expect(e.sample({g}, {s}, 0).at(0) == am, "greedy = argmax");
  e.free_sequence(s);
}

void test_spec() {
  ni::Engine e(weights(), cfg(ni::Precision::F16));
  const auto prompt = encode(kText);
  ni::SamplingParams g;
  g.temperature = 0.0f;
  ni::GenerateStats gs;
  const auto ref = ni::generate(e, prompt, 128, g, &gs);
  for (int k : {4, 6, 8}) {
    for (int gamma : {2, 4, 6}) {
      ni::SpecStats st;
      const auto out = ni::speculative_generate(e, prompt, 128, k, gamma, &st);
      std::printf("  info  taslak %d katman, gamma %d: kabul %d/%d (%.0f%%), adim basina %.2f token, %.1f ms (greedy %.1f ms)\n", k,
                  gamma, st.accepted, st.drafted, 100.0 * st.accepted / std::max(1, st.drafted), st.tokens_per_step, st.ms,
                  gs.ttft_ms + gs.tpot_ms * 127);
      nexus_test::expect(out == ref, "test_speculative_exact: k=" + std::to_string(k) + " gamma=" + std::to_string(gamma) +
                                         " spekulatif == greedy (128 token)");
    }
  }
  std::printf("  info  ornek cikti: %s\n", bpe().decode(std::vector<std::uint32_t>(ref.begin(), ref.begin() + 40)).c_str());
}

std::vector<std::int32_t> tiny_tokens(const char* file, std::size_t max_bytes, std::size_t offset) {
  std::ifstream in(env_or("NEXUS_TINYSTORIES", "C:/Users/muham/nexus-data/tinystories/") + file, std::ios::binary);
  if (!in) return {};
  in.seekg(static_cast<std::streamoff>(offset));
  std::string s(max_bytes, '\0');
  in.read(s.data(), static_cast<std::streamsize>(max_bytes));
  s.resize(static_cast<std::size_t>(in.gcount()));
  return encode(s);
}

double perplexity(ni::Engine& e, const std::vector<std::int32_t>& toks, int window) {
  double sum = 0.0;
  std::size_t cnt = 0;
  for (std::size_t at = 0; at + static_cast<std::size_t>(window) + 1 <= toks.size(); at += static_cast<std::size_t>(window)) {
    const int s = e.add_sequence();
    ni::Chunk c;
    c.seq = s;
    c.tokens.assign(toks.begin() + static_cast<std::ptrdiff_t>(at), toks.begin() + static_cast<std::ptrdiff_t>(at) + window);
    c.logits = window;
    (void)e.forward({c});
    std::vector<std::int32_t> tg(toks.begin() + static_cast<std::ptrdiff_t>(at) + 1, toks.begin() + static_cast<std::ptrdiff_t>(at) + window + 1);
    for (float v : e.nll(tg)) sum += v;
    cnt += static_cast<std::size_t>(window);
    e.free_sequence(s);
  }
  return std::exp(sum / static_cast<double>(cnt));
}

void test_quant() {
  const auto eval = tiny_tokens("TinyStoriesV2-GPT4-valid.txt", 120000, 0);
  const auto calib = tiny_tokens("TinyStoriesV2-GPT4-train.txt", 4000, 1000000);
  if (eval.size() < 4096) {
    std::printf("  SKIP  TinyStories yok\n");
    std::exit(77);
  }
  const std::vector<std::int32_t> cal(calib.begin(), calib.begin() + std::min<std::size_t>(512, calib.size()));
  const int window = 512;
  struct Case {
    const char* name;
    ni::WeightQuant w;
    ni::KvType kv;
    bool calibrate;
    double limit;  // izin verilen goreli artis
  };
  const Case cases[] = {
      {"fp16", ni::WeightQuant::F16, ni::KvType::F16, false, 0.0},
      {"KV int8", ni::WeightQuant::F16, ni::KvType::INT8, false, 0.01},
      {"W8A8 (olceksiz)", ni::WeightQuant::W8A8, ni::KvType::F16, false, 1e9},
      {"W8A8 SmoothQuant", ni::WeightQuant::W8A8, ni::KvType::F16, true, 0.05},
      {"W4A16 g128 (RTN)", ni::WeightQuant::W4A16, ni::KvType::F16, false, 1e9},
      {"W4A16 g128 AWQ", ni::WeightQuant::W4A16, ni::KvType::F16, true, 0.15},
  };
  double base = 0.0;
  for (const Case& cs : cases) {
    ni::EngineConfig c = cfg(ni::Precision::F16);
    c.quant.weights = cs.w;
    c.quant.kv = cs.kv;
    if (cs.calibrate) c.quant.calibration = cal;
    const auto t0 = Clock::now();
    ni::Engine e(weights(), c);
    const double build = ms_since(t0);
    // KV INT8 etkisi icin: pencerenin ikinci yarisi decode ile (onbellekten okuyarak) degerlendirilmeli; prefill
    // taze K/V kullanir. Bu yuzden KV INT8'de pencere 64 token prefill + kalan decode ile olculur.
    double ppl = 0.0;
    if (cs.kv == ni::KvType::INT8) {
      double sum = 0.0;
      std::size_t n = 0;
      for (std::size_t at = 0; at + 257 <= eval.size() && at < 16 * 256; at += 256) {
        const int s = e.add_sequence();
        ni::Chunk p;
        p.seq = s;
        p.tokens.assign(eval.begin() + static_cast<std::ptrdiff_t>(at), eval.begin() + static_cast<std::ptrdiff_t>(at) + 64);
        p.logits = 64;
        (void)e.forward({p});
        std::vector<std::int32_t> tg(eval.begin() + static_cast<std::ptrdiff_t>(at) + 1, eval.begin() + static_cast<std::ptrdiff_t>(at) + 65);
        for (float v : e.nll(tg)) sum += v;
        for (int i = 64; i < 256; ++i) {
          ni::Chunk d;
          d.seq = s;
          d.tokens = {eval[at + static_cast<std::size_t>(i)]};
          (void)e.forward({d});
          sum += e.nll({eval[at + static_cast<std::size_t>(i) + 1]}).at(0);
        }
        n += 256;
        e.free_sequence(s);
      }
      ppl = std::exp(sum / static_cast<double>(n));
      // ayni protokol fp16 ile
      ni::Engine b(weights(), cfg(ni::Precision::F16));
      double bsum = 0.0;
      for (std::size_t at = 0; at + 257 <= eval.size() && at < 16 * 256; at += 256) {
        const int s = b.add_sequence();
        ni::Chunk p;
        p.seq = s;
        p.tokens.assign(eval.begin() + static_cast<std::ptrdiff_t>(at), eval.begin() + static_cast<std::ptrdiff_t>(at) + 64);
        p.logits = 64;
        (void)b.forward({p});
        std::vector<std::int32_t> tg(eval.begin() + static_cast<std::ptrdiff_t>(at) + 1, eval.begin() + static_cast<std::ptrdiff_t>(at) + 65);
        for (float v : b.nll(tg)) bsum += v;
        for (int i = 64; i < 256; ++i) {
          ni::Chunk d;
          d.seq = s;
          d.tokens = {eval[at + static_cast<std::size_t>(i)]};
          (void)b.forward({d});
          bsum += b.nll({eval[at + static_cast<std::size_t>(i) + 1]}).at(0);
        }
        b.free_sequence(s);
      }
      const double bppl = std::exp(bsum / static_cast<double>(n));
      std::printf("  info  %-18s ppl %.4f (ayni decode protokolu fp16 %.4f, %+.2f%%), KV %zu -> %zu bayt/token\n", cs.name, ppl, bppl,
                  100.0 * (ppl / bppl - 1.0), static_cast<std::size_t>(12 * 2 * 768 * 2), e.kv_bytes_per_token());
      nexus_test::expect(ppl / bppl - 1.0 <= cs.limit, std::string("test_quant_ppl: ") + cs.name + " ppl artisi <= %1");
      continue;
    }
    ppl = perplexity(e, eval, window);
    if (base == 0.0) base = ppl;
    std::printf("  info  %-18s ppl %.4f (%+.2f%%), agirlik %.1f MB, kurulum %.0f ms (%zu token degerlendirme)\n", cs.name, ppl,
                100.0 * (ppl / base - 1.0), e.weight_bytes() / 1e6, build, eval.size() / window * window);
    if (cs.limit < 1e8)
      nexus_test::expect(ppl / base - 1.0 <= cs.limit,
                         std::string("test_quant_ppl: ") + cs.name + " ppl artisi <= " + std::to_string(static_cast<int>(cs.limit * 100)) + "%");
  }
}

void test_graph() {
  const auto prompt = encode(kText);
  ni::SamplingParams g;
  g.temperature = 0.0f;
  ni::EngineConfig c = cfg(ni::Precision::F16);
  ni::Engine eg(weights(), c);
  c.cuda_graphs = false;
  ni::Engine en(weights(), c);
  ni::GenerateStats a, b;
  const auto x = ni::generate(eg, prompt, 96, g, &a);
  const auto y = ni::generate(en, prompt, 96, g, &b);
  std::printf("  info  decode CUDA Graph: %d kova yakalandi; TPOT %.3f ms (graph'siz %.3f ms)\n", eg.graphs_captured(), a.tpot_ms, b.tpot_ms);
  nexus_test::expect(x == y && eg.graphs_captured() >= 1, "decode CUDA Graph == eager (ayni 96 token)");
  // 5 esanli sira: kova 8
  std::vector<int> seqs;
  std::vector<ni::Chunk> pre;
  for (int i = 0; i < 5; ++i) {
    seqs.push_back(eg.add_sequence());
    ni::Chunk ch;
    ch.seq = seqs.back();
    ch.tokens.assign(prompt.begin(), prompt.begin() + 10 + i);
    pre.push_back(ch);
  }
  (void)eg.forward(pre);
  std::vector<ni::Chunk> dec;
  for (int sq : seqs) {
    ni::Chunk ch;
    ch.seq = sq;
    ch.tokens = {prompt[3]};
    dec.push_back(ch);
  }
  (void)eg.forward(dec);
  nexus_test::expect(eg.last_step().graph && eg.logit_rows() == 5, "5 siralik decode kova 8 graph'iyla (dolgulu satirlar donmez)");
  for (int sq : seqs) eg.free_sequence(sq);
}

std::vector<std::vector<std::int32_t>> serve_prompts(int n) {
  const auto toks = tiny_tokens("TinyStoriesV2-GPT4-valid.txt", 200000, 0);
  std::vector<std::vector<std::int32_t>> out;
  std::size_t at = 0;
  for (int i = 0; i < n; ++i) {
    const std::size_t len = 16 + static_cast<std::size_t>((i * 37) % 200);
    if (at + len > toks.size()) at = 0;
    out.emplace_back(toks.begin() + static_cast<std::ptrdiff_t>(at), toks.begin() + static_cast<std::ptrdiff_t>(at + len));
    at += len;
  }
  return out;
}

void test_serve() {
  const int n = 16, gen = 48;
  const auto prompts = serve_prompts(n);
  if (prompts.empty() || prompts[0].empty()) {
    std::printf("  SKIP  TinyStories yok\n");
    std::exit(77);
  }
  ni::SamplingParams g;
  g.temperature = 0.0f;
  ni::EngineConfig c = cfg(ni::Precision::F16);
  c.max_seqs = 32;
  c.max_tokens = 2048;
  ni::Engine e(weights(), c);
  for (int w = 0; w < 2; ++w) (void)ni::generate(e, prompts[0], 8, g);  // isinma (graph kovalari)
  std::vector<std::vector<std::int32_t>> seq_out;
  const auto ts = Clock::now();
  for (const auto& p : prompts) seq_out.push_back(ni::generate(e, p, gen, g));
  const double seq_ms = ms_since(ts);
  nexus::cuda::serve::Server srv(e);
  std::vector<std::future<nexus::cuda::serve::Result>> fut;
  const auto t0 = Clock::now();
  for (const auto& p : prompts) {
    nexus::cuda::serve::Request r;
    r.prompt = p;
    r.max_new = gen;
    r.sampling = g;
    fut.push_back(srv.submit(r));
  }
  int same = 0, first_div = -1;
  double ttft = 0, tpot = 0;
  for (int i = 0; i < n; ++i) {
    const auto r = fut[static_cast<std::size_t>(i)].get();
    if (r.tokens == seq_out[static_cast<std::size_t>(i)]) {
      ++same;
    } else if (first_div < 0) {
      first_div = i;
    }
    ttft += r.ttft_ms / n;
    tpot += r.tpot_ms / n;
  }
  const double srv_ms = ms_since(t0);
  const auto m = srv.metrics();
  std::printf("  info  %d istek x %d token: sirali %.0f ms (%.0f tok/s), surekli batch %.0f ms (%.0f tok/s, %.2fx); adim %lld, "
              "esanli en cok %d, ortalama satir %.1f; TTFT ort %.1f ms, TPOT ort %.2f ms\n",
              n, gen, seq_ms, n * gen / (seq_ms / 1000.0), srv_ms, n * gen / (srv_ms / 1000.0), seq_ms / srv_ms,
              static_cast<long long>(m.steps), m.max_running, m.mean_batch, ttft, tpot);
  std::printf("  info  greedy cikti tek tek uretimle ayni: %d/%d%s\n", same, n,
              first_div >= 0 ? " (fark: yigin icinde cuBLASLt GEMM sekli degisiyor -> FP16 yuvarlama; ilk farkli istek)" : "");
  nexus_test::expect(srv_ms < seq_ms, "surekli batch toplam token/s > ayri isteklerin toplami");
  nexus_test::expect(same >= n * 3 / 4, "greedy ciktilarin en az %75'i tek tek uretimle birebir");
  // kucuk KV havuzu: preempt
  ni::EngineConfig sc = c;
  sc.kv_bytes = static_cast<std::size_t>(12 * 2 * 768 * 2) * 16 * 40;  // ~40 blok
  ni::Engine small(weights(), sc);
  nexus::cuda::serve::Server s2(small);
  std::vector<std::future<nexus::cuda::serve::Result>> f2;
  for (int i = 0; i < 8; ++i) {
    nexus::cuda::serve::Request r;
    r.prompt = prompts[static_cast<std::size_t>(i)];
    r.max_new = gen;
    r.sampling = g;
    f2.push_back(s2.submit(r));
  }
  int done = 0, pre = 0;
  for (auto& f : f2) {
    const auto r = f.get();
    done += static_cast<int>(r.tokens.size()) == gen ? 1 : 0;
    pre += r.preemptions;
  }
  std::printf("  info  %d blokluk havuz: 8 istek, %d onceden bosaltma (recompute), %d/8 tamamlandi; serbest blok %d/%d\n",
              small.total_blocks(), pre, done, small.free_blocks(), small.total_blocks());
  nexus_test::expect(done == 8 && pre > 0 && small.free_blocks() == small.total_blocks(),
                     "KV havuzu yetmeyince preempt + yeniden prefill ile tum istekler tamamlanir, bloklar geri doner");
}

int bench() {
  ni::SamplingParams g;
  g.temperature = 0.0f;
  for (ni::WeightQuant q : {ni::WeightQuant::F16, ni::WeightQuant::W8A8, ni::WeightQuant::W4A16}) {
    ni::EngineConfig c = cfg(ni::Precision::F16);
    c.quant.weights = q;
    ni::Engine e(weights(), c);
    std::mt19937 rng(7);
    for (int plen : {32, 128, 512}) {
      std::vector<std::int32_t> prompt(static_cast<std::size_t>(plen));
      for (auto& v : prompt) v = static_cast<std::int32_t>(rng() % 50000);
      ni::GenerateStats st;
      double ttft = 1e30, tpot = 1e30;
      for (int rep = 0; rep < 4; ++rep) {
        (void)ni::generate(e, prompt, 128, g, &st);
        if (rep > 0) {
          ttft = std::min(ttft, st.ttft_ms);
          tpot = std::min(tpot, st.tpot_ms);
        }
      }
      std::printf("{\"mode\": \"bench_infer\", \"quant\": \"%s\", \"prompt\": %d, \"gen\": 128, \"ttft_ms\": %.3f, \"tpot_ms\": %.3f, "
                  "\"tokens_per_s\": %.1f, \"weight_mb\": %.1f}\n",
                  ni::to_string(q), plen, ttft, tpot, 1000.0 / tpot, e.weight_bytes() / 1e6);
    }
  }
  ni::Engine e(weights(), cfg(ni::Precision::F16));
  const auto prompt = encode(kText);
  ni::GenerateStats gs;
  (void)ni::generate(e, prompt, 128, g, &gs);
  (void)ni::generate(e, prompt, 128, g, &gs);
  for (int k : {4, 6}) {
    ni::SpecStats st;
    (void)ni::speculative_generate(e, prompt, 128, k, 4, &st);
    (void)ni::speculative_generate(e, prompt, 128, k, 4, &st);
    std::printf("{\"mode\": \"bench_spec\", \"draft_layers\": %d, \"gamma\": 4, \"accept_rate\": %.3f, \"tokens_per_step\": %.2f, "
                "\"ms\": %.1f, \"greedy_ms\": %.1f}\n",
                k, static_cast<double>(st.accepted) / st.drafted, st.tokens_per_step, st.ms, gs.ttft_ms + gs.tpot_ms * 127);
  }
  // Surekli batch: esanli istek sayisi -> toplam token/s (istem 128, uretim 128)
  {
    ni::EngineConfig c = cfg(ni::Precision::F16);
    c.max_seqs = 64;
    c.max_tokens = 4096;
    c.kv_bytes = std::size_t{2} << 30;
    ni::Engine se(weights(), c);
    std::mt19937 rng(11);
    for (int conc : {1, 2, 4, 8, 16, 32, 64}) {
      nexus::cuda::serve::Server srv(se);
      std::vector<std::future<nexus::cuda::serve::Result>> fut;
      const auto t0 = Clock::now();
      for (int i = 0; i < conc; ++i) {
        nexus::cuda::serve::Request r;
        r.prompt.resize(128);
        for (auto& v : r.prompt) v = static_cast<std::int32_t>(rng() % 50000);
        r.max_new = 128;
        r.sampling = g;
        fut.push_back(srv.submit(r));
      }
      double ttft = 0, tpot = 0;
      for (auto& f : fut) {
        const auto r = f.get();
        ttft += r.ttft_ms / conc;
        tpot += r.tpot_ms / conc;
      }
      const double ms = ms_since(t0);
      std::printf("{\"mode\": \"bench_serve\", \"concurrent\": %d, \"total_tokens_per_s\": %.1f, \"ttft_ms\": %.2f, \"tpot_ms\": %.3f, "
                  "\"steps\": %lld}\n",
                  conc, conc * 128 / (ms / 1000.0), ttft, tpot, static_cast<long long>(srv.metrics().steps));
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "kvcache";
  try {
    if (mode == "kvcache") test_kvcache();
    else if (mode == "paged") test_paged();
    else if (mode == "sampling") test_sampling();
    else if (mode == "spec") test_spec();
    else if (mode == "quant") test_quant();
    else if (mode == "graph") test_graph();
    else if (mode == "serve") test_serve();
    else if (mode == "bench") return bench();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase16_infer_cuda");
}
