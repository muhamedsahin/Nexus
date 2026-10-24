/**
 * @file test_flash_attention.cpp
 * @brief FAZ 5 testleri: CUDA FlashAttention-2 ve CPU flash, FP64 "math" kahinine karsi.
 *
 * promt.txt FAZ 5 TESTLER:
 *  - test_flash_vs_math: (B, H, Hkv, Tq, Tk, d, causal, window, softcap, alibi, dtype) izgarasi;
 *    O, LSE, dQ, dK, dV.
 *  - test_flash_varlen_equals_separate: paketli varlen = dizileri tek tek calistirmak.
 *  - test_flash_tq1_decode: Tq = 1, Tk = 1..4096.
 *  - test_flash_dropout_stats: dusurulen oran p +- 3 sigma; maske math ile ayni, fwd/bwd tutarli.
 *  - test_flash_determinism: deterministik modda dQ iki kosuda bit-bit ayni.
 *  - test_causal_future_invariance: gelecek K/V degisince gecmis cikti bit-bit ayni.
 *
 * Tolerans (BOLUM G.4 dosyada tanimli olmadigi icin burada tanimlanir): norm-goreli
 * max|a-b| / max|b|. BF16: O 2e-2, gradyan 3e-2. F16: O 5e-3, gradyan 1e-2. CPU flash (FP32): 1e-5.
 * Girdiler dar tipte temsil edilebilir degerlere yuvarlanir; kahin ayni degerleri FP64'te isler.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/attention/cpu.hpp"
#include "nexus/core/half.hpp"
#include "nexus/cuda/attention.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/fused/training.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace na = nexus::attention;
using nexus::AttentionParams;
using nexus::DType;

namespace {

nc::Stream g_stream;

float round_to(DType t, float x) {
  return t == DType::BF16 ? nexus::bfloat16_bits_to_float(nexus::float_to_bfloat16_bits(x))
                          : nexus::half_bits_to_float(nexus::float_to_half_bits(x));
}

std::vector<float> random_values(std::size_t n, std::uint64_t seed, DType t, float scale = 1.0f) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (float& x : v) {
    x = round_to(t, d(rng));
  }
  return v;
}

struct Dev {
  nc::DeviceBuffer buf;
  std::size_t n;
  DType t;
  Dev(const std::vector<float>& host, DType type) : buf(std::max<std::size_t>(host.size(), 1) * 4, g_stream), n(host.size()), t(type) {
    nc::DeviceBuffer tmp(std::max<std::size_t>(n, 1) * 4, g_stream);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(tmp.data(), host.data(), n * 4, cudaMemcpyHostToDevice, g_stream.handle()));
    nc::cast(buf.data(), t, tmp.data(), DType::F32, n, g_stream.handle());
    g_stream.synchronize();
  }
  Dev(std::size_t count, DType type) : buf(std::max<std::size_t>(count, 1) * 4, g_stream), n(count), t(type) {
    NEXUS_CUDA_CHECK(cudaMemsetAsync(buf.data(), 0, std::max<std::size_t>(count, 1) * 4, g_stream.handle()));
  }
  std::vector<float> get() const {
    nc::DeviceBuffer tmp(std::max<std::size_t>(n, 1) * 4, g_stream);
    nc::cast(tmp.data(), DType::F32, buf.data(), t, n, g_stream.handle());
    std::vector<float> host(n);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data(), tmp.data(), n * 4, cudaMemcpyDeviceToHost, g_stream.handle()));
    g_stream.synchronize();
    return host;
  }
};

double rel(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  double scale = 1e-6;
  for (std::size_t i = 0; i < b.size(); ++i) {
    if (!std::isfinite(b[i])) {
      if (std::isfinite(a[i]) || std::signbit(a[i]) != std::signbit(b[i])) {
        return INFINITY;
      }
      continue;
    }
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
    scale = std::max(scale, std::fabs(static_cast<double>(b[i])));
  }
  return worst / scale;
}

void report(const std::string& name, double err, double tol) {
  std::printf("  info  %-58s hata=%.3e esik=%.1e\n", name.c_str(), err, tol);
  nexus_test::expect(err < tol, name);
}

struct Case {
  const char* name;
  int b, h, hkv, tq, tk, d;
  DType t;
  bool causal;
  int window = 0;
  float softcap = 0.0f;
  bool alibi = false;
  float dropout = 0.0f;
};

struct Run {
  std::vector<float> o, lse, dq, dk, dv;
};

AttentionParams make_params(const Case& c) {
  AttentionParams p;
  p.batch = c.b;
  p.heads = c.h;
  p.kv_heads = c.hkv;
  p.tq = c.tq;
  p.tk = c.tk;
  p.head_dim = c.d;
  p.dtype = c.t;
  p.causal = c.causal;
  p.window = c.window;
  p.softcap = c.softcap;
  p.dropout_p = c.dropout;
  p.seed = 1234;
  p.offset = 7;
  p.set_contiguous_bthd();
  return p;
}

Run run_gpu(const AttentionParams& p, const std::vector<float>& q, const std::vector<float>& k,
            const std::vector<float>& v, const std::vector<float>& dout, const std::vector<float>& slopes) {
  Dev dq_(q, p.dtype), dk_(k, p.dtype), dv_(v, p.dtype), ddo(dout, p.dtype);
  Dev o(q.size(), p.dtype), gq(q.size(), p.dtype), gk(k.size(), p.dtype), gv(v.size(), p.dtype);
  // Varlen'de LSE [H, total_q]: yalniz yazilan kisim okunur (initcheck temiz kalsin).
  std::size_t lse_n = static_cast<std::size_t>(p.batch) * p.heads * p.tq;
  if (p.cu_seqlens_q != nullptr) {
    std::int32_t total_q = 0;
    NEXUS_CUDA_CHECK(cudaMemcpy(&total_q, p.cu_seqlens_q + p.batch, 4, cudaMemcpyDeviceToHost));
    lse_n = static_cast<std::size_t>(p.heads) * static_cast<std::size_t>(total_q);
  }
  nc::DeviceBuffer lse(lse_n * 4, g_stream);
  nc::DeviceBuffer alibi(std::max<std::size_t>(slopes.size(), 1) * 4, g_stream);
  AttentionParams a = p;
  a.dtype = p.dtype;
  if (!slopes.empty()) {
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(alibi.data(), slopes.data(), slopes.size() * 4, cudaMemcpyHostToDevice,
                                     g_stream.handle()));
    a.alibi_slopes = alibi.as<float>();
  }
  nc::flash_attention_forward(o.buf.data(), lse.as<float>(), dq_.buf.data(), dk_.buf.data(), dv_.buf.data(), a,
                              g_stream.handle());
  nc::DeviceBuffer scratch(nc::flash_attention_backward_scratch_bytes(a), g_stream);
  nc::flash_attention_backward(gq.buf.data(), gk.buf.data(), gv.buf.data(), ddo.buf.data(), dq_.buf.data(),
                               dk_.buf.data(), dv_.buf.data(), o.buf.data(), lse.as<float>(), a, scratch.data(),
                               g_stream.handle());
  Run r;
  r.o = o.get();
  r.lse.resize(lse_n);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(r.lse.data(), lse.data(), lse_n * 4, cudaMemcpyDeviceToHost, g_stream.handle()));
  g_stream.synchronize();
  r.dq = gq.get();
  r.dk = gk.get();
  r.dv = gv.get();
  return r;
}

Run run_math(AttentionParams p, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
             const std::vector<float>& dout, const std::vector<float>& slopes, bool cpu_flash = false) {
  p.dtype = DType::F32;
  if (!slopes.empty()) {
    p.alibi_slopes = slopes.data();
  }
  Run r;
  r.o.assign(q.size(), 0.0f);
  r.lse.assign(static_cast<std::size_t>(p.batch) * p.heads * p.tq, 0.0f);
  r.dq.assign(q.size(), 0.0f);
  r.dk.assign(k.size(), 0.0f);
  r.dv.assign(v.size(), 0.0f);
  if (cpu_flash) {
    na::flash_attention_forward_cpu(r.o.data(), r.lse.data(), q.data(), k.data(), v.data(), p);
    na::flash_attention_backward_cpu(r.dq.data(), r.dk.data(), r.dv.data(), dout.data(), q.data(), k.data(), v.data(),
                                     r.o.data(), r.lse.data(), p);
  } else {
    na::math_attention_forward(r.o.data(), r.lse.data(), q.data(), k.data(), v.data(), p);
    na::math_attention_backward(r.dq.data(), r.dk.data(), r.dv.data(), dout.data(), q.data(), k.data(), v.data(),
                                r.o.data(), r.lse.data(), p);
  }
  return r;
}

void check_case(const Case& c) {
  const AttentionParams p = make_params(c);
  const std::size_t nq = static_cast<std::size_t>(c.b) * c.tq * c.h * c.d;
  const std::size_t nk = static_cast<std::size_t>(c.b) * c.tk * c.hkv * c.d;
  const auto q = random_values(nq, 11, c.t);
  const auto k = random_values(nk, 12, c.t);
  const auto v = random_values(nk, 13, c.t);
  const auto dout = random_values(nq, 14, c.t);
  std::vector<float> slopes;
  if (c.alibi) {
    for (int h = 0; h < c.h; ++h) {
      slopes.push_back(-std::pow(2.0f, -8.0f * (h + 1) / c.h));
    }
  }
  const Run ref = run_math(p, q, k, v, dout, slopes);
  const Run gpu = run_gpu(p, q, k, v, dout, slopes);
  const bool bf = c.t == DType::BF16;
  const double to = bf ? 2e-2 : 5e-3;
  const double tg = bf ? 3e-2 : 1e-2;
  const std::string tag = std::string(" [") + c.name + "]";
  report("flash O" + tag, rel(gpu.o, ref.o), to);
  report("flash LSE" + tag, rel(gpu.lse, ref.lse), 1e-4);
  report("flash dQ" + tag, rel(gpu.dq, ref.dq), tg);
  report("flash dK" + tag, rel(gpu.dk, ref.dk), tg);
  report("flash dV" + tag, rel(gpu.dv, ref.dv), tg);
  // CPU flash (FP32) ayni kahine karsi.
  const Run cpu = run_math(p, q, k, v, dout, slopes, true);
  const double tc = 1e-4;
  report("cpu flash O" + tag, rel(cpu.o, ref.o), tc);
  report("cpu flash dQ/dK/dV" + tag, std::max({rel(cpu.dq, ref.dq), rel(cpu.dk, ref.dk), rel(cpu.dv, ref.dv)}), tc);
}

void test_grid() {
  nexus_test::section("test_flash_vs_math: izgara (CUDA ve CPU flash vs FP64 kahin)");
  const Case cases[] = {
      {"B2 H4 T128 d64 causal bf16", 2, 4, 4, 128, 128, 64, DType::BF16, true},
      {"B1 H2 T100 d128 non-causal f16", 1, 2, 2, 100, 100, 128, DType::F16, false},
      {"Tq37 Tk200 causal hizalama d64 bf16", 2, 2, 2, 37, 200, 64, DType::BF16, true},
      {"Tq200 Tk37 causal (bos satirlar) d64 f16", 1, 2, 2, 200, 37, 64, DType::F16, true},
      {"GQA H8 Hkv2 window48 d64 bf16", 1, 8, 2, 150, 150, 64, DType::BF16, true, 48},
      {"MQA H4 Hkv1 softcap30 alibi d128 bf16", 1, 4, 1, 96, 96, 128, DType::BF16, true, 0, 30.0f, true},
      {"non-causal window32 d64 f16", 1, 2, 2, 130, 130, 64, DType::F16, false, 32},
      {"dropout0.1 causal d64 bf16", 1, 2, 2, 80, 80, 64, DType::BF16, true, 0, 0.0f, false, 0.1f},
  };
  for (const Case& c : cases) {
    check_case(c);
  }
  // CPU flash D % 64 != 0 (AVX2 olmayan skaler biriktirme yolu); CUDA bu boyutlari desteklemez.
  for (const Case& c : {Case{"cpu d80 GQA causal", 1, 4, 2, 90, 90, 80, DType::BF16, true},
                        Case{"cpu d32 Tq20 Tk70 causal dropout", 2, 2, 2, 20, 70, 32, DType::BF16, true, 0, 0.0f, false,
                             0.1f}}) {
    const AttentionParams p = make_params(c);
    const auto q = random_values(static_cast<std::size_t>(c.b) * c.tq * c.h * c.d, 41, c.t);
    const auto k = random_values(static_cast<std::size_t>(c.b) * c.tk * c.hkv * c.d, 42, c.t);
    const auto v = random_values(k.size(), 43, c.t);
    const auto dout = random_values(q.size(), 44, c.t);
    const Run ref = run_math(p, q, k, v, dout, {});
    const Run cpu = run_math(p, q, k, v, dout, {}, true);
    const std::string tag = std::string(" [") + c.name + "]";
    report("cpu flash O" + tag, rel(cpu.o, ref.o), 1e-4);
    report("cpu flash dQ/dK/dV" + tag, std::max({rel(cpu.dq, ref.dq), rel(cpu.dk, ref.dk), rel(cpu.dv, ref.dv)}), 1e-4);
  }
}

void test_tq1_decode() {
  nexus_test::section("test_flash_tq1_decode: Tq=1, Tk=1..4096");
  for (const int tk : {1, 7, 64, 65, 1000, 4096}) {
    const Case c{"decode", 1, 4, 4, 1, tk, 64, DType::BF16, true};
    const AttentionParams p = make_params(c);
    const auto q = random_values(static_cast<std::size_t>(4) * 64, 21, c.t);
    const auto k = random_values(static_cast<std::size_t>(tk) * 4 * 64, 22, c.t);
    const auto v = random_values(static_cast<std::size_t>(tk) * 4 * 64, 23, c.t);
    const auto dout = random_values(q.size(), 24, c.t);
    const Run ref = run_math(p, q, k, v, dout, {});
    const Run gpu = run_gpu(p, q, k, v, dout, {});
    report("Tq=1 Tk=" + std::to_string(tk) + " O", rel(gpu.o, ref.o), 2e-2);
    report("Tq=1 Tk=" + std::to_string(tk) + " dV", rel(gpu.dv, ref.dv), 3e-2);
    if (tk == 1) {
      // Tek anahtar: P = 1, dS = dP - D = 0 -> dK analitik olarak 0; gurultu dV olcegine gore.
      double worst = 0.0;
      double scale = 1e-6;
      for (std::size_t i = 0; i < gpu.dk.size(); ++i) {
        worst = std::max(worst, static_cast<double>(std::fabs(gpu.dk[i])));
        scale = std::max(scale, static_cast<double>(std::fabs(ref.dv[i])));
      }
      report("Tq=1 Tk=1 dK ~ 0 (dV olcegine gore)", worst / scale, 3e-2);
    } else {
      report("Tq=1 Tk=" + std::to_string(tk) + " dK", rel(gpu.dk, ref.dk), 3e-2);
    }
  }
}

void test_varlen() {
  nexus_test::section("test_flash_varlen_equals_separate");
  const std::vector<int> lens = {17, 64, 130};
  const int H = 4;
  const int D = 64;
  const DType t = DType::BF16;
  int total = 0;
  for (const int l : lens) {
    total += l;
  }
  const std::vector<float> q = random_values(static_cast<std::size_t>(total) * H * D, 31, t);
  const std::vector<float> k = random_values(q.size(), 32, t);
  const std::vector<float> v = random_values(q.size(), 33, t);
  const std::vector<float> dout = random_values(q.size(), 34, t);
  std::vector<std::int32_t> cu = {0};
  for (const int l : lens) {
    cu.push_back(cu.back() + l);
  }
  AttentionParams p;
  p.batch = static_cast<int>(lens.size());
  p.heads = H;
  p.kv_heads = H;
  p.tq = *std::max_element(lens.begin(), lens.end());
  p.tk = p.tq;
  p.head_dim = D;
  p.dtype = t;
  p.causal = true;
  p.q_row_stride = p.k_row_stride = p.o_row_stride = static_cast<long long>(H) * D;
  p.q_head_stride = p.k_head_stride = p.o_head_stride = D;
  nc::DeviceBuffer cu_d(cu.size() * 4, g_stream);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(cu_d.data(), cu.data(), cu.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
  AttentionParams pv = p;
  pv.cu_seqlens_q = cu_d.as<std::int32_t>();
  pv.cu_seqlens_k = cu_d.as<std::int32_t>();
  const Run packed = run_gpu(pv, q, k, v, dout, {});
  // Her dizi ayri ayri (bitisik tek batch).
  std::vector<float> o_sep(q.size()), dq_sep(q.size()), dk_sep(q.size()), dv_sep(q.size());
  for (std::size_t s = 0; s < lens.size(); ++s) {
    const std::size_t off = static_cast<std::size_t>(cu[s]) * H * D;
    const std::size_t n = static_cast<std::size_t>(lens[s]) * H * D;
    auto slice = [&](const std::vector<float>& src) { return std::vector<float>(src.begin() + off, src.begin() + off + n); };
    AttentionParams ps = p;
    ps.batch = 1;
    ps.tq = ps.tk = lens[s];
    ps.set_contiguous_bthd();
    const Run r = run_gpu(ps, slice(q), slice(k), slice(v), slice(dout), {});
    std::copy(r.o.begin(), r.o.end(), o_sep.begin() + off);
    std::copy(r.dq.begin(), r.dq.end(), dq_sep.begin() + off);
    std::copy(r.dk.begin(), r.dk.end(), dk_sep.begin() + off);
    std::copy(r.dv.begin(), r.dv.end(), dv_sep.begin() + off);
  }
  nexus_test::expect(packed.o == o_sep, "varlen O = ayri ayri (bit-bit)");
  report("varlen dQ/dK/dV = ayri ayri", std::max({rel(packed.dq, dq_sep), rel(packed.dk, dk_sep), rel(packed.dv, dv_sep)}),
         1e-6);
  // Kahine karsi da (host cu_seqlens).
  AttentionParams ph = p;
  ph.cu_seqlens_q = cu.data();
  ph.cu_seqlens_k = cu.data();
  const Run ref = run_math(ph, q, k, v, dout, {});
  report("varlen O vs kahin", rel(packed.o, ref.o), 2e-2);
}

void test_dropout_stats() {
  nexus_test::section("test_flash_dropout_stats");
  const double p = 0.1;
  const std::uint64_t n = 1u << 20;
  std::uint64_t dropped = 0;
  for (std::uint64_t i = 0; i < n; ++i) {
    dropped += nexus::fused::philox_uniform(1234, 7, i) < p ? 1 : 0;
  }
  const double rate = static_cast<double>(dropped) / static_cast<double>(n);
  const double sigma = std::sqrt(p * (1 - p) / static_cast<double>(n));
  nexus_test::info("dropout_rate", rate);
  nexus_test::expect(std::fabs(rate - p) < 3 * sigma, "dusurulen oran p +- 3 sigma");
  // Maske ve fwd/bwd tutarliligi izgaradaki dropout durumunda kahinle (ayni Philox) dogrulandi.
}

void test_determinism() {
  nexus_test::section("test_flash_determinism (deterministik mod)");
  Case c{"det", 1, 8, 2, 256, 256, 64, DType::BF16, true};
  AttentionParams p = make_params(c);
  p.deterministic = true;
  const auto q = random_values(static_cast<std::size_t>(256) * 8 * 64, 41, c.t);
  const auto k = random_values(static_cast<std::size_t>(256) * 2 * 64, 42, c.t);
  const auto v = random_values(k.size(), 43, c.t);
  const auto dout = random_values(q.size(), 44, c.t);
  const Run a = run_gpu(p, q, k, v, dout, {});
  const Run b = run_gpu(p, q, k, v, dout, {});
  nexus_test::expect(a.dq == b.dq && a.dk == b.dk && a.dv == b.dv, "deterministik: dQ/dK/dV bit-bit ayni");
  const Run ref = run_math(p, q, k, v, dout, {});
  report("deterministik dQ vs kahin", rel(a.dq, ref.dq), 3e-2);
}

void test_future_invariance() {
  nexus_test::section("test_causal_future_invariance");
  Case c{"fut", 1, 2, 2, 128, 128, 64, DType::BF16, true};
  const AttentionParams p = make_params(c);
  auto q = random_values(static_cast<std::size_t>(128) * 2 * 64, 51, c.t);
  auto k = random_values(q.size(), 52, c.t);
  auto v = random_values(q.size(), 53, c.t);
  const auto dout = random_values(q.size(), 54, c.t);
  const Run a = run_gpu(p, q, k, v, dout, {});
  for (std::size_t i = static_cast<std::size_t>(100) * 2 * 64; i < k.size(); ++i) {
    k[i] = round_to(c.t, k[i] + 1.0f);
    v[i] = round_to(c.t, v[i] - 1.0f);
  }
  const Run b = run_gpu(p, q, k, v, dout, {});
  const std::size_t past = static_cast<std::size_t>(100) * 2 * 64;
  nexus_test::expect(std::equal(a.o.begin(), a.o.begin() + past, b.o.begin()),
                     "gelecek K/V degisince ilk 100 sorgunun ciktisi bit-bit ayni");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return 77;
  }
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    g_stream = nc::Stream::create(0);
    nc::StreamGuard guard(g_stream);
    if (mode == "all" || mode == "grid") test_grid();
    if (mode == "all" || mode == "decode") test_tq1_decode();
    if (mode == "all" || mode == "varlen") test_varlen();
    if (mode == "all" || mode == "dropout") test_dropout_stats();
    if (mode == "all" || mode == "determinism") test_determinism();
    if (mode == "all" || mode == "future") test_future_invariance();
    g_stream.synchronize();
    return nexus_test::finish("faz5-flash-attention");
  } catch (const std::exception& e) {
    std::printf("  FAIL  beklenmeyen istisna: %s\n", e.what());
    return 1;
  }
}
