/**
 * @file test_phase18_cpu.cpp
 * @brief FAZ 18: CPU arka ucu — ozellik algilama, AVX2/AVX-512 GEMM (FP32 / BF16 / FP16 / INT8 / INT4 / VNNI),
 *        sabitli havuz / NUMA / buyuk sayfa, CPU FlashAttention (FP32 / BF16 / FP16), CPU Adam / ZeRO-Offload,
 *        CPU egitimi, FP32/BF16/INT8/INT4 LLM cikarimi ve CPU JIT fuzyon derleyicisi (18.6).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "nexus/attention/cpu.hpp"
#include "nexus/autograd/autograd.hpp"
#include "nexus/compile/compile.hpp"
#include "nexus/core/half.hpp"
#include "nexus/cpu/adam.hpp"
#include "nexus/cpu/features.hpp"
#include "nexus/cpu/gemm.hpp"
#include "nexus/cpu/threads.hpp"
#include "nexus/fused/training.hpp"
#include "nexus/interop/hf.hpp"
#include "nexus/lm/model.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace cpu = nexus::cpu;
namespace lm = nexus::lm;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::vector<float> randv(std::size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

float bf16_to_f(std::uint16_t h) {
  const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
float f16_to_f(std::uint16_t h) {
  const std::uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  std::uint32_t o;
  if (e == 0) {
    if (m == 0) {
      o = s;
    } else {
      int ee = -1;
      std::uint32_t mm = m;
      do {
        ++ee;
        mm <<= 1;
      } while ((mm & 0x400) == 0);
      o = s | ((127 - 15 - ee) << 23) | ((mm & 0x3FF) << 13);
    }
  } else if (e == 31) {
    o = s | 0x7F800000u | (m << 13);
  } else {
    o = s | ((e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &o, 4);
  return f;
}

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

void test_features() {
  std::printf("  info  %s\n", cpu::describe().c_str());
  std::printf("  info  kullanilan GEMM ISA yolu: %s (KernelIsa=%d)\n", cpu::isa(),
              static_cast<int>(cpu::active_gemm_isa()));
  const auto& f = cpu::features();
  nexus_test::expect(f.avx2 && f.fma && f.physical >= 1 && f.logical >= f.physical &&
                         static_cast<int>(f.core_of.size()) == f.logical,
                     "CPUID: AVX2 + FMA algilandi, topoloji tutarli");
  nexus_test::expect(cpu::active_gemm_isa() >= cpu::KernelIsa::Avx2Fma,
                     "active_gemm_isa en az Avx2Fma secildi");
}

struct Case {
  int M, N, K;
  cpu::BType bt;
  bool trans;
  float alpha, beta;
  bool bias;
};

double run_case(const Case& c, cpu::Pool* pool, std::vector<float>* out = nullptr) {
  const auto A = randv(static_cast<std::size_t>(c.M) * c.K, 1);
  const auto Bf = randv(static_cast<std::size_t>(c.K) * c.N, 2);
  const auto C0 = randv(static_cast<std::size_t>(c.M) * c.N, 3);
  const auto bias = randv(static_cast<std::size_t>(c.N), 4);
  std::vector<std::uint16_t> B16;
  std::vector<float> Bdec = Bf;  // B'nin acilmis degeri (referans icin)
  if (c.bt == cpu::BType::BF16) {
    B16 = cpu::to_bf16(Bf.data(), Bf.size());
    for (std::size_t i = 0; i < Bf.size(); ++i) Bdec[i] = bf16_to_f(B16[i]);
  } else if (c.bt == cpu::BType::F16) {
    B16 = cpu::to_f16(Bf.data(), Bf.size());
    for (std::size_t i = 0; i < Bf.size(); ++i) Bdec[i] = f16_to_f(B16[i]);
  }
  std::vector<float> C = C0;
  cpu::GemmArgs g;
  g.M = c.M;
  g.N = c.N;
  g.K = c.K;
  g.A = A.data();
  g.lda = c.K;
  g.B = c.bt == cpu::BType::F32 ? static_cast<const void*>(Bf.data()) : static_cast<const void*>(B16.data());
  g.btype = c.bt;
  g.b_trans = c.trans;
  g.ldb = c.trans ? c.K : c.N;
  g.C = C.data();
  g.ldc = c.N;
  g.alpha = c.alpha;
  g.beta = c.beta;
  g.bias = c.bias ? bias.data() : nullptr;
  cpu::gemm(g, pool);
  double err = 0.0, scale = 0.0;
  for (int m = 0; m < c.M; ++m)
    for (int n = 0; n < c.N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < c.K; ++k)
        acc += static_cast<double>(A[static_cast<std::size_t>(m) * c.K + k]) *
               Bdec[c.trans ? static_cast<std::size_t>(n) * c.K + k : static_cast<std::size_t>(k) * c.N + n];
      const double ref = c.alpha * acc + c.beta * C0[static_cast<std::size_t>(m) * c.N + n] +
                         (c.bias ? bias[static_cast<std::size_t>(n)] : 0.0);
      err = std::max(err, std::fabs(ref - C[static_cast<std::size_t>(m) * c.N + n]));
      scale = std::max(scale, std::fabs(ref));
    }
  if (out) *out = C;
  return err / std::max(scale, 1e-30);
}

void test_gemm() {
  std::vector<Case> cases;
  for (cpu::BType bt : {cpu::BType::F32, cpu::BType::BF16, cpu::BType::F16})
    for (bool tr : {false, true}) {
      cases.push_back({197, 131, 301, bt, tr, 1.0f, 0.0f, false});     // kenarlar (4 / 16 / 512'nin kati degil)
      cases.push_back({96, 2048 + 48, 513, bt, tr, 0.5f, 1.0f, true}); // KC sinirini asar, beta + bias
      cases.push_back({3, 1000, 777, bt, tr, 1.0f, 0.0f, true});       // GEMV yolu
      cases.push_back({1, 37, 5, bt, tr, 2.0f, 0.5f, false});          // kucuk
    }
  double worst = 0.0;
  for (const Case& c : cases) {
    const double e = run_case(c, nullptr);
    worst = std::max(worst, e);
    if (e > 1e-5)
      std::printf("  info  KOTU: M %d N %d K %d tip %d trans %d -> %.2e\n", c.M, c.N, c.K, static_cast<int>(c.bt), c.trans, e);
  }
  std::printf("  info  %zu durum (FP32/BF16/FP16 x [K,N]/[N,K] x kenar / KC asimi / GEMV / beta + bias): en kotu goreli %.2e\n",
              cases.size(), worst);
  nexus_test::expect(worst <= 1e-5, "test_cpu_gemm_avx2_vs_ref: tum tipler double referansla <= 1e-5");

  // Belirlenimcilik: 1 / 3 / 8 isci
  const Case big{301, 517, 1031, cpu::BType::BF16, true, 1.0f, 0.0f, true};
  std::vector<float> c1, c3, c8;
  cpu::Pool p1(1), p3(3), p8(8);
  (void)run_case(big, &p1, &c1);
  (void)run_case(big, &p3, &c3);
  (void)run_case(big, &p8, &c8);
  nexus_test::expect(c1 == c3 && c1 == c8, "GEMM sonucu isci sayisindan bagimsiz (1 / 3 / 8 isci bit bit)");

  // INT8 (PackedS8)
  const int M = 37, N = 83, K = 301;
  const auto A = randv(static_cast<std::size_t>(M) * K, 5), W = randv(static_cast<std::size_t>(N) * K, 6);
  const cpu::PackedS8 P = cpu::pack_s8(W.data(), N, K, true);
  std::vector<std::int8_t> Aq(static_cast<std::size_t>(M) * P.Kp);
  std::vector<float> sa(static_cast<std::size_t>(M)), C(static_cast<std::size_t>(M) * N);
  cpu::quantize_rows_s8(A.data(), M, K, K, Aq.data(), P.Kp, sa.data());
  cpu::gemm_s8(Aq.data(), sa.data(), M, P, C.data(), N);
  double ierr = 0.0, qerr = 0.0, qs = 0.0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      std::int64_t acc = 0;
      double exact = 0.0;
      for (int k = 0; k < K; ++k) {
        const int wq = P.data[(static_cast<std::size_t>(n / 16) * (P.Kp / 4) + k / 4) * 64 + static_cast<std::size_t>(n % 16) * 4 + k % 4];
        acc += static_cast<std::int64_t>(Aq[static_cast<std::size_t>(m) * P.Kp + k]) * wq;
        exact += static_cast<double>(A[static_cast<std::size_t>(m) * K + k]) * W[static_cast<std::size_t>(n) * K + k];
      }
      const double ref = static_cast<double>(acc) * sa[static_cast<std::size_t>(m)] * P.scale[static_cast<std::size_t>(n)];
      ierr = std::max(ierr, std::fabs(ref - C[static_cast<std::size_t>(m) * N + n]) / std::max(1e-6, std::fabs(ref)));
      qerr = std::max(qerr, std::fabs(exact - C[static_cast<std::size_t>(m) * N + n]));
      qs = std::max(qs, std::fabs(exact));
    }
  std::printf("  info  INT8 (s8 x s8, maddubs): tam sayi referansla goreli %.1e; FP32 dogru degere gore nicemleme hatasi %.2e\n", ierr,
              qerr / qs);
  nexus_test::expect(ierr <= 1e-6 && qerr / qs < 0.03, "INT8 GEMM == tam sayi referans (doygunluk yok); nicemleme hatasi < %3");

  // INT4 (PackedQ4)
  const cpu::PackedQ4 P4 = cpu::pack_q4(W.data(), N, K, true);
  std::vector<std::int8_t> Aq4(static_cast<std::size_t>(M) * P4.Kp);
  std::vector<float> sa4(static_cast<std::size_t>(M)), C4(static_cast<std::size_t>(M) * N);
  cpu::quantize_rows_s8(A.data(), M, K, K, Aq4.data(), P4.Kp, sa4.data());
  cpu::gemm_q4(Aq4.data(), sa4.data(), M, P4, C4.data(), N);
  std::vector<float> C_exact(static_cast<std::size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double exact = 0.0;
      for (int k = 0; k < K; ++k) exact += static_cast<double>(A[static_cast<std::size_t>(m) * K + k]) * W[static_cast<std::size_t>(n) * K + k];
      C_exact[static_cast<std::size_t>(m) * N + n] = static_cast<float>(exact);
    }
  const double cos4 = cosine_similarity(C4, C_exact);
  std::printf("  info  INT4 (q4 x s8, nibble maddubs): FP32 referansla kosinus benzerligi %.5f\n", cos4);
  nexus_test::expect(cos4 > 0.985, "INT4 GEMM FP32 referansla kosinus benzerligi > 0.985");
}

void test_gemm_avx512() {
  const auto& f = cpu::features();
  std::printf("  info  AVX-512F=%d AVX-512BF16=%d AVX-512VNNI=%d AMX_BF16=%d (aktif ISA: %s)\n",
              f.avx512f ? 1 : 0, f.avx512bf16 ? 1 : 0, f.avx512vnni ? 1 : 0, f.amx_bf16 ? 1 : 0, cpu::isa());
  // AVX-512 BF16 (vdpbf16ps) / AMX (tdpbf16ps) ciftli VNNI duzeni dogrulamasi:
  const int M = 29, N = 67, K = 195;
  const auto A = randv(static_cast<std::size_t>(M) * K, 31);
  const auto W = randv(static_cast<std::size_t>(N) * K, 32);
  const auto bias = randv(static_cast<std::size_t>(N), 33);
  const cpu::PackedBf16Vnni pv = cpu::pack_bf16_vnni(W.data(), N, K, /*b_trans=*/true);
  std::vector<float> C(static_cast<std::size_t>(M) * N, 0.0f);
  cpu::gemm_bf16_vnni(A.data(), M, K, pv, C.data(), N, bias.data());

  const auto Ab = cpu::to_bf16(A.data(), A.size());
  const auto Wb = cpu::to_bf16(W.data(), W.size());
  double err = 0.0, scale = 0.0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(bf16_to_f(Ab[static_cast<std::size_t>(m) * K + k])) *
               static_cast<double>(bf16_to_f(Wb[static_cast<std::size_t>(n) * K + k]));
      }
      const double ref = acc + bias[static_cast<std::size_t>(n)];
      err = std::max(err, std::fabs(ref - C[static_cast<std::size_t>(m) * N + n]));
      scale = std::max(scale, std::fabs(ref));
    }
  }
  const double rel = err / std::max(scale, 1e-30);
  std::printf("  info  test_cpu_gemm_avx512_vs_ref (PackedBf16Vnni vdpbf16ps/tdpbf16ps semantigi): en kotu goreli %.2e\n", rel);
  nexus_test::expect(rel <= 1e-5, "test_cpu_gemm_avx512_vs_ref: BF16 VNNI ciftli nokta carpimi referansla <= 1e-5");
}

void test_numa() {
  const auto& f = cpu::features();
  cpu::Pool& p = cpu::Pool::instance();
  const auto seen = p.observed_cpus();
  bool ok = true;
  std::set<int> cores;
  std::printf("  info  %d isci:", p.size());
  for (int w = 0; w < p.size(); ++w) {
    const int pin = p.pinned_cpu(w), got = seen[static_cast<std::size_t>(w)];
    std::printf(" %d->%d(%d)", w, pin, got);
    ok = ok && pin == got;
    cores.insert(f.core_of[static_cast<std::size_t>(pin)]);
  }
  std::printf("\n");
  nexus_test::expect(ok, "test_numa_affinity: her isci sabitlendigi mantiksal islemcide calisiyor (gozlem)");
  nexus_test::expect(static_cast<int>(cores.size()) == std::min(p.size(), f.physical),
                     "isciler farkli fiziksel cekirdeklerde (SMT esi paylasilmiyor)");
  bool nodes_ok = true;
  for (int w = 0; w < p.size(); ++w) nodes_ok = nodes_ok && f.node_of[static_cast<std::size_t>(p.pinned_cpu(w))] < f.numa_nodes;
  std::printf("  info  NUMA dugumu: %d (tek dugum: dugum basina yerlesim olculemez)\n", f.numa_nodes);
  nexus_test::expect(nodes_ok, "isci -> NUMA dugumu esleme tutarli");
  cpu::AllocOptions o;
  o.large_pages = true;
  o.numa_node = 0;
  cpu::Buffer b = cpu::alloc(std::size_t{64} << 20, o);
  std::printf("  info  64 MB buyuk sayfa istegi: %s%s%s\n", b.large_pages ? "2 MB sayfa ALINDI" : "4 KB sayfa",
              b.reason.empty() ? "" : " - ", b.reason.c_str());
  bool zero = true;
  for (std::size_t i = 0; i < b.bytes; i += 4096) zero = zero && static_cast<char*>(b.data)[i] == 0;
  nexus_test::expect(b.data != nullptr && zero && (b.large_pages || !b.reason.empty()),
                     "ayirici: first-touch sifirlandi; buyuk sayfa ya alindi ya da neden acikca bildirildi");
}

// ---- CPU FlashAttention vs math (FP64 kahin) ----------------------------------------------
double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
  double e = 0.0, s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    e = std::max(e, std::fabs(static_cast<double>(a[i]) - b[i]));
    s = std::max(s, std::fabs(static_cast<double>(b[i])));
  }
  return s < 1e-6 ? e : e / s;
}

void test_flash() {
  struct Case {
    const char* name;
    int B, H, Hk, tq, tk, D;
    bool causal;
    int window;
    float softcap;
    bool alibi;
  };
  const Case cases[] = {
      {"tam T=257 D=64", 2, 4, 4, 257, 257, 64, false, 0, 0.0f, false},
      {"causal T=130 GQA 4/2", 1, 4, 2, 130, 130, 64, true, 0, 0.0f, false},
      {"causal tq=37 tk=200", 1, 2, 2, 37, 200, 64, true, 0, 0.0f, false},
      {"pencere 48 causal", 1, 2, 1, 190, 190, 128, true, 48, 0.0f, false},
      {"pencere 33 cift yon", 1, 2, 2, 150, 150, 64, false, 33, 0.0f, false},
      {"softcap + alibi", 1, 3, 3, 100, 100, 64, true, 0, 30.0f, true},
      {"D=40 (skaler PV yolu)", 1, 2, 2, 70, 70, 40, true, 0, 0.0f, false},
      {"T=1", 1, 2, 2, 1, 1, 64, true, 0, 0.0f, false},
  };
  double worst = 0.0;
  for (const Case& c : cases) {
    nexus::AttentionParams p;
    p.batch = c.B;
    p.heads = c.H;
    p.kv_heads = c.Hk;
    p.tq = c.tq;
    p.tk = c.tk;
    p.head_dim = c.D;
    p.dtype = nexus::DType::F32;
    p.causal = c.causal;
    p.window = c.window;
    p.softcap = c.softcap;
    std::vector<float> slopes(static_cast<std::size_t>(c.H));
    for (int h = 0; h < c.H; ++h) slopes[static_cast<std::size_t>(h)] = 0.05f * (h + 1);
    if (c.alibi) p.alibi_slopes = slopes.data();
    p.set_contiguous_bthd();
    const std::size_t nq = static_cast<std::size_t>(c.B) * c.tq * c.H * c.D, nk = static_cast<std::size_t>(c.B) * c.tk * c.Hk * c.D;
    const auto q = randv(nq, 21), k = randv(nk, 22), v = randv(nk, 23), dout = randv(nq, 24);
    const std::size_t nl = static_cast<std::size_t>(c.B) * c.H * c.tq;
    std::vector<float> o0(nq), o1(nq), l0(nl), l1(nl), dq0(nq), dq1(nq), dk0(nk), dk1(nk), dv0(nk), dv1(nk);
    nexus::attention::math_attention_forward(o0.data(), l0.data(), q.data(), k.data(), v.data(), p);
    nexus::attention::math_attention_backward(dq0.data(), dk0.data(), dv0.data(), dout.data(), q.data(), k.data(), v.data(),
                                              o0.data(), l0.data(), p);
    nexus::attention::flash_attention_forward_cpu(o1.data(), l1.data(), q.data(), k.data(), v.data(), p);
    nexus::attention::flash_attention_backward_cpu(dq1.data(), dk1.data(), dv1.data(), dout.data(), q.data(), k.data(),
                                                   v.data(), o1.data(), l1.data(), p);
    const double e = std::max({max_rel(o1, o0), max_rel(l1, l0), max_rel(dq1, dq0), max_rel(dk1, dk0), max_rel(dv1, dv0)});
    std::printf("  info  %-24s O/LSE/dQ/dK/dV en kotu bagil hata %.2e\n", c.name, e);
    worst = std::max(worst, e);
  }
  nexus_test::expect(worst <= 1e-4, "test_cpu_flash_attn_vs_ref: F32 ileri + geri <= 1e-4 (8 durum)");

  // BF16 girdi/cikti testi (18.4: yuklemede FP32'ye genislet, FP32 biriktir, cikista BF16'ya daralt)
  {
    nexus::AttentionParams pb;
    pb.batch = 1;
    pb.heads = 4;
    pb.kv_heads = 2;
    pb.tq = 128;
    pb.tk = 128;
    pb.head_dim = 64;
    pb.dtype = nexus::DType::BF16;
    pb.causal = true;
    pb.set_contiguous_bthd();
    const std::size_t nq = 128 * 4 * 64, nk = 128 * 2 * 64, nl = 4 * 128;
    const auto qf = randv(nq, 41), kf = randv(nk, 42), vf = randv(nk, 43), do_f = randv(nq, 44);
    const auto qb = cpu::to_bf16(qf.data(), nq), kb = cpu::to_bf16(kf.data(), nk),
               vb = cpu::to_bf16(vf.data(), nk), dob = cpu::to_bf16(do_f.data(), nq);
    std::vector<std::uint16_t> ob0(nq), ob1(nq), dqb0(nq), dqb1(nq), dkb0(nk), dkb1(nk), dvb0(nk), dvb1(nk);
    std::vector<float> l0(nl), l1(nl);
    nexus::attention::math_attention_forward(ob0.data(), l0.data(), qb.data(), kb.data(), vb.data(), pb);
    nexus::attention::math_attention_backward(dqb0.data(), dkb0.data(), dvb0.data(), dob.data(), qb.data(), kb.data(),
                                              vb.data(), ob0.data(), l0.data(), pb);
    nexus::attention::flash_attention_forward_cpu(ob1.data(), l1.data(), qb.data(), kb.data(), vb.data(), pb);
    nexus::attention::flash_attention_backward_cpu(dqb1.data(), dkb1.data(), dvb1.data(), dob.data(), qb.data(), kb.data(),
                                                   vb.data(), ob1.data(), l1.data(), pb);
    auto dec = [](const std::vector<std::uint16_t>& v) {
      std::vector<float> o(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) o[i] = bf16_to_f(v[i]);
      return o;
    };
    const double eb = std::max({max_rel(dec(ob1), dec(ob0)), max_rel(l1, l0), max_rel(dec(dqb1), dec(dqb0)),
                                max_rel(dec(dkb1), dec(dkb0)), max_rel(dec(dvb1), dec(dvb0))});
    std::printf("  info  BF16 FlashAttention (T=128 GQA 4/2): math kahinine gore en kotu bagil hata %.2e\n", eb);
    nexus_test::expect(eb <= 2e-2, "test_cpu_flash_attn_vs_ref: BF16 ileri + geri <= 2e-2");
  }
}

int bench_attn() {
  for (const int threads : {1, 8}) {
    for (const int T : {512, 1024, 2048, 4096}) {
      for (const bool causal : {false, true}) {
        nexus::AttentionParams p;
        p.batch = 1;
        p.heads = 8;
        p.kv_heads = 8;
        p.tq = T;
        p.tk = T;
        p.head_dim = 64;
        p.dtype = nexus::DType::F32;
        p.causal = causal;
        p.set_contiguous_bthd();
        const std::size_t n = static_cast<std::size_t>(T) * 8 * 64;
        const auto q = randv(n, 1), k = randv(n, 2), v = randv(n, 3), dout = randv(n, 4);
        std::vector<float> o(n), lse(static_cast<std::size_t>(8) * T), dq(n), dk(n), dv(n);
        auto fwd = [&] { nexus::attention::flash_attention_forward_cpu(o.data(), lse.data(), q.data(), k.data(), v.data(), p, threads); };
        auto bwd = [&] {
          nexus::attention::flash_attention_backward_cpu(dq.data(), dk.data(), dv.data(), dout.data(), q.data(), k.data(), v.data(),
                                                         o.data(), lse.data(), p, threads);
        };
        auto med = [&](auto&& fn) {
          fn();
          std::vector<double> t;
          for (int r = 0; r < 5; ++r) {
            const auto t0 = Clock::now();
            fn();
            t.push_back(ms_since(t0));
          }
          std::sort(t.begin(), t.end());
          return t[2];
        };
        const double tf = med(fwd), tfb = med([&] {
          fwd();
          bwd();
        });
        std::printf("{\"mode\": \"bench_cpu_flash\", \"threads\": %d, \"T\": %d, \"causal\": %s, \"fwd_ms\": %.3f, \"fwd_bwd_ms\": %.3f}\n",
                    threads, T, causal ? "true" : "false", tf, tfb);
      }
    }
  }
  return 0;
}

// ---- CPU egitim paritesi: yeni AVX2 GEMM vs eski skaler GEMM (ayni model, ayni veri) -------
struct TrainRun {
  std::vector<float> loss;
  double ms_per_step = 0.0;
};

TrainRun train_lm(bool reference, int steps) {
  cpu::set_reference_gemm(reference);
  lm::LmConfig c;
  c.name = "faz18_train";
  c.vocab_size = 256;
  c.n_layer = 4;
  c.d_model = 256;
  c.n_head = 4;
  c.d_ff = 1024;
  c.max_position = 128;
  c.init = lm::InitKind::Gpt2;
  lm::LanguageModel model(c);
  nexus::fused::AdamWConfig ac;
  ac.lr = 1e-3;
  ac.beta2 = 0.95;
  ac.weight_decay = 0.1;
  ac.max_grad_norm = 1.0;
  lm::AdamW opt(model, ac);
  const int B = 4, T = 128;
  std::vector<std::array<int, 4>> succ(256);
  std::uint64_t st = 0x243F6A8885A308D3ULL;
  auto next = [&st]() {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return st >> 33;
  };
  for (auto& row : succ)
    for (int& x : row) x = static_cast<int>(next() % 256);
  TrainRun r;
  double total = 0.0;
  for (int s = 0; s < steps; ++s) {
    std::vector<std::int32_t> toks(static_cast<std::size_t>(B) * T);
    for (int b = 0; b < B; ++b) {
      int x = static_cast<int>(next() % 256);
      for (int t = 0; t < T; ++t) {
        toks[static_cast<std::size_t>(b) * T + t] = x;
        const std::uint64_t u = next() % 100;
        x = succ[static_cast<std::size_t>(x)][u < 50 ? 0 : (u < 75 ? 1 : (u < 90 ? 2 : 3))];
      }
    }
    const auto t0 = Clock::now();
    model.zero_grad();
    const nexus::autograd::Variable loss = model.loss(toks, B, T);
    loss.backward();
    (void)opt.step();
    if (s > 0) total += ms_since(t0);
    r.loss.push_back(loss.values()[0]);
  }
  cpu::set_reference_gemm(false);
  r.ms_per_step = total / std::max(1, steps - 1);
  return r;
}

void test_train() {
#if defined(__SANITIZE_ADDRESS__)
  const int steps = 8;
#else
  const int steps = 60;
#endif
  const TrainRun ref = train_lm(true, steps), neu = train_lm(false, steps);
  double worst = 0.0;
  for (int s = 0; s < steps; ++s)
    worst = std::max(worst, std::fabs(static_cast<double>(neu.loss[static_cast<std::size_t>(s)]) - ref.loss[static_cast<std::size_t>(s)]) /
                                ref.loss[static_cast<std::size_t>(s)]);
  std::printf("  info  4 katman d=256 T=128 B=4, %d adim: kayip %.4f -> %.4f (eski) / %.4f (yeni); en kotu adim bagil fark %.2e\n",
              steps, ref.loss[0], ref.loss.back(), neu.loss.back(), worst);
  std::printf("  info  adim suresi: eski skaler GEMM %.1f ms, AVX2 GEMM %.1f ms (%.2fx)\n",
              ref.ms_per_step, neu.ms_per_step, ref.ms_per_step / neu.ms_per_step);
  nexus_test::expect(neu.loss[0] == ref.loss[0] || std::fabs(neu.loss[0] - ref.loss[0]) / ref.loss[0] < 1e-6,
                     "ilk kayip ayni (ayni baslatma, yalniz toplama sirasi farkli)");
  nexus_test::expect(worst < 1e-3, "test_cpu_train_parity: kayip egrisi eski GEMM'e <= 1e-3 bagil");
  nexus_test::expect(neu.loss.back() < (steps >= 60 ? 0.7f : 0.98f) * neu.loss[0], "yeni GEMM ile model ogreniyor");
}

// ---- FAZ 18.5: FP32 / BF16 / INT8 / INT4 uctan uca dil modeli CPU cikarim paritesi -------
void test_quantized_inference() {
  lm::LmConfig c;
  c.name = "faz18_quant_lm";
  c.vocab_size = 512;
  c.n_layer = 4;
  c.d_model = 256;
  c.n_head = 8;
  c.n_kv_head = 2;
  c.d_ff = 768;
  c.max_position = 128;
  c.norm = lm::NormKind::RmsNorm;
  c.mlp = lm::MlpKind::SwiGlu;
  c.pos = lm::PosKind::Rope;
  c.init = lm::InitKind::Llama;
  lm::LanguageModel m(c);

  const std::vector<std::int32_t> prompt = {1, 17, 42, 99, 128, 255, 300, 12};
  nexus::autograd::NoGradGuard ng;

  // Tam gecis (cache = nullptr, Auto mod) vs F32 / BF16 / INT8 / INT4 KV-cache artimli cozme
  const auto full_logits = m.logits(prompt, 1, static_cast<int>(prompt.size())).values();
  const int Vp = c.padded_vocab();
  const std::vector<float> ref_last(full_logits.end() - Vp, full_logits.end());

  struct ModeInfo {
    lm::CpuWeightMode mode;
    const char* name;
    double min_cos;
  };
  const ModeInfo modes[] = {
      {lm::CpuWeightMode::F32, "F32", 0.999999},
      {lm::CpuWeightMode::BF16, "BF16", 0.9995},
      {lm::CpuWeightMode::INT8, "INT8", 0.998},
      {lm::CpuWeightMode::INT4, "INT4", 0.955},
  };
  std::size_t f32_bytes = 0;
  for (const auto& mi : modes) {
    m.prepare_cpu_inference(mi.mode);
    const std::size_t wbytes = m.cpu_inference_weight_bytes();
    if (mi.mode == lm::CpuWeightMode::F32) f32_bytes = wbytes;
    lm::KvCache cache = m.make_cache();
    lm::ForwardOptions fo;
    fo.cache = &cache;
    // Ilk 7 token prefill, 8. token tek-token decode!
    std::vector<std::int32_t> pre(prompt.begin(), prompt.end() - 1);
    (void)m.logits(pre, 1, static_cast<int>(pre.size()), fo);
    const auto last_v = m.logits({prompt.back()}, 1, 1, fo).values();
    const double cos = cosine_similarity(last_v, ref_last);
    std::printf("  info  CPU cikarim [%-4s] agirlik %.2f MB, son-token logit kosinus benzerligi = %.6f\n",
                mi.name, static_cast<double>(wbytes) / (1024.0 * 1024.0), cos);
    nexus_test::expect(cos >= mi.min_cos, std::string("test_cpu_quantized_inference: ") + mi.name + " kosinus benzerligi");
    if (mi.mode == lm::CpuWeightMode::INT4) {
      nexus_test::expect(wbytes * 3 < f32_bytes, "INT4 agirlik bellegi F32'nin 1/3'unden kucuk");
    }
  }
}

// ---- FAZ 18.6: CPU JIT fuzyon derleyicisi (AVX-512/AVX2 kod uretimi + paralel yurutucu) ---
void test_compile_fusion() {
  lm::LmConfig c;
  c.name = "faz18_compile_cpu";
  c.vocab_size = 256;
  c.n_layer = 2;
  c.d_model = 256;
  c.n_head = 4;
  c.n_kv_head = 2;
  c.d_ff = 512;
  c.max_position = 128;
  lm::LanguageModel m(c);

  nexus::compile::CompileOptions opt;
  opt.mode = "max-autotune";
  nexus::compile::CompiledLm cm = nexus::compile::compile(m, opt);

  const int B = 2, T = 64;
  std::vector<std::int32_t> toks(static_cast<std::size_t>(B * T));
  for (int i = 0; i < B * T; ++i) toks[static_cast<std::size_t>(i)] = (i * 13 + 7) % c.vocab_size;

  // Sayisal esdegerlik: istekli (eager) vs derlenmis (compiled) ileri + geri
  m.zero_grad();
  const nexus::autograd::Variable eager_loss_v = m.loss(toks, B, T);
  const float eager_loss = eager_loss_v.values()[0];
  eager_loss_v.backward();
  std::vector<std::vector<float>> comp_grads;
  const float comp_loss = cm.forward_backward(toks, B, T, &comp_grads);

  // Üretilen CPU kaynağında hem AVX-512 (__m512) hem AVX2 (__m256) dalları bulunmalı.
  std::vector<const nexus::ir::FusedOp*> fops;
  for (const auto& n : cm.joint_graph().nodes) {
    if (n.op == nexus::ir::Op::Fused && n.fused) fops.push_back(n.fused.get());
  }
  const std::string src = nexus::ir::cpu_source(fops);
  nexus_test::expect(src.find("_mm512_") != std::string::npos && src.find("_mm256_") != std::string::npos,
                     "test_cpu_compile_fusion: cpu_source hem AVX-512 hem AVX2 vektor dallarini uretiyor");

  const double loss_diff = std::fabs(eager_loss - comp_loss);
  std::printf("  info  18.6 CPU derleyici: eager loss=%.6f, compiled loss=%.6f (fark=%.2e), arena=%.2f MB / naif=%.2f MB (%d->%d cekirdek)\n",
              eager_loss, comp_loss, loss_diff,
              static_cast<double>(cm.stats().arena_bytes) / (1024.0 * 1024.0),
              static_cast<double>(cm.stats().naive_bytes) / (1024.0 * 1024.0),
              cm.stats().kernels_eager, cm.stats().kernels_compiled);
  nexus_test::expect(loss_diff <= 1e-4, "test_cpu_compile_fusion: derlenmis ileri+geri kayip istekliyle <= 1e-4");
  nexus_test::expect(cm.stats().arena_bytes > 0 && cm.stats().arena_bytes < cm.stats().naive_bytes,
                     "test_cpu_compile_fusion: statik bellek plani naif toplamdan kucuk");
}

// ---- CPU cikarim: TinyLlama-1.1B KV onbellekli greedy (F32 / BF16 / INT8 / INT4) ---------
int bench_decode() {
  const char* env = std::getenv("NEXUS_TINYLLAMA");
  const std::string dir = env ? env : "C:/Users/muham/nexus-data/tinyllama";
  if (!std::filesystem::exists(std::filesystem::path(dir) / "model.safetensors")) {
    std::printf("  SKIP  TinyLlama yok (%s)\n", dir.c_str());
    return 77;
  }
  const lm::LanguageModel m = nexus::interop::load_hf(dir, true, nullptr);
  const lm::LmConfig& c = m.config();
  const std::vector<std::int32_t> prompt = {1, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296};
  const int gen = 32;
  nexus::autograd::NoGradGuard ng;

  struct ModeEntry {
    lm::CpuWeightMode mode;
    const char* label;
  };
  const ModeEntry entries[] = {
      {lm::CpuWeightMode::F32, "f32"},
      {lm::CpuWeightMode::BF16, "bf16"},
      {lm::CpuWeightMode::INT8, "int8"},
      {lm::CpuWeightMode::INT4, "int4"},
  };
  for (const auto& me : entries) {
    m.prepare_cpu_inference(me.mode);
    const std::size_t wbytes = m.cpu_inference_weight_bytes();
    lm::KvCache cache = m.make_cache();
    std::vector<std::int32_t> feed = prompt;
    double prefill = 0.0, decode = 0.0;
    for (int i = 0; i < gen; ++i) {
      const auto t0 = Clock::now();
      lm::ForwardOptions o;
      o.cache = &cache;
      const auto lg = m.logits(feed, 1, static_cast<int>(feed.size()), o).values();
      (i == 0 ? prefill : decode) += ms_since(t0);
      const std::size_t V = lg.size() / feed.size();
      const float* r = lg.data() + (feed.size() - 1) * V;
      const auto nx = static_cast<std::int32_t>(std::max_element(r, r + c.vocab_size) - r);
      feed = {nx};
    }
    const double per = decode / (gen - 1);
    std::printf("{\"mode\": \"bench_cpu_decode\", \"model\": \"TinyLlama-1.1B\", \"weights\": \"%s\", \"weight_mb\": %.1f, "
                "\"prompt\": %zu, \"prefill_ms\": %.1f, \"decode_ms_per_token\": %.1f, \"tokens_per_s\": %.2f, \"weight_gb_s\": %.1f}\n",
                me.label, static_cast<double>(wbytes) / (1024.0 * 1024.0), prompt.size(), prefill, per, 1000.0 / per,
                static_cast<double>(wbytes) / (per * 1e6));
  }
  return 0;
}

// Eski ZeRO-Offload yolu (FAZ 7): g*clip kopyasi + fused::adamw_multi_tensor + skaler BF16 golge.
void ref_offload(std::vector<float>& p, std::vector<float>& m, std::vector<float>& v, const void* grad, cpu::GradType t,
                 float clip, double wd, std::int64_t step, std::size_t b, std::size_t n, std::uint16_t* shadow) {
  std::vector<float> g(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float x = t == cpu::GradType::F32 ? static_cast<const float*>(grad)[b + i]
                                            : bf16_to_f(static_cast<const std::uint16_t*>(grad)[b + i]);
    g[i] = x * clip;
  }
  nexus::fused::ParamSlot slot{p.data() + b, g.data(), m.data() + b, v.data() + b, n};
  nexus::fused::AdamWConfig c;
  c.lr = 3e-4;
  c.beta1 = 0.9;
  c.beta2 = 0.95;
  c.eps = 1e-8;
  c.weight_decay = wd;
  c.max_grad_norm = 0.0;
  (void)nexus::fused::adamw_multi_tensor(std::span<const nexus::fused::ParamSlot>(&slot, 1), c, step);
  for (std::size_t i = 0; i < n; ++i) shadow[b + i] = nexus::float_to_bfloat16_bits(p[b + i]);
}

void test_adam() {
  std::size_t worst = 0;
  bool all = true;
  for (const cpu::GradType t : {cpu::GradType::F32, cpu::GradType::BF16}) {
    for (const std::size_t n : {std::size_t{13}, std::size_t{4099}, std::size_t{1000003}}) {
      for (const double wd : {0.0, 0.1}) {
        auto p0 = randv(n, 5);
        std::vector<float> m0(n, 0.0f), v0(n, 0.0f);
        auto p1 = p0, m1 = m0, v1 = v0;
        std::vector<std::uint16_t> s0(n), s1(n);
        for (std::int64_t step = 1; step <= 5; ++step) {
          const auto gf = randv(n, static_cast<unsigned>(100 + step));
          const auto gb = cpu::to_bf16(gf.data(), n);
          const void* g = t == cpu::GradType::F32 ? static_cast<const void*>(gf.data()) : gb.data();
          const float clip = step == 3 ? 0.37f : 1.0f;
          const std::size_t h = n / 2 + 1;
          ref_offload(p0, m0, v0, g, t, clip, wd, step, 0, h, s0.data());
          ref_offload(p0, m0, v0, g, t, clip, wd, step, h, n - h, s0.data());
          const cpu::AdamCoeffs c = cpu::adam_coeffs(3e-4, 0.9, 0.95, 1e-8, wd, step);
          cpu::adamw_range(p1.data(), m1.data(), v1.data(), g, t, clip, c, wd != 0.0, s1.data(), 0, h);
          cpu::adamw_range(p1.data(), m1.data(), v1.data(), g, t, clip, c, wd != 0.0, s1.data(), h, n);
        }
        std::size_t diff = 0;
        for (std::size_t i = 0; i < n; ++i) {
          diff += std::memcmp(&p0[i], &p1[i], 4) != 0 || std::memcmp(&m0[i], &m1[i], 4) != 0 ||
                  std::memcmp(&v0[i], &v1[i], 4) != 0 || s0[i] != s1[i];
        }
        worst = std::max(worst, diff);
        all = all && diff == 0;
      }
    }
  }
  std::printf("  info  AdamW AVX2 vs fused skaler: 2 gradyan tipi x 3 boy x 2 wd x 5 adim, en kotu farkli oge: %zu\n", worst);
  nexus_test::expect(all, "test_offload_adam_bitwise: p / m / v / BF16 golge bit-bit ayni");
  std::vector<float> x = {0.0f, -0.0f, 1.0f, -1.5f, 1e-40f, -1e-40f, 3.4e38f, -3.4e38f};
  const std::uint32_t raw[] = {0x7F800000u, 0xFF800000u, 0x7FC00001u, 0xFF800001u, 0x3F808000u, 0x3F818000u, 0x3F807FFFu,
                               0x7F7FFFFFu, 0x00008000u};
  for (const std::uint32_t r : raw) {
    float f;
    std::memcpy(&f, &r, 4);
    x.push_back(f);
  }
  const auto rnd = randv(1001, 9);
  x.insert(x.end(), rnd.begin(), rnd.end());
  std::vector<std::uint16_t> o(x.size());
  cpu::to_bf16_range(x.data(), o.data(), x.size());
  bool ok = true;
  for (std::size_t i = 0; i < x.size(); ++i) ok = ok && o[i] == nexus::float_to_bfloat16_bits(x[i]);
  nexus_test::expect(ok, "to_bf16_range == float_to_bfloat16_bits (inf / NaN / esitlik / denormal)");
  const auto g = randv(1000003, 11);
  double ref = 0.0;
  for (const float a : g) ref += static_cast<double>(a) * a;
  const double got = cpu::sq_norm_range(g.data(), cpu::GradType::F32, 0, g.size());
  std::printf("  info  sq_norm bagil fark (FP64, sira farki): %.2e\n", std::fabs(got - ref) / ref);
  nexus_test::expect(std::fabs(got - ref) / ref < 1e-12, "sq_norm_range FP64 referansa yakin");
}

int bench_adam() {
  const std::size_t n = std::size_t{124} << 20, task = std::size_t{1} << 20;
  std::vector<float> p = randv(n, 1), m(n, 0.0f), v(n, 0.0f);
  const std::vector<float> gf = randv(n, 2);
  const auto gb = cpu::to_bf16(gf.data(), n);
  std::vector<std::uint16_t> sh(n);
  const std::size_t tasks = n / task;
  const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  auto par = [&](auto&& fn) {
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < hw; ++i)
      th.emplace_back([&] {
        for (std::size_t k = next.fetch_add(1); k < tasks; k = next.fetch_add(1)) fn(k);
      });
    for (auto& x : th) x.join();
  };
  auto old = [&](cpu::GradType t, std::int64_t step) {
    const void* g = t == cpu::GradType::F32 ? static_cast<const void*>(gf.data()) : gb.data();
    std::vector<double> part(tasks);
    par([&](std::size_t k) {
      double a = 0;
      for (std::size_t i = k * task; i < (k + 1) * task; ++i) {
        const double x = t == cpu::GradType::F32 ? gf[i] : bf16_to_f(gb[i]);
        a += x * x;
      }
      part[k] = a;
    });
    double sq = 0;
    for (double x : part) sq += x;
    const float clip = static_cast<float>(std::min(1.0, 1.0 / (std::sqrt(sq) + 1e-6)));
    par([&](std::size_t k) { ref_offload(p, m, v, g, t, clip, 0.1, step, k * task, task, sh.data()); });
  };
  cpu::Pool& pool = cpu::Pool::instance();
  auto neu = [&](cpu::GradType t, std::int64_t step) {
    const void* g = t == cpu::GradType::F32 ? static_cast<const void*>(gf.data()) : gb.data();
    std::vector<double> part(tasks);
    std::atomic<std::size_t> next{0};
    pool.each([&](int) {
      for (std::size_t k = next.fetch_add(1); k < tasks; k = next.fetch_add(1))
        part[k] = cpu::sq_norm_range(g, t, k * task, (k + 1) * task);
    });
    double sq = 0;
    for (double x : part) sq += x;
    const float clip = static_cast<float>(std::min(1.0, 1.0 / (std::sqrt(sq) + 1e-6)));
    const cpu::AdamCoeffs c = cpu::adam_coeffs(3e-4, 0.9, 0.95, 1e-8, 0.1, step);
    next.store(0);
    pool.each([&](int) {
      for (std::size_t k = next.fetch_add(1); k < tasks; k = next.fetch_add(1))
        cpu::adamw_range(p.data(), m.data(), v.data(), g, t, clip, c, true, sh.data(), k * task, (k + 1) * task);
    });
  };
  for (const cpu::GradType t : {cpu::GradType::F32, cpu::GradType::BF16}) {
    double bo = 1e30, bn = 1e30;
    for (int r = 0; r < 4; ++r) {
      auto t0 = Clock::now();
      old(t, r + 1);
      if (r) bo = std::min(bo, ms_since(t0));
      t0 = Clock::now();
      neu(t, r + 1);
      if (r) bn = std::min(bn, ms_since(t0));
    }
    const double bytes = static_cast<double>(n) * ((t == cpu::GradType::F32 ? 4 : 2) * 2 + 12 + 12 + 2);
    std::printf("{\"mode\": \"bench_offload_adam\", \"params\": %zu, \"grad\": \"%s\", \"old_ms\": %.1f, \"old_threads\": %d, "
                "\"new_ms\": %.1f, \"new_threads\": %d, \"speedup\": %.2f, \"new_gb_s\": %.1f}\n",
                n, t == cpu::GradType::F32 ? "f32" : "bf16", bo, hw, bn, pool.size(), bo / bn, bytes / (bn * 1e6));
  }
  return 0;
}

int bench() {
  std::printf("== %s\n", cpu::describe().c_str());
  cpu::Pool& pool = cpu::Pool::instance();
  cpu::Pool pool1(1, true);
  for (int n : {256, 512, 1024, 2048, 4096}) {
    const int M = n, N = n, K = n;
    const auto A = randv(static_cast<std::size_t>(M) * K, 1), B = randv(static_cast<std::size_t>(K) * N, 2);
    const auto Bb = cpu::to_bf16(B.data(), B.size()), Bh = cpu::to_f16(B.data(), B.size());
    std::vector<float> C(static_cast<std::size_t>(M) * N);
    auto time = [&](auto&& fn) {
      fn();
      double best = 1e30;
      for (int r = 0; r < (n <= 1024 ? 10 : 3); ++r) {
        const auto t0 = Clock::now();
        fn();
        best = std::min(best, ms_since(t0));
      }
      return best;
    };
    const double flop = 2.0 * M * N * K;
    auto run = [&](cpu::BType t, const void* b, cpu::Pool* pl) {
      cpu::GemmArgs g;
      g.M = M;
      g.N = N;
      g.K = K;
      g.A = A.data();
      g.lda = K;
      g.B = b;
      g.ldb = N;
      g.btype = t;
      g.C = C.data();
      g.ldc = N;
      return time([&] { cpu::gemm(g, pl); });
    };
    const double f32_1t = n <= 2048 ? run(cpu::BType::F32, B.data(), &pool1) : -1.0;
    const double f32 = run(cpu::BType::F32, B.data(), &pool), bf = run(cpu::BType::BF16, Bb.data(), &pool),
                 h = run(cpu::BType::F16, Bh.data(), &pool);
    const double old = n <= 2048 ? time([&] { nexus_model::kernels::gemm_nn(A.data(), B.data(), C.data(), M, K, N); }) : -1;
    const cpu::PackedS8 P = cpu::pack_s8(B.data(), N, K, false);
    const cpu::PackedQ4 P4 = cpu::pack_q4(B.data(), N, K, false);
    std::vector<std::int8_t> Aq(static_cast<std::size_t>(M) * P4.Kp);
    std::vector<float> sa(static_cast<std::size_t>(M));
    cpu::quantize_rows_s8(A.data(), M, K, K, Aq.data(), P.Kp, sa.data());
    const double i8 = time([&] { cpu::gemm_s8(Aq.data(), sa.data(), M, P, C.data(), N, nullptr, &pool); });
    cpu::quantize_rows_s8(A.data(), M, K, K, Aq.data(), P4.Kp, sa.data());
    const double i4 = time([&] { cpu::gemm_q4(Aq.data(), sa.data(), M, P4, C.data(), N, nullptr, &pool); });
    std::printf("{\"mode\": \"bench_cpu_gemm\", \"n\": %d, \"threads\": %d, \"f32_1t_gflops\": %.1f, \"f32_gflops\": %.1f, "
                "\"bf16_gflops\": %.1f, \"f16_gflops\": %.1f, \"int8_gops\": %.1f, \"int4_gops\": %.1f, \"old_gemm_nn_gflops\": %.1f}\n",
                n, pool.size(), f32_1t > 0 ? flop / f32_1t / 1e6 : -1.0, flop / f32 / 1e6, flop / bf / 1e6, flop / h / 1e6,
                flop / i8 / 1e6, flop / i4 / 1e6, old > 0 ? flop / old / 1e6 : -1.0);
  }
  // GEMV (M = 1, cikarim): B [N, K] agirlik
  for (int K : {2048, 4096}) {
    const int N = 4 * K / 2;
    const auto a = randv(static_cast<std::size_t>(K), 3), W = randv(static_cast<std::size_t>(N) * K, 4);
    const auto Wb = cpu::to_bf16(W.data(), W.size());
    const auto Ps8 = cpu::pack_s8(W.data(), N, K, true);
    const auto Pq4 = cpu::pack_q4(W.data(), N, K, true);
    std::vector<std::int8_t> aq(static_cast<std::size_t>(Pq4.Kp));
    float sa = 1.0f;
    std::vector<float> c(static_cast<std::size_t>(N));
    auto gv = [&](cpu::BType t, const void* w) {
      cpu::GemmArgs g;
      g.M = 1;
      g.N = N;
      g.K = K;
      g.A = a.data();
      g.lda = K;
      g.B = w;
      g.ldb = K;
      g.btype = t;
      g.b_trans = true;
      g.C = c.data();
      g.ldc = N;
      cpu::gemm(g, &pool);
      double best = 1e30;
      for (int r = 0; r < 20; ++r) {
        const auto t0 = Clock::now();
        cpu::gemm(g, &pool);
        best = std::min(best, ms_since(t0));
      }
      return best;
    };
    auto gv_fn = [&](auto&& fn) {
      fn();
      double best = 1e30;
      for (int r = 0; r < 20; ++r) {
        const auto t0 = Clock::now();
        fn();
        best = std::min(best, ms_since(t0));
      }
      return best;
    };
    const double f = gv(cpu::BType::F32, W.data()), b = gv(cpu::BType::BF16, Wb.data());
    const double i8 = gv_fn([&] {
      cpu::quantize_rows_s8(a.data(), 1, K, K, aq.data(), Ps8.Kp, &sa);
      cpu::gemm_s8(aq.data(), &sa, 1, Ps8, c.data(), N, nullptr, &pool);
    });
    const double i4 = gv_fn([&] {
      cpu::quantize_rows_s8(a.data(), 1, K, K, aq.data(), Pq4.Kp, &sa);
      cpu::gemm_q4(aq.data(), &sa, 1, Pq4, c.data(), N, nullptr, &pool);
    });
    std::printf("{\"mode\": \"bench_cpu_gemv\", \"N\": %d, \"K\": %d, \"f32_ms\": %.3f, \"f32_gb_s\": %.1f, "
                "\"bf16_ms\": %.3f, \"bf16_gb_s\": %.1f, \"int8_ms\": %.3f, \"int4_ms\": %.3f}\n",
                N, K, f, N * static_cast<double>(K) * 4 / (f * 1e6), b, N * static_cast<double>(K) * 2 / (b * 1e6), i8, i4);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    if (mode == "all") {
      test_features();
      test_gemm();
      test_gemm_avx512();
      test_numa();
      test_adam();
      test_flash();
      test_train();
      test_quantized_inference();
      test_compile_fusion();
    } else if (mode == "features") test_features();
    else if (mode == "gemm" || mode == "gemm_avx2") test_gemm();
    else if (mode == "gemm_avx512") test_gemm_avx512();
    else if (mode == "numa") test_numa();
    else if (mode == "adam" || mode == "offload") test_adam();
    else if (mode == "flash") test_flash();
    else if (mode == "train") test_train();
    else if (mode == "quant" || mode == "infer") test_quantized_inference();
    else if (mode == "compile" || mode == "fusion") test_compile_fusion();
    else if (mode == "bench_decode") return bench_decode();
    else if (mode == "bench_attn") return bench_attn();
    else if (mode == "bench") return bench();
    else if (mode == "bench_adam") return bench_adam();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase18_cpu");
}
