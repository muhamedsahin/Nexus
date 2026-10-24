/**
 * @file test_cuda_kernels.cpp
 * @brief FAZ 3 GPU: her cekirdek CPU FP64 (veya Nexus CPU) referansina karsi ileri/geri;
 *        sekil taramasi (V tek, V=50257 hizasiz, T=1, B=1, buyuk B*T).
 *
 * promt.txt FAZ 3 TESTLER:
 *  - Her cekirdek: CPU FP64 referansina karsi fwd/bwd; sekil taramasi.
 *  - test_adamw_fused_vs_reference: 1000 adim, CPU AdamW ile <= 1e-6.
 *  - test_dropout_reproducible: GPU maskesi CPU Philox maskesiyle bit-bit ayni.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/fused/training.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace fu = nexus::fused;

namespace {

cudaStream_t g_stream = nullptr;

std::vector<float> randv(std::size_t n, std::uint64_t seed, float scale = 1.0f) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (float& x : v) {
    x = d(rng);
  }
  return v;
}

struct Dev {
  nc::DeviceBuffer buffer;
  std::size_t n = 0;
  explicit Dev(std::size_t count) : buffer(std::max<std::size_t>(count, 1) * sizeof(float)), n(count) {
    NEXUS_CUDA_CHECK(cudaMemsetAsync(buffer.data(), 0, std::max<std::size_t>(count, 1) * sizeof(float), g_stream));
  }
  explicit Dev(const std::vector<float>& host) : Dev(host.size()) {
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(buffer.data(), host.data(), host.size() * 4, cudaMemcpyHostToDevice, g_stream));
  }
  float* p() const { return buffer.as<float>(); }
  std::vector<float> get() const {
    std::vector<float> host(n);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data(), buffer.data(), n * 4, cudaMemcpyDeviceToHost, g_stream));
    NEXUS_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return host;
  }
};

struct IntDev {
  nc::DeviceBuffer buffer;
  explicit IntDev(const std::vector<std::int32_t>& host) : buffer(host.size() * 4) {
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(buffer.data(), host.data(), host.size() * 4, cudaMemcpyHostToDevice, g_stream));
  }
  const std::int32_t* p() const { return buffer.as<std::int32_t>(); }
};

// Norm-bagil hata: max|a-b| / max(|b|, floor).
double rel_err(const std::vector<float>& a, const std::vector<double>& b, double floor = 1.0e-6) {
  double worst = 0.0;
  double scale = floor;
  for (std::size_t i = 0; i < b.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
    scale = std::max(scale, std::fabs(b[i]));
  }
  return worst / scale;
}

double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  return rel_err(a, std::vector<double>(b.begin(), b.end()));
}

void report(const std::string& name, double err, double tol) {
  if (err >= tol) {
    std::printf("  info  %s hata=%.3e esik=%.1e\n", name.c_str(), err, tol);
  }
  nexus_test::expect(err < tol, name);
}

double gelu_tanh(double x) { return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }
double gelu_tanh_grad(double x) {
  const double u = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  const double t = std::tanh(u);
  return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x);
}

// Satir-ana referans GEMM (FP64): C = op(A) op(B).
std::vector<double> ref_gemm(bool ta, bool tb, int m, int n, int k, const std::vector<float>& a,
                             const std::vector<float>& b) {
  std::vector<double> c(static_cast<std::size_t>(m) * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int t = 0; t < k; ++t) {
        const double av = ta ? a[static_cast<std::size_t>(t) * m + i] : a[static_cast<std::size_t>(i) * k + t];
        const double bv = tb ? b[static_cast<std::size_t>(j) * k + t] : b[static_cast<std::size_t>(t) * n + j];
        acc += av * bv;
      }
      c[static_cast<std::size_t>(i) * n + j] = acc;
    }
  }
  return c;
}

void test_gemm() {
  nexus_test::section("cuBLASLt GEMM: transpoze, hizasiz sekiller, epiloglar, batched");
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  struct Shape {
    int m, n, k;
  };
  for (const Shape s : {Shape{7, 13, 5}, Shape{64, 96, 128}, Shape{3, 50257, 32}, Shape{1, 1, 1}, Shape{257, 129, 65}}) {
    for (int mode = 0; mode < 4; ++mode) {
      const bool ta = (mode & 1) != 0;
      const bool tb = (mode & 2) != 0;
      const auto a = randv(static_cast<std::size_t>(s.m) * s.k, 1 + mode);
      const auto b = randv(static_cast<std::size_t>(s.k) * s.n, 9 + mode);
      Dev da(a), db(b), dc(static_cast<std::size_t>(s.m) * s.n);
      nc::GemmArgs g;
      g.trans_a = ta;
      g.trans_b = tb;
      g.m = s.m;
      g.n = s.n;
      g.k = s.k;
      g.a = da.p();
      g.lda = ta ? s.m : s.k;
      g.b = db.p();
      g.ldb = tb ? s.k : s.n;
      g.c = dc.p();
      g.ldc = s.n;
      nc::gemm(g, g_stream);
      const double err = rel_err(dc.get(), ref_gemm(ta, tb, s.m, s.n, s.k, a, b));
      report("gemm m=" + std::to_string(s.m) + " n=" + std::to_string(s.n) + " k=" + std::to_string(s.k) +
                 (ta ? " At" : "") + (tb ? " Bt" : ""),
             err, 1e-5);
    }
  }

  // Epiloglar: Linear ileri (bias+gelu, aux) ve geri (dgelu, bgrad).
  const int m = 96, n = 64, k = 48;
  const auto x = randv(static_cast<std::size_t>(m) * k, 21);
  const auto w = randv(static_cast<std::size_t>(k) * n, 22, 0.3f);
  const auto bias = randv(n, 23);
  const auto ref = ref_gemm(false, false, m, n, k, x, w);
  Dev dx(x), dw(w), dbias(bias), dy(static_cast<std::size_t>(m) * n), daux(static_cast<std::size_t>(m) * n);
  nc::GemmArgs g;
  g.m = m;
  g.n = n;
  g.k = k;
  g.a = dx.p();
  g.lda = k;
  g.b = dw.p();
  g.ldb = n;
  g.c = dy.p();
  g.ldc = n;
  g.bias = dbias.p();
  g.epilogue = nc::Epilogue::Bias;
  nc::gemm(g, g_stream);
  std::vector<double> pre(ref.size());
  std::vector<double> act(ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    pre[i] = ref[i] + bias[i % n];
    act[i] = gelu_tanh(pre[i]);
  }
  report("epilog bias", rel_err(dy.get(), pre), 1e-5);
  g.epilogue = nc::Epilogue::BiasGelu;
  g.aux = daux.p();
  nc::gemm(g, g_stream);
  report("epilog bias+gelu (tanh) cikti", rel_err(dy.get(), act), 1e-5);
  report("epilog bias+gelu aux = on-aktivasyon", rel_err(daux.get(), pre), 1e-5);

  // Geri: dpre = (dY W2^T) * gelu'(pre). dY [m, n2], W2 [n, n2] -> trans_b.
  const int n2 = 40;
  const auto dyy = randv(static_cast<std::size_t>(m) * n2, 31);
  const auto w2 = randv(static_cast<std::size_t>(n) * n2, 32, 0.3f);
  Dev ddyy(dyy), dw2(w2), dpre(static_cast<std::size_t>(m) * n);
  const auto prod = ref_gemm(false, true, m, n, n2, dyy, w2);
  std::vector<double> want(prod.size());
  const auto aux_host = daux.get();
  for (std::size_t i = 0; i < prod.size(); ++i) {
    want[i] = prod[i] * gelu_tanh_grad(aux_host[i]);
  }
  nc::GemmArgs gb;
  gb.trans_b = true;
  gb.m = m;
  gb.n = n;
  gb.k = n2;
  gb.a = ddyy.p();
  gb.lda = n2;
  gb.b = dw2.p();
  gb.ldb = n2;
  gb.c = dpre.p();
  gb.ldc = n;
  gb.epilogue = nc::Epilogue::DGelu;
  gb.aux = daux.p();
  nc::gemm(gb, g_stream);
  report("epilog dgelu (geri)", rel_err(dpre.get(), want), 1e-4);

  // dW = X^T dY ve db = sum_rows dY (BGradB).
  Dev ddw(static_cast<std::size_t>(k) * n), ddb(n);
  const auto dout = randv(static_cast<std::size_t>(m) * n, 41);
  Dev ddout(dout);
  nc::GemmArgs gw;
  gw.trans_a = true;
  gw.m = k;
  gw.n = n;
  gw.k = m;
  gw.a = dx.p();
  gw.lda = k;
  gw.b = ddout.p();
  gw.ldb = n;
  gw.c = ddw.p();
  gw.ldc = n;
  gw.epilogue = nc::Epilogue::BGradB;
  gw.bias_grad = ddb.p();
  nc::gemm(gw, g_stream);
  std::vector<double> db_ref(n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      db_ref[j] += dout[static_cast<std::size_t>(i) * n + j];
    }
  }
  report("dW = X^T dY (kopyasiz transpoze)", rel_err(ddw.get(), ref_gemm(true, false, k, n, m, x, dout)), 1e-5);
  report("epilog bgrad: db = sum dY", rel_err(ddb.get(), db_ref), 1e-5);

  // Batched: [4] x [T,D] * [T,D]^T.
  const int batch = 4, T = 33, D = 16;
  const auto q = randv(static_cast<std::size_t>(batch) * T * D, 51);
  const auto kk = randv(static_cast<std::size_t>(batch) * T * D, 52);
  Dev dq(q), dk(kk), ds(static_cast<std::size_t>(batch) * T * T);
  nc::GemmArgs bt;
  bt.trans_b = true;
  bt.m = T;
  bt.n = T;
  bt.k = D;
  bt.a = dq.p();
  bt.lda = D;
  bt.b = dk.p();
  bt.ldb = D;
  bt.c = ds.p();
  bt.ldc = T;
  bt.batch = batch;
  bt.stride_a = static_cast<long long>(T) * D;
  bt.stride_b = static_cast<long long>(T) * D;
  bt.stride_c = static_cast<long long>(T) * T;
  nc::gemm(bt, g_stream);
  const auto got = ds.get();
  double worst = 0.0;
  for (int b = 0; b < batch; ++b) {
    const std::vector<float> qb(q.begin() + b * T * D, q.begin() + (b + 1) * T * D);
    const std::vector<float> kb(kk.begin() + b * T * D, kk.begin() + (b + 1) * T * D);
    const auto r = ref_gemm(false, true, T, T, D, qb, kb);
    const std::vector<float> gotb(got.begin() + b * T * T, got.begin() + (b + 1) * T * T);
    worst = std::max(worst, rel_err(gotb, r));
  }
  report("strided batched GEMM (dikkat skoru)", worst, 1e-5);

  // TF32: daha gevsek ama yine dogru; politika degisikligi yeni plan uretir.
  nc::set_float32_matmul_precision("high");
  const auto plans_before = nc::gemm_cache_stats().plans;
  nc::GemmArgs tf = g;
  tf.epilogue = nc::Epilogue::None;
  tf.bias = nullptr;
  tf.aux = nullptr;
  nc::gemm(tf, g_stream);
  report("TF32 ('high') GEMM", rel_err(dy.get(), ref), 5e-3);
  nexus_test::expect(nc::gemm_cache_stats().plans == plans_before + 1, "hassasiyet politikasi plan anahtarinda");
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  const auto hits = nc::gemm_cache_stats().hits;
  nc::gemm(gw, g_stream);
  nexus_test::expect(nc::gemm_cache_stats().hits == hits + 1, "ayni sekil plan onbelleginden");
  bool threw = false;
  try {
    nc::GemmArgs bad = g;
    bad.epilogue = nc::Epilogue::DGelu;
    bad.aux = nullptr;
    nc::gemm(bad, g_stream);
  } catch (const std::exception&) {
    threw = true;
  }
  nexus_test::expect(threw, "DGelu aux olmadan acik hata");
}

void test_norms() {
  nexus_test::section("LayerNorm / RMSNorm / add&norm ileri-geri (FP64 referans)");
  struct S {
    int rows, cols;
  };
  for (const S s : {S{1, 1}, S{3, 7}, S{37, 769}, S{4096, 768}, S{2, 50257}}) {
    const std::size_t n = static_cast<std::size_t>(s.rows) * s.cols;
    const auto x = randv(n, 61, 2.0f);
    const auto w = randv(s.cols, 62);
    const auto b = randv(s.cols, 63);
    const auto dout = randv(n, 64);
    const float eps = 1e-5f;
    std::vector<double> y(n), dx(n), dw(s.cols, 0.0), dbias(s.cols, 0.0), ry(n), rdx(n), rdw(s.cols, 0.0);
    for (int r = 0; r < s.rows; ++r) {
      double mean = 0.0, var = 0.0, sq = 0.0;
      for (int c = 0; c < s.cols; ++c) {
        mean += x[static_cast<std::size_t>(r) * s.cols + c];
      }
      mean /= s.cols;
      for (int c = 0; c < s.cols; ++c) {
        const double d = x[static_cast<std::size_t>(r) * s.cols + c] - mean;
        var += d * d;
        sq += static_cast<double>(x[static_cast<std::size_t>(r) * s.cols + c]) * x[static_cast<std::size_t>(r) * s.cols + c];
      }
      var /= s.cols;
      const double rstd = 1.0 / std::sqrt(var + eps);
      const double rr = 1.0 / std::sqrt(sq / s.cols + eps);
      double m1 = 0.0, m2 = 0.0, q2 = 0.0;
      for (int c = 0; c < s.cols; ++c) {
        const std::size_t i = static_cast<std::size_t>(r) * s.cols + c;
        const double xh = (x[i] - mean) * rstd;
        y[i] = xh * w[c] + b[c];
        ry[i] = x[i] * rr * w[c];
        const double dn = static_cast<double>(dout[i]) * w[c];
        m1 += dn;
        m2 += dn * xh;
        q2 += dn * x[i];
        dw[c] += dout[i] * xh;
        dbias[c] += dout[i];
        rdw[c] += dout[i] * x[i] * rr;
      }
      m1 /= s.cols;
      m2 /= s.cols;
      q2 /= s.cols;
      for (int c = 0; c < s.cols; ++c) {
        const std::size_t i = static_cast<std::size_t>(r) * s.cols + c;
        const double xh = (x[i] - mean) * rstd;
        const double dn = static_cast<double>(dout[i]) * w[c];
        dx[i] = rstd * (dn - m1 - xh * m2);
        rdx[i] = rr * dn - x[i] * rr * rr * rr * q2;
      }
    }
    Dev dX(x), dW(w), dB(b), dO(dout), dY(n), dMean(s.rows), dRstd(s.rows), dDX(n), dDW(s.cols), dDB(s.cols);
    nc::DeviceBuffer scratch(nc::layernorm_backward_scratch_bytes(s.rows, s.cols));
    nc::layernorm_forward(dY.p(), dMean.p(), dRstd.p(), dX.p(), dW.p(), dB.p(), s.rows, s.cols, eps, g_stream);
    nc::layernorm_backward(dDX.p(), dDW.p(), dDB.p(), dO.p(), dX.p(), dW.p(), dMean.p(), dRstd.p(), s.rows, s.cols,
                           false, scratch.as<float>(), g_stream);
    const std::string tag = " [" + std::to_string(s.rows) + "x" + std::to_string(s.cols) + "]";
    report("layernorm ileri" + tag, rel_err(dY.get(), y), 1e-5);
    // cols=1'de dx analitik olarak ~0 (xhat=0); bagil olcum icin taban dout olcegindedir.
    report("layernorm dx" + tag, rel_err(dDX.get(), dx, 1e-2), 1e-4);
    report("layernorm dweight" + tag, rel_err(dDW.get(), dw), 1e-4);
    report("layernorm dbias" + tag, rel_err(dDB.get(), dbias), 1e-4);
    // Belirlenimcilik: ikinci kosu bit-bit ayni (atomiksiz).
    const auto first = dDW.get();
    nc::fill(dDW.p(), 0.0f, s.cols, g_stream);
    nc::layernorm_backward(dDX.p(), dDW.p(), dDB.p(), dO.p(), dX.p(), dW.p(), dMean.p(), dRstd.p(), s.rows, s.cols,
                           false, scratch.as<float>(), g_stream);
    nexus_test::expect(first == dDW.get(), "layernorm geri belirlenimci (bit-bit)" + tag);

    Dev rY(n), rR(s.rows), rDX(n), rDW(s.cols);
    nc::rmsnorm_forward(rY.p(), rR.p(), dX.p(), dW.p(), s.rows, s.cols, eps, g_stream);
    nc::rmsnorm_backward(rDX.p(), rDW.p(), dO.p(), dX.p(), dW.p(), rR.p(), s.rows, s.cols, false, scratch.as<float>(),
                         g_stream);
    report("rmsnorm ileri" + tag, rel_err(rY.get(), ry), 1e-5);
    report("rmsnorm dx" + tag, rel_err(rDX.get(), rdx, 1e-2), 1e-4);
    report("rmsnorm dweight" + tag, rel_err(rDW.get(), rdw), 1e-4);

    if (s.rows > 1) {
      // add & norm: a + b = x olacak sekilde ikiye bol.
      std::vector<float> a(n), bb(n);
      for (std::size_t i = 0; i < n; ++i) {
        a[i] = 0.25f * x[i];
        bb[i] = x[i] - a[i];
      }
      Dev dA(a), dBB(bb), dSum(n), dOut(n), m2(s.rows), r2(s.rows);
      nc::residual_layernorm_forward(dSum.p(), dOut.p(), m2.p(), r2.p(), dA.p(), dBB.p(), dW.p(), dB.p(), s.rows, s.cols,
                                     eps, g_stream);
      std::vector<float> sum_ref(n);
      for (std::size_t i = 0; i < n; ++i) {
        sum_ref[i] = a[i] + bb[i];
      }
      nexus_test::expect(dSum.get() == sum_ref, "add&norm toplam ciktisi bit-bit" + tag);
      report("add&norm normalize ciktisi" + tag, rel_err(dOut.get(), y), 1e-5);
    }
  }
}

void test_softmax() {
  nexus_test::section("nedensel softmax (maske cekirdekte) ileri-geri");
  for (const int t : {1, 5, 64, 1024}) {
    const int bh = t == 1024 ? 2 : 3;
    const float scale = 0.125f;
    const std::size_t n = static_cast<std::size_t>(bh) * t * t;
    const auto s = randv(n, 71, 3.0f);
    const auto dp = randv(n, 72);
    std::vector<double> p(n, 0.0), ds(n, 0.0);
    for (int r = 0; r < bh * t; ++r) {
      const int i = r % t;
      const std::size_t base = static_cast<std::size_t>(r) * t;
      double maxv = -1e300, total = 0.0;
      for (int j = 0; j <= i; ++j) {
        maxv = std::max(maxv, static_cast<double>(s[base + j]) * scale);
      }
      for (int j = 0; j <= i; ++j) {
        total += std::exp(s[base + j] * static_cast<double>(scale) - maxv);
      }
      double dot = 0.0;
      for (int j = 0; j <= i; ++j) {
        p[base + j] = std::exp(s[base + j] * static_cast<double>(scale) - maxv) / total;
        dot += p[base + j] * dp[base + j];
      }
      for (int j = 0; j <= i; ++j) {
        ds[base + j] = scale * p[base + j] * (dp[base + j] - dot);
      }
    }
    Dev dS(s), dP(n), dDP(dp), dDS(n);
    nc::softmax_causal_forward(dP.p(), dS.p(), bh, t, scale, g_stream);
    nc::softmax_causal_backward(dDS.p(), dDP.p(), dP.p(), bh, t, scale, g_stream);
    const auto got = dP.get();
    bool masked = true;
    for (int r = 0; r < bh * t; ++r) {
      for (int j = r % t + 1; j < t; ++j) {
        masked = masked && got[static_cast<std::size_t>(r) * t + j] == 0.0f;
      }
    }
    const std::string tag = " [T=" + std::to_string(t) + "]";
    nexus_test::expect(masked, "j > i maskeli (maske tensoru yok)" + tag);
    report("softmax ileri" + tag, rel_err(got, p), 1e-5);
    report("softmax geri" + tag, rel_err(dDS.get(), ds), 1e-4);
  }
}

void test_embedding() {
  nexus_test::section("gomme + konum ileri, atomik ve belirlenimci geri");
  const int batch = 4, time = 64, dim = 96, vocab = 50257;
  std::vector<std::int32_t> tokens(static_cast<std::size_t>(batch) * time);
  std::mt19937 rng(5);
  for (auto& t : tokens) {
    t = static_cast<std::int32_t>(rng() % 40);  // tekrar eden tokenlar: segment toplami sinanir
  }
  tokens[3] = vocab - 1;
  const auto wte = randv(static_cast<std::size_t>(vocab) * dim, 81);
  const auto wpe = randv(static_cast<std::size_t>(time) * dim, 82);
  const auto dout = randv(tokens.size() * dim, 83);
  std::vector<double> out(tokens.size() * dim), dwte(static_cast<std::size_t>(vocab) * dim, 0.0),
      dwpe(static_cast<std::size_t>(time) * dim, 0.0);
  for (std::size_t r = 0; r < tokens.size(); ++r) {
    for (int c = 0; c < dim; ++c) {
      const std::size_t i = r * dim + c;
      out[i] = static_cast<double>(wte[static_cast<std::size_t>(tokens[r]) * dim + c]) + wpe[(r % time) * dim + c];
      dwte[static_cast<std::size_t>(tokens[r]) * dim + c] += dout[i];
      dwpe[(r % time) * dim + c] += dout[i];
    }
  }
  IntDev dt(tokens);
  Dev dWte(wte), dWpe(wpe), dOut(tokens.size() * dim), dDout(dout);
  nc::embedding_pos_forward(dOut.p(), dt.p(), dWte.p(), dWpe.p(), batch, time, dim, g_stream);
  report("embedding_pos ileri", rel_err(dOut.get(), out), 1e-7);
  Dev aWte(wte.size()), aWpe(wpe.size());
  nc::embedding_backward_atomic(aWte.p(), aWpe.p(), dDout.p(), dt.p(), batch, time, dim, g_stream);
  report("embedding geri (atomik) dwte", rel_err(aWte.get(), dwte), 1e-5);
  report("embedding geri (atomik) dwpe", rel_err(aWpe.get(), dwpe), 1e-5);
  nc::DeviceBuffer scratch(nc::embedding_backward_scratch_bytes(batch, time));
  Dev sWte(wte.size()), sWpe(wpe.size());
  nc::embedding_backward_deterministic(sWte.p(), sWpe.p(), dDout.p(), dt.p(), batch, time, dim, vocab, -1,
                                       scratch.data(), g_stream);
  const auto first = sWte.get();
  report("embedding geri (siralama+segment) dwte", rel_err(first, dwte), 1e-5);
  report("embedding geri (belirlenimci) dwpe", rel_err(sWpe.get(), dwpe), 1e-5);
  nc::fill(sWte.p(), 0.0f, wte.size(), g_stream);
  nc::embedding_backward_deterministic(sWte.p(), nullptr, dDout.p(), dt.p(), batch, time, dim, vocab, -1,
                                       scratch.data(), g_stream);
  nexus_test::expect(first == sWte.get(), "belirlenimci yol bit-bit tekrarlanir");
  nc::fill(sWte.p(), 0.0f, wte.size(), g_stream);
  nc::embedding_backward_deterministic(sWte.p(), nullptr, dDout.p(), dt.p(), batch, time, dim, vocab, tokens[0],
                                       scratch.data(), g_stream);
  const auto padded = sWte.get();
  bool zero = true;
  for (int c = 0; c < dim; ++c) {
    zero = zero && padded[static_cast<std::size_t>(tokens[0]) * dim + c] == 0.0f;
  }
  nexus_test::expect(zero, "padding_idx satiri gradyan almaz");
}

void test_cross_entropy() {
  nexus_test::section("fuzyonlu kaydirilmis CE (GPU) vs Nexus CPU cekirdegi");
  struct S {
    int b, t, v;
    float smooth, z;
    bool ignore;
  };
  for (const S s : {S{2, 5, 7, 0, 0, false}, S{1, 2, 50257, 0, 0, false}, S{3, 4, 33, 0, 0, true},
                    S{2, 6, 19, 0.1f, 0, false}, S{2, 6, 19, 0, 1e-4f, true}, S{4, 1024, 1031, 0.05f, 1e-4f, true}}) {
    const std::size_t n = static_cast<std::size_t>(s.b) * s.t * s.v;
    const auto logits = randv(n, 91 + s.v, 2.0f);
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(s.b) * s.t);
    std::mt19937 rng(static_cast<unsigned>(s.v));
    for (auto& t : tokens) {
      t = static_cast<std::int32_t>(rng() % static_cast<unsigned>(s.v));
    }
    fu::CrossEntropyOptions cpu_opt;
    cpu_opt.label_smoothing = s.smooth;
    cpu_opt.z_loss = s.z;
    if (s.ignore) {
      tokens[1] = cpu_opt.ignore_index;
    }
    std::vector<float> cpu_grad(n);
    const auto cpu = fu::cross_entropy_shifted(logits.data(), tokens.data(), s.b, s.t, s.v, cpu_opt, cpu_grad.data());
    nc::CrossEntropyDeviceOptions opt;
    opt.label_smoothing = s.smooth;
    opt.z_loss = s.z;
    IntDev dt(tokens);
    Dev dl(logits), dloss(1);
    nc::DeviceBuffer valid(4);
    nc::DeviceBuffer scratch(nc::cross_entropy_scratch_bytes(s.b, s.t));
    nc::cross_entropy_shifted(dl.p(), dloss.p(), valid.as<std::int32_t>(), dl.p(), dt.p(), s.b, s.t, s.v, opt, 1.0f,
                              scratch.data(), g_stream);  // yerinde
    std::int32_t valid_host = 0;
    // Engellemeyen akis legacy akisla senkronlanmaz: okuma ayni akista + senkron.
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(&valid_host, valid.data(), 4, cudaMemcpyDeviceToHost, g_stream));
    NEXUS_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    const std::string tag = " [B=" + std::to_string(s.b) + " T=" + std::to_string(s.t) + " V=" + std::to_string(s.v) +
                            (s.smooth > 0 ? " ls" : "") + (s.z > 0 ? " z" : "") + (s.ignore ? " ign" : "") + "]";
    nexus_test::expect(valid_host == cpu.valid, "gecerli token sayisi" + tag);
    report("CE kayip" + tag, std::fabs(dloss.get()[0] - cpu.loss) / std::max(1.0f, std::fabs(cpu.loss)), 1e-5);
    report("CE dlogits (yerinde)" + tag, rel_err(dl.get(), cpu_grad), 1e-5);
  }
  // Sozluk disi hedef -> NaN kayip (sessiz 0 yok).
  std::vector<std::int32_t> tokens = {0, 9};
  IntDev dt(tokens);
  Dev dl(std::vector<float>(4, 0.0f)), dloss(1);
  nc::DeviceBuffer scratch(nc::cross_entropy_scratch_bytes(1, 2));
  nc::cross_entropy_shifted(nullptr, dloss.p(), nullptr, dl.p(), dt.p(), 1, 2, 2, {}, 1.0f, scratch.data(), g_stream);
  nexus_test::expect(std::isnan(dloss.get()[0]), "sozluk disi hedef kaybi NaN yapar");
}

void test_dropout() {
  nexus_test::section("test_dropout_reproducible (GPU Philox = CPU Philox)");
  const std::size_t n = 1000003;  // 4'e bolunmez: kuyruk serit sinanir
  const std::vector<float> ones(n, 1.0f);
  Dev dx(ones), dy(n), dg(n);
  nc::dropout_philox_forward(dy.p(), dx.p(), n, 0.3f, 1234, 77, g_stream);
  nc::dropout_philox_backward(dg.p(), dx.p(), n, 0.3f, 1234, 77, g_stream);
  const auto y = dy.get();
  std::size_t mismatch = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float expected = fu::philox_uniform(1234, 77, i) < 0.3f ? 0.0f : 1.0f / 0.7f;
    mismatch += y[i] != expected ? 1 : 0;
  }
  nexus_test::info("dropout_mask_mismatch", static_cast<double>(mismatch));
  nexus_test::expect(mismatch == 0, "GPU maskesi CPU Philox maskesiyle bit-bit ayni");
  nexus_test::expect(dg.get() == y, "geri geçis ayni (seed, offset) ile ayni maskeyi uretir");
}

void test_adamw() {
  nexus_test::section("test_adamw_fused_vs_reference (GPU vs CPU foreach AdamW, 1000 adim)");
  const std::vector<std::size_t> sizes = {7, 64, 129, 1, 70001};
  for (const double max_norm : {0.0, 0.5}) {
    std::vector<std::vector<float>> cp, cg, cm, cv;
    std::vector<nc::DeviceParamSlot> slots;
    std::vector<fu::ParamSlot> cpu_slots;
    for (std::size_t t = 0; t < sizes.size(); ++t) {
      cp.push_back(randv(sizes[t], 300 + t));
      cg.emplace_back(sizes[t], 0.0f);
      cm.emplace_back(sizes[t], 0.0f);
      cv.emplace_back(sizes[t], 0.0f);
    }
    for (std::size_t t = 0; t < sizes.size(); ++t) {
      cpu_slots.push_back({cp[t].data(), cg[t].data(), cm[t].data(), cv[t].data(), sizes[t]});
    }
    std::vector<Dev> dp, dg, dm, dv;
    dp.reserve(sizes.size());
    dg.reserve(sizes.size());
    dm.reserve(sizes.size());
    dv.reserve(sizes.size());
    for (std::size_t t = 0; t < sizes.size(); ++t) {
      dp.emplace_back(cp[t]);
      dg.emplace_back(sizes[t]);
      dm.emplace_back(sizes[t]);
      dv.emplace_back(sizes[t]);
      nc::DeviceParamSlot slot;
      slot.param = dp.back().p();
      slot.grad = dg.back().p();
      slot.exp_avg = dm.back().p();
      slot.exp_avg_sq = dv.back().p();
      slot.numel = sizes[t];
      slots.push_back(slot);
    }
    std::uint64_t total = 0;
    for (const auto s : sizes) {
      total += s;
    }
    nc::DeviceBuffer scratch(nc::adamw_scratch_bytes(static_cast<int>(slots.size()), total));
    Dev norm(1);
    nc::DeviceAdamWConfig gcfg;
    gcfg.lr = 1e-3;
    gcfg.weight_decay = 0.01;
    gcfg.max_grad_norm = max_norm;
    fu::AdamWConfig ccfg;
    ccfg.lr = 1e-3;
    ccfg.weight_decay = 0.01;
    ccfg.max_grad_norm = max_norm;
    double worst = 0.0;
    double norm_err = 0.0;
    for (int step = 1; step <= 1000; ++step) {
      // Gradyan = CPU parametresinin fonksiyonu; GPU'ya ayni gradyan yuklenir (iki yol ayni girdiyi gorur).
      for (std::size_t t = 0; t < sizes.size(); ++t) {
        for (std::size_t i = 0; i < sizes[t]; ++i) {
          cg[t][i] = 0.3f * cp[t][i] + 0.05f * std::sin(static_cast<float>(step * 7 + i * 3 + t));
        }
        NEXUS_CUDA_CHECK(cudaMemcpyAsync(dg[t].p(), cg[t].data(), sizes[t] * 4, cudaMemcpyHostToDevice, g_stream));
      }
      const auto info = fu::adamw_multi_tensor(cpu_slots, ccfg, step);
      nc::adamw_multi_tensor(slots.data(), static_cast<int>(slots.size()), gcfg, step, norm.p(), scratch.data(),
                             g_stream);
      if (step % 100 == 0 || step == 1) {
        for (std::size_t t = 0; t < sizes.size(); ++t) {
          worst = std::max(worst, rel_err(dp[t].get(), cp[t]));
          // GPU parametresini CPU'nunkiyle hizala: karsilastirma her adimin hatasini olcer, birikmeyi degil.
          NEXUS_CUDA_CHECK(cudaMemcpyAsync(dp[t].p(), cp[t].data(), sizes[t] * 4, cudaMemcpyHostToDevice, g_stream));
          NEXUS_CUDA_CHECK(cudaMemcpyAsync(dm[t].p(), cm[t].data(), sizes[t] * 4, cudaMemcpyHostToDevice, g_stream));
          NEXUS_CUDA_CHECK(cudaMemcpyAsync(dv[t].p(), cv[t].data(), sizes[t] * 4, cudaMemcpyHostToDevice, g_stream));
        }
        if (max_norm > 0.0) {
          norm_err = std::max(norm_err, std::fabs(norm.get()[0] - info.grad_norm) / info.grad_norm);
        }
      }
    }
    const std::string tag = max_norm > 0.0 ? " + global norm kirpma" : "";
    nexus_test::info(max_norm > 0.0 ? "gpu_adamw_rel_clip" : "gpu_adamw_rel", worst);
    report("GPU foreach AdamW = CPU foreach AdamW (1000 adim, <= 1e-6)" + tag, worst, 1e-6);
    if (max_norm > 0.0) {
      report("GPU global norm = CPU global norm", norm_err, 1e-6);
    }
  }
}

int run() {
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return 77;
  }
  nc::Stream stream = nc::Stream::create(0);
  g_stream = stream.handle();
  nc::StreamGuard guard(stream);
  test_gemm();
  test_norms();
  test_softmax();
  test_embedding();
  test_cross_entropy();
  test_dropout();
  test_adamw();
  NEXUS_CUDA_CHECK(cudaStreamSynchronize(g_stream));
  return nexus_test::finish("faz3-cuda-cekirdekleri");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    return run();
  } catch (const std::exception& error) {
    std::printf("  FAIL  beklenmeyen istisna: %s\n", error.what());
    return 1;
  }
}
