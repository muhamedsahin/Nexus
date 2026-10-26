/**
 * @file test_phase14_moe_cuda.cpp
 * @brief FAZ 14 (GPU): MoE uzman GEMM'leri — uzman basina cuBLASLt dongusu vs dolgulu strided batched (tek cagri).
 *
 *   test_phase14_moe_cuda grouped : gercek yonlendirme dagilimiyla (moe-124m boyutlari, B*T = 8192, k = 2) up + down
 *                                   GEMM'leri; BF16 girdi, FP32 biriktirme. Dogruluk: dongu == batched (dolgusuz satirlar)
 *                                   ve CPU double referansina gore <= 1e-2. Olcum: TFLOPS (yararli FLOP ile).
 *   CUTLASS grouped GEMM bu depoda yok -> olculmedi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/runtime.hpp"
#include "nexus/moe/moe.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace moe = nexus::moe;

namespace {

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

std::uint16_t to_bf16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}
float from_bf16(std::uint16_t h) {
  const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

nc::DeviceBuffer upload(const std::vector<std::uint16_t>& h) {
  nc::DeviceBuffer b(std::max<std::size_t>(h.size(), 1) * 2);
  NEXUS_CUDA_CHECK(cudaMemcpy(b.data(), h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  return b;
}
std::vector<float> download(const nc::DeviceBuffer& b, std::size_t n) {
  std::vector<float> h(n);
  NEXUS_CUDA_CHECK(cudaMemcpy(h.data(), b.data(), n * 4, cudaMemcpyDeviceToHost));
  return h;
}

// A [rows, K] (satir-ana, BF16) x B [K, Nn] -> C [rows, Nn] F32
nc::GemmArgs args(int m, int n, int k, const void* a, const void* b, void* c) {
  nc::GemmArgs g;
  g.m = m;
  g.n = n;
  g.k = k;
  g.a = a;
  g.lda = k;
  g.b = b;
  g.ldb = n;
  g.c = c;
  g.ldc = n;
  g.a_type = nexus::DType::BF16;
  g.b_type = nexus::DType::BF16;
  g.c_type = nexus::DType::F32;
  return g;
}

void test_grouped() {
  const int E = 8, k = 2, d = 512, F = 1856, N = 8192;
  moe::Spec s;
  s.experts = E;
  s.top_k = k;
  s.d_model = d;
  s.d_ff = F;
  s.capacity_factor = 1.25f;
  const auto x = randn(static_cast<std::size_t>(N) * d, 1, 1.0f);
  const auto w = randn(static_cast<std::size_t>(d) * E, 2, 0.05f);
  const moe::Routing r = moe::route(x.data(), w.data(), nullptr, N, s, true, 0);
  const std::int64_t M = r.rows();
  std::vector<int> cnt(E);
  int Cmax = 0;
  for (int e = 0; e < E; ++e) {
    cnt[static_cast<std::size_t>(e)] = static_cast<int>(r.offsets[static_cast<std::size_t>(e) + 1] - r.offsets[static_cast<std::size_t>(e)]);
    Cmax = std::max(Cmax, cnt[static_cast<std::size_t>(e)]);
  }
  std::printf("  info  satir %lld (kapasite %lld, dusen %lld); uzman basina:", static_cast<long long>(M),
              static_cast<long long>(r.capacity), static_cast<long long>(r.dropped));
  for (int c : cnt) std::printf(" %d", c);
  std::printf("; dolgu Cmax %d (yararli/dolgulu FLOP %.2f)\n", Cmax, static_cast<double>(M) / (static_cast<double>(Cmax) * E));

  // Girdiler (BF16): xp [M, d], up [E][d][2F], hp (down girdisi) [M, F], down [E][F][d]
  auto bf = [](const std::vector<float>& v) {
    std::vector<std::uint16_t> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) o[i] = to_bf16(v[i]);
    return o;
  };
  const auto xp = bf(randn(static_cast<std::size_t>(M) * d, 3, 1.0f));
  const auto up = bf(randn(static_cast<std::size_t>(E) * d * 2 * F, 4, 0.04f));
  const auto hp = bf(randn(static_cast<std::size_t>(M) * F, 5, 1.0f));
  const auto dn = bf(randn(static_cast<std::size_t>(E) * F * d, 6, 0.03f));
  // Dolgulu kopyalar [E][Cmax][*] (dolgu satirlari 0)
  std::vector<std::uint16_t> xpad(static_cast<std::size_t>(E) * Cmax * d, 0), hpad(static_cast<std::size_t>(E) * Cmax * F, 0);
  for (int e = 0; e < E; ++e) {
    const std::int64_t o = r.offsets[static_cast<std::size_t>(e)];
    std::copy_n(xp.begin() + o * d, static_cast<std::size_t>(cnt[static_cast<std::size_t>(e)]) * d, xpad.begin() + static_cast<std::ptrdiff_t>(e) * Cmax * d);
    std::copy_n(hp.begin() + o * F, static_cast<std::size_t>(cnt[static_cast<std::size_t>(e)]) * F, hpad.begin() + static_cast<std::ptrdiff_t>(e) * Cmax * F);
  }
  const nc::DeviceBuffer dxp = upload(xp), dup = upload(up), dhp = upload(hp), ddn = upload(dn), dxpad = upload(xpad), dhpad = upload(hpad);
  nc::DeviceBuffer h_loop(static_cast<std::size_t>(M) * 2 * F * 4), y_loop(static_cast<std::size_t>(M) * d * 4);
  nc::DeviceBuffer h_bat(static_cast<std::size_t>(E) * Cmax * 2 * F * 4), y_bat(static_cast<std::size_t>(E) * Cmax * d * 4);
  cudaStream_t st = nullptr;

  auto run_loop = [&] {
    for (int e = 0; e < E; ++e) {
      const int m = cnt[static_cast<std::size_t>(e)];
      if (m == 0) continue;
      const std::int64_t o = r.offsets[static_cast<std::size_t>(e)];
      nc::gemm(args(m, 2 * F, d, dxp.as<std::uint16_t>() + o * d, dup.as<std::uint16_t>() + static_cast<std::size_t>(e) * d * 2 * F,
                    h_loop.as<float>() + o * 2 * F),
               st);
      nc::gemm(args(m, d, F, dhp.as<std::uint16_t>() + o * F, ddn.as<std::uint16_t>() + static_cast<std::size_t>(e) * F * d,
                    y_loop.as<float>() + o * d),
               st);
    }
  };
  auto run_batched = [&] {
    nc::GemmArgs a = args(Cmax, 2 * F, d, dxpad.data(), dup.data(), h_bat.data());
    a.batch = E;
    a.stride_a = static_cast<long long>(Cmax) * d;
    a.stride_b = static_cast<long long>(d) * 2 * F;
    a.stride_c = static_cast<long long>(Cmax) * 2 * F;
    nc::gemm(a, st);
    nc::GemmArgs b = args(Cmax, d, F, dhpad.data(), ddn.data(), y_bat.data());
    b.batch = E;
    b.stride_a = static_cast<long long>(Cmax) * F;
    b.stride_b = static_cast<long long>(F) * d;
    b.stride_c = static_cast<long long>(Cmax) * d;
    nc::gemm(b, st);
  };
  auto time_it = [&](auto&& fn) {
    for (int i = 0; i < 5; ++i) fn();
    cudaEvent_t a, b;
    NEXUS_CUDA_CHECK(cudaEventCreate(&a));
    NEXUS_CUDA_CHECK(cudaEventCreate(&b));
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
      NEXUS_CUDA_CHECK(cudaEventRecord(a, st));
      for (int i = 0; i < 20; ++i) fn();
      NEXUS_CUDA_CHECK(cudaEventRecord(b, st));
      NEXUS_CUDA_CHECK(cudaEventSynchronize(b));
      float ms = 0.0f;
      NEXUS_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      best = std::min(best, ms / 20.0f);
    }
    NEXUS_CUDA_CHECK(cudaEventDestroy(a));
    NEXUS_CUDA_CHECK(cudaEventDestroy(b));
    return static_cast<double>(best);
  };
  const double tl = time_it(run_loop), tb = time_it(run_batched);
  NEXUS_CUDA_CHECK(cudaDeviceSynchronize());
  const auto hl = download(h_loop, static_cast<std::size_t>(M) * 2 * F), yl = download(y_loop, static_cast<std::size_t>(M) * d);
  const auto hb = download(h_bat, static_cast<std::size_t>(E) * Cmax * 2 * F), yb = download(y_bat, static_cast<std::size_t>(E) * Cmax * d);
  double dmax = 0.0, scale = 0.0;
  for (int e = 0; e < E; ++e) {
    const std::int64_t o = r.offsets[static_cast<std::size_t>(e)];
    for (int i = 0; i < cnt[static_cast<std::size_t>(e)]; ++i) {
      for (int j = 0; j < 2 * F; ++j) {
        const double a = hl[static_cast<std::size_t>((o + i) * 2 * F + j)], b = hb[(static_cast<std::size_t>(e) * Cmax + i) * 2 * F + j];
        dmax = std::max(dmax, std::fabs(a - b));
        scale = std::max(scale, std::fabs(a));
      }
      for (int j = 0; j < d; ++j) {
        const double a = yl[static_cast<std::size_t>((o + i) * d + j)], b = yb[(static_cast<std::size_t>(e) * Cmax + i) * d + j];
        dmax = std::max(dmax, std::fabs(a - b));
      }
    }
  }
  // CPU double referansi: her uzmanin ilk 4 satiri
  double ref_err = 0.0, ref_scale = 0.0;
  for (int e = 0; e < E; ++e) {
    const std::int64_t o = r.offsets[static_cast<std::size_t>(e)];
    for (int i = 0; i < std::min(4, cnt[static_cast<std::size_t>(e)]); ++i) {
      for (int j = 0; j < 2 * F; j += 37) {
        double acc = 0.0;
        for (int q = 0; q < d; ++q)
          acc += static_cast<double>(from_bf16(xp[static_cast<std::size_t>((o + i) * d + q)])) *
                 from_bf16(up[(static_cast<std::size_t>(e) * d + q) * 2 * F + j]);
        ref_err = std::max(ref_err, std::fabs(acc - hl[static_cast<std::size_t>((o + i) * 2 * F + j)]));
        ref_scale = std::max(ref_scale, std::fabs(acc));
      }
    }
  }
  const double useful = 2.0 * static_cast<double>(M) * (static_cast<double>(d) * 2 * F + static_cast<double>(F) * d);
  std::printf("  info  uzman basina cuBLASLt dongusu (%d cagri): %.3f ms, %.2f TFLOPS\n", 2 * E, tl, useful / tl / 1e9);
  std::printf("  info  dolgulu strided batched (2 cagri): %.3f ms, %.2f TFLOPS yararli (%.2f dolgulu dahil), %.2fx\n", tb,
              useful / tb / 1e9, useful * (static_cast<double>(Cmax) * E / M) / tb / 1e9, tl / tb);
  std::printf("  info  CUTLASS grouped GEMM: bu depoda yok, olculmedi\n");
  std::printf("  info  dongu vs batched max fark %.2e (olcek %.2f); CPU double referans goreli %.2e\n", dmax, scale, ref_err / ref_scale);
  std::printf("{\"mode\": \"bench_moe_grouped_gemm_cuda\", \"rows\": %lld, \"cmax\": %d, \"loop_ms\": %.4f, \"batched_ms\": %.4f, "
              "\"loop_tflops\": %.3f, \"batched_tflops_useful\": %.3f}\n",
              static_cast<long long>(M), Cmax, tl, tb, useful / tl / 1e9, useful / tb / 1e9);
  nexus_test::expect(dmax <= 1e-3 * std::max(1.0, scale), "uzman dongusu == dolgulu batched (dolgusuz satirlar)");
  nexus_test::expect(ref_err / ref_scale <= 1e-2, "BF16 uzman GEMM CPU double referansina gore <= 1e-2");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "grouped";
  try {
    if (mode == "grouped") test_grouped();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase14_moe_cuda");
}
