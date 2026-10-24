/** FAZ 4 CUDA: dar GEMM/epilog, CastCache ve FP8 emulasyon referans testleri. */
#include "nexus/core/fp8.hpp"
#include "nexus/core/half.hpp"
#include "nexus/cuda/cast_cache.hpp"
#include "nexus/cuda/fp8.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus_test.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace {
nc::Stream g_stream;
nc::DeviceBuffer upload(const std::vector<float>& x) {
  nc::DeviceBuffer d(x.size() * 4, g_stream);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), x.data(), x.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
  return d;
}
std::vector<float> download(const void* p, std::size_t n, nexus::DType type = nexus::DType::F32) {
  nc::DeviceBuffer f(n * 4, g_stream);
  nc::cast(f.data(), nexus::DType::F32, p, type, n, g_stream.handle());
  std::vector<float> out(n);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(out.data(), f.data(), n * 4, cudaMemcpyDeviceToHost, g_stream.handle()));
  g_stream.synchronize();
  return out;
}
nc::DeviceBuffer narrow(const nc::DeviceBuffer& f, std::size_t n, nexus::DType type) {
  nc::DeviceBuffer out(n * nexus::size_of(type), g_stream);
  nc::cast(out.data(), type, f.data(), nexus::DType::F32, n, g_stream.handle());
  return out;
}
float narrow_host(float x, nexus::DType t) {
  return t == nexus::DType::BF16 ? nexus::bfloat16_bits_to_float(nexus::float_to_bfloat16_bits(x))
                                 : nexus::half_bits_to_float(nexus::float_to_half_bits(x));
}
std::vector<float> random(std::size_t n, std::uint64_t seed, float scale = 1.0f) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> x(n);
  for (float& v : x) v = d(rng);
  return x;
}
double error(const std::vector<float>& got, const std::vector<double>& want) {
  double e = 0.0, s = 1e-12;
  for (std::size_t i = 0; i < want.size(); ++i) {
    e = std::max(e, std::fabs(static_cast<double>(got[i]) - want[i]));
    s = std::max(s, std::fabs(want[i]));
  }
  return e / s;
}
void check_error(const char* name, const std::vector<float>& got, const std::vector<double>& want, double tolerance) {
  const double e = error(got, want);
  std::printf("  info  %-38s rel=%.3e tol=%.1e\n", name, e, tolerance);
  nexus_test::expect(e < tolerance, name);
}
void gemm_tests() {
  nexus_test::section("FAZ 4.A BF16/F16 GEMM ve FP32 epilog ciktilari");
  constexpr int m = 48, n = 40, k = 32;
  const auto xh = random(m * k, 41), wh = random(k * n, 42, 0.25f), bh = random(n, 43);
  for (const nexus::DType type : {nexus::DType::BF16, nexus::DType::F16}) {
    const double tol = type == nexus::DType::BF16 ? 0.025 : 0.004;
    auto xq = xh, wq = wh, bq = bh;
    for (float& v : xq) v = narrow_host(v, type);
    for (float& v : wq) v = narrow_host(v, type);
    for (float& v : bq) v = narrow_host(v, type);
    auto dx = upload(xh), dw = upload(wh), db = upload(bh);
    auto dxq = narrow(dx, xh.size(), type), dwq = narrow(dw, wh.size(), type), dbq = narrow(db, bh.size(), type);
    nc::DeviceBuffer y(m * n * nexus::size_of(type), g_stream), aux(m * n * nexus::size_of(type), g_stream);
    nc::GemmArgs g;
    g.m = m; g.n = n; g.k = k; g.a = dxq.data(); g.lda = k; g.b = dwq.data(); g.ldb = n;
    g.c = y.data(); g.ldc = n; g.a_type = type; g.b_type = type; g.c_type = type; g.bias_type = type;
    g.bias = dbq.data(); g.aux = aux.data(); g.epilogue = nc::Epilogue::BiasGelu;
    nc::gemm(g, g_stream.handle());
    std::vector<double> ref(m * n);
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
      double sum = bq[j];
      for (int z = 0; z < k; ++z) sum += static_cast<double>(xq[i * k + z]) * wq[z * n + j];
      ref[i * n + j] = 0.5 * sum * (1.0 + std::tanh(0.7978845608028654 * (sum + 0.044715 * sum * sum * sum)));
    }
    check_error(type == nexus::DType::BF16 ? "BF16 bias+GELU" : "F16 bias+GELU", download(y.data(), m * n, type), ref, tol);
    nc::DeviceBuffer grad(k * n * 4, g_stream), dbgrad(n * 4, g_stream);
    g = nc::GemmArgs{}; g.trans_a = true; g.m = k; g.n = n; g.k = m;
    g.a = dxq.data(); g.lda = k; g.b = y.data(); g.ldb = n; g.c = grad.data(); g.ldc = n;
    g.a_type = type; g.b_type = type; g.c_type = nexus::DType::F32; g.bias_type = nexus::DType::F32;
    g.epilogue = nc::Epilogue::BGradB; g.bias_grad = dbgrad.data();
    nc::gemm(g, g_stream.handle());
    const auto yh = download(y.data(), m * n, type);
    std::vector<double> dwref(k * n), dbref(n, 0.0);
    for (int i = 0; i < k; ++i) for (int j = 0; j < n; ++j) {
      for (int z = 0; z < m; ++z) dwref[i * n + j] += static_cast<double>(xq[z * k + i]) * yh[z * n + j];
    }
    for (int z = 0; z < m; ++z) for (int j = 0; j < n; ++j) dbref[j] += yh[z * n + j];
    check_error("dar dW -> FP32", download(grad.data(), k * n), dwref, tol);
    check_error("dar BGradB -> FP32", download(dbgrad.data(), n), dbref, tol);
  }
}
void cache_tests() {
  nexus_test::section("FAZ 4.B autocast CastCache surum ve akis");
  const auto host = random(1027, 51);
  auto src = upload(host);
  nc::CastCache cache;
  const void* a = cache.get(src.as<float>(), host.size(), nexus::DType::BF16, 7, g_stream);
  const void* b = cache.get(src.as<float>(), host.size(), nexus::DType::BF16, 7, g_stream);
  nexus_test::expect(a == b && cache.hits() == 1 && cache.casts() == 1, "CastCache ayni surumde isabet");
  (void)cache.get(src.as<float>(), host.size(), nexus::DType::BF16, 8, g_stream);
  nexus_test::expect(cache.casts() == 2 && cache.entries() == 1, "CastCache surum degisiminde yeniden cast");
  nc::Stream second = nc::Stream::create(0);
  (void)cache.get(src.as<float>(), host.size(), nexus::DType::BF16, 8, second);
  nexus_test::expect(cache.casts() == 3 && cache.entries() == 2, "CastCache akislar arasi yarissa ayri kayit");
  second.synchronize();
  second.destroy();
  cache.clear();
  nexus_test::expect(cache.entries() == 0, "CastCache adim sonunda temizlenir");
}
void fp8_tests() {
  nexus_test::section("FAZ 4.5 FP8 CUDA/host bit esitligi ve emule GEMM");
  std::vector<float> host;
  for (int code = 0; code < 128; ++code) {
    for (const auto f : {nexus::Fp8Format::E4M3, nexus::Fp8Format::E5M2}) {
      const float v = nexus::fp8_to_float(static_cast<std::uint8_t>(code), f);
      if (std::isfinite(v)) host.push_back(v * 0.01f);
    }
  }
  const auto noise = random(4096, 61, 20.0f);
  host.insert(host.end(), noise.begin(), noise.end());
  auto x = upload(host);
  for (const auto format : {nexus::Fp8Format::E4M3, nexus::Fp8Format::E5M2}) {
    constexpr float scale = 1.75f;
    nc::DeviceBuffer q(host.size(), g_stream), amax(4, g_stream);
    NEXUS_CUDA_CHECK(cudaMemsetAsync(amax.data(), 0, 4, g_stream.handle()));
    nc::fp8_quantize(q.as<std::uint8_t>(), x.as<float>(), host.size(), scale, format, amax.as<float>(), g_stream.handle());
    std::vector<std::uint8_t> got(host.size());
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(got.data(), q.data(), got.size(), cudaMemcpyDeviceToHost, g_stream.handle()));
    const float got_amax = download(amax.data(), 1)[0];
    std::vector<float> deq(host.size() * 4);
    nc::DeviceBuffer deq_d(host.size() * 4, g_stream);
    nc::fp8_dequantize(deq_d.data(), nexus::DType::F32, q.as<std::uint8_t>(), host.size(), scale, format, g_stream.handle());
    deq = download(deq_d.data(), host.size());
    float want_amax = 0.0f;
    bool bits_equal = true;
    double deq_error = 0.0;
    for (std::size_t i = 0; i < host.size(); ++i) {
      const auto want = nexus::float_to_fp8(host[i] * scale, format);
      bits_equal &= got[i] == want;
      const float expected = nexus::fp8_to_float(want, format) / scale;
      deq_error = std::max(deq_error, static_cast<double>(std::fabs(deq[i] - expected)) /
                                          std::max(1.0f, std::fabs(expected)));
      want_amax = std::max(want_amax, std::fabs(host[i]));
    }
    nexus_test::expect(bits_equal, "FP8 CUDA quantization host RNE ile bit-bit ayni");
    nexus_test::expect(deq_error < 1e-6, "FP8 dequantization olcegi (1 ULP carpma/bolme toleransi)");
    nexus_test::expect(got_amax == want_amax, "FP8 amax indirgeme");
  }
  constexpr int m = 16, n = 24, k = 32;
  constexpr float sa = 3.0f, sb = 5.0f;
  const auto ah = random(m * k, 71, 20.0f), bh = random(k * n, 72, 8.0f);
  auto a = upload(ah), b = upload(bh);
  nc::DeviceBuffer c(m * n * 4, g_stream), scratch(nc::fp8_gemm_emulated_scratch_bytes(m, n, k), g_stream);
  nc::fp8_gemm_emulated(c.as<float>(), a.as<float>(), b.as<float>(), m, n, k, sa, sb,
                        nexus::Fp8Format::E4M3, scratch.data(), g_stream.handle());
  std::vector<double> ref(m * n, 0.0);
  for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) for (int z = 0; z < k; ++z) {
    const float av = nexus::fp8_to_float(nexus::float_to_fp8(ah[i * k + z] * sa, nexus::Fp8Format::E4M3), nexus::Fp8Format::E4M3) / sa;
    const float bv = nexus::fp8_to_float(nexus::float_to_fp8(bh[z * n + j] * sb, nexus::Fp8Format::E4M3), nexus::Fp8Format::E4M3) / sb;
    ref[i * n + j] += static_cast<double>(av) * bv;
  }
  check_error("FP8 E4M3 emule GEMM", download(c.data(), m * n), ref, 2e-5);
  if (!nc::fp8_native_supported(0)) {
    bool threw = false;
    try { nc::require_fp8_native(0); } catch (const nexus::TensorError&) { threw = true; }
    nexus_test::expect(threw, "sm_86 yerel FP8 icin acik hata");
  }
}
}  // namespace
int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) { std::printf("  SKIP  CUDA aygiti yok\n"); return 77; }
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    g_stream = nc::Stream::create(0);
    nc::StreamGuard guard(g_stream);
    if (mode == "all" || mode == "gemm") gemm_tests();
    if (mode == "all" || mode == "cache") cache_tests();
    if (mode == "all" || mode == "fp8") fp8_tests();
    g_stream.synchronize();
    g_stream.destroy();
    return nexus_test::finish("phase4_cuda_kernels");
  } catch (const std::exception& e) {
    std::printf("  FAIL  %s\n", e.what());
    return 1;
  }
}

