/**
 * @file test_tensor_cuda.cpp
 * @brief Dalga 2: Tensor dagiticisinin CUDA cekirdekleri, host cekirdeklerine / double referansa karsi.
 *
 *   test_tensor_cuda copy   : H2D / D2H / D2D, adimli + dtype donusumlu kopya (host ile bit bit), hata bayragi
 *   test_tensor_cuda ops    : add / mul yayin + adimli (host ile bit bit), sum, fill, matmul (devrik gorunum)
 *   test_tensor_cuda errors : kayitsiz (FP8) / farkli cihaz / kayit oncesi davranis
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/core/tensor.hpp"
#include "nexus/core/tensor_ops.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/runtime.hpp"
#include "nexus/cuda/tensor_kernels.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace ops = nexus::ops;
using nexus::DeviceRef;
using nexus::DType;
using nexus::MemorySpace;
using nexus::Shape;
using nexus::Tensor;

namespace {

const DeviceRef kGpu{nexus::DeviceKind::Cuda, 0};

bool throws(const std::function<void()>& f, const char* needle = nullptr) {
  try {
    f();
  } catch (const std::exception& e) {
    return needle == nullptr || std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

std::vector<double> randv(std::size_t n, unsigned seed, double scale = 1.0) {
  std::mt19937 g(seed);
  std::normal_distribution<double> d(0.0, scale);
  std::vector<double> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

Tensor gpu(const Tensor& h) { return h.to(MemorySpace::Device, kGpu); }
Tensor host(const Tensor& d) { return d.to(MemorySpace::Host, DeviceRef::cpu()); }

// Iki tensörün bayt bayt (mantiksal sirada) esitligi
bool same_bits(const Tensor& a, const Tensor& b) {
  const Tensor x = host(a).contiguous(), y = host(b).contiguous();
  return x.dtype() == y.dtype() && x.shape() == y.shape() &&
         std::memcmp(x.raw_data(), y.raw_data(), x.numel() * x.element_size()) == 0;
}

void test_copy() {
  nexus_test::section("copy: H2D / D2H / D2D, adimli + donusumlu");
  const Tensor h = Tensor::from_values(Shape{3, 4, 5}, randv(60, 1, 3.0), DType::F32);
  const Tensor d = gpu(h);
  nexus_test::expect(d.space() == MemorySpace::Device && same_bits(d, h), "bitisik H2D -> D2H gidis-donus bit bit");
  const std::size_t order[] = {2, 0, 1};
  const Tensor hp = h.permute(order).contiguous();
  const Tensor dp = d.permute(order).contiguous();  // cihazda adimli -> bitisik cekirdek
  nexus_test::expect(dp.space() == MemorySpace::Device && same_bits(dp, hp), "cihazda permute().contiguous() == host");
  bool ok = true;
  for (const DType t : {DType::BF16, DType::F16, DType::F64, DType::I32, DType::I64, DType::I8, DType::U8, DType::Bool}) {
    const Tensor src = (t == DType::I8 || t == DType::U8) ? Tensor::from_values(Shape{60}, std::vector<double>(60, 7.75), DType::F32) : h;
    const Tensor want = src.transpose(0, src.rank() - 1).to(t);
    const Tensor got = gpu(src).transpose(0, src.rank() - 1).to(t);
    const bool eq = same_bits(got, want);
    if (!eq) std::printf("  info  dtype %s farkli\n", std::string(nexus::to_string(t)).c_str());
    ok = ok && eq;
  }
  nexus_test::expect(ok, "cihaz devrik + donusum (BF16/F16/F64/I32/I64/I8/U8/Bool) host ile bit bit");
  const Tensor back = gpu(h.to(DType::BF16)).to(DType::F32);
  nexus_test::expect(same_bits(back, h.to(DType::BF16).to(DType::F32)), "cihaz BF16 -> F32");
  // host adimli -> cihaz adimli hedef
  Tensor dst = Tensor::zeros(Shape{5, 4}, DType::F64, MemorySpace::Device, kGpu);
  Tensor dst_view = dst.transpose(0, 1);  // [4,5] adimli
  dst_view.copy_from(h.select(0, 1));     // host [4,5] F32 -> cihaz adimli F64
  nexus_test::expect(same_bits(dst, h.select(0, 1).transpose(0, 1).to(DType::F64)), "host -> cihaz adimli + donusumlu hedef");
  // cihaz -> host adimli hedef
  Tensor hd = Tensor::zeros(Shape{5, 4}, DType::F32);
  hd.transpose(0, 1).copy_from(d.select(0, 2));
  nexus_test::expect(same_bits(hd, h.select(0, 2).transpose(0, 1)), "cihaz -> host adimli hedef");
  const Tensor nan = gpu(Tensor::from_values(Shape{2}, std::vector<double>{1.0, NAN}, DType::F32));
  nexus_test::expect(throws([&] { (void)nan.to(DType::I32); }, "NaN"), "cihaz NaN -> I32 acik hata (bayrak)");
  const Tensor big = gpu(Tensor::from_values(Shape{1}, std::vector<double>{300.0}, DType::F32));
  nexus_test::expect(throws([&] { (void)big.to(DType::I8); }, "aralik"), "cihaz 300 -> I8 acik hata");
}

double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0, s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    e = std::max(e, std::fabs(a[i] - b[i]));
    s = std::max(s, std::fabs(b[i]));
  }
  return e / std::max(s, 1e-30);
}

std::vector<double> ref_mm(const Tensor& a, const Tensor& b) {
  const Tensor x = host(a).to(DType::F64), y = host(b).to(DType::F64);
  const std::size_t M = x.shape()[0], K = x.shape()[1], N = y.shape()[1];
  const auto xv = x.to_double_vector(), yv = y.to_double_vector();
  std::vector<double> c(M * N, 0.0);
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t p = 0; p < K; ++p)
      for (std::size_t j = 0; j < N; ++j) c[i * N + j] += xv[i * K + p] * yv[p * N + j];
  return c;
}

void test_ops() {
  nexus_test::section("ops: cihaz cekirdekleri");
  const Tensor a = Tensor::from_values(Shape{4, 1, 6}, randv(24, 2), DType::F32);
  const Tensor b = Tensor::from_values(Shape{3, 6}, randv(18, 3), DType::F32);
  bool ok = true;
  for (const DType t : {DType::F32, DType::BF16, DType::F16, DType::F64}) {
    const Tensor ha = a.to(t), hb = b.to(t);
    ok = ok && same_bits(ops::add(gpu(ha), gpu(hb)), ops::add(ha, hb));
    ok = ok && same_bits(ops::mul(gpu(ha).transpose(0, 2), gpu(ha).transpose(0, 2)), ops::mul(ha.transpose(0, 2), ha.transpose(0, 2)));
  }
  const Tensor ia = Tensor::from_values(Shape{5}, std::vector<double>{1, -2, 3, 40000, -5}, DType::F32).to(DType::I64);
  ok = ok && same_bits(ops::mul(gpu(ia), gpu(ia)), ops::mul(ia, ia));
  nexus_test::expect(ok, "add / mul (yayin [4,1,6]+[3,6], adimli) F32/BF16/F16/F64/I64 host ile bit bit");
  const Tensor big = Tensor::from_values(Shape{1 << 16}, randv(1 << 16, 4), DType::F32);
  const double hs = ops::sum(big)[0];
  const Tensor ds1 = host(ops::sum(gpu(big))), ds2 = host(ops::sum(gpu(big)));
  std::printf("  info  sum 65536: host %.9g cihaz %.9g\n", hs, static_cast<double>(ds1[0]));
  nexus_test::expect(std::fabs(ds1[0] - hs) <= 1e-6 * std::fabs(hs) + 1e-6 && same_bits(ds1, ds2),
                     "sum: FP64 biriktirme (host'a <= 1e-6), cihazda belirlenimci (iki cagri bit bit)");
  nexus_test::expect(host(ops::sum(gpu(Tensor::from_values(Shape{4}, std::vector<double>{1, 2, 3, 4}, DType::I32)))).data_as<std::int64_t>()[0] == 10,
                     "sum I32 -> I64 cihaz");
  Tensor f = Tensor::zeros(Shape{4, 6}, DType::BF16, MemorySpace::Device, kGpu);
  Tensor fv = f.slice(1, 1, 6, 2);
  ops::fill_(fv, 2.5);
  const auto fh = host(f).to(DType::F32).to_double_vector();
  ok = true;
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t c = 0; c < 6; ++c) ok = ok && fh[r * 6 + c] == (c % 2 == 1 ? 2.5 : 0.0);
  nexus_test::expect(ok, "fill_ adimli cihaz gorunumu (komsulara dokunmaz)");
  nexus::cuda::set_float32_matmul_precision(nexus::cuda::MatmulPrecision::Highest);
  double w32 = 0, w16 = 0;
  for (const auto& [M, K, N] : std::vector<std::array<std::size_t, 3>>{{5, 7, 3}, {128, 256, 96}, {1, 300, 17}}) {
    const Tensor A = gpu(Tensor::from_values(Shape{M, K}, randv(M * K, 5), DType::F32));
    const Tensor B = gpu(Tensor::from_values(Shape{K, N}, randv(K * N, 6), DType::F32));
    const Tensor Bt = gpu(Tensor::from_values(Shape{N, K}, randv(N * K, 7), DType::F32)).transpose(0, 1);
    const Tensor At = gpu(Tensor::from_values(Shape{K, M}, randv(K * M, 8), DType::F32)).transpose(0, 1);
    for (const auto& [x, y] : std::vector<std::pair<const Tensor*, const Tensor*>>{{&A, &B}, {&A, &Bt}, {&At, &B}, {&At, &Bt}}) {
      w32 = std::max(w32, max_rel(host(ops::matmul(*x, *y)).to_double_vector(), ref_mm(*x, *y)));
      for (const DType t : {DType::BF16, DType::F16}) {
        const Tensor xs = x->to(t), ys = y->to(t);
        w16 = std::max(w16, max_rel(host(ops::matmul(xs, ys)).to(DType::F64).to_double_vector(), ref_mm(xs, ys)));
      }
    }
  }
  std::printf("  info  cihaz matmul F32 (Highest) en kotu %.2e; BF16/F16 %.2e\n", w32, w16);
  nexus_test::expect(w32 <= 1e-5, "cihaz matmul F32 (devrik gorunumler dahil) double referansla <= 1e-5");
  nexus_test::expect(w16 <= 1e-2, "cihaz matmul BF16/F16 (FP32 biriktirme, cikti yuvarlamasi) <= 1e-2");
}

namespace ag = nexus::autograd;

std::vector<double> gvals(const ag::Variable& v) {
  return v.grad().is_tensor_backed() ? host(v.grad().tensor()).to(DType::F64).to_double_vector()
                                     : std::vector<double>(v.grad().values().begin(), v.grad().values().end());
}

void test_autograd() {
  nexus_test::section("autograd: cihazda Tensor yolu");
  nexus::cuda::set_float32_matmul_precision(nexus::cuda::MatmulPrecision::Highest);
  const std::size_t M = 64, K = 48, N = 32;
  const auto av = randv(M * K, 21), bv = randv(K * N, 22), cv = randv(M * N, 23);
  auto leaf = [](const std::vector<double>& v, std::size_t r, std::size_t c, DType t, bool on_gpu) {
    Tensor h = Tensor::from_values(Shape{r, c}, v, t);
    return ag::Variable::wrap(on_gpu ? gpu(h) : h, true);
  };
  auto graph = [](const ag::Variable& A, const ag::Variable& B, const ag::Variable& C) {
    return ag::sum(ag::add(ag::mul(ag::matmul(A, B), C), C));
  };
  double worst32 = 0, worst16 = 0;
  for (const DType t : {DType::F32, DType::BF16}) {
    ag::Variable A = leaf(av, M, K, t, true), B = leaf(bv, K, N, t, true), C = leaf(cv, M, N, t, true);
    ag::Variable Ah = leaf(av, M, K, DType::F32, false), Bh = leaf(bv, K, N, DType::F32, false), Ch = leaf(cv, M, N, DType::F32, false);
    graph(A, B, C).backward();
    graph(Ah, Bh, Ch).backward();
    const bool on = A.grad().on_device() && B.grad().on_device() && A.grad().dtype() == t;
    nexus_test::expect(on, std::string("cihaz gradyanlari cihazda, dtype ") + std::string(nexus::to_string(t)));
    const double e = std::max({max_rel(gvals(A), gvals(Ah)), max_rel(gvals(B), gvals(Bh)), max_rel(gvals(C), gvals(Ch))});
    (t == DType::F32 ? worst32 : worst16) = e;
  }
  std::printf("  info  cihaz autograd vs host Tensor yolu: F32 %.2e, BF16 %.2e\n", worst32, worst16);
  nexus_test::expect(worst32 <= 1e-5, "cihaz F32 autograd gradyanlari host'a <= 1e-5");
  nexus_test::expect(worst16 <= 3e-2, "cihaz BF16 autograd gradyanlari host F32'ye <= 3e-2");
  // host FAZ 2 vector yapragi -> to(GPU) -> cihazda islem -> gradyan host vector olarak geri
  ag::Variable x = ag::Variable::make(std::vector<float>(av.begin(), av.end()), ag::Shape{{static_cast<std::int64_t>(M), static_cast<std::int64_t>(K)}}, true);
  ag::Variable w = leaf(bv, K, N, DType::F32, true);
  ag::Variable y = ag::sum(ag::matmul(ag::to(x, MemorySpace::Device, kGpu), w));
  y.backward();
  bool ok = !x.grad().is_tensor_backed() && x.grad().values().size() == M * K;
  // dx = 1 * W^T: satir i'nin elemani k = sum_j W[k,j]
  for (std::size_t k = 0; k < K && ok; ++k) {
    double rs = 0;
    for (std::size_t j = 0; j < N; ++j) rs += static_cast<float>(bv[k * N + j]);
    ok = std::fabs(x.grad().values()[k] - rs) <= 1e-4 * std::max(1.0, std::fabs(rs));
  }
  nexus_test::expect(ok, "host vector yaprak -> to(GPU) -> matmul: gradyan host F32 vector olarak doner, dogru");
}

void test_errors() {
  nexus_test::section("errors: acik hatalar");
  const Tensor d = gpu(Tensor::from_values(Shape{4}, randv(4, 9), DType::F32));
  nexus_test::expect(throws([&] { (void)d.to(DType::F8E4M3); }, "cekirdek yok"), "cihaz FP8 donusumu kayitsiz -> acik hata");
  nexus_test::expect(throws([&] { (void)ops::matmul(gpu(Tensor::zeros(Shape{2, 2}, DType::I32)), gpu(Tensor::zeros(Shape{2, 2}, DType::I32))); },
                            "cekirdek yok"),
                     "cihaz I32 matmul kayitsiz -> acik hata");
  nexus_test::expect(throws([&] { (void)ops::add(d, host(d)); }, "ayni cihaz"), "host + cihaz karisik ikili islem acik hata");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nexus::cuda::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return 77;
  }
  const std::string mode = argc > 1 ? argv[1] : "copy";
  try {
    // kayit oncesi: cihaz cekirdegi yok -> acik hata ve oneri
    if (mode == "errors") {
      const Tensor d = Tensor::from_values(Shape{2, 2}, std::vector<double>{1, 2, 3, 4}, DType::F32).to(MemorySpace::Device, kGpu);
      nexus_test::expect(throws([&] { (void)ops::add(d, d); }, "register_tensor_kernels"),
                         "kayit oncesi cihaz islemi acik hata + register_tensor_kernels onerisi");
    }
    nexus::cuda::register_tensor_kernels();
    nexus::cuda::register_tensor_kernels();  // idempotent
    if (mode == "copy") test_copy();
    else if (mode == "ops") test_ops();
    else if (mode == "errors") test_errors();
    else if (mode == "autograd") test_autograd();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_tensor_cuda");
}
