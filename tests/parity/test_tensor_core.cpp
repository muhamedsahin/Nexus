/**
 * @file test_tensor_core.cpp
 * @brief Dalga 2: dtype'li adimli Tensor + dagitici.
 *
 *   test_tensor_core views    : kopyasiz gorunumler (paylasim, adimlar, surum), bitisiklik, reshape
 *   test_tensor_core dtype    : tum dtype'lar arasi donusum (RNE, kesme, aralik disi hata), from_values
 *   test_tensor_core ops      : add / mul yayin + adimli girdi, sum, matmul (devrik gorunum, BF16/F16/F64)
 *                               double referansa karsi; eksik cekirdek ve dtype uyusmazligi acik hata
 *   test_tensor_core legacy   : FAZ 1 API'si (alias / adopt_shared / derin kopya / operator[]) ayni anlam
 *   test_tensor_core bench    : eager islem ek yuku (us / islem), JSON satirlari
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/core/half.hpp"
#include "nexus/core/tensor.hpp"
#include "nexus/core/tensor_ops.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace ops = nexus::ops;
using nexus::DType;
using nexus::Shape;
using nexus::Tensor;

namespace {

bool throws(const std::function<void()>& f, const char* needle = nullptr) {
  try {
    f();
  } catch (const nexus::TensorError& e) {
    return needle == nullptr || std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

Tensor iota(Shape s, DType t = DType::F32) {
  std::vector<double> v(s.numel());
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<double>(i);
  return Tensor::from_values(s, v, t);
}

std::vector<double> randv(std::size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<double> d(0.0, 1.0);
  std::vector<double> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

void test_views() {
  nexus_test::section("views: kopyasiz gorunumler");
  Tensor t = iota(Shape{2, 3, 4});
  nexus_test::expect(t.is_contiguous() && t.stride(0) == 12 && t.stride(2) == 1, "bitisik adimlar 12/4/1");
  Tensor tr = t.transpose(0, 2);  // [4,3,2]
  nexus_test::expect(tr.shares_storage_with(t) && !tr.is_contiguous() && tr.shape() == Shape({4, 3, 2}),
                     "transpose: ayni depolama, bitisik degil, sekil 4x3x2");
  const std::int64_t i1[] = {3, 2, 1};
  nexus_test::expect(tr.item_at(i1) == 1 * 12 + 2 * 4 + 3, "transpose eleman t[1,2,3] = 23");
  Tensor sl = t.slice(2, 1, 4, 2);  // son eksen 1,3
  const std::int64_t i2[] = {1, 2, 1};
  nexus_test::expect(sl.shape() == Shape({2, 3, 2}) && sl.item_at(i2) == 12 + 8 + 3 && sl.stride(2) == 2,
                     "slice(step 2): sekil, adim 2, eleman");
  // yazi gorunumde gorunur + surum artar
  const std::uint64_t v0 = t.version();
  Tensor col = t.select(2, 0);  // [2,3], adim 4
  ops::fill_(col, -1.0);
  const std::int64_t i3[] = {1, 1, 0};
  const std::int64_t i4[] = {1, 1, 1};
  nexus_test::expect(t.item_at(i3) == -1.0 && t.item_at(i4) == 17.0 && t.version() > v0,
                     "select + fill_: ayni depolamaya yazar, komsuya dokunmaz, surum artar");
  Tensor e = iota(Shape{3, 1}).expand(Shape{2, 3, 4});
  const std::int64_t i5[] = {1, 2, 3};
  nexus_test::expect(e.stride(0) == 0 && e.stride(2) == 0 && e.item_at(i5) == 2.0, "expand: adim 0 yayin");
  nexus_test::expect(throws([&] { (void)t.transpose(0, 1).view_as(Shape{24}); }, "bitisik"),
                     "view bitisik olmayanda acik hata");
  Tensor rs = t.transpose(0, 1).reshape(Shape{24});
  nexus_test::expect(!rs.shares_storage_with(t) && rs.is_contiguous(), "reshape bitisik degilse kopyalar");
  Tensor same = t.reshape(Shape{6, 4});
  nexus_test::expect(same.shares_storage_with(t), "reshape bitisikse gorunum");
  const std::size_t order[] = {2, 0, 1};
  Tensor p = t.permute(order);
  const std::int64_t i6[] = {3, 1, 2};
  nexus_test::expect(p.shape() == Shape({4, 2, 3}) && p.item_at(i6) == 12 + 8 + 3, "permute");
  Tensor c = p.contiguous();
  bool ok = c.is_contiguous() && !c.shares_storage_with(t);
  const auto pv = p.to_double_vector(), cv = c.to_double_vector();
  ok = ok && pv == cv;
  nexus_test::expect(ok, "contiguous kopya mantiksal sirayi korur");
  Tensor u = t.unsqueeze(1);
  nexus_test::expect(u.shape() == Shape({2, 1, 3, 4}) && u.squeeze(1).shape() == t.shape() && u.shares_storage_with(t),
                     "unsqueeze / squeeze");
  nexus_test::expect(throws([&] { (void)t.slice(1, 2, 5); }) && throws([&] { (void)t.expand(Shape{2, 5, 4}); }) &&
                         throws([&] { (void)t.permute(std::vector<std::size_t>{0, 0, 1}); }),
                     "aralik disi slice / gecersiz expand / tekrarli permute acik hata");
  // derin kopya: depolama paylasmaz
  Tensor deep = sl;
  nexus_test::expect(!deep.shares_storage_with(t) && deep.is_contiguous() && deep.to_double_vector() == sl.to_double_vector(),
                     "kopya kurucusu derin + bitisik");
}

void test_dtype() {
  nexus_test::section("dtype: donusumler");
  const std::vector<double> v = {0.0, -0.0, 1.0, -1.5, 3.14159, 65504.0, 1e-3, -2.75};
  Tensor f = Tensor::from_values(Shape{8}, v, DType::F32);
  Tensor b = f.to(DType::BF16), h = f.to(DType::F16), d = f.to(DType::F64);
  bool ok = true;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const auto fv = static_cast<float>(v[i]);
    ok = ok && b.data_as<std::uint16_t>()[i] == nexus::float_to_bfloat16_bits(fv);
    ok = ok && h.data_as<std::uint16_t>()[i] == nexus::float_to_half_bits(fv);
    ok = ok && d.data_as<double>()[i] == static_cast<double>(fv);
  }
  nexus_test::expect(ok, "F32 -> BF16 / F16 (RNE, half.hpp ile ayni bit) / F64 tam");
  Tensor back = b.to(DType::F32);
  nexus_test::expect(back.dtype() == DType::F32 && back[3] == -1.5f, "BF16 -> F32 gidis-donus");
  Tensor i = Tensor::from_values(Shape{4}, std::vector<double>{2.9, -2.9, 0.4, -0.0}, DType::F32).to(DType::I32);
  nexus_test::expect(i.data_as<std::int32_t>()[0] == 2 && i.data_as<std::int32_t>()[1] == -2 && i.data_as<std::int32_t>()[2] == 0,
                     "kayan -> tamsayi: sifira dogru kesme");
  nexus_test::expect(throws([] { (void)Tensor::from_values(Shape{1}, std::vector<double>{300.0}, DType::F32).to(DType::I8); }, "aralik"),
                     "aralik disi (300 -> I8) acik hata");
  nexus_test::expect(throws([] { (void)Tensor::from_values(Shape{1}, std::vector<double>{NAN}, DType::F32).to(DType::I32); }, "NaN"),
                     "NaN -> tamsayi acik hata");
  nexus_test::expect(throws([] { (void)Tensor::from_values(Shape{1}, std::vector<double>{1.5}, DType::I32); }),
                     "from_values tamsayiya kesirli deger acik hata");
  Tensor big = Tensor::empty(Shape{2}, DType::I64);
  big.data_as<std::int64_t>()[0] = (std::int64_t{1} << 62) + 1;  // double'da temsil edilemez
  big.data_as<std::int64_t>()[1] = -7;
  Tensor big2 = big.slice(0, 0, 2, 1).to(DType::I64);
  nexus_test::expect(big2.data_as<std::int64_t>()[0] == (std::int64_t{1} << 62) + 1 && big2.data_as<std::int64_t>()[1] == -7,
                     "I64 kopya tam (2^62+1, double'dan gecmez)");
  nexus_test::expect(throws([&] { (void)big.to(DType::I32); }, "aralik"), "I64 2^62 -> I32 aralik disi acik hata");
  Tensor bo = Tensor::from_values(Shape{3}, std::vector<double>{0.0, 2.5, -1.0}, DType::F32).to(DType::Bool);
  nexus_test::expect(bo.data_as<std::uint8_t>()[0] == 0 && bo.data_as<std::uint8_t>()[1] == 1 && bo.data_as<std::uint8_t>()[2] == 1,
                     "kayan -> Bool: sifir disi 1");
  Tensor e4 = Tensor::from_values(Shape{2}, std::vector<double>{1.0, 1000.0}, DType::F32).to(DType::F8E4M3);
  nexus_test::expect(e4.to(DType::F32)[0] == 1.0f && e4.to(DType::F32)[1] == 448.0f, "F8E4M3 doygun (1000 -> 448)");
  nexus_test::expect(throws([&] { (void)b.data(); }, "F32") && throws([&] { (void)b.data_as<float>(); }),
                     "BF16 tensörde data() / yanlis boyutlu data_as acik hata");
  // adimli + donusumlu kopya
  Tensor m = iota(Shape{3, 4});
  Tensor mt = m.transpose(0, 1).to(DType::F64);
  const std::int64_t ix[] = {3, 2};
  nexus_test::expect(mt.is_contiguous() && mt.item_at(ix) == 11.0, "devrik gorunum -> F64 bitisik kopya");
}

double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0, s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    e = std::max(e, std::fabs(a[i] - b[i]));
    s = std::max(s, std::fabs(b[i]));
  }
  return e / std::max(s, 1e-30);
}

std::vector<double> ref_matmul(const Tensor& a, const Tensor& b) {
  const std::size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
  std::vector<double> c(M * N, 0.0);
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      double acc = 0;
      for (std::size_t p = 0; p < K; ++p) {
        const std::int64_t ia[] = {static_cast<std::int64_t>(i), static_cast<std::int64_t>(p)};
        const std::int64_t ib[] = {static_cast<std::int64_t>(p), static_cast<std::int64_t>(j)};
        acc += a.item_at(ia) * b.item_at(ib);
      }
      c[i * N + j] = acc;
    }
  return c;
}

void test_ops() {
  nexus_test::section("ops: dagitici + cekirdekler");
  Tensor a = Tensor::from_values(Shape{2, 3}, randv(6, 1), DType::F32);
  Tensor row = Tensor::from_values(Shape{3}, std::vector<double>{10, 20, 30}, DType::F32);
  Tensor s = ops::add(a, row);
  bool ok = s.shape() == Shape({2, 3});
  for (std::size_t i = 0; i < 6; ++i) ok = ok && s[i] == a[i] + row[i % 3];
  nexus_test::expect(ok, "add yayin [2,3] + [3]");
  Tensor at = Tensor::from_values(Shape{3, 2}, randv(6, 2), DType::F32).transpose(0, 1);
  Tensor m = ops::mul(at, a);
  ok = true;
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      const std::int64_t ix[] = {static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)};
      ok = ok && m.item_at(ix) == static_cast<double>(static_cast<float>(at.item_at(ix) * a.item_at(ix)));
    }
  nexus_test::expect(ok, "mul adimli (devrik) girdi");
  Tensor ab = ops::add(a.to(DType::BF16), row.to(DType::BF16));
  ok = true;
  for (std::size_t i = 0; i < 6; ++i)
    ok = ok && ab.data_as<std::uint16_t>()[i] ==
                   nexus::float_to_bfloat16_bits(nexus::bfloat16_bits_to_float(nexus::float_to_bfloat16_bits(a[i])) +
                                                 nexus::bfloat16_bits_to_float(nexus::float_to_bfloat16_bits(row[i % 3])));
  nexus_test::expect(ok, "BF16 add: FP32'de hesap, RNE");
  Tensor ii = ops::add(iota(Shape{4}, DType::I64), iota(Shape{4}, DType::I64));
  nexus_test::expect(ii.dtype() == DType::I64 && ii.data_as<std::int64_t>()[3] == 6, "I64 add");
  Tensor big = Tensor::from_values(Shape{1000}, randv(1000, 3), DType::F32);
  double ref = 0;
  for (std::size_t i = 0; i < 1000; ++i) ref += big[i];
  Tensor sm = ops::sum(big);
  nexus_test::expect(sm.shape() == Shape({1}) && sm[0] == static_cast<float>(ref), "sum FP64 biriktirir");
  nexus_test::expect(ops::sum(iota(Shape{5}, DType::I32)).dtype() == DType::I64 &&
                         ops::sum(iota(Shape{5}, DType::I32)).data_as<std::int64_t>()[0] == 10,
                     "sum I32 -> I64");
  // matmul: dtype x yerlesim, double referans
  double worst32 = 0, worst16 = 0;
  for (const auto& [M, K, N] : std::vector<std::array<std::size_t, 3>>{{5, 7, 3}, {64, 128, 96}, {1, 300, 17}, {33, 1, 9}}) {
    Tensor A = Tensor::from_values(Shape{M, K}, randv(M * K, 4), DType::F32);
    Tensor B = Tensor::from_values(Shape{K, N}, randv(K * N, 5), DType::F32);
    Tensor Bt = Tensor::from_values(Shape{N, K}, randv(N * K, 6), DType::F32).transpose(0, 1);  // [K,N] devrik gorunum
    Tensor At = Tensor::from_values(Shape{K, M}, randv(K * M, 7), DType::F32).transpose(0, 1);
    for (const auto& [x, y] : std::vector<std::pair<Tensor*, Tensor*>>{{&A, &B}, {&A, &Bt}, {&At, &B}, {&At, &Bt}}) {
      worst32 = std::max(worst32, max_rel(ops::matmul(*x, *y).to_double_vector(), ref_matmul(*x, *y)));
      const Tensor xb = x->to(DType::BF16), yb = y->transpose(0, 1).to(DType::BF16).transpose(0, 1);
      worst16 = std::max(worst16, max_rel(ops::matmul(xb, yb).to(DType::F64).to_double_vector(), ref_matmul(xb, yb)));
      const Tensor xh = x->to(DType::F16), yh = y->to(DType::F16);
      worst16 = std::max(worst16, max_rel(ops::matmul(xh, yh).to(DType::F64).to_double_vector(), ref_matmul(xh, yh)));
    }
    Tensor A64 = A.to(DType::F64), B64 = B.to(DType::F64);
    worst32 = std::max(worst32, max_rel(ops::matmul(A64, B64).to_double_vector(), ref_matmul(A64, B64)) * 1e6);  // F64 ~1e-15
  }
  std::printf("  info  matmul F32 (devrik gorunumler dahil) en kotu goreli %.2e; BF16/F16 (cikti RNE) %.2e\n", worst32, worst16);
  nexus_test::expect(worst32 <= 1e-5, "matmul F32 + F64 double referansla <= 1e-5 (F64 x1e6 olcekli)");
  nexus_test::expect(worst16 <= 1e-2, "matmul BF16 / F16: FP32 biriktirme, yalniz cikti yuvarlamasi (<= 1e-2)");
  nexus_test::expect(throws([&] { (void)ops::add(a, a.to(DType::F64)); }, "ayni dtype"), "dtype uyusmazligi acik hata");
  nexus_test::expect(throws([&] { (void)ops::add(a, Tensor::zeros(Shape{4})); }, "yayin"), "yayin uyusmazligi acik hata");
  const Tensor f8 = a.to(DType::F8E4M3);
  nexus_test::expect(throws([&] { (void)ops::add(f8, f8); }, "cekirdek yok"), "kayitsiz cekirdek (F8 add) acik hata");
  nexus_test::expect(throws([&] { (void)ops::matmul(iota(Shape{2, 2}, DType::I32), iota(Shape{2, 2}, DType::I32)); }, "cekirdek yok"),
                     "I32 matmul cekirdegi yok -> acik hata");
  nexus_test::expect(throws([] { ops::register_kernel(ops::Op::Add, DType::F32, ops::DeviceKind::Host, [](const ops::KernelArgs&) {}); },
                            "zaten kayitli"),
                     "ayni anahtara ikinci kayit acik hata");
  Tensor out = Tensor::empty(Shape{2, 3}, DType::F32);
  ops::add_out(out, a, row);
  nexus_test::expect(out.to_double_vector() == s.to_double_vector(), "add_out == add");
}

void test_legacy() {
  nexus_test::section("legacy: FAZ 1 API");
  std::vector<float> buf = {1, 2, 3, 4};
  Tensor al = Tensor::alias(buf.data(), Shape{4});
  al[1] = 20.0f;
  nexus_test::expect(buf[1] == 20.0f && al.data() == buf.data() && !al.owns_storage(), "alias: ayni adres, sahip degil");
  auto keep = std::make_shared<std::vector<float>>(std::vector<float>{5, 6});
  Tensor ad = Tensor::adopt_shared(keep->data(), Shape{2}, nexus::MemorySpace::Host, nexus::DeviceRef::cpu(), keep);
  const float* kp = keep->data();
  keep.reset();
  nexus_test::expect(ad.data() == kp && ad[1] == 6.0f, "adopt_shared yasam tutucuyu saklar");
  Tensor z = Tensor::zeros(Shape{3, 2});
  Tensor cp = z;
  cp[0] = 1.0f;
  nexus_test::expect(z[0] == 0.0f && cp.owns_storage() && z.dtype() == DType::F32, "kopya derin; zeros F32");
  Tensor fl = Tensor::full(Shape{3}, 2.5f);
  nexus_test::expect(fl[2] == 2.5f && fl.view().contiguous() && fl.view().numel() == 3, "full + view()");
  Tensor mv = std::move(fl);
  nexus_test::expect(mv[0] == 2.5f && fl.empty(), "tasima");
  nexus_test::expect(throws([&] { (void)z.transpose(0, 1)[1]; }, "bitisik"), "operator[] adimli tensörde acik hata");
}

void test_bridge() {
  nexus_test::section("bridge: autograd Variable <-> Tensor");
  namespace ag = nexus::autograd;
  ag::Variable v = ag::Variable::make({1, 2, 3, 4, 5, 6}, ag::Shape{{2, 3}}, true);
  Tensor tv = v.tensor_view();
  nexus_test::expect(tv.data() == v.values().data() && tv.shape() == Shape({2, 3}) && tv.dtype() == DType::F32,
                     "tensor_view kopyasiz (ayni adres), sekil 2x3, F32");
  Tensor col = tv.transpose(0, 1);
  const std::int64_t ix[] = {2, 1};
  nexus_test::expect(col.item_at(ix) == 6.0, "Variable verisi uzerinde adimli gorunum");
  Tensor kept;
  {
    ag::Variable tmp = ag::Variable::make({7, 8}, ag::Shape{{2}}, false);
    kept = tmp.tensor_view();
  }
  nexus_test::expect(kept[1] == 8.0f, "tensor_view Variable'dan uzun yasar (yasam tutucu)");
  ag::Variable sc = ag::Variable::make({3.5f}, ag::Shape{}, false);
  nexus_test::expect(sc.tensor_view().shape() == Shape({1}) && sc.tensor_view()[0] == 3.5f, "autograd skaleri {1} gorunur");
  Tensor b = Tensor::from_values(Shape{3, 2}, std::vector<double>{1, 2, 3, 4, 5, 6}, DType::F32).to(DType::BF16).transpose(0, 1);
  ag::Variable fv = ag::Variable::from_tensor(b, true);
  bool ok = fv.requires_grad() && fv.shape().dims == std::vector<std::int64_t>{2, 3};
  const std::vector<float> want = {1, 3, 5, 2, 4, 6};
  ok = ok && fv.values() == want;
  nexus_test::expect(ok, "from_tensor: BF16 devrik gorunum -> F32 bitisik Variable");
  // Tensor matmul == autograd matmul (ayni veri)
  ag::Variable x = ag::Variable::randn(ag::Shape{{5, 7}}, 1, false), w = ag::Variable::randn(ag::Shape{{7, 3}}, 2, false);
  const Tensor y = ops::matmul(x.tensor_view(), w.tensor_view());
  const ag::Variable yr = ag::matmul(x, w);
  double e = 0;
  for (std::size_t i = 0; i < y.numel(); ++i) e = std::max(e, static_cast<double>(std::fabs(y[i] - yr.values()[i])));
  std::printf("  info  ops::matmul(tensor_view) vs autograd::matmul en buyuk fark %.2e\n", e);
  nexus_test::expect(e <= 1e-5, "Tensor matmul == autograd matmul (ayni veri, <= 1e-5)");
}

int bench() {
  // Kucuk tensörde islem basina ek yuk (ayirma + dagitim + cekirdek). PyTorch referansi:
  // benchmarks/runner/dalga2_eager_overhead.py (ayni sekiller).
  for (const std::size_t n : {std::size_t{1}, std::size_t{64}, std::size_t{4096}}) {
    Tensor a = Tensor::from_values(Shape{n}, randv(n, 1), DType::F32), b = a.clone();
    Tensor out = Tensor::empty(Shape{n}, DType::F32);
    auto time = [&](auto&& f) {
      for (int i = 0; i < 1000; ++i) f();
      std::vector<double> best;
      for (int r = 0; r < 7; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10000; ++i) f();
        best.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 10000);
      }
      std::sort(best.begin(), best.end());
      return best[3];
    };
    const double alloc = time([&] { Tensor c = ops::add(a, b); });
    const double into = time([&] { ops::add_out(out, a, b); });
    const double view = time([&] { Tensor t = a.view_as(Shape{1, n}); });
    std::printf("{\"mode\": \"bench_eager_add\", \"n\": %zu, \"add_us\": %.3f, \"add_out_us\": %.3f, \"view_us\": %.3f}\n", n, alloc,
                into, view);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "views";
  try {
    if (mode == "views") test_views();
    else if (mode == "dtype") test_dtype();
    else if (mode == "ops") test_ops();
    else if (mode == "legacy") test_legacy();
    else if (mode == "bridge") test_bridge();
    else if (mode == "bench") return bench();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_tensor_core");
}
