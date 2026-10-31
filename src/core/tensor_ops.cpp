/**
 * @file tensor_ops.cpp
 * @brief Dalga 2: dagitici tablosu + islem girisleri + yerlesik CPU cekirdekleri.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor_ops.hpp"

#include "nexus/core/half.hpp"
#include "nexus/cpu/gemm.hpp"

#include <array>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

namespace nexus {
namespace ops {

namespace {

constexpr std::size_t kOps = static_cast<std::size_t>(Op::Count);
constexpr std::size_t kTypes = std::size(kAllDTypes);
constexpr std::size_t kDevs = static_cast<std::size_t>(DeviceKind::Count);

std::size_t dtype_index(DType t) noexcept { return static_cast<std::size_t>(t); }

struct Table {
  std::array<Kernel, kOps * kTypes * kDevs> k{};
  std::mutex write;
};
Table& table() {
  static Table t;
  return t;
}
std::size_t slot(Op op, DType t, DeviceKind d) noexcept {
  return (static_cast<std::size_t>(op) * kTypes + dtype_index(t)) * kDevs + static_cast<std::size_t>(d);
}

void register_builtin_cpu();  // asagida
void ensure_builtin() {
  static std::once_flag once;
  std::call_once(once, register_builtin_cpu);
}

std::string_view device_name(DeviceKind d) noexcept { return d == DeviceKind::Host ? "host" : "cihaz"; }

// ---- eleman erisimi (FP32 hesap tipleri) -------------------------------------------------------------------
template <DType T>
struct Elem;
template <> struct Elem<DType::F32> { using S = float; using C = float; static C ld(S v) { return v; } static S st(C v) { return v; } };
template <> struct Elem<DType::F64> { using S = double; using C = double; static C ld(S v) { return v; } static S st(C v) { return v; } };
template <> struct Elem<DType::BF16> {
  using S = std::uint16_t; using C = float;
  static C ld(S v) { return bfloat16_bits_to_float(v); }
  static S st(C v) { return float_to_bfloat16_bits(v); }
};
template <> struct Elem<DType::F16> {
  using S = std::uint16_t; using C = float;
  static C ld(S v) { return half_bits_to_float(v); }
  static S st(C v) { return float_to_half_bits(v); }
};
template <> struct Elem<DType::I32> { using S = std::int32_t; using C = std::int64_t; static C ld(S v) { return v; } static S st(C v) { return static_cast<S>(v); } };
template <> struct Elem<DType::I64> { using S = std::int64_t; using C = std::int64_t; static C ld(S v) { return v; } static S st(C v) { return v; } };

// Uc islenenli satir-ana gezinme: f(out_off, a_off, b_off) eleman cinsinden.
template <typename F>
void for_each3(const Shape& shape, const Strides& so, std::int64_t o0, const Strides& sa, std::int64_t a0,
               const Strides& sb, std::int64_t b0, F&& f) {
  const std::size_t n = shape.numel();
  if (n == 0) return;
  const std::size_t r = shape.rank();
  std::array<std::size_t, kMaxRank> idx{};
  std::int64_t oo = o0, oa = a0, ob = b0;
  const std::size_t inner = shape[r - 1];
  const std::int64_t io = so[r - 1], ia = sa[r - 1], ib = sb[r - 1];
  for (std::size_t done = 0; done < n; done += inner) {
    for (std::size_t k = 0; k < inner; ++k) {
      const auto kk = static_cast<std::int64_t>(k);
      f(oo + kk * io, oa + kk * ia, ob + kk * ib);
    }
    for (int axis = static_cast<int>(r) - 2; axis >= 0; --axis) {
      const auto ax = static_cast<std::size_t>(axis);
      ++idx[ax];
      oo += so[ax];
      oa += sa[ax];
      ob += sb[ax];
      if (idx[ax] < shape[ax]) break;
      const auto len = static_cast<std::int64_t>(shape[ax]);
      oo -= so[ax] * len;
      oa -= sa[ax] * len;
      ob -= sb[ax] * len;
      idx[ax] = 0;
    }
  }
}

template <DType T, typename F>
void binary(const KernelArgs& k, F&& f) {
  using E = Elem<T>;
  using S = typename E::S;
  auto* o = static_cast<S*>(k.out->storage()->data());
  const auto* a = static_cast<const S*>(k.a->storage()->data());
  const auto* b = static_cast<const S*>(k.b->storage()->data());
  const std::size_t n = k.out->numel();
  // sicak yol: hepsi bitisik ve ayni sekil (yayin yok) -> duz dongu (derleyici vektorler)
  if (k.out->is_contiguous() && k.a->is_contiguous() && k.b->is_contiguous() && k.a->shape() == k.out->shape() &&
      k.b->shape() == k.out->shape()) {
    S* op = o + k.out->offset();
    const S* ap = a + k.a->offset();
    const S* bp = b + k.b->offset();
    for (std::size_t i = 0; i < n; ++i) op[i] = E::st(f(E::ld(ap[i]), E::ld(bp[i])));
    return;
  }
  for_each3(k.out->shape(), k.out->strides(), k.out->offset(), k.a->strides(), k.a->offset(), k.b->strides(),
            k.b->offset(), [&](std::int64_t io, std::int64_t ia, std::int64_t ib) { o[io] = E::st(f(E::ld(a[ia]), E::ld(b[ib]))); });
}

template <DType T>
void add_k(const KernelArgs& k) {
  binary<T>(k, [](auto x, auto y) { return x + y; });
}
template <DType T>
void mul_k(const KernelArgs& k) {
  binary<T>(k, [](auto x, auto y) { return x * y; });
}

template <DType T>
void sum_k(const KernelArgs& k) {
  using E = Elem<T>;
  using S = typename E::S;
  const auto* a = static_cast<const S*>(k.a->storage()->data());
  using Acc = std::conditional_t<std::is_integral_v<typename E::C>, std::int64_t, double>;
  Acc acc = 0;
  const Strides z{};
  for_each3(k.a->shape(), z, 0, k.a->strides(), k.a->offset(), z, 0,
            [&](std::int64_t, std::int64_t ia, std::int64_t) { acc += static_cast<Acc>(E::ld(a[ia])); });
  if constexpr (std::is_integral_v<Acc>) {
    *static_cast<std::int64_t*>(k.out->raw_data()) = acc;
  } else {
    *static_cast<S*>(k.out->raw_data()) = E::st(static_cast<typename E::C>(acc));
  }
}

// Host kopya / doldurma: Tensor::copy_from'un genel yolu (her dtype) zaten host'ta calisir; dagitici girisi
// ayni sonuca gitsin diye copy_ cekirdegi copy_from'u cagirir.
void copy_host(const KernelArgs& k) { k.out->copy_from(*k.a); }

void fill_host(const KernelArgs& k) {
  const Tensor one = Tensor::from_values(Shape{1}, std::span<const double>(&k.scalar, 1), k.out->dtype());
  // tek elemani tum sekle yay: adim 0 gorunum
  k.out->copy_from(one.expand(k.out->shape()));
}

// ---- matmul --------------------------------------------------------------------------------------------------
// 2B gorunumu gemm duzenine cevir: ic adim 1 -> satir-ana (ld = adim0); dis adim 1 -> devrik (ld = adim1).
bool gemm_layout(const Tensor& t, bool& trans, int& ld) {
  const std::int64_t s0 = t.strides()[0], s1 = t.strides()[1];
  if (s1 == 1 && s0 >= static_cast<std::int64_t>(t.shape()[1])) {
    trans = false;
    ld = static_cast<int>(s0);
    return true;
  }
  if (s0 == 1 && s1 >= static_cast<std::int64_t>(t.shape()[0])) {
    trans = true;
    ld = static_cast<int>(s1);
    return true;
  }
  return false;
}

void matmul_gemm(const KernelArgs& k) {
  const Tensor& A0 = *k.a;
  const Tensor& B0 = *k.b;
  const DType dt = A0.dtype();
  const int M = static_cast<int>(A0.shape()[0]), K = static_cast<int>(A0.shape()[1]), N = static_cast<int>(B0.shape()[1]);
  // A FP32 olmali (cpu::gemm sozlesmesi): F32 disinda bir kez FP32 kopya
  Tensor A = dt == DType::F32 ? A0.shallow() : A0.to(DType::F32);
  bool at = false, bt = false;
  int lda = 0, ldb = 0;
  if (!gemm_layout(A, at, lda)) {
    A = A.contiguous();
    gemm_layout(A, at, lda);
  }
  Tensor B = B0.shallow();
  if (!gemm_layout(B, bt, ldb)) {
    B = B.contiguous();
    gemm_layout(B, bt, ldb);
  }
  cpu::GemmArgs g;
  g.M = M;
  g.N = N;
  g.K = K;
  g.A = static_cast<const float*>(A.raw_data());
  g.lda = lda;
  g.a_trans = at;
  g.B = B.raw_data();
  g.ldb = ldb;
  g.b_trans = bt;
  g.btype = dt == DType::F32 ? cpu::BType::F32 : dt == DType::BF16 ? cpu::BType::BF16 : cpu::BType::F16;
  if (dt == DType::F32) {
    g.C = static_cast<float*>(k.out->raw_data());
    g.ldc = N;
    cpu::gemm(g);
    return;
  }
  Tensor C = Tensor::empty(k.out->shape(), DType::F32);
  g.C = C.data();
  g.ldc = N;
  cpu::gemm(g);
  k.out->copy_from(C);  // FP32 -> BF16 / F16 RNE
}

void matmul_f64(const KernelArgs& k) {
  const Tensor& A = *k.a;
  const Tensor& B = *k.b;
  const std::size_t M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
  const auto* a = static_cast<const double*>(A.storage()->data());
  const auto* b = static_cast<const double*>(B.storage()->data());
  auto* c = static_cast<double*>(k.out->raw_data());
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      double acc = 0;
      for (std::size_t p = 0; p < K; ++p)
        acc += a[A.offset() + static_cast<std::int64_t>(i) * A.strides()[0] + static_cast<std::int64_t>(p) * A.strides()[1]] *
               b[B.offset() + static_cast<std::int64_t>(p) * B.strides()[0] + static_cast<std::int64_t>(j) * B.strides()[1]];
      c[i * N + j] = acc;
    }
}

void register_builtin_cpu() {
  constexpr auto H = DeviceKind::Host;
  for (const DType t : kAllDTypes) {
    register_kernel(Op::Copy, t, H, copy_host);
    register_kernel(Op::Fill, t, H, fill_host);
  }
  register_kernel(Op::Add, DType::F32, H, add_k<DType::F32>);
  register_kernel(Op::Add, DType::F64, H, add_k<DType::F64>);
  register_kernel(Op::Add, DType::BF16, H, add_k<DType::BF16>);
  register_kernel(Op::Add, DType::F16, H, add_k<DType::F16>);
  register_kernel(Op::Add, DType::I32, H, add_k<DType::I32>);
  register_kernel(Op::Add, DType::I64, H, add_k<DType::I64>);
  register_kernel(Op::Mul, DType::F32, H, mul_k<DType::F32>);
  register_kernel(Op::Mul, DType::F64, H, mul_k<DType::F64>);
  register_kernel(Op::Mul, DType::BF16, H, mul_k<DType::BF16>);
  register_kernel(Op::Mul, DType::F16, H, mul_k<DType::F16>);
  register_kernel(Op::Mul, DType::I32, H, mul_k<DType::I32>);
  register_kernel(Op::Mul, DType::I64, H, mul_k<DType::I64>);
  register_kernel(Op::Sum, DType::F32, H, sum_k<DType::F32>);
  register_kernel(Op::Sum, DType::F64, H, sum_k<DType::F64>);
  register_kernel(Op::Sum, DType::BF16, H, sum_k<DType::BF16>);
  register_kernel(Op::Sum, DType::F16, H, sum_k<DType::F16>);
  register_kernel(Op::Sum, DType::I32, H, sum_k<DType::I32>);
  register_kernel(Op::Sum, DType::I64, H, sum_k<DType::I64>);
  register_kernel(Op::Matmul, DType::F32, H, matmul_gemm);
  register_kernel(Op::Matmul, DType::BF16, H, matmul_gemm);
  register_kernel(Op::Matmul, DType::F16, H, matmul_gemm);
  register_kernel(Op::Matmul, DType::F64, H, matmul_f64);
}

void require_same(const Tensor& a, const Tensor& b, Op op) {
  if (a.dtype() != b.dtype())
    throw TensorError(std::string("Nexus: ") + std::string(op_name(op)) + " girdileri ayni dtype olmali (" +
                      std::string(to_string(a.dtype())) + " vs " + std::string(to_string(b.dtype())) +
                      "); otomatik yukseltme yok, Tensor::to kullanin");
  if (device_kind(a.space()) != device_kind(b.space()) || a.device() != b.device())
    throw TensorError(std::string("Nexus: ") + std::string(op_name(op)) + " girdileri ayni cihazda olmali");
}

void binary_out(Op op, Tensor& out, const Tensor& a, const Tensor& b) {
  require_same(a, b, op);
  const Shape s = broadcast_shapes(a.shape(), b.shape());
  if (out.shape() != s || out.dtype() != a.dtype()) throw TensorError("Nexus: cikti sekli / dtype'i uyusmuyor");
  const Kernel kern = find_kernel(op, a.dtype(), device_kind(a.space()));
  const Tensor ea = a.shape() == s ? a.shallow() : a.expand(s);
  const Tensor eb = b.shape() == s ? b.shallow() : b.expand(s);
  KernelArgs args;
  args.out = &out;
  args.a = &ea;
  args.b = &eb;
  kern(args);
  out.bump_version();
}

Tensor binary(Op op, const Tensor& a, const Tensor& b) {
  require_same(a, b, op);
  Tensor out = Tensor::empty(broadcast_shapes(a.shape(), b.shape()), a.dtype(), a.space(), a.device());
  binary_out(op, out, a, b);
  return out;
}

}  // namespace

std::string_view op_name(Op op) noexcept {
  switch (op) {
    case Op::Copy: return "copy";
    case Op::Fill: return "fill";
    case Op::Add: return "add";
    case Op::Mul: return "mul";
    case Op::Matmul: return "matmul";
    case Op::Sum: return "sum";
    case Op::Count: break;
  }
  return "?";
}

void register_kernel(Op op, DType dtype, DeviceKind device, Kernel kernel, bool replace) {
  if (op == Op::Count || device == DeviceKind::Count || kernel == nullptr) throw TensorError("Nexus: gecersiz cekirdek kaydi");
  Table& t = table();
  std::lock_guard<std::mutex> lock(t.write);
  Kernel& k = t.k[slot(op, dtype, device)];
  if (k != nullptr && !replace)
    throw TensorError(std::string("Nexus: ") + std::string(op_name(op)) + " " + std::string(to_string(dtype)) + " " +
                      std::string(device_name(device)) + " cekirdegi zaten kayitli");
  k = kernel;
}

bool has_kernel(Op op, DType dtype, DeviceKind device) {
  ensure_builtin();
  return table().k[slot(op, dtype, device)] != nullptr;
}

Kernel find_kernel(Op op, DType dtype, DeviceKind device) {
  ensure_builtin();
  const Kernel k = table().k[slot(op, dtype, device)];
  if (k == nullptr)
    throw TensorError(std::string("Nexus: ") + std::string(op_name(op)) + " " + std::string(to_string(dtype)) + " " +
                      std::string(device_name(device)) + " icin cekirdek yok" +
                      (device == DeviceKind::Device ? " (CUDA: nexus::cuda::register_tensor_kernels() cagrildi mi?)" : ""));
  return k;
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
  const std::size_t r = std::max(a.rank(), b.rank());
  std::vector<std::size_t> d(r);
  for (std::size_t i = 0; i < r; ++i) {
    const std::size_t x = i < r - a.rank() ? 1 : a[i - (r - a.rank())];
    const std::size_t y = i < r - b.rank() ? 1 : b[i - (r - b.rank())];
    if (x != y && x != 1 && y != 1) throw TensorError("Nexus: sekiller yayinla uyusmuyor");
    d[i] = x == 1 ? y : x;
  }
  return Shape(d);
}

Tensor add(const Tensor& a, const Tensor& b) { return binary(Op::Add, a, b); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary(Op::Mul, a, b); }
void add_out(Tensor& out, const Tensor& a, const Tensor& b) { binary_out(Op::Add, out, a, b); }
void mul_out(Tensor& out, const Tensor& a, const Tensor& b) { binary_out(Op::Mul, out, a, b); }

Tensor matmul(const Tensor& a, const Tensor& b) {
  require_same(a, b, Op::Matmul);
  if (a.rank() != 2 || b.rank() != 2) throw TensorError("Nexus: matmul yalniz 2B tensörler ([m,k] x [k,n])");
  if (a.shape()[1] != b.shape()[0]) throw TensorError("Nexus: matmul ic boyutlar uyusmuyor");
  const Kernel kern = find_kernel(Op::Matmul, a.dtype(), device_kind(a.space()));
  Tensor out = Tensor::empty(Shape{a.shape()[0], b.shape()[1]}, a.dtype(), a.space(), a.device());
  if (out.numel() == 0) return out;
  if (a.shape()[1] == 0) {
    fill_(out, 0.0);
    return out;
  }
  KernelArgs args;
  args.out = &out;
  args.a = &a;
  args.b = &b;
  kern(args);
  out.bump_version();
  return out;
}

Tensor sum(const Tensor& a) {
  const DType od = (a.dtype() == DType::I32 || a.dtype() == DType::I64) ? DType::I64 : a.dtype();
  const Kernel kern = find_kernel(Op::Sum, a.dtype(), device_kind(a.space()));
  Tensor out = Tensor::zeros(Shape{1}, od, a.space(), a.device());
  if (a.numel() == 0) return out;
  KernelArgs args;
  args.out = &out;
  args.a = &a;
  kern(args);
  out.bump_version();
  return out;
}

void fill_(Tensor& t, double value) {
  if (t.numel() == 0) return;
  KernelArgs args;
  args.out = &t;
  args.scalar = value;
  find_kernel(Op::Fill, t.dtype(), device_kind(t.space()))(args);
}

void copy_(Tensor& dst, const Tensor& src) {
  const DeviceKind d = device_kind(dst.space()) == DeviceKind::Device || device_kind(src.space()) == DeviceKind::Device
                           ? DeviceKind::Device
                           : DeviceKind::Host;
  KernelArgs args;
  args.out = &dst;
  args.a = &src;
  find_kernel(Op::Copy, dst.dtype(), d)(args);
}

}  // namespace ops
}  // namespace nexus
