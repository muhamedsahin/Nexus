/**
 * @file tensor_ops_autograd.cpp
 * @brief Dalga 2: autograd Tensor yolu icin CPU cekirdekleri + API girisleri (bkz. tensor_ops.hpp).
 *
 * Formuller FAZ 2 autograd (src/autograd/ops.cpp) ile ayni: FP32'de hesap (F64'te FP64), ayni sira; BF16 / F16
 * FP32'de hesaplanip RNE ile yuvarlanir. AdamW fused::adamw_multi_tensor ile ayni islem sirasi (F32 bit bit).
 * API girisleri girdileri bitisik yapar (cekirdekler bitisik varsayar); cihaz cekirdekleri nexus_cuda'da.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/half.hpp"
#include "nexus/core/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>

namespace nexus {
namespace ops {
namespace {

template <DType T> struct E;
template <> struct E<DType::F32> { using S = float; using C = float; static C ld(S v) { return v; } static S st(C v) { return v; } };
template <> struct E<DType::F64> { using S = double; using C = double; static C ld(S v) { return v; } static S st(C v) { return v; } };
template <> struct E<DType::BF16> {
  using S = std::uint16_t; using C = float;
  static C ld(S v) { return bfloat16_bits_to_float(v); }
  static S st(C v) { return float_to_bfloat16_bits(v); }
};
template <> struct E<DType::F16> {
  using S = std::uint16_t; using C = float;
  static C ld(S v) { return half_bits_to_float(v); }
  static S st(C v) { return float_to_half_bits(v); }
};

template <typename C>
C unary_f(UnaryFn fn, C x, C s) {
  constexpr C kInvSqrt2 = static_cast<C>(0.7071067811865475);
  switch (fn) {
    case UnaryFn::Neg: return -x;
    case UnaryFn::AddScalar: return x + s;
    case UnaryFn::MulScalar: return x * s;
    case UnaryFn::Pow: return std::pow(x, s);
    case UnaryFn::Relu: return x > C(0) ? x : C(0);
    case UnaryFn::Sigmoid: return C(1) / (C(1) + std::exp(-x));
    case UnaryFn::Tanh: return std::tanh(x);
    case UnaryFn::Exp: return std::exp(x);
    case UnaryFn::Log: return std::log(x);
    case UnaryFn::Gelu: return C(0.5) * x * (C(1) + std::erf(x * kInvSqrt2));
    case UnaryFn::Silu: return x / (C(1) + std::exp(-x));
  }
  return x;
}

// g: gelen gradyan, v: saved (x ya da y)
template <typename C>
C unary_grad_f(UnaryFn fn, C g, C v, C s) {
  constexpr C kInvSqrt2 = static_cast<C>(0.7071067811865475);
  constexpr C kInvSqrt2Pi = static_cast<C>(0.3989422804014327);
  switch (fn) {
    case UnaryFn::Neg: return -g;
    case UnaryFn::AddScalar: return g;
    case UnaryFn::MulScalar: return g * s;
    case UnaryFn::Pow: return g * s * std::pow(v, s - C(1));
    case UnaryFn::Relu: return v > C(0) ? g : C(0);
    case UnaryFn::Sigmoid: return g * v * (C(1) - v);
    case UnaryFn::Tanh: return g * (C(1) - v * v);
    case UnaryFn::Exp: return g * v;
    case UnaryFn::Log: return g / v;
    case UnaryFn::Gelu: {
      const C cdf = C(0.5) * (C(1) + std::erf(v * kInvSqrt2));
      const C gaussian = std::exp(C(-0.5) * v * v);
      return g * (cdf + v * kInvSqrt2Pi * gaussian);
    }
    case UnaryFn::Silu: {
      const C sig = C(1) / (C(1) + std::exp(-v));
      return g * sig * (C(1) + v * (C(1) - sig));
    }
  }
  return g;
}

template <DType T>
typename E<T>::S* ptr(const Tensor& t) {
  return static_cast<typename E<T>::S*>(t.raw_data());
}

template <DType T>
void unary_k(const KernelArgs& k) {
  using C = typename E<T>::C;
  const auto* x = ptr<T>(*k.a);
  auto* y = ptr<T>(*k.out);
  const auto fn = static_cast<UnaryFn>(k.fn);
  const C s = static_cast<C>(k.scalar);
  for (std::size_t i = 0, n = k.out->numel(); i < n; ++i) y[i] = E<T>::st(unary_f<C>(fn, E<T>::ld(x[i]), s));
}

template <DType T>
void unary_grad_k(const KernelArgs& k) {
  using C = typename E<T>::C;
  const auto* g = ptr<T>(*k.a);
  const auto* v = ptr<T>(*k.b);
  auto* d = ptr<T>(*k.out);
  const auto fn = static_cast<UnaryFn>(k.fn);
  const C s = static_cast<C>(k.scalar);
  for (std::size_t i = 0, n = k.out->numel(); i < n; ++i) d[i] = E<T>::st(unary_grad_f<C>(fn, E<T>::ld(g[i]), E<T>::ld(v[i]), s));
}

template <DType T>
void row_softmax_k(const KernelArgs& k) {
  using C = typename E<T>::C;
  const auto* x = ptr<T>(*k.a);
  auto* y = ptr<T>(*k.out);
  const std::size_t n = k.a->shape()[0], c = k.a->shape()[1];
  std::vector<C> e(c);
  for (std::size_t i = 0; i < n; ++i) {
    const auto* row = x + i * c;
    C maxv = E<T>::ld(row[0]);
    for (std::size_t j = 1; j < c; ++j) maxv = std::fmax(maxv, E<T>::ld(row[j]));
    if (k.fn == 0) {  // softmax: FAZ 2 sirasi (FP32 toplam)
      C total = 0;
      for (std::size_t j = 0; j < c; ++j) {
        e[j] = std::exp(E<T>::ld(row[j]) - maxv);
        total += e[j];
      }
      for (std::size_t j = 0; j < c; ++j) y[i * c + j] = E<T>::st(e[j] / total);
    } else {  // log_softmax: FP64 toplam (FAZ 2 gibi)
      double total = 0.0;
      for (std::size_t j = 0; j < c; ++j) total += std::exp(static_cast<double>(E<T>::ld(row[j]) - maxv));
      const C lse = maxv + static_cast<C>(std::log(total));
      for (std::size_t j = 0; j < c; ++j) y[i * c + j] = E<T>::st(E<T>::ld(row[j]) - lse);
    }
  }
}

template <DType T>
void row_softmax_grad_k(const KernelArgs& k) {
  using C = typename E<T>::C;
  const auto* g = ptr<T>(*k.a);
  const auto* y = ptr<T>(*k.b);
  auto* d = ptr<T>(*k.out);
  const std::size_t n = k.a->shape()[0], c = k.a->shape()[1];
  for (std::size_t i = 0; i < n; ++i) {
    C acc = 0;
    if (k.fn == 0) {
      for (std::size_t j = 0; j < c; ++j) acc += E<T>::ld(g[i * c + j]) * E<T>::ld(y[i * c + j]);
      for (std::size_t j = 0; j < c; ++j) d[i * c + j] = E<T>::st(E<T>::ld(y[i * c + j]) * (E<T>::ld(g[i * c + j]) - acc));
    } else {
      for (std::size_t j = 0; j < c; ++j) acc += E<T>::ld(g[i * c + j]);
      for (std::size_t j = 0; j < c; ++j) d[i * c + j] = E<T>::st(E<T>::ld(g[i * c + j]) - std::exp(E<T>::ld(y[i * c + j])) * acc);
    }
  }
}

// FAZ 2 cross_entropy ile ayni sira (olasiliklar, sirali toplam, alt sinir 1e-12). Hesap tipi: F32 / BF16 / F16 icin
// float (FAZ 2 ile bit bit), F64 icin double (F64 hassasiyeti sessizce dusmesin).
template <DType T>
using CeT = std::conditional_t<T == DType::F64, double, float>;

template <DType T>
void softmax_row_probs(const typename E<T>::S* row, std::size_t c, std::vector<CeT<T>>& probs) {
  using A = CeT<T>;
  A maxv = static_cast<A>(E<T>::ld(row[0]));
  for (std::size_t j = 1; j < c; ++j) maxv = std::fmax(maxv, static_cast<A>(E<T>::ld(row[j])));
  A total = 0;
  for (std::size_t j = 0; j < c; ++j) {
    probs[j] = std::exp(static_cast<A>(E<T>::ld(row[j])) - maxv);
    total += probs[j];
  }
  for (std::size_t j = 0; j < c; ++j) probs[j] /= total;
}

template <DType T>
void cross_entropy_k(const KernelArgs& k) {
  const auto* x = ptr<T>(*k.a);
  const auto* t = static_cast<const std::int64_t*>(k.b->raw_data());
  const std::size_t n = k.a->shape()[0], c = k.a->shape()[1];
  using A = CeT<T>;
  std::vector<A> probs(c);
  A loss = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (t[i] < 0 || static_cast<std::size_t>(t[i]) >= c) throw TensorError("Nexus: cross_entropy hedefi sinif araligi disinda");
    softmax_row_probs<T>(x + i * c, c, probs);
    loss += -std::log(std::fmax(probs[static_cast<std::size_t>(t[i])], static_cast<A>(1.0e-12)));
  }
  *static_cast<A*>(k.out->raw_data()) = loss / static_cast<A>(n);
}

template <DType T>
void cross_entropy_grad_k(const KernelArgs& k) {
  const auto* x = ptr<T>(*k.a);
  const auto* t = static_cast<const std::int64_t*>(k.b->raw_data());
  auto* d = ptr<T>(*k.out);
  const std::size_t n = k.a->shape()[0], c = k.a->shape()[1];
  using A = CeT<T>;
  const A scale = *static_cast<const A*>(k.c->raw_data()) / static_cast<A>(n);
  std::vector<A> probs(c);
  for (std::size_t i = 0; i < n; ++i) {
    softmax_row_probs<T>(x + i * c, c, probs);
    for (std::size_t j = 0; j < c; ++j) {
      const A onehot = static_cast<std::int64_t>(j) == t[i] ? A(1) : A(0);
      d[i * c + j] = E<T>::st(static_cast<typename E<T>::C>((probs[j] - onehot) * scale));
    }
  }
}

template <DType T>
void reduce_rows_k(const KernelArgs& k) {
  const auto* x = ptr<T>(*k.a);
  auto* y = ptr<T>(*k.out);
  const std::size_t c = k.out->numel(), rows = k.a->numel() / std::max<std::size_t>(c, 1);
  for (std::size_t j = 0; j < c; ++j) {
    double acc = 0;
    for (std::size_t i = 0; i < rows; ++i) acc += static_cast<double>(E<T>::ld(x[i * c + j]));
    y[j] = E<T>::st(static_cast<typename E<T>::C>(acc));
  }
}

template <DType T>
void gather_rows_k(const KernelArgs& k) {
  const auto* w = ptr<T>(*k.a);
  const auto* idx = static_cast<const std::int64_t*>(k.b->raw_data());
  auto* y = ptr<T>(*k.out);
  const std::size_t L = k.b->numel(), D = k.a->shape()[1];
  const auto V = static_cast<std::int64_t>(k.a->shape()[0]);
  for (std::size_t i = 0; i < L; ++i) {
    if (idx[i] < 0 || idx[i] >= V) throw TensorError("Nexus: gather_rows / embedding indeksi sozluk disinda");
    std::copy(w + static_cast<std::size_t>(idx[i]) * D, w + static_cast<std::size_t>(idx[i] + 1) * D, y + i * D);
  }
}

template <DType T>
void embedding_grad_k(const KernelArgs& k) {
  using C = typename E<T>::C;
  const auto* g = ptr<T>(*k.a);
  const auto* plan = static_cast<const std::int64_t*>(k.b->raw_data());
  auto* d = ptr<T>(*k.out);
  const auto L = static_cast<std::size_t>(k.p[0]), U = static_cast<std::size_t>(k.p[1]);
  const auto pad = static_cast<std::int64_t>(k.p[2]);
  const std::size_t D = k.out->shape()[1];
  const std::int64_t* perm = plan;
  const std::int64_t* seg_row = plan + L;
  const std::int64_t* seg_off = plan + L + U;
  for (std::size_t s = 0; s < U; ++s) {
    if (seg_row[s] == pad) continue;
    for (std::size_t j = 0; j < D; ++j) {
      C acc = 0;  // FAZ 2: FP32, indeks sirasiyla
      for (std::int64_t q = seg_off[s]; q < seg_off[s + 1]; ++q) acc += E<T>::ld(g[static_cast<std::size_t>(perm[q]) * D + j]);
      d[static_cast<std::size_t>(seg_row[s]) * D + j] = E<T>::st(acc);
    }
  }
}

template <DType T>
void sq_norm_k(const KernelArgs& k) {
  const auto* x = ptr<T>(*k.a);
  double acc = 0;
  for (std::size_t i = 0, n = k.a->numel(); i < n; ++i) {
    const double v = static_cast<double>(E<T>::ld(x[i]));
    acc += v * v;
  }
  *static_cast<double*>(k.out->raw_data()) = acc;
}

template <DType T>
void adamw_k(const KernelArgs& k) {
  auto* p = ptr<T>(*k.out);
  const auto* g = ptr<T>(*k.a);
  auto* m = static_cast<float*>(k.out2->raw_data());
  auto* v = static_cast<float*>(k.out3->raw_data());
  const auto f = [&](int i) { return static_cast<float>(k.p[i]); };
  // p[]: 0 b1, 1 ob1, 2 b2, 3 ob2, 4 inv1, 5 inv2, 6 lr, 7 lr_wd; scalar: eps; fn: decay; c: clip {1} F32 host
  const float b1 = f(0), ob1 = f(1), b2 = f(2), ob2 = f(3), inv1 = f(4), inv2 = f(5), lr = f(6), lr_wd = f(7);
  const auto eps = static_cast<float>(k.scalar);
  const float clip = *static_cast<const float*>(k.c->raw_data());
  const bool decay = k.fn != 0;
  for (std::size_t i = 0, n = k.out->numel(); i < n; ++i) {
    float pp = static_cast<float>(E<T>::ld(p[i]));
    const float gg = static_cast<float>(E<T>::ld(g[i])) * clip;
    if (decay) pp -= lr_wd * pp;
    const float mt = b1 * m[i] + ob1 * gg;
    const float vt = b2 * v[i] + ob2 * gg * gg;
    m[i] = mt;
    v[i] = vt;
    p[i] = E<T>::st(static_cast<typename E<T>::C>(pp - lr * (mt * inv1) / (std::sqrt(vt * inv2) + eps)));
  }
}

template <DType T>
void register_float() {
  constexpr auto H = DeviceKind::Host;
  register_kernel(Op::Unary, T, H, unary_k<T>);
  register_kernel(Op::UnaryGrad, T, H, unary_grad_k<T>);
  register_kernel(Op::RowSoftmax, T, H, row_softmax_k<T>);
  register_kernel(Op::RowSoftmaxGrad, T, H, row_softmax_grad_k<T>);
  register_kernel(Op::CrossEntropy, T, H, cross_entropy_k<T>);
  register_kernel(Op::CrossEntropyGrad, T, H, cross_entropy_grad_k<T>);
  register_kernel(Op::ReduceRows, T, H, reduce_rows_k<T>);
  register_kernel(Op::GatherRows, T, H, gather_rows_k<T>);
  register_kernel(Op::EmbeddingGrad, T, H, embedding_grad_k<T>);
  register_kernel(Op::SqNorm, T, H, sq_norm_k<T>);
  if constexpr (T != DType::F64) register_kernel(Op::AdamW, T, H, adamw_k<T>);
}

bool floating(DType t) { return t == DType::F32 || t == DType::F64 || t == DType::BF16 || t == DType::F16; }

void require_float(const Tensor& t, const char* op) {
  if (!floating(t.dtype()))
    throw TensorError(std::string("Nexus: ") + op + " yalniz kayan dtype (F32 F64 BF16 F16), verilen " +
                      std::string(to_string(t.dtype())));
}

void require_same_place(const Tensor& a, const Tensor& b, const char* op) {
  if (device_kind(a.space()) != device_kind(b.space()) || a.device() != b.device())
    throw TensorError(std::string("Nexus: ") + op + " girdileri ayni cihazda olmali");
}

Tensor like(const Tensor& t, Shape shape, DType dtype) { return Tensor::empty(std::move(shape), dtype, t.space(), t.device()); }

void run(Op op, KernelArgs& args, DType dtype, const Tensor& where) {
  find_kernel(op, dtype, device_kind(where.space()))(args);
  if (args.out) args.out->bump_version();
}

void require_2d(const Tensor& t, const char* op) {
  if (t.rank() != 2) throw TensorError(std::string("Nexus: ") + op + " 2B tensör ister");
}

}  // namespace

namespace detail {
void register_autograd_cpu_kernels() {
  register_float<DType::F32>();
  register_float<DType::F64>();
  register_float<DType::BF16>();
  register_float<DType::F16>();
}
}  // namespace detail

Tensor unary(UnaryFn fn, const Tensor& x, double scalar) {
  require_float(x, "unary");
  const Tensor xc = x.contiguous();
  Tensor out = like(x, x.shape(), x.dtype());
  if (out.numel() == 0) return out;
  KernelArgs a;
  a.out = &out;
  a.a = &xc;
  a.fn = static_cast<int>(fn);
  a.scalar = scalar;
  run(Op::Unary, a, x.dtype(), x);
  return out;
}

Tensor unary_grad(UnaryFn fn, const Tensor& g, const Tensor& saved, double scalar) {
  require_float(g, "unary_grad");
  if (g.dtype() != saved.dtype() || g.shape() != saved.shape()) throw TensorError("Nexus: unary_grad g / saved sekil-dtype uyusmuyor");
  require_same_place(g, saved, "unary_grad");
  const Tensor gc = g.contiguous(), sc = saved.contiguous();
  Tensor out = like(g, g.shape(), g.dtype());
  if (out.numel() == 0) return out;
  KernelArgs a;
  a.out = &out;
  a.a = &gc;
  a.b = &sc;
  a.fn = static_cast<int>(fn);
  a.scalar = scalar;
  run(Op::UnaryGrad, a, g.dtype(), g);
  return out;
}

Tensor row_softmax(const Tensor& x, bool log) {
  require_float(x, "row_softmax");
  require_2d(x, "row_softmax");
  const Tensor xc = x.contiguous();
  Tensor out = like(x, x.shape(), x.dtype());
  if (out.numel() == 0) return out;
  KernelArgs a;
  a.out = &out;
  a.a = &xc;
  a.fn = log ? 1 : 0;
  run(Op::RowSoftmax, a, x.dtype(), x);
  return out;
}

Tensor row_softmax_grad(const Tensor& g, const Tensor& y, bool log) {
  require_float(g, "row_softmax_grad");
  require_2d(g, "row_softmax_grad");
  if (g.dtype() != y.dtype() || g.shape() != y.shape()) throw TensorError("Nexus: row_softmax_grad g / y uyusmuyor");
  require_same_place(g, y, "row_softmax_grad");
  const Tensor gc = g.contiguous(), yc = y.contiguous();
  Tensor out = like(g, g.shape(), g.dtype());
  if (out.numel() == 0) return out;
  KernelArgs a;
  a.out = &out;
  a.a = &gc;
  a.b = &yc;
  a.fn = log ? 1 : 0;
  run(Op::RowSoftmaxGrad, a, g.dtype(), g);
  return out;
}

namespace {
void check_ce(const Tensor& logits, const Tensor& targets) {
  require_float(logits, "cross_entropy");
  require_2d(logits, "cross_entropy");
  if (targets.dtype() != DType::I64 || targets.rank() != 1 || targets.shape()[0] != logits.shape()[0])
    throw TensorError("Nexus: cross_entropy hedefleri I64 [N] olmali");
  require_same_place(logits, targets, "cross_entropy");
  if (logits.shape()[0] == 0) throw TensorError("Nexus: cross_entropy bos yigin");
}
}  // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
  check_ce(logits, targets);
  const Tensor lc = logits.contiguous(), tc = targets.contiguous();
  // kayip F32 (autocast gibi); F64 logitlerde F64 (hassasiyet dusmez)
  Tensor out = like(logits, Shape{1}, logits.dtype() == DType::F64 ? DType::F64 : DType::F32);
  KernelArgs a;
  a.out = &out;
  a.a = &lc;
  a.b = &tc;
  run(Op::CrossEntropy, a, logits.dtype(), logits);
  return out;
}

Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets, const Tensor& g) {
  check_ce(logits, targets);
  const DType loss_dt = logits.dtype() == DType::F64 ? DType::F64 : DType::F32;
  if (g.dtype() != loss_dt || g.numel() != 1) throw TensorError("Nexus: cross_entropy_grad g kayip dtype'inda {1} olmali (F64 logit: F64, digerleri F32)");
  require_same_place(logits, g, "cross_entropy_grad");
  const Tensor lc = logits.contiguous(), tc = targets.contiguous(), gc = g.contiguous();
  Tensor out = like(logits, logits.shape(), logits.dtype());
  KernelArgs a;
  a.out = &out;
  a.a = &lc;
  a.b = &tc;
  a.c = &gc;
  run(Op::CrossEntropyGrad, a, logits.dtype(), logits);
  return out;
}

Tensor reduce_rows(const Tensor& x) {
  require_float(x, "reduce_rows");
  if (x.rank() < 1) throw TensorError("Nexus: reduce_rows en az 1 eksen ister");
  const Tensor xc = x.contiguous();
  Tensor out = like(x, Shape{x.shape()[x.rank() - 1]}, x.dtype());
  if (out.numel() == 0) return out;
  KernelArgs a;
  a.out = &out;
  a.a = &xc;
  run(Op::ReduceRows, a, x.dtype(), x);
  return out;
}

Tensor gather_rows(const Tensor& weight, const Tensor& idx) {
  require_float(weight, "gather_rows");
  require_2d(weight, "gather_rows");
  if (idx.dtype() != DType::I64 || idx.rank() != 1) throw TensorError("Nexus: gather_rows indeksleri I64 [L] olmali");
  require_same_place(weight, idx, "gather_rows");
  const Tensor wc = weight.contiguous(), ic = idx.contiguous();
  Tensor out = like(weight, Shape{idx.shape()[0], weight.shape()[1]}, weight.dtype());
  if (out.numel() == 0) return out;
  KernelArgs a;
  a.out = &out;
  a.a = &wc;
  a.b = &ic;
  run(Op::GatherRows, a, weight.dtype(), weight);
  return out;
}

Tensor embedding_grad(const Tensor& g, const Tensor& plan, std::int64_t vocab, std::int64_t count, std::int64_t segments,
                      std::int64_t padding_idx) {
  require_float(g, "embedding_grad");
  require_2d(g, "embedding_grad");
  if (plan.dtype() != DType::I64 || static_cast<std::int64_t>(plan.numel()) != count + 2 * segments + 1)
    throw TensorError("Nexus: embedding_grad plani I64 [L + 2U + 1] olmali");
  require_same_place(g, plan, "embedding_grad");
  const Tensor gc = g.contiguous(), pc = plan.contiguous();
  Tensor out = Tensor::zeros(Shape{static_cast<std::size_t>(vocab), g.shape()[1]}, g.dtype(), g.space(), g.device());
  if (out.numel() == 0 || count == 0) return out;
  KernelArgs a;
  a.out = &out;
  a.a = &gc;
  a.b = &pc;
  a.p[0] = static_cast<double>(count);
  a.p[1] = static_cast<double>(segments);
  a.p[2] = static_cast<double>(padding_idx);
  run(Op::EmbeddingGrad, a, g.dtype(), g);
  return out;
}

Tensor sq_norm(const Tensor& x) {
  require_float(x, "sq_norm");
  const Tensor xc = x.contiguous();
  Tensor out = Tensor::zeros(Shape{1}, DType::F64, x.space(), x.device());
  if (x.numel() == 0) return out;
  KernelArgs a;
  a.out = &out;
  a.a = &xc;
  run(Op::SqNorm, a, x.dtype(), x);
  return out;
}

void adamw_(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, const AdamWCoeffs& c) {
  if (!param.is_contiguous() || !m.is_contiguous() || !v.is_contiguous())
    throw TensorError("Nexus: adamw_ param / m / v bitisik olmali");
  if (grad.dtype() != param.dtype() || grad.shape() != param.shape() || m.dtype() != DType::F32 || v.dtype() != DType::F32 ||
      m.shape() != param.shape() || v.shape() != param.shape())
    throw TensorError("Nexus: adamw_ sekil / dtype uyusmuyor (grad = param dtype, m / v F32)");
  require_same_place(param, grad, "adamw_");
  require_same_place(param, m, "adamw_");
  require_same_place(param, v, "adamw_");
  if (param.numel() == 0) return;
  const Tensor gc = grad.contiguous();
  // clip: cekirdek tek elemanli F32 tensörden okur (cihazda cihaz bellegi)
  const float clipf = static_cast<float>(c.clip);
  const Tensor clip = Tensor(Shape{1}, std::span<const float>(&clipf, 1)).to(param.space(), param.device());
  KernelArgs a;
  a.out = &param;
  a.a = &gc;
  a.out2 = &m;
  a.out3 = &v;
  a.c = &clip;
  const double ps[8] = {c.b1, c.ob1, c.b2, c.ob2, c.inv1, c.inv2, c.lr, c.lr_wd};
  for (int i = 0; i < 8; ++i) a.p[i] = ps[i];
  a.scalar = c.eps;
  a.fn = c.decay ? 1 : 0;
  run(Op::AdamW, a, param.dtype(), param);
  m.bump_version();
  v.bump_version();
}

}  // namespace ops
}  // namespace nexus
