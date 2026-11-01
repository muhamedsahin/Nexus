#pragma once

/**
 * @file tensor_ops.hpp
 * @brief Dalga 2: Tensor hesap islemleri + dagitici (islem x dtype x cihaz -> cekirdek).
 *
 * Tablo duz dizidir (arama O(1), kilit yok). CPU cekirdekleri ilk kullanimda tembel kaydedilir (statik kutuphanede
 * kendini kaydeden statik nesneler baglayici tarafindan atilabilir). Cihaz kutuphaneleri (orn. nexus_cuda)
 * `register_kernel` ile ekler. Kayitsiz birlesim ACIK HATA verir: "Nexus: <islem> <dtype> <cihaz> icin cekirdek yok"
 * (sessiz host'a / FP32'ye kacis yok).
 *
 * Kurallar:
 *   - Ikili islemlerde iki girdi AYNI dtype olmali (otomatik yukseltme yok; `Tensor::to` ile acikca).
 *   - Yayin (broadcast) NumPy kurali: sondan hizali, boyu 1 olan eksen genisler.
 *   - BF16 / F16 / FP8 eleman islemleri FP32'de hesaplanir, sonuc RNE ile yuvarlanir.
 *   - sum: kayanlar FP64 biriktirir, sonuc girdi dtype'i; tamsayilar I64. Sekil {1} (Shape rank 0 = 0 eleman).
 *   - matmul: 2B [m,k] x [k,n]; F32 / BF16 / F16 nexus::cpu::gemm (AVX2) uzerinden, F64 dogrudan. Devrik
 *     gorunumler (transpose) kopyasiz gemm devrik bayragina cevrilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor.hpp"

#include <cstdint>
#include <string_view>

namespace nexus {
namespace ops {

enum class Op : std::uint8_t {
  Copy, Fill, Add, Mul, Matmul, Sum,
  // Dalga 2 (autograd Tensor yolu)
  Sub, Div, Unary, UnaryGrad, RowSoftmax, RowSoftmaxGrad, CrossEntropy, CrossEntropyGrad, ReduceRows, GatherRows,
  EmbeddingGrad, SqNorm, AdamW,
  Count
};

/// Unary / UnaryGrad islev kimligi (FAZ 2 autograd formulleriyle ayni: GELU erf, sigmoid 1/(1+e^-x), ...).
enum class UnaryFn : std::uint8_t { Neg, AddScalar, MulScalar, Pow, Relu, Sigmoid, Tanh, Exp, Log, Gelu, Silu };
enum class DeviceKind : std::uint8_t { Host, Device, Count };

[[nodiscard]] std::string_view op_name(Op op) noexcept;
[[nodiscard]] constexpr DeviceKind device_kind(MemorySpace space) noexcept {
  return space == MemorySpace::Device ? DeviceKind::Device : DeviceKind::Host;
}

/// Cekirdek girdileri. out her zaman ayrilmis (Copy / Fill'de hedef), a / b islem girdileri (yoksa nullptr).
struct KernelArgs {
  Tensor* out = nullptr;
  const Tensor* a = nullptr;
  const Tensor* b = nullptr;
  double scalar = 0.0;
  // Dalga 2: cok ciktili / parametreli cekirdekler (AdamW: out = param, out2 = m, out3 = v; p = katsayilar)
  Tensor* out2 = nullptr;
  Tensor* out3 = nullptr;
  const Tensor* c = nullptr;
  int fn = 0;
  double p[8] = {};
};
using Kernel = void (*)(const KernelArgs&);

/// Ayni anahtara ikinci kayit acik hata (cekirdek degistirmek icin `replace = true`).
void register_kernel(Op op, DType dtype, DeviceKind device, Kernel kernel, bool replace = false);
[[nodiscard]] bool has_kernel(Op op, DType dtype, DeviceKind device);
/// Kayitsizsa TensorError.
[[nodiscard]] Kernel find_kernel(Op op, DType dtype, DeviceKind device);

// ---- islemler ----------------------------------------------------------------------------------------
[[nodiscard]] Shape broadcast_shapes(const Shape& a, const Shape& b);
[[nodiscard]] Tensor add(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor mul(const Tensor& a, const Tensor& b);
/// out = a + b (yerinde degil; out onceden ayrilmis, sekli yayin sekli, dtype ayni). Ayirmasiz sicak yol.
void add_out(Tensor& out, const Tensor& a, const Tensor& b);
void mul_out(Tensor& out, const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor sum(const Tensor& a);
// ---- Dalga 2: autograd Tensor yolu cekirdekleri (kayan dtype'lar: F32 F64 BF16 F16; hesap FP32 / FP64) --------
[[nodiscard]] Tensor sub(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor div(const Tensor& a, const Tensor& b);
/// y = f(x[, scalar]) eleman eleman.
[[nodiscard]] Tensor unary(UnaryFn fn, const Tensor& x, double scalar = 0.0);
/// dx = g * f'(...). `saved`: Pow / Relu / Log / Gelu / Silu icin girdi x; Sigmoid / Tanh / Exp icin cikti y.
/// Neg / AddScalar / MulScalar icin saved kullanilmaz (g'yi verin).
[[nodiscard]] Tensor unary_grad(UnaryFn fn, const Tensor& g, const Tensor& saved, double scalar = 0.0);
/// 2B [N, C], son eksen: softmax (log = false) ya da log_softmax (log = true).
[[nodiscard]] Tensor row_softmax(const Tensor& x, bool log);
/// softmax: dx = y (g - sum(g y)); log_softmax: dx = g - exp(y) sum(g). y ileri ciktisi.
[[nodiscard]] Tensor row_softmax_grad(const Tensor& g, const Tensor& y, bool log);
/// logits [N, C], targets I64 [N] (ayni cihaz) -> ortalama kayip {1}: F32 / BF16 / F16 logitte F32 (FAZ 2 sirasi),
/// F64 logitte F64 hesap ve kayip. Olasilik alt siniri 1e-12.
[[nodiscard]] Tensor cross_entropy(const Tensor& logits, const Tensor& targets);
/// dlogits = (softmax - onehot) * g / N; g {1} kayip dtype'inda (ayni cihaz).
[[nodiscard]] Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets, const Tensor& g);
/// [.., C] -> [C]: son eksen disindaki tum eksenler uzerinde toplam (FP64 biriktirme, belirlenimci).
[[nodiscard]] Tensor reduce_rows(const Tensor& x);
/// weight [V, D], idx I64 [L] -> [L, D]. Aralik disi indeks acik hata.
[[nodiscard]] Tensor gather_rows(const Tensor& weight, const Tensor& idx);
/// Belirlenimci embedding geri gecisi. plan I64: [perm (L) | seg_row (U) | seg_off (U+1)] (indeks sirali segmentler,
/// segment icinde ozgun sira). padding satiri sifir kalir. Cikti [V, D], g ile ayni dtype.
[[nodiscard]] Tensor embedding_grad(const Tensor& g, const Tensor& plan, std::int64_t vocab, std::int64_t count,
                                    std::int64_t segments, std::int64_t padding_idx);
/// sum(x^2) F64 {1}, FP64 biriktirme (gradyan kirpma normu).
[[nodiscard]] Tensor sq_norm(const Tensor& x);
/// Yerinde AdamW (fused::adamw_multi_tensor ile ayni islem sirasi): param (F32 / BF16 / F16), grad (param dtype'i),
/// m / v (F32). coeffs: b1 ob1 b2 ob2 inv1 inv2 lr lr_wd eps clip (bkz. cpu::adam_coeffs); decay: lr_wd uygula.
struct AdamWCoeffs {
  double b1, ob1, b2, ob2, inv1, inv2, lr, lr_wd, eps, clip;
  bool decay;
};
void adamw_(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, const AdamWCoeffs& c);

/// Yerinde doldurma (her dtype; tamsayi tiplerinde tam sayi olmayan deger acik hata).
void fill_(Tensor& t, double value);
/// Yerinde kopya: dst.dtype'a donusum + adimlar (ayni sekil). Host <-> host Tensor::copy_from'da; bu giris
/// dagiticiya gider (cihaz cekirdekleri burada kayitli olmali).
void copy_(Tensor& dst, const Tensor& src);

}  // namespace ops
}  // namespace nexus
