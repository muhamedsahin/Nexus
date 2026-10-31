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

enum class Op : std::uint8_t { Copy, Fill, Add, Mul, Matmul, Sum, Count };
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
/// Yerinde doldurma (her dtype; tamsayi tiplerinde tam sayi olmayan deger acik hata).
void fill_(Tensor& t, double value);
/// Yerinde kopya: dst.dtype'a donusum + adimlar (ayni sekil). Host <-> host Tensor::copy_from'da; bu giris
/// dagiticiya gider (cihaz cekirdekleri burada kayitli olmali).
void copy_(Tensor& dst, const Tensor& src);

}  // namespace ops
}  // namespace nexus
