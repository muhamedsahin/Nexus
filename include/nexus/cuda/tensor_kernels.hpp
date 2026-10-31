#pragma once

/**
 * @file tensor_kernels.hpp
 * @brief Dalga 2: nexus::Tensor dagiticisina CUDA cekirdekleri (ops::DeviceKind::Device).
 *
 * `register_tensor_kernels()` acikca cagrilir (idempotent; statik kutuphanede kendini kaydeden statik nesne
 * baglayici tarafindan atilabilir). Kayitlar:
 *   - copy  : F32 F16 BF16 F64 I32 I64 I8 U8 Bool (hedef dtype). Adimli + dtype donusumlu; host <-> cihaz karisik
 *             (host tarafi bitisik ara tamponla). Kayan -> tamsayi NaN / aralik disi: cihaz bayragi -> acik hata.
 *   - fill  : ayni dtype'lar, adimli hedef.
 *   - add / mul : F32 F16 BF16 F64 I32 I64; yayin (adim 0). BF16 / F16 FP32'de hesap, RNE.
 *   - sum   : F32 F16 BF16 F64 (FP64 biriktirme) ve I32 I64 (I64); belirlenimci iki gecis (blok kismi + sirali).
 *   - matmul: F32 F16 BF16 cuBLASLt (nexus::cuda::gemm; FP32 biriktirme, F32'de float32_matmul_precision
 *             ayari gecerli); devrik gorunum kopyasiz trans bayragina cevrilir.
 * FP8 cihaz donusumu kayitli DEGIL (acik hata).
 *
 * Akis: tum cekirdekler ESKI VARSAYILAN AKISTA (stream 0) calisir; boylece Tensor'un `copy_bytes` hizli yolu
 * (cudaMemcpy) ile dogal sirali. Nexus akislari cudaStreamNonBlocking oldugundan baska akista calisan kodla
 * (orn. GptModel) paylasilan tamponlarda cagiran acik senkronizasyon yapmalidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

namespace nexus {
namespace cuda {

/// Idempotent; ilk cagri kaydeder.
void register_tensor_kernels();

}  // namespace cuda
}  // namespace nexus
