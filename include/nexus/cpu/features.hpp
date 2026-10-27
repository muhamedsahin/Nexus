#pragma once

/**
 * @file features.hpp
 * @brief FAZ 18.A: calisma zamani CPU ozellik algilama (CPUID + XGETBV) ve topoloji (fiziksel cekirdek, SMT esleri,
 *        NUMA dugumleri, onbellek boyutlari).
 *
 * Yol secimi: AVX2+FMA yolu bu derlemenin ana yoludur. AVX-512 (F/BW/VL/VNNI/BF16/FP16) ve AMX bayraklari algilanir ve
 * raporlanir; bu depoda AVX-512 / AMX cekirdegi YOKTUR (FAZ 18 kapsami: gelistirme makinesi Zen 3, calistirilamayan
 * kod yazilmadi) — isa() hangi yolun kullanildigini acikca soyler. OS destegi: AVX icin XCR0 bit 1-2, AVX-512 icin
 * 5-7, AMX icin 17-18 (XGETBV); islemci destekleyip OS etkinlestirmediyse bayrak false.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstddef>
#include <string>
#include <vector>

namespace nexus {
namespace cpu {

struct Features {
  std::string vendor, brand;
  bool sse42 = false, avx = false, avx2 = false, fma = false, f16c = false, bmi2 = false;
  bool avx512f = false, avx512bw = false, avx512vl = false, avx512vnni = false, avx512bf16 = false, avx512fp16 = false;
  bool avx_vnni = false;
  bool amx_tile = false, amx_bf16 = false, amx_int8 = false;
  int logical = 1, physical = 1, numa_nodes = 1;
  std::size_t l1d = 0, l2 = 0, l3 = 0;  // bayt (cekirdek basina L1/L2, paylasilan L3)
  /// mantiksal islemci -> fiziksel cekirdek kimligi; -> NUMA dugumu
  std::vector<int> core_of, node_of;
};

[[nodiscard]] const Features& features();
/// Kullanilan cekirdek yolu: "avx2-fma" (bu derleme). AVX2 yoksa acik hata verilir (skaler yedek yok).
[[nodiscard]] const char* isa();
[[nodiscard]] std::string describe();

}  // namespace cpu
}  // namespace nexus
