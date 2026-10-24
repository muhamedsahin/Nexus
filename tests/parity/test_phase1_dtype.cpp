/**
 * @file test_phase1_dtype.cpp
 * @brief FAZ 1 spesifikasyon testleri: DType, half/bfloat16, tip yukseltme,
 *        ayirici stres testi (10^6 rastgele ayirma) ve gorunum semantigi.
 *
 * promt.txt FAZ 1 TESTLER listesi:
 *  - test_dtype_roundtrip  -> tum 65536 binary16 deseni gidis-donus + F16C capraz kontrol
 *  - test_bf16_rounding    -> RNE tablo + 65536 desen gidis-donus
 *  - test_promote_table    -> 11 x 11 tablo
 *  - test_caching_allocator -> 10^6 rastgele boyutlu ayirma/serbest, sizinti yok
 *  - test_view_semantics   -> view() sifir kopya, Tensor kopyasi derin (bugunku sozlesme)
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/allocator.hpp"
#include "nexus/core/dtype.hpp"
#include "nexus/core/half.hpp"
#include "nexus/core/memory.hpp"
#include "nexus/core/tensor.hpp"
#include "nexus_optim/core/half.hpp"
#include "nexus_test.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
// MSVC intrinsic'leri /arch olmadan da sunar; calisma zamani cpuid ile korunur.
#if defined(_MSC_VER) || defined(__F16C__)
#include <immintrin.h>
#define NEXUS_TEST_HAS_F16C_INTRIN 1
#endif

namespace {

[[nodiscard]] bool cpu_has_f16c() {
#if defined(_MSC_VER)
  int info[4] = {0, 0, 0, 0};
  __cpuid(info, 1);
  return (info[2] & (1 << 29)) != 0;
#elif defined(__GNUC__)
  return __builtin_cpu_supports("f16c");
#else
  return false;
#endif
}

#if defined(NEXUS_TEST_HAS_F16C_INTRIN)
// MSVC skaler _cvtss_sh/_cvtsh_ss sunmaz; 128 bit vektor formunun ilk seridi kullanilir.
[[nodiscard]] float f16c_to_float(std::uint16_t bits) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(bits))));
}
[[nodiscard]] std::uint16_t f16c_to_half(float value) {
  return static_cast<std::uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(value), _MM_FROUND_TO_NEAREST_INT)) & 0xFFFF);
}
#endif

[[nodiscard]] bool is_half_nan(std::uint16_t bits) { return (bits & 0x7C00u) == 0x7C00u && (bits & 0x03FFu) != 0u; }
[[nodiscard]] bool is_bf16_nan(std::uint16_t bits) { return (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0u; }

int run() {
  using nexus::DType;

  nexus_test::section("dtype ozellikleri");
  {
    nexus_test::expect(nexus::size_of(DType::F32) == 4 && nexus::size_of(DType::BF16) == 2 &&
                           nexus::size_of(DType::F64) == 8 && nexus::size_of(DType::F8E4M3) == 1,
                       "size_of degerleri");
    nexus_test::expect(nexus::is_floating(DType::BF16) && !nexus::is_floating(DType::I32), "is_floating");
    nexus_test::expect(nexus::is_integral(DType::U8) && !nexus::is_integral(DType::Bool), "is_integral (bool haric)");
    nexus_test::expect(nexus::to_string(DType::F8E4M3) == "float8_e4m3fn", "to_string PyTorch adlari");
  }

  nexus_test::section("test_dtype_roundtrip: 65536 binary16 deseni");
  {
    std::uint32_t mismatches = 0;
    std::uint32_t optim_mismatches = 0;
    for (std::uint32_t raw = 0; raw <= 0xFFFFu; ++raw) {
      const auto bits = static_cast<std::uint16_t>(raw);
      const float value = nexus::half_bits_to_float(bits);
      const std::uint16_t back = nexus::float_to_half_bits(value);
      // Sinyal veren NaN sessiz NaN'a doner; diger her desen bit-bit ayni.
      const std::uint16_t expected = is_half_nan(bits) ? static_cast<std::uint16_t>(bits | 0x0200u) : bits;
      if (back != expected) {
        ++mismatches;
      }
      if (!is_half_nan(bits) && nexus_optim::float16::from_float(value).bits != bits) {
        ++optim_mismatches;
      }
    }
    nexus_test::info("half_roundtrip_mismatch", mismatches);
    nexus_test::expect(mismatches == 0, "tum 65536 half deseni gidis-donus");
    nexus_test::expect(optim_mismatches == 0, "NexusOptim float16 ayni bitleri uretir (tek tanim)");
    nexus_test::expect(nexus::half(65504.0f).bits == 0x7BFFu && nexus::half(65520.0f).bits == 0x7C00u,
                       "en buyuk sonlu ve tasma (RNE)");
    nexus_test::expect(nexus::half(5.960464477539063e-08f).bits == 0x0001u, "en kucuk alt-normal 2^-24");
    nexus_test::expect(nexus::half(2.98023223876953125e-08f).bits == 0x0000u, "2^-25 tam yari -> cift (0)");
    nexus_test::expect(nexus::half(-0.0f).bits == 0x8000u, "-0 isareti korunur");
  }

  nexus_test::section("F16C donanim capraz kontrolu");
  {
#if defined(NEXUS_TEST_HAS_F16C_INTRIN)
    if (cpu_has_f16c()) {
      std::uint32_t to_half_bad = 0;
      std::uint32_t to_float_bad = 0;
      for (std::uint32_t raw = 0; raw <= 0xFFFFu; ++raw) {
        const auto bits = static_cast<std::uint16_t>(raw);
        const float soft = nexus::half_bits_to_float(bits);
        const float hard = f16c_to_float(bits);
        // F16C sinyal veren NaN'i sessizlestirir (0x00400000); yazilim yolu yuku aynen
        // tasir (c10 fp16_ieee_to_fp32 ile ayni). NaN'da isaret + NaN'lik karsilastirilir.
        const bool same = is_half_nan(bits)
                              ? std::isnan(soft) && std::isnan(hard) && std::signbit(soft) == std::signbit(hard)
                              : std::bit_cast<std::uint32_t>(soft) == std::bit_cast<std::uint32_t>(hard);
        if (!same) {
          ++to_float_bad;
        }
      }
      // float -> half: 2^32 uzayindan asal adimla ~1M ornek + tum sinir bolgesi.
      std::uint64_t samples = 0;
      for (std::uint64_t raw = 0; raw <= 0xFFFFFFFFull; raw += 4093u) {
        const float value = std::bit_cast<float>(static_cast<std::uint32_t>(raw));
        const std::uint16_t soft = nexus::float_to_half_bits(value);
        const auto hard = f16c_to_half(value);
        ++samples;
        if (std::isnan(value) ? !(is_half_nan(soft) && is_half_nan(hard) && (soft & 0x8000u) == (hard & 0x8000u))
                              : soft != hard) {
          ++to_half_bad;
        }
      }
      for (std::uint32_t raw = 0x33000000u - 4096u; raw <= 0x38800000u + 4096u; raw += 1u) {
        const float value = std::bit_cast<float>(raw);
        ++samples;
        if (nexus::float_to_half_bits(value) != f16c_to_half(value)) {
          ++to_half_bad;
        }
      }
      nexus_test::info("f16c_float_to_half_samples", static_cast<double>(samples));
      nexus_test::expect(to_float_bad == 0, "half->float F16C ile bit-bit ayni (65536; NaN yuku haric)");
      nexus_test::expect(to_half_bad == 0, "float->half F16C ile bit-bit ayni (ornekler + alt-normal bolge)");
    } else {
      std::printf("  info  F16C yok: donanim capraz kontrolu atlandi (SKIPPED)\n");
    }
#else
    std::printf("  info  F16C intrinsic derlenmedi: capraz kontrol atlandi (SKIPPED)\n");
#endif
  }

  nexus_test::section("test_bf16_rounding");
  {
    struct Case {
      std::uint32_t input;
      std::uint16_t expected;
      const char* name;
    };
    const Case cases[] = {
        {0x3F800000u, 0x3F80u, "1.0 tam"},
        {0x3F808000u, 0x3F80u, "tam yari, alt cift -> asagi"},
        {0x3F818000u, 0x3F82u, "tam yari, alt tek -> yukari"},
        {0x3F808001u, 0x3F81u, "yarinin ustu -> yukari"},
        {0x3F807FFFu, 0x3F80u, "yarinin alti -> asagi"},
        {0x7F7FFFFFu, 0x7F80u, "FLT_MAX RNE ile Inf"},
        {0x7F800000u, 0x7F80u, "+Inf"},
        {0xFF800000u, 0xFF80u, "-Inf"},
        {0x80000000u, 0x8000u, "-0"},
        {0x00000001u, 0x0000u, "en kucuk alt-normal -> 0"},
        {0x7F800001u, 0x7FC0u, "sinyal NaN -> sessiz NaN (Inf'e yuvarlanmaz)"},
    };
    for (const Case& c : cases) {
      nexus_test::expect(nexus::float_to_bfloat16_bits(std::bit_cast<float>(c.input)) == c.expected,
                         std::string("bf16 ") + c.name);
    }
    std::uint32_t mismatches = 0;
    std::uint32_t optim_mismatches = 0;
    for (std::uint32_t raw = 0; raw <= 0xFFFFu; ++raw) {
      const auto bits = static_cast<std::uint16_t>(raw);
      const float value = nexus::bfloat16_bits_to_float(bits);
      const std::uint16_t back = nexus::float_to_bfloat16_bits(value);
      const std::uint16_t expected = is_bf16_nan(bits) ? static_cast<std::uint16_t>(bits | 0x0040u) : bits;
      if (back != expected) {
        ++mismatches;
      }
      if (!is_bf16_nan(bits) && nexus_optim::bfloat16::from_float(value).bits != bits) {
        ++optim_mismatches;
      }
    }
    nexus_test::expect(mismatches == 0, "tum 65536 bf16 deseni gidis-donus");
    nexus_test::expect(optim_mismatches == 0, "NexusOptim bfloat16 ayni bitleri uretir (tek tanim)");
  }

  nexus_test::section("test_promote_table (11 x 11)");
  {
    // Satir/sutun sirasi: F32 F16 BF16 F64 I32 I64 I8 U8 Bool ; FP8 ayri denetlenir.
    const DType order[] = {DType::F32, DType::F16, DType::BF16, DType::F64, DType::I32,
                           DType::I64, DType::I8,  DType::U8,   DType::Bool};
    const DType F32 = DType::F32, F16 = DType::F16, B16 = DType::BF16, F64 = DType::F64, I32 = DType::I32,
                I64 = DType::I64, I8 = DType::I8, U8 = DType::U8, BOL = DType::Bool;
    const DType expected[9][9] = {
        /* F32  */ {F32, F32, F32, F64, F32, F32, F32, F32, F32},
        /* F16  */ {F32, F16, F32, F64, F16, F16, F16, F16, F16},
        /* BF16 */ {F32, F32, B16, F64, B16, B16, B16, B16, B16},
        /* F64  */ {F64, F64, F64, F64, F64, F64, F64, F64, F64},
        /* I32  */ {F32, F16, B16, F64, I32, I64, I32, I32, I32},
        /* I64  */ {F32, F16, B16, F64, I64, I64, I64, I64, I64},
        /* I8   */ {F32, F16, B16, F64, I32, I64, I8, I32, I8},
        /* U8   */ {F32, F16, B16, F64, I32, I64, I32, U8, U8},
        /* Bool */ {F32, F16, B16, F64, I32, I64, I8, U8, BOL},
    };
    int bad = 0;
    for (int i = 0; i < 9; ++i) {
      for (int j = 0; j < 9; ++j) {
        if (nexus::promote(order[i], order[j]) != expected[i][j]) {
          std::printf("  info  promote(%s,%s) beklenmeyen\n", std::string(nexus::to_string(order[i])).c_str(),
                      std::string(nexus::to_string(order[j])).c_str());
          ++bad;
        }
      }
    }
    nexus_test::expect(bad == 0, "9x9 tablo PyTorch kurallariyla ayni (U8+I8 -> I32 belgelenmis sapma)");
    int fp8_ok = 0;
    for (const DType other : nexus::kAllDTypes) {
      for (const DType f8 : {DType::F8E4M3, DType::F8E5M2}) {
        if (other == f8) {
          fp8_ok += nexus::promote(f8, f8) == f8 ? 1 : 0;
          continue;
        }
        try {
          (void)nexus::promote(f8, other);
        } catch (const nexus::TensorError&) {
          ++fp8_ok;
        }
      }
    }
    nexus_test::expect(fp8_ok == 22, "FP8 yalniz kendisiyle; digerleriyle acik hata (2 x 11)");
  }

  nexus_test::section("test_caching_allocator: 10^6 rastgele ayirma/serbest");
  {
    auto& pool = nexus::host_allocator();
    pool.reset();
    const std::uint64_t live_before = nexus::live_allocation_count();
    std::mt19937_64 rng(20261024);
    std::uniform_int_distribution<int> size_pick(1, 1 << 16);
    std::vector<void*> live;
    live.reserve(4096);
    constexpr int kOps = 1000000;
    bool aligned = true;
    for (int op = 0; op < kOps; ++op) {
      const bool do_free = !live.empty() && (live.size() >= 4096 || (rng() & 1u) == 0u);
      if (do_free) {
        const std::size_t pick = static_cast<std::size_t>(rng() % live.size());
        nexus::release_owned(live[pick]);
        live[pick] = live.back();
        live.pop_back();
      } else {
        void* p = nexus::allocate_aligned(static_cast<std::size_t>(size_pick(rng)));
        aligned = aligned && (reinterpret_cast<std::uintptr_t>(p) % 64u) == 0u;
        live.push_back(p);
      }
    }
    for (void* p : live) {
      nexus::release_owned(p);
    }
    const auto stats = pool.stats();
    nexus_test::expect(aligned, "her ayirma 64 bayt hizali");
    nexus_test::expect(nexus::live_allocation_count() == live_before, "sizinti yok: canli kapi sayaci basa dondu");
    nexus_test::expect(stats.current_bytes == 0, "havuzda canli bayt kalmadi");
    nexus_test::info("allocator_cache_hits", static_cast<double>(stats.cache_hits));
    nexus_test::expect(stats.cache_hits > static_cast<std::uint64_t>(kOps) / 4, "onbellek isabeti baskin");
    pool.empty_cache();
    nexus_test::expect(pool.stats().reserved_bytes == 0, "empty_cache tum onbellegi OS'a verir");
  }

  nexus_test::section("test_view_semantics (bugunku sozlesme)");
  {
    nexus::Tensor tensor = nexus::Tensor::zeros(nexus::Shape{2, 3});
    auto view = tensor.view();
    nexus_test::expect(view.data() == tensor.data(), "view() sifir kopya: ayni data_ptr");
    nexus_test::expect(view.strides()[0] == 3 && view.strides()[1] == 1, "bitisik stride [3,1]");
    const nexus::Tensor copy = tensor;
    nexus_test::expect(copy.data() != tensor.data(), "Tensor kopya kurucusu derin kopyadir (KURAL 5)");
  }

  return nexus_test::finish("faz1-dtype-ve-ayirici");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    return run();
  } catch (const std::exception& error) {
    std::printf("  FAIL  beklenmeyen istisna: %s\n", error.what());
    return 1;
  }
}
