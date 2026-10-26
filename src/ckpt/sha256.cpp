/**
 * @file sha256.cpp
 * @brief FAZ 15: kayit ozeti icin SHA-256 — x86 SHA uzantilari (SHA-NI) varsa donanim yolu, yoksa NexusData'nin
 *        skaler uygulamasi. Secim calisma aninda CPUID ile (yaprak 7, EBX bit 29 + SSSE3/SSE4.1); iki yol ayni
 *        ozeti verir (test_crash_injection karsilastirir).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/ckpt/checkpoint.hpp"

#include "nexusdata/text/bpe.hpp"

#include <cstring>

#if defined(_M_X64) || defined(__x86_64__)
#define NEXUS_CKPT_SHA_NI 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace nexus {
namespace ckpt {

namespace {

#if defined(NEXUS_CKPT_SHA_NI)
alignas(16) constexpr std::uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

bool cpu_has_sha_ni() {
  int r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuidex(r, 1, 0);
  const bool ssse3 = (r[2] >> 9) & 1, sse41 = (r[2] >> 19) & 1;
  __cpuidex(r, 7, 0);
#else
  unsigned a, b, c, d;
  if (__get_cpuid_max(0, nullptr) < 7) return false;
  __cpuid_count(1, 0, a, b, c, d);
  const bool ssse3 = (c >> 9) & 1, sse41 = (c >> 19) & 1;
  __cpuid_count(7, 0, a, b, c, d);
  r[1] = static_cast<int>(b);
#endif
  return ssse3 && sse41 && ((r[1] >> 29) & 1);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("sha,sse4.1,ssse3")))
#endif
void blocks_sha_ni(std::uint32_t state[8], const unsigned char* data, std::size_t blocks) {
  const __m128i mask = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
  __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
  __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
  tmp = _mm_shuffle_epi32(tmp, 0xB1);        // CDAB
  s1 = _mm_shuffle_epi32(s1, 0x1B);          // EFGH
  __m128i s0 = _mm_alignr_epi8(tmp, s1, 8);  // ABEF
  s1 = _mm_blend_epi16(s1, tmp, 0xF0);       // CDGH
  while (blocks-- > 0) {
    const __m128i abef = s0, cdgh = s1;
    __m128i w[16];
    for (int g = 0; g < 16; ++g) {
      if (g < 4) {
        w[g] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * g)), mask);
      } else {
        // W_t = s1(W_t-2) + W_t-7 + s0(W_t-15) + W_t-16 (4'lu gruplar halinde)
        const __m128i t = _mm_add_epi32(_mm_sha256msg1_epu32(w[g - 4], w[g - 3]), _mm_alignr_epi8(w[g - 1], w[g - 2], 4));
        w[g] = _mm_sha256msg2_epu32(t, w[g - 1]);
      }
      __m128i msg = _mm_add_epi32(w[g], _mm_load_si128(reinterpret_cast<const __m128i*>(&kK[4 * g])));
      s1 = _mm_sha256rnds2_epu32(s1, s0, msg);
      msg = _mm_shuffle_epi32(msg, 0x0E);
      s0 = _mm_sha256rnds2_epu32(s0, s1, msg);
    }
    s0 = _mm_add_epi32(s0, abef);
    s1 = _mm_add_epi32(s1, cdgh);
    data += 64;
  }
  tmp = _mm_shuffle_epi32(s0, 0x1B);    // FEBA
  s1 = _mm_shuffle_epi32(s1, 0xB1);     // DCHG
  s0 = _mm_blend_epi16(tmp, s1, 0xF0);  // DCBA
  s1 = _mm_alignr_epi8(s1, tmp, 8);     // ABEF
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), s0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), s1);
}

std::array<std::uint8_t, 32> sha256_ni(const unsigned char* p, std::size_t n) {
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const std::size_t full = n / 64;
  blocks_sha_ni(h, p, full);
  unsigned char tail[128] = {};
  const std::size_t rem = n - full * 64;
  std::memcpy(tail, p + full * 64, rem);
  tail[rem] = 0x80;
  const std::size_t tail_blocks = rem + 1 + 8 <= 64 ? 1 : 2;
  const std::uint64_t bits = static_cast<std::uint64_t>(n) * 8;
  for (int i = 0; i < 8; ++i) tail[tail_blocks * 64 - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
  blocks_sha_ni(h, tail, tail_blocks);
  std::array<std::uint8_t, 32> out{};
  for (int i = 0; i < 8; ++i) {
    out[4 * i] = static_cast<std::uint8_t>(h[i] >> 24);
    out[4 * i + 1] = static_cast<std::uint8_t>(h[i] >> 16);
    out[4 * i + 2] = static_cast<std::uint8_t>(h[i] >> 8);
    out[4 * i + 3] = static_cast<std::uint8_t>(h[i]);
  }
  return out;
}
#endif

bool use_hw() {
#if defined(NEXUS_CKPT_SHA_NI)
  static const bool hw = cpu_has_sha_ni();
  return hw;
#else
  return false;
#endif
}

}  // namespace

std::array<std::uint8_t, 32> sha256(const void* data, std::size_t size, bool allow_hw) {
#if defined(NEXUS_CKPT_SHA_NI)
  if (allow_hw && use_hw()) return sha256_ni(static_cast<const unsigned char*>(data), size);
#endif
  nexusdata::text::Sha256 h;
  h.update(data, size);
  return h.finish();
}

const char* sha256_backend() { return use_hw() ? "sha-ni" : "skaler (NexusData)"; }

}  // namespace ckpt
}  // namespace nexus
