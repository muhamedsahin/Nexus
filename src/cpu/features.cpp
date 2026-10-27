/**
 * @file features.cpp
 * @brief FAZ 18.A: CPUID / XGETBV algilama, Windows / Linux topolojisi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cpu/features.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace nexus {
namespace cpu {

namespace {

void cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER)
  int v[4];
  __cpuidex(v, leaf, sub);
  for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(v[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

unsigned long long xgetbv0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
}

Features detect() {
  Features f;
  unsigned r[4];
  cpuid(0, 0, r);
  const unsigned max_leaf = r[0];
  char v[13] = {};
  std::memcpy(v, &r[1], 4);
  std::memcpy(v + 4, &r[3], 4);
  std::memcpy(v + 8, &r[2], 4);
  f.vendor = v;
  cpuid(0x80000000, 0, r);
  if (r[0] >= 0x80000004) {
    char b[49] = {};
    for (int i = 0; i < 3; ++i) {
      cpuid(0x80000002 + i, 0, r);
      std::memcpy(b + 16 * i, r, 16);
    }
    f.brand = b;
    while (!f.brand.empty() && f.brand.back() == ' ') f.brand.pop_back();
  }
  cpuid(1, 0, r);
  const bool osxsave = (r[2] >> 27) & 1;
  f.sse42 = (r[2] >> 20) & 1;
  f.fma = (r[2] >> 12) & 1;
  f.f16c = (r[2] >> 29) & 1;
  const bool cpu_avx = (r[2] >> 28) & 1;
  const unsigned long long xcr0 = osxsave ? xgetbv0() : 0;
  const bool os_avx = (xcr0 & 0x6) == 0x6;
  const bool os_avx512 = (xcr0 & 0xE6) == 0xE6;
  const bool os_amx = (xcr0 & 0x60000) == 0x60000;
  f.avx = cpu_avx && os_avx;
  f.fma = f.fma && f.avx;
  f.f16c = f.f16c && f.avx;
  if (max_leaf >= 7) {
    cpuid(7, 0, r);
    f.avx2 = ((r[1] >> 5) & 1) && f.avx;
    f.bmi2 = (r[1] >> 8) & 1;
    f.avx512f = ((r[1] >> 16) & 1) && os_avx512;
    f.avx512bw = ((r[1] >> 30) & 1) && os_avx512;
    f.avx512vl = ((r[1] >> 31) & 1) && os_avx512;
    f.avx512vnni = ((r[2] >> 11) & 1) && os_avx512;
    f.avx512fp16 = ((r[3] >> 23) & 1) && os_avx512;
    f.amx_bf16 = ((r[3] >> 22) & 1) && os_amx;
    f.amx_tile = ((r[3] >> 24) & 1) && os_amx;
    f.amx_int8 = ((r[3] >> 25) & 1) && os_amx;
    cpuid(7, 1, r);
    f.avx_vnni = ((r[0] >> 4) & 1) && f.avx;
    f.avx512bf16 = ((r[0] >> 5) & 1) && os_avx512;
  }
  f.logical = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
#if defined(_WIN32)
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
  std::vector<char> buf(len);
  f.core_of.assign(static_cast<std::size_t>(f.logical), -1);
  f.node_of.assign(static_cast<std::size_t>(f.logical), 0);
  if (GetLogicalProcessorInformationEx(RelationAll, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
    int core = 0;
    std::set<int> nodes;
    for (char* p = buf.data(); p < buf.data() + len;) {
      auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
      if (info->Relationship == RelationProcessorCore) {
        const KAFFINITY m = info->Processor.GroupMask[0].Mask;
        for (int b = 0; b < 64 && b < f.logical; ++b)
          if (m & (KAFFINITY{1} << b)) f.core_of[static_cast<std::size_t>(b)] = core;
        ++core;
      } else if (info->Relationship == RelationNumaNode) {
        const KAFFINITY m = info->NumaNode.GroupMask.Mask;
        nodes.insert(static_cast<int>(info->NumaNode.NodeNumber));
        for (int b = 0; b < 64 && b < f.logical; ++b)
          if (m & (KAFFINITY{1} << b)) f.node_of[static_cast<std::size_t>(b)] = static_cast<int>(info->NumaNode.NodeNumber);
      } else if (info->Relationship == RelationCache) {
        const auto& c = info->Cache;
        if (c.Level == 1 && c.Type == CacheData) f.l1d = c.CacheSize;
        if (c.Level == 2) f.l2 = c.CacheSize;
        if (c.Level == 3) f.l3 = std::max<std::size_t>(f.l3, c.CacheSize);
      }
      p += info->Size;
    }
    f.physical = std::max(1, core);
    f.numa_nodes = std::max<int>(1, static_cast<int>(nodes.size()));
  }
#else
  f.core_of.resize(static_cast<std::size_t>(f.logical));
  for (int i = 0; i < f.logical; ++i) f.core_of[static_cast<std::size_t>(i)] = i;
  f.node_of.assign(static_cast<std::size_t>(f.logical), 0);
  f.physical = f.logical;
#endif
  return f;
}

}  // namespace

const Features& features() {
  static const Features f = detect();
  return f;
}

const char* isa() {
  const Features& f = features();
  if (!f.avx2 || !f.fma) throw std::runtime_error("nexus::cpu: AVX2 + FMA gerekli (bu islemcide yok; skaler yedek yol yok)");
  return "avx2-fma";
}

std::string describe() {
  const Features& f = features();
  std::ostringstream o;
  o << f.brand << " (" << f.vendor << "): " << f.physical << " cekirdek / " << f.logical << " mantiksal, " << f.numa_nodes
    << " NUMA dugumu; L1d " << f.l1d / 1024 << " KB, L2 " << f.l2 / 1024 << " KB, L3 " << f.l3 / (1024 * 1024) << " MB; ISA:"
    << (f.avx2 ? " avx2" : "") << (f.fma ? " fma" : "") << (f.f16c ? " f16c" : "") << (f.avx_vnni ? " avx-vnni" : "")
    << (f.avx512f ? " avx512f" : "") << (f.avx512bw ? " avx512bw" : "") << (f.avx512vnni ? " avx512vnni" : "")
    << (f.avx512bf16 ? " avx512bf16" : "") << (f.avx512fp16 ? " avx512fp16" : "") << (f.amx_tile ? " amx" : "")
    << (f.amx_bf16 ? " amx-bf16" : "") << (f.amx_int8 ? " amx-int8" : "");
  return o.str();
}

}  // namespace cpu
}  // namespace nexus
