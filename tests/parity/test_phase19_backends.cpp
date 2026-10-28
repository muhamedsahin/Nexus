/**
 * @file test_phase19_backends.cpp
 * @brief FAZ 19: Coklu hizlandirici soyutlamasi (CPU, CUDA, ROCm/HIP, Vulkan SPIR-V,
 *        Metal MSL, DirectML DX12, OpenXLA PJRT) parite, geri dusus politikasi,
 *        Vulkan/DirectML uctan uca model egitimi ve StableHLO+GSPMD -> PJRT testleri.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/backend/backend.hpp"
#include "nexus/backend/hip_compat.hpp"
#include "nexus/backend/pjrt.hpp"
#include "nexus/compile/compile.hpp"
#include "nexus/compile/ir.hpp"
#include "nexus/compile/trace.hpp"
#include "nexus/core/device.hpp"
#include "nexus/core/mirrored_tensor.hpp"
#include "nexus/dist/dtensor.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus/interop/json.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace nb = nexus::backend;
namespace ir = nexus::ir;
using nexus::DeviceKind;
using nexus::DeviceRef;
using nexus::MirroredTensor;
using nexus::MirrorState;
using nexus::Shape;
using nexus_test::expect;

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
  float worst = 0.0f;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<float>::infinity();
    worst = std::max(worst, std::fabs(a[i] - b[i]));
  }
  return worst;
}

std::vector<float> make_data(std::size_t n, float scale = 0.05f, int seed = 17) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const int v = static_cast<int>((i * static_cast<std::size_t>(seed) + 13u) % 97u) - 48;
    out[i] = static_cast<float>(v) * scale;
  }
  return out;
}

// ============================================================================
//  1) test_backend_detection (19.A + 19.D Metal unified)
// ============================================================================

void test_backend_detection() {
  nexus_test::section("FAZ 19.A: Backend algilama, Device cozumleme ve geri dusus politikasi");

  // 1.1 DeviceRef::parse kontrolu
  expect(DeviceRef::parse("cpu") == DeviceRef::cpu(), "parse('cpu') == cpu");
  expect(DeviceRef::parse("cuda:0") == DeviceRef::cuda(0), "parse('cuda:0') == cuda(0)");
  expect(DeviceRef::parse("rocm:1") == DeviceRef::rocm(1), "parse('rocm:1') == rocm(1)");
  expect(DeviceRef::parse("hip:0") == DeviceRef::rocm(0), "parse('hip:0') == rocm(0)");
  expect(DeviceRef::parse("vulkan:0") == DeviceRef::vulkan(0), "parse('vulkan:0') == vulkan(0)");
  expect(DeviceRef::parse("mps:0") == DeviceRef::metal(0), "parse('mps:0') == metal(0)");
  expect(DeviceRef::parse("metal:0") == DeviceRef::metal(0), "parse('metal:0') == metal(0)");
  expect(DeviceRef::parse("dml:0") == DeviceRef::directml(0), "parse('dml:0') == directml(0)");
  expect(DeviceRef::parse("directml:1") == DeviceRef::directml(1), "parse('directml:1') == directml(1)");
  expect(DeviceRef::parse("tpu:0") == DeviceRef::tpu(0), "parse('tpu:0') == tpu(0)");
  expect(DeviceRef::parse("pjrt:0") == DeviceRef::tpu(0), "parse('pjrt:0') == tpu(0)");

  bool bad_threw = false;
  try {
    (void)DeviceRef::parse("unknown_acc:0");
  } catch (const nexus::TensorError&) {
    bad_threw = true;
  }
  expect(bad_threw, "bilinmeyen cihaz dizgesi acik TensorError firlatir");

  // 1.2 Tum 7 backend turunun kayitli oldugunu ve durumlarini raporladigini dogrula
  auto& reg = nb::Registry::instance();
  expect(reg.all().size() >= 7, "Registry 7 backend surucusunu (cpu, cuda, hip, vulkan, metal, directml, pjrt) icerir");

  for (nb::Driver* d : reg.all()) {
    std::printf("  info  backend=%-10s name=%-13s version=%-8s available=%s devices=%zu reason='%s'\n",
                nb::to_string(d->kind()), std::string(d->name()).c_str(), std::string(d->version()).c_str(),
                d->available() ? "evet" : "hayir", d->devices().size(), d->unavailable_reason().c_str());
    for (const nb::DeviceInfo& info : d->devices()) {
      std::printf("        -> dev=%s '%s' vendor=%s api=%s mem=%.1f MB subgroup=%u unified=%s\n",
                  info.device.to_string().c_str(), info.name.c_str(), info.vendor.c_str(),
                  info.api_version.c_str(), static_cast<double>(info.memory_bytes) / (1024.0 * 1024.0),
                  info.subgroup_size, info.unified_memory ? "evet" : "hayir");
    }
    if (!d->available()) {
      expect(!d->unavailable_reason().empty(),
             std::string("mevcut olmayan backend (") + nb::to_string(d->kind()) + ") acik neden raporlar");
    }
  }

  expect(reg.require(nb::Kind::Cpu).available(), "CPU surucusu her zaman kullanilabilir");
  expect(reg.require(nb::Kind::Pjrt).available(), "Nexus StableHLO CPU reference interpreter available");

  // 1.3 Apple Metal MirroredTensor unified=true (FAZ 19.D)
  MirroredTensor mps_tensor(Shape{4, 8}, DeviceRef::metal(0));
  expect(!mps_tensor.unified(), "Metal host allocation must not impersonate a shared GPU buffer");
  if (!reg.require(nb::Kind::Metal).available()) {
    bool unavailable = false;
    try { (void)mps_tensor.device_data(); } catch (const nexus::TensorError&) { unavailable = true; }
    expect(unavailable, "Unavailable Metal device access throws instead of returning host storage");
  }

  // 1.4 FallbackPolicy::Error, WarnOnce, WarnAlways (FAZ 19.A — sessiz geri dusus yasak)
  nb::reset_fallback_stats();
  nb::set_fallback_policy(nb::FallbackPolicy::Error);
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> out(4, 0.0f);

  bool error_policy_threw = false;
  try {
    nb::elementwise(nb::Operation::Add, DeviceRef::metal(0), a.data(), b.data(), out.data(), 4);
  } catch (const nexus::TensorError&) {
    error_policy_threw = true;
  }
  expect(error_policy_threw, "FallbackPolicy::Error mevcut olmayan backend isleminde sessizce dusmez, hata firlatir");

  nb::set_fallback_policy(nb::FallbackPolicy::WarnOnce);
  nb::elementwise(nb::Operation::Add, DeviceRef::metal(0), a.data(), b.data(), out.data(), 4);
  nb::elementwise(nb::Operation::Add, DeviceRef::metal(0), a.data(), b.data(), out.data(), 4);
  const nb::FallbackStats st = nb::fallback_stats(nb::Kind::Metal, nb::Operation::Add);
  expect(st.calls == 2 && st.elements == 8, "FallbackPolicy::WarnOnce geri dusus sayaclarini kaydeder");
  expect(out[0] == 6.0f && out[3] == 12.0f, "WarnOnce geri dususu dogru CPU sonucunu uretir");
  nb::set_fallback_policy(nb::FallbackPolicy::Error);
}

// ============================================================================
//  2) test_hip_parity (19.B: ROCm/HIP hipify + wave64 denetimi + 64KB LDS)
// ============================================================================

void test_hip_source_compat() {
  nexus_test::section("FAZ 19.B: ROCm / HIP uyumlulugu, hipify donusumu, wave64 ve 64 KB LDS");

  const auto& sym_table = nb::hip::symbol_table();
  expect(sym_table.size() >= 30, "CUDA->HIP sembol esleme tablosu >= 30 API icerir");

  const std::string sample_cuda = R"CUDA(
#include <cuda_runtime.h>
#include <cuda_bf16.h>
void launch() {
  void* p = nullptr;
  cudaMalloc(&p, 1024);
  cudaMemcpyAsync(p, p, 1024, cudaMemcpyDeviceToDevice, 0);
  cublasLtMatmul(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
  ncclAllReduce(p, p, 256, 0, 0, nullptr, 0);
  cudaFree(p);
}
)CUDA";
  const std::string hipified = nb::hip::hipify_source(sample_cuda);
  expect(hipified.find("<hip/hip_runtime.h>") != std::string::npos &&
             hipified.find("hipMalloc(") != std::string::npos &&
             hipified.find("hipMemcpyAsync(") != std::string::npos &&
             hipified.find("hipblasLtMatmul(") != std::string::npos &&
             hipified.find("rcclAllReduce(") != std::string::npos &&
             hipified.find("hipFree(") != std::string::npos,
         "hipify_source cudaMalloc->hipMalloc, cublasLtMatmul->hipblasLtMatmul, nccl*->rccl* donusumu yapar");

  // ROCm 64 KB LDS FlashAttention karo secimi
  for (int d : {64, 128}) {
    const nb::hip::FlashTileConfig tile = nb::hip::select_rocm_flash_tiles(d, 64);
    std::printf("  info  ROCm FlashAttention D=%d wave=%d -> Br=%d, Bc=%d, LDS=%zu bayt (limit=%zu)\n",
                d, tile.wavefront_size, tile.br, tile.bc, tile.lds_bytes, nb::hip::kRocmLdsLimitBytes);
    expect(tile.fits_lds && tile.lds_bytes <= nb::hip::kRocmLdsLimitBytes,
           "ROCm FlashAttention karo boyutu 64 KB LDS sinirina sigar (D=" + std::to_string(d) + ")");
  }

  // Monorepo CUDA cekirdeklerinin wave64/wave32 denetimi
#if defined(NEXUS_SOURCE_DIR)
  const nb::hip::KernelAuditReport audit = nb::hip::audit_cuda_kernels_for_hip(NEXUS_SOURCE_DIR);
  std::printf("  info  HIP kaynak denetimi: taranan=%d, wave-uyumlu=%d, ihlal=%d\n",
              audit.files_scanned, audit.wave_adaptive_reductions, audit.hardcoded_warp_violations);
  expect(audit.all_clean, "src/cuda cekirdeklerinde sabit 32 yerine wave64/32 uyumlu kWarp kullaniliyor");
#endif

  expect(!nb::hip::embedded_hip_kernels_source().empty(), "gomulu HIP cekirdek kaynagi hazir");
}

// ============================================================================
//  3) test_vulkan_parity (19.C: SPIR-V cekirdekleri + uctan uca egitim)
// ============================================================================

void run_vulkan_device_suite(int dev_idx, const nb::DeviceInfo& info) {
  std::printf("  info  Vulkan cihaz [%d] test ediliyor: %s (subgroup=%u, unified=%s)\n",
              dev_idx, info.name.c_str(), info.subgroup_size, info.unified_memory ? "evet" : "hayir");
  nb::Driver& vk = nb::Registry::instance().require(nb::Kind::Vulkan);
  nb::Driver& cpu = nb::Registry::instance().require(nb::Kind::Cpu);
  const DeviceRef dev = DeviceRef::vulkan(dev_idx);

  // 3.1 MirroredTensor H2D / D2H ve elementwise (Add, Mul, Silu, Exp, Tanh, Axpy)
  const std::size_t N = 1024;
  MirroredTensor ma(Shape{N}, dev), mb(Shape{N}, dev), mout(Shape{N}, dev);
  const auto va = make_data(N, 0.03f, 7);
  const auto vb = make_data(N, 0.03f, 19);
  std::memcpy(ma.host_data(), va.data(), N * sizeof(float));
  std::memcpy(mb.host_data(), vb.data(), N * sizeof(float));
  ma.modify_host();
  mb.modify_host();
  ma.sync_device();
  mb.sync_device();
  mout.sync_device();

  for (nb::Operation op : {nb::Operation::Add, nb::Operation::Sub, nb::Operation::Mul,
                           nb::Operation::Silu, nb::Operation::Tanh, nb::Operation::Axpy}) {
    mout.modify_device();
    nb::elementwise(op, dev, ma.device_data(), mb.device_data(), mout.device_data(), N, 0.5f);
    mout.sync_host();
    std::vector<float> ref(N);
    cpu.elementwise(op, DeviceRef::cpu(), va.data(), vb.data(), ref.data(), N, 0.5f);
    std::vector<float> got(mout.host_data(), mout.host_data() + N);
    expect(max_abs_diff(got, ref) < 1e-5f,
           std::string("Vulkan[") + std::to_string(dev_idx) + "] elementwise " + nb::to_string(op) + " < 1e-5");
  }

  // 3.2 ReduceSum
  MirroredTensor msum(Shape{1}, dev);
  msum.sync_device();
  msum.modify_device();
  nb::reduce_sum(dev, ma.device_data(), msum.device_data(), N);
  msum.sync_host();
  float ref_sum = 0.0f;
  cpu.reduce_sum(DeviceRef::cpu(), va.data(), &ref_sum, N);
  expect(std::fabs(msum.host_data()[0] - ref_sum) < 1e-3f,
         "Vulkan[" + std::to_string(dev_idx) + "] reduce_sum < 1e-3");

  // 3.3 Tiled GEMM (normal ve transpozlu)
  {
    const int M = 67, K = 96, Ncols = 53;
    const auto hA = make_data(static_cast<std::size_t>(M * K), 0.02f, 11);
    const auto hB = make_data(static_cast<std::size_t>(Ncols * K), 0.02f, 23);
    MirroredTensor dA(Shape{M, K}, dev), dB(Shape{Ncols, K}, dev), dC(Shape{M, Ncols}, dev);
    std::memcpy(dA.host_data(), hA.data(), hA.size() * sizeof(float));
    std::memcpy(dB.host_data(), hB.data(), hB.size() * sizeof(float));
    dA.modify_host(); dB.modify_host();
    dA.sync_device(); dB.sync_device(); dC.sync_device();
    dC.modify_device();

    nb::MatMulArgs mm{M, Ncols, K, dA.device_data(), K, false, dB.device_data(), K, true, dC.device_data(), Ncols, 1.0f, 0.0f};
    nb::matmul(dev, mm);
    dC.sync_host();

    std::vector<float> refC(static_cast<std::size_t>(M * Ncols), 0.0f);
    nb::MatMulArgs cmm = mm;
    cmm.a = hA.data(); cmm.b = hB.data(); cmm.c = refC.data();
    cpu.matmul(DeviceRef::cpu(), cmm);

    std::vector<float> gotC(dC.host_data(), dC.host_data() + M * Ncols);
    expect(max_abs_diff(gotC, refC) < 1e-4f,
           "Vulkan[" + std::to_string(dev_idx) + "] karolu GEMM (A @ B^T, 67x53x96) < 1e-4");
  }

  // 3.4 Softmax, RMSNorm, LayerNorm, RoPE, SwiGLU
  {
    const int rows = 16, cols = 64;
    const auto hx = make_data(rows * cols, 0.04f, 31);
    const auto hw = make_data(cols, 0.02f, 41);
    const auto hb = make_data(cols, 0.01f, 43);
    MirroredTensor dx(Shape{rows, cols}, dev), dw(Shape{cols}, dev), db(Shape{cols}, dev), dy(Shape{rows, cols}, dev);
    std::memcpy(dx.host_data(), hx.data(), hx.size() * sizeof(float));
    std::memcpy(dw.host_data(), hw.data(), hw.size() * sizeof(float));
    std::memcpy(db.host_data(), hb.data(), hb.size() * sizeof(float));
    dx.modify_host(); dw.modify_host(); db.modify_host();
    dx.sync_device(); dw.sync_device(); db.sync_device(); dy.sync_device();

    std::vector<float> ref(rows * cols);
    dy.modify_device();
    nb::softmax(dev, dx.device_data(), dy.device_data(), rows, cols);
    dy.sync_host();
    cpu.softmax(DeviceRef::cpu(), hx.data(), ref.data(), rows, cols);
    expect(max_abs_diff(std::vector<float>(dy.host_data(), dy.host_data() + rows * cols), ref) < 1e-5f,
           "Vulkan[" + std::to_string(dev_idx) + "] Softmax < 1e-5");

    dy.modify_device();
    nb::rms_norm(dev, dx.device_data(), dw.device_data(), dy.device_data(), rows, cols, 1e-5f);
    dy.sync_host();
    cpu.rms_norm(DeviceRef::cpu(), hx.data(), hw.data(), ref.data(), rows, cols, 1e-5f);
    expect(max_abs_diff(std::vector<float>(dy.host_data(), dy.host_data() + rows * cols), ref) < 1e-5f,
           "Vulkan[" + std::to_string(dev_idx) + "] RMSNorm < 1e-5");

    dy.modify_device();
    nb::layer_norm(dev, dx.device_data(), dw.device_data(), db.device_data(), dy.device_data(), rows, cols, 1e-5f);
    dy.sync_host();
    cpu.layer_norm(DeviceRef::cpu(), hx.data(), hw.data(), hb.data(), ref.data(), rows, cols, 1e-5f);
    expect(max_abs_diff(std::vector<float>(dy.host_data(), dy.host_data() + rows * cols), ref) < 1e-5f,
           "Vulkan[" + std::to_string(dev_idx) + "] LayerNorm < 1e-5");

    dy.modify_device();
    nb::rope(dev, dx.device_data(), dy.device_data(), 2, 8, 64, 3, 10000.0f);
    dy.sync_host();
    cpu.rope(DeviceRef::cpu(), hx.data(), ref.data(), 2, 8, 64, 3, 10000.0f);
    expect(max_abs_diff(std::vector<float>(dy.host_data(), dy.host_data() + rows * cols), ref) < 1e-4f,
           "Vulkan[" + std::to_string(dev_idx) + "] RoPE < 1e-4");

    MirroredTensor d_swiglu_out(Shape{rows, cols / 2}, dev);
    d_swiglu_out.sync_device();
    d_swiglu_out.modify_device();
    nb::swiglu(dev, dx.device_data(), d_swiglu_out.device_data(), rows, cols / 2);
    d_swiglu_out.sync_host();
    std::vector<float> ref_swiglu(rows * (cols / 2));
    cpu.swiglu(DeviceRef::cpu(), hx.data(), ref_swiglu.data(), rows, cols / 2);
    expect(max_abs_diff(std::vector<float>(d_swiglu_out.host_data(), d_swiglu_out.host_data() + ref_swiglu.size()), ref_swiglu) < 1e-5f,
           "Vulkan[" + std::to_string(dev_idx) + "] SwiGLU < 1e-5");
  }

  // 3.5 Karolu Dikkat (Attention)
  {
    const int BH = 2, T = 16, D = 32;
    const std::size_t total = static_cast<std::size_t>(BH * T * D);
    const auto hq = make_data(total, 0.05f, 5);
    const auto hk = make_data(total, 0.05f, 13);
    const auto hv = make_data(total, 0.05f, 29);
    MirroredTensor dq(Shape{total}, dev), dk(Shape{total}, dev), dv(Shape{total}, dev), dout(Shape{total}, dev);
    std::memcpy(dq.host_data(), hq.data(), total * sizeof(float));
    std::memcpy(dk.host_data(), hk.data(), total * sizeof(float));
    std::memcpy(dv.host_data(), hv.data(), total * sizeof(float));
    dq.modify_host(); dk.modify_host(); dv.modify_host();
    dq.sync_device(); dk.sync_device(); dv.sync_device(); dout.sync_device();
    dout.modify_device();

    nb::attention(dev, dq.device_data(), dk.device_data(), dv.device_data(), dout.device_data(), BH, T, D, true);
    dout.sync_host();

    std::vector<float> ref(total);
    cpu.attention(DeviceRef::cpu(), hq.data(), hk.data(), hv.data(), ref.data(), BH, T, D, true);
    expect(max_abs_diff(std::vector<float>(dout.host_data(), dout.host_data() + total), ref) < 1e-4f,
           "Vulkan[" + std::to_string(dev_idx) + "] Causal FlashAttention < 1e-4");
  }

  // 3.6 Q8 ve Q4 Nicemlenmis MatMul (FAZ 19.C cikarim cekirdegi)
  {
    const int M = 8, Ncols = 32, K = 64, G = 32;
    const auto hA = make_data(M * K, 0.05f, 17);
    const auto hW = make_data(Ncols * K, 0.05f, 37);

    std::vector<std::int8_t> w_q8(Ncols * K);
    std::vector<float> s_q8(Ncols * (K / G));
    for (int col = 0; col < Ncols; ++col) {
      for (int g = 0; g < K / G; ++g) {
        float amax = 0.0f;
        for (int j = 0; j < G; ++j) amax = std::max(amax, std::fabs(hW[col * K + g * G + j]));
        const float sc = amax > 0.0f ? (amax / 127.0f) : 1.0f;
        s_q8[col * (K / G) + g] = sc;
        for (int j = 0; j < G; ++j) {
          const int q = std::clamp(static_cast<int>(std::round(hW[col * K + g * G + j] / sc)), -127, 127);
          w_q8[col * K + g * G + j] = static_cast<std::int8_t>(q);
        }
      }
    }

    std::vector<std::uint8_t> w_q4(Ncols * (K / 2), 0u);
    std::vector<float> s_q4(Ncols * (K / G));
    for (int col = 0; col < Ncols; ++col) {
      for (int g = 0; g < K / G; ++g) {
        float amax = 0.0f;
        for (int j = 0; j < G; ++j) amax = std::max(amax, std::fabs(hW[col * K + g * G + j]));
        const float sc = amax > 0.0f ? (amax / 7.0f) : 1.0f;
        s_q4[col * (K / G) + g] = sc;
        for (int j = 0; j < G; ++j) {
          const std::size_t flat_idx = static_cast<std::size_t>(col) * K + g * G + j;
          const int q = std::clamp(static_cast<int>(std::round(hW[flat_idx] / sc)), -7, 7) + 8;
          if (flat_idx & 1u) {
            w_q4[flat_idx >> 1u] |= static_cast<std::uint8_t>((q & 0xF) << 4u);
          } else {
            w_q4[flat_idx >> 1u] |= static_cast<std::uint8_t>(q & 0xF);
          }
        }
      }
    }

    void* dA = vk.allocate(dev_idx, hA.size() * sizeof(float));
    void* dWq8 = vk.allocate(dev_idx, w_q8.size());
    void* dSq8 = vk.allocate(dev_idx, s_q8.size() * sizeof(float));
    void* dWq4 = vk.allocate(dev_idx, w_q4.size());
    void* dSq4 = vk.allocate(dev_idx, s_q4.size() * sizeof(float));
    void* dC = vk.allocate(dev_idx, static_cast<std::size_t>(M * Ncols) * sizeof(float));

    vk.copy(dev, dA, DeviceRef::cpu(), hA.data(), hA.size() * sizeof(float));
    vk.copy(dev, dWq8, DeviceRef::cpu(), w_q8.data(), w_q8.size());
    vk.copy(dev, dSq8, DeviceRef::cpu(), s_q8.data(), s_q8.size() * sizeof(float));
    vk.copy(dev, dWq4, DeviceRef::cpu(), w_q4.data(), w_q4.size());
    vk.copy(dev, dSq4, DeviceRef::cpu(), s_q4.data(), s_q4.size() * sizeof(float));

    std::vector<float> got_q8(M * Ncols), ref_q8(M * Ncols);
    nb::quant_matmul_q8(dev, static_cast<const float*>(dA), dWq8, static_cast<const float*>(dSq8),
                        static_cast<float*>(dC), M, Ncols, K, G);
    vk.copy(DeviceRef::cpu(), got_q8.data(), dev, dC, got_q8.size() * sizeof(float));
    cpu.quant_matmul_q8(DeviceRef::cpu(), hA.data(), w_q8.data(), s_q8.data(), ref_q8.data(), M, Ncols, K, G);
    expect(max_abs_diff(got_q8, ref_q8) < 1e-4f,
           "Vulkan[" + std::to_string(dev_idx) + "] Q8 nicemlenmis GEMM < 1e-4");

    std::vector<float> got_q4(M * Ncols), ref_q4(M * Ncols);
    nb::quant_matmul_q4(dev, static_cast<const float*>(dA), dWq4, static_cast<const float*>(dSq4),
                        static_cast<float*>(dC), M, Ncols, K, G);
    vk.copy(DeviceRef::cpu(), got_q4.data(), dev, dC, got_q4.size() * sizeof(float));
    cpu.quant_matmul_q4(DeviceRef::cpu(), hA.data(), w_q4.data(), s_q4.data(), ref_q4.data(), M, Ncols, K, G);
    expect(max_abs_diff(got_q4, ref_q4) < 1e-4f,
           "Vulkan[" + std::to_string(dev_idx) + "] Q4 nicemlenmis GEMM < 1e-4");

    vk.deallocate(dev_idx, dA);
    vk.deallocate(dev_idx, dWq8);
    vk.deallocate(dev_idx, dSq8);
    vk.deallocate(dev_idx, dWq4);
    vk.deallocate(dev_idx, dSq4);
    vk.deallocate(dev_idx, dC);
  }
}

void test_vulkan_end_to_end_training(int dev_idx, const std::string& dev_name) {
  // FAZ 19 Kabul Kriteri: "En az bir ek arka ucta (ROCm veya Vulkan) uctan uca kucuk model egitimi kayip dusurur"
  // 2 katmanli MLP (X[B, D] -> W1[D, H] -> ReLU -> W2[H, D] -> MSE kaybi + AdamW SPIR-V guncellemesi)
  const DeviceRef dev = DeviceRef::vulkan(dev_idx);
  const int B = 16, D = 32, H = 64;
  const float inv_bd = 1.0f / static_cast<float>(B * D);

  MirroredTensor X(Shape{B, D}, dev), Y(Shape{B, D}, dev);
  MirroredTensor W1(Shape{D, H}, dev), mW1(Shape{D, H}, dev), vW1(Shape{D, H}, dev), gW1(Shape{D, H}, dev);
  MirroredTensor W2(Shape{H, D}, dev), mW2(Shape{H, D}, dev), vW2(Shape{H, D}, dev), gW2(Shape{H, D}, dev);
  MirroredTensor Z1(Shape{B, H}, dev), A1(Shape{B, H}, dev), Pred(Shape{B, D}, dev);
  MirroredTensor Diff(Shape{B, D}, dev), Sq(Shape{B, D}, dev), dPred(Shape{B, D}, dev);
  MirroredTensor dA1(Shape{B, H}, dev), Mask(Shape{B, H}, dev), dZ1(Shape{B, H}, dev);
  MirroredTensor LossBuf(Shape{1}, dev);

  const auto hx = make_data(B * D, 0.05f, 3);
  const auto hy = make_data(B * D, 0.05f, 19);
  const auto hw1 = make_data(D * H, 0.02f, 29);
  const auto hw2 = make_data(H * D, 0.02f, 37);

  std::memcpy(X.host_data(), hx.data(), hx.size() * sizeof(float));
  std::memcpy(Y.host_data(), hy.data(), hy.size() * sizeof(float));
  std::memcpy(W1.host_data(), hw1.data(), hw1.size() * sizeof(float));
  std::memcpy(W2.host_data(), hw2.data(), hw2.size() * sizeof(float));
  X.modify_host(); Y.modify_host(); W1.modify_host(); W2.modify_host();
  mW1.modify_host(); vW1.modify_host(); mW2.modify_host(); vW2.modify_host();
  X.sync_device(); Y.sync_device(); W1.sync_device(); W2.sync_device();
  mW1.sync_device(); vW1.sync_device(); mW2.sync_device(); vW2.sync_device();
  gW1.sync_device(); gW2.sync_device(); Z1.sync_device(); A1.sync_device();
  Pred.sync_device(); Diff.sync_device(); Sq.sync_device(); dPred.sync_device();
  dA1.sync_device(); Mask.sync_device(); dZ1.sync_device(); LossBuf.sync_device();

  float first_loss = 0.0f, last_loss = 0.0f;
  for (int step = 1; step <= 25; ++step) {
    // Ileri gecis: Z1 = X @ W1, A1 = ReLU(Z1), Pred = A1 @ W2
    Z1.modify_device(); A1.modify_device(); Pred.modify_device();
    nb::matmul(dev, {B, H, D, X.device_data(), D, false, W1.device_data(), H, false, Z1.device_data(), H, 1.0f, 0.0f});
    nb::elementwise(nb::Operation::Relu, dev, Z1.device_data(), nullptr, A1.device_data(), B * H);
    nb::matmul(dev, {B, D, H, A1.device_data(), H, false, W2.device_data(), D, false, Pred.device_data(), D, 1.0f, 0.0f});

    // Kayip: Diff = Pred - Y, Sq = Diff * Diff, Loss = sum(Sq) / (B*D)
    Diff.modify_device(); Sq.modify_device(); LossBuf.modify_device();
    nb::elementwise(nb::Operation::Sub, dev, Pred.device_data(), Y.device_data(), Diff.device_data(), B * D);
    nb::elementwise(nb::Operation::Mul, dev, Diff.device_data(), Diff.device_data(), Sq.device_data(), B * D);
    nb::reduce_sum(dev, Sq.device_data(), LossBuf.device_data(), B * D);
    LossBuf.sync_host();
    const float loss = LossBuf.host_data()[0] * inv_bd;
    if (step == 1) first_loss = loss;
    last_loss = loss;

    // Geri gecis: dPred = (2 / (B*D)) * Diff
    dPred.modify_device(); gW2.modify_device(); dA1.modify_device();
    nb::elementwise(nb::Operation::Scale, dev, Diff.device_data(), nullptr, dPred.device_data(), B * D, 2.0f * inv_bd);
    // gW2 = A1^T @ dPred
    nb::matmul(dev, {H, D, B, A1.device_data(), H, true, dPred.device_data(), D, false, gW2.device_data(), D, 1.0f, 0.0f});
    // dA1 = dPred @ W2^T
    nb::matmul(dev, {B, H, D, dPred.device_data(), D, false, W2.device_data(), D, true, dA1.device_data(), H, 1.0f, 0.0f});
    // ReLU turevi: A1 > 0 ise 1.0, degilse 0.0 -> Z1 > 0 kontrolu icin Sign/Step:
    // ReLU(Z1) / (Z1 + 1e-12) yerine CPU/SPIR-V uzerinde ReLU maskesi:
    Z1.sync_host(); dA1.sync_host();
    dZ1.modify_host();
    for (int i = 0; i < B * H; ++i) {
      dZ1.host_data()[i] = Z1.host_data()[i] > 0.0f ? dA1.host_data()[i] : 0.0f;
    }
    dZ1.sync_device();

    // gW1 = X^T @ dZ1
    gW1.modify_device();
    nb::matmul(dev, {D, H, B, X.device_data(), D, true, dZ1.device_data(), H, false, gW1.device_data(), H, 1.0f, 0.0f});

    // AdamW SPIR-V guncellemesi
    W1.modify_device(); mW1.modify_device(); vW1.modify_device();
    W2.modify_device(); mW2.modify_device(); vW2.modify_device();
    nb::adamw_step(dev, W1.device_data(), gW1.device_data(), mW1.device_data(), vW1.device_data(),
                   D * H, 1e-2f, 0.9f, 0.999f, 1e-8f, 1e-2f, step);
    nb::adamw_step(dev, W2.device_data(), gW2.device_data(), mW2.device_data(), vW2.device_data(),
                   H * D, 1e-2f, 0.9f, 0.999f, 1e-8f, 1e-2f, step);
  }
  std::printf("  info  Vulkan[%d] (%s) uctan uca 25 adim MLP egitimi: ilk_kayip=%.6f -> son_kayip=%.6f\n",
              dev_idx, dev_name.c_str(), first_loss, last_loss);
  expect(std::isfinite(first_loss) && std::isfinite(last_loss) && last_loss < first_loss * 0.35f,
         "Vulkan[" + std::to_string(dev_idx) + "] uctan uca model egitimi kaybi anlamli sekilde dusurur");
}

void test_vulkan_parity() {
  nexus_test::section("FAZ 19.C: Vulkan SPIR-V parite ve uctan uca model egitimi");
  nb::Driver& vk = nb::Registry::instance().require(nb::Kind::Vulkan);
  if (!vk.available()) {
    std::printf("  SKIP  Vulkan kullanilamiyor: %s\n", vk.unavailable_reason().c_str());
    return;
  }
  for (std::size_t i = 0; i < vk.devices().size(); ++i) {
    run_vulkan_device_suite(static_cast<int>(i), vk.devices()[i]);
    test_vulkan_end_to_end_training(static_cast<int>(i), vk.devices()[i].name);
  }
}

// ============================================================================
//  4) test_directml_parity (19.D: Windows DirectX 12 + DirectML)
// ============================================================================

void test_directml_parity() {
  nexus_test::section("FAZ 19.D: Windows DirectML (DirectX 12) parite testi");
  nb::Driver& dml = nb::Registry::instance().require(nb::Kind::DirectMl);
  nb::Driver& cpu = nb::Registry::instance().require(nb::Kind::Cpu);
  if (!dml.available()) {
    std::printf("  info  DirectML devre disi: %s\n", dml.unavailable_reason().c_str());
    return;
  }
  for (std::size_t i = 0; i < dml.devices().size(); ++i) {
    const int idx = static_cast<int>(i);
    const DeviceRef dev = DeviceRef::directml(idx);
    const int M = 32, K = 64, Ncols = 48;
    const auto hA = make_data(M * K, 0.03f, 7);
    const auto hB = make_data(K * Ncols, 0.03f, 13);
    MirroredTensor dA(Shape{M, K}, dev), dB(Shape{K, Ncols}, dev), dC(Shape{M, Ncols}, dev);
    std::memcpy(dA.host_data(), hA.data(), hA.size() * sizeof(float));
    std::memcpy(dB.host_data(), hB.data(), hB.size() * sizeof(float));
    dA.modify_host(); dB.modify_host();
    dA.sync_device(); dB.sync_device(); dC.sync_device();
    dC.modify_device();

    nb::MatMulArgs mm{M, Ncols, K, dA.device_data(), K, false, dB.device_data(), Ncols, false, dC.device_data(), Ncols, 1.0f, 0.0f};
    nb::matmul(dev, mm);
    dC.sync_host();

    std::vector<float> refC(M * Ncols, 0.0f);
    nb::MatMulArgs cmm = mm;
    cmm.a = hA.data(); cmm.b = hB.data(); cmm.c = refC.data();
    cpu.matmul(DeviceRef::cpu(), cmm);

    const float err = max_abs_diff(std::vector<float>(dC.host_data(), dC.host_data() + M * Ncols), refC);
    std::printf("  info  DirectML[%d] (%s) DML_OPERATOR_GEMM max_abs=%.2e\n",
                idx, dml.devices()[i].name.c_str(), err);
    expect(err < 1e-4f, "DirectML[" + std::to_string(idx) + "] DML_OPERATOR_GEMM < 1e-4");
  }
}

// ============================================================================
//  5) test_pjrt_export (19.E: Grafik -> StableHLO + GSPMD -> PJRT C API -> Yurutme)
// ============================================================================

void test_pjrt_export(const char* export_path = nullptr) {
  nexus_test::section("FAZ 19.E: StableHLO + GSPMD (mhlo.sharding) -> PJRT C API derleme ve yurutme");

  // 5.1 DeviceMesh + DTensor -> GSPMD mhlo.sharding dogrulamasi
  nexus::dist::ThreadWorld world(4);
  std::string sharding_rep, sharding_2d, sharding_partial;
  std::string sharding_swapped, sharding_replicated;
  bool partial_rejected = false;
  {
    std::vector<std::thread> threads;
    for (int r = 0; r < 4; ++r) {
      threads.emplace_back([&, r] {
        auto grp = world.create_group(r);
        nexus::dist::DeviceMesh mesh(grp, {2, 2}, {"dp", "tp"});
        if (r == 0) {
          sharding_rep = nb::pjrt::to_mhlo_sharding({16, 32}, mesh,
              {nexus::dist::Placement::replicate(), nexus::dist::Placement::replicate()});
          sharding_2d = nb::pjrt::to_mhlo_sharding({16, 32}, mesh,
              {nexus::dist::Placement::shard(0), nexus::dist::Placement::shard(1)});
          sharding_swapped = nb::pjrt::to_mhlo_sharding({16, 32}, mesh,
              {nexus::dist::Placement::shard(1), nexus::dist::Placement::shard(0)});
          sharding_replicated = nb::pjrt::to_mhlo_sharding({16, 32}, mesh,
              {nexus::dist::Placement::replicate(), nexus::dist::Placement::shard(0)});
          try {
            sharding_partial = nb::pjrt::to_mhlo_sharding({16, 32}, mesh,
                {nexus::dist::Placement::replicate(), nexus::dist::Placement::partial()});
          } catch (const nexus::TensorError&) { partial_rejected = true; }
        }
      });
    }
    for (auto& t : threads) t.join();
  }
  std::printf("  info  GSPMD Replicate -> %s\n", sharding_rep.c_str());
  std::printf("  info  GSPMD Shard(0,1)-> %s\n", sharding_2d.c_str());
  std::printf("  info  GSPMD Partial   -> %s\n", sharding_partial.c_str());
  expect(sharding_rep == "{replicated}", "Replicate yerlesimi {replicated} uretir");
  expect(sharding_2d == "{devices=[2,2]0,1,2,3}", "2B Shard(0),Shard(1) yerlesimi {devices=[2,2]0,1,2,3} uretir");
  expect(partial_rejected, "Partial(sum) cannot be misrepresented as manual XLA sharding");
  expect(sharding_swapped == "{devices=[2,2]0,2,1,3}", "Swapped mesh axes reorder XLA tile devices");
  expect(sharding_replicated == "{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}",
         "Replication dimension follows tensor tile dimensions");

  // 5.2 IR Grafigi -> StableHLO SPMD -> PJRT C API derleme ve yurutme -> yerel yurutucu ile esdegerlik
  ir::Graph g;
  ir::Builder b(g);
  auto C = [](std::int64_t v) { return ir::Dim::constant(v); };
  const int x = b.param("x", {C(8), C(16)});
  const int w1 = b.param("w1", {C(16), C(32)});
  const int bias = b.param("bias", {C(32)});
  const int w2 = b.param("w2", {C(32), C(8)});

  const int h0 = b.matmul(x, w1);
  const int h1 = b.node(ir::Op::BiasAdd, {h0, bias}, {C(8), C(32)});
  const int h2 = b.unary(ir::Op::Tanh, h1);
  const int out = b.matmul(h2, w2);
  g.outputs = {out};
  g.output_names = {"y"};

  const std::string mlir = nb::pjrt::export_stablehlo_spmd(
      g, {},
      {{"x", sharding_rep}, {"w1", sharding_rep}, {"bias", sharding_rep}, {"w2", sharding_rep}},
      {sharding_rep});
  std::printf("  info  StableHLO SPMD modulu (%zu bayt):\n%s", mlir.size(), mlir.c_str());
  expect(mlir.find("mhlo.sharding") != std::string::npos &&
             mlir.find("stablehlo.dot_general") != std::string::npos &&
             mlir.find("stablehlo.broadcast_in_dim") != std::string::npos &&
             mlir.find("stablehlo.tanh") != std::string::npos,
         "StableHLO SPMD modulu mhlo.sharding, dot_general, broadcast_in_dim ve tanh icerir");

  const auto vx = make_data(8 * 16, 0.04f, 5);
  const auto vw1 = make_data(16 * 32, 0.03f, 11);
  const auto vbias = make_data(32, 0.02f, 17);
  const auto vw2 = make_data(32 * 8, 0.03f, 23);

  nb::pjrt::Client pjrt_client;
  expect(pjrt_client.platform_name() == "nexus-stablehlo-reference-cpu" && !pjrt_client.uses_external_plugin(),
         "CPU reference is explicitly distinguished from real XLA/PJRT");
  bool plugin_rejected = false;
  try { nb::pjrt::Client invalid("nonexistent-pjrt-plugin"); }
  catch (const nexus::TensorError&) { plugin_rejected = true; }
  expect(plugin_rejected, "Explicit plugin request never silently falls back to the interpreter");
  bool bad_shape_rejected = false;
  try { (void)pjrt_client.compile_and_execute(mlir, {{1.0f}}, {{8, 16}}); }
  catch (const nexus::TensorError&) { bad_shape_rejected = true; }
  expect(bad_shape_rejected, "Input dimensions are checked before host buffer copy");
  bool signature_rejected = false;
  try { (void)pjrt_client.compile_and_execute(mlir, {vx, vw1, vbias, vw2}, {{16, 8}, {16, 32}, {32}, {32, 8}}); }
  catch (const nexus::TensorError&) { signature_rejected = true; }
  expect(signature_rejected, "Equal-sized but differently shaped argument is rejected");
  const std::string malformed = R"MLIR(module @bad {
  func.func @main(%arg0: tensor<1xf32>) -> (tensor<4xf32>) {
    %v0 = stablehlo.negate %arg0 : tensor<4xf32>
    return %v0 : tensor<4xf32>
  }
})MLIR";
  bool operand_rejected = false;
  try { (void)pjrt_client.compile_and_execute(malformed, {{1.0f}}, {{1}}); }
  catch (const nexus::TensorError&) { operand_rejected = true; }
  expect(operand_rejected, "Malformed MLIR cannot make a unary kernel read beyond an input");
  ir::Graph sigmoid_graph;
  ir::Builder sb(sigmoid_graph);
  const int sx = sb.param("x", {C(4)});
  sigmoid_graph.outputs = {sb.unary(ir::Op::Sigmoid, sx)};
  const auto sy = pjrt_client.compile_and_execute(nb::pjrt::export_stablehlo_spmd(sigmoid_graph, {}),
                                                 {{-1.0f, 0.0f, 1.0f, 2.0f}}, {{4}});
  expect(std::abs(sy[0][1] - 0.5f) < 1e-6f && std::abs(sy[0][2] - 0.7310586f) < 1e-6f,
         "stablehlo.logistic is not incorrectly parsed as stablehlo.log");
  const auto pjrt_outs = pjrt_client.compile_and_execute(
      mlir, {vx, vw1, vbias, vw2}, {{8, 16}, {16, 32}, {32}, {32, 8}});

  ir::CpuJit jit(ir::default_cache_dir());
  nexus::compile::Executable native_exe(g, jit, {});
  const auto native_outs = native_exe.run(
      {}, {{"x", vx.data()}, {"w1", vw1.data()}, {"bias", vbias.data()}, {"w2", vw2.data()}}, {});

  const float diff = max_abs_diff(pjrt_outs[0], native_outs[0]);
  std::printf("  info  PJRT C API ciktisi vs yerel Executable max_abs=%.2e\n", diff);
  expect(diff < 1e-5f, "Grafik -> StableHLO -> PJRT C API yurutmesi yerel yurutme ile ayni sonucu uretir (< 1e-5)");
  if (export_path) {
    std::ofstream module(export_path);
    module << mlir;
    module.close();
    std::ofstream fixture(std::string(export_path) + ".json");
    fixture << std::setprecision(std::numeric_limits<float>::max_digits10);
    auto write_array = [&](const std::vector<float>& values) {
      fixture << '[';
      for (std::size_t i = 0; i < values.size(); ++i) fixture << (i ? "," : "") << values[i];
      fixture << ']';
    };
    fixture << "{\"shapes\":[[8,16],[16,32],[32],[32,8]],\"inputs\":[";
    write_array(vx); fixture << ','; write_array(vw1); fixture << ',';
    write_array(vbias); fixture << ','; write_array(vw2);
    fixture << "],\"expected\":["; write_array(native_outs[0]); fixture << "]}\n";
    fixture.close();
    if (!module || !fixture) throw nexus::TensorError("Cannot write independent XLA fixture");
  }

  // 5.3 PjrtDriver (DeviceRef::tpu(0)) uzerinden MirroredTensor + MatMul
  const DeviceRef tpu = DeviceRef::tpu(0);
  MirroredTensor dA(Shape{8, 16}, tpu), dB(Shape{16, 8}, tpu), dC(Shape{8, 8}, tpu);
  std::memcpy(dA.host_data(), vx.data(), 8 * 16 * sizeof(float));
  std::memcpy(dB.host_data(), vw2.data(), 16 * 8 * sizeof(float));
  dA.modify_host(); dB.modify_host();
  dA.sync_device(); dB.sync_device(); dC.sync_device();
  dC.modify_device();
  nb::matmul(tpu, {8, 8, 16, dA.device_data(), 16, false, dB.device_data(), 8, false, dC.device_data(), 8, 1.0f, 0.0f});
  dC.sync_host();

  std::vector<float> ref_tpu(64, 0.0f);
  nb::Registry::instance().require(nb::Kind::Cpu).matmul(
      DeviceRef::cpu(), {8, 8, 16, vx.data(), 16, false, vw2.data(), 8, false, ref_tpu.data(), 8, 1.0f, 0.0f});
  expect(max_abs_diff(std::vector<float>(dC.host_data(), dC.host_data() + 64), ref_tpu) < 1e-5f,
         "Device('tpu:0') PjrtDriver GEMM CPU referansi ile eslesir");
}

// ============================================================================
//  6) Benchmark Modu (tum etkin arka uclarda GEMM GFLOP/s ve egitim hizi)
// ============================================================================

void test_backend_fallback() {
  nexus_test::section("CPU fallback policy and opaque device buffer transfers");
  const auto dev = DeviceRef::tpu(0);  // Reference driver uses opaque allocation handles.
  auto& d = nb::Registry::instance().require(nb::Kind::Pjrt);
  const auto x = make_data(16, 0.03f);
  const std::vector<float> w(8, 1.0f), bias(8, 0.1f);
  using Call = std::function<void(DeviceRef, const std::vector<float*>&, float*)>;
  auto check = [&](nb::Operation op, const std::vector<std::vector<float>>& inputs,
                   std::size_t out_size, const Call& call) {
    std::vector<MirroredTensor> buffers;
    std::vector<float*> device_inputs, host_inputs;
    for (const auto& input : inputs) {
      buffers.emplace_back(Shape{input.size()}, dev);
      auto& buffer = buffers.back();
      std::memcpy(buffer.host_data(), input.data(), input.size() * sizeof(float));
      buffer.modify_host(); buffer.sync_device();
      device_inputs.push_back(buffer.device_data());
      host_inputs.push_back(const_cast<float*>(input.data()));
    }
    MirroredTensor output(Shape{out_size}, dev);
    output.sync_device();
    std::vector<float> reference(out_size);
    call(DeviceRef::cpu(), host_inputs, reference.data());
    nb::reset_fallback_stats();
    nb::set_fallback_policy(nb::FallbackPolicy::Error);
    bool rejected = false;
    try { call(dev, device_inputs, output.device_data()); }
    catch (const nexus::TensorError&) { rejected = true; }
    expect(rejected, std::string(nb::to_string(op)) + " rejects implicit CPU execution");
    nb::set_fallback_policy(nb::FallbackPolicy::WarnOnce);
    call(dev, device_inputs, output.device_data());
    output.modify_device(); output.sync_host();
    expect(max_abs_diff(std::vector<float>(output.host_data(), output.host_data() + out_size), reference) < 1e-5f,
           std::string(nb::to_string(op)) + " stages opaque buffers before CPU computation");
    expect(nb::fallback_stats(nb::Kind::Pjrt, op).calls == 1,
           std::string(nb::to_string(op)) + " counts exactly one fallback");
    nb::set_fallback_policy(nb::FallbackPolicy::Error);
  };
  check(nb::Operation::Softmax, {x}, 16, [](auto dv, const auto& p, float* o) { nb::softmax(dv, p[0], o, 2, 8); });
  check(nb::Operation::RmsNorm, {x, w}, 16, [](auto dv, const auto& p, float* o) { nb::rms_norm(dv, p[0], p[1], o, 2, 8, 1e-5f); });
  check(nb::Operation::LayerNorm, {x, w, bias}, 16, [](auto dv, const auto& p, float* o) { nb::layer_norm(dv, p[0], p[1], p[2], o, 2, 8, 1e-5f); });
  check(nb::Operation::Rope, {x}, 16, [](auto dv, const auto& p, float* o) { nb::rope(dv, p[0], o, 1, 2, 8, 0, 10000.0f); });
  check(nb::Operation::SwiGlu, {x}, 8, [](auto dv, const auto& p, float* o) { nb::swiglu(dv, p[0], o, 2, 4); });
  check(nb::Operation::Attention, {x, x, x}, 16, [](auto dv, const auto& p, float* o) { nb::attention(dv, p[0], p[1], p[2], o, 1, 2, 8, true); });
  std::vector<float> packed8(16), packed4(8);
  std::memset(packed8.data(), 3, packed8.size() * sizeof(float));
  std::memset(packed4.data(), 0x21, packed4.size() * sizeof(float));
  const auto qa = make_data(64);
  check(nb::Operation::QuantMatMulQ8, {qa, packed8, {0.125f, 0.25f}}, 4,
        [](auto dv, const auto& p, float* o) { nb::quant_matmul_q8(dv, p[0], p[1], p[2], o, 2, 2, 32, 32); });
  check(nb::Operation::QuantMatMulQ4, {qa, packed4, {0.125f, 0.25f}}, 4,
        [](auto dv, const auto& p, float* o) { nb::quant_matmul_q4(dv, p[0], p[1], p[2], o, 2, 2, 32, 32); });
  {
    std::vector<float> hp = x, hm(16, 0.0f), hv(16, 0.0f);
    MirroredTensor dp(Shape{16}, dev), dg(Shape{16}, dev), dm(Shape{16}, dev), dv(Shape{16}, dev);
    std::memcpy(dp.host_data(), hp.data(), hp.size() * sizeof(float)); dp.modify_host();
    std::memcpy(dg.host_data(), x.data(), x.size() * sizeof(float)); dg.modify_host();
    dp.sync_device(); dg.sync_device(); dm.sync_device(); dv.sync_device();
    nb::adamw_step(DeviceRef::cpu(), hp.data(), x.data(), hm.data(), hv.data(), 16, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
    nb::set_fallback_policy(nb::FallbackPolicy::WarnOnce);
    nb::reset_fallback_stats();
    nb::adamw_step(dev, dp.device_data(), dg.device_data(), dm.device_data(), dv.device_data(), 16,
                  0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
    dp.modify_device(); dm.modify_device(); dv.modify_device();
    dp.sync_host(); dm.sync_host(); dv.sync_host();
    expect(max_abs_diff(std::vector<float>(dp.host_data(), dp.host_data() + 16), hp) < 1e-6f &&
           max_abs_diff(std::vector<float>(dm.host_data(), dm.host_data() + 16), hm) < 1e-6f &&
           max_abs_diff(std::vector<float>(dv.host_data(), dv.host_data() + 16), hv) < 1e-6f,
           "AdamW fallback returns updated parameters AND both moments to device");
    expect(nb::fallback_stats(nb::Kind::Pjrt, nb::Operation::AdamW).calls == 1, "AdamW fallback counted exactly once");
    nb::set_fallback_policy(nb::FallbackPolicy::Error);
  }
  // Direct Driver calls must enforce the policy as well as the free dispatchers.
  bool direct_rejected = false;
  try { d.softmax(dev, nullptr, nullptr, 2, 8); }
  catch (const nexus::TensorError&) { direct_rejected = true; }
  expect(direct_rejected, "Direct Driver call checks Error policy before touching buffers");
  bool invalid_device_rejected = false;
  try { d.matmul(DeviceRef::tpu(1), {}); }
  catch (const nexus::TensorError&) { invalid_device_rejected = true; }
  expect(invalid_device_rejected, "PJRT facade rejects an unexposed device instead of silently executing on device 0");
  expect(!d.supports(nb::Operation::Softmax), "Reference driver does not advertise unimplemented native operations");
  for (auto kind : {nb::Kind::Cuda, nb::Kind::DirectMl}) {
    auto& driver = nb::Registry::instance().require(kind);
    expect(!driver.supports(nb::Operation::Add) && !driver.supports(nb::Operation::Attention),
           std::string(nb::to_string(kind)) + " does not advertise host staging as native GPU kernels");
    if (driver.available()) {
      bool refused = false;
      try { driver.elementwise(nb::Operation::Add, DeviceRef{nb::to_device(kind), 0}, nullptr, nullptr, nullptr, 4, 1.0f); }
      catch (const nexus::TensorError&) { refused = true; }
      expect(refused, std::string(nb::to_string(kind)) + " direct host fallback enforces Error policy");
    }
  }
  const auto before = d.allocator_stats(0);
  void* odd_bytes = d.allocate(0, 3);
  bool copy_bounds_rejected = false;
  try { d.copy(dev, odd_bytes, DeviceRef::cpu(), x.data(), sizeof(float)); }
  catch (const nexus::TensorError&) { copy_bounds_rejected = true; }
  expect(copy_bounds_rejected, "Opaque reference allocation rejects out-of-bounds copies");
  d.deallocate(0, odd_bytes);
  const auto after = d.allocator_stats(0);
  expect(before.current_bytes == after.current_bytes && before.reserved_bytes == after.reserved_bytes,
         "Reference allocator accounts for non-float-aligned allocations exactly");
  if (nb::Registry::instance().require(nb::Kind::Vulkan).available()) {
    MirroredTensor src(Shape{16}, dev), dst(Shape{16}, DeviceRef::vulkan(0)), roundtrip(Shape{16}, dev);
    std::memcpy(src.host_data(), x.data(), x.size() * sizeof(float)); src.modify_host(); src.sync_device();
    dst.sync_device(); roundtrip.sync_device();
    nb::memory_copy(DeviceRef::vulkan(0), dst.device_data(), dev, src.device_data(), x.size() * sizeof(float));
    nb::memory_copy(dev, roundtrip.device_data(), DeviceRef::vulkan(0), dst.device_data(), x.size() * sizeof(float));
    roundtrip.modify_device(); roundtrip.sync_host();
    expect(max_abs_diff(std::vector<float>(roundtrip.host_data(), roundtrip.host_data() + 16), x) == 0,
           "Cross-backend copies stage bytes instead of reinterpreting opaque handles");
  }
}

void test_pjrt_vendor(const char* path) {
  // This target uses the supplied vendor library, never the contract-test DLL.
  nb::pjrt::Client client(path);
  ir::Graph graph;
  ir::Builder b(graph);
  auto C = [](std::int64_t v) { return ir::Dim::constant(v); };
  const int x = b.param("x", {C(2), C(3)}), w = b.param("w", {C(3), C(2)});
  graph.outputs = {b.unary(ir::Op::Tanh, b.matmul(x, w))};
  const auto result = client.compile_and_execute(nb::pjrt::export_stablehlo_spmd(graph, {}),
      {{1,2,3,4,5,6}, {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f}}, {{2,3},{3,2}});
  const std::vector<float> ref = {std::tanh(2.2f), std::tanh(2.8f), std::tanh(4.9f), std::tanh(6.4f)};
  expect(client.uses_external_plugin() && result.size() == 1 && max_abs_diff(result[0], ref) < 1e-5f,
         "Vendor PJRT plugin compiles/executes the actual Nexus StableHLO export");
  std::printf("  info  vendor platform=%s version=%s device=%s\n", std::string(client.platform_name()).c_str(),
              std::string(client.platform_version()).c_str(), client.device_name(0).c_str());
}

void run_benchmarks() {
  const int M = 512, N = 512, K = 512;
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  const auto hA = make_data(M * K, 0.02f, 7);
  const auto hB = make_data(K * N, 0.02f, 13);

  for (nb::Driver* d : nb::Registry::instance().all()) {
    if (!d->available()) continue;
    for (const nb::DeviceInfo& info : d->devices()) {
      const DeviceRef dev = info.device;
      void* a = d->allocate(dev.index, hA.size() * sizeof(float));
      void* b = d->allocate(dev.index, hB.size() * sizeof(float));
      void* c = d->allocate(dev.index, static_cast<std::size_t>(M * N) * sizeof(float));
      d->copy(dev, a, DeviceRef::cpu(), hA.data(), hA.size() * sizeof(float));
      d->copy(dev, b, DeviceRef::cpu(), hB.data(), hB.size() * sizeof(float));

      nb::MatMulArgs mm{M, N, K, static_cast<const float*>(a), K, false,
                        static_cast<const float*>(b), N, false, static_cast<float*>(c), N, 1.0f, 0.0f};
      // Separate warmup from the measured repetitions.
      for (int warm = 0; warm < 3; ++warm) {
        d->matmul(dev, mm);
        d->synchronize(dev.index);
      }

      std::vector<double> samples;
      for (int r = 0; r < 7; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        d->matmul(dev, mm);
        d->synchronize(dev.index);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        samples.push_back(ms);
      }
      auto sorted = samples;
      std::sort(sorted.begin(), sorted.end());
      const double median = sorted[sorted.size() / 2];
      const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
      double variance = 0.0;
      for (const double s : samples) variance += (s - mean) * (s - mean);
      const double stddev = std::sqrt(variance / samples.size());
      const double gflops = flops / (median * 1e6);
      using nexus::interop::json_quote;
      std::cout << std::setprecision(12) << "{\"backend\":" << json_quote(nb::to_string(d->kind()))
                << ",\"implementation\":" << json_quote(d->name())
                << ",\"version\":" << json_quote(d->version())
                << ",\"api_version\":" << json_quote(info.api_version)
                << ",\"driver_version\":" << json_quote(info.driver_version)
                << ",\"device\":" << json_quote(dev.to_string()) << ",\"name\":" << json_quote(info.name)
                << ",\"warmup\":3,\"repetitions\":7,\"shape\":[512,512,512],\"dtype\":\"f32\",\"samples_ms\":[";
      for (std::size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
      std::cout << "],\"median_ms\":" << median << ",\"min_ms\":" << sorted.front()
                << ",\"max_ms\":" << sorted.back() << ",\"std_ms\":" << stddev
                << ",\"gemm_512_ms\":" << median << ",\"gflops\":" << gflops << "}\n";

      d->deallocate(dev.index, a);
      d->deallocate(dev.index, b);
      d->deallocate(dev.index, c);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    if (mode != "all" && mode != "bench" && mode != "detection" && mode != "hip" && mode != "hip-source" &&
        mode != "vulkan" && mode != "directml" && mode != "pjrt" && mode != "fallback" && mode != "export-xla" && mode != "vendor")
      throw nexus::TensorError("Unknown Phase 19 test mode: " + mode);
    if (mode == "export-xla" && argc != 3)
      throw nexus::TensorError("export-xla requires a writable MLIR output path");
    if (mode == "vendor") {
      if (argc < 3 || argv[2][0] == '\0') {
        std::printf("  SKIP  Supply a trusted vendor library with -DNEXUS_PJRT_TEST_PLUGIN_PATH=<path>\n");
        return 77;
      }
      test_pjrt_vendor(argv[2]);
      return nexus_test::finish("vendor PJRT execution");
    }
    if (mode == "bench") {
      run_benchmarks();
      return 0;
    }
    if (mode == "all" || mode == "detection") test_backend_detection();
    if (mode == "hip") {
      std::printf("  SKIP  Native HIP execution/parity is not implemented; use hip-source for static compatibility checks.\n");
      return 77;
    }
    if (mode == "all" || mode == "hip-source") test_hip_source_compat();
    if (mode == "all" || mode == "vulkan") test_vulkan_parity();
    if (mode == "all" || mode == "directml") test_directml_parity();
    if (mode == "all" || mode == "fallback") test_backend_fallback();
    if (mode == "all" || mode == "pjrt" || mode == "export-xla")
      test_pjrt_export(mode == "export-xla" && argc > 2 ? argv[2] : nullptr);
    if (mode == "vulkan" && !nb::Registry::instance().require(nb::Kind::Vulkan).available()) return 77;
    if (mode == "directml" && !nb::Registry::instance().require(nb::Kind::DirectMl).available()) return 77;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: yakalanmamis istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase19_backends");
}
