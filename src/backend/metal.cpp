/**
 * @file metal.cpp
 * @brief FAZ 19.D: sample MSL sources and explicit unavailable driver.
 *        Native allocation/command submission is not implemented; no zero-copy claim.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/backend/backend.hpp"
#include "nexus/core/error.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace nexus::backend {

namespace {

constexpr std::string_view kMetalMslShaders = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void nexus_msl_elementwise(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& scalar [[buffer(4)]],
    constant uint& op [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  float x = a[gid];
  float y = b[gid];
  switch (op) {
    case 0: out[gid] = x + y; break;
    case 1: out[gid] = x - y; break;
    case 2: out[gid] = x * y; break;
    case 3: out[gid] = x / y; break;
    case 4: out[gid] = -x; break;
    case 5: out[gid] = exp(x); break;
    case 6: out[gid] = log(x); break;
    case 7: out[gid] = sqrt(x); break;
    case 8: out[gid] = rsqrt(x); break;
    case 9: out[gid] = tanh(x); break;
    case 10: out[gid] = 1.0f / (1.0f + exp(-x)); break;
    case 11: out[gid] = x / (1.0f + exp(-x)); break;
    case 12: out[gid] = max(x, 0.0f); break;
    case 13: out[gid] = x * scalar; break;
    case 14: out[gid] = fma(x, scalar, y); break;
    default: out[gid] = x; break;
  }
}

kernel void nexus_msl_gemm_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],  // m, n, k, flags
    constant float2& ab [[buffer(4)]],   // alpha, beta
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {
  threadgroup float ta[16][17];
  threadgroup float tb[16][17];
  uint m = dims.x, n = dims.y, k = dims.z;
  float sum = 0.0f;
  for (uint tile = 0; tile < k; tile += 16) {
    uint ak = tile + lid.x;
    uint bk = tile + lid.y;
    ta[lid.y][lid.x] = (gid.y < m && ak < k) ? a[gid.y * k + ak] : 0.0f;
    tb[lid.y][lid.x] = (gid.x < n && bk < k) ? b[bk * n + gid.x] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 16; ++j) sum = fma(ta[lid.y][j], tb[j][lid.x], sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (gid.y < m && gid.x < n) {
    uint idx = gid.y * n + gid.x;
    c[idx] = (ab.y == 0.0f) ? (ab.x * sum) : fma(ab.x, sum, ab.y * c[idx]);
  }
}
)MSL";

class MetalDriver final : public Driver {
 public:
  MetalDriver() {
#if defined(__APPLE__)
    reason_ = "Metal execution backend (Objective-C++ allocator/command queue) uygulanmadi; MSL kaynaklari tek basina calistirici degildir";
#else
    reason_ = "Apple Metal (MPS/MSL) yalniz macOS / Apple Silicon (M1-M4) uzerinde kullanilabilir "
              "(yalniz ornek MSL kaynaklari: " + std::to_string(kMetalMslShaders.size()) + " bayt; execution backend uygulanmadi)";
#endif
  }

  Kind kind() const noexcept override { return Kind::Metal; }
  std::string_view name() const noexcept override { return "metal-msl"; }
  std::string_view version() const noexcept override { return "0"; }
  const std::vector<DeviceInfo>& devices() const noexcept override { return devices_; }
  std::string unavailable_reason() const override { return reason_; }
  bool supports(Operation) const noexcept override { return false; }

  void* allocate(int, std::size_t) override { fail(); }
  void deallocate(int, void*) noexcept override {}
  void copy(DeviceRef, void*, DeviceRef, const void*, std::size_t) override { fail(); }
  void synchronize(int) override { fail(); }
  void elementwise(Operation, DeviceRef, const float*, const float*, float*, std::size_t, float) override { fail(); }
  void matmul(DeviceRef, const MatMulArgs&) override { fail(); }
  void reduce_sum(DeviceRef, const float*, float*, std::size_t) override { fail(); }

 private:
  [[noreturn]] void fail() const { throw TensorError("Nexus Metal: " + reason_); }
  std::vector<DeviceInfo> devices_;
  std::string reason_;
};

}  // namespace

std::unique_ptr<Driver> make_metal_driver() { return std::make_unique<MetalDriver>(); }

}  // namespace nexus::backend
