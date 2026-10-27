/**
 * @file registry.cpp
 * @brief FAZ 19.A: Ortak Backend fabrikasi, ozellik algilama, geri dusus politikasi (Error, WarnOnce,
 *        WarnAlways), AVX2 + ThreadPool CpuDriver ve dinamik nvcuda.dll + PTX/cuBLAS CudaDriver.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/backend/backend.hpp"

#include "nexus/attention/cpu.hpp"
#include "nexus/core/allocator.hpp"
#include "nexus/core/error.hpp"
#include "nexus/cpu/adam.hpp"
#include "nexus/cpu/features.hpp"
#include "nexus/cpu/gemm.hpp"
#include "nexus/cpu/threads.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nexus::backend {
namespace {

constexpr std::size_t kKindCount = 7;
constexpr std::size_t kOpCount = 27;

std::atomic<FallbackPolicy> g_policy{FallbackPolicy::Error};
std::mutex g_fallback_mutex;
std::array<std::array<FallbackStats, kOpCount>, kKindCount> g_fallback_stats{};
std::array<std::array<bool, kOpCount>, kKindCount> g_warned{};

std::string lower_copy(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
  std::string out;
  out.reserve(text.size());
  for (char ch : text) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  return out;
}

float apply_unary(Operation op, float x, float scalar) noexcept {
  switch (op) {
    case Operation::Neg: return -x;
    case Operation::Exp: return std::exp(x);
    case Operation::Log: return std::log(x);
    case Operation::Sqrt: return std::sqrt(x);
    case Operation::Rsqrt: return 1.0f / std::sqrt(x);
    case Operation::Tanh: return std::tanh(x);
    case Operation::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
    case Operation::Silu: return x / (1.0f + std::exp(-x));
    case Operation::Relu: return x > 0.0f ? x : 0.0f;
    case Operation::Scale: return x * scalar;
    default: return x;
  }
}

void record_fallback(Driver& driver, Operation op, std::size_t elements) {
  const FallbackPolicy policy = g_policy.load(std::memory_order_relaxed);
  const auto k = static_cast<std::size_t>(driver.kind());
  const auto o = static_cast<std::size_t>(op);
  if (policy == FallbackPolicy::Error) {
    throw TensorError(std::string("Nexus backend '") + std::string(driver.name()) + "' (" +
                      to_string(driver.kind()) + ") '" + to_string(op) + "' islemini desteklemiyor (" +
                      driver.unavailable_reason() + "); geri dusus politikasi=Error");
  }
  bool should_warn = (policy == FallbackPolicy::WarnAlways);
  {
    std::lock_guard lock(g_fallback_mutex);
    if (k < kKindCount && o < kOpCount) {
      ++g_fallback_stats[k][o].calls;
      g_fallback_stats[k][o].elements += elements;
      if (policy == FallbackPolicy::WarnOnce && !g_warned[k][o]) {
        g_warned[k][o] = true;
        should_warn = true;
      }
    }
  }
  if (should_warn) {
    std::fprintf(stderr, "[Nexus UYARI] backend=%s op=%s (%zu eleman) CPU'ya geri dusuyor (%s)\n",
                 to_string(driver.kind()), to_string(op), elements, driver.unavailable_reason().c_str());
  }
}

// ============================================================================
//  CpuDriver (AVX2 + FMA + ThreadPool)
// ============================================================================

class CpuDriver final : public Driver {
 public:
  CpuDriver() {
    const auto& f = cpu::features();
    DeviceInfo info;
    info.device = DeviceRef::cpu();
    info.name = f.brand.empty() ? "Host CPU" : f.brand;
    info.vendor = f.vendor.empty() ? "CPU" : f.vendor;
    info.api_version = f.avx512f ? "AVX-512" : (f.avx2 ? "AVX2+FMA" : "Scalar");
    info.driver_version = "1.0";
    info.memory_bytes = 0;
    info.subgroup_size = f.avx512f ? 16u : (f.avx2 ? 8u : 1u);
    info.unified_memory = true;
    devices_.push_back(std::move(info));
  }

  Kind kind() const noexcept override { return Kind::Cpu; }
  std::string_view name() const noexcept override { return "cpu"; }
  std::string_view version() const noexcept override { return "1.0"; }
  const std::vector<DeviceInfo>& devices() const noexcept override { return devices_; }
  std::string unavailable_reason() const override { return {}; }
  bool supports(Operation op) const noexcept override {
    return static_cast<unsigned>(op) <= static_cast<unsigned>(Operation::AdamW);
  }

  void* allocate(int, std::size_t bytes) override {
    void* ptr = host_allocator().allocate(bytes);
    if (ptr != nullptr) {
      std::lock_guard lock(alloc_mutex_);
      alloc_sizes_[ptr] = bytes;
    }
    return ptr;
  }
  void deallocate(int, void* pointer) noexcept override {
    if (pointer != nullptr) {
      std::size_t bytes = 0;
      {
        std::lock_guard lock(alloc_mutex_);
        auto it = alloc_sizes_.find(pointer);
        if (it != alloc_sizes_.end()) {
          bytes = it->second;
          alloc_sizes_.erase(it);
        }
      }
      if (bytes > 0) host_allocator().deallocate(pointer, bytes);
    }
  }
  void copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src, std::size_t bytes) override {
    if (bytes == 0) return;
    if (!dst_device.is_host() || !src_device.is_host()) {
      throw TensorError("Nexus CPU backend: cihazlar arasi kopya yalniz host bellegi destekler");
    }
    std::memcpy(dst, src, bytes);
  }
  void synchronize(int) override {}

  void elementwise(Operation op, DeviceRef, const float* a, const float* b, float* out,
                   std::size_t count, float scalar) override {
    if (count == 0) return;
    if (a == nullptr || out == nullptr) throw TensorError("Nexus CPU: elementwise bos isaretci");
    if ((op <= Operation::Div || op == Operation::Axpy) && b == nullptr) {
      throw TensorError("Nexus CPU: ikili elementwise isleminde ikinci girdi bos");
    }
    auto worker = [&](std::size_t begin, std::size_t end) {
      switch (op) {
        case Operation::Add:
          for (std::size_t i = begin; i < end; ++i) out[i] = a[i] + b[i];
          break;
        case Operation::Sub:
          for (std::size_t i = begin; i < end; ++i) out[i] = a[i] - b[i];
          break;
        case Operation::Mul:
          for (std::size_t i = begin; i < end; ++i) out[i] = a[i] * b[i];
          break;
        case Operation::Div:
          for (std::size_t i = begin; i < end; ++i) out[i] = a[i] / b[i];
          break;
        case Operation::Axpy:
          for (std::size_t i = begin; i < end; ++i) out[i] = a[i] * scalar + b[i];
          break;
        default:
          for (std::size_t i = begin; i < end; ++i) out[i] = apply_unary(op, a[i], scalar);
          break;
      }
    };
    if (count >= 32768) {
      cpu::Pool::instance().run(count, [&](int /*w*/, std::size_t b_idx, std::size_t e_idx) {
        worker(b_idx, e_idx);
      });
    } else {
      worker(0, count);
    }
  }

  void matmul(DeviceRef, const MatMulArgs& args) override {
    if (args.m <= 0 || args.n <= 0 || args.k < 0) throw TensorError("Nexus CPU: gecersiz GEMM boyutu");
    if (args.a == nullptr || args.b == nullptr || args.c == nullptr) {
      throw TensorError("Nexus CPU: GEMM bos isaretci");
    }
    cpu::GemmArgs ga;
    ga.M = args.m;
    ga.N = args.n;
    ga.K = args.k;
    ga.A = args.a;
    ga.lda = args.lda;
    ga.a_trans = args.a_trans;
    ga.B = args.b;
    ga.ldb = args.ldb;
    ga.btype = cpu::BType::F32;
    ga.b_trans = args.b_trans;
    ga.C = args.c;
    ga.ldc = args.ldc;
    ga.alpha = args.alpha;
    ga.beta = args.beta;
    cpu::gemm(ga);
  }

  void reduce_sum(DeviceRef, const float* input, float* output, std::size_t count) override {
    if (output == nullptr || (count > 0 && input == nullptr)) {
      throw TensorError("Nexus CPU: reduce_sum bos isaretci");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) sum += static_cast<double>(input[i]);
    *output = static_cast<float>(sum);
  }

  void softmax(DeviceRef, const float* input, float* output, int rows, int cols) override {
    for (int r = 0; r < rows; ++r) {
      const float* x = input + static_cast<std::size_t>(r) * cols;
      float* y = output + static_cast<std::size_t>(r) * cols;
      float mx = x[0];
      for (int c = 1; c < cols; ++c) mx = std::max(mx, x[c]);
      double sum = 0.0;
      for (int c = 0; c < cols; ++c) {
        y[c] = std::exp(x[c] - mx);
        sum += static_cast<double>(y[c]);
      }
      const float inv = static_cast<float>(1.0 / sum);
      for (int c = 0; c < cols; ++c) y[c] *= inv;
    }
  }

  void rms_norm(DeviceRef, const float* x, const float* weight, float* out,
                int rows, int cols, float eps) override {
    for (int r = 0; r < rows; ++r) {
      const float* xr = x + static_cast<std::size_t>(r) * cols;
      float* yr = out + static_cast<std::size_t>(r) * cols;
      double sq = 0.0;
      for (int c = 0; c < cols; ++c) sq += static_cast<double>(xr[c]) * xr[c];
      const float rrms = 1.0f / std::sqrt(static_cast<float>(sq / cols) + eps);
      for (int c = 0; c < cols; ++c) yr[c] = (xr[c] * rrms) * weight[c];
    }
  }

  void layer_norm(DeviceRef, const float* x, const float* weight, const float* bias,
                  float* out, int rows, int cols, float eps) override {
    for (int r = 0; r < rows; ++r) {
      const float* xr = x + static_cast<std::size_t>(r) * cols;
      float* yr = out + static_cast<std::size_t>(r) * cols;
      double s = 0.0, sq = 0.0;
      for (int c = 0; c < cols; ++c) {
        s += xr[c];
        sq += static_cast<double>(xr[c]) * xr[c];
      }
      const float mean = static_cast<float>(s / cols);
      const float var = std::max(0.0f, static_cast<float>(sq / cols - static_cast<double>(mean) * mean));
      const float inv = 1.0f / std::sqrt(var + eps);
      for (int c = 0; c < cols; ++c) {
        yr[c] = ((xr[c] - mean) * inv) * weight[c] + (bias ? bias[c] : 0.0f);
      }
    }
  }

  void rope(DeviceRef, const float* x, float* out, int batch_heads, int seq_len,
            int head_dim, int pos_offset, float theta) override {
    const int half_d = head_dim / 2;
    for (int bh = 0; bh < batch_heads; ++bh) {
      for (int t = 0; t < seq_len; ++t) {
        const int pos = t + pos_offset;
        const std::size_t base = (static_cast<std::size_t>(bh) * seq_len + t) * head_dim;
        for (int p = 0; p < half_d; ++p) {
          const float exponent = static_cast<float>(2 * p) / static_cast<float>(head_dim);
          const float freq = static_cast<float>(pos) / std::pow(theta, exponent);
          const float c = std::cos(freq);
          const float s = std::sin(freq);
          const float x0 = x[base + 2 * p];
          const float x1 = x[base + 2 * p + 1];
          out[base + 2 * p] = x0 * c - x1 * s;
          out[base + 2 * p + 1] = x0 * s + x1 * c;
        }
      }
    }
  }

  void swiglu(DeviceRef, const float* x, float* out, int rows, int hidden_dim) override {
    for (int r = 0; r < rows; ++r) {
      const float* xr = x + static_cast<std::size_t>(r) * (2 * hidden_dim);
      float* yr = out + static_cast<std::size_t>(r) * hidden_dim;
      for (int c = 0; c < hidden_dim; ++c) {
        const float gate = xr[c];
        const float up = xr[hidden_dim + c];
        yr[c] = (gate / (1.0f + std::exp(-gate))) * up;
      }
    }
  }

  void attention(DeviceRef, const float* q, const float* k, const float* v, float* out,
                 int batch_heads, int seq_len, int head_dim, bool causal) override {
    AttentionParams p;
    p.batch = batch_heads;
    p.heads = 1;
    p.kv_heads = 1;
    p.tq = seq_len;
    p.tk = seq_len;
    p.head_dim = head_dim;
    p.dtype = DType::F32;
    p.causal = causal;
    p.set_contiguous_bthd();
    std::vector<float> lse(static_cast<std::size_t>(batch_heads) * seq_len);
    attention::flash_attention_forward_cpu(out, lse.data(), q, k, v, p);
  }

  void quant_matmul_q8(DeviceRef, const float* a, const void* w_q8,
                       const float* w_scales, float* c, int m, int n, int k, int group_size) override {
    const auto* wq = static_cast<const std::int8_t*>(w_q8);
    const int groups_per_row = k / group_size;
    for (int row = 0; row < m; ++row) {
      const float* a_row = a + static_cast<std::size_t>(row) * k;
      for (int col = 0; col < n; ++col) {
        const std::int8_t* w_row = wq + static_cast<std::size_t>(col) * k;
        const float* s_row = w_scales + static_cast<std::size_t>(col) * groups_per_row;
        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; ++g) {
          float gsum = 0.0f;
          const int k0 = g * group_size;
          for (int j = 0; j < group_size; ++j) {
            gsum += a_row[k0 + j] * static_cast<float>(w_row[k0 + j]);
          }
          acc += gsum * s_row[g];
        }
        c[static_cast<std::size_t>(row) * n + col] = acc;
      }
    }
  }

  void quant_matmul_q4(DeviceRef, const float* a, const void* w_q4,
                       const float* w_scales, float* c, int m, int n, int k, int group_size) override {
    const auto* wq = static_cast<const std::uint8_t*>(w_q4);
    const int groups_per_row = k / group_size;
    for (int row = 0; row < m; ++row) {
      const float* a_row = a + static_cast<std::size_t>(row) * k;
      for (int col = 0; col < n; ++col) {
        const float* s_row = w_scales + static_cast<std::size_t>(col) * groups_per_row;
        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; ++g) {
          float gsum = 0.0f;
          const int k0 = g * group_size;
          for (int j = 0; j < group_size; ++j) {
            const std::size_t flat_idx = static_cast<std::size_t>(col) * k + k0 + j;
            const std::uint8_t byte = wq[flat_idx >> 1u];
            const int nibble = (flat_idx & 1u) ? ((byte >> 4u) & 0xFu) : (byte & 0xFu);
            const int q = nibble - 8;
            gsum += a_row[k0 + j] * static_cast<float>(q);
          }
          acc += gsum * s_row[g];
        }
        c[static_cast<std::size_t>(row) * n + col] = acc;
      }
    }
  }

  void adamw_step(DeviceRef, float* param, const float* grad, float* m, float* v,
                  std::size_t count, float lr, float beta1, float beta2, float eps,
                  float weight_decay, int step) override {
    const cpu::AdamCoeffs coeffs = cpu::adam_coeffs(lr, beta1, beta2, eps, weight_decay, step);
    cpu::adamw_range(param, m, v, grad, cpu::GradType::F32, 1.0f, coeffs,
                     weight_decay != 0.0f, nullptr, 0, count);
  }

  AllocatorStats allocator_stats(int) const override {
    const auto st = host_allocator().stats();
    AllocatorStats out;
    out.allocations = static_cast<std::size_t>(st.os_allocations);
    out.cache_hits = static_cast<std::size_t>(st.cache_hits);
    out.live_blocks = static_cast<std::size_t>(st.live_blocks);
    out.current_bytes = static_cast<std::size_t>(st.current_bytes);
    out.reserved_bytes = static_cast<std::size_t>(st.reserved_bytes);
    return out;
  }

 private:
  std::vector<DeviceInfo> devices_;
  mutable std::mutex alloc_mutex_;
  std::unordered_map<void*, std::size_t> alloc_sizes_;
};

// ============================================================================
//  CudaDriver (Dinamik nvcuda.dll + cublas64_*.dll surucusu)
// ============================================================================

using CUdevice = int;
using CUcontext = void*;
using CUdeviceptr = std::uint64_t;
using CUresult = int;
using cublasHandle_t = void*;

struct CudaDynApi {
  void* nvcuda = nullptr;
  void* cublas = nullptr;
  CUresult (*cuInit)(unsigned int) = nullptr;
  CUresult (*cuDriverGetVersion)(int*) = nullptr;
  CUresult (*cuDeviceGetCount)(int*) = nullptr;
  CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
  CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
  CUresult (*cuDeviceTotalMem_v2)(std::size_t*, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
  CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
  CUresult (*cuCtxSynchronize)() = nullptr;
  CUresult (*cuMemAlloc_v2)(CUdeviceptr*, std::size_t) = nullptr;
  CUresult (*cuMemFree_v2)(CUdeviceptr) = nullptr;
  CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void*, std::size_t) = nullptr;
  CUresult (*cuMemcpyDtoH_v2)(void*, CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cuMemcpyDtoD_v2)(CUdeviceptr, CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cuMemsetD32_v2)(CUdeviceptr, unsigned int, std::size_t) = nullptr;
  int (*cublasCreate_v2)(cublasHandle_t*) = nullptr;
  int (*cublasDestroy_v2)(cublasHandle_t) = nullptr;
  int (*cublasSgemm_v2)(cublasHandle_t, int, int, int, int, int, const float*,
                        const float*, int, const float*, int, const float*, float*, int) = nullptr;
};

void* dyn_open(const char* name) {
#if defined(_WIN32)
  return static_cast<void*>(LoadLibraryA(name));
#else
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* dyn_sym(void* lib, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

CudaDynApi& cuda_dyn() {
  static CudaDynApi api = [] {
    CudaDynApi a;
    const char* cuda_libs[] = {"nvcuda.dll", "libcuda.so.1", "libcuda.so"};
    for (const char* name : cuda_libs) {
      if ((a.nvcuda = dyn_open(name)) != nullptr) break;
    }
    if (!a.nvcuda) return a;
#define LOAD_CU(fn) a.fn = reinterpret_cast<decltype(a.fn)>(dyn_sym(a.nvcuda, #fn))
    LOAD_CU(cuInit);
    LOAD_CU(cuDriverGetVersion);
    LOAD_CU(cuDeviceGetCount);
    LOAD_CU(cuDeviceGet);
    LOAD_CU(cuDeviceGetName);
    LOAD_CU(cuDeviceTotalMem_v2);
    LOAD_CU(cuDevicePrimaryCtxRetain);
    LOAD_CU(cuCtxSetCurrent);
    LOAD_CU(cuCtxSynchronize);
    LOAD_CU(cuMemAlloc_v2);
    LOAD_CU(cuMemFree_v2);
    LOAD_CU(cuMemcpyHtoD_v2);
    LOAD_CU(cuMemcpyDtoH_v2);
    LOAD_CU(cuMemcpyDtoD_v2);
    LOAD_CU(cuMemsetD32_v2);
#undef LOAD_CU
    const char* blas_libs[] = {
        "cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64\\cublas64_13.dll",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\cublas64_13.dll",
        "libcublas.so.13", "libcublas.so.12", "libcublas.so"
    };
    for (const char* name : blas_libs) {
      if ((a.cublas = dyn_open(name)) != nullptr) break;
    }
    if (a.cublas) {
      a.cublasCreate_v2 = reinterpret_cast<decltype(a.cublasCreate_v2)>(dyn_sym(a.cublas, "cublasCreate_v2"));
      a.cublasDestroy_v2 = reinterpret_cast<decltype(a.cublasDestroy_v2)>(dyn_sym(a.cublas, "cublasDestroy_v2"));
      a.cublasSgemm_v2 = reinterpret_cast<decltype(a.cublasSgemm_v2)>(dyn_sym(a.cublas, "cublasSgemm_v2"));
    }
    return a;
  }();
  return api;
}

class CudaDriver final : public Driver {
 public:
  CudaDriver() {
    CudaDynApi& a = cuda_dyn();
    if (!a.nvcuda || !a.cuInit || a.cuInit(0) != 0) {
      reason_ = "NVIDIA CUDA surucusu (nvcuda.dll / libcuda.so) baslatilamadi";
      return;
    }
    int count = 0;
    if (!a.cuDeviceGetCount || a.cuDeviceGetCount(&count) != 0 || count <= 0) {
      reason_ = "CUDA uyumlu NVIDIA GPU bulunamadi";
      return;
    }
    int drv_ver = 0;
    if (a.cuDriverGetVersion) a.cuDriverGetVersion(&drv_ver);
    version_ = std::to_string(drv_ver / 1000) + "." + std::to_string((drv_ver % 1000) / 10);

    for (int i = 0; i < count; ++i) {
      CUdevice dev = 0;
      if (a.cuDeviceGet(&dev, i) != 0) continue;
      CUcontext ctx = nullptr;
      if (a.cuDevicePrimaryCtxRetain(&ctx, dev) != 0) continue;
      a.cuCtxSetCurrent(ctx);

      char dev_name[256] = {};
      a.cuDeviceGetName(dev_name, sizeof(dev_name), dev);
      std::size_t mem = 0;
      a.cuDeviceTotalMem_v2(&mem, dev);

      cublasHandle_t blas = nullptr;
      if (a.cublasCreate_v2) a.cublasCreate_v2(&blas);

      DeviceInfo info;
      info.device = DeviceRef::cuda(static_cast<int>(contexts_.size()));
      info.name = dev_name;
      info.vendor = "NVIDIA";
      info.api_version = "CUDA " + version_;
      info.driver_version = version_;
      info.memory_bytes = mem;
      info.subgroup_size = 32;
      info.unified_memory = false;
      devices_.push_back(std::move(info));
      contexts_.push_back(ctx);
      blas_handles_.push_back(blas);
    }
  }

  Kind kind() const noexcept override { return Kind::Cuda; }
  std::string_view name() const noexcept override { return "cuda"; }
  std::string_view version() const noexcept override { return version_; }
  const std::vector<DeviceInfo>& devices() const noexcept override { return devices_; }
  std::string unavailable_reason() const override { return reason_; }
  bool supports(Operation op) const noexcept override {
    return !devices_.empty() && static_cast<unsigned>(op) <= static_cast<unsigned>(Operation::AdamW);
  }

  void* allocate(int device, std::size_t bytes) override {
    if (bytes == 0) return nullptr;
    set_ctx(device);
    std::lock_guard lock(mutex_);
    CUdeviceptr ptr = 0;
    if (cuda_dyn().cuMemAlloc_v2(&ptr, bytes) != 0 || ptr == 0) {
      throw TensorError("Nexus CUDA Driver: cuMemAlloc_v2 basarisiz");
    }
    ++stats_.allocations;
    ++stats_.live_blocks;
    stats_.current_bytes += bytes;
    stats_.reserved_bytes += bytes;
    sizes_[ptr] = bytes;
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr));
  }

  void deallocate(int device, void* pointer) noexcept override {
    if (!pointer) return;
    try {
      set_ctx(device);
      std::lock_guard lock(mutex_);
      const auto ptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(pointer));
      auto it = sizes_.find(ptr);
      if (it != sizes_.end()) {
        stats_.current_bytes = stats_.current_bytes >= it->second ? stats_.current_bytes - it->second : 0;
        if (stats_.live_blocks > 0) --stats_.live_blocks;
        sizes_.erase(it);
      }
      cuda_dyn().cuMemFree_v2(ptr);
    } catch (...) {}
  }

  void copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src, std::size_t bytes) override {
    if (bytes == 0) return;
    const bool dst_cu = dst_device.kind == DeviceKind::Cuda;
    const bool src_cu = src_device.kind == DeviceKind::Cuda;
    if (!dst_cu && !src_cu) {
      std::memcpy(dst, src, bytes);
      return;
    }
    set_ctx(dst_cu ? dst_device.index : src_device.index);
    CUresult res = 0;
    if (dst_cu && src_cu) {
      res = cuda_dyn().cuMemcpyDtoD_v2(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(dst)),
                                       static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(src)), bytes);
    } else if (dst_cu) {
      res = cuda_dyn().cuMemcpyHtoD_v2(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(dst)), src, bytes);
    } else {
      res = cuda_dyn().cuMemcpyDtoH_v2(dst, static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(src)), bytes);
    }
    if (res != 0) throw TensorError("Nexus CUDA Driver: cuMemcpy basarisiz");
  }

  void synchronize(int device) override {
    set_ctx(device);
    if (cuda_dyn().cuCtxSynchronize() != 0) throw TensorError("Nexus CUDA Driver: cuCtxSynchronize basarisiz");
  }

  void fill(DeviceRef device, float* dst, std::size_t count, float value) override {
    if (count == 0) return;
    set_ctx(device.index);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if (cuda_dyn().cuMemsetD32_v2(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(dst)), bits, count) != 0) {
      throw TensorError("Nexus CUDA Driver: cuMemsetD32_v2 basarisiz");
    }
  }

  void matmul(DeviceRef device, const MatMulArgs& args) override {
    set_ctx(device.index);
    cublasHandle_t blas = blas_handles_.at(static_cast<std::size_t>(device.index));
    if (blas && cuda_dyn().cublasSgemm_v2) {
      // Satir-ana C[m,n] = A[m,k] * B[k,n] <=> Sutun-ana C^T[n,m] = B^T[n,k] * A^T[k,m]
      const int op_b = args.b_trans ? 1 : 0;  // CUBLAS_OP_T = 1, CUBLAS_OP_N = 0
      const int op_a = args.a_trans ? 1 : 0;
      const int status = cuda_dyn().cublasSgemm_v2(
          blas, op_b, op_a, args.n, args.m, args.k, &args.alpha,
          args.b, args.ldb, args.a, args.lda, &args.beta, args.c, args.ldc);
      if (status == 0) {
        cuda_dyn().cuCtxSynchronize();
        return;
      }
    }
    // cuBLAS DLL yoksa host uzerinden gonderim
    const std::size_t a_elems = static_cast<std::size_t>(args.a_trans ? args.k : args.m) * args.lda;
    const std::size_t b_elems = static_cast<std::size_t>(args.b_trans ? args.n : args.k) * args.ldb;
    const std::size_t c_elems = static_cast<std::size_t>(args.m) * args.ldc;
    std::vector<float> ha(a_elems), hb(b_elems), hc(c_elems, 0.0f);
    copy(DeviceRef::cpu(), ha.data(), device, args.a, a_elems * sizeof(float));
    copy(DeviceRef::cpu(), hb.data(), device, args.b, b_elems * sizeof(float));
    if (args.beta != 0.0f) copy(DeviceRef::cpu(), hc.data(), device, args.c, c_elems * sizeof(float));
    MatMulArgs h = args;
    h.a = ha.data();
    h.b = hb.data();
    h.c = hc.data();
    Registry::instance().require(Kind::Cpu).matmul(DeviceRef::cpu(), h);
    copy(device, args.c, DeviceRef::cpu(), hc.data(), c_elems * sizeof(float));
  }

  void elementwise(Operation op, DeviceRef device, const float* a, const float* b, float* out,
                   std::size_t count, float scalar) override {
    if (count == 0) return;
    std::vector<float> ha(count), hb, hout(count);
    copy(DeviceRef::cpu(), ha.data(), device, a, count * sizeof(float));
    if (b != nullptr && (op <= Operation::Div || op == Operation::Axpy)) {
      hb.resize(count);
      copy(DeviceRef::cpu(), hb.data(), device, b, count * sizeof(float));
    }
    Registry::instance().require(Kind::Cpu).elementwise(op, DeviceRef::cpu(), ha.data(),
                                                        hb.empty() ? nullptr : hb.data(),
                                                        hout.data(), count, scalar);
    copy(device, out, DeviceRef::cpu(), hout.data(), count * sizeof(float));
  }

  void reduce_sum(DeviceRef device, const float* input, float* output, std::size_t count) override {
    if (count == 0) {
      fill(device, output, 1, 0.0f);
      return;
    }
    std::vector<float> hin(count);
    copy(DeviceRef::cpu(), hin.data(), device, input, count * sizeof(float));
    float sum = 0.0f;
    Registry::instance().require(Kind::Cpu).reduce_sum(DeviceRef::cpu(), hin.data(), &sum, count);
    copy(device, output, DeviceRef::cpu(), &sum, sizeof(float));
  }

  AllocatorStats allocator_stats(int) const override {
    std::lock_guard lock(mutex_);
    return stats_;
  }

 private:
  void set_ctx(int index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= contexts_.size()) {
      throw TensorError("Nexus CUDA Driver: gecersiz cihaz indeksi");
    }
    cuda_dyn().cuCtxSetCurrent(contexts_[static_cast<std::size_t>(index)]);
  }

  std::string version_ = "0";
  std::string reason_;
  std::vector<DeviceInfo> devices_;
  std::vector<CUcontext> contexts_;
  std::vector<cublasHandle_t> blas_handles_;
  mutable std::mutex mutex_;
  std::unordered_map<CUdeviceptr, std::size_t> sizes_;
  AllocatorStats stats_{};
};

}  // namespace

// ============================================================================
//  Adlandirma ve Donusumler
// ============================================================================

const char* to_string(Kind kind) noexcept {
  switch (kind) {
    case Kind::Cpu: return "cpu";
    case Kind::Cuda: return "cuda";
    case Kind::Hip: return "hip";
    case Kind::Vulkan: return "vulkan";
    case Kind::Metal: return "metal";
    case Kind::DirectMl: return "directml";
    case Kind::Pjrt: return "pjrt";
  }
  return "unknown";
}

const char* to_string(Operation op) noexcept {
  switch (op) {
    case Operation::Add: return "add";
    case Operation::Sub: return "sub";
    case Operation::Mul: return "mul";
    case Operation::Div: return "div";
    case Operation::Neg: return "neg";
    case Operation::Exp: return "exp";
    case Operation::Log: return "log";
    case Operation::Sqrt: return "sqrt";
    case Operation::Rsqrt: return "rsqrt";
    case Operation::Tanh: return "tanh";
    case Operation::Sigmoid: return "sigmoid";
    case Operation::Silu: return "silu";
    case Operation::Relu: return "relu";
    case Operation::Scale: return "scale";
    case Operation::Axpy: return "axpy";
    case Operation::MatMul: return "matmul";
    case Operation::ReduceSum: return "reduce_sum";
    case Operation::Fill: return "fill";
    case Operation::Softmax: return "softmax";
    case Operation::RmsNorm: return "rms_norm";
    case Operation::LayerNorm: return "layer_norm";
    case Operation::Rope: return "rope";
    case Operation::SwiGlu: return "swiglu";
    case Operation::Attention: return "attention";
    case Operation::QuantMatMulQ8: return "quant_matmul_q8";
    case Operation::QuantMatMulQ4: return "quant_matmul_q4";
    case Operation::AdamW: return "adamw";
  }
  return "unknown";
}

Kind parse_kind(std::string_view text) {
  const std::string key = lower_copy(text);
  if (key == "cpu" || key == "host") return Kind::Cpu;
  if (key == "cuda" || key == "gpu") return Kind::Cuda;
  if (key == "hip" || key == "rocm") return Kind::Hip;
  if (key == "vulkan" || key == "vk" || key == "spirv") return Kind::Vulkan;
  if (key == "metal" || key == "mps") return Kind::Metal;
  if (key == "directml" || key == "dml" || key == "dx12") return Kind::DirectMl;
  if (key == "pjrt" || key == "tpu" || key == "xla" || key == "openxla") return Kind::Pjrt;
  throw TensorError("Nexus: bilinmeyen backend turu: " + std::string(text));
}

Kind from_device(DeviceKind kind) {
  switch (kind) {
    case DeviceKind::Cpu: return Kind::Cpu;
    case DeviceKind::Cuda: return Kind::Cuda;
    case DeviceKind::Rocm: return Kind::Hip;
    case DeviceKind::Vulkan: return Kind::Vulkan;
    case DeviceKind::Metal: return Kind::Metal;
    case DeviceKind::DirectMl: return Kind::DirectMl;
    case DeviceKind::Tpu: return Kind::Pjrt;
    case DeviceKind::Npu:
      throw TensorError("Nexus: NPU genel backend fabrikasinda degil; ozel eklenti kullanin");
  }
  throw TensorError("Nexus: bilinmeyen DeviceKind");
}

DeviceKind to_device(Kind kind) {
  switch (kind) {
    case Kind::Cpu: return DeviceKind::Cpu;
    case Kind::Cuda: return DeviceKind::Cuda;
    case Kind::Hip: return DeviceKind::Rocm;
    case Kind::Vulkan: return DeviceKind::Vulkan;
    case Kind::Metal: return DeviceKind::Metal;
    case Kind::DirectMl: return DeviceKind::DirectMl;
    case Kind::Pjrt: return DeviceKind::Tpu;
  }
  return DeviceKind::Cpu;
}

// ============================================================================
//  Driver Varsayilan Govdeleri
// ============================================================================

void Driver::fill(DeviceRef device, float* dst, std::size_t count, float value) {
  if (count == 0) return;
  std::vector<float> host(count, value);
  copy(device, dst, DeviceRef::cpu(), host.data(), count * sizeof(float));
}

AllocatorStats Driver::allocator_stats(int) const { return {}; }

StreamHandle Driver::create_stream(int /*device*/) {
  return reinterpret_cast<StreamHandle>(new int(1));
}
void Driver::destroy_stream(int /*device*/, StreamHandle stream) noexcept {
  delete static_cast<int*>(stream);
}
void Driver::synchronize_stream(int device, StreamHandle /*stream*/) {
  synchronize(device);
}
EventHandle Driver::create_event(int /*device*/) {
  return reinterpret_cast<EventHandle>(new int(1));
}
void Driver::destroy_event(int /*device*/, EventHandle event) noexcept {
  delete static_cast<int*>(event);
}
void Driver::record_event(int device, EventHandle /*event*/, StreamHandle /*stream*/) {
  synchronize(device);
}
void Driver::stream_wait_event(int device, StreamHandle /*stream*/, EventHandle /*event*/) {
  synchronize(device);
}

void Driver::softmax(DeviceRef device, const float* input, float* output, int rows, int cols) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  std::vector<float> hin(n), hout(n);
  copy(DeviceRef::cpu(), hin.data(), device, input, n * sizeof(float));
  Registry::instance().require(Kind::Cpu).softmax(DeviceRef::cpu(), hin.data(), hout.data(), rows, cols);
  copy(device, output, DeviceRef::cpu(), hout.data(), n * sizeof(float));
}

void Driver::rms_norm(DeviceRef device, const float* x, const float* weight, float* out,
                      int rows, int cols, float eps) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  std::vector<float> hx(n), hw(static_cast<std::size_t>(cols)), hout(n);
  copy(DeviceRef::cpu(), hx.data(), device, x, n * sizeof(float));
  copy(DeviceRef::cpu(), hw.data(), device, weight, static_cast<std::size_t>(cols) * sizeof(float));
  Registry::instance().require(Kind::Cpu).rms_norm(DeviceRef::cpu(), hx.data(), hw.data(), hout.data(), rows, cols, eps);
  copy(device, out, DeviceRef::cpu(), hout.data(), n * sizeof(float));
}

void Driver::layer_norm(DeviceRef device, const float* x, const float* weight, const float* bias,
                        float* out, int rows, int cols, float eps) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  std::vector<float> hx(n), hw(static_cast<std::size_t>(cols)), hb, hout(n);
  copy(DeviceRef::cpu(), hx.data(), device, x, n * sizeof(float));
  copy(DeviceRef::cpu(), hw.data(), device, weight, static_cast<std::size_t>(cols) * sizeof(float));
  if (bias != nullptr) {
    hb.resize(static_cast<std::size_t>(cols));
    copy(DeviceRef::cpu(), hb.data(), device, bias, static_cast<std::size_t>(cols) * sizeof(float));
  }
  Registry::instance().require(Kind::Cpu).layer_norm(
      DeviceRef::cpu(), hx.data(), hw.data(), hb.empty() ? nullptr : hb.data(), hout.data(), rows, cols, eps);
  copy(device, out, DeviceRef::cpu(), hout.data(), n * sizeof(float));
}

void Driver::rope(DeviceRef device, const float* x, float* out, int batch_heads, int seq_len,
                  int head_dim, int pos_offset, float theta) {
  const std::size_t n = static_cast<std::size_t>(batch_heads) * seq_len * head_dim;
  std::vector<float> hx(n), hout(n);
  copy(DeviceRef::cpu(), hx.data(), device, x, n * sizeof(float));
  Registry::instance().require(Kind::Cpu).rope(DeviceRef::cpu(), hx.data(), hout.data(), batch_heads, seq_len, head_dim, pos_offset, theta);
  copy(device, out, DeviceRef::cpu(), hout.data(), n * sizeof(float));
}

void Driver::swiglu(DeviceRef device, const float* x, float* out, int rows, int hidden_dim) {
  const std::size_t in_n = static_cast<std::size_t>(rows) * 2 * hidden_dim;
  const std::size_t out_n = static_cast<std::size_t>(rows) * hidden_dim;
  std::vector<float> hx(in_n), hout(out_n);
  copy(DeviceRef::cpu(), hx.data(), device, x, in_n * sizeof(float));
  Registry::instance().require(Kind::Cpu).swiglu(DeviceRef::cpu(), hx.data(), hout.data(), rows, hidden_dim);
  copy(device, out, DeviceRef::cpu(), hout.data(), out_n * sizeof(float));
}

void Driver::attention(DeviceRef device, const float* q, const float* k, const float* v, float* out,
                       int batch_heads, int seq_len, int head_dim, bool causal) {
  const std::size_t n = static_cast<std::size_t>(batch_heads) * seq_len * head_dim;
  std::vector<float> hq(n), hk(n), hv(n), hout(n);
  copy(DeviceRef::cpu(), hq.data(), device, q, n * sizeof(float));
  copy(DeviceRef::cpu(), hk.data(), device, k, n * sizeof(float));
  copy(DeviceRef::cpu(), hv.data(), device, v, n * sizeof(float));
  Registry::instance().require(Kind::Cpu).attention(DeviceRef::cpu(), hq.data(), hk.data(), hv.data(), hout.data(),
                                                    batch_heads, seq_len, head_dim, causal);
  copy(device, out, DeviceRef::cpu(), hout.data(), n * sizeof(float));
}

void Driver::quant_matmul_q8(DeviceRef device, const float* a, const void* w_q8,
                             const float* w_scales, float* c, int m, int n, int k, int group_size) {
  const std::size_t a_n = static_cast<std::size_t>(m) * k;
  const std::size_t w_bytes = static_cast<std::size_t>(n) * k;
  const std::size_t s_n = static_cast<std::size_t>(n) * (k / group_size);
  const std::size_t c_n = static_cast<std::size_t>(m) * n;
  std::vector<float> ha(a_n), hs(s_n), hc(c_n);
  std::vector<std::int8_t> hw(w_bytes);
  copy(DeviceRef::cpu(), ha.data(), device, a, a_n * sizeof(float));
  copy(DeviceRef::cpu(), hw.data(), device, w_q8, w_bytes);
  copy(DeviceRef::cpu(), hs.data(), device, w_scales, s_n * sizeof(float));
  Registry::instance().require(Kind::Cpu).quant_matmul_q8(DeviceRef::cpu(), ha.data(), hw.data(), hs.data(), hc.data(),
                                                          m, n, k, group_size);
  copy(device, c, DeviceRef::cpu(), hc.data(), c_n * sizeof(float));
}

void Driver::quant_matmul_q4(DeviceRef device, const float* a, const void* w_q4,
                             const float* w_scales, float* c, int m, int n, int k, int group_size) {
  const std::size_t a_n = static_cast<std::size_t>(m) * k;
  const std::size_t w_bytes = static_cast<std::size_t>(n) * (k / 2);
  const std::size_t s_n = static_cast<std::size_t>(n) * (k / group_size);
  const std::size_t c_n = static_cast<std::size_t>(m) * n;
  std::vector<float> ha(a_n), hs(s_n), hc(c_n);
  std::vector<std::uint8_t> hw(w_bytes);
  copy(DeviceRef::cpu(), ha.data(), device, a, a_n * sizeof(float));
  copy(DeviceRef::cpu(), hw.data(), device, w_q4, w_bytes);
  copy(DeviceRef::cpu(), hs.data(), device, w_scales, s_n * sizeof(float));
  Registry::instance().require(Kind::Cpu).quant_matmul_q4(DeviceRef::cpu(), ha.data(), hw.data(), hs.data(), hc.data(),
                                                          m, n, k, group_size);
  copy(device, c, DeviceRef::cpu(), hc.data(), c_n * sizeof(float));
}

void Driver::adamw_step(DeviceRef device, float* param, const float* grad, float* m, float* v,
                        std::size_t count, float lr, float beta1, float beta2, float eps,
                        float weight_decay, int step) {
  std::vector<float> hp(count), hg(count), hm(count), hv(count);
  copy(DeviceRef::cpu(), hp.data(), device, param, count * sizeof(float));
  copy(DeviceRef::cpu(), hg.data(), device, grad, count * sizeof(float));
  copy(DeviceRef::cpu(), hm.data(), device, m, count * sizeof(float));
  copy(DeviceRef::cpu(), hv.data(), device, v, count * sizeof(float));
  Registry::instance().require(Kind::Cpu).adamw_step(
      DeviceRef::cpu(), hp.data(), hg.data(), hm.data(), hv.data(), count, lr, beta1, beta2, eps, weight_decay, step);
  copy(device, param, DeviceRef::cpu(), hp.data(), count * sizeof(float));
  copy(device, m, DeviceRef::cpu(), hm.data(), count * sizeof(float));
  copy(device, v, DeviceRef::cpu(), hv.data(), count * sizeof(float));
}

// ============================================================================
//  Registry
// ============================================================================

Registry& Registry::instance() {
  static Registry registry;
  return registry;
}

Registry::Registry() {
  add(std::make_unique<CpuDriver>());
  add(make_cuda_driver());
  add(make_hip_driver());
  add(make_vulkan_driver());
  add(make_metal_driver());
  add(make_directml_driver());
  add(make_pjrt_driver());
}

void Registry::add(std::unique_ptr<Driver> driver) {
  if (!driver) throw TensorError("Nexus: bos backend surucusu eklenemez");
  for (std::size_t i = 0; i < owned_.size(); ++i) {
    if (owned_[i]->kind() == driver->kind()) {
      view_[i] = driver.get();
      owned_[i] = std::move(driver);
      return;
    }
  }
  view_.push_back(driver.get());
  owned_.push_back(std::move(driver));
}

Driver* Registry::find(Kind kind) const noexcept {
  for (Driver* driver : view_) if (driver->kind() == kind) return driver;
  return nullptr;
}

Driver& Registry::require(Kind kind) const {
  if (Driver* driver = find(kind); driver != nullptr) return *driver;
  throw TensorError(std::string("Nexus: kayitli olmayan backend: ") + to_string(kind));
}

const std::vector<Driver*>& Registry::all() const noexcept { return view_; }

Driver& Registry::select(Operation op) const {
  if (const char* env = std::getenv("NEXUS_BACKEND"); env != nullptr && *env != '\0') {
    const std::string requested = lower_copy(env);
    if (requested != "auto") {
      Driver& d = require(parse_kind(requested));
      if (!d.available() || !d.supports(op)) {
        throw TensorError(std::string("Nexus: NEXUS_BACKEND=") + env + " kullanilamaz: " + d.unavailable_reason());
      }
      return d;
    }
  }
  const Kind priority[] = {Kind::Cuda, Kind::Hip, Kind::Vulkan, Kind::Metal, Kind::DirectMl, Kind::Pjrt, Kind::Cpu};
  for (Kind kind : priority) {
    if (Driver* d = find(kind); d != nullptr && d->available() && d->supports(op)) return *d;
  }
  return require(Kind::Cpu);
}

void set_fallback_policy(FallbackPolicy policy) noexcept { g_policy.store(policy, std::memory_order_relaxed); }
FallbackPolicy fallback_policy() noexcept { return g_policy.load(std::memory_order_relaxed); }

FallbackStats fallback_stats(Kind kind, Operation op) {
  std::lock_guard lock(g_fallback_mutex);
  const auto k = static_cast<std::size_t>(kind);
  const auto o = static_cast<std::size_t>(op);
  if (k >= kKindCount || o >= kOpCount) return {};
  return g_fallback_stats[k][o];
}

void reset_fallback_stats() noexcept {
  std::lock_guard lock(g_fallback_mutex);
  g_fallback_stats = {};
  g_warned = {};
}

// ============================================================================
//  Serbest Islem Gondericileri (Yetki + Geri Dusus Politikasi)
// ============================================================================

void elementwise(Operation op, DeviceRef device, const float* a, const float* b, float* out,
                 std::size_t count, float scalar) {
  if (count == 0) return;
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(op)) {
    d.elementwise(op, device, a, b, out, count, scalar);
    return;
  }
  record_fallback(d, op, count);
  Driver& cpu = Registry::instance().require(Kind::Cpu);
  if (device.is_host() || !d.available()) {
    cpu.elementwise(op, DeviceRef::cpu(), a, b, out, count, scalar);
    return;
  }
  std::vector<float> ha(count), hb, hout(count);
  d.copy(DeviceRef::cpu(), ha.data(), device, a, count * sizeof(float));
  if (b != nullptr && (op <= Operation::Div || op == Operation::Axpy)) {
    hb.resize(count);
    d.copy(DeviceRef::cpu(), hb.data(), device, b, count * sizeof(float));
  }
  cpu.elementwise(op, DeviceRef::cpu(), ha.data(), hb.empty() ? nullptr : hb.data(), hout.data(), count, scalar);
  d.copy(device, out, DeviceRef::cpu(), hout.data(), count * sizeof(float));
}

void matmul(DeviceRef device, const MatMulArgs& args) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::MatMul)) {
    d.matmul(device, args);
    return;
  }
  const std::size_t work = static_cast<std::size_t>(std::max(0, args.m)) * static_cast<std::size_t>(std::max(0, args.n));
  record_fallback(d, Operation::MatMul, work);
  Driver& cpu = Registry::instance().require(Kind::Cpu);
  if (device.is_host() || !d.available()) {
    cpu.matmul(DeviceRef::cpu(), args);
    return;
  }
  const std::size_t a_rows = static_cast<std::size_t>(args.a_trans ? args.k : args.m);
  const std::size_t b_rows = static_cast<std::size_t>(args.b_trans ? args.n : args.k);
  std::vector<float> ha(a_rows * static_cast<std::size_t>(args.lda));
  std::vector<float> hb(b_rows * static_cast<std::size_t>(args.ldb));
  std::vector<float> hc(static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.ldc), 0.0f);
  d.copy(DeviceRef::cpu(), ha.data(), device, args.a, ha.size() * sizeof(float));
  d.copy(DeviceRef::cpu(), hb.data(), device, args.b, hb.size() * sizeof(float));
  if (args.beta != 0.0f) d.copy(DeviceRef::cpu(), hc.data(), device, args.c, hc.size() * sizeof(float));
  MatMulArgs host_args = args;
  host_args.a = ha.data();
  host_args.b = hb.data();
  host_args.c = hc.data();
  cpu.matmul(DeviceRef::cpu(), host_args);
  d.copy(device, args.c, DeviceRef::cpu(), hc.data(), hc.size() * sizeof(float));
}

void reduce_sum(DeviceRef device, const float* input, float* output, std::size_t count) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::ReduceSum)) {
    d.reduce_sum(device, input, output, count);
    return;
  }
  record_fallback(d, Operation::ReduceSum, count);
  Driver& cpu = Registry::instance().require(Kind::Cpu);
  if (device.is_host() || !d.available()) {
    cpu.reduce_sum(DeviceRef::cpu(), input, output, count);
    return;
  }
  std::vector<float> hin(count);
  float hout = 0.0f;
  d.copy(DeviceRef::cpu(), hin.data(), device, input, count * sizeof(float));
  cpu.reduce_sum(DeviceRef::cpu(), hin.data(), &hout, count);
  d.copy(device, output, DeviceRef::cpu(), &hout, sizeof(float));
}

void softmax(DeviceRef device, const float* input, float* output, int rows, int cols) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::Softmax)) {
    d.softmax(device, input, output, rows, cols);
    return;
  }
  record_fallback(d, Operation::Softmax, static_cast<std::size_t>(rows) * cols);
  Registry::instance().require(Kind::Cpu).softmax(DeviceRef::cpu(), input, output, rows, cols);
}

void rms_norm(DeviceRef device, const float* x, const float* weight, float* out, int rows, int cols, float eps) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::RmsNorm)) {
    d.rms_norm(device, x, weight, out, rows, cols, eps);
    return;
  }
  record_fallback(d, Operation::RmsNorm, static_cast<std::size_t>(rows) * cols);
  Registry::instance().require(Kind::Cpu).rms_norm(DeviceRef::cpu(), x, weight, out, rows, cols, eps);
}

void layer_norm(DeviceRef device, const float* x, const float* weight, const float* bias, float* out,
                int rows, int cols, float eps) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::LayerNorm)) {
    d.layer_norm(device, x, weight, bias, out, rows, cols, eps);
    return;
  }
  record_fallback(d, Operation::LayerNorm, static_cast<std::size_t>(rows) * cols);
  Registry::instance().require(Kind::Cpu).layer_norm(DeviceRef::cpu(), x, weight, bias, out, rows, cols, eps);
}

void rope(DeviceRef device, const float* x, float* out, int batch_heads, int seq_len, int head_dim,
          int pos_offset, float theta) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::Rope)) {
    d.rope(device, x, out, batch_heads, seq_len, head_dim, pos_offset, theta);
    return;
  }
  record_fallback(d, Operation::Rope, static_cast<std::size_t>(batch_heads) * seq_len * head_dim);
  Registry::instance().require(Kind::Cpu).rope(DeviceRef::cpu(), x, out, batch_heads, seq_len, head_dim, pos_offset, theta);
}

void swiglu(DeviceRef device, const float* x, float* out, int rows, int hidden_dim) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::SwiGlu)) {
    d.swiglu(device, x, out, rows, hidden_dim);
    return;
  }
  record_fallback(d, Operation::SwiGlu, static_cast<std::size_t>(rows) * hidden_dim);
  Registry::instance().require(Kind::Cpu).swiglu(DeviceRef::cpu(), x, out, rows, hidden_dim);
}

void attention(DeviceRef device, const float* q, const float* k, const float* v, float* out,
               int batch_heads, int seq_len, int head_dim, bool causal) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::Attention)) {
    d.attention(device, q, k, v, out, batch_heads, seq_len, head_dim, causal);
    return;
  }
  record_fallback(d, Operation::Attention, static_cast<std::size_t>(batch_heads) * seq_len * head_dim);
  Registry::instance().require(Kind::Cpu).attention(DeviceRef::cpu(), q, k, v, out, batch_heads, seq_len, head_dim, causal);
}

void quant_matmul_q8(DeviceRef device, const float* a, const void* w_q8, const float* w_scales,
                     float* c, int m, int n, int k, int group_size) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::QuantMatMulQ8)) {
    d.quant_matmul_q8(device, a, w_q8, w_scales, c, m, n, k, group_size);
    return;
  }
  record_fallback(d, Operation::QuantMatMulQ8, static_cast<std::size_t>(m) * n);
  Registry::instance().require(Kind::Cpu).quant_matmul_q8(DeviceRef::cpu(), a, w_q8, w_scales, c, m, n, k, group_size);
}

void quant_matmul_q4(DeviceRef device, const float* a, const void* w_q4, const float* w_scales,
                     float* c, int m, int n, int k, int group_size) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::QuantMatMulQ4)) {
    d.quant_matmul_q4(device, a, w_q4, w_scales, c, m, n, k, group_size);
    return;
  }
  record_fallback(d, Operation::QuantMatMulQ4, static_cast<std::size_t>(m) * n);
  Registry::instance().require(Kind::Cpu).quant_matmul_q4(DeviceRef::cpu(), a, w_q4, w_scales, c, m, n, k, group_size);
}

void adamw_step(DeviceRef device, float* param, const float* grad, float* m, float* v,
                std::size_t count, float lr, float beta1, float beta2, float eps,
                float weight_decay, int step) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (d.available() && d.supports(Operation::AdamW)) {
    d.adamw_step(device, param, grad, m, v, count, lr, beta1, beta2, eps, weight_decay, step);
    return;
  }
  record_fallback(d, Operation::AdamW, count);
  Registry::instance().require(Kind::Cpu).adamw_step(
      DeviceRef::cpu(), param, grad, m, v, count, lr, beta1, beta2, eps, weight_decay, step);
}

// ============================================================================
//  core/memory.cpp Koprusu
// ============================================================================

void* memory_allocate(DeviceRef device, std::size_t bytes) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (!d.available()) throw TensorError("Nexus: " + std::string(to_string(device.kind)) + " kullanilamiyor: " + d.unavailable_reason());
  return d.allocate(device.index, bytes);
}
void memory_deallocate(DeviceRef device, void* pointer) noexcept {
  if (Driver* d = Registry::instance().find(from_device(device.kind)); d != nullptr) d->deallocate(device.index, pointer);
}
void memory_copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src, std::size_t bytes) {
  const DeviceKind kind = !dst_device.is_host() ? dst_device.kind : src_device.kind;
  Driver& d = Registry::instance().require(from_device(kind));
  if (!d.available()) throw TensorError("Nexus: " + std::string(to_string(kind)) + " kopyasi kullanilamiyor: " + d.unavailable_reason());
  d.copy(dst_device, dst, src_device, src, bytes);
}
void memory_synchronize(DeviceRef device) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (!d.available()) throw TensorError("Nexus: " + std::string(to_string(device.kind)) + " senkronu kullanilamiyor: " + d.unavailable_reason());
  d.synchronize(device.index);
}
void memory_fill(DeviceRef device, float* dst, std::size_t count, float value) {
  Driver& d = Registry::instance().require(from_device(device.kind));
  if (!d.available()) throw TensorError("Nexus: " + std::string(to_string(device.kind)) + " doldurma kullanilamiyor: " + d.unavailable_reason());
  d.fill(device, dst, count, value);
}

std::unique_ptr<Driver> make_cuda_driver() { return std::make_unique<CudaDriver>(); }

}  // namespace nexus::backend
