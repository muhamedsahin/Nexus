#pragma once

/**
 * @file backend.hpp
 * @brief FAZ 19: surumlu hizlandirici fabrikasi, yetenekler ve denetlenebilir CPU geri dususu.
 *
 * Cihaz tamponlari opak `float*` taniticilaridir; host bunlari dereference etmez.
 * Suruculer is parcacigi guvenlidir. Bir cagri dondugunde sonuc tensörde hazirdir;
 * boylece mevcut Tensor semantigi ve eski CPU/CUDA yollari degismez.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/device.hpp"
#include "nexus/core/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace nexus::backend {

enum class Kind : std::uint8_t { Cpu = 0, Cuda, Hip, Vulkan, Metal, DirectMl, Pjrt };
using BackendKind = Kind;

enum class Operation : std::uint16_t {
  Add = 0,
  Sub,
  Mul,
  Div,
  Neg,
  Exp,
  Log,
  Sqrt,
  Rsqrt,
  Tanh,
  Sigmoid,
  Silu,
  Relu,
  Scale,
  Axpy,
  MatMul,
  ReduceSum,
  Fill,
  Softmax,
  RmsNorm,
  LayerNorm,
  Rope,
  SwiGlu,
  Attention,
  QuantMatMulQ8,
  QuantMatMulQ4,
  AdamW,
};

[[nodiscard]] const char* to_string(Kind kind) noexcept;
[[nodiscard]] const char* to_string(Operation op) noexcept;
[[nodiscard]] Kind parse_kind(std::string_view text);
[[nodiscard]] Kind from_device(DeviceKind kind);
[[nodiscard]] DeviceKind to_device(Kind kind);
[[nodiscard]] inline DeviceRef parse_device(std::string_view spec) { return DeviceRef::parse(spec); }

struct DeviceInfo {
  DeviceRef device{};
  std::string name;
  std::string vendor;
  std::string api_version;
  std::string driver_version;
  std::uint64_t memory_bytes = 0;
  std::uint32_t subgroup_size = 0;
  bool unified_memory = false;
};

struct MatMulArgs {
  int m = 0, n = 0, k = 0;
  const float* a = nullptr;
  int lda = 0;
  bool a_trans = false;
  const float* b = nullptr;
  int ldb = 0;
  bool b_trans = false;
  float* c = nullptr;
  int ldc = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
};

struct AllocatorStats {
  std::uint64_t allocations = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t live_blocks = 0;
  std::uint64_t current_bytes = 0;
  std::uint64_t reserved_bytes = 0;
};

class Driver {
 public:
  virtual ~Driver() = default;
  [[nodiscard]] virtual Kind kind() const noexcept = 0;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual std::string_view version() const noexcept = 0;
  [[nodiscard]] virtual const std::vector<DeviceInfo>& devices() const noexcept = 0;
  [[nodiscard]] virtual std::string unavailable_reason() const = 0;
  [[nodiscard]] virtual bool supports(Operation op) const noexcept = 0;
  [[nodiscard]] bool available() const noexcept { return !devices().empty(); }

  // Bellek arayuzu
  virtual void* allocate(int device, std::size_t bytes) = 0;
  virtual void deallocate(int device, void* pointer) noexcept = 0;
  virtual void copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src,
                    std::size_t bytes) = 0;
  virtual void synchronize(int device) = 0;
  virtual void fill(DeviceRef device, float* dst, std::size_t count, float value);
  [[nodiscard]] virtual AllocatorStats allocator_stats(int device) const;

  // Akis ve olay (Stream & Event) soyutlamasi
  virtual StreamHandle create_stream(int device);
  virtual void destroy_stream(int device, StreamHandle stream) noexcept;
  virtual void synchronize_stream(int device, StreamHandle stream);
  virtual EventHandle create_event(int device);
  virtual void destroy_event(int device, EventHandle event) noexcept;
  virtual void record_event(int device, EventHandle event, StreamHandle stream);
  virtual void stream_wait_event(int device, StreamHandle stream, EventHandle event);

  // Temel cekirdekler
  virtual void elementwise(Operation op, DeviceRef device, const float* a, const float* b,
                           float* out, std::size_t count, float scalar = 0.0f) = 0;
  virtual void matmul(DeviceRef device, const MatMulArgs& args) = 0;
  virtual void reduce_sum(DeviceRef device, const float* input, float* output, std::size_t count) = 0;

  // Ozel cekirdekler (FAZ 19.A / 19.C: attention, norm, rope, swiglu, softmax, nicemlenmis GEMM, adamw)
  virtual void softmax(DeviceRef device, const float* input, float* output, int rows, int cols);
  virtual void rms_norm(DeviceRef device, const float* x, const float* weight, float* out,
                        int rows, int cols, float eps = 1e-5f);
  virtual void layer_norm(DeviceRef device, const float* x, const float* weight, const float* bias,
                          float* out, int rows, int cols, float eps = 1e-5f);
  virtual void rope(DeviceRef device, const float* x, float* out, int batch_heads, int seq_len,
                    int head_dim, int pos_offset = 0, float theta = 10000.0f);
  virtual void swiglu(DeviceRef device, const float* x, float* out, int rows, int hidden_dim);
  virtual void attention(DeviceRef device, const float* q, const float* k, const float* v, float* out,
                         int batch_heads, int seq_len, int head_dim, bool causal = true);
  virtual void quant_matmul_q8(DeviceRef device, const float* a, const void* w_q8,
                               const float* w_scales, float* c, int m, int n, int k,
                               int group_size = 32);
  virtual void quant_matmul_q4(DeviceRef device, const float* a, const void* w_q4,
                               const float* w_scales, float* c, int m, int n, int k,
                               int group_size = 32);
  virtual void adamw_step(DeviceRef device, float* param, const float* grad, float* m, float* v,
                          std::size_t count, float lr, float beta1, float beta2, float eps,
                          float weight_decay, int step);
};

class Registry {
 public:
  static Registry& instance();
  void add(std::unique_ptr<Driver> driver);
  [[nodiscard]] Driver* find(Kind kind) const noexcept;
  [[nodiscard]] Driver& require(Kind kind) const;
  [[nodiscard]] const std::vector<Driver*>& all() const noexcept;
  /// NEXUS_BACKEND=auto|cpu|cuda|hip|vulkan|metal|dml|pjrt; auto deterministiktir.
  [[nodiscard]] Driver& select(Operation op = Operation::MatMul) const;

 private:
  Registry();
  std::vector<std::unique_ptr<Driver>> owned_;
  std::vector<Driver*> view_;
};

enum class FallbackPolicy : std::uint8_t { Error = 0, WarnOnce, WarnAlways };
struct FallbackStats {
  std::uint64_t calls = 0;
  std::uint64_t elements = 0;
};
void set_fallback_policy(FallbackPolicy policy) noexcept;
/// Enforces and records CPU fallback, including calls made directly through Driver.
void require_cpu_fallback(Driver& driver, Operation op, std::size_t elements);
[[nodiscard]] FallbackPolicy fallback_policy() noexcept;
[[nodiscard]] FallbackStats fallback_stats(Kind kind, Operation op);
void reset_fallback_stats() noexcept;

/// Yetenek varsa dogrudan surucude, yoksa politikaya gore gorunur CPU geri dususunde calisir.
void elementwise(Operation op, DeviceRef device, const float* a, const float* b, float* out,
                 std::size_t count, float scalar = 0.0f);
void matmul(DeviceRef device, const MatMulArgs& args);
void reduce_sum(DeviceRef device, const float* input, float* output, std::size_t count);
void softmax(DeviceRef device, const float* input, float* output, int rows, int cols);
void rms_norm(DeviceRef device, const float* x, const float* weight, float* out, int rows, int cols,
              float eps = 1e-5f);
void layer_norm(DeviceRef device, const float* x, const float* weight, const float* bias, float* out,
                int rows, int cols, float eps = 1e-5f);
void rope(DeviceRef device, const float* x, float* out, int batch_heads, int seq_len, int head_dim,
          int pos_offset = 0, float theta = 10000.0f);
void swiglu(DeviceRef device, const float* x, float* out, int rows, int hidden_dim);
void attention(DeviceRef device, const float* q, const float* k, const float* v, float* out,
               int batch_heads, int seq_len, int head_dim, bool causal = true);
void quant_matmul_q8(DeviceRef device, const float* a, const void* w_q8, const float* w_scales,
                     float* c, int m, int n, int k, int group_size = 32);
void quant_matmul_q4(DeviceRef device, const float* a, const void* w_q4, const float* w_scales,
                     float* c, int m, int n, int k, int group_size = 32);
void adamw_step(DeviceRef device, float* param, const float* grad, float* m, float* v,
                std::size_t count, float lr, float beta1, float beta2, float eps,
                float weight_decay, int step);

// core/memory.cpp icin ince kopru; uygulama kodu Driver/Registry kullanmalidir.
void* memory_allocate(DeviceRef device, std::size_t bytes);
void memory_deallocate(DeviceRef device, void* pointer) noexcept;
void memory_copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src, std::size_t bytes);
void memory_synchronize(DeviceRef device);
void memory_fill(DeviceRef device, float* dst, std::size_t count, float value);

std::unique_ptr<Driver> make_cuda_driver();
std::unique_ptr<Driver> make_hip_driver();
std::unique_ptr<Driver> make_vulkan_driver();
std::unique_ptr<Driver> make_metal_driver();
std::unique_ptr<Driver> make_directml_driver();
std::unique_ptr<Driver> make_pjrt_driver();

}  // namespace nexus::backend
