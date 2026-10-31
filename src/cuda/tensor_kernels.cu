/**
 * @file tensor_kernels.cu
 * @brief Dalga 2: nexus::Tensor dagiticisi icin CUDA cekirdekleri (bkz. include/nexus/cuda/tensor_kernels.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor_ops.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/runtime.hpp"
#include "nexus/cuda/tensor_kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <mutex>
#include <type_traits>
#include <climits>
#include <string>
#include <vector>

namespace nexus {
namespace cuda {
namespace {

constexpr int kThreads = 256;
constexpr cudaStream_t kStream = nullptr;  // eski varsayilan akis (copy_bytes ile sirali)

struct Geo {
  int rank;
  long long size[kMaxRank];
  long long so[kMaxRank], sa[kMaxRank], sb[kMaxRank];
  long long oo, oa, ob;
};

Geo make_geo(const Tensor& out, const Tensor* a, const Tensor* b) {
  Geo g{};
  g.rank = out.rank();
  for (int d = 0; d < g.rank; ++d) {
    g.size[d] = static_cast<long long>(out.shape()[static_cast<std::size_t>(d)]);
    g.so[d] = out.strides()[static_cast<std::size_t>(d)];
    g.sa[d] = a ? a->strides()[static_cast<std::size_t>(d)] : 0;
    g.sb[d] = b ? b->strides()[static_cast<std::size_t>(d)] : 0;
  }
  g.oo = out.offset();
  g.oa = a ? a->offset() : 0;
  g.ob = b ? b->offset() : 0;
  return g;
}

__device__ __forceinline__ void offsets(const Geo& g, long long lin, long long& o, long long& a, long long& b) {
  o = g.oo;
  a = g.oa;
  b = g.ob;
  for (int d = g.rank - 1; d >= 0; --d) {
    const long long idx = lin % g.size[d];
    lin /= g.size[d];
    o += idx * g.so[d];
    a += idx * g.sa[d];
    b += idx * g.sb[d];
  }
}

unsigned blocks_for(long long n) {
  const long long b = (n + kThreads - 1) / kThreads;
  return static_cast<unsigned>(b < 65535 ? b : 65535);  // izgara-adimli donguyle
}

// ---- dtype'tan bagimsiz eleman (cihaz) ----------------------------------------------------------------------
struct Val {
  bool is_int;
  long long i;
  double d;
};

__device__ __forceinline__ Val load(int dt, const void* base, long long off) {
  switch (static_cast<DType>(dt)) {
    case DType::F32: return {false, 0, static_cast<const float*>(base)[off]};
    case DType::F64: return {false, 0, static_cast<const double*>(base)[off]};
    case DType::F16: return {false, 0, __half2float(static_cast<const __half*>(base)[off])};
    case DType::BF16: return {false, 0, __bfloat162float(static_cast<const __nv_bfloat16*>(base)[off])};
    case DType::I32: return {true, static_cast<const int*>(base)[off], 0};
    case DType::I64: return {true, static_cast<const long long*>(base)[off], 0};
    case DType::I8: return {true, static_cast<const signed char*>(base)[off], 0};
    case DType::U8: return {true, static_cast<const unsigned char*>(base)[off], 0};
    case DType::Bool: return {true, static_cast<const unsigned char*>(base)[off] != 0 ? 1 : 0, 0};
    default: return {false, 0, 0};
  }
}

__device__ __forceinline__ long long to_int(const Val& v, long long lo, long long hi, int* err) {
  if (v.is_int) {
    if (v.i < lo || v.i > hi) atomicExch(err, 1);
    return v.i;
  }
  if (!isfinite(v.d)) {
    atomicExch(err, 1);
    return 0;
  }
  const double t = trunc(v.d);
  if (t < static_cast<double>(lo) || t >= static_cast<double>(hi) + 1.0) atomicExch(err, 1);
  return static_cast<long long>(t);
}

__device__ __forceinline__ void store(int dt, void* base, long long off, const Val& v, int* err) {
  const double d = v.is_int ? static_cast<double>(v.i) : v.d;
  switch (static_cast<DType>(dt)) {
    case DType::F32: static_cast<float*>(base)[off] = v.is_int ? static_cast<float>(v.i) : static_cast<float>(v.d); return;
    case DType::F64: static_cast<double*>(base)[off] = d; return;
    case DType::F16: static_cast<__half*>(base)[off] = __float2half_rn(static_cast<float>(d)); return;
    case DType::BF16: static_cast<__nv_bfloat16*>(base)[off] = __float2bfloat16_rn(static_cast<float>(d)); return;
    case DType::I32: static_cast<int*>(base)[off] = static_cast<int>(to_int(v, -2147483648LL, 2147483647LL, err)); return;
    case DType::I64: static_cast<long long*>(base)[off] = v.is_int ? v.i : to_int(v, LLONG_MIN, LLONG_MAX, err); return;
    case DType::I8: static_cast<signed char*>(base)[off] = static_cast<signed char>(to_int(v, -128, 127, err)); return;
    case DType::U8: static_cast<unsigned char*>(base)[off] = static_cast<unsigned char>(to_int(v, 0, 255, err)); return;
    case DType::Bool: static_cast<unsigned char*>(base)[off] = (v.is_int ? v.i != 0 : v.d != 0.0) ? 1 : 0; return;
    default: atomicExch(err, 2); return;
  }
}

__global__ void copy_cast_kernel(void* dst, int dt, const void* src, int st, Geo g, long long n, int* err) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    long long o, a, b;
    offsets(g, i, o, a, b);
    store(dt, dst, o, load(st, src, a), err);
  }
}

// ---- ikili islemler ---------------------------------------------------------------------------------------------
template <typename T> struct Calc { using C = T; static __device__ C ld(T v) { return v; } static __device__ T st(C v) { return v; } };
template <> struct Calc<__half> { using C = float; static __device__ C ld(__half v) { return __half2float(v); } static __device__ __half st(C v) { return __float2half_rn(v); } };
template <> struct Calc<__nv_bfloat16> {
  using C = float;
  static __device__ C ld(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __nv_bfloat16 st(C v) { return __float2bfloat16_rn(v); }
};

struct AddOp { template <typename C> __device__ C operator()(C x, C y) const { return x + y; } };
struct MulOp { template <typename C> __device__ C operator()(C x, C y) const { return x * y; } };

template <typename T, typename Op>
__global__ void binary_kernel(T* out, const T* a, const T* b, Geo g, long long n, bool flat) {
  using K = Calc<T>;
  Op op;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    long long o = g.oo + i, ia = g.oa + i, ib = g.ob + i;
    if (!flat) offsets(g, i, o, ia, ib);
    out[o] = K::st(op(K::ld(a[ia]), K::ld(b[ib])));
  }
}

// ---- toplam (belirlenimci) -----------------------------------------------------------------------------------
template <typename T, typename Acc>
__global__ void sum_partial_kernel(const T* a, Geo g, long long n, Acc* partial) {
  __shared__ Acc sh[kThreads];
  Acc acc = 0;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    long long o, ia, ib;
    offsets(g, i, o, ia, ib);
    acc += static_cast<Acc>(Calc<T>::ld(a[ia]));
  }
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

template <typename Acc, typename O>
__global__ void sum_final_kernel(const Acc* partial, int count, O* out) {
  Acc acc = 0;
  for (int i = 0; i < count; ++i) acc += partial[i];  // sabit sira
  *out = Calc<O>::st(static_cast<typename Calc<O>::C>(acc));
}

// ---- yardimcilar (host) -------------------------------------------------------------------------------------------
int* error_flag(int device) {
  static std::mutex mu;
  static std::vector<int*> flags;
  std::lock_guard<std::mutex> lock(mu);
  if (static_cast<int>(flags.size()) <= device) flags.resize(static_cast<std::size_t>(device) + 1, nullptr);
  if (!flags[static_cast<std::size_t>(device)]) NEXUS_CUDA_CHECK(cudaMalloc(&flags[static_cast<std::size_t>(device)], sizeof(int)));
  return flags[static_cast<std::size_t>(device)];
}

void* base(const Tensor& t) { return t.storage()->data(); }

int device_of(const Tensor& a) { return a.device().index; }

void require_same_device(const Tensor& a, const Tensor& b) {
  if (device_of(a) != device_of(b)) throw TensorError("Nexus cuda: tensörler farkli cihazlarda (adimli esler arasi kopya yok)");
}

void launch_copy(Tensor& dst, const Tensor& src) {
  require_same_device(dst, src);
  const long long n = static_cast<long long>(dst.numel());
  if (n == 0) return;
  DeviceGuard guard(device_of(dst));
  int* err = error_flag(device_of(dst));
  const bool checked = (dst.dtype() == DType::I32 || dst.dtype() == DType::I64 || dst.dtype() == DType::I8 ||
                        dst.dtype() == DType::U8);
  if (checked) NEXUS_CUDA_CHECK(cudaMemsetAsync(err, 0, sizeof(int), kStream));
  copy_cast_kernel<<<blocks_for(n), kThreads, 0, kStream>>>(base(dst), static_cast<int>(dst.dtype()), base(src),
                                                          static_cast<int>(src.dtype()), make_geo(dst, &src, nullptr), n, err);
  NEXUS_CUDA_CHECK(cudaGetLastError());
  if (checked) {
    int h = 0;
    NEXUS_CUDA_CHECK(cudaMemcpy(&h, err, sizeof(int), cudaMemcpyDeviceToHost));  // eski akista: kernelden sonra
    if (h) throw TensorError("Nexus cuda: kayan -> tamsayi donusumunde NaN / aralik disi deger");
  }
}

bool fp8(DType t) { return t == DType::F8E4M3 || t == DType::F8E5M2; }

// ---- dagitici cekirdekleri -----------------------------------------------------------------------------------------
void copy_device(const ops::KernelArgs& k) {
  Tensor& dst = *k.out;
  const Tensor& src = *k.a;
  if (fp8(dst.dtype()) || fp8(src.dtype())) throw TensorError("Nexus cuda: FP8 cihaz donusumu kayitli degil");
  const bool dd = dst.space() == MemorySpace::Device, sd = src.space() == MemorySpace::Device;
  if (dd && sd) {
    launch_copy(dst, src);
  } else if (dd) {  // host -> cihaz: host'ta bitisik, ayni dtype ile tasi, cihazda donustur / yerlestir
    Tensor tmp = src.contiguous().to(MemorySpace::Device, dst.device());
    launch_copy(dst, tmp);
  } else {  // cihaz -> host: cihazda hedef dtype'a bitisik, host'a tasi, host'ta yerlestir
    Tensor tmp = Tensor::empty(dst.shape(), dst.dtype(), MemorySpace::Device, src.device());
    launch_copy(tmp, src);
    const Tensor host = tmp.to(MemorySpace::Host, DeviceRef::cpu());
    dst.copy_from(host);
  }
  dst.bump_version();
}

void fill_device(const ops::KernelArgs& k) {
  Tensor& out = *k.out;
  if (fp8(out.dtype())) throw TensorError("Nexus cuda: FP8 cihaz doldurma kayitli degil");
  const Tensor one = Tensor::from_values(Shape{1}, std::span<const double>(&k.scalar, 1), out.dtype())
                         .to(MemorySpace::Device, out.device());
  launch_copy(out, one.expand(out.shape()));
  out.bump_version();
}

template <typename T, typename Op>
void binary_device(const ops::KernelArgs& k) {
  require_same_device(*k.out, *k.a);
  require_same_device(*k.out, *k.b);
  const long long n = static_cast<long long>(k.out->numel());
  if (n == 0) return;
  DeviceGuard guard(device_of(*k.out));
  const bool flat = k.out->is_contiguous() && k.a->is_contiguous() && k.b->is_contiguous() && k.a->shape() == k.out->shape() &&
                    k.b->shape() == k.out->shape();
  binary_kernel<T, Op><<<blocks_for(n), kThreads, 0, kStream>>>(static_cast<T*>(base(*k.out)), static_cast<const T*>(base(*k.a)),
                                                               static_cast<const T*>(base(*k.b)), make_geo(*k.out, k.a, k.b), n, flat);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename Acc, typename O>
void sum_device(const ops::KernelArgs& k) {
  const long long n = static_cast<long long>(k.a->numel());
  if (n == 0) return;
  DeviceGuard guard(device_of(*k.a));
  const unsigned blocks = blocks_for(n) < 1024 ? blocks_for(n) : 1024;  // sabit izgara -> belirlenimci
  Tensor partial = Tensor::empty(Shape{blocks}, sizeof(Acc) == 8 && std::is_integral_v<Acc> ? DType::I64 : DType::F64,
                                 MemorySpace::Device, k.a->device());
  // a'yi kendi seklinde gez: out geometrisi olarak a'nin sekli, a adimlari
  Geo g = make_geo(*k.a, k.a, nullptr);
  sum_partial_kernel<T, Acc><<<blocks, kThreads, 0, kStream>>>(static_cast<const T*>(base(*k.a)), g, n,
                                                               static_cast<Acc*>(partial.raw_data()));
  NEXUS_CUDA_CHECK(cudaGetLastError());
  sum_final_kernel<Acc, O><<<1, 1, 0, kStream>>>(static_cast<const Acc*>(partial.raw_data()), static_cast<int>(blocks),
                                                 static_cast<O*>(k.out->raw_data()));
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

bool gemm_layout(const Tensor& t, bool& trans, int& ld) {
  const std::int64_t s0 = t.strides()[0], s1 = t.strides()[1];
  if (s1 == 1 && s0 >= static_cast<std::int64_t>(t.shape()[1]) && s0 >= 1) {
    trans = false;
    ld = static_cast<int>(s0);
    return true;
  }
  if (s0 == 1 && s1 >= static_cast<std::int64_t>(t.shape()[0]) && s1 >= 1) {
    trans = true;
    ld = static_cast<int>(s1);
    return true;
  }
  return false;
}

void matmul_device(const ops::KernelArgs& k) {
  require_same_device(*k.a, *k.b);
  DeviceGuard guard(device_of(*k.a));
  Tensor A = k.a->shallow(), B = k.b->shallow();
  bool ta = false, tb = false;
  int lda = 0, ldb = 0;
  if (!gemm_layout(A, ta, lda)) {
    A = A.contiguous();
    gemm_layout(A, ta, lda);
  }
  if (!gemm_layout(B, tb, ldb)) {
    B = B.contiguous();
    gemm_layout(B, tb, ldb);
  }
  GemmArgs g;
  g.trans_a = ta;
  g.trans_b = tb;
  g.m = static_cast<int>(A.shape()[0]);
  g.k = static_cast<int>(A.shape()[1]);
  g.n = static_cast<int>(B.shape()[1]);
  g.a = A.raw_data();
  g.lda = lda;
  g.b = B.raw_data();
  g.ldb = ldb;
  g.c = k.out->raw_data();
  g.ldc = g.n;
  g.a_type = g.b_type = g.c_type = A.dtype();
  gemm(g, kStream);
}

void register_all() {
  using ops::DeviceKind;
  using ops::Op;
  constexpr auto D = DeviceKind::Device;
  for (const DType t : {DType::F32, DType::F16, DType::BF16, DType::F64, DType::I32, DType::I64, DType::I8, DType::U8, DType::Bool}) {
    ops::register_kernel(Op::Copy, t, D, copy_device);
    ops::register_kernel(Op::Fill, t, D, fill_device);
  }
  ops::register_kernel(Op::Add, DType::F32, D, binary_device<float, AddOp>);
  ops::register_kernel(Op::Add, DType::F64, D, binary_device<double, AddOp>);
  ops::register_kernel(Op::Add, DType::F16, D, binary_device<__half, AddOp>);
  ops::register_kernel(Op::Add, DType::BF16, D, binary_device<__nv_bfloat16, AddOp>);
  ops::register_kernel(Op::Add, DType::I32, D, binary_device<int, AddOp>);
  ops::register_kernel(Op::Add, DType::I64, D, binary_device<long long, AddOp>);
  ops::register_kernel(Op::Mul, DType::F32, D, binary_device<float, MulOp>);
  ops::register_kernel(Op::Mul, DType::F64, D, binary_device<double, MulOp>);
  ops::register_kernel(Op::Mul, DType::F16, D, binary_device<__half, MulOp>);
  ops::register_kernel(Op::Mul, DType::BF16, D, binary_device<__nv_bfloat16, MulOp>);
  ops::register_kernel(Op::Mul, DType::I32, D, binary_device<int, MulOp>);
  ops::register_kernel(Op::Mul, DType::I64, D, binary_device<long long, MulOp>);
  ops::register_kernel(Op::Sum, DType::F32, D, sum_device<float, double, float>);
  ops::register_kernel(Op::Sum, DType::F64, D, sum_device<double, double, double>);
  ops::register_kernel(Op::Sum, DType::F16, D, sum_device<__half, double, __half>);
  ops::register_kernel(Op::Sum, DType::BF16, D, sum_device<__nv_bfloat16, double, __nv_bfloat16>);
  ops::register_kernel(Op::Sum, DType::I32, D, sum_device<int, long long, long long>);
  ops::register_kernel(Op::Sum, DType::I64, D, sum_device<long long, long long, long long>);
  ops::register_kernel(Op::Matmul, DType::F32, D, matmul_device);
  ops::register_kernel(Op::Matmul, DType::F16, D, matmul_device);
  ops::register_kernel(Op::Matmul, DType::BF16, D, matmul_device);
}

}  // namespace

void register_tensor_kernels() {
  static std::once_flag once;
  std::call_once(once, register_all);
}

}  // namespace cuda
}  // namespace nexus
