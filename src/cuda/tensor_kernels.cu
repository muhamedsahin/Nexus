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
#include <cmath>
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
struct SubOp { template <typename C> __device__ C operator()(C x, C y) const { return x - y; } };
struct DivOp { template <typename C> __device__ C operator()(C x, C y) const { return x / y; } };

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


// ---- Dalga 2: autograd Tensor yolu cihaz cekirdekleri (girdiler bitisik; API girisleri garanti eder) -----------
template <typename C>
__device__ __forceinline__ C unary_d(int fn, C x, C s) {
  const C kInvSqrt2 = static_cast<C>(0.7071067811865475);
  switch (static_cast<ops::UnaryFn>(fn)) {
    case ops::UnaryFn::Neg: return -x;
    case ops::UnaryFn::AddScalar: return x + s;
    case ops::UnaryFn::MulScalar: return x * s;
    case ops::UnaryFn::Pow: return pow(x, s);
    case ops::UnaryFn::Relu: return x > C(0) ? x : C(0);
    case ops::UnaryFn::Sigmoid: return C(1) / (C(1) + exp(-x));
    case ops::UnaryFn::Tanh: return tanh(x);
    case ops::UnaryFn::Exp: return exp(x);
    case ops::UnaryFn::Log: return log(x);
    case ops::UnaryFn::Gelu: return C(0.5) * x * (C(1) + erf(x * kInvSqrt2));
    case ops::UnaryFn::Silu: return x / (C(1) + exp(-x));
  }
  return x;
}

template <typename C>
__device__ __forceinline__ C unary_grad_d(int fn, C g, C v, C s) {
  const C kInvSqrt2 = static_cast<C>(0.7071067811865475);
  const C kInvSqrt2Pi = static_cast<C>(0.3989422804014327);
  switch (static_cast<ops::UnaryFn>(fn)) {
    case ops::UnaryFn::Neg: return -g;
    case ops::UnaryFn::AddScalar: return g;
    case ops::UnaryFn::MulScalar: return g * s;
    case ops::UnaryFn::Pow: return g * s * pow(v, s - C(1));
    case ops::UnaryFn::Relu: return v > C(0) ? g : C(0);
    case ops::UnaryFn::Sigmoid: return g * v * (C(1) - v);
    case ops::UnaryFn::Tanh: return g * (C(1) - v * v);
    case ops::UnaryFn::Exp: return g * v;
    case ops::UnaryFn::Log: return g / v;
    case ops::UnaryFn::Gelu: {
      const C cdf = C(0.5) * (C(1) + erf(v * kInvSqrt2));
      return g * (cdf + v * kInvSqrt2Pi * exp(C(-0.5) * v * v));
    }
    case ops::UnaryFn::Silu: {
      const C sig = C(1) / (C(1) + exp(-v));
      return g * sig * (C(1) + v * (C(1) - sig));
    }
  }
  return g;
}

template <typename T>
__global__ void unary_kernel(T* y, const T* x, long long n, int fn, double s) {
  using K = Calc<T>;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x)
    y[i] = K::st(unary_d<typename K::C>(fn, K::ld(x[i]), static_cast<typename K::C>(s)));
}

template <typename T>
__global__ void unary_grad_kernel(T* d, const T* g, const T* v, long long n, int fn, double s) {
  using K = Calc<T>;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x)
    d[i] = K::st(unary_grad_d<typename K::C>(fn, K::ld(g[i]), K::ld(v[i]), static_cast<typename K::C>(s)));
}

template <typename A>
__device__ A block_reduce_sum(A v, A* sh) {
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  const A r = sh[0];
  __syncthreads();
  return r;
}

template <typename A>
__device__ A block_reduce_max(A v, A* sh) {
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + s]);
    __syncthreads();
  }
  const A r = sh[0];
  __syncthreads();
  return r;
}

// blok basina bir satir
template <typename T>
__global__ void row_softmax_kernel(T* y, const T* x, long long c, int log_mode) {
  using K = Calc<T>;
  using C = typename K::C;
  __shared__ double shd[kThreads];
  __shared__ C shc[kThreads];
  const T* row = x + blockIdx.x * c;
  T* out = y + blockIdx.x * c;
  C mx = -INFINITY;
  for (long long j = threadIdx.x; j < c; j += blockDim.x) mx = fmax(mx, K::ld(row[j]));
  mx = block_reduce_max<C>(mx, shc);
  double total = 0;
  for (long long j = threadIdx.x; j < c; j += blockDim.x) total += exp(static_cast<double>(K::ld(row[j]) - mx));
  total = block_reduce_sum<double>(total, shd);
  if (log_mode) {
    const C lse = mx + static_cast<C>(log(total));
    for (long long j = threadIdx.x; j < c; j += blockDim.x) out[j] = K::st(K::ld(row[j]) - lse);
  } else {
    for (long long j = threadIdx.x; j < c; j += blockDim.x) out[j] = K::st(static_cast<C>(exp(K::ld(row[j]) - mx) / static_cast<C>(total)));
  }
}

template <typename T>
__global__ void row_softmax_grad_kernel(T* d, const T* g, const T* y, long long c, int log_mode) {
  using K = Calc<T>;
  using C = typename K::C;
  __shared__ double shd[kThreads];
  const long long base = blockIdx.x * c;
  double acc = 0;
  for (long long j = threadIdx.x; j < c; j += blockDim.x)
    acc += log_mode ? static_cast<double>(K::ld(g[base + j])) : static_cast<double>(K::ld(g[base + j])) * K::ld(y[base + j]);
  const C dot = static_cast<C>(block_reduce_sum<double>(acc, shd));
  for (long long j = threadIdx.x; j < c; j += blockDim.x) {
    const C gj = K::ld(g[base + j]), yj = K::ld(y[base + j]);
    d[base + j] = K::st(log_mode ? gj - exp(yj) * dot : yj * (gj - dot));
  }
}

// CE: satir basina kayip (kismi), sonra sirali toplam / N. Olasilik FP32 (alt sinir 1e-12).
template <typename T>
__device__ void ce_row_stats(const T* row, long long c, float* sh, float& mx, float& total) {
  using K = Calc<T>;
  float m = -INFINITY;
  for (long long j = threadIdx.x; j < c; j += blockDim.x) m = fmaxf(m, static_cast<float>(K::ld(row[j])));
  mx = block_reduce_max<float>(m, sh);
  double t = 0;
  for (long long j = threadIdx.x; j < c; j += blockDim.x) t += expf(static_cast<float>(K::ld(row[j])) - mx);
  __shared__ double shd[kThreads];
  total = static_cast<float>(block_reduce_sum<double>(t, shd));
}

template <typename T>
__global__ void ce_rows_kernel(const T* x, const long long* tgt, long long c, float* row_loss, int* err) {
  using K = Calc<T>;
  __shared__ float sh[kThreads];
  const T* row = x + blockIdx.x * c;
  float mx, total;
  ce_row_stats<T>(row, c, sh, mx, total);
  if (threadIdx.x == 0) {
    const long long t = tgt[blockIdx.x];
    if (t < 0 || t >= c) {
      atomicExch(err, 3);
      row_loss[blockIdx.x] = 0;
      return;
    }
    const float p = expf(static_cast<float>(K::ld(row[t])) - mx) / total;
    row_loss[blockIdx.x] = -logf(fmaxf(p, 1.0e-12f));
  }
}

__global__ void ce_final_kernel(const float* row_loss, long long n, float* out) {
  float acc = 0;
  for (long long i = 0; i < n; ++i) acc += row_loss[i];  // sabit sira
  *out = acc / static_cast<float>(n);
}

template <typename T>
__global__ void ce_grad_kernel(T* d, const T* x, const long long* tgt, const float* g, long long n, long long c) {
  using K = Calc<T>;
  __shared__ float sh[kThreads];
  const T* row = x + blockIdx.x * c;
  float mx, total;
  ce_row_stats<T>(row, c, sh, mx, total);
  const float scale = *g / static_cast<float>(n);
  const long long t = tgt[blockIdx.x];
  for (long long j = threadIdx.x; j < c; j += blockDim.x) {
    const float p = expf(static_cast<float>(K::ld(row[j])) - mx) / total;
    d[blockIdx.x * c + j] = K::st(static_cast<typename K::C>((p - (j == t ? 1.0f : 0.0f)) * scale));
  }
}

template <typename T>
__global__ void reduce_rows_kernel(T* y, const T* x, long long rows, long long c) {
  using K = Calc<T>;
  for (long long j = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; j < c; j += static_cast<long long>(gridDim.x) * blockDim.x) {
    double acc = 0;
    for (long long i = 0; i < rows; ++i) acc += static_cast<double>(K::ld(x[i * c + j]));  // sabit sira
    y[j] = K::st(static_cast<typename K::C>(acc));
  }
}

template <typename T>
__global__ void gather_rows_kernel(T* y, const T* w, const long long* idx, long long L, long long D, long long V, int* err) {
  const long long n = L * D;
  for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < n; e += static_cast<long long>(gridDim.x) * blockDim.x) {
    const long long i = e / D, j = e % D;
    const long long r = idx[i];
    if (r < 0 || r >= V) {
      atomicExch(err, 4);
      continue;
    }
    y[e] = w[r * D + j];
  }
}

template <typename T>
__global__ void embedding_grad_kernel(T* d, const T* g, const long long* plan, long long L, long long U, long long D, long long pad) {
  using K = Calc<T>;
  const long long* perm = plan;
  const long long* seg_row = plan + L;
  const long long* seg_off = plan + L + U;
  const long long n = U * D;
  for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < n; e += static_cast<long long>(gridDim.x) * blockDim.x) {
    const long long s = e / D, j = e % D;
    if (seg_row[s] == pad) continue;
    typename K::C acc = 0;
    for (long long q = seg_off[s]; q < seg_off[s + 1]; ++q) acc += K::ld(g[perm[q] * D + j]);  // indeks sirasi
    d[seg_row[s] * D + j] = K::st(acc);
  }
}

template <typename T>
__global__ void sq_partial_kernel(const T* x, long long n, double* partial) {
  __shared__ double sh[kThreads];
  double acc = 0;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const double v = static_cast<double>(Calc<T>::ld(x[i]));
    acc += v * v;
  }
  acc = block_reduce_sum<double>(acc, sh);
  if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}

__global__ void sq_final_kernel(const double* partial, int count, double* out) {
  double acc = 0;
  for (int i = 0; i < count; ++i) acc += partial[i];
  *out = acc;
}

// FAZ 2 / CPU ile ayni sira; __f*_rn: nvcc'nin FMA kasilmasini engeller (CPU ile bit bit)
template <typename T>
__global__ void adamw_kernel(T* p, const T* g, float* m, float* v, long long n, float b1, float ob1, float b2, float ob2, float inv1,
                             float inv2, float lr, float lr_wd, float eps, const float* clip, int decay) {
  using K = Calc<T>;
  const float cl = *clip;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    float pp = static_cast<float>(K::ld(p[i]));
    const float gg = __fmul_rn(static_cast<float>(K::ld(g[i])), cl);
    if (decay) pp = __fsub_rn(pp, __fmul_rn(lr_wd, pp));
    const float mt = __fadd_rn(__fmul_rn(b1, m[i]), __fmul_rn(ob1, gg));
    const float vt = __fadd_rn(__fmul_rn(b2, v[i]), __fmul_rn(__fmul_rn(ob2, gg), gg));
    m[i] = mt;
    v[i] = vt;
    const float upd = __fdiv_rn(__fmul_rn(lr, __fmul_rn(mt, inv1)), __fadd_rn(__fsqrt_rn(__fmul_rn(vt, inv2)), eps));
    p[i] = K::st(static_cast<typename K::C>(__fsub_rn(pp, upd)));
  }
}

// ---- host sarmalayicilari ------------------------------------------------------------------------------------------
template <typename T> T* dp(const Tensor& t) { return static_cast<T*>(t.raw_data()); }

void check_flag(int device, const char* what) {
  int h = 0;
  NEXUS_CUDA_CHECK(cudaMemcpy(&h, error_flag(device), sizeof(int), cudaMemcpyDeviceToHost));
  if (h) throw TensorError(std::string("Nexus cuda: ") + what);
}

template <typename T>
void unary_dev(const ops::KernelArgs& k) {
  const long long n = static_cast<long long>(k.out->numel());
  DeviceGuard guard(device_of(*k.out));
  unary_kernel<T><<<blocks_for(n), kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), n, k.fn, k.scalar);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void unary_grad_dev(const ops::KernelArgs& k) {
  const long long n = static_cast<long long>(k.out->numel());
  DeviceGuard guard(device_of(*k.out));
  unary_grad_kernel<T><<<blocks_for(n), kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), dp<T>(*k.b), n, k.fn, k.scalar);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void row_softmax_dev(const ops::KernelArgs& k) {
  DeviceGuard guard(device_of(*k.out));
  const auto rows = static_cast<unsigned>(k.a->shape()[0]);
  row_softmax_kernel<T><<<rows, kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), static_cast<long long>(k.a->shape()[1]), k.fn);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void row_softmax_grad_dev(const ops::KernelArgs& k) {
  DeviceGuard guard(device_of(*k.out));
  const auto rows = static_cast<unsigned>(k.a->shape()[0]);
  row_softmax_grad_kernel<T><<<rows, kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), dp<T>(*k.b),
                                                            static_cast<long long>(k.a->shape()[1]), k.fn);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void ce_dev(const ops::KernelArgs& k) {
  const int dev = device_of(*k.a);
  DeviceGuard guard(dev);
  const long long n = static_cast<long long>(k.a->shape()[0]), c = static_cast<long long>(k.a->shape()[1]);
  Tensor rows = Tensor::empty(Shape{static_cast<std::size_t>(n)}, DType::F32, MemorySpace::Device, k.a->device());
  int* err = error_flag(dev);
  NEXUS_CUDA_CHECK(cudaMemsetAsync(err, 0, sizeof(int), kStream));
  ce_rows_kernel<T><<<static_cast<unsigned>(n), kThreads, 0, kStream>>>(dp<T>(*k.a), dp<long long>(*k.b), c, dp<float>(rows), err);
  NEXUS_CUDA_CHECK(cudaGetLastError());
  ce_final_kernel<<<1, 1, 0, kStream>>>(dp<float>(rows), n, dp<float>(*k.out));
  NEXUS_CUDA_CHECK(cudaGetLastError());
  check_flag(dev, "cross_entropy hedefi sinif araligi disinda");
}

template <typename T>
void ce_grad_dev(const ops::KernelArgs& k) {
  DeviceGuard guard(device_of(*k.a));
  const long long n = static_cast<long long>(k.a->shape()[0]), c = static_cast<long long>(k.a->shape()[1]);
  ce_grad_kernel<T><<<static_cast<unsigned>(n), kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), dp<long long>(*k.b), dp<float>(*k.c), n, c);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void reduce_rows_dev(const ops::KernelArgs& k) {
  DeviceGuard guard(device_of(*k.a));
  const long long c = static_cast<long long>(k.out->numel()), rows = static_cast<long long>(k.a->numel()) / c;
  reduce_rows_kernel<T><<<blocks_for(c), kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), rows, c);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void gather_rows_dev(const ops::KernelArgs& k) {
  const int dev = device_of(*k.a);
  DeviceGuard guard(dev);
  const long long L = static_cast<long long>(k.b->numel()), D = static_cast<long long>(k.a->shape()[1]),
                  V = static_cast<long long>(k.a->shape()[0]);
  int* err = error_flag(dev);
  NEXUS_CUDA_CHECK(cudaMemsetAsync(err, 0, sizeof(int), kStream));
  gather_rows_kernel<T><<<blocks_for(L * D), kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), dp<long long>(*k.b), L, D, V, err);
  NEXUS_CUDA_CHECK(cudaGetLastError());
  check_flag(dev, "gather_rows / embedding indeksi sozluk disinda");
}

template <typename T>
void embedding_grad_dev(const ops::KernelArgs& k) {
  DeviceGuard guard(device_of(*k.a));
  const auto L = static_cast<long long>(k.p[0]), U = static_cast<long long>(k.p[1]), pad = static_cast<long long>(k.p[2]);
  const long long D = static_cast<long long>(k.out->shape()[1]);
  embedding_grad_kernel<T><<<blocks_for(U * D), kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), dp<long long>(*k.b), L, U, D, pad);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void sq_norm_dev(const ops::KernelArgs& k) {
  DeviceGuard guard(device_of(*k.a));
  const long long n = static_cast<long long>(k.a->numel());
  const unsigned blocks = blocks_for(n) < 1024 ? blocks_for(n) : 1024;
  Tensor partial = Tensor::empty(Shape{blocks}, DType::F64, MemorySpace::Device, k.a->device());
  sq_partial_kernel<T><<<blocks, kThreads, 0, kStream>>>(dp<T>(*k.a), n, dp<double>(partial));
  NEXUS_CUDA_CHECK(cudaGetLastError());
  sq_final_kernel<<<1, 1, 0, kStream>>>(dp<double>(partial), static_cast<int>(blocks), dp<double>(*k.out));
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void adamw_dev(const ops::KernelArgs& k) {
  DeviceGuard guard(device_of(*k.out));
  const long long n = static_cast<long long>(k.out->numel());
  const auto f = [&](int i) { return static_cast<float>(k.p[i]); };
  adamw_kernel<T><<<blocks_for(n), kThreads, 0, kStream>>>(dp<T>(*k.out), dp<T>(*k.a), dp<float>(*k.out2), dp<float>(*k.out3), n, f(0), f(1),
                                                         f(2), f(3), f(4), f(5), f(6), f(7), static_cast<float>(k.scalar),
                                                         dp<float>(*k.c), k.fn);
  NEXUS_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void register_float_dev(DType t) {
  using ops::Op;
  constexpr auto D = ops::DeviceKind::Device;
  ops::register_kernel(Op::Sub, t, D, binary_device<T, SubOp>);
  ops::register_kernel(Op::Div, t, D, binary_device<T, DivOp>);
  ops::register_kernel(Op::Unary, t, D, unary_dev<T>);
  ops::register_kernel(Op::UnaryGrad, t, D, unary_grad_dev<T>);
  ops::register_kernel(Op::RowSoftmax, t, D, row_softmax_dev<T>);
  ops::register_kernel(Op::RowSoftmaxGrad, t, D, row_softmax_grad_dev<T>);
  if (t != DType::F64) {  // cihaz CE FP32 hesaplar ve F32 kayip yazar; F64 logit (F64 kayip) cihazda kayitli degil
    ops::register_kernel(Op::CrossEntropy, t, D, ce_dev<T>);
    ops::register_kernel(Op::CrossEntropyGrad, t, D, ce_grad_dev<T>);
  }
  ops::register_kernel(Op::ReduceRows, t, D, reduce_rows_dev<T>);
  ops::register_kernel(Op::GatherRows, t, D, gather_rows_dev<T>);
  ops::register_kernel(Op::EmbeddingGrad, t, D, embedding_grad_dev<T>);
  ops::register_kernel(Op::SqNorm, t, D, sq_norm_dev<T>);
  if (t != DType::F64) ops::register_kernel(Op::AdamW, t, D, adamw_dev<T>);
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
  register_float_dev<float>(DType::F32);
  register_float_dev<double>(DType::F64);
  register_float_dev<__half>(DType::F16);
  register_float_dev<__nv_bfloat16>(DType::BF16);
}

}  // namespace

void register_tensor_kernels() {
  static std::once_flag once;
  std::call_once(once, register_all);
}

}  // namespace cuda
}  // namespace nexus
