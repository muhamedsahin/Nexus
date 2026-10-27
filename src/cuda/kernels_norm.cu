/**
 * @file kernels_norm.cu
 * @brief FAZ 3.2: LayerNorm / RMSNorm / add&norm ileri-geri (FP32 istatistik).
 *
 * Ileri: satir basina bir warp, Welford (Chan birlestirmesi) ile ortalama/varyans.
 * Geri (belirlenimci, atomiksiz):
 *   k1 satir istatistikleri  (warp/satir): mean(dnorm), mean(dnorm * xhat)
 *   k2 dx                    (eleman eleman)
 *   k3 sutun kismi toplamlari (blok = 256 satirlik dilim x 256 sutun)
 *   k4 dilimlerin sabit sirali toplami -> dweight, dbias (+=)
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "common.cuh"
#include "nexus/core/error.hpp"
#include "nexus/cuda/kernels.hpp"

#include <algorithm>

namespace nexus {
namespace cuda {
namespace {

using detail::kWarp;

constexpr int kRowsPerChunk = 256;
constexpr int kNormThreads = 256;
constexpr int kRowsPerBlock = kNormThreads / kWarp;

struct Welford {
  float mean = 0.0f;
  float m2 = 0.0f;
  float n = 0.0f;
};

__device__ __forceinline__ Welford welford_merge(Welford a, Welford b) {
  const float n = a.n + b.n;
  if (n == 0.0f) {
    return a;
  }
  const float delta = b.mean - a.mean;
  Welford out;
  out.n = n;
  out.mean = a.mean + delta * (b.n / n);
  out.m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / n);
  return out;
}

__device__ __forceinline__ Welford warp_welford(Welford w) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    Welford other;
    other.mean = __shfl_xor_sync(0xFFFFFFFFu, w.mean, offset);
    other.m2 = __shfl_xor_sync(0xFFFFFFFFu, w.m2, offset);
    other.n = __shfl_xor_sync(0xFFFFFFFFu, w.n, offset);
    w = welford_merge(w, other);
  }
  return w;
}

__device__ __forceinline__ void row_moments(const float* row, int cols, float& mean, float& var) {
  const int lane = threadIdx.x % kWarp;
  Welford w;
  for (int j = lane; j < cols; j += kWarp) {
    const float x = row[j];
    w.n += 1.0f;
    const float delta = x - w.mean;
    w.mean += delta / w.n;
    w.m2 += delta * (x - w.mean);
  }
  w = warp_welford(w);
  mean = w.mean;
  var = w.m2 / static_cast<float>(cols);
}

template <typename OutT>
__global__ void layernorm_forward_kernel(OutT* __restrict__ out, float* __restrict__ mean_out,
                                         float* __restrict__ rstd_out, const float* __restrict__ x,
                                         const float* __restrict__ weight, const float* __restrict__ bias, int rows,
                                         int cols, float eps) {
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
  if (row >= rows) {
    return;
  }
  const float* xr = x + static_cast<std::size_t>(row) * cols;
  float mean = 0.0f;
  float var = 0.0f;
  row_moments(xr, cols, mean, var);
  const float rstd = 1.0f / sqrtf(var + eps);
  OutT* orow = out + static_cast<std::size_t>(row) * cols;
  for (int j = threadIdx.x % kWarp; j < cols; j += kWarp) {
    orow[j] = detail::Num<OutT>::store((xr[j] - mean) * rstd * weight[j] + bias[j]);
  }
  if (threadIdx.x % kWarp == 0) {
    if (mean_out != nullptr) {
      mean_out[row] = mean;
    }
    rstd_out[row] = rstd;
  }
}

template <typename OutT>
__global__ void residual_layernorm_forward_kernel(float* __restrict__ sum, OutT* __restrict__ out,
                                                  float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                  const float* __restrict__ a, const float* __restrict__ b,
                                                  const float* __restrict__ weight, const float* __restrict__ bias,
                                                  int rows, int cols, float eps) {
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
  if (row >= rows) {
    return;
  }
  const std::size_t base = static_cast<std::size_t>(row) * cols;
  for (int j = threadIdx.x % kWarp; j < cols; j += kWarp) {
    sum[base + j] = a[base + j] + b[base + j];
  }
  __syncwarp();
  float mean = 0.0f;
  float var = 0.0f;
  row_moments(sum + base, cols, mean, var);
  const float rstd = 1.0f / sqrtf(var + eps);
  for (int j = threadIdx.x % kWarp; j < cols; j += kWarp) {
    out[base + j] = detail::Num<OutT>::store((sum[base + j] - mean) * rstd * weight[j] + bias[j]);
  }
  if (threadIdx.x % kWarp == 0) {
    mean_out[row] = mean;
    rstd_out[row] = rstd;
  }
}

// k1: satir istatistikleri. LN: m1 = mean(dnorm), m2 = mean(dnorm * xhat). RMS: m2 = mean(dnorm * x).
template <bool kRms>
__global__ void norm_row_stats_kernel(float2* __restrict__ stats, const float* __restrict__ dout,
                                      const float* __restrict__ x, const float* __restrict__ weight,
                                      const float* __restrict__ mean, const float* __restrict__ rstd, int rows,
                                      int cols) {
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
  if (row >= rows) {
    return;
  }
  const std::size_t base = static_cast<std::size_t>(row) * cols;
  const float mu = kRms ? 0.0f : mean[row];
  const float r = rstd[row];
  float s1 = 0.0f;
  float s2 = 0.0f;
  for (int j = threadIdx.x % kWarp; j < cols; j += kWarp) {
    const float dn = dout[base + j] * weight[j];
    s1 += dn;
    s2 += kRms ? dn * x[base + j] : dn * (x[base + j] - mu) * r;
  }
  s1 = detail::warp_sum(s1);
  s2 = detail::warp_sum(s2);
  if (threadIdx.x % kWarp == 0) {
    stats[row] = make_float2(s1 / static_cast<float>(cols), s2 / static_cast<float>(cols));
  }
}

template <bool kRms>
__global__ void norm_dx_kernel(float* __restrict__ dx, const float* __restrict__ dout, const float* __restrict__ x,
                               const float* __restrict__ weight, const float* __restrict__ mean,
                               const float* __restrict__ rstd, const float2* __restrict__ stats, int rows, int cols,
                               bool accumulate) {
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(i / cols);
    const int col = static_cast<int>(i % cols);
    const float r = rstd[row];
    const float dn = dout[i] * weight[col];
    float value;
    if (kRms) {
      // dx = r*dn - x*r^3*mean(dn*x)
      value = r * dn - x[i] * r * r * r * stats[row].y;
    } else {
      const float xhat = (x[i] - mean[row]) * r;
      value = r * (dn - stats[row].x - xhat * stats[row].y);
    }
    dx[i] = accumulate ? dx[i] + value : value;
  }
}

// k3: blockIdx.y = satir dilimi, her ip bir sutun.
template <bool kRms>
__global__ void norm_param_partials_kernel(float* __restrict__ partial_w, float* __restrict__ partial_b,
                                           const float* __restrict__ dout, const float* __restrict__ x,
                                           const float* __restrict__ mean, const float* __restrict__ rstd, int rows,
                                           int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  const int start = blockIdx.y * kRowsPerChunk;
  const int end = min(rows, start + kRowsPerChunk);
  float sw = 0.0f;
  float sb = 0.0f;
  for (int row = start; row < end; ++row) {
    const std::size_t i = static_cast<std::size_t>(row) * cols + col;
    const float xn = kRms ? x[i] * rstd[row] : (x[i] - mean[row]) * rstd[row];
    sw += dout[i] * xn;
    sb += dout[i];
  }
  partial_w[static_cast<std::size_t>(blockIdx.y) * cols + col] = sw;
  if (partial_b != nullptr) {
    partial_b[static_cast<std::size_t>(blockIdx.y) * cols + col] = sb;
  }
}

__global__ void norm_param_final_kernel(float* __restrict__ dweight, float* __restrict__ dbias,
                                        const float* __restrict__ partial_w, const float* __restrict__ partial_b,
                                        int chunks, int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  float sw = 0.0f;
  float sb = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    sw += partial_w[static_cast<std::size_t>(c) * cols + col];
    if (partial_b != nullptr) {
      sb += partial_b[static_cast<std::size_t>(c) * cols + col];
    }
  }
  dweight[col] += sw;
  if (dbias != nullptr) {
    dbias[col] += sb;
  }
}

void validate_norm(int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    throw TensorError("Nexus CUDA norm: rows ve cols pozitif olmali");
  }
}

// ---- Tek gecisli fuzyonlu geri (cols <= 2048) -------------------------------------------
// Sabit 512 warp; warp w satirlari w, w+512, ... isler (donanimdan bagimsiz atama ->
// belirlenimci). Her ip kendi sutunlarinin (lane + 32*i) x, dout degerlerini yazmacta tutar:
// satir istatistigi, dx ve dweight/dbias kismi toplamlari DRAM'den tek okumayla cikar.
constexpr int kFusedWarps = 512;
constexpr int kFusedMaxCols = 2048;

template <bool kRms, int kPerLane>
__global__ void norm_backward_fused_kernel(float* __restrict__ dx, float* __restrict__ partial_w,
                                           float* __restrict__ partial_b, const float* __restrict__ dout,
                                           const float* __restrict__ x, const float* __restrict__ weight,
                                           const float* __restrict__ mean, const float* __restrict__ rstd, int rows,
                                           int cols, bool accumulate) {
  const int warp = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  float acc_w[kPerLane];
  float acc_b[kPerLane];
  float wv[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    acc_w[i] = 0.0f;
    acc_b[i] = 0.0f;
    const int j = lane + kWarp * i;
    wv[i] = j < cols ? weight[j] : 0.0f;
  }
  for (int row = warp; row < rows; row += kFusedWarps) {
    const std::size_t base = static_cast<std::size_t>(row) * cols;
    const float mu = kRms ? 0.0f : mean[row];
    const float r = rstd[row];
    float xs[kPerLane];
    float gs[kPerLane];
    float s1 = 0.0f;
    float s2 = 0.0f;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int j = lane + kWarp * i;
      xs[i] = j < cols ? x[base + j] : 0.0f;
      gs[i] = j < cols ? dout[base + j] : 0.0f;
      const float dn = gs[i] * wv[i];
      const float xn = kRms ? xs[i] * r : (xs[i] - mu) * r;
      s1 += dn;
      s2 += kRms ? dn * xs[i] : dn * xn;
      acc_w[i] += gs[i] * xn;
      acc_b[i] += gs[i];
    }
    s1 = detail::warp_sum(s1) / static_cast<float>(cols);
    s2 = detail::warp_sum(s2) / static_cast<float>(cols);
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int j = lane + kWarp * i;
      if (j < cols) {
        const float dn = gs[i] * wv[i];
        float value;
        if (kRms) {
          value = r * dn - xs[i] * r * r * r * s2;
        } else {
          const float xn = (xs[i] - mu) * r;
          value = r * (dn - s1 - xn * s2);
        }
        dx[base + j] = accumulate ? dx[base + j] + value : value;
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const int j = lane + kWarp * i;
    if (j < cols) {
      partial_w[static_cast<std::size_t>(warp) * cols + j] = acc_w[i];
      if (partial_b != nullptr) {
        partial_b[static_cast<std::size_t>(warp) * cols + j] = acc_b[i];
      }
    }
  }
}

template <bool kRms, int kPerLane>
void launch_fused(float* dx, float* partial_w, float* partial_b, const float* dout, const float* x,
                  const float* weight, const float* mean, const float* rstd, int rows, int cols, bool accumulate,
                  cudaStream_t stream) {
  norm_backward_fused_kernel<kRms, kPerLane><<<kFusedWarps / kRowsPerBlock, kNormThreads, 0, stream>>>(
      dx, partial_w, partial_b, dout, x, weight, mean, rstd, rows, cols, accumulate);
}

template <bool kRms>
bool norm_backward_fused(float* dx, float* dweight, float* dbias, const float* dout, const float* x,
                         const float* weight, const float* mean, const float* rstd, int rows, int cols,
                         bool accumulate_dx, float* scratch, cudaStream_t stream) {
  if (cols > kFusedMaxCols) {
    return false;
  }
  float* partial_w = scratch;
  float* partial_b = scratch + static_cast<std::size_t>(kFusedWarps) * cols;
  float* pb = kRms ? nullptr : partial_b;
  const int per_lane = (cols + kWarp - 1) / kWarp;
  if (per_lane <= 8) {
    launch_fused<kRms, 8>(dx, partial_w, pb, dout, x, weight, mean, rstd, rows, cols, accumulate_dx, stream);
  } else if (per_lane <= 16) {
    launch_fused<kRms, 16>(dx, partial_w, pb, dout, x, weight, mean, rstd, rows, cols, accumulate_dx, stream);
  } else if (per_lane <= 24) {
    launch_fused<kRms, 24>(dx, partial_w, pb, dout, x, weight, mean, rstd, rows, cols, accumulate_dx, stream);
  } else if (per_lane <= 32) {
    launch_fused<kRms, 32>(dx, partial_w, pb, dout, x, weight, mean, rstd, rows, cols, accumulate_dx, stream);
  } else if (per_lane <= 48) {
    launch_fused<kRms, 48>(dx, partial_w, pb, dout, x, weight, mean, rstd, rows, cols, accumulate_dx, stream);
  } else {
    launch_fused<kRms, 64>(dx, partial_w, pb, dout, x, weight, mean, rstd, rows, cols, accumulate_dx, stream);
  }
  detail::check_launch("norm_backward_fused_kernel");
  norm_param_final_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(dweight, kRms ? nullptr : dbias, partial_w, pb,
                                                                  kFusedWarps, cols);
  detail::check_launch("norm_param_final_kernel");
  return true;
}

template <bool kRms>
void norm_backward(float* dx, float* dweight, float* dbias, const float* dout, const float* x, const float* weight,
                   const float* mean, const float* rstd, int rows, int cols, bool accumulate_dx, float* scratch,
                   cudaStream_t stream) {
  validate_norm(rows, cols);
  if (scratch == nullptr) {
    throw TensorError("Nexus CUDA norm geri: scratch gerekli (layernorm_backward_scratch_bytes)");
  }
  if (norm_backward_fused<kRms>(dx, dweight, dbias, dout, x, weight, mean, rstd, rows, cols, accumulate_dx, scratch,
                                stream)) {
    return;
  }
  // Genis satirlar (cols > 2048): dort cekirdekli yol (yine atomiksiz).
  const int chunks = (rows + kRowsPerChunk - 1) / kRowsPerChunk;
  auto* stats = reinterpret_cast<float2*>(scratch);
  float* partial_w = scratch + 2 * static_cast<std::size_t>(rows);
  float* partial_b = partial_w + static_cast<std::size_t>(chunks) * cols;
  const unsigned int row_blocks = static_cast<unsigned int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  norm_row_stats_kernel<kRms><<<row_blocks, kNormThreads, 0, stream>>>(stats, dout, x, weight, mean, rstd, rows, cols);
  detail::check_launch("norm_row_stats_kernel");
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  norm_dx_kernel<kRms><<<detail::blocks_for(total, 256), 256, 0, stream>>>(dx, dout, x, weight, mean, rstd, stats,
                                                                            rows, cols, accumulate_dx);
  detail::check_launch("norm_dx_kernel");
  const dim3 grid((cols + 255) / 256, static_cast<unsigned int>(chunks));
  norm_param_partials_kernel<kRms><<<grid, 256, 0, stream>>>(partial_w, kRms ? nullptr : partial_b, dout, x, mean,
                                                             rstd, rows, cols);
  detail::check_launch("norm_param_partials_kernel");
  norm_param_final_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(dweight, kRms ? nullptr : dbias, partial_w,
                                                                  kRms ? nullptr : partial_b, chunks, cols);
  detail::check_launch("norm_param_final_kernel");
}

__global__ void rmsnorm_forward_kernel(float* __restrict__ out, float* __restrict__ rstd_out,
                                       const float* __restrict__ x, const float* __restrict__ weight, int rows,
                                       int cols, float eps) {
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
  if (row >= rows) {
    return;
  }
  const float* xr = x + static_cast<std::size_t>(row) * cols;
  float sq = 0.0f;
  for (int j = threadIdx.x % kWarp; j < cols; j += kWarp) {
    sq += xr[j] * xr[j];
  }
  sq = detail::warp_sum(sq);
  const float r = 1.0f / sqrtf(sq / static_cast<float>(cols) + eps);
  float* orow = out + static_cast<std::size_t>(row) * cols;
  for (int j = threadIdx.x % kWarp; j < cols; j += kWarp) {
    orow[j] = xr[j] * r * weight[j];
  }
  if (threadIdx.x % kWarp == 0) {
    rstd_out[row] = r;
  }
}

}  // namespace

void layernorm_forward(float* out, float* mean, float* rstd, const float* x, const float* weight, const float* bias,
                       int rows, int cols, float eps, cudaStream_t stream) {
  validate_norm(rows, cols);
  const unsigned int blocks = static_cast<unsigned int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  layernorm_forward_kernel<float><<<blocks, kNormThreads, 0, stream>>>(out, mean, rstd, x, weight, bias, rows, cols, eps);
  detail::check_launch("layernorm_forward_kernel");
}

std::size_t layernorm_backward_scratch_bytes(int rows, int cols) {
  const std::size_t chunks = (static_cast<std::size_t>(rows) + kRowsPerChunk - 1) / kRowsPerChunk;
  const std::size_t general = 2 * static_cast<std::size_t>(rows) + 2 * chunks * static_cast<std::size_t>(cols);
  const std::size_t fused = 2 * static_cast<std::size_t>(kFusedWarps) * static_cast<std::size_t>(cols);
  return sizeof(float) * (cols <= kFusedMaxCols ? std::max(general, fused) : general);
}

void layernorm_backward(float* dx, float* dweight, float* dbias, const float* dout, const float* x,
                        const float* weight, const float* mean, const float* rstd, int rows, int cols,
                        bool accumulate_dx, float* scratch, cudaStream_t stream) {
  norm_backward<false>(dx, dweight, dbias, dout, x, weight, mean, rstd, rows, cols, accumulate_dx, scratch, stream);
}

void rmsnorm_forward(float* out, float* rstd, const float* x, const float* weight, int rows, int cols, float eps,
                     cudaStream_t stream) {
  validate_norm(rows, cols);
  const unsigned int blocks = static_cast<unsigned int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  rmsnorm_forward_kernel<<<blocks, kNormThreads, 0, stream>>>(out, rstd, x, weight, rows, cols, eps);
  detail::check_launch("rmsnorm_forward_kernel");
}

void rmsnorm_backward(float* dx, float* dweight, const float* dout, const float* x, const float* weight,
                      const float* rstd, int rows, int cols, bool accumulate_dx, float* scratch, cudaStream_t stream) {
  norm_backward<true>(dx, dweight, nullptr, dout, x, weight, nullptr, rstd, rows, cols, accumulate_dx, scratch,
                      stream);
}

void residual_layernorm_forward(float* sum, float* out, float* mean, float* rstd, const float* a, const float* b,
                                const float* weight, const float* bias, int rows, int cols, float eps,
                                cudaStream_t stream) {
  validate_norm(rows, cols);
  const unsigned int blocks = static_cast<unsigned int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  residual_layernorm_forward_kernel<float><<<blocks, kNormThreads, 0, stream>>>(sum, out, mean, rstd, a, b, weight,
                                                                                bias, rows, cols, eps);
  detail::check_launch("residual_layernorm_forward_kernel");
}

void layernorm_forward(void* out, DType out_type, float* mean, float* rstd, const float* x, const float* weight,
                       const float* bias, int rows, int cols, float eps, cudaStream_t stream) {
  validate_norm(rows, cols);
  const unsigned int blocks = static_cast<unsigned int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  detail::dispatch_storage(out_type, "layernorm_forward", [&](auto tag) {
    using T = decltype(tag);
    layernorm_forward_kernel<T><<<blocks, kNormThreads, 0, stream>>>(static_cast<T*>(out), mean, rstd, x, weight, bias,
                                                                     rows, cols, eps);
  });
  detail::check_launch("layernorm_forward_kernel");
}

void residual_layernorm_forward(float* sum, void* out, DType out_type, float* mean, float* rstd, const float* a,
                                const float* b, const float* weight, const float* bias, int rows, int cols, float eps,
                                cudaStream_t stream) {
  validate_norm(rows, cols);
  const unsigned int blocks = static_cast<unsigned int>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  detail::dispatch_storage(out_type, "residual_layernorm_forward", [&](auto tag) {
    using T = decltype(tag);
    residual_layernorm_forward_kernel<T><<<blocks, kNormThreads, 0, stream>>>(sum, static_cast<T*>(out), mean, rstd, a,
                                                                              b, weight, bias, rows, cols, eps);
  });
  detail::check_launch("residual_layernorm_forward_kernel");
}

}  // namespace cuda
}  // namespace nexus
