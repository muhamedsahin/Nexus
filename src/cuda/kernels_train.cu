/**
 * @file kernels_train.cu
 * @brief FAZ 3.4-3.7: gomme (+konum) ileri/geri, fuzyonlu kaydirilmis CE,
 *        Philox dropout ve foreach AdamW + global norm kirpma.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "common.cuh"
#include "nexus/core/error.hpp"
#include "nexus/cuda/kernels.hpp"

#include <cub/device/device_radix_sort.cuh>

#include <cmath>
#include <vector>

namespace nexus {
namespace cuda {
namespace {

#define NEXUS_GRID_STRIDE(i, n)                                                                         \
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < (n); \
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)

std::size_t align_up(std::size_t bytes) { return (bytes + 255u) & ~static_cast<std::size_t>(255u); }

// ================================ Gomme ==========================================

__global__ void embedding_pos_forward_kernel(float* __restrict__ out, const std::int32_t* __restrict__ tokens,
                                             const float* __restrict__ wte, const float* __restrict__ wpe, int time,
                                             int dim, std::size_t total) {
  NEXUS_GRID_STRIDE(i, total) {
    const std::size_t row = i / dim;
    const int c = static_cast<int>(i % dim);
    const int t = static_cast<int>(row % time);
    out[i] = wte[static_cast<std::size_t>(tokens[row]) * dim + c] + wpe[static_cast<std::size_t>(t) * dim + c];
  }
}

__global__ void embedding_backward_atomic_kernel(float* __restrict__ dwte, float* __restrict__ dwpe,
                                                 const float* __restrict__ dout, const std::int32_t* __restrict__ tokens,
                                                 int time, int dim, std::size_t total) {
  NEXUS_GRID_STRIDE(i, total) {
    const std::size_t row = i / dim;
    const int c = static_cast<int>(i % dim);
    const int t = static_cast<int>(row % time);
    atomicAdd(&dwte[static_cast<std::size_t>(tokens[row]) * dim + c], dout[i]);
    if (dwpe != nullptr) {
      atomicAdd(&dwpe[static_cast<std::size_t>(t) * dim + c], dout[i]);
    }
  }
}

// dwpe[t, c] += sum_b dout[b, t, c]  (b sirasi sabit -> belirlenimci)
__global__ void wpe_backward_kernel(float* __restrict__ dwpe, const float* __restrict__ dout, int batch, int time,
                                    int dim) {
  const std::size_t total = static_cast<std::size_t>(time) * dim;
  NEXUS_GRID_STRIDE(i, total) {
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
      acc += dout[static_cast<std::size_t>(b) * total + i];
    }
    dwpe[i] += acc;
  }
}

__global__ void iota_kernel(std::int32_t* __restrict__ out, int n) {
  NEXUS_GRID_STRIDE(i, static_cast<std::size_t>(n)) { out[i] = static_cast<std::int32_t>(i); }
}

// Siralanmis (token, konum) ciftlerinde her segmentin ilk elemani segmenti sirayla toplar.
__global__ void embedding_segment_kernel(float* __restrict__ dwte, const float* __restrict__ dout,
                                         const std::int32_t* __restrict__ keys, const std::int32_t* __restrict__ rows,
                                         int n, int dim, int padding_idx) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dim) {
    return;
  }
  for (int i = blockIdx.y; i < n; i += gridDim.y) {
    const std::int32_t key = keys[i];
    if ((i > 0 && keys[i - 1] == key) || key == padding_idx) {
      continue;  // segment basi degil (veya padding)
    }
    float acc = 0.0f;
    for (int j = i; j < n && keys[j] == key; ++j) {
      acc += dout[static_cast<std::size_t>(rows[j]) * dim + c];
    }
    dwte[static_cast<std::size_t>(key) * dim + c] += acc;
  }
}

// ================================ Kaydirilmis CE =================================

constexpr int kCeThreads = 512;

__global__ void ce_count_kernel(std::int32_t* __restrict__ valid, float* __restrict__ inv_valid,
                                const std::int32_t* __restrict__ tokens, int batch, int time, int vocab,
                                std::int32_t ignore_index, int* __restrict__ bad_target) {
  __shared__ float shared[32];
  float count = 0.0f;
  const int rows = batch * time;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const int t = r % time;
    if (t + 1 == time) {
      continue;
    }
    const std::int32_t target = tokens[r + 1];
    if (target == ignore_index) {
      continue;
    }
    if (target < 0 || target >= vocab) {
      atomicExch(bad_target, 1);
      continue;
    }
    count += 1.0f;
  }
  count = detail::block_sum(count, shared);
  if (threadIdx.x == 0) {
    const auto n = static_cast<std::int32_t>(count);
    *valid = n;
    *inv_valid = n > 0 ? 1.0f / static_cast<float>(n) : 0.0f;
  }
}

__global__ void ce_row_kernel(float* __restrict__ dlogits, float* __restrict__ row_loss, const float* logits,
                              const std::int32_t* __restrict__ tokens, int time, int vocab,
                              std::int32_t ignore_index, float smoothing, float z_coef, float grad_scale,
                              const float* __restrict__ inv_valid) {
  __shared__ float shared[32];
  const std::size_t row = blockIdx.x;
  const int t = static_cast<int>(row % time);
  const float* x = logits + row * vocab;
  float* d = dlogits != nullptr ? dlogits + row * vocab : nullptr;
  const std::int32_t target = (t + 1 == time) ? ignore_index : tokens[row + 1];
  if (target == ignore_index || target < 0 || target >= vocab) {
    if (d != nullptr) {
      for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        d[j] = 0.0f;
      }
    }
    if (threadIdx.x == 0) {
      row_loss[row] = 0.0f;
    }
    return;
  }
  float maxv = -INFINITY;
  for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
    maxv = fmaxf(maxv, x[j]);
  }
  maxv = detail::block_max(maxv, shared);
  float total = 0.0f;
  float logit_sum = 0.0f;
  for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
    total += expf(x[j] - maxv);
    logit_sum += x[j];
  }
  total = detail::block_sum(total, shared);
  logit_sum = detail::block_sum(logit_sum, shared);
  const float lse = maxv + logf(total);
  const float x_target = x[target];
  __syncthreads();  // tum ipler satiri okudu: yerinde yazmak guvenli
  const float on_target = 1.0f - smoothing;
  if (threadIdx.x == 0) {
    float loss = on_target * (lse - x_target);
    if (smoothing > 0.0f) {
      loss += smoothing * (lse - logit_sum / static_cast<float>(vocab));
    }
    row_loss[row] = loss + z_coef * lse * lse;
  }
  if (d != nullptr) {
    const float scale = *inv_valid * grad_scale;
    const float uniform = smoothing / static_cast<float>(vocab);
    const float z_grad = 2.0f * z_coef * lse;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
      const float p = expf(x[j] - lse);
      float g = p - uniform + z_grad * p;
      if (j == target) {
        g -= on_target;
      }
      d[j] = g * scale;
    }
  }
}

__global__ void ce_reduce_kernel(float* __restrict__ loss_out, const float* __restrict__ row_loss, int rows,
                                 const float* __restrict__ inv_valid, const int* __restrict__ bad_target) {
  __shared__ double shared[32];
  double acc = 0.0;
  for (int i = threadIdx.x; i < rows; i += blockDim.x) {
    acc += row_loss[i];
  }
  acc = detail::block_sum(acc, shared);
  if (threadIdx.x == 0) {
    // Sozluk disi hedef: kayip NaN (anomali/NaN denetimi bunu yakalar; sessiz 0 yok).
    *loss_out = *bad_target != 0 ? NAN : static_cast<float>(acc * static_cast<double>(*inv_valid));
  }
}

// ================================ Philox =========================================

__device__ __forceinline__ uint4 philox4x32_10(uint4 ctr, uint2 key) {
  constexpr unsigned int kMul0 = 0xD2511F53u;
  constexpr unsigned int kMul1 = 0xCD9E8D57u;
  constexpr unsigned int kWeyl0 = 0x9E3779B9u;
  constexpr unsigned int kWeyl1 = 0xBB67AE85u;
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    const unsigned int hi0 = __umulhi(kMul0, ctr.x);
    const unsigned int lo0 = kMul0 * ctr.x;
    const unsigned int hi1 = __umulhi(kMul1, ctr.z);
    const unsigned int lo1 = kMul1 * ctr.z;
    ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
    key.x += kWeyl0;
    key.y += kWeyl1;
  }
  return ctr;
}

// CPU fused::philox_uniform ile ayni: blok = offset + i/4, serit = i%4, ust 24 bit.
template <bool kBackward>
__global__ void dropout_kernel(float* __restrict__ out, const float* __restrict__ in, std::size_t n, float p,
                               float scale, std::uint64_t seed, std::uint64_t offset) {
  const std::size_t groups = (n + 3) / 4;
  NEXUS_GRID_STRIDE(g, groups) {
    const std::uint64_t block = offset + g;
    const uint4 r = philox4x32_10(make_uint4(static_cast<unsigned int>(block), static_cast<unsigned int>(block >> 32), 0u, 0u),
                                  make_uint2(static_cast<unsigned int>(seed), static_cast<unsigned int>(seed >> 32)));
    const unsigned int lanes[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const std::size_t i = g * 4 + l;
      if (i < n) {
        const float u = static_cast<float>(lanes[l] >> 8) * (1.0f / 16777216.0f);
        out[i] = u < p ? 0.0f : in[i] * scale;
      }
    }
  }
}

// ================================ Foreach AdamW ==================================

struct Chunk {
  std::uint32_t slot;
  std::uint32_t pad;
  std::uint64_t start;
  std::uint64_t length;
};

constexpr std::uint64_t kChunk = 65536;
constexpr int kAdamThreads = 256;

__global__ void grad_norm_partial_kernel(double* __restrict__ partial, const DeviceParamSlot* __restrict__ slots,
                                         const Chunk* __restrict__ chunks) {
  __shared__ double shared[32];
  const Chunk chunk = chunks[blockIdx.x];
  const float* g = slots[chunk.slot].grad + chunk.start;
  double acc = 0.0;
  for (std::uint64_t i = threadIdx.x; i < chunk.length; i += blockDim.x) {
    const double v = g[i];
    acc += v * v;
  }
  acc = detail::block_sum(acc, shared);
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = acc;
  }
}

__global__ void grad_norm_final_kernel(float* __restrict__ clip_out, float* __restrict__ norm_out,
                                       const double* __restrict__ partial, int count, double max_norm) {
  __shared__ double shared[32];
  double acc = 0.0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    acc += partial[i];
  }
  acc = detail::block_sum(acc, shared);
  if (threadIdx.x == 0) {
    const double norm = sqrt(acc);
    if (norm_out != nullptr) {
      *norm_out = static_cast<float>(norm);
    }
    *clip_out = max_norm > 0.0 ? static_cast<float>(fmin(1.0, max_norm / (norm + 1.0e-6))) : 1.0f;
  }
}

struct AdamCoeffs {
  float b1, ob1, b2, ob2, inv1, inv2, lr, lr_wd, eps;
};

// Islem sirasi CPU fused::adamw_multi_tensor ile ayni; __f*_rn FMA birlestirmesini engeller.
__global__ void adamw_update_kernel(const DeviceParamSlot* __restrict__ slots, const Chunk* __restrict__ chunks,
                                    const float* __restrict__ clip_ptr, AdamCoeffs c) {
  const Chunk chunk = chunks[blockIdx.x];
  const DeviceParamSlot slot = slots[chunk.slot];
  const float clip = *clip_ptr;
  for (std::uint64_t k = threadIdx.x; k < chunk.length; k += blockDim.x) {
    const std::uint64_t i = chunk.start + k;
    float p = slot.param[i];
    const float g = __fmul_rn(slot.grad[i], clip);
    const float lr_wd = __fmul_rn(c.lr_wd, slot.weight_decay_scale);
    if (lr_wd != 0.0f) {
      p = __fsub_rn(p, __fmul_rn(lr_wd, p));
    }
    const float mt = __fadd_rn(__fmul_rn(c.b1, slot.exp_avg[i]), __fmul_rn(c.ob1, g));
    const float vt = __fadd_rn(__fmul_rn(c.b2, slot.exp_avg_sq[i]), __fmul_rn(__fmul_rn(c.ob2, g), g));
    slot.exp_avg[i] = mt;
    slot.exp_avg_sq[i] = vt;
    const float denom = __fadd_rn(__fsqrt_rn(__fmul_rn(vt, c.inv2)), c.eps);
    p = __fsub_rn(p, __fdiv_rn(__fmul_rn(c.lr, __fmul_rn(mt, c.inv1)), denom));
    slot.param[i] = p;
  }
}

std::vector<Chunk> make_chunks(const DeviceParamSlot* slots, int num_slots) {
  std::vector<Chunk> chunks;
  for (int s = 0; s < num_slots; ++s) {
    for (std::uint64_t start = 0; start < slots[s].numel; start += kChunk) {
      chunks.push_back({static_cast<std::uint32_t>(s), 0u, start, std::min<std::uint64_t>(kChunk, slots[s].numel - start)});
    }
  }
  return chunks;
}

std::uint64_t max_chunks(int num_slots, std::uint64_t total) {
  return total / kChunk + static_cast<std::uint64_t>(num_slots) + 1;
}

}  // namespace

// ================================ Gomme API ======================================

void embedding_pos_forward(float* out, const std::int32_t* tokens, const float* wte, const float* wpe, int batch,
                           int time, int dim, cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(batch) * time * dim;
  embedding_pos_forward_kernel<<<detail::blocks_for(total, 256), 256, 0, stream>>>(out, tokens, wte, wpe, time, dim,
                                                                                   total);
  detail::check_launch("embedding_pos_forward_kernel");
}

void embedding_backward_atomic(float* dwte, float* dwpe, const float* dout, const std::int32_t* tokens, int batch,
                               int time, int dim, cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(batch) * time * dim;
  embedding_backward_atomic_kernel<<<detail::blocks_for(total, 256), 256, 0, stream>>>(dwte, dwpe, dout, tokens, time,
                                                                                       dim, total);
  detail::check_launch("embedding_backward_atomic_kernel");
}

std::size_t embedding_backward_scratch_bytes(int batch, int time) {
  const int n = batch * time;
  std::size_t temp = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp, static_cast<const std::int32_t*>(nullptr),
                                  static_cast<std::int32_t*>(nullptr), static_cast<const std::int32_t*>(nullptr),
                                  static_cast<std::int32_t*>(nullptr), n);
  return 3 * align_up(sizeof(std::int32_t) * static_cast<std::size_t>(n)) + align_up(temp);
}

void embedding_backward_deterministic(float* dwte, float* dwpe, const float* dout, const std::int32_t* tokens,
                                      int batch, int time, int dim, int vocab, int padding_idx, void* scratch,
                                      cudaStream_t stream) {
  if (batch <= 0 || time <= 0 || dim <= 0 || vocab <= 0 || scratch == nullptr) {
    throw TensorError("Nexus CUDA embedding geri: gecersiz sekil veya scratch yok");
  }
  const int n = batch * time;
  auto* base = static_cast<char*>(scratch);
  const std::size_t block = align_up(sizeof(std::int32_t) * static_cast<std::size_t>(n));
  auto* rows_in = reinterpret_cast<std::int32_t*>(base);
  auto* keys_out = reinterpret_cast<std::int32_t*>(base + block);
  auto* rows_out = reinterpret_cast<std::int32_t*>(base + 2 * block);
  void* temp = base + 3 * block;
  std::size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, tokens, keys_out, rows_in, rows_out, n);
  iota_kernel<<<detail::blocks_for(static_cast<std::size_t>(n), 256), 256, 0, stream>>>(rows_in, n);
  detail::check_launch("iota_kernel");
  // Radix siralama kararli: ayni token icin konum sirasi korunur -> sabit toplama sirasi.
  check(cub::DeviceRadixSort::SortPairs(temp, temp_bytes, tokens, keys_out, rows_in, rows_out, n, 0,
                                        static_cast<int>(sizeof(std::int32_t) * 8), stream),
        "cub::DeviceRadixSort::SortPairs");
  const dim3 grid((dim + 127) / 128, static_cast<unsigned int>(std::min(n, 4096)));
  embedding_segment_kernel<<<grid, 128, 0, stream>>>(dwte, dout, keys_out, rows_out, n, dim, padding_idx);
  detail::check_launch("embedding_segment_kernel");
  if (dwpe != nullptr) {
    const std::size_t total = static_cast<std::size_t>(time) * dim;
    wpe_backward_kernel<<<detail::blocks_for(total, 256), 256, 0, stream>>>(dwpe, dout, batch, time, dim);
    detail::check_launch("wpe_backward_kernel");
  }
}

// ================================ CE API =========================================

std::size_t cross_entropy_scratch_bytes(int batch, int time) {
  return align_up(sizeof(float) * static_cast<std::size_t>(batch) * time) + 256;
}

void cross_entropy_shifted(float* dlogits, float* loss_out, std::int32_t* valid_out, const float* logits,
                           const std::int32_t* tokens, int batch, int time, int vocab,
                           const CrossEntropyDeviceOptions& options, float grad_scale, void* scratch,
                           cudaStream_t stream) {
  if (batch < 1 || time < 2 || vocab < 2) {
    throw TensorError("Nexus CUDA CE: batch>=1, time>=2, vocab>=2 olmali");
  }
  if (!(options.label_smoothing >= 0.0f && options.label_smoothing < 1.0f) || !(options.z_loss >= 0.0f)) {
    throw TensorError("Nexus CUDA CE: label_smoothing [0,1), z_loss >= 0 olmali");
  }
  if (scratch == nullptr || loss_out == nullptr) {
    throw TensorError("Nexus CUDA CE: scratch ve loss_out gerekli");
  }
  const int rows = batch * time;
  auto* base = static_cast<char*>(scratch);
  auto* row_loss = reinterpret_cast<float*>(base);
  char* tail = base + align_up(sizeof(float) * static_cast<std::size_t>(rows));
  auto* inv_valid = reinterpret_cast<float*>(tail);
  auto* valid = reinterpret_cast<std::int32_t*>(tail + 4);
  auto* bad = reinterpret_cast<int*>(tail + 8);
  check(cudaMemsetAsync(bad, 0, sizeof(int), stream), "cudaMemsetAsync");
  ce_count_kernel<<<1, 1024, 0, stream>>>(valid, inv_valid, tokens, batch, time, vocab, options.ignore_index, bad);
  detail::check_launch("ce_count_kernel");
  ce_row_kernel<<<static_cast<unsigned int>(rows), kCeThreads, 0, stream>>>(
      dlogits, row_loss, logits, tokens, time, vocab, options.ignore_index, options.label_smoothing, options.z_loss,
      grad_scale, inv_valid);
  detail::check_launch("ce_row_kernel");
  ce_reduce_kernel<<<1, 1024, 0, stream>>>(loss_out, row_loss, rows, inv_valid, bad);
  detail::check_launch("ce_reduce_kernel");
  if (valid_out != nullptr) {
    check(cudaMemcpyAsync(valid_out, valid, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync");
  }
}

// ================================ Dropout API ====================================

void dropout_philox_forward(float* out, const float* x, std::size_t n, float p, std::uint64_t seed,
                            std::uint64_t offset, cudaStream_t stream) {
  if (!(p >= 0.0f && p < 1.0f)) {
    throw TensorError("Nexus CUDA dropout: p [0,1) araliginda olmali");
  }
  const std::size_t groups = (n + 3) / 4;
  dropout_kernel<false><<<detail::blocks_for(groups, 256), 256, 0, stream>>>(out, x, n, p, 1.0f / (1.0f - p), seed,
                                                                             offset);
  detail::check_launch("dropout_forward_kernel");
}

void dropout_philox_backward(float* dx, const float* dout, std::size_t n, float p, std::uint64_t seed,
                             std::uint64_t offset, cudaStream_t stream) {
  if (!(p >= 0.0f && p < 1.0f)) {
    throw TensorError("Nexus CUDA dropout: p [0,1) araliginda olmali");
  }
  const std::size_t groups = (n + 3) / 4;
  dropout_kernel<true><<<detail::blocks_for(groups, 256), 256, 0, stream>>>(dx, dout, n, p, 1.0f / (1.0f - p), seed,
                                                                            offset);
  detail::check_launch("dropout_backward_kernel");
}

// ================================ AdamW API ======================================

std::size_t adamw_scratch_bytes(int num_slots, std::uint64_t total_numel) {
  const std::uint64_t chunks = max_chunks(num_slots, total_numel);
  return align_up(sizeof(DeviceParamSlot) * static_cast<std::size_t>(num_slots)) +
         align_up(sizeof(Chunk) * static_cast<std::size_t>(chunks)) + align_up(sizeof(double) * static_cast<std::size_t>(chunks)) +
         256;
}

void adamw_multi_tensor(const DeviceParamSlot* slots, int num_slots, const DeviceAdamWConfig& config,
                        std::int64_t step, float* grad_norm_out, void* scratch, cudaStream_t stream) {
  if (step < 1) {
    throw TensorError("Nexus CUDA AdamW: step 1'den baslar");
  }
  if (num_slots <= 0 || slots == nullptr || scratch == nullptr) {
    throw TensorError("Nexus CUDA AdamW: bos slot listesi veya scratch yok");
  }
  std::uint64_t total = 0;
  for (int s = 0; s < num_slots; ++s) {
    total += slots[s].numel;
  }
  const std::vector<Chunk> chunks = make_chunks(slots, num_slots);
  auto* base = static_cast<char*>(scratch);
  auto* d_slots = reinterpret_cast<DeviceParamSlot*>(base);
  base += align_up(sizeof(DeviceParamSlot) * static_cast<std::size_t>(num_slots));
  auto* d_chunks = reinterpret_cast<Chunk*>(base);
  base += align_up(sizeof(Chunk) * static_cast<std::size_t>(max_chunks(num_slots, total)));
  auto* partial = reinterpret_cast<double*>(base);
  base += align_up(sizeof(double) * static_cast<std::size_t>(max_chunks(num_slots, total)));
  auto* clip = reinterpret_cast<float*>(base);
  // Sayfalanabilir kaynaktan asenkron kopya, kaynak hazirlik tamponuna alindiginda doner:
  // host dizileri bu cagridan sonra serbest kalabilir.
  check(cudaMemcpyAsync(d_slots, slots, sizeof(DeviceParamSlot) * static_cast<std::size_t>(num_slots),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(slots)");
  check(cudaMemcpyAsync(d_chunks, chunks.data(), sizeof(Chunk) * chunks.size(), cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(chunks)");
  const auto count = static_cast<unsigned int>(chunks.size());
  if (config.max_grad_norm > 0.0 || grad_norm_out != nullptr) {
    grad_norm_partial_kernel<<<count, kAdamThreads, 0, stream>>>(partial, d_slots, d_chunks);
    detail::check_launch("grad_norm_partial_kernel");
    grad_norm_final_kernel<<<1, 1024, 0, stream>>>(clip, grad_norm_out, partial, static_cast<int>(count),
                                                   config.max_grad_norm);
    detail::check_launch("grad_norm_final_kernel");
  } else {
    fill(clip, 1.0f, 1, stream);
  }
  const double bc1 = 1.0 - std::pow(config.beta1, static_cast<double>(step));
  const double bc2 = 1.0 - std::pow(config.beta2, static_cast<double>(step));
  AdamCoeffs c;
  c.b1 = static_cast<float>(config.beta1);
  c.ob1 = static_cast<float>(1.0 - config.beta1);
  c.b2 = static_cast<float>(config.beta2);
  c.ob2 = static_cast<float>(1.0 - config.beta2);
  c.inv1 = static_cast<float>(1.0 / bc1);
  c.inv2 = static_cast<float>(1.0 / bc2);
  c.lr = static_cast<float>(config.lr);
  c.lr_wd = c.lr * static_cast<float>(config.weight_decay);
  c.eps = static_cast<float>(config.eps);
  adamw_update_kernel<<<count, kAdamThreads, 0, stream>>>(d_slots, d_chunks, clip, c);
  detail::check_launch("adamw_update_kernel");
}

}  // namespace cuda
}  // namespace nexus
