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

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace nexus {
namespace cuda {
namespace {

#define NEXUS_GRID_STRIDE(i, n)                                                                         \
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < (n); \
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)

std::size_t align_up(std::size_t bytes) { return (bytes + 255u) & ~static_cast<std::size_t>(255u); }

// ================================ Gomme ==========================================

template <typename W>
__global__ void embedding_pos_forward_kernel(float* __restrict__ out, const std::int32_t* __restrict__ tokens,
                                             const W* __restrict__ wte, const W* __restrict__ wpe, int time, int dim,
                                             std::size_t total) {
  using N = detail::Num<W>;
  NEXUS_GRID_STRIDE(i, total) {
    const std::size_t row = i / dim;
    const int c = static_cast<int>(i % dim);
    const int t = static_cast<int>(row % time);
    out[i] = N::load(wte[static_cast<std::size_t>(tokens[row]) * dim + c]) +
             N::load(wpe[static_cast<std::size_t>(t) * dim + c]);
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

// row_offset: parcali CE (FAZ 7.5). logits/dlogits parcaya gore yerel, row_loss/tokens kuresel.
__global__ void ce_row_kernel(float* __restrict__ dlogits, float* __restrict__ row_loss, const float* logits,
                              const std::int32_t* __restrict__ tokens, int time, int vocab,
                              std::int32_t ignore_index, float smoothing, float z_coef, float grad_scale,
                              const float* __restrict__ inv_valid, const float* __restrict__ device_scale,
                              std::size_t row_offset) {
  __shared__ float shared[32];
  const std::size_t local = blockIdx.x;
  const std::size_t row = row_offset + local;
  const int t = static_cast<int>(row % time);
  const float* x = logits + local * vocab;
  float* d = dlogits != nullptr ? dlogits + local * vocab : nullptr;
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
    const float scale = *inv_valid * grad_scale * (device_scale != nullptr ? *device_scale : 1.0f);
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

// Olcekli gradyanlarin kare toplamindan: unscale edilmis norm, kirpma, inf/NaN bayragi.
// Cikti katsayisi = kirpma * 1/olcek (guncelleme cekirdegi gradyani bununla carpar).
__global__ void grad_norm_final_kernel(float* __restrict__ clip_out, float* __restrict__ norm_out,
                                       const double* __restrict__ partial, int count, double max_norm,
                                       const float* __restrict__ loss_scale, int* __restrict__ found_inf) {
  __shared__ double shared[32];
  double acc = 0.0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    acc += partial[i];
  }
  acc = detail::block_sum(acc, shared);
  if (threadIdx.x == 0) {
    const double inv_scale = loss_scale != nullptr ? 1.0 / static_cast<double>(*loss_scale) : 1.0;
    const bool bad = !isfinite(acc);
    const double norm = sqrt(acc) * inv_scale;
    if (norm_out != nullptr) {
      *norm_out = static_cast<float>(norm);
    }
    if (found_inf != nullptr) {
      *found_inf = bad ? 1 : 0;
    }
    const float clip = max_norm > 0.0 && !bad ? static_cast<float>(fmin(1.0, max_norm / (norm + 1.0e-6))) : 1.0f;
    *clip_out = loss_scale != nullptr ? clip * static_cast<float>(inv_scale) : clip;
  }
}

// f32 -> bf16 stokastik yuvarlama: alt 16 bite duzgun gurultu eklenip kesilir (beklenen deger
// korunur). NaN/Inf degismeden RN ile gecer.
__device__ __forceinline__ __nv_bfloat16 bf16_stochastic(float x, unsigned int noise) {
  if (!isfinite(x)) {
    return __float2bfloat16_rn(x);
  }
  const unsigned int bits = __float_as_uint(x) + (noise & 0xFFFFu);
  return __ushort_as_bfloat16(static_cast<unsigned short>(bits >> 16));
}

__device__ __forceinline__ unsigned int hash32(unsigned long long v) {
  v ^= v >> 33;
  v *= 0xff51afd7ed558ccdULL;
  v ^= v >> 33;
  v *= 0xc4ceb9fe1a85ec53ULL;
  v ^= v >> 33;
  return static_cast<unsigned int>(v);
}

struct AdamCoeffs {
  float b1, ob1, b2, ob2, inv1, inv2, lr, lr_wd, eps;
  double beta1, beta2;  // olcekli yolda kabul edilen adim icin aygitta bias correction
};

// Islem sirasi CPU fused::adamw_multi_tensor ile ayni; __f*_rn FMA birlestirmesini engeller.
template <typename S>
__global__ void adamw_update_kernel(const DeviceParamSlot* __restrict__ slots, const Chunk* __restrict__ chunks,
                                    const float* __restrict__ clip_ptr, AdamCoeffs c, const int* __restrict__ found_inf,
                                    const long long* __restrict__ skipped_steps, long long attempt_step,
                                     bool stochastic, unsigned long long sr_seed) {
  if (found_inf != nullptr && *found_inf != 0) {
    return;  // inf/NaN: adim atlanir (parametre ve momentler degismez)
  }
  float inv1 = c.inv1;
  float inv2 = c.inv2;
  if (skipped_steps != nullptr) {
    // Denemelerden onceki reddedilenleri cikar: bias correction yalniz kabul edilen
    // Adam guncellemelerini sayar. Host'a senkron okuma gerekmez.
    const long long accepted_step = attempt_step - *skipped_steps;
    inv1 = static_cast<float>(1.0 / (1.0 - pow(c.beta1, static_cast<double>(accepted_step))));
    inv2 = static_cast<float>(1.0 / (1.0 - pow(c.beta2, static_cast<double>(accepted_step))));
  }
  const Chunk chunk = chunks[blockIdx.x];
  const DeviceParamSlot slot = slots[chunk.slot];
  const float clip = *clip_ptr;
  S* shadow = static_cast<S*>(slot.shadow);
  for (std::uint64_t k = threadIdx.x; k < chunk.length; k += blockDim.x) {
    const std::uint64_t i = chunk.start + k;
    float p = slot.param != nullptr ? slot.param[i] : detail::Num<S>::load(shadow[i]);
    const float g = __fmul_rn(slot.grad[i], clip);
    const float lr_wd = __fmul_rn(c.lr_wd, slot.weight_decay_scale);
    if (lr_wd != 0.0f) {
      p = __fsub_rn(p, __fmul_rn(lr_wd, p));
    }
    const float mt = __fadd_rn(__fmul_rn(c.b1, slot.exp_avg[i]), __fmul_rn(c.ob1, g));
    const float vt = __fadd_rn(__fmul_rn(c.b2, slot.exp_avg_sq[i]), __fmul_rn(__fmul_rn(c.ob2, g), g));
    slot.exp_avg[i] = mt;
    slot.exp_avg_sq[i] = vt;
    const float denom = __fadd_rn(__fsqrt_rn(__fmul_rn(vt, inv2)), c.eps);
    p = __fsub_rn(p, __fdiv_rn(__fmul_rn(c.lr, __fmul_rn(mt, inv1)), denom));
    if (slot.param != nullptr) {
      slot.param[i] = p;  // FP32 ana kopya
    }
    if (shadow != nullptr) {
      if constexpr (std::is_same_v<S, __nv_bfloat16>) {
        // Ana kopyasiz mod (param == nullptr): stokastik yuvarlama; aksi halde RN dar kopya.
        shadow[i] = (stochastic && slot.param == nullptr)
                        ? bf16_stochastic(p, hash32(sr_seed ^ (static_cast<unsigned long long>(chunk.slot) << 40) ^ i))
                        : __float2bfloat16_rn(p);
      } else {
        shadow[i] = detail::Num<S>::store(p);
      }
    }
  }
}

__global__ void loss_scale_update_kernel(DeviceLossScaleState* __restrict__ state, double growth_factor,
                                         double backoff_factor, int growth_interval, double min_scale,
                                         double max_scale) {
  // nexus_train::LossScaleState::update ile ayni kurallar (tek politika).
  DeviceLossScaleState st = *state;
  st.steps += 1;
  if (st.found_inf != 0) {
    const double s = static_cast<double>(st.scale) * backoff_factor;
    st.scale = static_cast<float>(s < min_scale ? min_scale : s);
    st.growth_tracker = 0;
    st.skipped_steps += 1;
  } else {
    st.growth_tracker += 1;
    if (st.growth_tracker >= growth_interval) {
      const double g = static_cast<double>(st.scale) * growth_factor;
      st.scale = static_cast<float>(g > max_scale ? max_scale : g);
      st.growth_tracker = 0;
    }
  }
  st.scale_history[(st.steps - 1) % kLossScaleHistorySize] = st.scale;
  *state = st;
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
  embedding_pos_forward_kernel<float><<<detail::blocks_for(total, 256), 256, 0, stream>>>(out, tokens, wte, wpe, time,
                                                                                          dim, total);
  detail::check_launch("embedding_pos_forward_kernel");
}

void embedding_pos_forward(float* out, const std::int32_t* tokens, const void* wte, const void* wpe, DType weight_type,
                           int batch, int time, int dim, cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(batch) * time * dim;
  detail::dispatch_storage(weight_type, "embedding_pos_forward", [&](auto tag) {
    using W = decltype(tag);
    embedding_pos_forward_kernel<W><<<detail::blocks_for(total, 256), 256, 0, stream>>>(
        out, tokens, static_cast<const W*>(wte), static_cast<const W*>(wpe), time, dim, total);
  });
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
  cross_entropy_shifted(dlogits, loss_out, valid_out, logits, tokens, batch, time, vocab, options, grad_scale, nullptr,
                        scratch, stream);
}

void cross_entropy_shifted(float* dlogits, float* loss_out, std::int32_t* valid_out, const float* logits,
                           const std::int32_t* tokens, int batch, int time, int vocab,
                           const CrossEntropyDeviceOptions& options, float grad_scale, const float* device_grad_scale,
                           void* scratch, cudaStream_t stream) {
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
      grad_scale, inv_valid, device_grad_scale, 0);
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
  // Norm gecisi ayni zamanda unscale + isfinite denetimidir (tek cok-tensorlu indirgeme, tek bayrak).
  if (config.max_grad_norm > 0.0 || grad_norm_out != nullptr || config.loss_scale != nullptr ||
      config.found_inf != nullptr) {
    grad_norm_partial_kernel<<<count, kAdamThreads, 0, stream>>>(partial, d_slots, d_chunks);
    detail::check_launch("grad_norm_partial_kernel");
    grad_norm_final_kernel<<<1, 1024, 0, stream>>>(clip, grad_norm_out, partial, static_cast<int>(count),
                                                   config.max_grad_norm, config.loss_scale, config.found_inf);
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
  c.beta1 = config.beta1;
  c.beta2 = config.beta2;
  if (config.stochastic_rounding && config.shadow_type != DType::BF16) {
    throw TensorError("Nexus CUDA AdamW: stokastik yuvarlama yalniz BF16 golge kopyada desteklenir");
  }
  detail::dispatch_storage(config.shadow_type, "adamw_multi_tensor", [&](auto tag) {
    using S = decltype(tag);
    adamw_update_kernel<S><<<count, kAdamThreads, 0, stream>>>(d_slots, d_chunks, clip, c, config.found_inf,
                                                               config.skipped_steps, step, config.stochastic_rounding,
                                                               static_cast<unsigned long long>(config.sr_seed) ^
                                                                   static_cast<unsigned long long>(step) * 0x9E3779B97F4A7C15ULL);
  });
  detail::check_launch("adamw_update_kernel");
}

namespace {
struct CeScratch {
  float* row_loss;
  float* inv_valid;
  std::int32_t* valid;
  int* bad;
};
CeScratch ce_scratch(void* scratch, int rows) {
  auto* base = static_cast<char*>(scratch);
  char* tail = base + align_up(sizeof(float) * static_cast<std::size_t>(rows));
  return {reinterpret_cast<float*>(base), reinterpret_cast<float*>(tail), reinterpret_cast<std::int32_t*>(tail + 4),
          reinterpret_cast<int*>(tail + 8)};
}
}  // namespace

void cross_entropy_chunked_begin(const std::int32_t* tokens, int batch, int time, int vocab,
                                 const CrossEntropyDeviceOptions& options, void* scratch, cudaStream_t stream) {
  if (batch < 1 || time < 2 || vocab < 2 || scratch == nullptr) {
    throw TensorError("Nexus CUDA CE (parcali): batch>=1, time>=2, vocab>=2, scratch gerekli");
  }
  const CeScratch c = ce_scratch(scratch, batch * time);
  check(cudaMemsetAsync(c.bad, 0, sizeof(int), stream), "cudaMemsetAsync");
  ce_count_kernel<<<1, 1024, 0, stream>>>(c.valid, c.inv_valid, tokens, batch, time, vocab, options.ignore_index, c.bad);
  detail::check_launch("ce_count_kernel");
}

void cross_entropy_chunk(float* dlogits, const float* logits, const std::int32_t* tokens, int batch, int time, int vocab,
                         const CrossEntropyDeviceOptions& options, float grad_scale, const float* device_grad_scale,
                         int row_begin, int rows, void* scratch, cudaStream_t stream) {
  if (row_begin < 0 || rows < 1 || row_begin + rows > batch * time) {
    throw TensorError("Nexus CUDA CE (parcali): satir araligi gecersiz");
  }
  const CeScratch c = ce_scratch(scratch, batch * time);
  ce_row_kernel<<<static_cast<unsigned int>(rows), kCeThreads, 0, stream>>>(
      dlogits, c.row_loss, logits, tokens, time, vocab, options.ignore_index, options.label_smoothing, options.z_loss,
      grad_scale, c.inv_valid, device_grad_scale, static_cast<std::size_t>(row_begin));
  detail::check_launch("ce_row_kernel");
}

void cross_entropy_chunked_end(float* loss_out, std::int32_t* valid_out, int batch, int time, void* scratch,
                               cudaStream_t stream) {
  const CeScratch c = ce_scratch(scratch, batch * time);
  ce_reduce_kernel<<<1, 1024, 0, stream>>>(loss_out, c.row_loss, batch * time, c.inv_valid, c.bad);
  detail::check_launch("ce_reduce_kernel");
  if (valid_out != nullptr) {
    check(cudaMemcpyAsync(valid_out, c.valid, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync");
  }
}

// ================================ 8-bit AdamW (FAZ 7.3) ===========================
// bitsandbytes blok bazli dinamik nicemleme: blok = 2048 eleman, blok basina absmax; m
// isaretli, v isaretsiz dinamik harita (create_dynamic_map, 256 deger). Guncelleme tam
// hassasiyetli yeni m, v ile yapilir; sonra yeni absmax ile yeniden nicemlenir.
namespace {

__constant__ float c_qmap_signed[256];
__constant__ float c_qmap_unsigned[256];
constexpr int kQ8Threads = 256;
constexpr int kQ8PerThread = kQ8Block / kQ8Threads;

struct Q8Item {
  std::uint32_t slot;
  std::uint32_t block;
};

// bitsandbytes.functional.create_dynamic_map(signed, max_exponent_bits=7, total_bits=8).
std::vector<float> dynamic_map(bool is_signed) {
  std::vector<float> data;
  const int max_exp = 7;
  const int non_sign = 7;
  for (int i = 0; i < max_exp; ++i) {
    const int items = is_signed ? (1 << (i + non_sign - max_exp)) + 1 : (1 << (i + non_sign - max_exp + 1)) + 1;
    const double scale = std::pow(10.0, -(max_exp - 1) + i);
    for (int k = 0; k + 1 < items; ++k) {
      // torch.linspace(0.1, 1, items): ardisik sinirlarin ortasi
      const double a = 0.1 + 0.9 * k / (items - 1);
      const double b = 0.1 + 0.9 * (k + 1) / (items - 1);
      data.push_back(static_cast<float>(scale * (a + b) / 2.0));
      if (is_signed) {
        data.push_back(static_cast<float>(-scale * (a + b) / 2.0));
      }
    }
  }
  data.push_back(0.0f);
  data.push_back(1.0f);
  while (data.size() < 256) {
    data.push_back(0.0f);
  }
  std::sort(data.begin(), data.end());
  return data;
}

void upload_qmaps() {
  static bool done = false;
  if (done) {
    return;
  }
  const std::vector<float> s = dynamic_map(true);
  const std::vector<float> u = dynamic_map(false);
  check(cudaMemcpyToSymbol(c_qmap_signed, s.data(), 256 * sizeof(float)), "cudaMemcpyToSymbol(qmap)");
  check(cudaMemcpyToSymbol(c_qmap_unsigned, u.data(), 256 * sizeof(float)), "cudaMemcpyToSymbol(qmap)");
  done = true;
}

__device__ __forceinline__ std::uint8_t quantize_nearest(float x, const float* code) {
  int lo = 0;
  int hi = 255;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (code[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return static_cast<std::uint8_t>((x - code[lo] <= code[hi] - x) ? lo : hi);
}

template <typename S>
__global__ void adamw8bit_kernel(const Device8bitSlot* __restrict__ slots, const Q8Item* __restrict__ items,
                                 const float* __restrict__ clip_ptr, AdamCoeffs c, const int* __restrict__ found_inf,
                                 bool stochastic, unsigned long long sr_seed) {
  if (found_inf != nullptr && *found_inf != 0) {
    return;
  }
  __shared__ float red_m[kQ8Threads / 32];
  __shared__ float red_v[kQ8Threads / 32];
  const Q8Item item = items[blockIdx.x];
  const Device8bitSlot slot = slots[item.slot];
  const std::uint64_t base = static_cast<std::uint64_t>(item.block) * kQ8Block;
  const float am = slot.absmax_m[item.block];
  const float av = slot.absmax_v[item.block];
  const float clip = *clip_ptr;
  S* shadow = static_cast<S*>(slot.shadow);
  float mt[kQ8PerThread];
  float vt[kQ8PerThread];
  float local_m = 0.0f;
  float local_v = 0.0f;
#pragma unroll
  for (int k = 0; k < kQ8PerThread; ++k) {
    const std::uint64_t i = base + threadIdx.x + static_cast<std::uint64_t>(k) * kQ8Threads;
    mt[k] = 0.0f;
    vt[k] = 0.0f;
    if (i >= slot.numel) {
      continue;
    }
    float p = slot.param != nullptr ? slot.param[i] : detail::Num<S>::load(shadow[i]);
    const float g = __fmul_rn(slot.grad[i], clip);
    const float lr_wd = __fmul_rn(c.lr_wd, slot.weight_decay_scale);
    if (lr_wd != 0.0f) {
      p = __fsub_rn(p, __fmul_rn(lr_wd, p));
    }
    const float m0 = c_qmap_signed[slot.qm[i]] * am;
    const float v0 = c_qmap_unsigned[slot.qv[i]] * av;
    mt[k] = __fadd_rn(__fmul_rn(c.b1, m0), __fmul_rn(c.ob1, g));
    vt[k] = __fadd_rn(__fmul_rn(c.b2, v0), __fmul_rn(__fmul_rn(c.ob2, g), g));
    const float denom = __fadd_rn(__fsqrt_rn(__fmul_rn(vt[k], c.inv2)), c.eps);
    p = __fsub_rn(p, __fdiv_rn(__fmul_rn(c.lr, __fmul_rn(mt[k], c.inv1)), denom));
    if (slot.param != nullptr) {
      slot.param[i] = p;
    }
    if (shadow != nullptr) {
      if constexpr (std::is_same_v<S, __nv_bfloat16>) {
        shadow[i] = (stochastic && slot.param == nullptr)
                        ? bf16_stochastic(p, hash32(sr_seed ^ (static_cast<unsigned long long>(item.slot) << 40) ^ i))
                        : __float2bfloat16_rn(p);
      } else {
        shadow[i] = detail::Num<S>::store(p);
      }
    }
    local_m = fmaxf(local_m, fabsf(mt[k]));
    local_v = fmaxf(local_v, vt[k]);
  }
  // Blok absmax (warp + paylasimli bellek).
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    local_m = fmaxf(local_m, __shfl_xor_sync(0xFFFFFFFFu, local_m, off));
    local_v = fmaxf(local_v, __shfl_xor_sync(0xFFFFFFFFu, local_v, off));
  }
  if (threadIdx.x % 32 == 0) {
    red_m[threadIdx.x / 32] = local_m;
    red_v[threadIdx.x / 32] = local_v;
  }
  __syncthreads();
  float nm = 0.0f;
  float nv = 0.0f;
#pragma unroll
  for (int w = 0; w < kQ8Threads / 32; ++w) {
    nm = fmaxf(nm, red_m[w]);
    nv = fmaxf(nv, red_v[w]);
  }
  if (threadIdx.x == 0) {
    slot.absmax_m[item.block] = nm;
    slot.absmax_v[item.block] = nv;
  }
  const float inv_m = nm > 0.0f ? 1.0f / nm : 0.0f;
  const float inv_v = nv > 0.0f ? 1.0f / nv : 0.0f;
#pragma unroll
  for (int k = 0; k < kQ8PerThread; ++k) {
    const std::uint64_t i = base + threadIdx.x + static_cast<std::uint64_t>(k) * kQ8Threads;
    if (i < slot.numel) {
      slot.qm[i] = quantize_nearest(mt[k] * inv_m, c_qmap_signed);
      slot.qv[i] = quantize_nearest(vt[k] * inv_v, c_qmap_unsigned);
    }
  }
}

std::vector<Q8Item> q8_items(const Device8bitSlot* slots, int n) {
  std::vector<Q8Item> out;
  for (int s = 0; s < n; ++s) {
    const std::uint64_t blocks = (slots[s].numel + kQ8Block - 1) / kQ8Block;
    for (std::uint64_t b = 0; b < blocks; ++b) {
      out.push_back({static_cast<std::uint32_t>(s), static_cast<std::uint32_t>(b)});
    }
  }
  return out;
}

}  // namespace

namespace {
__global__ void grad_sq_accumulate_kernel(double* __restrict__ accum, const DeviceParamSlot* __restrict__ slots,
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
    atomicAdd(accum, acc);
  }
}

__global__ void delayed_clip_kernel(float* __restrict__ clip, float* __restrict__ norm_out, double* __restrict__ accum,
                                    double max_norm) {
  const double norm = sqrt(*accum);
  if (norm_out != nullptr) {
    *norm_out = static_cast<float>(norm);
  }
  *clip = max_norm > 0.0 && isfinite(norm) ? static_cast<float>(fmin(1.0, max_norm / (norm + 1.0e-6))) : 1.0f;
  *accum = 0.0;
}
}  // namespace

void adamw_fixed_clip_update(const DeviceParamSlot* slots32, int n32, const Device8bitSlot* slots8, int n8,
                             const DeviceAdamWConfig& config, std::int64_t step, const float* clip, double* norm_accum,
                             void* scratch, cudaStream_t stream) {
  if (step < 1 || scratch == nullptr || clip == nullptr || norm_accum == nullptr || n32 + n8 == 0) {
    throw TensorError("Nexus CUDA AdamW (sabit kirpma): step>=1, scratch, clip, norm_accum gerekli");
  }
  upload_qmaps();
  std::vector<DeviceParamSlot> all(slots32, slots32 + n32);
  std::uint64_t total = 0;
  for (int i = 0; i < n32; ++i) {
    total += slots32[i].numel;
  }
  for (int i = 0; i < n8; ++i) {
    DeviceParamSlot d;
    d.grad = slots8[i].grad;
    d.numel = slots8[i].numel;
    all.push_back(d);
    total += slots8[i].numel;
  }
  const int n = n32 + n8;
  const std::vector<Chunk> chunks_all = make_chunks(all.data(), n);
  const std::vector<Chunk> chunks32 = n32 > 0 ? make_chunks(slots32, n32) : std::vector<Chunk>{};
  const std::vector<Q8Item> items = q8_items(slots8, n8);
  const std::uint64_t mc = max_chunks(n, total);
  auto* base = static_cast<char*>(scratch);
  auto* d_all = reinterpret_cast<DeviceParamSlot*>(base);
  base += align_up(sizeof(DeviceParamSlot) * static_cast<std::size_t>(n));
  auto* d_chunks_all = reinterpret_cast<Chunk*>(base);
  base += align_up(sizeof(Chunk) * mc);
  auto* d_chunks32 = reinterpret_cast<Chunk*>(base);
  base += align_up(sizeof(Chunk) * mc);
  base += align_up(sizeof(double) * mc);
  auto* d_slots8 = reinterpret_cast<Device8bitSlot*>(base);
  base += align_up(sizeof(Device8bitSlot) * static_cast<std::size_t>(n8 + 1));
  auto* d_items = reinterpret_cast<Q8Item*>(base);
  check(cudaMemcpyAsync(d_all, all.data(), sizeof(DeviceParamSlot) * all.size(), cudaMemcpyHostToDevice, stream), "memcpy");
  check(cudaMemcpyAsync(d_chunks_all, chunks_all.data(), sizeof(Chunk) * chunks_all.size(), cudaMemcpyHostToDevice, stream),
        "memcpy");
  if (!chunks32.empty()) {
    check(cudaMemcpyAsync(d_chunks32, chunks32.data(), sizeof(Chunk) * chunks32.size(), cudaMemcpyHostToDevice, stream),
          "memcpy");
  }
  if (n8 > 0) {
    check(cudaMemcpyAsync(d_slots8, slots8, sizeof(Device8bitSlot) * static_cast<std::size_t>(n8), cudaMemcpyHostToDevice,
                          stream),
          "memcpy");
    check(cudaMemcpyAsync(d_items, items.data(), sizeof(Q8Item) * items.size(), cudaMemcpyHostToDevice, stream), "memcpy");
  }
  // Bu gradyanlarin kare toplami bir sonraki adimin kirpmasi icin biriktirilir (guncellemeden once okunur).
  grad_sq_accumulate_kernel<<<static_cast<unsigned int>(chunks_all.size()), kAdamThreads, 0, stream>>>(norm_accum, d_all,
                                                                                                        d_chunks_all);
  detail::check_launch("grad_sq_accumulate_kernel");
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
  c.beta1 = config.beta1;
  c.beta2 = config.beta2;
  const unsigned long long seed =
      static_cast<unsigned long long>(config.sr_seed) ^ static_cast<unsigned long long>(step) * 0x9E3779B97F4A7C15ULL;
  detail::dispatch_storage(config.shadow_type, "adamw_fixed_clip_update", [&](auto tag) {
    using S = decltype(tag);
    if (!chunks32.empty()) {
      adamw_update_kernel<S><<<static_cast<unsigned int>(chunks32.size()), kAdamThreads, 0, stream>>>(
          d_all, d_chunks32, clip, c, nullptr, nullptr, step, config.stochastic_rounding, seed);
      detail::check_launch("adamw_update_kernel");
    }
    if (n8 > 0) {
      adamw8bit_kernel<S><<<static_cast<unsigned int>(items.size()), kQ8Threads, 0, stream>>>(
          d_slots8, d_items, clip, c, nullptr, config.stochastic_rounding, seed);
      detail::check_launch("adamw8bit_kernel");
    }
  });
}

void delayed_clip_finish(float* clip, float* norm_out, double* norm_accum, double max_norm, cudaStream_t stream) {
  delayed_clip_kernel<<<1, 1, 0, stream>>>(clip, norm_out, norm_accum, max_norm);
  detail::check_launch("delayed_clip_kernel");
}

std::size_t adamw8bit_scratch_bytes(int n32, int n8, std::uint64_t total32, std::uint64_t total8) {
  const int n = n32 + n8;
  const std::uint64_t chunks = max_chunks(n, total32 + total8);
  const std::uint64_t blocks = total8 / kQ8Block + static_cast<std::uint64_t>(n8) + 1;
  return align_up(sizeof(DeviceParamSlot) * static_cast<std::size_t>(n)) + align_up(sizeof(Chunk) * chunks) * 2 +
         align_up(sizeof(double) * chunks) + align_up(sizeof(Device8bitSlot) * static_cast<std::size_t>(n8 + 1)) +
         align_up(sizeof(Q8Item) * blocks) + 512;
}

void adamw_mixed_multi_tensor(const DeviceParamSlot* slots32, int n32, const Device8bitSlot* slots8, int n8,
                              const DeviceAdamWConfig& config, std::int64_t step, float* grad_norm_out, void* scratch,
                              cudaStream_t stream) {
  if (step < 1 || scratch == nullptr || n32 < 0 || n8 < 0 || n32 + n8 == 0) {
    throw TensorError("Nexus CUDA AdamW8: step>=1, scratch ve en az bir slot gerekli");
  }
  if (config.loss_scale != nullptr && config.skipped_steps != nullptr) {
    throw TensorError("Nexus CUDA AdamW8: dinamik kayip olcegi desteklenmez (BF16 ile kullanin)");
  }
  upload_qmaps();
  // Norm: tum gradyanlar tek coklu-tensör indirgemesinde.
  std::vector<DeviceParamSlot> all(slots32, slots32 + n32);
  std::uint64_t total32 = 0;
  std::uint64_t total8 = 0;
  for (int i = 0; i < n32; ++i) {
    total32 += slots32[i].numel;
  }
  for (int i = 0; i < n8; ++i) {
    DeviceParamSlot d;
    d.grad = slots8[i].grad;
    d.numel = slots8[i].numel;
    all.push_back(d);
    total8 += slots8[i].numel;
  }
  const int n = n32 + n8;
  const std::vector<Chunk> chunks_all = make_chunks(all.data(), n);
  const std::vector<Chunk> chunks32 = n32 > 0 ? make_chunks(slots32, n32) : std::vector<Chunk>{};
  const std::vector<Q8Item> items = q8_items(slots8, n8);
  const std::uint64_t mc = max_chunks(n, total32 + total8);
  auto* base = static_cast<char*>(scratch);
  auto* d_all = reinterpret_cast<DeviceParamSlot*>(base);
  base += align_up(sizeof(DeviceParamSlot) * static_cast<std::size_t>(n));
  auto* d_chunks_all = reinterpret_cast<Chunk*>(base);
  base += align_up(sizeof(Chunk) * mc);
  auto* d_chunks32 = reinterpret_cast<Chunk*>(base);
  base += align_up(sizeof(Chunk) * mc);
  auto* partial = reinterpret_cast<double*>(base);
  base += align_up(sizeof(double) * mc);
  auto* d_slots8 = reinterpret_cast<Device8bitSlot*>(base);
  base += align_up(sizeof(Device8bitSlot) * static_cast<std::size_t>(n8 + 1));
  auto* d_items = reinterpret_cast<Q8Item*>(base);
  base += align_up(sizeof(Q8Item) * (total8 / kQ8Block + static_cast<std::uint64_t>(n8) + 1));
  auto* clip = reinterpret_cast<float*>(base);
  check(cudaMemcpyAsync(d_all, all.data(), sizeof(DeviceParamSlot) * all.size(), cudaMemcpyHostToDevice, stream), "memcpy");
  check(cudaMemcpyAsync(d_chunks_all, chunks_all.data(), sizeof(Chunk) * chunks_all.size(), cudaMemcpyHostToDevice, stream),
        "memcpy");
  if (!chunks32.empty()) {
    check(cudaMemcpyAsync(d_chunks32, chunks32.data(), sizeof(Chunk) * chunks32.size(), cudaMemcpyHostToDevice, stream),
          "memcpy");
  }
  if (n8 > 0) {
    check(cudaMemcpyAsync(d_slots8, slots8, sizeof(Device8bitSlot) * static_cast<std::size_t>(n8), cudaMemcpyHostToDevice,
                          stream),
          "memcpy");
    check(cudaMemcpyAsync(d_items, items.data(), sizeof(Q8Item) * items.size(), cudaMemcpyHostToDevice, stream), "memcpy");
  }
  grad_norm_partial_kernel<<<static_cast<unsigned int>(chunks_all.size()), kAdamThreads, 0, stream>>>(partial, d_all,
                                                                                                        d_chunks_all);
  detail::check_launch("grad_norm_partial_kernel");
  grad_norm_final_kernel<<<1, 1024, 0, stream>>>(clip, grad_norm_out, partial, static_cast<int>(chunks_all.size()),
                                                 config.max_grad_norm, config.loss_scale, config.found_inf);
  detail::check_launch("grad_norm_final_kernel");
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
  c.beta1 = config.beta1;
  c.beta2 = config.beta2;
  const unsigned long long seed =
      static_cast<unsigned long long>(config.sr_seed) ^ static_cast<unsigned long long>(step) * 0x9E3779B97F4A7C15ULL;
  detail::dispatch_storage(config.shadow_type, "adamw_mixed_multi_tensor", [&](auto tag) {
    using S = decltype(tag);
    if (!chunks32.empty()) {
      adamw_update_kernel<S><<<static_cast<unsigned int>(chunks32.size()), kAdamThreads, 0, stream>>>(
          d_all, d_chunks32, clip, c, config.found_inf, nullptr, step, config.stochastic_rounding, seed);
      detail::check_launch("adamw_update_kernel");
    }
    if (n8 > 0) {
      adamw8bit_kernel<S><<<static_cast<unsigned int>(items.size()), kQ8Threads, 0, stream>>>(
          d_slots8, d_items, clip, c, config.found_inf, config.stochastic_rounding, seed);
      detail::check_launch("adamw8bit_kernel");
    }
  });
}

void loss_scale_update(DeviceLossScaleState* state, const nexus_train::LossScalePolicy& policy, cudaStream_t stream) {
  policy.validate();
  loss_scale_update_kernel<<<1, 1, 0, stream>>>(state, policy.growth_factor, policy.backoff_factor,
                                                policy.growth_interval, policy.min_scale, policy.max_scale);
  detail::check_launch("loss_scale_update_kernel");
}

void loss_scale_init(DeviceLossScaleState* state, const nexus_train::LossScalePolicy& policy, cudaStream_t stream) {
  policy.validate();
  DeviceLossScaleState host{};
  host.scale = static_cast<float>(policy.init_scale);
  // Pinned olmayan kaynak: cudaMemcpyAsync kaynagi kopyalayip doner (host degiskeni guvenle biter).
  check(cudaMemcpyAsync(state, &host, sizeof(host), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(loss_scale)");
}

}  // namespace cuda
}  // namespace nexus
