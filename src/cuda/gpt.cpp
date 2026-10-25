/**
 * @file gpt.cpp
 * @brief FAZ 3/4 GPU GPT-2 egitim yolu: FP32/TF32 ve BF16/F16 karisik hassasiyet,
 *        ileri, geri, foreach AdamW (ana kopya modlari), dinamik kayip olcegi, egitmen.
 *        FAZ 7: yeniden hesaplama (selective/full/every_k), parcali lm_head+CE, 8-bit AdamW,
 *        ZeRO-Offload (host AdamW, katman katman gradyan akisi).
 *
 * Satir-ana duzen; agirliklar [giris, cikis] (y = x W). Katman l'nin girisi res[l],
 * cikisi res[l+1]; res[0] gomme ciktisidir. add&norm fuzyonu ile residual toplami
 * ve sonraki LayerNorm tek cekirdektir.
 *
 * Tip kurallari (autocast tablosu): matmul/bmm (Lower) compute_dtype'ta; softmax, CE,
 * LayerNorm (Fp32) FP32 biriktirir. Depolama: residual, LN istatistikleri, logit ve tum
 * gradyan tamponlari FP32; GEMM'den GEMM'e gecen aktivasyonlar compute_dtype. GEMM'in FP32
 * tampona yazdigi yerlerde (residual'a eklenen projeksiyonlar, dW, dX -> LN geri) cikti
 * tipi FP32 secilir; boylece dar gradyan toplami olmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gpt.hpp"

#include "nexus/amp/autocast.hpp"
#include "nexus/cuda/attention.hpp"
#include "nexus/core/error.hpp"
#include "nexus/core/half.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/fused/training.hpp"
#include "nexus/prof/memory_ledger.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <span>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace nexus {
namespace cuda {
namespace {

// Matrisler once, kucuk parametreler (LN, bias) sonra: ana kopyasiz modda (C) FP32 ana
// kopya yalniz kuyruk bolgesini kapsar.
enum Param : int {
  kWte = 0,
  kWpe,
  kQkvW,
  kAttProjW,
  kFcW,
  kFcProjW,
  kLn1W,
  kLn1B,
  kQkvB,
  kAttProjB,
  kLn2W,
  kLn2B,
  kFcB,
  kFcProjB,
  kLnfW,
  kLnfB,
  kParamCount
};
constexpr int kFirstSmall = kLn1W;

struct Mat {
  const void* p;
  DType t;
};

struct OutBuf {
  void* p;
  DType t;
};

// Satir-ana GEMM: C[m,n] = op(A) op(B) (+ epilog). A ve B ayni tip, C F32 ya da A tipi.
void mm(cudaStream_t s, bool ta, bool tb, int m, int n, int k, Mat a, Mat b, OutBuf c, Epilogue epi = Epilogue::None,
        const void* bias = nullptr, void* aux = nullptr, void* bias_grad = nullptr, float beta = 0.0f) {
  GemmArgs g;
  g.trans_a = ta;
  g.trans_b = tb;
  g.m = m;
  g.n = n;
  g.k = k;
  g.a = a.p;
  g.lda = ta ? m : k;
  g.b = b.p;
  g.ldb = tb ? k : n;
  g.c = c.p;
  g.ldc = n;
  g.a_type = a.t;
  g.b_type = b.t;
  g.c_type = c.t;
  g.bias_type = c.t;  // bias/bias_grad cikti tipinde (dar cikti -> dar bias golgesi)
  g.epilogue = epi;
  g.bias = bias;
  g.aux = aux;
  g.bias_grad = bias_grad;
  g.beta = beta;
  gemm(g, s);
}

// Strided batched: count adet bagimsiz [m,n] = op(A)[m,k] op(B)[k,n]; tum tamponlar tip t.
void bmm(cudaStream_t s, DType t, bool ta, bool tb, int count, int m, int n, int k, const void* a, long long sa,
         const void* b, long long sb, void* c, long long sc) {
  GemmArgs g;
  g.trans_a = ta;
  g.trans_b = tb;
  g.m = m;
  g.n = n;
  g.k = k;
  g.a = a;
  g.lda = ta ? m : k;
  g.b = b;
  g.ldb = tb ? k : n;
  g.c = c;
  g.ldc = n;
  g.a_type = t;
  g.b_type = t;
  g.c_type = t;
  g.batch = count;
  g.stride_a = sa;
  g.stride_b = sb;
  g.stride_c = sc;
  gemm(g, s);
}

constexpr std::size_t kHostCastChunk = std::size_t{1} << 24;  // 16M eleman (64 MB FP32) yukleme/indirme parcasi

void* at(const DeviceBuffer& buffer, std::size_t elements, DType t) {
  return static_cast<char*>(buffer.data()) + elements * size_of(t);
}

bool sdpa_debug() {
#ifdef _MSC_VER
  char* value = nullptr;
  std::size_t length = 0;
  const bool on = _dupenv_s(&value, &length, "NEXUS_SDPA_DEBUG") == 0 && value != nullptr && value[0] == '1';
  std::free(value);
  return on;
#else
  const char* value = std::getenv("NEXUS_SDPA_DEBUG");
  return value != nullptr && value[0] == '1';
#endif
}

// GPT'nin [B,T,3,H,D] qkv tamponunu kopyasiz okuyan causal oz-dikkat parametreleri.
AttentionParams gpt_attention_params(int B, int T, int H, int D, DType t, bool deterministic) {
  AttentionParams p;
  p.batch = B;
  p.heads = H;
  p.kv_heads = H;
  p.tq = T;
  p.tk = T;
  p.head_dim = D;
  p.dtype = t;
  p.causal = true;
  p.deterministic = deterministic;
  const long long C = static_cast<long long>(H) * D;
  p.q_row_stride = p.k_row_stride = 3 * C;
  p.q_head_stride = p.k_head_stride = D;
  p.q_batch_stride = p.k_batch_stride = static_cast<long long>(T) * 3 * C;
  p.o_row_stride = C;
  p.o_head_stride = D;
  p.o_batch_stride = static_cast<long long>(T) * C;
  return p;
}

}  // namespace

const char* to_string(AttentionBackend backend) noexcept {
  switch (backend) {
    case AttentionBackend::Auto:
      return "auto";
    case AttentionBackend::Math:
      return "math";
    case AttentionBackend::Flash:
      return "flash";
  }
  return "unknown";
}

const char* to_string(MasterWeights mode) noexcept {
  switch (mode) {
    case MasterWeights::CastPerForward:
      return "A:cast_per_forward";
    case MasterWeights::Fp32Master:
      return "B:fp32_master";
    case MasterWeights::NoMaster:
      return "C:no_master_stochastic";
  }
  return "unknown";
}

const char* to_string(Recompute mode) noexcept {
  switch (mode) {
    case Recompute::None:
      return "none";
    case Recompute::Selective:
      return "selective";
    case Recompute::Full:
      return "full";
    case Recompute::EveryK:
      return "every_k";
  }
  return "unknown";
}

const char* to_string(OptimizerKind kind) noexcept {
  switch (kind) {
    case OptimizerKind::AdamW:
      return "adamw32";
    case OptimizerKind::AdamW8bit:
      return "adamw8bit";
    case OptimizerKind::ZeroOffload:
      return "zero_offload";
    case OptimizerKind::AdamW8bitFused:
      return "adamw8bit_fused";
  }
  return "unknown";
}

namespace {
// Aktivasyon turleri (yeniden hesaplama yuvasi icin).
enum Act : int { kActLn1 = 0, kActQkv, kActAtt, kActXmid, kActLn2, kActFcPre, kActFcAct, kActMath, kActCount };
}  // namespace

struct GptModel::Buffers {
  // Katman basina saklanan aktivasyonlar (geri gecis icin). (t) = compute_dtype, (f) = FP32.
  // FAZ 7: buyuk aktivasyonlar "yuva"lara yazilir; Recompute moduna gore yuva sayisi L ya da 1.
  DeviceBuffer res;                      // (f) [L+1, BT, C]  her zaman (katman girisi)
  DeviceBuffer ln1, ln1_mean, ln1_rstd;  // (t) [yuva, BT, C], (f) [L, BT]
  DeviceBuffer qkv;                      // (t) [yuva, BT, 3C]
  DeviceBuffer q, k, v;                  // (t) [yuva, B, H, T, D]       yalniz math
  DeviceBuffer probs;                    // (t) [yuva, B, H, T, T]       yalniz math (hesap FP32)
  DeviceBuffer atty;                     // (t) [yuva, B, H, T, D]       yalniz math
  DeviceBuffer lse;                      // (f) [L, B, H, T]             yalniz flash, her zaman L
  DeviceBuffer att;                      // (t) [yuva, BT, C]
  DeviceBuffer xmid;                     // (f) [yuva, BT, C]
  DeviceBuffer ln2, ln2_mean, ln2_rstd;  // (t) [yuva] / (f) [L]
  DeviceBuffer fc_pre, fc_act;           // (t) [yuva, BT, 4C]
  DeviceBuffer lnf, lnf_mean, lnf_rstd;  // (t) / (f)
  DeviceBuffer logits;                   // (f) [BT ya da parca, V]  (yerinde dlogits)
  // Gecici
  DeviceBuffer scores;         // (t) [B, H, T, T]  yalniz math
  DeviceBuffer flash_scratch;  // yalniz flash: D vektoru + dQ FP32 biriktirici
  DeviceBuffer proj;    // (f) [BT, C]
  // Geri gecis gecicileri
  DeviceBuffer dres, dln;                                      // (f)
  DeviceBuffer dres_t, dlogits_t;                              // (t) yalniz dar modda
  DeviceBuffer dqkv, dq, dk, dv, datty, datt, dfc;              // (t)
  DeviceBuffer norm_scratch, emb_scratch, ce_scratch, adam_scratch;
  DeviceBuffer bias_grad;  // FAZ 9.2: biriktirmede BGRAD epilogu ciktisi (sonra += ile eklenir), [4C]
  // Yuva eslemesi: slot[tur][katman] -> yuva indeksi; recompute[katman]: geri geciste yeniden ileri.
  std::vector<int> slot[kActCount];
  std::vector<char> recompute;
  bool selective = false;
  int logit_rows = 0;
};

// ZeroOffload host durumu ve akis tamponlari.
struct GptModel::Offload {
  std::vector<float> master, m, v;  // host FP32 [num_params]
  void* pinned_grad = nullptr;      // [num_params] offload_grad_dtype
  std::uint16_t* pinned_w = nullptr;  // [num_params] BF16 (BF16 gradyanda pinned_grad ile ayni)
  bool alias = false;
  DeviceBuffer scratch[2], staging[2], rest_staging;
  std::size_t layer_numel = 0;      // katman basina matris elemanlari (hizali)
  std::size_t mat_local[kParamCount] = {};
  Stream copy;
  Event layer_ready[2], copy_done[2], grads_ready;
  bool slot_used[2] = {false, false};
  // RAII: kurucu yolda istisna (orn. host bad_alloc) olsa da pinned bellek ve kopya akisi birakilir.
  ~Offload() {
    try {
      if (copy.handle() != nullptr) {
        copy.synchronize();
      }
    } catch (...) {
    }
    if (!alias && pinned_w != nullptr) {
      pinned_allocator().deallocate(pinned_w);
    }
    if (pinned_grad != nullptr) {
      pinned_allocator().deallocate(pinned_grad);
    }
    try {
      if (copy.handle() != nullptr) {
        copy.destroy();
      }
    } catch (...) {
    }
  }
};

std::size_t GptModel::rest_index(std::size_t offset) const noexcept {
  return offset < matrices_begin_ ? offset : offset - (small_begin_ - matrices_begin_);
}

float* GptModel::master(std::size_t offset) const {
  if (params_.data() == nullptr) {
    return nullptr;
  }
  if (offload()) {
    // ZeroOffload: aygitta yalniz wte/wpe + kucuk parametrelerin FP32 kopyasi (ileri yol mod B ile ayni).
    return in_layer_matrices(offset) ? nullptr : params_.as<float>() + rest_index(offset);
  }
  if (offset < master_begin_) {
    return nullptr;
  }
  return params_.as<float>() + (offset - master_begin_);
}

void* GptModel::shadow(std::size_t offset) const {
  if (!narrow()) {
    return nullptr;
  }
  return at(shadow_, offset, config_.compute_dtype);
}

const void* GptModel::weight(std::size_t offset) const {
  return narrow() ? shadow(offset) : static_cast<const void*>(master(offset));
}

GptModel::GptModel(const GptConfig& config, int batch, int time, std::uint64_t seed, const Stream& stream)
    : config_(config), batch_(batch), time_(time), stream_(stream) {
  if (batch < 1 || time < 2 || time > config.max_seq || config.dim % config.heads != 0 || config.layers < 1 ||
      config.vocab < 2) {
    throw TensorError("Nexus GPT: gecersiz yapilandirma (time <= max_seq, dim % heads == 0)");
  }
  const DType ct = config_.compute_dtype;
  if (ct != DType::F32 && ct != DType::BF16 && ct != DType::F16) {
    throw TensorError("Nexus GPT: compute_dtype F32, BF16 veya F16 olmali");
  }
  if (config_.master == MasterWeights::NoMaster && ct != DType::BF16) {
    throw TensorError("Nexus GPT: ana kopyasiz mod (C) yalniz BF16'da (stokastik yuvarlama) desteklenir");
  }
  if (offload() && (ct != DType::BF16 || config_.dynamic_loss_scale)) {
    throw TensorError("Nexus GPT: ZeroOffload yalniz BF16 ve dinamik kayip olceksiz desteklenir");
  }
  if (offload() && config_.offload_grad_dtype != DType::BF16 && config_.offload_grad_dtype != DType::F32) {
    throw TensorError("Nexus GPT: offload_grad_dtype BF16 ya da F32 olmali");
  }
  if (config_.optimizer == OptimizerKind::AdamW8bit && (ct == DType::F16 || config_.dynamic_loss_scale)) {
    throw TensorError("Nexus GPT: AdamW8bit F32/BF16 ve dinamik kayip olceksiz desteklenir");
  }
  if (fused() && (ct != DType::BF16 || config_.master != MasterWeights::NoMaster || config_.dynamic_loss_scale)) {
    throw TensorError("Nexus GPT: AdamW8bitFused BF16 + ana kopyasiz (C) + kayip olceksiz ister");
  }
  if (config_.recompute == Recompute::EveryK && config_.recompute_every < 1) {
    throw TensorError("Nexus GPT: recompute_every >= 1 olmali");
  }
  if (config_.ce_chunk_rows < 0) {
    throw TensorError("Nexus GPT: ce_chunk_rows >= 0 olmali");
  }
  // F16'nin dar us araligi gradyani sifira dusurur: dinamik kayip olcegi zorunlu.
  use_loss_scale_ = config_.dynamic_loss_scale || ct == DType::F16;
  build_layout();
  for (const GptParamInfo& p : params_info_) {
    scales_.push_back(GptParamScale{p.decay ? 1.0f : 0.0f, 1.0f});
  }
  rest_count_ = matrices_begin_ + (num_params_ - small_begin_);
  {
    const int D = config_.dim / config_.heads;
    std::string why = "compute_dtype F32 (flash yalniz BF16/F16)";
    const bool can_flash =
        narrow() && flash_supported(gpt_attention_params(batch, time, config_.heads, D, ct, false), &why);
    if (config_.attention == AttentionBackend::Flash && !can_flash) {
      throw TensorError("Nexus GPT: attention=Flash istendi ama desteklenmiyor: " + why);
    }
    if (config_.attention == AttentionBackend::Math) {
      why = "attention=Math istendi";
    }
    flash_ = config_.attention != AttentionBackend::Math && can_flash;
    if (sdpa_debug()) {
      if (flash_) {
        std::fprintf(stderr, "[nexus sdpa] GPT dikkat yolu: flash\n");
      } else {
        std::fprintf(stderr, "[nexus sdpa] GPT dikkat yolu: math (neden: %s)\n", why.c_str());
      }
    }
  }
  if (config_.recompute == Recompute::Selective && !flash_) {
    throw TensorError("Nexus GPT: selective recompute flash dikkat ister (math yolu olasilik tamponu saklar)");
  }
  const bool no_master = narrow() && config_.master == MasterWeights::NoMaster;
  master_begin_ = no_master ? small_begin_ : 0;
  const std::size_t master_count = offload() ? rest_count_ : num_params_ - master_begin_;
  const std::size_t sh = narrow() ? num_params_ * size_of(ct) : 0;
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::Parameter);
    params_ = DeviceBuffer(master_count * sizeof(float), stream_);
    if (narrow()) {
      shadow_ = DeviceBuffer(sh, stream_);
    }
  }
  cudaStream_t s = stream_.handle();
  const std::size_t grad_count = layer_grad_stream() ? rest_count_ : num_params_;
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::Gradient);
    grads_ = DeviceBuffer(grad_count * sizeof(float), stream_);
  }
  NEXUS_CUDA_CHECK(cudaMemsetAsync(grads_.data(), 0, grad_count * 4, s));
  std::size_t opt_bytes = 0;
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::OptimizerState);
    if (config_.optimizer == OptimizerKind::AdamW) {
      exp_avg_ = DeviceBuffer(num_params_ * sizeof(float), stream_);
      exp_avg_sq_ = DeviceBuffer(num_params_ * sizeof(float), stream_);
      NEXUS_CUDA_CHECK(cudaMemsetAsync(exp_avg_.data(), 0, num_params_ * 4, s));
      NEXUS_CUDA_CHECK(cudaMemsetAsync(exp_avg_sq_.data(), 0, num_params_ * 4, s));
      opt_bytes = num_params_ * 8;
    } else if (config_.optimizer == OptimizerKind::AdamW8bit || fused()) {
      // 8-bit bolge: katman matrisleri (+ istege bagli gomme). 32-bit durumlar sikistirilmis
      // "rest" indeksiyle (gomme 32-bit ise wte/wpe + kucukler; degilse yalniz kucukler da rest'te tutulur).
      q8_begin_ = config_.adam8_embedding_fp32 ? matrices_begin_ : 0;
      const std::size_t n8 = small_begin_ - q8_begin_;
      std::size_t blocks = 0;
      for (int id = 0; id < kFirstSmall; ++id) {
        if (params_info_[static_cast<std::size_t>(id)].offset >= q8_begin_) {
          blocks += (params_info_[static_cast<std::size_t>(id)].numel + kQ8Block - 1) / kQ8Block;
        }
      }
      q8_m_ = DeviceBuffer(n8, stream_);
      q8_v_ = DeviceBuffer(n8, stream_);
      q8_absmax_m_ = DeviceBuffer(blocks * 4, stream_);
      q8_absmax_v_ = DeviceBuffer(blocks * 4, stream_);
      NEXUS_CUDA_CHECK(cudaMemsetAsync(q8_m_.data(), 0, n8, s));
      NEXUS_CUDA_CHECK(cudaMemsetAsync(q8_v_.data(), 0, n8, s));
      NEXUS_CUDA_CHECK(cudaMemsetAsync(q8_absmax_m_.data(), 0, blocks * 4, s));
      NEXUS_CUDA_CHECK(cudaMemsetAsync(q8_absmax_v_.data(), 0, blocks * 4, s));
      // 32-bit durumlar yalniz 32-bit kume icin: gomme 8-bit ise yalniz kucuk parametreler.
      const std::size_t count32 = q8_begin_ == 0 ? num_params_ - small_begin_ : rest_count_;
      exp_avg_ = DeviceBuffer(count32 * sizeof(float), stream_);
      exp_avg_sq_ = DeviceBuffer(count32 * sizeof(float), stream_);
      NEXUS_CUDA_CHECK(cudaMemsetAsync(exp_avg_.data(), 0, count32 * 4, s));
      NEXUS_CUDA_CHECK(cudaMemsetAsync(exp_avg_sq_.data(), 0, count32 * 4, s));
      opt_bytes = 2 * n8 + 8 * blocks + count32 * 8;
    }
  }
  device_state_bytes_ = master_count * 4 + sh + grad_count * 4 + opt_bytes;
  state_bytes_ = device_state_bytes_;
  loss_ = DeviceBuffer(sizeof(float), stream_);
  grad_norm_ = DeviceBuffer(sizeof(float), stream_);
  loss_scale_state_ = DeviceBuffer(sizeof(DeviceLossScaleState), stream_);
  nexus_train::LossScalePolicy policy = config_.loss_scale;
  if (!use_loss_scale_) {
    policy.init_scale = 1.0;
  }
  loss_scale_init(loss_scale_state_.as<DeviceLossScaleState>(), policy, s);

  const std::size_t L = static_cast<std::size_t>(config_.layers);
  const std::size_t C = static_cast<std::size_t>(config_.dim);
  if (layer_grad_stream()) {
    offload_ = std::make_unique<Offload>();
    Offload& o = *offload_;
    std::size_t local = 0;
    for (int id = kQkvW; id <= kFcProjW; ++id) {
      o.mat_local[id] = local;
      const std::size_t per = params_info_[static_cast<std::size_t>(id)].numel / L;
      if (fused() && per % kQ8Block != 0) {
        throw TensorError("Nexus GPT: AdamW8bitFused katman dilimi 2048'in kati olmali (8-bit blok hizasi)");
      }
      local += (per + 63) / 64 * 64;
    }
    o.layer_numel = local;
  }
  if (fused()) {
    Offload& o = *offload_;
    o.scratch[0] = DeviceBuffer(o.layer_numel * 4, stream_);
    fused_scratch_ = DeviceBuffer(adamw8bit_scratch_bytes(kParamCount, kParamCount, rest_count_, num_params_), stream_);
    fused_clip_ = DeviceBuffer(sizeof(float), stream_);
    fused_norm_accum_ = DeviceBuffer(sizeof(double), stream_);
    fill(fused_clip_.as<float>(), 1.0f, 1, s);  // ilk adim kirpmasiz (onceki norm yok)
    NEXUS_CUDA_CHECK(cudaMemsetAsync(fused_norm_accum_.data(), 0, sizeof(double), s));
    device_state_bytes_ += o.layer_numel * 4;
    state_bytes_ = device_state_bytes_;
  }
  if (offload()) {
    Offload& o = *offload_;
    o.copy = Stream::create(0);
    const std::size_t local = o.layer_numel;
    const std::size_t gsz = size_of(config_.offload_grad_dtype);
    for (int i = 0; i < 2; ++i) {
      o.scratch[i] = DeviceBuffer(local * 4, stream_);
      if (config_.offload_grad_dtype != DType::F32) {
        o.staging[i] = DeviceBuffer(local * gsz, stream_);
      }
    }
    if (config_.offload_grad_dtype != DType::F32) {
      o.rest_staging = DeviceBuffer(rest_count_ * gsz, stream_);
    }
    o.pinned_grad = pinned_allocator().allocate(num_params_ * gsz);
    o.alias = config_.offload_grad_dtype == DType::BF16;
    o.pinned_w = o.alias ? static_cast<std::uint16_t*>(o.pinned_grad)
                         : static_cast<std::uint16_t*>(pinned_allocator().allocate(num_params_ * 2));
    o.m.assign(num_params_, 0.0f);
    o.v.assign(num_params_, 0.0f);
    host_state_bytes_ = num_params_ * 12 + num_params_ * gsz + (o.alias ? 0 : num_params_ * 2);
    device_state_bytes_ += 2 * local * 4 + (config_.offload_grad_dtype != DType::F32 ? 2 * local * gsz + rest_count_ * gsz : 0);
    state_bytes_ = device_state_bytes_ + host_state_bytes_;
  }

  // GPT-2 baslatmasi (host'ta, tohumla).
  std::vector<float> host(num_params_, 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float residual_scale = 0.02f / std::sqrt(2.0f * static_cast<float>(config_.layers));
  for (const GptParamInfo& p : params_info_) {
    float* dst = host.data() + p.offset;
    if (p.name == "ln1_w" || p.name == "ln2_w" || p.name == "lnf_w") {
      std::fill(dst, dst + p.numel, 1.0f);
    } else if (p.decay) {
      const float std = (p.name == "attproj_w" || p.name == "fcproj_w") ? residual_scale : 0.02f;
      for (std::size_t i = 0; i < p.numel; ++i) {
        dst[i] = std * normal(rng);
      }
    }
  }
  if (offload()) {
    offload_->master = std::move(host);  // 1.5B'de ikinci 6 GB kopya olusmasin
    sync_device_from_host_master(true);
  } else {
    upload_parameters(host);
  }

  // Aktivasyonlar.
  prof::AllocationLabelScope label(prof::MemoryLabel::Activation);
  buffers_ = std::make_unique<Buffers>();
  Buffers& b = *buffers_;
  const std::size_t BT = static_cast<std::size_t>(batch) * time;
  const std::size_t H = static_cast<std::size_t>(config_.heads);
  const std::size_t T = static_cast<std::size_t>(time);
  const std::size_t V = static_cast<std::size_t>(config_.vocab);
  const std::size_t BHTT = static_cast<std::size_t>(batch) * H * T * T;
  const std::size_t et = size_of(ct);
  // Yuva eslemesi.
  b.recompute.assign(L, 0);
  b.selective = config_.recompute == Recompute::Selective;
  int stored = 0;
  std::vector<int> layer_slot(L, 0);
  for (std::size_t l = 0; l < L; ++l) {
    const bool rc = config_.recompute == Recompute::Full ||
                    (config_.recompute == Recompute::EveryK && l % static_cast<std::size_t>(config_.recompute_every) == 0);
    b.recompute[l] = rc ? 1 : 0;
    layer_slot[l] = rc ? -1 : stored++;
  }
  const bool any_rc = std::any_of(b.recompute.begin(), b.recompute.end(), [](char c) { return c != 0; });
  const int workspace = stored;  // yeniden hesaplanan katmanlarin ortak yuvasi
  const int full_slots = stored + (any_rc ? 1 : 0);
  for (int kind = 0; kind < kActCount; ++kind) {
    b.slot[kind].assign(L, 0);
    for (std::size_t l = 0; l < L; ++l) {
      if (b.selective) {
        const bool cheap = kind == kActLn1 || kind == kActLn2 || kind == kActFcAct;
        b.slot[kind][l] = cheap ? 0 : static_cast<int>(l);
      } else {
        b.slot[kind][l] = layer_slot[l] < 0 ? workspace : layer_slot[l];
      }
    }
  }
  auto slots = [&](int kind) -> std::size_t {
    if (b.selective) {
      const bool cheap = kind == kActLn1 || kind == kActLn2 || kind == kActFcAct;
      return cheap ? 1 : L;
    }
    return static_cast<std::size_t>(full_slots);
  };
  std::size_t total = 0;
  auto alloc = [&](DeviceBuffer& buf, std::size_t count, std::size_t elem) {
    buf = DeviceBuffer(count * elem, stream_);
    total += count * elem;
  };
  alloc(b.res, (L + 1) * BT * C, 4);
  alloc(b.ln1, slots(kActLn1) * BT * C, et);
  alloc(b.ln1_mean, L * BT, 4);
  alloc(b.ln1_rstd, L * BT, 4);
  alloc(b.qkv, slots(kActQkv) * BT * 3 * C, et);
  if (flash_) {
    alloc(b.lse, L * BT * H, 4);
  } else {
    alloc(b.q, slots(kActMath) * BT * C, et);
    alloc(b.k, slots(kActMath) * BT * C, et);
    alloc(b.v, slots(kActMath) * BT * C, et);
    alloc(b.probs, slots(kActMath) * BHTT, et);
    alloc(b.atty, slots(kActMath) * BT * C, et);
  }
  alloc(b.att, slots(kActAtt) * BT * C, et);
  alloc(b.xmid, slots(kActXmid) * BT * C, 4);
  alloc(b.ln2, slots(kActLn2) * BT * C, et);
  alloc(b.ln2_mean, L * BT, 4);
  alloc(b.ln2_rstd, L * BT, 4);
  alloc(b.fc_pre, slots(kActFcPre) * BT * 4 * C, et);
  alloc(b.fc_act, slots(kActFcAct) * BT * 4 * C, et);
  alloc(b.lnf, BT * C, et);
  alloc(b.lnf_mean, BT, 4);
  alloc(b.lnf_rstd, BT, 4);
  b.logit_rows = config_.ce_chunk_rows > 0 ? std::min(config_.ce_chunk_rows, static_cast<int>(BT)) : static_cast<int>(BT);
  alloc(b.logits, static_cast<std::size_t>(b.logit_rows) * V, 4);
  if (flash_) {
    const AttentionParams ap = gpt_attention_params(batch, time, config_.heads, static_cast<int>(C / H), ct,
                                                    config_.deterministic_attention);
    alloc(b.flash_scratch, flash_attention_backward_scratch_bytes(ap), 1);
  } else {
    alloc(b.scores, BHTT, et);
  }
  alloc(b.proj, BT * C, 4);
  alloc(b.dres, BT * C, 4);
  alloc(b.dln, BT * C, 4);
  if (narrow()) {
    alloc(b.dres_t, BT * C, et);
    alloc(b.dlogits_t, static_cast<std::size_t>(b.logit_rows) * V, et);
  }
  alloc(b.dqkv, BT * 3 * C, et);
  if (!flash_) {
    alloc(b.dq, BT * C, et);
    alloc(b.dk, BT * C, et);
    alloc(b.dv, BT * C, et);
    alloc(b.datty, BT * C, et);
  }
  alloc(b.datt, BT * C, et);
  alloc(b.dfc, BT * 4 * C, et);
  b.norm_scratch = DeviceBuffer(layernorm_backward_scratch_bytes(static_cast<int>(BT), static_cast<int>(C)), stream_);
  b.bias_grad = DeviceBuffer(4 * C * sizeof(float), stream_);
  b.emb_scratch = DeviceBuffer(embedding_backward_scratch_bytes(batch, time), stream_);
  b.ce_scratch = DeviceBuffer(cross_entropy_scratch_bytes(batch, time), stream_);
  if (config_.optimizer == OptimizerKind::AdamW8bit || fused()) {
    b.adam_scratch = DeviceBuffer(adamw8bit_scratch_bytes(kParamCount, kParamCount, rest_count_, num_params_), stream_);
  } else {
    b.adam_scratch = DeviceBuffer(adamw_scratch_bytes(kParamCount, num_params_), stream_);
  }
  activation_bytes_ = total;
}

GptModel::~GptModel() {
  if (offload_ && offload()) {
    try {
      stream_.synchronize();  // kopyalar pinned tamponlari okurken birakilmasin (Offload yikicisi birakir)
    } catch (...) {
    }
  }
}

void GptModel::build_layout() {
  const int L = config_.layers;
  const int C = config_.dim;
  const int V = config_.vocab;
  const int P = config_.max_seq;
  struct Spec {
    const char* name;
    std::vector<int> shape;
    bool decay;
  };
  const Spec specs[kParamCount] = {
      {"wte", {V, C}, true},           {"wpe", {P, C}, true},
      {"qkv_w", {L, C, 3 * C}, true},  {"attproj_w", {L, C, C}, true},
      {"fc_w", {L, C, 4 * C}, true},   {"fcproj_w", {L, 4 * C, C}, true},
      {"ln1_w", {L, C}, false},        {"ln1_b", {L, C}, false},
      {"qkv_b", {L, 3 * C}, false},    {"attproj_b", {L, C}, false},
      {"ln2_w", {L, C}, false},        {"ln2_b", {L, C}, false},
      {"fc_b", {L, 4 * C}, false},     {"fcproj_b", {L, C}, false},
      {"lnf_w", {C}, false},           {"lnf_b", {C}, false},
  };
  std::size_t offset = 0;
  for (int id = 0; id < kParamCount; ++id) {
    const Spec& spec = specs[id];
    GptParamInfo info;
    info.name = spec.name;
    info.shape = spec.shape;
    info.numel = 1;
    for (const int d : spec.shape) {
      info.numel *= static_cast<std::size_t>(d);
    }
    // Her tensör 64 elemana hizalanir (cuBLASLt ve vektor yukleme hizasi).
    offset = (offset + 63) / 64 * 64;
    if (id == kFirstSmall) {
      small_begin_ = offset;
    }
    if (id == kQkvW) {
      matrices_begin_ = offset;
    }
    info.offset = offset;
    info.decay = spec.decay;
    offset += info.numel;
    params_info_.push_back(info);
  }
  num_params_ = offset;
}

void GptModel::refresh_shadow_from_master() {
  if (!narrow()) {
    return;
  }
  const DType ct = config_.compute_dtype;
  cast(shadow(master_begin_), ct, params_.data(), DType::F32, num_params_ - master_begin_, stream_.handle());
}

// Katman l'nin ileri gecisi. recompute = true: geri geciste yeniden hesaplama; ln1 res[l]'den
// (fuzyonlu add&norm ile ayni moment kodu: bit-bit ayni) kurulur, fc_proj + sonraki add&norm atlanir.
void GptModel::layer_forward(int l, bool recompute) {
  Buffers& b = *buffers_;
  cudaStream_t s = stream_.handle();
  const int L = config_.layers;
  const int B = batch_;
  const int T = time_;
  const int C = config_.dim;
  const int H = config_.heads;
  const int D = C / H;
  const int BT = B * T;
  const std::size_t btc = static_cast<std::size_t>(BT) * C;
  const std::size_t bhtt = static_cast<std::size_t>(B) * H * T * T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float eps = config_.ln_eps;
  const DType mt = config_.compute_dtype;
  const auto& info = params_info_;
  auto off = [&](int id) { return info[static_cast<std::size_t>(id)].offset; };
  auto w = [&](int id, std::size_t extra = 0) { return Mat{weight(off(id) + extra), mt}; };
  auto f32 = [&](int id, std::size_t extra = 0) { return static_cast<const float*>(master(off(id) + extra)); };
  auto act = [&](const DeviceBuffer& buf, std::size_t elements) { return at(buf, elements, mt); };
  auto slot = [&](int kind) { return static_cast<std::size_t>(b.slot[kind][static_cast<std::size_t>(l)]); };
  const std::size_t lc = static_cast<std::size_t>(l) * C;
  const std::size_t lt = static_cast<std::size_t>(l) * btc;
  float* res = b.res.as<float>();
  float* x_in = res + lt;
  float* x_out = res + static_cast<std::size_t>(l + 1) * btc;
  void* ln1 = act(b.ln1, slot(kActLn1) * btc);
  void* qkv = act(b.qkv, slot(kActQkv) * 3 * btc);
  void* att = act(b.att, slot(kActAtt) * btc);
  float* xmid = b.xmid.as<float>() + slot(kActXmid) * btc;
  void* ln2 = act(b.ln2, slot(kActLn2) * btc);
  void* fc_pre = act(b.fc_pre, slot(kActFcPre) * 4 * btc);
  void* fc_act = act(b.fc_act, slot(kActFcAct) * 4 * btc);
  float* ln1_mean = b.ln1_mean.as<float>() + static_cast<std::size_t>(l) * BT;
  float* ln1_rstd = b.ln1_rstd.as<float>() + static_cast<std::size_t>(l) * BT;
  if (recompute) {
    layernorm_forward(ln1, mt, ln1_mean, ln1_rstd, x_in, f32(kLn1W, lc), f32(kLn1B, lc), BT, C, eps, s);
  }
  // QKV = ln1 W + b  (dar cikti, dar bias golgesi)
  mm(s, false, false, BT, 3 * C, C, Mat{ln1, mt}, w(kQkvW, lc * 3 * C), OutBuf{qkv, mt}, Epilogue::Bias,
     weight(off(kQkvB) + lc * 3));
  if (flash_) {
    // FlashAttention-2: Q/K/V qkv'den adimla okunur, O dogrudan att'a ([B,T,C]) yazilir.
    const AttentionParams ap = gpt_attention_params(B, T, H, D, mt, config_.deterministic_attention);
    flash_attention_forward(att, b.lse.as<float>() + static_cast<std::size_t>(l) * BT * H, qkv,
                            static_cast<char*>(qkv) + static_cast<std::size_t>(C) * size_of(mt),
                            static_cast<char*>(qkv) + static_cast<std::size_t>(2 * C) * size_of(mt), ap, s);
  } else {
    void* q = act(b.q, slot(kActMath) * btc);
    void* k = act(b.k, slot(kActMath) * btc);
    void* v = act(b.v, slot(kActMath) * btc);
    void* probs = act(b.probs, slot(kActMath) * bhtt);
    void* atty = act(b.atty, slot(kActMath) * btc);
    split_qkv_heads(q, k, v, qkv, mt, B, T, H, D, s);
    bmm(s, mt, false, true, B * H, T, T, D, q, static_cast<long long>(T) * D, k, static_cast<long long>(T) * D,
        b.scores.data(), static_cast<long long>(T) * T);
    softmax_causal_forward(probs, b.scores.data(), mt, B * H, T, scale, s);
    bmm(s, mt, false, false, B * H, T, D, T, probs, static_cast<long long>(T) * T, v, static_cast<long long>(T) * D,
        atty, static_cast<long long>(T) * D);
    merge_heads(att, atty, mt, B, T, H, D, s);
  }
  // proj (FP32 cikti, FP32 bias) + residual, ardindan LN2 (add&norm)
  mm(s, false, false, BT, C, C, Mat{att, mt}, w(kAttProjW, lc * C), OutBuf{b.proj.data(), DType::F32},
     Epilogue::Bias, f32(kAttProjB, lc));
  residual_layernorm_forward(xmid, ln2, mt, b.ln2_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                             b.ln2_rstd.as<float>() + static_cast<std::size_t>(l) * BT, x_in, b.proj.as<float>(),
                             f32(kLn2W, lc), f32(kLn2B, lc), BT, C, eps, s);
  if (b.selective) {
    // Selective: fc_pre GEMM ciktisi saklanir, GELU ayri cekirdekle (geri geciste ayni cekirdekle
    // yeniden uretilir: ileri ve yeniden hesaplanan fc_act bit-bit ayni).
    mm(s, false, false, BT, 4 * C, C, Mat{ln2, mt}, w(kFcW, lc * 4 * C), OutBuf{fc_pre, mt}, Epilogue::Bias,
       weight(off(kFcB) + lc * 4));
    gelu_forward(fc_act, fc_pre, mt, btc * 4, s);
  } else {
    // MLP: GELU on-aktivasyonu aux olarak saklanir (geri geciste DGELU epilogu okur).
    mm(s, false, false, BT, 4 * C, C, Mat{ln2, mt}, w(kFcW, lc * 4 * C), OutBuf{fc_act, mt}, Epilogue::BiasGelu,
       weight(off(kFcB) + lc * 4), fc_pre);
  }
  if (recompute) {
    return;  // x_out ve sonraki ln1 zaten var
  }
  mm(s, false, false, BT, C, 4 * C, Mat{fc_act, mt}, w(kFcProjW, lc * 4 * C), OutBuf{b.proj.data(), DType::F32},
     Epilogue::Bias, f32(kFcProjB, lc));
  if (l + 1 < L) {
    const std::size_t next = static_cast<std::size_t>(l + 1);
    void* ln1_next = act(b.ln1, static_cast<std::size_t>(b.slot[kActLn1][next]) * btc);
    residual_layernorm_forward(x_out, ln1_next, mt, b.ln1_mean.as<float>() + next * BT,
                               b.ln1_rstd.as<float>() + next * BT, xmid, b.proj.as<float>(), f32(kLn1W, next * C),
                               f32(kLn1B, next * C), BT, C, eps, s);
  } else {
    residual_layernorm_forward(x_out, b.lnf.data(), mt, b.lnf_mean.as<float>(), b.lnf_rstd.as<float>(), xmid,
                               b.proj.as<float>(), f32(kLnfW), f32(kLnfB), BT, C, eps, s);
  }
}

void GptModel::run_forward(const std::int32_t* d_tokens, bool for_backward) {
  Buffers& b = *buffers_;
  cudaStream_t s = stream_.handle();
  const int L = config_.layers;
  const int B = batch_;
  const int T = time_;
  const int C = config_.dim;
  const int BT = B * T;
  const float eps = config_.ln_eps;
  // Autocast tablosu: matmul dar tipte, softmax/LN/CE FP32 hesap.
  const DeviceKind cuda_kind = DeviceKind::Cuda;
  amp::AutocastGuard autocast(cuda_kind, narrow() ? config_.compute_dtype : DType::BF16, narrow());
  const DType mt = amp::compute_dtype(cuda_kind, amp::OpClass::Linear, {DType::F32});
  if (amp::compute_dtype(cuda_kind, amp::OpClass::LayerNorm, {mt}) != DType::F32 ||
      amp::compute_dtype(cuda_kind, amp::OpClass::Softmax, {mt}) != DType::F32) {
    throw TensorError("Nexus GPT: autocast tablosu LN/softmax'i FP32 disina dusurdu");
  }
  if (narrow() && config_.master == MasterWeights::CastPerForward && !offload()) {
    refresh_shadow_from_master();  // A modu: her ileri geciste FP32 -> dar
  }
  const auto& info = params_info_;
  auto off = [&](int id) { return info[static_cast<std::size_t>(id)].offset; };
  auto f32 = [&](int id, std::size_t extra = 0) { return static_cast<const float*>(master(off(id) + extra)); };
  float* res = b.res.as<float>();
  if (master(off(kWte)) != nullptr) {
    embedding_pos_forward(res, d_tokens, f32(kWte), f32(kWpe), B, T, C, s);
  } else {
    embedding_pos_forward(res, d_tokens, shadow(off(kWte)), shadow(off(kWpe)), mt, B, T, C, s);
  }
  layernorm_forward(at(b.ln1, static_cast<std::size_t>(b.slot[kActLn1][0]) * BT * C, mt), mt, b.ln1_mean.as<float>(),
                    b.ln1_rstd.as<float>(), res, f32(kLn1W), f32(kLn1B), BT, C, eps, s);
  for (int l = 0; l < L; ++l) {
    layer_forward(l, false);
  }
  head_forward(d_tokens, for_backward);
}

// Bagli kafa + CE. Parcali modda (ce_chunk_rows > 0) ve egitimde kafanin geri gecisi de burada
// yapilir: her parcada logit -> CE ileri+geri -> dlnf parcasi ve dwte'ye biriktirme; logit
// tamponu tek parca boyutunda yeniden kullanilir.
void GptModel::head_forward(const std::int32_t* d_tokens, bool for_backward) {
  Buffers& b = *buffers_;
  cudaStream_t s = stream_.handle();
  const int B = batch_;
  const int T = time_;
  const int C = config_.dim;
  const int V = config_.vocab;
  const int BT = B * T;
  const DType mt = config_.compute_dtype;
  const std::size_t wte_off = params_info_[kWte].offset;
  const Mat wte{weight(wte_off), mt};
  CrossEntropyDeviceOptions options;
  const float* device_scale = use_loss_scale_ ? &loss_scale_state_.as<DeviceLossScaleState>()->scale : nullptr;
  if (config_.ce_chunk_rows <= 0) {
    // Logit FP32'ye yukseltilmis cikti (4.6) -> CE FP32.
    mm(s, false, true, BT, V, C, Mat{b.lnf.data(), mt}, wte, OutBuf{b.logits.data(), DType::F32});
    cross_entropy_shifted(for_backward ? b.logits.as<float>() : nullptr, loss_.as<float>(), nullptr,
                          b.logits.as<float>(), d_tokens, B, T, V, options, grad_scale_, device_scale, b.ce_scratch.data(), s);
    return;
  }
  float* dwte = grads_.as<float>() + (offload() ? rest_index(wte_off) : wte_off);
  float* dln = b.dln.as<float>();
  cross_entropy_chunked_begin(d_tokens, B, T, V, options, b.ce_scratch.data(), s);
  for (int r0 = 0; r0 < BT; r0 += b.logit_rows) {
    const int rows = std::min(b.logit_rows, BT - r0);
    const void* lnf = at(b.lnf, static_cast<std::size_t>(r0) * C, mt);
    mm(s, false, true, rows, V, C, Mat{lnf, mt}, wte, OutBuf{b.logits.data(), DType::F32});
    cross_entropy_chunk(for_backward ? b.logits.as<float>() : nullptr, b.logits.as<float>(), d_tokens, B, T, V, options,
                        grad_scale_, device_scale, r0, rows, b.ce_scratch.data(), s);
    if (!for_backward) {
      continue;
    }
    Mat dlogits{b.logits.data(), DType::F32};
    if (narrow()) {
      cast(b.dlogits_t.data(), mt, b.logits.data(), DType::F32, static_cast<std::size_t>(rows) * V, s);
      dlogits = Mat{b.dlogits_t.data(), mt};
    }
    mm(s, false, false, rows, C, V, dlogits, wte, OutBuf{dln + static_cast<std::size_t>(r0) * C, DType::F32});
    mm(s, true, false, V, C, rows, dlogits, Mat{lnf, mt}, OutBuf{dwte, DType::F32}, Epilogue::None, nullptr, nullptr,
       nullptr, (r0 > 0 || accumulate_) ? 1.0f : 0.0f);  // dwte: ilk parca yazar, sonrakiler (ve biriktirme) ekler
  }
  cross_entropy_chunked_end(loss_.as<float>(), nullptr, B, T, b.ce_scratch.data(), s);
}

void GptModel::run_backward(const std::int32_t* d_tokens) {
  Buffers& b = *buffers_;
  cudaStream_t s = stream_.handle();
  const int L = config_.layers;
  const int B = batch_;
  const int T = time_;
  const int C = config_.dim;
  const int H = config_.heads;
  const int D = C / H;
  const int V = config_.vocab;
  const int BT = B * T;
  const std::size_t btc = static_cast<std::size_t>(BT) * C;
  const std::size_t bhtt = static_cast<std::size_t>(B) * H * T * T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float eps = config_.ln_eps;
  const DType mt = config_.compute_dtype;
  float* G = grads_.as<float>();
  Offload* o = offload_.get();  // ZeroOffload ya da fused: katman matris gradyanlari tek katmanlik tamponda
  int cur = 0;                   // ZeroOffload: bu katmanin gradyan yuvasi (fused: hep 0)
  const auto& info = params_info_;
  auto off = [&](int id) { return info[static_cast<std::size_t>(id)].offset; };
  auto w = [&](int id, std::size_t extra = 0) { return Mat{weight(off(id) + extra), mt}; };
  auto f32 = [&](int id, std::size_t extra = 0) { return static_cast<const float*>(master(off(id) + extra)); };
  auto grad = [&](int id, std::size_t extra = 0) -> float* {
    const std::size_t g = off(id) + extra;
    if (o == nullptr) {
      return G + g;
    }
    if (in_layer_matrices(g)) {
      const std::size_t per = info[static_cast<std::size_t>(id)].numel / static_cast<std::size_t>(L);
      return o->scratch[cur].as<float>() + o->mat_local[id] + extra % per;
    }
    return G + rest_index(g);
  };
  auto gout = [&](int id, std::size_t extra = 0) { return OutBuf{grad(id, extra), DType::F32}; };
  auto act = [&](const DeviceBuffer& buf, std::size_t elements) { return at(buf, elements, mt); };
  float* scratch = b.norm_scratch.as<float>();
  float* dres = b.dres.as<float>();
  float* dln = b.dln.as<float>();
  // FP32 residual gradyani dar GEMM girisi olarak: dar modda bir kopya, FP32'de ayni tampon.
  auto dres_as_input = [&]() -> Mat {
    if (!narrow()) {
      return Mat{dres, DType::F32};
    }
    cast(b.dres_t.data(), mt, dres, DType::F32, btc, s);
    return Mat{b.dres_t.data(), mt};
  };

  // Agirlik gradyani GEMM'i + BGRADB epilogunda bias gradyani. FAZ 9.2 biriktirme: agirlik beta = 1 ile
  // eklenir; epilog bias gradyanini uzerine yazdigi icin ara tampona yazilip += ile eklenir.
  auto wgrad = [&](int m, int n, Mat a, Mat bm, int wid, std::size_t woff, int bid, std::size_t boff, int bn) {
    if (!accumulate_) {
      mm(s, true, false, m, n, BT, a, bm, gout(wid, woff), Epilogue::BGradB, nullptr, nullptr, grad(bid, boff));
      return;
    }
    mm(s, true, false, m, n, BT, a, bm, gout(wid, woff), Epilogue::BGradB, nullptr, nullptr, b.bias_grad.data(), 1.0f);
    add_inplace(grad(bid, boff), b.bias_grad.as<float>(), static_cast<std::size_t>(bn), s);
  };

  if (config_.ce_chunk_rows <= 0) {
    // Kafa: dlogits FP32 (CE yerinde yazdi); dar modda GEMM icin dar kopya.
    Mat dlogits{b.logits.data(), DType::F32};
    if (narrow()) {
      cast(b.dlogits_t.data(), mt, b.logits.data(), DType::F32, static_cast<std::size_t>(BT) * V, s);
      dlogits = Mat{b.dlogits_t.data(), mt};
    }
    mm(s, false, false, BT, C, V, dlogits, w(kWte), OutBuf{dln, DType::F32});              // dlnf = dlogits wte
    mm(s, true, false, V, C, BT, dlogits, Mat{b.lnf.data(), mt}, gout(kWte), Epilogue::None, nullptr, nullptr, nullptr,
       accumulate_ ? 1.0f : 0.0f);  // dwte (ilk yazim; biriktirmede +=), FP32
  }  // parcali modda dln ve dwte head_forward'da yazildi
  float* res = b.res.as<float>();
  layernorm_backward(dres, grad(kLnfW), grad(kLnfB), dln, res + static_cast<std::size_t>(L) * btc, f32(kLnfW),
                     b.lnf_mean.as<float>(), b.lnf_rstd.as<float>(), BT, C, false, scratch, s);
  if (backward_hook_) {
    backward_hook_(kStageHead);  // lnf gradyanlari kuyrukta tamam (dwte gomme geri gecisini bekler)
  }

  for (int l = L - 1; l >= 0; --l) {
    const std::size_t lc = static_cast<std::size_t>(l) * C;
    const std::size_t lt = static_cast<std::size_t>(l) * btc;
    auto slot = [&](int kind) { return static_cast<std::size_t>(b.slot[kind][static_cast<std::size_t>(l)]); };
    float* x_in = res + lt;
    if (b.recompute[static_cast<std::size_t>(l)] != 0) {
      layer_forward(l, true);
    }
    void* ln1 = act(b.ln1, slot(kActLn1) * btc);
    void* att = act(b.att, slot(kActAtt) * btc);
    float* xmid = b.xmid.as<float>() + slot(kActXmid) * btc;
    void* ln2 = act(b.ln2, slot(kActLn2) * btc);
    void* fc_pre = act(b.fc_pre, slot(kActFcPre) * 4 * btc);
    void* fc_act = act(b.fc_act, slot(kActFcAct) * 4 * btc);
    void* qkv = act(b.qkv, slot(kActQkv) * 3 * btc);
    if (b.selective) {
      // Ucuz-ama-buyuk aktivasyonlar: norm ciktilari ve GELU ciktisi (ileriyle ayni cekirdekler).
      layernorm_forward(ln1, mt, b.ln1_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                        b.ln1_rstd.as<float>() + static_cast<std::size_t>(l) * BT, x_in, f32(kLn1W, lc), f32(kLn1B, lc),
                        BT, C, eps, s);
      layernorm_forward(ln2, mt, b.ln2_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                        b.ln2_rstd.as<float>() + static_cast<std::size_t>(l) * BT, xmid, f32(kLn2W, lc), f32(kLn2B, lc),
                        BT, C, eps, s);
      gelu_forward(fc_act, fc_pre, mt, btc * 4, s);
    }
    if (o != nullptr && offload()) {
      cur = l % 2;
      if (o->slot_used[cur]) {
        o->copy_done[cur].wait(stream_);  // bu yuvanin onceki D2H'si bitmeden ustune yazma
      }
    }

    // --- MLP --- dres = d x_out
    Mat dres_in = dres_as_input();
    mm(s, false, true, BT, 4 * C, C, dres_in, w(kFcProjW, lc * 4 * C), OutBuf{b.dfc.data(), mt}, Epilogue::DGelu,
       nullptr, fc_pre);
    wgrad(4 * C, C, Mat{fc_act, mt}, dres_in, kFcProjW, lc * 4 * C, kFcProjB, lc, C);
    mm(s, false, true, BT, C, 4 * C, Mat{b.dfc.data(), mt}, w(kFcW, lc * 4 * C), OutBuf{dln, DType::F32});
    wgrad(C, 4 * C, Mat{ln2, mt}, Mat{b.dfc.data(), mt}, kFcW, lc * 4 * C, kFcB, lc * 4, 4 * C);
    layernorm_backward(dres, grad(kLn2W, lc), grad(kLn2B, lc), dln, xmid, f32(kLn2W, lc),
                       b.ln2_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                       b.ln2_rstd.as<float>() + static_cast<std::size_t>(l) * BT, BT, C, true, scratch, s);
    // --- Dikkat --- dres = d xmid
    dres_in = dres_as_input();
    mm(s, false, true, BT, C, C, dres_in, w(kAttProjW, lc * C), OutBuf{b.datt.data(), mt});
    wgrad(C, C, Mat{att, mt}, dres_in, kAttProjW, lc * C, kAttProjB, lc, C);
    if (flash_) {
      // dO = datt ([B,T,C]); dQ/dK/dV dogrudan dqkv'nin [B,T,3,H,D] dilimlerine yazilir.
      const AttentionParams ap = gpt_attention_params(B, T, H, D, mt, config_.deterministic_attention);
      const std::size_t es = size_of(mt);
      flash_attention_backward(b.dqkv.data(), at(b.dqkv, C, mt), at(b.dqkv, 2 * C, mt), b.datt.data(), qkv,
                               static_cast<char*>(qkv) + C * es, static_cast<char*>(qkv) + 2 * C * es, att,
                               b.lse.as<float>() + static_cast<std::size_t>(l) * BT * H, ap, b.flash_scratch.data(), s);
    } else {
      void* q = act(b.q, slot(kActMath) * btc);
      void* k = act(b.k, slot(kActMath) * btc);
      void* v = act(b.v, slot(kActMath) * btc);
      void* probs = act(b.probs, slot(kActMath) * bhtt);
      split_heads(b.datty.data(), b.datt.data(), mt, B, T, H, D, s);
      void* dscores = b.scores.data();
      bmm(s, mt, false, true, B * H, T, T, D, b.datty.data(), static_cast<long long>(T) * D, v,
          static_cast<long long>(T) * D, dscores, static_cast<long long>(T) * T);
      bmm(s, mt, true, false, B * H, T, D, T, probs, static_cast<long long>(T) * T, b.datty.data(),
          static_cast<long long>(T) * D, b.dv.data(), static_cast<long long>(T) * D);
      softmax_causal_backward(dscores, dscores, probs, mt, B * H, T, scale, s);  // yerinde, FP32 hesap
      bmm(s, mt, false, false, B * H, T, D, T, dscores, static_cast<long long>(T) * T, k,
          static_cast<long long>(T) * D, b.dq.data(), static_cast<long long>(T) * D);
      bmm(s, mt, true, false, B * H, T, D, T, dscores, static_cast<long long>(T) * T, q,
          static_cast<long long>(T) * D, b.dk.data(), static_cast<long long>(T) * D);
      merge_qkv_heads(b.dqkv.data(), b.dq.data(), b.dk.data(), b.dv.data(), mt, B, T, H, D, s);
    }
    mm(s, false, true, BT, C, 3 * C, Mat{b.dqkv.data(), mt}, w(kQkvW, lc * 3 * C), OutBuf{dln, DType::F32});
    wgrad(C, 3 * C, Mat{ln1, mt}, Mat{b.dqkv.data(), mt}, kQkvW, lc * 3 * C, kQkvB, static_cast<std::size_t>(l) * 3 * C, 3 * C);
    layernorm_backward(dres, grad(kLn1W, lc), grad(kLn1B, lc), dln, x_in, f32(kLn1W, lc),
                       b.ln1_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                       b.ln1_rstd.as<float>() + static_cast<std::size_t>(l) * BT, BT, C, true, scratch, s);
    if (backward_hook_) {
      backward_hook_(l);  // katman l'nin tum dilimleri kuyrukta tamam
    }
    if (fused()) {
      fused_update_layer(l, o->scratch[0].as<float>());  // ayni akis: sonraki katman tamponu guvenle yeniden yazar
    }
    if (o != nullptr && offload()) {
      // Katman l'nin dort matris gradyani hazir: (BF16'ya cevir) -> kopya akisinda pinned host'a.
      const DType gdt = config_.offload_grad_dtype;
      const std::size_t gsz = size_of(gdt);
      const void* src = o->scratch[cur].data();
      if (gdt != DType::F32) {
        cast(o->staging[cur].data(), gdt, o->scratch[cur].data(), DType::F32, o->layer_numel, s);
        src = o->staging[cur].data();
      }
      o->layer_ready[cur].record(stream_);
      o->layer_ready[cur].wait(o->copy);
      for (int id = kQkvW; id <= kFcProjW; ++id) {
        const std::size_t per = info[static_cast<std::size_t>(id)].numel / static_cast<std::size_t>(L);
        char* dst = static_cast<char*>(o->pinned_grad) + (off(id) + static_cast<std::size_t>(l) * per) * gsz;
        NEXUS_CUDA_CHECK(cudaMemcpyAsync(dst, static_cast<const char*>(src) + o->mat_local[id] * gsz, per * gsz,
                                         cudaMemcpyDeviceToHost, o->copy.handle()));
      }
      o->copy_done[cur].record(o->copy);
      o->slot_used[cur] = true;
    }
  }
  // Gomme: dwte'ye (kafanin yazdigi) EKLER; dwpe sifirdan. FP32.
  if (config_.deterministic) {
    embedding_backward_deterministic(grad(kWte), grad(kWpe), dres, d_tokens, B, T, C, V, -1, b.emb_scratch.data(), s);
  } else {
    embedding_backward_atomic(grad(kWte), grad(kWpe), dres, d_tokens, B, T, C, s);
  }
  if (backward_hook_) {
    backward_hook_(kStageEmbedding);
  }
  if (fused()) {
    fused_update_rest();
  }
  if (o != nullptr && offload()) {
    // wte/wpe + kucuk parametre gradyanlari (sikistirilmis) -> pinned host'ta kendi ofsetlerine.
    const DType gdt = config_.offload_grad_dtype;
    const std::size_t gsz = size_of(gdt);
    const void* src = G;
    if (gdt != DType::F32) {
      cast(o->rest_staging.data(), gdt, G, DType::F32, rest_count_, s);
      src = o->rest_staging.data();
    }
    o->grads_ready.record(stream_);
    o->grads_ready.wait(o->copy);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(o->pinned_grad, src, matrices_begin_ * gsz, cudaMemcpyDeviceToHost, o->copy.handle()));
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(o->pinned_grad) + small_begin_ * gsz,
                                     static_cast<const char*>(src) + matrices_begin_ * gsz,
                                     (num_params_ - small_begin_) * gsz, cudaMemcpyDeviceToHost, o->copy.handle()));
    o->grads_ready.record(o->copy);
  }
}

void GptModel::forward(const std::int32_t* d_tokens) { run_forward(d_tokens, false); }

void GptModel::forward_backward(const std::int32_t* d_tokens, float grad_scale, bool accumulate) {
  if (!std::isfinite(grad_scale)) {
    throw TensorError("Nexus GPT: grad_scale sonlu olmali");
  }
  if (accumulate && layer_grad_stream()) {
    throw TensorError("Nexus GPT: gradyan biriktirme ZeroOffload/AdamW8bitFused ile desteklenmez (tam gradyan tamponu yok)");
  }
  grad_scale_ = grad_scale;
  accumulate_ = accumulate;
  // LN agirlik/bias, wte/wpe (gomme geri) gradyanlari "+=" ile yazilir; biriktirme yoksa her adimda sifirlanir.
  if (!accumulate) {
    const std::size_t count = layer_grad_stream() ? rest_count_ : num_params_;
    NEXUS_CUDA_CHECK(cudaMemsetAsync(grads_.data(), 0, count * sizeof(float), stream_.handle()));
  }
  if (fused()) {
    if (!fused_ready_) {
      throw TensorError("Nexus GPT: AdamW8bitFused icin once set_fused_optimizer cagirin");
    }
    step_ += 1;  // guncelleme bu geri geciste (bias duzeltmesi icin adim)
  }
  run_forward(d_tokens, true);
  run_backward(d_tokens);
}

void GptModel::set_nonfinite_guard(bool enabled) {
  if (enabled && layer_grad_stream()) {
    throw TensorError("Nexus GPT: NaN korumasi ZeroOffload/AdamW8bitFused ile desteklenmez");
  }
  guard_ = enabled;
}

void GptModel::set_param_scales(const std::vector<GptParamScale>& scales) {
  if (scales.empty()) {
    scales_.clear();
    for (const GptParamInfo& p : params_info_) {
      scales_.push_back(GptParamScale{p.decay ? 1.0f : 0.0f, 1.0f});
    }
    return;
  }
  if (scales.size() != params_info_.size()) {
    throw TensorError("Nexus GPT: set_param_scales parametre sayisi kadar giris ister");
  }
  for (std::size_t i = 0; i < scales.size(); ++i) {
    if (!(scales[i].weight_decay >= 0.0f) || !(scales[i].lr >= 0.0f) || !std::isfinite(scales[i].lr) ||
        !std::isfinite(scales[i].weight_decay)) {
      throw TensorError("Nexus GPT: parametre carpanlari sonlu ve >= 0 olmali");
    }
    // ZeroOffload'in CPU AdamW'i yalniz decay bayragini bilir: farkli carpan sessizce yok sayilmasin.
    if (offload() && (scales[i].lr != 1.0f || scales[i].weight_decay != (params_info_[i].decay ? 1.0f : 0.0f))) {
      throw TensorError("Nexus GPT: ZeroOffload parametre grubu carpanlarini desteklemez");
    }
  }
  scales_ = scales;
}

GptStateSnapshot GptModel::save_state() const {
  if (offload()) {
    throw TensorError("Nexus GPT: save_state ZeroOffload'da desteklenmez (durum host'ta)");
  }
  GptStateSnapshot snap;
  snap.step = step_;
  const DeviceBuffer* buffers[] = {&params_, &shadow_, &exp_avg_, &exp_avg_sq_, &q8_m_, &q8_v_,
                                   &q8_absmax_m_, &q8_absmax_v_, &loss_scale_state_, &fused_clip_, &fused_norm_accum_};
  for (const DeviceBuffer* buf : buffers) {
    std::vector<unsigned char> host(buf->bytes());
    if (!host.empty()) {
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data(), buf->data(), host.size(), cudaMemcpyDeviceToHost, stream_.handle()));
    }
    snap.blobs.push_back(std::move(host));
  }
  stream_.synchronize();
  return snap;
}

void GptModel::load_state(const GptStateSnapshot& snap) {
  if (offload()) {
    throw TensorError("Nexus GPT: load_state ZeroOffload'da desteklenmez");
  }
  DeviceBuffer* buffers[] = {&params_, &shadow_, &exp_avg_, &exp_avg_sq_, &q8_m_, &q8_v_,
                             &q8_absmax_m_, &q8_absmax_v_, &loss_scale_state_, &fused_clip_, &fused_norm_accum_};
  if (snap.blobs.size() != std::size(buffers)) {
    throw TensorError("Nexus GPT: durum kopyasi bicimi uyusmuyor");
  }
  for (std::size_t i = 0; i < snap.blobs.size(); ++i) {
    if (snap.blobs[i].size() != buffers[i]->bytes()) {
      throw TensorError("Nexus GPT: durum kopyasi baska bir yapilandirmadan (tampon boyutu uyusmuyor)");
    }
  }
  for (std::size_t i = 0; i < snap.blobs.size(); ++i) {
    if (!snap.blobs[i].empty()) {
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(buffers[i]->data(), snap.blobs[i].data(), snap.blobs[i].size(),
                                       cudaMemcpyHostToDevice, stream_.handle()));
    }
  }
  step_ = snap.step;
  stream_.synchronize();
}

const float* GptModel::device_token_losses() const noexcept {
  // CE scratch'in ilk B*T float'i konum basina kayiptir (ce_scratch duzeni, kernels_train.cu).
  return buffers_->ce_scratch.as<float>();
}

namespace {
// Katmanli tensorlerde (ilk boyut L) dilim adlari "h.<l>.<ad>"; digerleri ad olarak.
template <typename Fn>
void for_each_slice(const std::vector<GptParamInfo>& info, int layers, Fn&& fn) {
  for (const GptParamInfo& p : info) {
    const bool layered = p.shape.size() >= 2 && p.shape.front() == layers && p.name != "wte" && p.name != "wpe";
    if (!layered) {
      fn(p.name, p.offset, p.numel);
      continue;
    }
    const std::size_t per = p.numel / static_cast<std::size_t>(layers);
    for (int l = 0; l < layers; ++l) {
      fn("h." + std::to_string(l) + "." + p.name, p.offset + static_cast<std::size_t>(l) * per, per);
    }
  }
}

double l2_norm(const float* x, std::size_t n) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += static_cast<double>(x[i]) * x[i];
  }
  return std::sqrt(acc);
}

bool all_finite(const float* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(x[i])) {
      return false;
    }
  }
  return true;
}
}  // namespace

std::vector<GptModel::NormEntry> GptModel::layer_norms() const {
  const std::vector<float> w = download_parameters();
  const std::vector<float> g = download_gradients();
  std::vector<NormEntry> out;
  for_each_slice(params_info_, config_.layers, [&](const std::string& name, std::size_t offset, std::size_t n) {
    out.push_back(NormEntry{name, l2_norm(w.data() + offset, n), l2_norm(g.data() + offset, n)});
  });
  return out;
}

GptNonFiniteReport GptModel::diagnose_nonfinite(const std::int32_t* d_tokens) {
  GptNonFiniteReport report;
  {
    const std::vector<float> w = download_parameters();
    for_each_slice(params_info_, config_.layers, [&](const std::string& name, std::size_t offset, std::size_t n) {
      if (!all_finite(w.data() + offset, n)) {
        report.parameters.push_back(name);
      }
    });
  }
  // Ileri: res[0] = gomme ciktisi, res[l + 1] = katman l ciktisi (her modda (L+1) dilim saklanir).
  run_forward(d_tokens, false);
  const int L = config_.layers;
  const std::size_t btc = static_cast<std::size_t>(batch_) * time_ * config_.dim;
  DeviceBuffer counts((static_cast<std::size_t>(L) + 1) * sizeof(unsigned long long), stream_);
  NEXUS_CUDA_CHECK(cudaMemsetAsync(counts.data(), 0, counts.bytes(), stream_.handle()));
  for (int k = 0; k <= L; ++k) {
    count_nonfinite(counts.as<unsigned long long>() + k, buffers_->res.as<float>() + static_cast<std::size_t>(k) * btc, btc,
                    stream_.handle());
  }
  std::vector<unsigned long long> host(static_cast<std::size_t>(L) + 1);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data(), counts.data(), counts.bytes(), cudaMemcpyDeviceToHost, stream_.handle()));
  report.loss = read_loss();  // senkron
  for (int k = 0; k <= L && report.first_activation.empty(); ++k) {
    if (host[static_cast<std::size_t>(k)] != 0) {
      report.first_activation = k == 0 ? std::string("embedding") : "h." + std::to_string(k - 1);
    }
  }
  if (report.first_activation.empty() && !std::isfinite(report.loss)) {
    report.first_activation = "lm_head";  // son LN, bagli kafa veya CE (ya da sozluk disi hedef token)
  }
  // Geri: gradyanlar, geri gecis sirasiyla (kafa ve son katman once).
  if (!layer_grad_stream()) {
    forward_backward(d_tokens, grad_scale_, false);
    const std::vector<float> g = download_gradients();
    std::vector<std::pair<int, std::string>> bad;
    for_each_slice(params_info_, L, [&](const std::string& name, std::size_t offset, std::size_t n) {
      if (all_finite(g.data() + offset, n)) {
        return;
      }
      int order = -1;  // lnf: en once
      if (name.rfind("h.", 0) == 0) {
        order = L - std::stoi(name.substr(2));
      } else if (name == "wte" || name == "wpe") {
        order = L + 1;
      }
      bad.emplace_back(order, name);
    });
    std::stable_sort(bad.begin(), bad.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& entry : bad) {
      report.gradients.push_back(std::move(entry.second));
    }
  }
  return report;
}

void GptModel::set_fused_optimizer(const DeviceAdamWConfig& config) {
  if (!fused()) {
    throw TensorError("Nexus GPT: set_fused_optimizer yalniz AdamW8bitFused modunda");
  }
  fused_config_ = config;
  fused_config_.shadow_type = DType::BF16;
  fused_config_.stochastic_rounding = true;
  fused_config_.loss_scale = nullptr;
  fused_config_.found_inf = nullptr;
  fused_config_.skipped_steps = nullptr;
  fused_ready_ = true;
}

namespace {
std::size_t q8_block_base(const std::vector<GptParamInfo>& info, std::size_t q8_begin, std::size_t small_begin, int id) {
  std::size_t base = 0;
  for (int j = 0; j < id; ++j) {
    const GptParamInfo& p = info[static_cast<std::size_t>(j)];
    if (p.offset >= q8_begin && p.offset < small_begin) {
      base += (p.numel + kQ8Block - 1) / kQ8Block;
    }
  }
  return base;
}
}  // namespace

void GptModel::fused_update_layer(int l, const float* layer_grads) {
  const std::size_t L = static_cast<std::size_t>(config_.layers);
  std::vector<Device8bitSlot> slots;
  for (int id = kQkvW; id <= kFcProjW; ++id) {
    const GptParamInfo& p = params_info_[static_cast<std::size_t>(id)];
    const std::size_t per = p.numel / L;
    const std::size_t o = p.offset + static_cast<std::size_t>(l) * per;
    Device8bitSlot q;
    q.param = master(o);
    q.shadow = shadow(o);
    q.grad = layer_grads + offload_->mat_local[id];
    q.qm = q8_m_.as<std::uint8_t>() + (o - q8_begin_);
    q.qv = q8_v_.as<std::uint8_t>() + (o - q8_begin_);
    const std::size_t blk = q8_block_base(params_info_, q8_begin_, small_begin_, id) + static_cast<std::size_t>(l) * (per / kQ8Block);
    q.absmax_m = q8_absmax_m_.as<float>() + blk;
    q.absmax_v = q8_absmax_v_.as<float>() + blk;
    q.numel = per;
    q.weight_decay_scale = scale_of(p).weight_decay;
    q.lr_scale = scale_of(p).lr;
    slots.push_back(q);
  }
  DeviceAdamWConfig c = fused_config_;
  c.sr_seed = 0x5EEDull ^ (static_cast<std::uint64_t>(l + 1) << 32);
  adamw_fixed_clip_update(nullptr, 0, slots.data(), static_cast<int>(slots.size()), c, step_, fused_clip_.as<float>(),
                          fused_norm_accum_.as<double>(), fused_scratch_.data(), stream_.handle());
}

void GptModel::fused_update_rest() {
  std::vector<DeviceParamSlot> slots32;
  std::vector<Device8bitSlot> slots8;
  for (int id = 0; id < kParamCount; ++id) {
    const GptParamInfo& p = params_info_[static_cast<std::size_t>(id)];
    if (in_layer_matrices(p.offset)) {
      continue;  // katman matrisleri geri geciste guncellendi
    }
    if (p.offset >= q8_begin_ && p.offset < small_begin_) {  // 8-bit gomme (adam8_embedding_fp32 = false)
      Device8bitSlot q;
      q.param = master(p.offset);
      q.shadow = shadow(p.offset);
      q.grad = grads_.as<float>() + rest_index(p.offset);
      q.qm = q8_m_.as<std::uint8_t>() + (p.offset - q8_begin_);
      q.qv = q8_v_.as<std::uint8_t>() + (p.offset - q8_begin_);
      const std::size_t blk = q8_block_base(params_info_, q8_begin_, small_begin_, id);
      q.absmax_m = q8_absmax_m_.as<float>() + blk;
      q.absmax_v = q8_absmax_v_.as<float>() + blk;
      q.numel = p.numel;
      q.weight_decay_scale = scale_of(p).weight_decay;
      q.lr_scale = scale_of(p).lr;
      slots8.push_back(q);
      continue;
    }
    DeviceParamSlot slot;
    slot.param = master(p.offset);
    slot.shadow = shadow(p.offset);
    slot.grad = grads_.as<float>() + rest_index(p.offset);
    slot.exp_avg = exp_avg_.as<float>() + state32_index(p.offset);
    slot.exp_avg_sq = exp_avg_sq_.as<float>() + state32_index(p.offset);
    slot.numel = p.numel;
    slot.weight_decay_scale = scale_of(p).weight_decay;
    slot.lr_scale = scale_of(p).lr;
    slots32.push_back(slot);
  }
  DeviceAdamWConfig c = fused_config_;
  c.sr_seed = 0x5EEDull;
  adamw_fixed_clip_update(slots32.data(), static_cast<int>(slots32.size()), slots8.data(), static_cast<int>(slots8.size()), c,
                          step_, fused_clip_.as<float>(), fused_norm_accum_.as<double>(), fused_scratch_.data(),
                          stream_.handle());
  // Bu adimin global normu -> sonraki adimin kirpmasi (gecikmeli kirpma).
  delayed_clip_finish(fused_clip_.as<float>(), grad_norm_.as<float>(), fused_norm_accum_.as<double>(),
                      fused_config_.max_grad_norm, stream_.handle());
}

void GptModel::optimizer_step(const DeviceAdamWConfig& base) {
  if (fused()) {
    // Guncelleme forward_backward icinde yapildi; sonraki adimlar icin hiperparametreler.
    set_fused_optimizer(base);
    return;
  }
  step_ += 1;
  if (offload()) {
    cpu_optimizer_step(base);
    return;
  }
  DeviceAdamWConfig config = base;
  auto* scale_state = loss_scale_state_.as<DeviceLossScaleState>();
  if (use_loss_scale_) {
    config.loss_scale = &scale_state->scale;
    config.found_inf = &scale_state->found_inf;
    config.skipped_steps = &scale_state->skipped_steps;
  } else if (guard_) {
    config.found_inf = &scale_state->found_inf;  // olcek 1 kalir; yalniz bayrak ve sayaclar
    config.skipped_steps = &scale_state->skipped_steps;
  }
  const bool write_shadow = narrow() && config_.master != MasterWeights::CastPerForward;
  config.shadow_type = write_shadow ? config_.compute_dtype : DType::F32;
  config.stochastic_rounding = narrow() && config_.master == MasterWeights::NoMaster;
  config.sr_seed = 0x5EEDull;
  if (config_.optimizer == OptimizerKind::AdamW8bit) {
    std::vector<DeviceParamSlot> slots32;
    std::vector<Device8bitSlot> slots8;
    std::size_t block = 0;
    for (const GptParamInfo& p : params_info_) {
      if (p.offset >= q8_begin_ && p.offset < small_begin_) {
        Device8bitSlot q;
        q.param = master(p.offset);
        q.shadow = write_shadow ? shadow(p.offset) : nullptr;
        q.grad = grads_.as<float>() + p.offset;
        q.qm = q8_m_.as<std::uint8_t>() + (p.offset - q8_begin_);
        q.qv = q8_v_.as<std::uint8_t>() + (p.offset - q8_begin_);
        q.absmax_m = q8_absmax_m_.as<float>() + block;
        q.absmax_v = q8_absmax_v_.as<float>() + block;
        q.numel = p.numel;
        q.weight_decay_scale = scale_of(p).weight_decay;
        q.lr_scale = scale_of(p).lr;
        block += (p.numel + kQ8Block - 1) / kQ8Block;
        slots8.push_back(q);
      } else {
        DeviceParamSlot slot;
        slot.param = master(p.offset);
        slot.shadow = write_shadow ? shadow(p.offset) : nullptr;
        slot.grad = grads_.as<float>() + p.offset;
        slot.exp_avg = exp_avg_.as<float>() + state32_index(p.offset);
        slot.exp_avg_sq = exp_avg_sq_.as<float>() + state32_index(p.offset);
        slot.numel = p.numel;
        slot.weight_decay_scale = scale_of(p).weight_decay;
        slot.lr_scale = scale_of(p).lr;
        slots32.push_back(slot);
      }
    }
    adamw_mixed_multi_tensor(slots32.data(), static_cast<int>(slots32.size()), slots8.data(),
                             static_cast<int>(slots8.size()), config, step_, grad_norm_.as<float>(),
                             buffers_->adam_scratch.data(), stream_.handle());
    if (use_loss_scale_) {
      loss_scale_update(scale_state, config_.loss_scale, stream_.handle());
    } else if (guard_) {
      nonfinite_guard_update(scale_state, stream_.handle());
    }
    return;
  }
  std::vector<DeviceParamSlot> slots;
  slots.reserve(params_info_.size());
  for (const GptParamInfo& p : params_info_) {
    DeviceParamSlot slot;
    slot.param = master(p.offset);
    slot.shadow = write_shadow ? shadow(p.offset) : nullptr;
    slot.grad = grads_.as<float>() + p.offset;
    slot.exp_avg = exp_avg_.as<float>() + p.offset;
    slot.exp_avg_sq = exp_avg_sq_.as<float>() + p.offset;
    slot.numel = p.numel;
    slot.weight_decay_scale = scale_of(p).weight_decay;
    slot.lr_scale = scale_of(p).lr;
    slots.push_back(slot);
  }
  adamw_multi_tensor(slots.data(), static_cast<int>(slots.size()), config, step_, grad_norm_.as<float>(),
                     buffers_->adam_scratch.data(), stream_.handle());
  if (use_loss_scale_) {
    loss_scale_update(scale_state, config_.loss_scale, stream_.handle());
  } else if (guard_) {
    nonfinite_guard_update(scale_state, stream_.handle());
  }
}

namespace {

// [0, count) gorevlerini n is parcacigina dagitir (adim basina bir kez: thread acma maliyeti saniyelik
// CPU guncellemesi yaninda ihmal edilebilir).
void parallel_tasks(int count, int threads, const std::function<void(int)>& fn) {
  threads = std::max(1, std::min(threads, count));
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      for (int i = next.fetch_add(1); i < count; i = next.fetch_add(1)) {
        fn(i);
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }
}

float grad_at(const void* base, DType t, std::size_t i) {
  return t == DType::F32 ? static_cast<const float*>(base)[i]
                         : bfloat16_bits_to_float(static_cast<const std::uint16_t*>(base)[i]);
}

}  // namespace

// ZeroOffload: gradyanlar pinned host'ta. Global norm (FP64) -> kirpma -> parca parca CPU AdamW
// (fused::adamw_multi_tensor, GPU cekirdegiyle ayni islem sirasi) -> BF16 RN golge -> H2D.
void GptModel::cpu_optimizer_step(const DeviceAdamWConfig& base) {
  Offload& o = *offload_;
  const auto start = std::chrono::steady_clock::now();
  o.copy.synchronize();  // tum D2H tamam
  const DType gdt = config_.offload_grad_dtype;
  const int threads = config_.offload_threads > 0 ? config_.offload_threads
                                                  : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  constexpr std::size_t kTask = 1u << 20;
  struct Task {
    std::size_t begin, length;
    bool decay;
  };
  std::vector<Task> tasks;
  for (const GptParamInfo& p : params_info_) {
    for (std::size_t i = 0; i < p.numel; i += kTask) {
      tasks.push_back({p.offset + i, std::min(kTask, p.numel - i), p.decay});
    }
  }
  std::vector<double> partial(tasks.size(), 0.0);
  parallel_tasks(static_cast<int>(tasks.size()), threads, [&](int t) {
    double acc = 0.0;
    const Task& task = tasks[static_cast<std::size_t>(t)];
    for (std::size_t i = task.begin; i < task.begin + task.length; ++i) {
      const double g = grad_at(o.pinned_grad, gdt, i);
      acc += g * g;
    }
    partial[static_cast<std::size_t>(t)] = acc;
  });
  double sq = 0.0;
  for (const double x : partial) {
    sq += x;
  }
  const double norm = std::sqrt(sq);
  const float clip = base.max_grad_norm > 0.0 ? static_cast<float>(std::min(1.0, base.max_grad_norm / (norm + 1.0e-6))) : 1.0f;
  parallel_tasks(static_cast<int>(tasks.size()), threads, [&](int t) {
    const Task& task = tasks[static_cast<std::size_t>(t)];
    std::vector<float> g(task.length);
    for (std::size_t i = 0; i < task.length; ++i) {
      g[i] = grad_at(o.pinned_grad, gdt, task.begin + i) * clip;  // GPU: __fmul_rn(grad, clip)
    }
    fused::ParamSlot slot{o.master.data() + task.begin, g.data(), o.m.data() + task.begin, o.v.data() + task.begin,
                          task.length};
    fused::AdamWConfig c;
    c.lr = base.lr;
    c.beta1 = base.beta1;
    c.beta2 = base.beta2;
    c.eps = base.eps;
    c.weight_decay = task.decay ? base.weight_decay : 0.0;
    c.max_grad_norm = 0.0;
    (void)fused::adamw_multi_tensor(std::span<const fused::ParamSlot>(&slot, 1), c, step_);
    for (std::size_t i = 0; i < task.length; ++i) {
      o.pinned_w[task.begin + i] = float_to_bfloat16_bits(o.master[task.begin + i]);
    }
  });
  last_cpu_update_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const float norm_f = static_cast<float>(norm);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(grad_norm_.data(), &norm_f, sizeof(float), cudaMemcpyHostToDevice, stream_.handle()));
  sync_device_from_host_master(false);
}

// Host FP32 ana kopyadan aygit: BF16 golge (pinned_w'den) + wte/wpe/kucuk FP32 kopya.
// pinned_w guncel olmali (cpu_optimizer_step yazar; kurulum/yuklemede burada doldurulur).
void GptModel::sync_device_from_host_master(bool rebuild_bf16) {
  Offload& o = *offload_;
  cudaStream_t s = stream_.handle();
  if (rebuild_bf16) {
    const int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const std::size_t n = num_params_;
    const int parts = threads * 4;
    parallel_tasks(parts, threads, [&](int t) {
      const std::size_t a = n * static_cast<std::size_t>(t) / parts;
      const std::size_t e = n * static_cast<std::size_t>(t + 1) / parts;
      for (std::size_t i = a; i < e; ++i) {
        o.pinned_w[i] = float_to_bfloat16_bits(o.master[i]);
      }
    });
  }
  // Onceki kopyalarin bitmesini bekle (pinned_w, pinned_grad ile ayni olabilir).
  o.copy.synchronize();
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(shadow_.data(), o.pinned_w, num_params_ * 2, cudaMemcpyHostToDevice, s));
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(params_.data(), o.master.data(), matrices_begin_ * 4, cudaMemcpyHostToDevice, s));
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(params_.as<float>() + matrices_begin_, o.master.data() + small_begin_,
                                   (num_params_ - small_begin_) * 4, cudaMemcpyHostToDevice, s));
  // H2D bitmeden pinned_w (= pinned_grad) uzerine yeni gradyan yazilmasin: kopya akisi bekler.
  o.grads_ready.record(stream_);
  o.grads_ready.wait(o.copy);
}

std::vector<float> GptModel::download_parameters() const {
  std::vector<float> host(num_params_, 0.0f);
  cudaStream_t s = stream_.handle();
  if (offload()) {
    stream_.synchronize();
    return offload_->master;
  }
  if (master_begin_ > 0) {
    // Ana kopyasiz matrisler: dar golgeden FP32'ye, sabit boyutlu parcalarla (1.3B'de 4.8 GB'lik
    // gecici tampon ayiriciya rezerve olarak kalip WDDM tasmasina yol aciyordu).
    const std::size_t chunk = std::min<std::size_t>(master_begin_, kHostCastChunk);
    DeviceBuffer tmp(chunk * sizeof(float), stream_);
    for (std::size_t i = 0; i < master_begin_; i += chunk) {
      const std::size_t n = std::min(chunk, master_begin_ - i);
      cast(tmp.data(), DType::F32, shadow(i), config_.compute_dtype, n, s);
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data() + i, tmp.data(), n * 4, cudaMemcpyDeviceToHost, s));
      stream_.synchronize();
    }
  }
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data() + master_begin_, params_.data(), (num_params_ - master_begin_) * 4,
                                   cudaMemcpyDeviceToHost, s));
  stream_.synchronize();
  return host;
}

void GptModel::upload_parameters(const std::vector<float>& values) {
  if (values.size() != num_params_) {
    throw TensorError("Nexus GPT: parametre sayisi uyusmuyor");
  }
  cudaStream_t s = stream_.handle();
  if (offload()) {
    stream_.synchronize();
    offload_->master = values;
    sync_device_from_host_master(true);
    stream_.synchronize();
    return;
  }
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(params_.data(), values.data() + master_begin_, (num_params_ - master_begin_) * 4,
                                   cudaMemcpyHostToDevice, s));
  if (narrow()) {
    if (master_begin_ > 0) {
      const std::size_t chunk = std::min<std::size_t>(master_begin_, kHostCastChunk);
      DeviceBuffer tmp(chunk * sizeof(float), stream_);
      for (std::size_t i = 0; i < master_begin_; i += chunk) {
        const std::size_t n = std::min(chunk, master_begin_ - i);
        NEXUS_CUDA_CHECK(cudaMemcpyAsync(tmp.data(), values.data() + i, n * 4, cudaMemcpyHostToDevice, s));
        cast(shadow(i), config_.compute_dtype, tmp.data(), DType::F32, n, s);
        stream_.synchronize();
      }
    }
    refresh_shadow_from_master();
  }
  stream_.synchronize();
}

std::vector<float> GptModel::download_gradients() const {
  std::vector<float> host(num_params_, 0.0f);
  if (offload()) {
    stream_.synchronize();
    offload_->copy.synchronize();
    for (const GptParamInfo& p : params_info_) {
      for (std::size_t i = p.offset; i < p.offset + p.numel; ++i) {
        host[i] = grad_at(offload_->pinned_grad, config_.offload_grad_dtype, i);
      }
    }
    return host;
  }
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data(), grads_.data(), num_params_ * 4, cudaMemcpyDeviceToHost, stream_.handle()));
  stream_.synchronize();
  return host;
}

float GptModel::read_loss() const {
  float value = 0.0f;
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(&value, loss_.data(), sizeof(float), cudaMemcpyDeviceToHost, stream_.handle()));
  stream_.synchronize();
  return value;
}

DeviceLossScaleState GptModel::read_loss_scale() const {
  DeviceLossScaleState host{};
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(&host, loss_scale_state_.data(), sizeof(host), cudaMemcpyDeviceToHost,
                                   stream_.handle()));
  stream_.synchronize();
  return host;
}

// ================================ Egitmen =========================================

GptTrainer::GptTrainer(GptModel& model, DeviceAdamWConfig optimizer, int loss_read_interval)
    : model_(&model), optimizer_(optimizer), interval_(loss_read_interval < 1 ? 1 : loss_read_interval) {
  const std::size_t count = static_cast<std::size_t>(model.batch()) * model.time();
  for (int i = 0; i < 2; ++i) {
    pinned_tokens_[i] = static_cast<std::int32_t*>(pinned_allocator().allocate(count * sizeof(std::int32_t)));
    device_tokens_[i] = DeviceBuffer(count * sizeof(std::int32_t), model.stream());
  }
  pinned_loss_ = static_cast<float*>(pinned_allocator().allocate(sizeof(float)));
}

GptTrainer::~GptTrainer() {
  try {
    synchronize();
  } catch (...) {
  }
  for (std::int32_t* p : pinned_tokens_) {
    pinned_allocator().deallocate(p);
  }
  pinned_allocator().deallocate(pinned_loss_);
}

float GptTrainer::step(const std::int32_t* tokens) {
  const std::size_t count = static_cast<std::size_t>(model_->batch()) * model_->time();
  const Stream& stream = model_->stream();
  // Bu pinned tamponun bir onceki kopyasi bitmeden ustune yazilmaz (cift tampon).
  copy_done_[slot_].synchronize();
  std::memcpy(pinned_tokens_[slot_], tokens, count * sizeof(std::int32_t));
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(device_tokens_[slot_].data(), pinned_tokens_[slot_], count * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream.handle()));
  copy_done_[slot_].record(stream);
  const std::int32_t* d_tokens = device_tokens_[slot_].as<std::int32_t>();
  slot_ ^= 1;
  return step_device(d_tokens);
}

float GptTrainer::step_device(const std::int32_t* d_tokens) {
  const Stream& stream = model_->stream();
  model_->forward_backward(d_tokens);
  model_->optimizer_step(optimizer_);
  steps_ += 1;
  if (steps_ % interval_ == 0) {
    if (loss_pending_) {
      loss_ready_.synchronize();
      last_loss_ = *pinned_loss_;
      has_loss_ = true;
    }
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(pinned_loss_, model_->device_loss(), sizeof(float), cudaMemcpyDeviceToHost,
                                     stream.handle()));
    loss_ready_.record(stream);
    loss_pending_ = true;
  }
  return has_loss_ ? last_loss_ : NAN;
}

float GptTrainer::synchronize() {
  model_->stream().synchronize();
  if (loss_pending_) {
    loss_ready_.synchronize();
    last_loss_ = *pinned_loss_;
    has_loss_ = true;
    loss_pending_ = false;
  }
  return last_loss_;
}

}  // namespace cuda
}  // namespace nexus
