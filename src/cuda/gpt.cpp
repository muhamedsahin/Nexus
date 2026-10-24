/**
 * @file gpt.cpp
 * @brief FAZ 3/4 GPU GPT-2 egitim yolu: FP32/TF32 ve BF16/F16 karisik hassasiyet,
 *        ileri, geri, foreach AdamW (ana kopya modlari), dinamik kayip olcegi, egitmen.
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
#include "nexus/cuda/gemm.hpp"
#include "nexus/prof/memory_ledger.hpp"

#include <cmath>
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
        const void* bias = nullptr, void* aux = nullptr, void* bias_grad = nullptr) {
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

struct GptModel::Buffers {
  // Katman basina saklanan aktivasyonlar (geri gecis icin). (t) = compute_dtype, (f) = FP32.
  DeviceBuffer res;                      // (f) [L+1, BT, C]
  DeviceBuffer ln1, ln1_mean, ln1_rstd;  // (t) [L, BT, C], (f) [L, BT]
  DeviceBuffer qkv;                      // (t) [L, BT, 3C]
  DeviceBuffer q, k, v;                  // (t) [L, B, H, T, D]       yalniz math
  DeviceBuffer probs;                    // (t) [L, B, H, T, T]       yalniz math (hesap FP32)
  DeviceBuffer atty;                     // (t) [L, B, H, T, D]       yalniz math
  DeviceBuffer lse;                      // (f) [L, B, H, T]          yalniz flash
  DeviceBuffer att;                      // (t) [L, BT, C]
  DeviceBuffer xmid;                     // (f) [L, BT, C]
  DeviceBuffer ln2, ln2_mean, ln2_rstd;  // (t) / (f)
  DeviceBuffer fc_pre, fc_act;           // (t) [L, BT, 4C]
  DeviceBuffer lnf, lnf_mean, lnf_rstd;  // (t) / (f)
  DeviceBuffer logits;                   // (f) [BT, V]  (geri geciste yerinde dlogits)
  // Gecici
  DeviceBuffer scores;         // (t) [B, H, T, T]  yalniz math
  DeviceBuffer flash_scratch;  // yalniz flash: D vektoru + dQ FP32 biriktirici
  DeviceBuffer proj;    // (f) [BT, C]
  // Geri gecis gecicileri
  DeviceBuffer dres, dln;                                      // (f)
  DeviceBuffer dres_t, dlogits_t;                              // (t) yalniz dar modda
  DeviceBuffer dqkv, dq, dk, dv, datty, datt, dfc;              // (t)
  DeviceBuffer norm_scratch, emb_scratch, ce_scratch, adam_scratch;
};

float* GptModel::master(std::size_t offset) const {
  if (offset < master_begin_ || params_.data() == nullptr) {
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
  // F16'nin dar us araligi gradyani sifira dusurur: dinamik kayip olcegi zorunlu.
  use_loss_scale_ = config_.dynamic_loss_scale || ct == DType::F16;
  build_layout();
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
  const bool no_master = narrow() && config_.master == MasterWeights::NoMaster;
  master_begin_ = no_master ? small_begin_ : 0;
  const std::size_t master_count = num_params_ - master_begin_;
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::Parameter);
    params_ = DeviceBuffer(master_count * sizeof(float), stream_);
    if (narrow()) {
      shadow_ = DeviceBuffer(num_params_ * size_of(ct), stream_);
    }
  }
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::Gradient);
    grads_ = DeviceBuffer(num_params_ * sizeof(float), stream_);
  }
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::OptimizerState);
    exp_avg_ = DeviceBuffer(num_params_ * sizeof(float), stream_);
    exp_avg_sq_ = DeviceBuffer(num_params_ * sizeof(float), stream_);
  }
  state_bytes_ = master_count * 4 + (narrow() ? num_params_ * size_of(ct) : 0) + num_params_ * 12;
  cudaStream_t s = stream_.handle();
  NEXUS_CUDA_CHECK(cudaMemsetAsync(exp_avg_.data(), 0, num_params_ * 4, s));
  NEXUS_CUDA_CHECK(cudaMemsetAsync(exp_avg_sq_.data(), 0, num_params_ * 4, s));
  NEXUS_CUDA_CHECK(cudaMemsetAsync(grads_.data(), 0, num_params_ * 4, s));
  loss_ = DeviceBuffer(sizeof(float), stream_);
  grad_norm_ = DeviceBuffer(sizeof(float), stream_);
  loss_scale_state_ = DeviceBuffer(sizeof(DeviceLossScaleState), stream_);
  nexus_train::LossScalePolicy policy = config_.loss_scale;
  if (!use_loss_scale_) {
    policy.init_scale = 1.0;
  }
  loss_scale_init(loss_scale_state_.as<DeviceLossScaleState>(), policy, s);

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
  upload_parameters(host);

  // Aktivasyonlar.
  prof::AllocationLabelScope label(prof::MemoryLabel::Activation);
  buffers_ = std::make_unique<Buffers>();
  Buffers& b = *buffers_;
  const std::size_t L = static_cast<std::size_t>(config_.layers);
  const std::size_t BT = static_cast<std::size_t>(batch) * time;
  const std::size_t C = static_cast<std::size_t>(config_.dim);
  const std::size_t H = static_cast<std::size_t>(config_.heads);
  const std::size_t T = static_cast<std::size_t>(time);
  const std::size_t V = static_cast<std::size_t>(config_.vocab);
  const std::size_t BHTT = static_cast<std::size_t>(batch) * H * T * T;
  const std::size_t et = size_of(ct);
  std::size_t total = 0;
  auto alloc = [&](DeviceBuffer& buf, std::size_t count, std::size_t elem) {
    buf = DeviceBuffer(count * elem, stream_);
    total += count * elem;
  };
  alloc(b.res, (L + 1) * BT * C, 4);
  alloc(b.ln1, L * BT * C, et);
  alloc(b.ln1_mean, L * BT, 4);
  alloc(b.ln1_rstd, L * BT, 4);
  alloc(b.qkv, L * BT * 3 * C, et);
  if (flash_) {
    alloc(b.lse, L * BT * H, 4);
  } else {
    alloc(b.q, L * BT * C, et);
    alloc(b.k, L * BT * C, et);
    alloc(b.v, L * BT * C, et);
    alloc(b.probs, L * BHTT, et);
    alloc(b.atty, L * BT * C, et);
  }
  alloc(b.att, L * BT * C, et);
  alloc(b.xmid, L * BT * C, 4);
  alloc(b.ln2, L * BT * C, et);
  alloc(b.ln2_mean, L * BT, 4);
  alloc(b.ln2_rstd, L * BT, 4);
  alloc(b.fc_pre, L * BT * 4 * C, et);
  alloc(b.fc_act, L * BT * 4 * C, et);
  alloc(b.lnf, BT * C, et);
  alloc(b.lnf_mean, BT, 4);
  alloc(b.lnf_rstd, BT, 4);
  alloc(b.logits, BT * V, 4);
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
    alloc(b.dlogits_t, BT * V, et);
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
  b.emb_scratch = DeviceBuffer(embedding_backward_scratch_bytes(batch, time), stream_);
  b.ce_scratch = DeviceBuffer(cross_entropy_scratch_bytes(batch, time), stream_);
  b.adam_scratch = DeviceBuffer(adamw_scratch_bytes(kParamCount, num_params_), stream_);
  activation_bytes_ = total;
}

GptModel::~GptModel() = default;

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

void GptModel::run_forward(const std::int32_t* d_tokens, bool for_backward) {
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
  // Autocast tablosu: matmul dar tipte, softmax/LN/CE FP32 hesap.
  const DeviceKind cuda_kind = DeviceKind::Cuda;
  amp::AutocastGuard autocast(cuda_kind, narrow() ? config_.compute_dtype : DType::BF16, narrow());
  const DType mt = amp::compute_dtype(cuda_kind, amp::OpClass::Linear, {DType::F32});
  if (amp::compute_dtype(cuda_kind, amp::OpClass::LayerNorm, {mt}) != DType::F32 ||
      amp::compute_dtype(cuda_kind, amp::OpClass::Softmax, {mt}) != DType::F32) {
    throw TensorError("Nexus GPT: autocast tablosu LN/softmax'i FP32 disina dusurdu");
  }
  if (narrow() && config_.master == MasterWeights::CastPerForward) {
    refresh_shadow_from_master();  // A modu: her ileri geciste FP32 -> dar
  }
  const auto& info = params_info_;
  auto off = [&](int id) { return info[static_cast<std::size_t>(id)].offset; };
  auto w = [&](int id, std::size_t extra = 0) { return Mat{weight(off(id) + extra), mt}; };
  auto f32 = [&](int id, std::size_t extra = 0) { return static_cast<const float*>(master(off(id) + extra)); };
  auto act = [&](const DeviceBuffer& buf, std::size_t elements) { return at(buf, elements, mt); };

  float* res = b.res.as<float>();
  if (master(off(kWte)) != nullptr) {
    embedding_pos_forward(res, d_tokens, f32(kWte), f32(kWpe), B, T, C, s);
  } else {
    embedding_pos_forward(res, d_tokens, shadow(off(kWte)), shadow(off(kWpe)), mt, B, T, C, s);
  }
  layernorm_forward(b.ln1.data(), mt, b.ln1_mean.as<float>(), b.ln1_rstd.as<float>(), res, f32(kLn1W), f32(kLn1B), BT,
                    C, eps, s);
  for (int l = 0; l < L; ++l) {
    const std::size_t lc = static_cast<std::size_t>(l) * C;
    const std::size_t lt = static_cast<std::size_t>(l) * btc;
    float* x_in = res + lt;
    float* x_out = res + static_cast<std::size_t>(l + 1) * btc;
    void* ln1 = act(b.ln1, lt);
    void* qkv = act(b.qkv, 3 * lt);
    void* att = act(b.att, lt);
    float* xmid = b.xmid.as<float>() + lt;
    void* ln2 = act(b.ln2, lt);
    void* fc_pre = act(b.fc_pre, 4 * lt);
    void* fc_act = act(b.fc_act, 4 * lt);

    // QKV = ln1 W + b  (dar cikti, dar bias golgesi)
    mm(s, false, false, BT, 3 * C, C, Mat{ln1, mt}, w(kQkvW, lc * 3 * C), OutBuf{qkv, mt}, Epilogue::Bias,
       weight(off(kQkvB) + lc * 3));
    if (flash_) {
      // FlashAttention-2: Q/K/V qkv'den adimla okunur, O dogrudan att'a ([B,T,C]) yazilir.
      const AttentionParams ap = gpt_attention_params(B, T, H, D, mt, config_.deterministic_attention);
      flash_attention_forward(att, b.lse.as<float>() + static_cast<std::size_t>(l) * BT * H, qkv,
                              at(b.qkv, 3 * lt + C, mt), at(b.qkv, 3 * lt + 2 * C, mt), ap, s);
    } else {
      void* q = act(b.q, lt);
      void* k = act(b.k, lt);
      void* v = act(b.v, lt);
      void* probs = act(b.probs, static_cast<std::size_t>(l) * bhtt);
      void* atty = act(b.atty, lt);
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
    // MLP: GELU on-aktivasyonu aux olarak saklanir (geri geciste DGELU epilogu okur).
    mm(s, false, false, BT, 4 * C, C, Mat{ln2, mt}, w(kFcW, lc * 4 * C), OutBuf{fc_act, mt}, Epilogue::BiasGelu,
       weight(off(kFcB) + lc * 4), fc_pre);
    mm(s, false, false, BT, C, 4 * C, Mat{fc_act, mt}, w(kFcProjW, lc * 4 * C), OutBuf{b.proj.data(), DType::F32},
       Epilogue::Bias, f32(kFcProjB, lc));
    if (l + 1 < L) {
      const std::size_t next = static_cast<std::size_t>(l + 1);
      residual_layernorm_forward(x_out, act(b.ln1, next * btc), mt, b.ln1_mean.as<float>() + next * BT,
                                 b.ln1_rstd.as<float>() + next * BT, xmid, b.proj.as<float>(), f32(kLn1W, next * C),
                                 f32(kLn1B, next * C), BT, C, eps, s);
    } else {
      residual_layernorm_forward(x_out, b.lnf.data(), mt, b.lnf_mean.as<float>(), b.lnf_rstd.as<float>(), xmid,
                                 b.proj.as<float>(), f32(kLnfW), f32(kLnfB), BT, C, eps, s);
    }
  }
  // Bagli kafa: logits = lnf wte^T, FP32'ye yukseltilmis cikti (4.6) -> CE FP32.
  mm(s, false, true, BT, V, C, Mat{b.lnf.data(), mt}, w(kWte), OutBuf{b.logits.data(), DType::F32});
  CrossEntropyDeviceOptions options;
  const float* device_scale = use_loss_scale_ ? &loss_scale_state_.as<DeviceLossScaleState>()->scale : nullptr;
  cross_entropy_shifted(for_backward ? b.logits.as<float>() : nullptr, loss_.as<float>(), nullptr,
                        b.logits.as<float>(), d_tokens, B, T, V, options, 1.0f, device_scale, b.ce_scratch.data(), s);
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
  const DType mt = config_.compute_dtype;
  float* G = grads_.as<float>();
  const auto& info = params_info_;
  auto off = [&](int id) { return info[static_cast<std::size_t>(id)].offset; };
  auto w = [&](int id, std::size_t extra = 0) { return Mat{weight(off(id) + extra), mt}; };
  auto f32 = [&](int id, std::size_t extra = 0) { return static_cast<const float*>(master(off(id) + extra)); };
  auto grad = [&](int id, std::size_t extra = 0) { return G + off(id) + extra; };
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

  // Kafa: dlogits FP32 (CE yerinde yazdi); dar modda GEMM icin dar kopya.
  Mat dlogits{b.logits.data(), DType::F32};
  if (narrow()) {
    cast(b.dlogits_t.data(), mt, b.logits.data(), DType::F32, static_cast<std::size_t>(BT) * V, s);
    dlogits = Mat{b.dlogits_t.data(), mt};
  }
  mm(s, false, false, BT, C, V, dlogits, w(kWte), OutBuf{dln, DType::F32});              // dlnf = dlogits wte
  mm(s, true, false, V, C, BT, dlogits, Mat{b.lnf.data(), mt}, gout(kWte));             // dwte (ilk yazim, FP32)
  float* res = b.res.as<float>();
  layernorm_backward(dres, grad(kLnfW), grad(kLnfB), dln, res + static_cast<std::size_t>(L) * btc, f32(kLnfW),
                     b.lnf_mean.as<float>(), b.lnf_rstd.as<float>(), BT, C, false, scratch, s);

  for (int l = L - 1; l >= 0; --l) {
    const std::size_t lc = static_cast<std::size_t>(l) * C;
    const std::size_t lt = static_cast<std::size_t>(l) * btc;
    float* x_in = res + lt;
    void* ln1 = act(b.ln1, lt);
    void* att = act(b.att, lt);
    float* xmid = b.xmid.as<float>() + lt;
    void* ln2 = act(b.ln2, lt);
    void* fc_pre = act(b.fc_pre, 4 * lt);
    void* fc_act = act(b.fc_act, 4 * lt);

    // --- MLP --- dres = d x_out
    Mat dres_in = dres_as_input();
    mm(s, false, true, BT, 4 * C, C, dres_in, w(kFcProjW, lc * 4 * C), OutBuf{b.dfc.data(), mt}, Epilogue::DGelu,
       nullptr, fc_pre);
    mm(s, true, false, 4 * C, C, BT, Mat{fc_act, mt}, dres_in, gout(kFcProjW, lc * 4 * C), Epilogue::BGradB, nullptr,
       nullptr, grad(kFcProjB, lc));
    mm(s, false, true, BT, C, 4 * C, Mat{b.dfc.data(), mt}, w(kFcW, lc * 4 * C), OutBuf{dln, DType::F32});
    mm(s, true, false, C, 4 * C, BT, Mat{ln2, mt}, Mat{b.dfc.data(), mt}, gout(kFcW, lc * 4 * C), Epilogue::BGradB,
       nullptr, nullptr, grad(kFcB, lc * 4));
    layernorm_backward(dres, grad(kLn2W, lc), grad(kLn2B, lc), dln, xmid, f32(kLn2W, lc),
                       b.ln2_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                       b.ln2_rstd.as<float>() + static_cast<std::size_t>(l) * BT, BT, C, true, scratch, s);
    // --- Dikkat --- dres = d xmid
    dres_in = dres_as_input();
    mm(s, false, true, BT, C, C, dres_in, w(kAttProjW, lc * C), OutBuf{b.datt.data(), mt});
    mm(s, true, false, C, C, BT, Mat{att, mt}, dres_in, gout(kAttProjW, lc * C), Epilogue::BGradB, nullptr, nullptr,
       grad(kAttProjB, lc));
    if (flash_) {
      // dO = datt ([B,T,C]); dQ/dK/dV dogrudan dqkv'nin [B,T,3,H,D] dilimlerine yazilir.
      const AttentionParams ap = gpt_attention_params(B, T, H, D, mt, config_.deterministic_attention);
      flash_attention_backward(b.dqkv.data(), at(b.dqkv, C, mt), at(b.dqkv, 2 * C, mt), b.datt.data(),
                               act(b.qkv, 3 * lt), at(b.qkv, 3 * lt + C, mt), at(b.qkv, 3 * lt + 2 * C, mt), att,
                               b.lse.as<float>() + static_cast<std::size_t>(l) * BT * H, ap, b.flash_scratch.data(), s);
    } else {
      void* q = act(b.q, lt);
      void* k = act(b.k, lt);
      void* v = act(b.v, lt);
      void* probs = act(b.probs, static_cast<std::size_t>(l) * bhtt);
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
    mm(s, true, false, C, 3 * C, BT, Mat{ln1, mt}, Mat{b.dqkv.data(), mt}, gout(kQkvW, lc * 3 * C), Epilogue::BGradB,
       nullptr, nullptr, grad(kQkvB, static_cast<std::size_t>(l) * 3 * C));
    layernorm_backward(dres, grad(kLn1W, lc), grad(kLn1B, lc), dln, x_in, f32(kLn1W, lc),
                       b.ln1_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                       b.ln1_rstd.as<float>() + static_cast<std::size_t>(l) * BT, BT, C, true, scratch, s);
  }
  // Gomme: dwte'ye (kafanin yazdigi) EKLER; dwpe sifirdan. FP32.
  if (config_.deterministic) {
    embedding_backward_deterministic(grad(kWte), grad(kWpe), dres, d_tokens, B, T, C, V, -1, b.emb_scratch.data(), s);
  } else {
    embedding_backward_atomic(grad(kWte), grad(kWpe), dres, d_tokens, B, T, C, s);
  }
}

void GptModel::forward(const std::int32_t* d_tokens) { run_forward(d_tokens, false); }

void GptModel::forward_backward(const std::int32_t* d_tokens) {
  // LN agirlik/bias ve wpe gradyanlari "+=" ile yazilir; tumu her adimda sifirlanir.
  NEXUS_CUDA_CHECK(cudaMemsetAsync(grads_.data(), 0, num_params_ * sizeof(float), stream_.handle()));
  run_forward(d_tokens, true);
  run_backward(d_tokens);
}

void GptModel::optimizer_step(const DeviceAdamWConfig& base) {
  step_ += 1;
  DeviceAdamWConfig config = base;
  auto* scale_state = loss_scale_state_.as<DeviceLossScaleState>();
  if (use_loss_scale_) {
    config.loss_scale = &scale_state->scale;
    config.found_inf = &scale_state->found_inf;
    config.skipped_steps = &scale_state->skipped_steps;
  }
  const bool write_shadow = narrow() && config_.master != MasterWeights::CastPerForward;
  config.shadow_type = write_shadow ? config_.compute_dtype : DType::F32;
  config.stochastic_rounding = narrow() && config_.master == MasterWeights::NoMaster;
  config.sr_seed = 0x5EEDull;
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
    slot.weight_decay_scale = p.decay ? 1.0f : 0.0f;
    slots.push_back(slot);
  }
  adamw_multi_tensor(slots.data(), static_cast<int>(slots.size()), config, step_, grad_norm_.as<float>(),
                     buffers_->adam_scratch.data(), stream_.handle());
  if (use_loss_scale_) {
    loss_scale_update(scale_state, config_.loss_scale, stream_.handle());
  }
}

std::vector<float> GptModel::download_parameters() const {
  std::vector<float> host(num_params_, 0.0f);
  cudaStream_t s = stream_.handle();
  if (master_begin_ > 0) {
    // Ana kopyasiz matrisler: dar golgeden FP32'ye.
    DeviceBuffer tmp(master_begin_ * sizeof(float), stream_);
    cast(tmp.data(), DType::F32, shadow(0), config_.compute_dtype, master_begin_, s);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data(), tmp.data(), master_begin_ * 4, cudaMemcpyDeviceToHost, s));
    stream_.synchronize();
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
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(params_.data(), values.data() + master_begin_, (num_params_ - master_begin_) * 4,
                                   cudaMemcpyHostToDevice, s));
  if (narrow()) {
    if (master_begin_ > 0) {
      DeviceBuffer tmp(master_begin_ * sizeof(float), stream_);
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(tmp.data(), values.data(), master_begin_ * 4, cudaMemcpyHostToDevice, s));
      cast(shadow(0), config_.compute_dtype, tmp.data(), DType::F32, master_begin_, s);
      stream_.synchronize();
    }
    refresh_shadow_from_master();
  }
  stream_.synchronize();
}

std::vector<float> GptModel::download_gradients() const {
  std::vector<float> host(num_params_);
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
  model_->forward_backward(device_tokens_[slot_].as<std::int32_t>());
  model_->optimizer_step(optimizer_);
  steps_ += 1;
  slot_ ^= 1;
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
