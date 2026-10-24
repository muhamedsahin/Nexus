/**
 * @file gpt.cpp
 * @brief FAZ 3 GPU GPT-2 egitim yolu: ileri, geri, foreach AdamW, asenkron egitmen.
 *
 * Satir-ana duzen; agirliklar [giris, cikis] (y = x W). Katman l'nin girisi res[l],
 * cikisi res[l+1]; res[0] gomme ciktisidir. add&norm fuzyonu ile residual toplami
 * ve sonraki LayerNorm tek cekirdektir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gpt.hpp"

#include "nexus/core/error.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/prof/memory_ledger.hpp"

#include <cmath>
#include <cstring>
#include <random>

namespace nexus {
namespace cuda {
namespace {

enum Param : int {
  kWte = 0,
  kWpe,
  kLn1W,
  kLn1B,
  kQkvW,
  kQkvB,
  kAttProjW,
  kAttProjB,
  kLn2W,
  kLn2B,
  kFcW,
  kFcB,
  kFcProjW,
  kFcProjB,
  kLnfW,
  kLnfB,
  kParamCount
};

// Satir-ana GEMM kisayolu: C[m,n] = op(A) op(B) (+ epilog).
void mm(cudaStream_t s, bool ta, bool tb, int m, int n, int k, const float* a, const float* b, float* c,
        Epilogue epi = Epilogue::None, const float* bias = nullptr, float* aux = nullptr, float* bias_grad = nullptr,
        float beta = 0.0f) {
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
  g.beta = beta;
  g.epilogue = epi;
  g.bias = bias;
  g.aux = aux;
  g.bias_grad = bias_grad;
  gemm(g, s);
}

// Strided batched: count adet bagimsiz [m,n] = op(A)[m,k] op(B)[k,n].
void bmm(cudaStream_t s, bool ta, bool tb, int count, int m, int n, int k, const float* a, long long sa,
         const float* b, long long sb, float* c, long long sc) {
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
  g.batch = count;
  g.stride_a = sa;
  g.stride_b = sb;
  g.stride_c = sc;
  gemm(g, s);
}

}  // namespace

struct GptModel::Buffers {
  // Katman basina saklanan aktivasyonlar (geri gecis icin).
  DeviceBuffer res;       // [L+1, BT, C]
  DeviceBuffer ln1, ln1_mean, ln1_rstd;  // [L, BT, C], [L, BT]
  DeviceBuffer qkv;       // [L, BT, 3C]
  DeviceBuffer q, k, v;   // [L, B, H, T, D]
  DeviceBuffer probs;     // [L, B, H, T, T]
  DeviceBuffer atty;      // [L, B, H, T, D]
  DeviceBuffer att;       // [L, BT, C]
  DeviceBuffer xmid;      // [L, BT, C]  (dikkat sonrasi residual)
  DeviceBuffer ln2, ln2_mean, ln2_rstd;
  DeviceBuffer fc_pre, fc_act;  // [L, BT, 4C]
  DeviceBuffer lnf, lnf_mean, lnf_rstd;
  DeviceBuffer logits;    // [BT, V]  (geri geciste yerinde dlogits)
  // Gecici
  DeviceBuffer scores;    // [B, H, T, T]
  DeviceBuffer proj;      // [BT, C]
  // Geri gecis gecicileri
  DeviceBuffer dres, dln, dqkv, dq, dk, dv, datty, datt, dfc;
  DeviceBuffer norm_scratch, emb_scratch, ce_scratch, adam_scratch;
};

GptModel::GptModel(const GptConfig& config, int batch, int time, std::uint64_t seed, const Stream& stream)
    : config_(config), batch_(batch), time_(time), stream_(stream) {
  if (batch < 1 || time < 2 || time > config.max_seq || config.dim % config.heads != 0 || config.layers < 1 ||
      config.vocab < 2) {
    throw TensorError("Nexus GPT: gecersiz yapilandirma (time <= max_seq, dim % heads == 0)");
  }
  build_layout();
  const std::size_t bytes = num_params_ * sizeof(float);
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::Parameter);
    params_ = DeviceBuffer(bytes, stream_);
  }
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::Gradient);
    grads_ = DeviceBuffer(bytes, stream_);
  }
  {
    prof::AllocationLabelScope label(prof::MemoryLabel::OptimizerState);
    exp_avg_ = DeviceBuffer(bytes, stream_);
    exp_avg_sq_ = DeviceBuffer(bytes, stream_);
  }
  cudaStream_t s = stream_.handle();
  NEXUS_CUDA_CHECK(cudaMemsetAsync(exp_avg_.data(), 0, bytes, s));
  NEXUS_CUDA_CHECK(cudaMemsetAsync(exp_avg_sq_.data(), 0, bytes, s));
  NEXUS_CUDA_CHECK(cudaMemsetAsync(grads_.data(), 0, bytes, s));
  loss_ = DeviceBuffer(sizeof(float), stream_);
  grad_norm_ = DeviceBuffer(sizeof(float), stream_);

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
  std::size_t total = 0;
  auto alloc = [&](DeviceBuffer& buf, std::size_t count) {
    buf = DeviceBuffer(count * sizeof(float), stream_);
    total += count * sizeof(float);
  };
  alloc(b.res, (L + 1) * BT * C);
  alloc(b.ln1, L * BT * C);
  alloc(b.ln1_mean, L * BT);
  alloc(b.ln1_rstd, L * BT);
  alloc(b.qkv, L * BT * 3 * C);
  alloc(b.q, L * BT * C);
  alloc(b.k, L * BT * C);
  alloc(b.v, L * BT * C);
  alloc(b.probs, L * BHTT);
  alloc(b.atty, L * BT * C);
  alloc(b.att, L * BT * C);
  alloc(b.xmid, L * BT * C);
  alloc(b.ln2, L * BT * C);
  alloc(b.ln2_mean, L * BT);
  alloc(b.ln2_rstd, L * BT);
  alloc(b.fc_pre, L * BT * 4 * C);
  alloc(b.fc_act, L * BT * 4 * C);
  alloc(b.lnf, BT * C);
  alloc(b.lnf_mean, BT);
  alloc(b.lnf_rstd, BT);
  alloc(b.logits, BT * V);
  alloc(b.scores, BHTT);
  alloc(b.proj, BT * C);
  alloc(b.dres, BT * C);
  alloc(b.dln, BT * C);
  alloc(b.dqkv, BT * 3 * C);
  alloc(b.dq, BT * C);
  alloc(b.dk, BT * C);
  alloc(b.dv, BT * C);
  alloc(b.datty, BT * C);
  alloc(b.datt, BT * C);
  alloc(b.dfc, BT * 4 * C);
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
      {"ln1_w", {L, C}, false},        {"ln1_b", {L, C}, false},
      {"qkv_w", {L, C, 3 * C}, true},  {"qkv_b", {L, 3 * C}, false},
      {"attproj_w", {L, C, C}, true},  {"attproj_b", {L, C}, false},
      {"ln2_w", {L, C}, false},        {"ln2_b", {L, C}, false},
      {"fc_w", {L, C, 4 * C}, true},   {"fc_b", {L, 4 * C}, false},
      {"fcproj_w", {L, 4 * C, C}, true}, {"fcproj_b", {L, C}, false},
      {"lnf_w", {C}, false},           {"lnf_b", {C}, false},
  };
  std::size_t offset = 0;
  for (const Spec& spec : specs) {
    GptParamInfo info;
    info.name = spec.name;
    info.shape = spec.shape;
    info.numel = 1;
    for (const int d : spec.shape) {
      info.numel *= static_cast<std::size_t>(d);
    }
    // Her tensör 64 elemana hizalanir (cuBLASLt ve vektor yukleme hizasi).
    offset = (offset + 63) / 64 * 64;
    info.offset = offset;
    info.decay = spec.decay;
    offset += info.numel;
    params_info_.push_back(info);
  }
  num_params_ = offset;
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
  float* P = params_.as<float>();
  auto param = [&](int id) { return P + params_info_[static_cast<std::size_t>(id)].offset; };

  float* res = b.res.as<float>();
  embedding_pos_forward(res, d_tokens, param(kWte), param(kWpe), B, T, C, s);
  layernorm_forward(b.ln1.as<float>(), b.ln1_mean.as<float>(), b.ln1_rstd.as<float>(), res, param(kLn1W),
                    param(kLn1B), BT, C, eps, s);
  for (int l = 0; l < L; ++l) {
    const std::size_t lc = static_cast<std::size_t>(l) * C;
    float* x_in = res + static_cast<std::size_t>(l) * btc;
    float* x_out = res + static_cast<std::size_t>(l + 1) * btc;
    float* ln1 = b.ln1.as<float>() + static_cast<std::size_t>(l) * btc;
    float* qkv = b.qkv.as<float>() + static_cast<std::size_t>(l) * 3 * btc;
    float* q = b.q.as<float>() + static_cast<std::size_t>(l) * btc;
    float* k = b.k.as<float>() + static_cast<std::size_t>(l) * btc;
    float* v = b.v.as<float>() + static_cast<std::size_t>(l) * btc;
    float* probs = b.probs.as<float>() + static_cast<std::size_t>(l) * bhtt;
    float* atty = b.atty.as<float>() + static_cast<std::size_t>(l) * btc;
    float* att = b.att.as<float>() + static_cast<std::size_t>(l) * btc;
    float* xmid = b.xmid.as<float>() + static_cast<std::size_t>(l) * btc;
    float* ln2 = b.ln2.as<float>() + static_cast<std::size_t>(l) * btc;
    float* fc_pre = b.fc_pre.as<float>() + static_cast<std::size_t>(l) * 4 * btc;
    float* fc_act = b.fc_act.as<float>() + static_cast<std::size_t>(l) * 4 * btc;

    // QKV = ln1 W + b
    mm(s, false, false, BT, 3 * C, C, ln1, param(kQkvW) + lc * 3 * C, qkv, Epilogue::Bias,
       param(kQkvB) + static_cast<std::size_t>(l) * 3 * C);
    split_qkv_heads(q, k, v, qkv, B, T, H, D, s);
    // skor = q k^T (B*H adet [T,T]); nedensel softmax cekirdek icinde maskeler.
    bmm(s, false, true, B * H, T, T, D, q, static_cast<long long>(T) * D, k, static_cast<long long>(T) * D,
        b.scores.as<float>(), static_cast<long long>(T) * T);
    softmax_causal_forward(probs, b.scores.as<float>(), B * H, T, scale, s);
    bmm(s, false, false, B * H, T, D, T, probs, static_cast<long long>(T) * T, v, static_cast<long long>(T) * D, atty,
        static_cast<long long>(T) * D);
    merge_heads(att, atty, B, T, H, D, s);
    // proj + residual, ardindan LN2 (add&norm)
    mm(s, false, false, BT, C, C, att, param(kAttProjW) + lc * C, b.proj.as<float>(), Epilogue::Bias,
       param(kAttProjB) + lc);
    residual_layernorm_forward(xmid, ln2, b.ln2_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                               b.ln2_rstd.as<float>() + static_cast<std::size_t>(l) * BT, x_in, b.proj.as<float>(),
                               param(kLn2W) + lc, param(kLn2B) + lc, BT, C, eps, s);
    // MLP: GELU on-aktivasyonu aux olarak saklanir (geri geciste DGELU epilogu okur).
    mm(s, false, false, BT, 4 * C, C, ln2, param(kFcW) + lc * 4 * C, fc_act, Epilogue::BiasGelu,
       param(kFcB) + lc * 4, fc_pre);
    mm(s, false, false, BT, C, 4 * C, fc_act, param(kFcProjW) + lc * 4 * C, b.proj.as<float>(), Epilogue::Bias,
       param(kFcProjB) + lc);
    // x_out = xmid + mlp; sonraki katmanin LN1'i (veya son LN) ayni cekirdekte.
    if (l + 1 < L) {
      const std::size_t next = static_cast<std::size_t>(l + 1);
      residual_layernorm_forward(x_out, b.ln1.as<float>() + next * btc, b.ln1_mean.as<float>() + next * BT,
                                 b.ln1_rstd.as<float>() + next * BT, xmid, b.proj.as<float>(),
                                 param(kLn1W) + next * C, param(kLn1B) + next * C, BT, C, eps, s);
    } else {
      residual_layernorm_forward(x_out, b.lnf.as<float>(), b.lnf_mean.as<float>(), b.lnf_rstd.as<float>(), xmid,
                                 b.proj.as<float>(), param(kLnfW), param(kLnfB), BT, C, eps, s);
    }
  }
  // Bagli kafa: logits = lnf wte^T
  mm(s, false, true, BT, V, C, b.lnf.as<float>(), param(kWte), b.logits.as<float>());
  CrossEntropyDeviceOptions options;
  cross_entropy_shifted(for_backward ? b.logits.as<float>() : nullptr, loss_.as<float>(), nullptr,
                        b.logits.as<float>(), d_tokens, B, T, V, options, 1.0f, b.ce_scratch.data(), s);
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
  float* P = params_.as<float>();
  float* G = grads_.as<float>();
  auto param = [&](int id) { return P + params_info_[static_cast<std::size_t>(id)].offset; };
  auto grad = [&](int id) { return G + params_info_[static_cast<std::size_t>(id)].offset; };
  float* scratch = b.norm_scratch.as<float>();
  float* dres = b.dres.as<float>();
  float* dln = b.dln.as<float>();

  // Kafa: dlogits logits tamponunda (CE yerinde yazdi).
  float* dlogits = b.logits.as<float>();
  mm(s, false, false, BT, C, V, dlogits, param(kWte), dln);              // dlnf = dlogits wte
  mm(s, true, false, V, C, BT, dlogits, b.lnf.as<float>(), grad(kWte));  // dwte = dlogits^T lnf (ilk yazim)
  float* res = b.res.as<float>();
  layernorm_backward(dres, grad(kLnfW), grad(kLnfB), dln, res + static_cast<std::size_t>(L) * btc, param(kLnfW),
                     b.lnf_mean.as<float>(), b.lnf_rstd.as<float>(), BT, C, false, scratch, s);

  for (int l = L - 1; l >= 0; --l) {
    const std::size_t lc = static_cast<std::size_t>(l) * C;
    float* x_in = res + static_cast<std::size_t>(l) * btc;
    float* ln1 = b.ln1.as<float>() + static_cast<std::size_t>(l) * btc;
    float* q = b.q.as<float>() + static_cast<std::size_t>(l) * btc;
    float* k = b.k.as<float>() + static_cast<std::size_t>(l) * btc;
    float* v = b.v.as<float>() + static_cast<std::size_t>(l) * btc;
    float* probs = b.probs.as<float>() + static_cast<std::size_t>(l) * bhtt;
    float* att = b.att.as<float>() + static_cast<std::size_t>(l) * btc;
    float* xmid = b.xmid.as<float>() + static_cast<std::size_t>(l) * btc;
    float* ln2 = b.ln2.as<float>() + static_cast<std::size_t>(l) * btc;
    float* fc_pre = b.fc_pre.as<float>() + static_cast<std::size_t>(l) * 4 * btc;
    float* fc_act = b.fc_act.as<float>() + static_cast<std::size_t>(l) * 4 * btc;

    // --- MLP --- dres = d x_out
    // dfc_pre = (dres Wproj^T) * gelu'(fc_pre)   (DGELU epilogu)
    mm(s, false, true, BT, 4 * C, C, dres, param(kFcProjW) + lc * 4 * C, b.dfc.as<float>(), Epilogue::DGelu, nullptr,
       fc_pre);
    mm(s, true, false, 4 * C, C, BT, fc_act, dres, grad(kFcProjW) + lc * 4 * C, Epilogue::BGradB, nullptr, nullptr,
       grad(kFcProjB) + lc);
    mm(s, false, true, BT, C, 4 * C, b.dfc.as<float>(), param(kFcW) + lc * 4 * C, dln);
    mm(s, true, false, C, 4 * C, BT, ln2, b.dfc.as<float>(), grad(kFcW) + lc * 4 * C, Epilogue::BGradB, nullptr,
       nullptr, grad(kFcB) + lc * 4);
    layernorm_backward(dres, grad(kLn2W) + lc, grad(kLn2B) + lc, dln, xmid, param(kLn2W) + lc,
                       b.ln2_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                       b.ln2_rstd.as<float>() + static_cast<std::size_t>(l) * BT, BT, C, true, scratch, s);
    // --- Dikkat --- dres = d xmid
    mm(s, false, true, BT, C, C, dres, param(kAttProjW) + lc * C, b.datt.as<float>());
    mm(s, true, false, C, C, BT, att, dres, grad(kAttProjW) + lc * C, Epilogue::BGradB, nullptr, nullptr,
       grad(kAttProjB) + lc);
    split_heads(b.datty.as<float>(), b.datt.as<float>(), B, T, H, D, s);
    float* dscores = b.scores.as<float>();
    // dprobs = datty v^T ; dv = probs^T datty
    bmm(s, false, true, B * H, T, T, D, b.datty.as<float>(), static_cast<long long>(T) * D, v,
        static_cast<long long>(T) * D, dscores, static_cast<long long>(T) * T);
    bmm(s, true, false, B * H, T, D, T, probs, static_cast<long long>(T) * T, b.datty.as<float>(),
        static_cast<long long>(T) * D, b.dv.as<float>(), static_cast<long long>(T) * D);
    softmax_causal_backward(dscores, dscores, probs, B * H, T, scale, s);  // yerinde (satir once okunur)
    // dq = dscores k ; dk = dscores^T q
    bmm(s, false, false, B * H, T, D, T, dscores, static_cast<long long>(T) * T, k, static_cast<long long>(T) * D,
        b.dq.as<float>(), static_cast<long long>(T) * D);
    bmm(s, true, false, B * H, T, D, T, dscores, static_cast<long long>(T) * T, q, static_cast<long long>(T) * D,
        b.dk.as<float>(), static_cast<long long>(T) * D);
    merge_qkv_heads(b.dqkv.as<float>(), b.dq.as<float>(), b.dk.as<float>(), b.dv.as<float>(), B, T, H, D, s);
    mm(s, false, true, BT, C, 3 * C, b.dqkv.as<float>(), param(kQkvW) + lc * 3 * C, dln);
    mm(s, true, false, C, 3 * C, BT, ln1, b.dqkv.as<float>(), grad(kQkvW) + lc * 3 * C, Epilogue::BGradB, nullptr,
       nullptr, grad(kQkvB) + static_cast<std::size_t>(l) * 3 * C);
    layernorm_backward(dres, grad(kLn1W) + lc, grad(kLn1B) + lc, dln, x_in, param(kLn1W) + lc,
                       b.ln1_mean.as<float>() + static_cast<std::size_t>(l) * BT,
                       b.ln1_rstd.as<float>() + static_cast<std::size_t>(l) * BT, BT, C, true, scratch, s);
  }
  // Gomme: dwte'ye (kafanin yazdigi) EKLER; dwpe sifirdan.
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

void GptModel::optimizer_step(const DeviceAdamWConfig& config) {
  step_ += 1;
  std::vector<DeviceParamSlot> slots;
  slots.reserve(params_info_.size());
  for (const GptParamInfo& p : params_info_) {
    DeviceParamSlot slot;
    slot.param = params_.as<float>() + p.offset;
    slot.grad = grads_.as<float>() + p.offset;
    slot.exp_avg = exp_avg_.as<float>() + p.offset;
    slot.exp_avg_sq = exp_avg_sq_.as<float>() + p.offset;
    slot.numel = p.numel;
    slot.weight_decay_scale = p.decay ? 1.0f : 0.0f;
    slots.push_back(slot);
  }
  adamw_multi_tensor(slots.data(), static_cast<int>(slots.size()), config, step_, grad_norm_.as<float>(),
                     buffers_->adam_scratch.data(), stream_.handle());
}

std::vector<float> GptModel::download_parameters() const {
  std::vector<float> host(num_params_);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data(), params_.data(), num_params_ * 4, cudaMemcpyDeviceToHost, stream_.handle()));
  stream_.synchronize();
  return host;
}

void GptModel::upload_parameters(const std::vector<float>& values) {
  if (values.size() != num_params_) {
    throw TensorError("Nexus GPT: parametre sayisi uyusmuyor");
  }
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(params_.data(), values.data(), num_params_ * 4, cudaMemcpyHostToDevice, stream_.handle()));
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
