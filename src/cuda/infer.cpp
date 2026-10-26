/**
 * @file infer.cpp
 * @brief FAZ 16: GPT-2 GPU cikarim motoru (bkz. nexus/cuda/infer.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/infer.hpp"

#include "infer_kernels.hpp"
#include "nexus/cuda/attention.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/cuda/runtime.hpp"

#include <cublasLt.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace nexus {
namespace cuda {
namespace infer {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error("Nexus infer: " + why); }

void lt_check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) fail(std::string("cuBLASLt INT8 ") + what + " hata " + std::to_string(static_cast<int>(s)));
}

std::vector<std::uint16_t> to_half(const float* p, std::size_t n) {
  std::vector<std::uint16_t> h(n);
  for (std::size_t i = 0; i < n; ++i) {
    const __half v = __float2half_rn(p[i]);
    std::memcpy(&h[i], &v, 2);
  }
  return h;
}

struct Buf {
  DeviceBuffer b;
  std::size_t bytes = 0;
  void alloc(std::size_t n, const Stream& s) {
    bytes = std::max<std::size_t>(n, 16);
    b = DeviceBuffer(bytes, s);
    NEXUS_CUDA_CHECK(cudaMemsetAsync(b.data(), 0, bytes, s.handle()));
  }
  template <typename T>
  T* p() const {
    return static_cast<T*>(b.data());
  }
  void* v() const { return b.data(); }
};

void upload(Buf& dst, const void* src, std::size_t bytes, const Stream& s) {
  dst.alloc(bytes, s);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(dst.v(), src, bytes, cudaMemcpyHostToDevice, s.handle()));
}

}  // namespace

// ==========================================================================================
// Agirliklar
// ==========================================================================================

std::int64_t Gpt2Weights::parameters() const {
  std::int64_t n = static_cast<std::int64_t>(wte.size() + wpe.size() + ln_f_w.size() + ln_f_b.size());
  for (const Layer& l : h) {
    n += static_cast<std::int64_t>(l.ln1_w.size() + l.ln1_b.size() + l.attn_w.size() + l.attn_b.size() + l.proj_w.size() +
                                   l.proj_b.size() + l.ln2_w.size() + l.ln2_b.size() + l.fc_w.size() + l.fc_b.size() +
                                   l.fc2_w.size() + l.fc2_b.size());
  }
  return n;
}

Gpt2Weights load_gpt2(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("agirlik dosyasi acilamadi: " + path + " (benchmarks/runner/faz16_gpt2.py export)");
  char magic[4];
  in.read(magic, 4);
  if (std::memcmp(magic, "NXG2", 4) != 0) fail(path + ": NXG2 degil");
  std::uint32_t hdr[5];
  in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  Gpt2Weights w;
  w.layers = static_cast<int>(hdr[0]);
  w.heads = static_cast<int>(hdr[1]);
  w.d = static_cast<int>(hdr[2]);
  w.vocab = static_cast<int>(hdr[3]);
  w.ctx = static_cast<int>(hdr[4]);
  const std::size_t d = static_cast<std::size_t>(w.d);
  auto rd = [&](std::vector<float>& v, std::size_t n) {
    v.resize(n);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
  };
  rd(w.wte, static_cast<std::size_t>(w.vocab) * d);
  rd(w.wpe, static_cast<std::size_t>(w.ctx) * d);
  w.h.resize(static_cast<std::size_t>(w.layers));
  for (auto& l : w.h) {
    rd(l.ln1_w, d);
    rd(l.ln1_b, d);
    rd(l.attn_w, d * 3 * d);
    rd(l.attn_b, 3 * d);
    rd(l.proj_w, d * d);
    rd(l.proj_b, d);
    rd(l.ln2_w, d);
    rd(l.ln2_b, d);
    rd(l.fc_w, d * 4 * d);
    rd(l.fc_b, 4 * d);
    rd(l.fc2_w, 4 * d * d);
    rd(l.fc2_b, d);
  }
  rd(w.ln_f_w, d);
  rd(w.ln_f_b, d);
  if (!in) fail(path + ": dosya kisa (bozuk)");
  return w;
}

const char* to_string(WeightQuant q) noexcept {
  switch (q) {
    case WeightQuant::F16: return "fp16";
    case WeightQuant::W8A8: return "w8a8";
    case WeightQuant::W4A16: return "w4a16";
  }
  return "?";
}

// ==========================================================================================
// Motor
// ==========================================================================================

struct Engine::Impl {
  EngineConfig cfg;
  Stream stream;
  bool own_stream = false;
  int L = 0, H = 0, d = 0, V = 0, Vp = 0, ctx = 0, D = 64;
  DType act = DType::F16;     // aktivasyon / agirlik
  DType cache = DType::F16;   // KV
  std::size_t wbytes = 0;

  struct Linear {
    int k = 0, n = 0;
    Buf w, bias, bias16;            // F16/F32 [n, k] (ince GEMM + cuBLASLt trans_b); bias FP32 (+ FP16 kopya)
    Buf w8, sw;                     // W8A8: INT8 [n, k], olcek [n]
    Buf packed, scale, zero, inv_s; // W4A16
  };
  struct LayerW {
    Buf ln1w, ln1b, ln2w, ln2b;
    Linear qkv, proj, fc, fc2;
  };
  std::vector<LayerW> lw;
  Buf wte, wpe, lnfw, lnfb;  // wte [Vp, d] act tipi (sozluk dolgulu, lm_head icin de)

  // KV havuzu
  int num_blocks = 0, max_blocks = 0;
  std::vector<Buf> kc, vc, ks, vs;

  // Statik arena
  int N = 0, S = 0;
  Buf x, h, qkvb, o, f, mean, rstd, xr, hr, logits, lse, cu;
  Buf tok, pos, rowseq, slot, isout, gidx, btable, counts, out_counts, srows, sout, nllout, nlltg;
  Buf q8, sx, acc32, w4scratch;
  Buf sscratch;  // ornekleme: [kSampleRows, Vp]
  static constexpr int kSampleRows = 64;

  // INT8 GEMM (cuBLASLt)
  cublasLtHandle_t lt = nullptr;
  Buf ltws;

  // Siralar ve bloklar (host)
  struct Seq {
    bool used = false;
    std::vector<int> blocks;
    std::vector<std::int32_t> tokens;
    std::vector<std::uint8_t> is_output;
  };
  std::vector<Seq> seqs;
  std::vector<int> free_list;
  std::vector<int> refs;
  std::vector<std::int32_t> table_host;

  // Son adim
  int rows = 0;
  StepStats stats;
  std::map<int, cudaGraphExec_t> graphs;
  std::vector<std::vector<float>>* capture = nullptr;  // kalibrasyon: dogrusal girdileri [katman*4 + i]

  explicit Impl(EngineConfig c) : cfg(std::move(c)) {}
  ~Impl() {
    for (auto& g : graphs) cudaGraphExecDestroy(g.second);
    if (lt) cublasLtDestroy(lt);
    if (own_stream) {
      stream.synchronize();
      stream.destroy();
    }
  }

  cudaStream_t s() const { return stream.handle(); }
  std::size_t asz() const { return act == DType::F16 ? 2 : 4; }

  // ---------------------------------------------------------------------------------------
  void make_linear(Linear& li, const std::vector<float>& w, const std::vector<float>& b, int k, int n,
                   const std::vector<float>* smooth, bool smooth_in_kernel) {
    li.k = k;
    li.n = n;
    upload(li.bias, b.data(), b.size() * 4, stream);
    if (act == DType::F16) {
      const auto hb = to_half(b.data(), b.size());
      upload(li.bias16, hb.data(), hb.size() * 2, stream);
    }
    const WeightQuant q = cfg.quant.weights;
    std::vector<float> ws = w;  // [k, n]
    if (smooth != nullptr) {
      for (int kk = 0; kk < k; ++kk)
        for (int c = 0; c < n; ++c) ws[static_cast<std::size_t>(kk) * n + c] *= (*smooth)[static_cast<std::size_t>(kk)];
    }
    if (q == WeightQuant::F16 || act == DType::F32) {
      std::vector<float> wt(ws.size());  // [k, n] -> [n, k]
      for (int kk = 0; kk < k; ++kk)
        for (int c = 0; c < n; ++c) wt[static_cast<std::size_t>(c) * k + kk] = ws[static_cast<std::size_t>(kk) * n + c];
      ws.swap(wt);
      if (act == DType::F16) {
        const auto hw = to_half(ws.data(), ws.size());
        upload(li.w, hw.data(), hw.size() * 2, stream);
        wbytes += hw.size() * 2;
      } else {
        upload(li.w, ws.data(), ws.size() * 4, stream);
        wbytes += ws.size() * 4;
      }
    } else if (q == WeightQuant::W8A8) {
      std::vector<std::int8_t> w8(static_cast<std::size_t>(n) * k);
      std::vector<float> sw(static_cast<std::size_t>(n));
      for (int c = 0; c < n; ++c) {
        float amax = 0.0f;
        for (int kk = 0; kk < k; ++kk) amax = std::max(amax, std::fabs(ws[static_cast<std::size_t>(kk) * n + c]));
        const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
        sw[static_cast<std::size_t>(c)] = sc;
        for (int kk = 0; kk < k; ++kk) {
          const float v = std::nearbyint(ws[static_cast<std::size_t>(kk) * n + c] / sc);
          w8[static_cast<std::size_t>(c) * k + kk] = static_cast<std::int8_t>(std::max(-127.0f, std::min(127.0f, v)));
        }
      }
      upload(li.w8, w8.data(), w8.size(), stream);
      upload(li.sw, sw.data(), sw.size() * 4, stream);
      wbytes += w8.size() + sw.size() * 4;
    } else {  // W4A16
      const int g = cfg.quant.w4_group;
      if (k % g != 0) fail("W4 grup boyu K'yi bolmeli");
      const int groups = k / g;
      std::vector<std::uint8_t> packed(static_cast<std::size_t>(n) * (k / 2), 0), zero(static_cast<std::size_t>(n) * groups);
      std::vector<float> scale(static_cast<std::size_t>(n) * groups);
      for (int c = 0; c < n; ++c) {
        for (int gi = 0; gi < groups; ++gi) {
          float mn = 1e30f, mx = -1e30f;
          for (int kk = gi * g; kk < (gi + 1) * g; ++kk) {
            const float v = ws[static_cast<std::size_t>(kk) * n + c];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
          }
          const float sc = mx > mn ? (mx - mn) / 15.0f : 1.0f;
          const float z = std::max(0.0f, std::min(15.0f, std::nearbyint(-mn / sc)));
          scale[static_cast<std::size_t>(c) * groups + gi] = sc;
          zero[static_cast<std::size_t>(c) * groups + gi] = static_cast<std::uint8_t>(z);
          for (int kk = gi * g; kk < (gi + 1) * g; ++kk) {
            const float v = ws[static_cast<std::size_t>(kk) * n + c];
            const int qv = static_cast<int>(std::max(0.0f, std::min(15.0f, std::nearbyint(v / sc) + z)));
            std::uint8_t& byte = packed[static_cast<std::size_t>(c) * (k / 2) + kk / 2];
            byte = static_cast<std::uint8_t>((kk & 1) ? (byte | (qv << 4)) : (byte | qv));
          }
        }
      }
      upload(li.packed, packed.data(), packed.size(), stream);
      upload(li.scale, scale.data(), scale.size() * 4, stream);
      upload(li.zero, zero.data(), zero.size(), stream);
      wbytes += packed.size() + scale.size() * 4 + zero.size();
      if (smooth != nullptr && smooth_in_kernel) {
        std::vector<float> inv(static_cast<std::size_t>(k));
        for (int kk = 0; kk < k; ++kk) inv[static_cast<std::size_t>(kk)] = 1.0f / (*smooth)[static_cast<std::size_t>(kk)];
        upload(li.inv_s, inv.data(), inv.size() * 4, stream);
      }
    }
  }

  void upload_act(Buf& dst, const float* p, std::size_t n) {
    if (act == DType::F16) {
      const auto hw = to_half(p, n);
      upload(dst, hw.data(), hw.size() * 2, stream);
      wbytes += n * 2;
    } else {
      upload(dst, p, n * 4, stream);
      wbytes += n * 4;
    }
  }

  // ---------------------------------------------------------------------------------------
  void linear(const Linear& li, const void* in, void* out, bool accumulate, bool gelu, int m) {
    if (m == 0) return;
    const DType ytype = accumulate ? DType::F32 : act;
    const WeightQuant q = act == DType::F32 ? WeightQuant::F16 : cfg.quant.weights;
    if (q == WeightQuant::F16) {
      if (m <= k::kSkinnyMaxRows) {
        k::SkinnyArgs a;
        a.x = in;
        a.xtype = act;
        a.w = li.w.v();
        a.wtype = act;
        a.w_nk = true;
        a.bias = li.bias.p<float>();
        a.y = out;
        a.ytype = ytype;
        a.accumulate = accumulate;
        a.gelu = gelu;
        a.m = m;
        a.n = li.n;
        a.k = li.k;
        k::skinny_gemm(a, s());
        return;
      }
      GemmArgs g;
      g.trans_b = true;
      g.m = m;
      g.n = li.n;
      g.k = li.k;
      g.a = in;
      g.lda = li.k;
      g.b = li.w.v();
      g.ldb = li.k;
      g.c = out;
      g.ldc = li.n;
      g.a_type = act;
      g.b_type = act;
      g.c_type = ytype;
      g.beta = accumulate ? 1.0f : 0.0f;
      g.epilogue = gelu ? Epilogue::BiasGelu : Epilogue::Bias;
      g.bias = ytype == DType::F16 ? li.bias16.v() : li.bias.v();
      g.bias_type = ytype == DType::F16 ? DType::F16 : DType::F32;
      gemm(g, s());
      return;
    }
    if (q == WeightQuant::W8A8) {
      k::quant_rows_i8(q8.p<std::int8_t>(), sx.p<float>(), in, nullptr, m, li.k, s());
      int8_gemm(acc32.p<std::int32_t>(), q8.p<std::int8_t>(), li.w8.p<std::int8_t>(), m, li.n, li.k);
      k::dequant_i32(out, ytype, accumulate, gelu, acc32.p<std::int32_t>(), sx.p<float>(), li.sw.p<float>(),
                     li.bias.p<float>(), m, li.n, s());
      return;
    }
    // W4A16
    if (m <= k::kSkinnyMaxRows) {
      k::W4Args a;
      a.x = in;
      a.inv_s = li.inv_s.bytes > 16 ? li.inv_s.p<float>() : nullptr;
      a.packed = li.packed.p<std::uint8_t>();
      a.scale = li.scale.p<float>();
      a.zero = li.zero.p<std::uint8_t>();
      a.group = cfg.quant.w4_group;
      a.bias = li.bias.p<float>();
      a.y = out;
      a.ytype = ytype;
      a.accumulate = accumulate;
      a.gelu = gelu;
      a.m = m;
      a.n = li.n;
      a.k = li.k;
      k::w4_gemv(a, s());
      return;
    }
    k::w4_dequant(w4scratch.v(), li.packed.p<std::uint8_t>(), li.scale.p<float>(), li.zero.p<std::uint8_t>(),
                  cfg.quant.w4_group, li.n, li.k, s());
    if (li.inv_s.bytes > 16) k::scale_cols(const_cast<void*>(in), li.inv_s.p<float>(), m, li.k, s());
    GemmArgs g;
    g.m = m;
    g.n = li.n;
    g.k = li.k;
    g.a = in;
    g.lda = li.k;
    g.b = w4scratch.v();
    g.ldb = li.n;
    g.c = out;
    g.ldc = li.n;
    g.a_type = DType::F16;
    g.b_type = DType::F16;
    g.c_type = ytype;
    g.beta = accumulate ? 1.0f : 0.0f;
    g.epilogue = gelu ? Epilogue::BiasGelu : Epilogue::Bias;
    g.bias = ytype == DType::F16 ? li.bias16.v() : li.bias.v();
    g.bias_type = ytype == DType::F16 ? DType::F16 : DType::F32;
    gemm(g, s());
  }

  struct LtPlan {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
  };
  std::map<std::tuple<int, int, int>, LtPlan> lt_plans;

  /// acc[m, n] (INT32, satir-ana) = Xq[m, k] * W8[n, k]^T. Sutun-ana: C[n x m] = W8^T(T) * Xq(N) ("TN").
  void int8_gemm(std::int32_t* acc, const std::int8_t* xq, const std::int8_t* w8, int m, int n, int kk) {
    if (lt == nullptr) {
      lt_check(cublasLtCreate(&lt), "handle");
      ltws.alloc(std::size_t{32} << 20, stream);
    }
    auto key = std::make_tuple(m, n, kk);
    auto it = lt_plans.find(key);
    if (it == lt_plans.end()) {
      LtPlan p;
      lt_check(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32I, CUDA_R_32I), "desc");
      const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
      lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
      lt_check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
      lt_check(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8I, static_cast<std::uint64_t>(kk), static_cast<std::uint64_t>(n), kk), "A");
      lt_check(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8I, static_cast<std::uint64_t>(kk), static_cast<std::uint64_t>(m), kk), "B");
      lt_check(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32I, static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(m), n), "C");
      cublasLtMatmulPreference_t pref = nullptr;
      lt_check(cublasLtMatmulPreferenceCreate(&pref), "pref");
      const std::size_t ws = ltws.bytes;
      lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)), "ws");
      cublasLtMatmulHeuristicResult_t res{};
      int found = 0;
      lt_check(cublasLtMatmulAlgoGetHeuristic(lt, p.desc, p.a, p.b, p.c, p.c, pref, 1, &res, &found), "heuristic");
      cublasLtMatmulPreferenceDestroy(pref);
      if (found == 0) fail("cuBLASLt INT8 GEMM algoritma bulamadi (m=" + std::to_string(m) + ")");
      p.algo = res.algo;
      it = lt_plans.emplace(key, p).first;
    }
    const std::int32_t alpha = 1, beta = 0;
    lt_check(cublasLtMatmul(lt, it->second.desc, &alpha, w8, it->second.a, xq, it->second.b, &beta, acc, it->second.c, acc,
                            it->second.c, &it->second.algo, ltws.v(), ltws.bytes, s()),
             "matmul");
  }

  // ---------------------------------------------------------------------------------------
  void capture_linear_input(int layer, int which, const void* in, int m, int kk) {
    if (capture == nullptr) return;
    std::vector<float>& dst = (*capture)[static_cast<std::size_t>(layer) * 4 + which];
    dst.resize(static_cast<std::size_t>(m) * kk);
    if (act != DType::F32) fail("kalibrasyon yakalama FP32 motor ister");
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(dst.data(), in, dst.size() * 4, cudaMemcpyDeviceToHost, s()));
    NEXUS_CUDA_CHECK(cudaStreamSynchronize(s()));
  }

  /// n token (ilk nf tanesi taze prefill: flash varlen), mrows logit satiri, `layers` katman.
  void compute(int n, int nf, int nfseq, int maxlen, int mrows, int layers) {
    const int HD = H * D;
    k::embed(x.p<float>(), tok.p<std::int32_t>(), pos.p<std::int32_t>(), wte.v(), wpe.v(), act, n, d, s());
    for (int l = 0; l < layers; ++l) {
      LayerW& w = lw[static_cast<std::size_t>(l)];
      layernorm_forward(h.v(), act, mean.p<float>(), rstd.p<float>(), x.p<float>(), w.ln1w.p<float>(), w.ln1b.p<float>(), n, d,
                        1e-5f, s());
      capture_linear_input(l, 0, h.v(), n, d);
      linear(w.qkv, h.v(), qkvb.v(), false, false, n);
      k::kv_append(kc[static_cast<std::size_t>(l)].v(), vc[static_cast<std::size_t>(l)].v(),
                   cache == DType::I8 ? ks[static_cast<std::size_t>(l)].p<float>() : nullptr,
                   cache == DType::I8 ? vs[static_cast<std::size_t>(l)].p<float>() : nullptr, cache, qkvb.v(), act,
                   slot.p<std::int32_t>(), n, H, D, s());
      if (nf > 0) {
        AttentionParams p;
        p.batch = nfseq;
        p.heads = H;
        p.kv_heads = H;
        p.tq = maxlen;
        p.tk = maxlen;
        p.head_dim = D;
        p.dtype = DType::F16;
        p.causal = true;
        p.cu_seqlens_q = cu.p<std::int32_t>();
        p.cu_seqlens_k = cu.p<std::int32_t>();
        p.q_row_stride = 3LL * HD;
        p.q_head_stride = D;
        p.k_row_stride = 3LL * HD;
        p.k_head_stride = D;
        p.o_row_stride = HD;
        p.o_head_stride = D;
        const auto* q = static_cast<const __half*>(qkvb.v());
        flash_attention_forward(o.v(), lse.p<float>(), q, q + HD, q + 2 * HD, p, s());
      }
      if (n - nf > 0) {
        k::PagedArgs a;
        a.qkv = static_cast<const char*>(qkvb.v()) + static_cast<std::size_t>(nf) * 3 * HD * asz();
        a.out = static_cast<char*>(o.v()) + static_cast<std::size_t>(nf) * HD * asz();
        a.act = act;
        a.kc = kc[static_cast<std::size_t>(l)].v();
        a.vc = vc[static_cast<std::size_t>(l)].v();
        a.kscale = cache == DType::I8 ? ks[static_cast<std::size_t>(l)].p<float>() : nullptr;
        a.vscale = cache == DType::I8 ? vs[static_cast<std::size_t>(l)].p<float>() : nullptr;
        a.cache = cache;
        a.row_seq = rowseq.p<std::int32_t>() + nf;
        a.row_pos = pos.p<std::int32_t>() + nf;
        a.block_table = btable.p<std::int32_t>();
        a.max_blocks = max_blocks;
        a.block_size = cfg.block_size;
        a.n = n - nf;
        a.H = H;
        a.D = D;
        a.max_ctx = ctx;
        a.scale = 1.0f / std::sqrt(static_cast<float>(D));
        k::paged_attention(a, s());
      }
      capture_linear_input(l, 1, o.v(), n, d);
      linear(w.proj, o.v(), x.v(), true, false, n);
      layernorm_forward(h.v(), act, mean.p<float>(), rstd.p<float>(), x.p<float>(), w.ln2w.p<float>(), w.ln2b.p<float>(), n, d,
                        1e-5f, s());
      capture_linear_input(l, 2, h.v(), n, d);
      linear(w.fc, h.v(), f.v(), false, true, n);
      capture_linear_input(l, 3, f.v(), n, 4 * d);
      linear(w.fc2, f.v(), x.v(), true, false, n);
    }
    if (mrows > 0) {
      k::gather_rows(xr.p<float>(), x.p<float>(), gidx.p<std::int32_t>(), mrows, d, s());
      layernorm_forward(hr.v(), act, mean.p<float>(), rstd.p<float>(), xr.p<float>(), lnfw.p<float>(), lnfb.p<float>(), mrows, d,
                        1e-5f, s());
      if (mrows <= k::kSkinnyMaxRows) {
        k::SkinnyArgs a;
        a.x = hr.v();
        a.xtype = act;
        a.w = wte.v();
        a.wtype = act;
        a.w_nk = true;
        a.y = logits.v();
        a.ytype = DType::F32;
        a.m = mrows;
        a.n = Vp;
        a.k = d;
        k::skinny_gemm(a, s());
      } else {
        GemmArgs g;
        g.trans_b = true;
        g.m = mrows;
        g.n = Vp;
        g.k = d;
        g.a = hr.v();
        g.lda = d;
        g.b = wte.v();
        g.ldb = d;
        g.c = logits.v();
        g.ldc = Vp;
        g.a_type = act;
        g.b_type = act;
        g.c_type = DType::F32;
        gemm(g, s());
      }
    }
    k::count_tokens(counts.p<std::int32_t>(), out_counts.p<std::int32_t>(), rowseq.p<std::int32_t>(), tok.p<std::int32_t>(),
                    isout.p<std::uint8_t>(), n, V, 1, s());
  }

  // ---- bloklar -----------------------------------------------------------------------------
  int alloc_block() {
    if (free_list.empty()) fail("KV bloklari tukendi (" + std::to_string(num_blocks - 1) + " blok)");
    const int b = free_list.back();
    free_list.pop_back();
    refs[static_cast<std::size_t>(b)] = 1;
    return b;
  }
  void release_block(int b) {
    if (--refs[static_cast<std::size_t>(b)] == 0) free_list.push_back(b);
  }
  void copy_block(int dst, int src) {
    const std::size_t per = static_cast<std::size_t>(cfg.block_size) * H * D * (cache == DType::I8 ? 1 : asz());
    const std::size_t sper = static_cast<std::size_t>(cfg.block_size) * H * 4;
    for (int l = 0; l < L; ++l) {
      auto cp = [&](Buf& b, std::size_t unit) {
        NEXUS_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(b.v()) + dst * unit, static_cast<char*>(b.v()) + src * unit, unit,
                                         cudaMemcpyDeviceToDevice, s()));
      };
      cp(kc[static_cast<std::size_t>(l)], per);
      cp(vc[static_cast<std::size_t>(l)], per);
      if (cache == DType::I8) {
        cp(ks[static_cast<std::size_t>(l)], sper);
        cp(vs[static_cast<std::size_t>(l)], sper);
      }
    }
  }
  Seq& seq_at(int i) {
    if (i < 0 || i >= S || !seqs[static_cast<std::size_t>(i)].used) fail("gecersiz sira " + std::to_string(i));
    return seqs[static_cast<std::size_t>(i)];
  }
  const Seq& seq_at(int i) const { return const_cast<Impl*>(this)->seq_at(i); }
  void sync_table(int i) {
    const Seq& q = seqs[static_cast<std::size_t>(i)];
    std::int32_t* row = table_host.data() + static_cast<std::size_t>(i) * max_blocks;
    std::fill(row, row + max_blocks, 0);
    for (std::size_t b = 0; b < q.blocks.size(); ++b) row[b] = q.blocks[b];
  }
  void reserve(int i, int new_tokens) {
    Seq& q = seq_at(i);
    const int len = static_cast<int>(q.tokens.size());
    if (len + new_tokens > ctx) fail("sira " + std::to_string(i) + " baglam sinirini asiyor (" + std::to_string(ctx) + ")");
    const int bs = cfg.block_size;
    if (len % bs != 0 && !q.blocks.empty() && refs[static_cast<std::size_t>(q.blocks.back())] > 1) {  // copy-on-write
      const int nb = alloc_block();
      copy_block(nb, q.blocks.back());
      release_block(q.blocks.back());
      q.blocks.back() = nb;
    }
    while (static_cast<int>(q.blocks.size()) * bs < len + new_tokens) q.blocks.push_back(alloc_block());
    sync_table(i);
  }

  // ---- ileri -------------------------------------------------------------------------------
  const float* forward(const std::vector<Chunk>& chunks, int layers) {
    const auto t0 = Clock::now();
    if (layers < 0) layers = L;
    if (layers < 1 || layers > L) fail("katman sayisi 1.." + std::to_string(L));
    // taze prefill parcalari once (flash varlen; FP32 kipinde yok)
    std::vector<int> order;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const Chunk& ch = chunks[c];
      if (ch.tokens.empty()) fail("bos parca");
      if (act == DType::F16 && seq_at(ch.seq).tokens.empty() && ch.tokens.size() > 1) order.push_back(static_cast<int>(c));
    }
    const int nfseq = static_cast<int>(order.size());
    for (std::size_t c = 0; c < chunks.size(); ++c)
      if (std::find(order.begin(), order.begin() + nfseq, static_cast<int>(c)) == order.begin() + nfseq) order.push_back(static_cast<int>(c));
    int n = 0, nf = 0, maxlen = 0, mrows = 0;
    for (const Chunk& ch : chunks) {
      n += static_cast<int>(ch.tokens.size());
      mrows += std::min(ch.logits, static_cast<int>(ch.tokens.size()));
    }
    if (n > N) fail("adim token sayisi " + std::to_string(n) + " > max_tokens " + std::to_string(N));
    for (std::size_t c = 0; c < chunks.size(); ++c)
      for (std::size_t c2 = c + 1; c2 < chunks.size(); ++c2)
        if (chunks[c].seq == chunks[c2].seq) fail("ayni sira bir adimda iki kez");
    for (const Chunk& ch : chunks) reserve(ch.seq, static_cast<int>(ch.tokens.size()));
    // decode CUDA Graph kovasi
    bool decode = cfg.cuda_graphs && layers == L && capture == nullptr && !chunks.empty() && static_cast<int>(chunks.size()) <= S;
    for (const Chunk& ch : chunks) decode = decode && ch.tokens.size() == 1 && ch.logits == 1 && !seq_at(ch.seq).tokens.empty();
    int bucket = n;
    if (decode) {
      bucket = 1;
      while (bucket < n) bucket *= 2;
      bucket = std::min(bucket, S);
    }
    const int ntot = decode ? bucket : n;
    std::vector<std::int32_t> htok(static_cast<std::size_t>(ntot), 0), hpos(static_cast<std::size_t>(ntot), 0),
        hseq(static_cast<std::size_t>(ntot), S), hslot(static_cast<std::size_t>(ntot), 0), hidx;
    std::vector<std::uint8_t> hout(static_cast<std::size_t>(ntot), 0);
    std::vector<int> start(chunks.size());
    std::vector<std::int32_t> hcu(1, 0);
    int at = 0;
    for (int oi = 0; oi < static_cast<int>(order.size()); ++oi) {
      const int c = order[static_cast<std::size_t>(oi)];
      const Chunk& ch = chunks[static_cast<std::size_t>(c)];
      Seq& q = seqs[static_cast<std::size_t>(ch.seq)];
      start[static_cast<std::size_t>(c)] = at;
      const int len = static_cast<int>(q.tokens.size());
      for (std::size_t i = 0; i < ch.tokens.size(); ++i, ++at) {
        const int p = len + static_cast<int>(i);
        htok[static_cast<std::size_t>(at)] = ch.tokens[i];
        hpos[static_cast<std::size_t>(at)] = p;
        hseq[static_cast<std::size_t>(at)] = ch.seq;
        hslot[static_cast<std::size_t>(at)] = q.blocks[static_cast<std::size_t>(p / cfg.block_size)] * cfg.block_size + p % cfg.block_size;
        hout[static_cast<std::size_t>(at)] = ch.output ? 1 : 0;
      }
      if (oi < nfseq) {
        nf = at;
        hcu.push_back(at);
        maxlen = std::max(maxlen, static_cast<int>(ch.tokens.size()));
      }
    }
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const int nt = static_cast<int>(chunks[c].tokens.size());
      for (int i = nt - std::min(chunks[c].logits, nt); i < nt; ++i) hidx.push_back(start[c] + i);
    }
    if (decode)
      for (int i = n; i < bucket; ++i) hidx.push_back(i);
    const int mtot = static_cast<int>(hidx.size());
    auto put = [&](Buf& b, const void* src, std::size_t bytes) {
      if (bytes) NEXUS_CUDA_CHECK(cudaMemcpyAsync(b.v(), src, bytes, cudaMemcpyHostToDevice, s()));
    };
    put(tok, htok.data(), htok.size() * 4);
    put(pos, hpos.data(), hpos.size() * 4);
    put(rowseq, hseq.data(), hseq.size() * 4);
    put(slot, hslot.data(), hslot.size() * 4);
    put(isout, hout.data(), hout.size());
    put(gidx, hidx.data(), hidx.size() * 4);
    put(btable, table_host.data(), table_host.size() * 4);
    if (nfseq > 0) put(cu, hcu.data(), hcu.size() * 4);
    stats = StepStats{};
    stats.tokens = n;
    stats.prefill_tokens = nf;
    stats.decode_tokens = n - nf;
    if (decode) {
      auto it = graphs.find(bucket);
      if (it == graphs.end()) {
        compute(bucket, 0, 0, 0, bucket, L);  // isinma: cuBLASLt planlari + calisma alani yakalamadan once
        // isinma tokenlari say(a)ci iki kez artirmasin: yakalama oncesi geri al
        k::count_tokens(counts.p<std::int32_t>(), out_counts.p<std::int32_t>(), rowseq.p<std::int32_t>(),
                        tok.p<std::int32_t>(), isout.p<std::uint8_t>(), bucket, V, -1, s());
        cudaGraph_t g = nullptr;
        NEXUS_CUDA_CHECK(cudaStreamBeginCapture(s(), cudaStreamCaptureModeThreadLocal));
        compute(bucket, 0, 0, 0, bucket, L);
        NEXUS_CUDA_CHECK(cudaStreamEndCapture(s(), &g));
        cudaGraphExec_t exec = nullptr;
        NEXUS_CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
        cudaGraphDestroy(g);
        it = graphs.emplace(bucket, exec).first;
      }
      NEXUS_CUDA_CHECK(cudaGraphLaunch(it->second, s()));
      stats.graph = true;
    } else {
      compute(n, nf, nfseq, maxlen, mtot, layers);
    }
    for (const Chunk& ch : chunks) {
      Seq& q = seqs[static_cast<std::size_t>(ch.seq)];
      q.tokens.insert(q.tokens.end(), ch.tokens.begin(), ch.tokens.end());
      q.is_output.insert(q.is_output.end(), ch.tokens.size(), ch.output ? 1 : 0);
    }
    rows = decode ? n : mtot;
    NEXUS_CUDA_CHECK(cudaStreamSynchronize(s()));
    stats.ms = ms_since(t0);
    return logits.p<float>();
  }
};

// ------------------------------------------------------------------------------------------

namespace {

/// Kalibrasyon girdileri: FP32 motorla dogrusal katman girdileri (katman*4 + {qkv, proj, fc, fc2}).
std::vector<std::vector<float>> calibration_inputs(const Gpt2Weights& w, const std::vector<std::int32_t>& tokens) {
  EngineConfig c;
  c.precision = Precision::F32;
  c.max_seqs = 1;
  c.max_tokens = static_cast<int>(tokens.size());
  c.kv_bytes = (tokens.size() + 64) * static_cast<std::size_t>(w.layers) * 2 * w.d * 4 * 2;
  c.cuda_graphs = false;
  Engine e(w, c);
  std::vector<std::vector<float>> cap(static_cast<std::size_t>(w.layers) * 4);
  e.impl_capture(&cap);
  const int s = e.add_sequence();
  Chunk ch;
  ch.seq = s;
  ch.tokens = tokens;
  ch.logits = 0;
  (void)e.forward({ch});
  return cap;
}

}  // namespace

void Engine::impl_capture(std::vector<std::vector<float>>* cap) { impl_->capture = cap; }

Engine::Engine(const Gpt2Weights& w, EngineConfig config, cudaStream_t stream) : impl_(std::make_unique<Impl>(std::move(config))) {
  Impl& m = *impl_;
  const EngineConfig& c = m.cfg;
  int dev = 0;
  NEXUS_CUDA_CHECK(cudaGetDevice(&dev));
  if (stream == nullptr) {
    m.stream = Stream::create(dev);
    m.own_stream = true;
  } else {
    m.stream = Stream(stream, dev);
  }
  m.L = w.layers;
  m.H = w.heads;
  m.d = w.d;
  m.V = w.vocab;
  m.Vp = (w.vocab + 63) / 64 * 64;
  m.ctx = w.ctx;
  m.D = w.d / w.heads;
  if (m.D != 64) fail("yalniz head_dim 64 (GPT-2)");
  m.act = c.precision == Precision::F32 ? DType::F32 : DType::F16;
  if (c.quant.weights != WeightQuant::F16 && c.precision == Precision::F32) fail("nicemleme FP16 kipini ister");
  m.cache = c.quant.kv == KvType::INT8 ? DType::I8 : m.act;
  if (c.precision == Precision::F32) set_float32_matmul_precision(MatmulPrecision::Highest);
  m.N = c.max_tokens;
  m.S = c.max_seqs;
  const std::size_t d = static_cast<std::size_t>(m.d);

  // ---- olcekler (SmoothQuant / AWQ) ----
  std::vector<std::vector<float>> smooth(static_cast<std::size_t>(m.L) * 4);
  const bool quant = c.quant.weights != WeightQuant::F16;
  if (quant && !c.quant.calibration.empty()) {
    const auto cap = calibration_inputs(w, c.quant.calibration);
    for (int l = 0; l < m.L; ++l) {
      const Gpt2Weights::Layer& lyr = w.h[static_cast<std::size_t>(l)];
      const std::vector<float>* W[4] = {&lyr.attn_w, &lyr.proj_w, &lyr.fc_w, &lyr.fc2_w};
      const int K[4] = {m.d, m.d, m.d, 4 * m.d}, Nn[4] = {3 * m.d, m.d, 4 * m.d, m.d};
      for (int i = 0; i < 4; ++i) {
        const std::vector<float>& X = cap[static_cast<std::size_t>(l) * 4 + i];
        const int kk = K[i], nn = Nn[i];
        const std::size_t rows = X.size() / static_cast<std::size_t>(kk);
        std::vector<float> xmax(static_cast<std::size_t>(kk), 0.0f), xmean(static_cast<std::size_t>(kk), 0.0f),
            wmax(static_cast<std::size_t>(kk), 0.0f);
        for (std::size_t r = 0; r < rows; ++r)
          for (int j = 0; j < kk; ++j) {
            const float v = std::fabs(X[r * kk + j]);
            xmax[static_cast<std::size_t>(j)] = std::max(xmax[static_cast<std::size_t>(j)], v);
            xmean[static_cast<std::size_t>(j)] += v / static_cast<float>(rows);
          }
        for (int j = 0; j < kk; ++j)
          for (int o = 0; o < nn; ++o)
            wmax[static_cast<std::size_t>(j)] = std::max(wmax[static_cast<std::size_t>(j)], std::fabs((*W[i])[static_cast<std::size_t>(j) * nn + o]));
        std::vector<float>& sv = smooth[static_cast<std::size_t>(l) * 4 + i];
        sv.assign(static_cast<std::size_t>(kk), 1.0f);
        const bool foldable = i == 0 || i == 2;  // qkv / fc: 1/s onceki layernorm'a katlanir
        if (c.quant.weights == WeightQuant::W8A8) {
          if (!foldable) continue;  // proj / fc2 girdisi katlanamaz: olcek 1
          const float a = c.quant.smooth_alpha;
          for (int j = 0; j < kk; ++j) {
            const float s = std::pow(std::max(xmax[static_cast<std::size_t>(j)], 1e-5f), a) /
                            std::pow(std::max(wmax[static_cast<std::size_t>(j)], 1e-5f), 1.0f - a);
            sv[static_cast<std::size_t>(j)] = std::max(s, 1e-5f);
          }
        } else {  // AWQ: s = mean|x|^alpha, alfa izgarada cikis hatasina gore (ilk 128 satir)
          const std::size_t probe = std::min<std::size_t>(rows, 128);
          std::vector<double> ref(probe * static_cast<std::size_t>(nn), 0.0);
          for (std::size_t r = 0; r < probe; ++r)
            for (int j = 0; j < kk; ++j) {
              const double xv = X[r * kk + j];
              if (xv == 0.0) continue;
              const float* wr = W[i]->data() + static_cast<std::size_t>(j) * nn;
              double* rr = ref.data() + r * nn;
              for (int o = 0; o < nn; ++o) rr[o] += xv * wr[o];
            }
          double best = 1e300;
          std::vector<float> best_s(static_cast<std::size_t>(kk), 1.0f);
          const int g = c.quant.w4_group;
          for (float alpha : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
            std::vector<float> s(static_cast<std::size_t>(kk));
            double lo = 1e30, hi = 0;
            for (int j = 0; j < kk; ++j) {
              s[static_cast<std::size_t>(j)] = std::pow(std::max(xmean[static_cast<std::size_t>(j)], 1e-4f), alpha);
              lo = std::min(lo, static_cast<double>(s[static_cast<std::size_t>(j)]));
              hi = std::max(hi, static_cast<double>(s[static_cast<std::size_t>(j)]));
            }
            const float norm = static_cast<float>(std::sqrt(lo * hi));
            for (float& v : s) v /= norm;
            // Q(diag(s) W) / s  (grup-bazli asimetrik 4 bit) -> hata
            std::vector<float> wq(static_cast<std::size_t>(kk) * nn);
            for (int o = 0; o < nn; ++o)
              for (int gi = 0; gi < kk / g; ++gi) {
                float mn = 1e30f, mx = -1e30f;
                for (int j = gi * g; j < (gi + 1) * g; ++j) {
                  const float v = (*W[i])[static_cast<std::size_t>(j) * nn + o] * s[static_cast<std::size_t>(j)];
                  mn = std::min(mn, v);
                  mx = std::max(mx, v);
                }
                const float sc = mx > mn ? (mx - mn) / 15.0f : 1.0f;
                const float z = std::max(0.0f, std::min(15.0f, std::nearbyint(-mn / sc)));
                for (int j = gi * g; j < (gi + 1) * g; ++j) {
                  const float v = (*W[i])[static_cast<std::size_t>(j) * nn + o] * s[static_cast<std::size_t>(j)];
                  const float q = std::max(0.0f, std::min(15.0f, std::nearbyint(v / sc) + z));
                  wq[static_cast<std::size_t>(j) * nn + o] = (q - z) * sc / s[static_cast<std::size_t>(j)];
                }
              }
            double err = 0.0;
            std::vector<double> out(static_cast<std::size_t>(nn));
            for (std::size_t r = 0; r < probe; ++r) {
              std::fill(out.begin(), out.end(), 0.0);
              for (int j = 0; j < kk; ++j) {
                const double xv = X[r * kk + j];
                if (xv == 0.0) continue;
                const float* wr = wq.data() + static_cast<std::size_t>(j) * nn;
                for (int o = 0; o < nn; ++o) out[static_cast<std::size_t>(o)] += xv * wr[o];
              }
              for (int o = 0; o < nn; ++o) {
                const double dlt = out[static_cast<std::size_t>(o)] - ref[r * nn + o];
                err += dlt * dlt;
              }
            }
            if (err < best) {
              best = err;
              best_s = s;
            }
          }
          sv = best_s;
        }
      }
    }
  }

  // ---- agirliklar ----
  std::vector<float> wte_p(static_cast<std::size_t>(m.Vp) * d, 0.0f);
  std::copy(w.wte.begin(), w.wte.end(), wte_p.begin());
  m.upload_act(m.wte, wte_p.data(), wte_p.size());
  m.upload_act(m.wpe, w.wpe.data(), w.wpe.size());
  upload(m.lnfw, w.ln_f_w.data(), d * 4, m.stream);
  upload(m.lnfb, w.ln_f_b.data(), d * 4, m.stream);
  m.lw.resize(static_cast<std::size_t>(m.L));
  for (int l = 0; l < m.L; ++l) {
    const Gpt2Weights::Layer& lyr = w.h[static_cast<std::size_t>(l)];
    Impl::LayerW& o = m.lw[static_cast<std::size_t>(l)];
    const auto& s0 = smooth[static_cast<std::size_t>(l) * 4 + 0];
    const auto& s1 = smooth[static_cast<std::size_t>(l) * 4 + 1];
    const auto& s2 = smooth[static_cast<std::size_t>(l) * 4 + 2];
    const auto& s3 = smooth[static_cast<std::size_t>(l) * 4 + 3];
    auto fold = [&](const std::vector<float>& v, const std::vector<float>& sv) {  // LN gamma/beta / s
      std::vector<float> r = v;
      if (!sv.empty())
        for (std::size_t j = 0; j < r.size(); ++j) r[j] /= sv[j];
      return r;
    };
    const auto g1 = fold(lyr.ln1_w, s0), b1 = fold(lyr.ln1_b, s0), g2 = fold(lyr.ln2_w, s2), b2 = fold(lyr.ln2_b, s2);
    upload(o.ln1w, g1.data(), d * 4, m.stream);
    upload(o.ln1b, b1.data(), d * 4, m.stream);
    upload(o.ln2w, g2.data(), d * 4, m.stream);
    upload(o.ln2b, b2.data(), d * 4, m.stream);
    m.make_linear(o.qkv, lyr.attn_w, lyr.attn_b, m.d, 3 * m.d, s0.empty() ? nullptr : &s0, false);
    m.make_linear(o.proj, lyr.proj_w, lyr.proj_b, m.d, m.d, s1.empty() ? nullptr : &s1, true);
    m.make_linear(o.fc, lyr.fc_w, lyr.fc_b, m.d, 4 * m.d, s2.empty() ? nullptr : &s2, false);
    m.make_linear(o.fc2, lyr.fc2_w, lyr.fc2_b, 4 * m.d, m.d, s3.empty() ? nullptr : &s3, true);
  }

  // ---- KV havuzu ----
  const std::size_t per_tok = kv_bytes_per_token();
  m.max_blocks = (m.ctx + c.block_size - 1) / c.block_size;
  m.num_blocks = static_cast<int>(c.kv_bytes / (per_tok * static_cast<std::size_t>(c.block_size)));
  if (m.num_blocks < 2) fail("kv_bytes cok kucuk");
  const std::size_t slots = static_cast<std::size_t>(m.num_blocks) * c.block_size;
  const std::size_t elem = m.cache == DType::I8 ? 1 : m.asz();
  m.kc.resize(static_cast<std::size_t>(m.L));
  m.vc.resize(static_cast<std::size_t>(m.L));
  m.ks.resize(static_cast<std::size_t>(m.L));
  m.vs.resize(static_cast<std::size_t>(m.L));
  for (int l = 0; l < m.L; ++l) {
    m.kc[static_cast<std::size_t>(l)].alloc(slots * m.H * m.D * elem, m.stream);
    m.vc[static_cast<std::size_t>(l)].alloc(slots * m.H * m.D * elem, m.stream);
    if (m.cache == DType::I8) {
      m.ks[static_cast<std::size_t>(l)].alloc(slots * m.H * 4, m.stream);
      m.vs[static_cast<std::size_t>(l)].alloc(slots * m.H * 4, m.stream);
    }
  }
  m.refs.assign(static_cast<std::size_t>(m.num_blocks), 0);
  for (int b = m.num_blocks - 1; b >= 1; --b) m.free_list.push_back(b);  // blok 0: dolgu tokenlarinin yuvasi
  m.seqs.resize(static_cast<std::size_t>(m.S));
  m.table_host.assign(static_cast<std::size_t>(m.S + 1) * m.max_blocks, 0);

  // ---- arena ----
  const std::size_t N = static_cast<std::size_t>(m.N), as = m.asz();
  m.x.alloc(N * d * 4, m.stream);
  m.h.alloc(N * d * as, m.stream);
  m.qkvb.alloc(N * 3 * d * as, m.stream);
  m.o.alloc(N * d * as, m.stream);
  m.f.alloc(N * 4 * d * as, m.stream);
  m.mean.alloc(N * 4, m.stream);
  m.rstd.alloc(N * 4, m.stream);
  m.xr.alloc(N * d * 4, m.stream);
  m.hr.alloc(N * d * as, m.stream);
  m.logits.alloc(N * static_cast<std::size_t>(m.Vp) * 4, m.stream);
  m.lse.alloc(N * static_cast<std::size_t>(m.H) * 4, m.stream);
  m.cu.alloc(static_cast<std::size_t>(m.S + 1) * 4, m.stream);
  m.tok.alloc(N * 4, m.stream);
  m.pos.alloc(N * 4, m.stream);
  m.rowseq.alloc(N * 4, m.stream);
  m.slot.alloc(N * 4, m.stream);
  m.isout.alloc(N, m.stream);
  m.gidx.alloc(N * 4, m.stream);
  m.btable.alloc(m.table_host.size() * 4, m.stream);
  m.counts.alloc(static_cast<std::size_t>(m.S + 1) * m.V * 4, m.stream);
  m.out_counts.alloc(static_cast<std::size_t>(m.S + 1) * m.V * 4, m.stream);
  m.srows.alloc(N * sizeof(k::SampleRow), m.stream);
  m.sscratch.alloc(static_cast<std::size_t>(Impl::kSampleRows) * m.Vp * 4, m.stream);
  m.sout.alloc(N * 4, m.stream);
  m.nllout.alloc(N * 4, m.stream);
  m.nlltg.alloc(N * 4, m.stream);
  if (c.quant.weights == WeightQuant::W8A8) {
    m.q8.alloc(N * 4 * d, m.stream);
    m.sx.alloc(N * 4, m.stream);
    m.acc32.alloc(N * 4 * d * 4, m.stream);
  }
  if (c.quant.weights == WeightQuant::W4A16) m.w4scratch.alloc(4 * d * d * 2, m.stream);
  NEXUS_CUDA_CHECK(cudaStreamSynchronize(m.s()));
}

Engine::~Engine() = default;

int Engine::add_sequence() {
  Impl& m = *impl_;
  for (int i = 0; i < m.S; ++i) {
    if (!m.seqs[static_cast<std::size_t>(i)].used) {
      m.seqs[static_cast<std::size_t>(i)] = Impl::Seq{};
      m.seqs[static_cast<std::size_t>(i)].used = true;
      NEXUS_CUDA_CHECK(cudaMemsetAsync(m.counts.p<std::int32_t>() + static_cast<std::size_t>(i) * m.V, 0,
                                       static_cast<std::size_t>(m.V) * 4, m.s()));
      NEXUS_CUDA_CHECK(cudaMemsetAsync(m.out_counts.p<std::int32_t>() + static_cast<std::size_t>(i) * m.V, 0,
                                       static_cast<std::size_t>(m.V) * 4, m.s()));
      m.sync_table(i);
      return i;
    }
  }
  fail("sira yuvasi yok (max_seqs " + std::to_string(m.S) + ")");
}

int Engine::fork_sequence(int src) {
  Impl& m = *impl_;
  const Impl::Seq copy = m.seq_at(src);
  const int dst = add_sequence();
  Impl::Seq& q = m.seqs[static_cast<std::size_t>(dst)];
  q.blocks = copy.blocks;
  q.tokens = copy.tokens;
  q.is_output = copy.is_output;
  for (int b : q.blocks) m.refs[static_cast<std::size_t>(b)] += 1;
  const std::size_t row = static_cast<std::size_t>(m.V) * 4;
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(m.counts.p<std::int32_t>() + static_cast<std::size_t>(dst) * m.V,
                                   m.counts.p<std::int32_t>() + static_cast<std::size_t>(src) * m.V, row,
                                   cudaMemcpyDeviceToDevice, m.s()));
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(m.out_counts.p<std::int32_t>() + static_cast<std::size_t>(dst) * m.V,
                                   m.out_counts.p<std::int32_t>() + static_cast<std::size_t>(src) * m.V, row,
                                   cudaMemcpyDeviceToDevice, m.s()));
  m.sync_table(dst);
  return dst;
}

void Engine::free_sequence(int seq) {
  Impl& m = *impl_;
  Impl::Seq& q = m.seq_at(seq);
  for (int b : q.blocks) m.release_block(b);
  q = Impl::Seq{};
  m.sync_table(seq);
}

void Engine::truncate(int seq, int length) {
  Impl& m = *impl_;
  Impl::Seq& q = m.seq_at(seq);
  const int len = static_cast<int>(q.tokens.size());
  if (length < 0 || length > len) fail("truncate: gecersiz uzunluk");
  if (length == len) return;
  // ceza sayaclarini geri al
  const int n = len - length;
  std::vector<std::int32_t> tk(q.tokens.begin() + length, q.tokens.end()), sq(static_cast<std::size_t>(n), seq);
  std::vector<std::uint8_t> io(q.is_output.begin() + length, q.is_output.end());
  if (n > m.N) fail("truncate: max_tokens'tan uzun geri alma");
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(m.tok.v(), tk.data(), tk.size() * 4, cudaMemcpyHostToDevice, m.s()));
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(m.rowseq.v(), sq.data(), sq.size() * 4, cudaMemcpyHostToDevice, m.s()));
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(m.isout.v(), io.data(), io.size(), cudaMemcpyHostToDevice, m.s()));
  k::count_tokens(m.counts.p<std::int32_t>(), m.out_counts.p<std::int32_t>(), m.rowseq.p<std::int32_t>(), m.tok.p<std::int32_t>(),
                  m.isout.p<std::uint8_t>(), n, m.V, -1, m.s());
  NEXUS_CUDA_CHECK(cudaStreamSynchronize(m.s()));
  q.tokens.resize(static_cast<std::size_t>(length));
  q.is_output.resize(static_cast<std::size_t>(length));
  const int keep = (length + m.cfg.block_size - 1) / m.cfg.block_size;
  while (static_cast<int>(q.blocks.size()) > keep) {
    m.release_block(q.blocks.back());
    q.blocks.pop_back();
  }
  m.sync_table(seq);
}

int Engine::length(int seq) const { return static_cast<int>(impl_->seq_at(seq).tokens.size()); }
const std::vector<std::int32_t>& Engine::tokens(int seq) const { return impl_->seq_at(seq).tokens; }
int Engine::free_blocks() const noexcept { return static_cast<int>(impl_->free_list.size()); }
int Engine::total_blocks() const noexcept { return impl_->num_blocks - 1; }
int Engine::block_size() const noexcept { return impl_->cfg.block_size; }
int Engine::blocks_needed(int seq, int new_tokens) const {
  const Impl& m = *impl_;
  const Impl::Seq& q = m.seq_at(seq);
  const int len = static_cast<int>(q.tokens.size()), bs = m.cfg.block_size;
  int need = (len + new_tokens + bs - 1) / bs - static_cast<int>(q.blocks.size());
  if (len % bs != 0 && !q.blocks.empty() && m.refs[static_cast<std::size_t>(q.blocks.back())] > 1) need += 1;
  return std::max(0, need);
}

const float* Engine::forward(const std::vector<Chunk>& chunks, int layers) { return impl_->forward(chunks, layers); }
int Engine::logit_rows() const noexcept { return impl_->rows; }
int Engine::vocab() const noexcept { return impl_->V; }
int Engine::vocab_padded() const noexcept { return impl_->Vp; }
const StepStats& Engine::last_step() const noexcept { return impl_->stats; }
std::size_t Engine::weight_bytes() const noexcept { return impl_->wbytes; }
int Engine::layers() const noexcept { return impl_->L; }
int Engine::graphs_captured() const noexcept { return static_cast<int>(impl_->graphs.size()); }
const EngineConfig& Engine::config() const noexcept { return impl_->cfg; }

std::size_t Engine::kv_bytes_per_token() const noexcept {
  const Impl& m = *impl_;
  const std::size_t elem = m.cache == DType::I8 ? 1 : m.asz();
  return static_cast<std::size_t>(m.L) * 2 * m.H * m.D * elem + (m.cache == DType::I8 ? static_cast<std::size_t>(m.L) * 2 * m.H * 4 : 0);
}

std::vector<float> Engine::logits_host(int row) const {
  const Impl& m = *impl_;
  if (row < 0 || row >= m.rows) fail("logit satiri yok");
  std::vector<float> v(static_cast<std::size_t>(m.V));
  NEXUS_CUDA_CHECK(cudaMemcpy(v.data(), m.logits.p<float>() + static_cast<std::size_t>(row) * m.Vp, v.size() * 4,
                              cudaMemcpyDeviceToHost));
  return v;
}

std::vector<std::int32_t> Engine::sample(const std::vector<SamplingParams>& params, const std::vector<int>& seqs,
                                         std::uint64_t offset) {
  return sample(params, seqs, std::vector<std::uint64_t>(params.size(), offset));
}

std::vector<std::int32_t> Engine::sample(const std::vector<SamplingParams>& params, const std::vector<int>& seqs,
                                         const std::vector<std::uint64_t>& offsets) {
  Impl& m = *impl_;
  if (static_cast<int>(params.size()) != m.rows || seqs.size() != params.size() || offsets.size() != params.size())
    fail("sample: satir sayisi uyusmuyor");
  std::vector<k::SampleRow> rows(params.size());
  for (std::size_t i = 0; i < params.size(); ++i) {
    const SamplingParams& p = params[i];
    rows[i] = k::SampleRow{p.temperature, p.top_p, p.min_p, p.repetition_penalty, p.frequency_penalty, p.presence_penalty,
                           p.top_k, seqs[i], static_cast<unsigned long long>(p.seed), static_cast<unsigned long long>(offsets[i])};
  }
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(m.srows.v(), rows.data(), rows.size() * sizeof(k::SampleRow), cudaMemcpyHostToDevice, m.s()));
  for (int r0 = 0; r0 < m.rows; r0 += Impl::kSampleRows) {
    const int nr = std::min(Impl::kSampleRows, m.rows - r0);
    k::sample(m.sout.p<std::int32_t>() + r0, m.logits.p<float>() + static_cast<std::size_t>(r0) * m.Vp, m.sscratch.p<float>(),
              m.Vp, m.V, m.srows.p<k::SampleRow>() + r0, nr, m.counts.p<std::int32_t>(), m.out_counts.p<std::int32_t>(), m.s());
  }
  std::vector<std::int32_t> out(params.size());
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(out.data(), m.sout.v(), out.size() * 4, cudaMemcpyDeviceToHost, m.s()));
  NEXUS_CUDA_CHECK(cudaStreamSynchronize(m.s()));
  return out;
}

std::vector<float> Engine::nll(const std::vector<std::int32_t>& targets) {
  Impl& m = *impl_;
  if (static_cast<int>(targets.size()) != m.rows) fail("nll: satir sayisi uyusmuyor");
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(m.nlltg.v(), targets.data(), targets.size() * 4, cudaMemcpyHostToDevice, m.s()));
  k::row_nll(m.nllout.p<float>(), m.logits.p<float>(), m.Vp, m.V, m.nlltg.p<std::int32_t>(), m.rows, m.s());
  std::vector<float> out(targets.size());
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(out.data(), m.nllout.v(), out.size() * 4, cudaMemcpyDeviceToHost, m.s()));
  NEXUS_CUDA_CHECK(cudaStreamSynchronize(m.s()));
  return out;
}

// ==========================================================================================

std::vector<std::int32_t> generate(Engine& e, const std::vector<std::int32_t>& prompt, int max_new, const SamplingParams& sp,
                                   GenerateStats* stats) {
  const int s = e.add_sequence();
  std::vector<std::int32_t> out;
  const auto t0 = Clock::now();
  Chunk c;
  c.seq = s;
  c.tokens = prompt;
  (void)e.forward({c});
  std::int32_t t = e.sample({sp}, {s}, 0).at(0);
  out.push_back(t);
  const double ttft = ms_since(t0);
  const auto t1 = Clock::now();
  for (int i = 1; i < max_new; ++i) {
    Chunk d;
    d.seq = s;
    d.tokens = {t};
    d.output = true;
    (void)e.forward({d});
    t = e.sample({sp}, {s}, static_cast<std::uint64_t>(i)).at(0);
    out.push_back(t);
  }
  if (stats) {
    stats->ttft_ms = ttft;
    stats->tpot_ms = max_new > 1 ? ms_since(t1) / (max_new - 1) : 0.0;
    stats->tokens = max_new;
  }
  e.free_sequence(s);
  return out;
}

std::vector<std::int32_t> speculative_generate(Engine& e, const std::vector<std::int32_t>& prompt, int max_new,
                                               int draft_layers, int gamma, SpecStats* stats) {
  if (gamma < 1 || gamma + 1 > k::kSkinnyMaxRows) fail("gamma 1..15");
  if (draft_layers < 1 || draft_layers >= e.layers()) fail("taslak katman sayisi 1..L-1");
  const auto t0 = Clock::now();
  const int s = e.add_sequence();
  SamplingParams greedy;
  greedy.temperature = 0.0f;
  Chunk c;
  c.seq = s;
  c.tokens = prompt;
  (void)e.forward({c});
  std::int32_t last = e.sample({greedy}, {s}, 0).at(0);
  std::vector<std::int32_t> out{last};
  SpecStats st;
  while (static_cast<int>(out.size()) < max_new) {
    const int base = e.length(s);
    const int g = std::min(gamma, max_new - static_cast<int>(out.size()));
    std::vector<std::int32_t> draft;
    std::int32_t cur = last;
    for (int i = 0; i < g; ++i) {  // taslak: ilk k katman, tek token
      Chunk d;
      d.seq = s;
      d.tokens = {cur};
      d.output = true;
      (void)e.forward({d}, draft_layers);
      cur = e.sample({greedy}, {s}, 0).at(0);
      draft.push_back(cur);
    }
    e.truncate(s, base);
    Chunk v;  // dogrulama: [son, d1..dg] tum katmanlar, g + 1 logit
    v.seq = s;
    v.tokens.push_back(last);
    v.tokens.insert(v.tokens.end(), draft.begin(), draft.end());
    v.logits = g + 1;
    v.output = true;
    (void)e.forward({v});
    std::vector<SamplingParams> gp(static_cast<std::size_t>(g + 1), greedy);
    std::vector<int> sq(static_cast<std::size_t>(g + 1), s);
    const std::vector<std::int32_t> tgt = e.sample(gp, sq, 0);
    int acc = 0;
    while (acc < g && tgt[static_cast<std::size_t>(acc)] == draft[static_cast<std::size_t>(acc)]) ++acc;
    for (int i = 0; i < acc; ++i) out.push_back(draft[static_cast<std::size_t>(i)]);
    last = tgt[static_cast<std::size_t>(acc)];
    if (static_cast<int>(out.size()) < max_new) out.push_back(last);
    e.truncate(s, base + 1 + acc);  // reddedilen taslaklarin KV'si atilir
    st.steps += 1;
    st.drafted += g;
    st.accepted += acc;
  }
  out.resize(static_cast<std::size_t>(max_new));
  e.free_sequence(s);
  st.ms = ms_since(t0);
  st.tokens_per_step = st.steps > 0 ? static_cast<double>(max_new - 1) / st.steps : 0.0;
  if (stats) *stats = st;
  return out;
}

}  // namespace infer
}  // namespace cuda
}  // namespace nexus
