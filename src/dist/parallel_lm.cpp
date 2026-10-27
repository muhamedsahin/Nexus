/**
 * @file parallel_lm.cpp
 * @brief FAZ 12.B-12.E, 12.H: TP (+ sozluk-parcali CE), SP, async TP, PP zamanlamalari, CP, DP + ZeRO-1.
 *
 * Satir (token) duzeni: bu cp rank'inin yerel tokenlari parca-ana [nch][B][Tc] (N = B T / cp).
 * SP'de tp rank'i t ardisik [t Ns, (t + 1) Ns) satirlarini tutar (Ns = N / tp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cpu/gemm.hpp"
#include "nexus/dist/parallel_lm.hpp"

#include "nexus/attention/cpu.hpp"
#include "nexus/lm/model.hpp"
#include "nexus/lm/rope.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <numeric>

namespace nexus {
namespace dist {

namespace {

namespace K = nexus_model::kernels;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

constexpr int kTagAg = 11;
constexpr int kTagRs = 12;
constexpr std::int32_t kIgnore = -100;
constexpr float kInvSqrt2 = 0.70710678118654752f;
constexpr float kInvSqrt2Pi = 0.39894228040143268f;

// lm ops.cpp ile ayni: SiLU a s(a); GELU (erf).
template <bool Silu>
inline void act(float a, float* f, float* df) {
  if constexpr (Silu) {
    const float sg = 1.0f / (1.0f + std::exp(-a));
    *f = a * sg;
    *df = sg * (1.0f + a * (1.0f - sg));
  } else {
    const float cdf = 0.5f * (1.0f + std::erf(a * kInvSqrt2));
    *f = a * cdf;
    *df = cdf + a * kInvSqrt2Pi * std::exp(-0.5f * a * a);
  }
}

void norm_forward(bool rms, const float* x, const float* g, const float* b, float* y, float* mean, float* rstd,
                  std::int64_t rows, int c, float eps) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * c;
    float* yr = y + r * c;
    double mu = 0.0;
    if (!rms) {
      for (int i = 0; i < c; ++i) mu += xr[i];
      mu /= c;
    }
    double var = 0.0;
    for (int i = 0; i < c; ++i) var += (xr[i] - mu) * (xr[i] - mu);
    var /= c;
    const float rs = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mu);
    for (int i = 0; i < c; ++i) {
      const float xh = (xr[i] - m) * rs;
      yr[i] = b != nullptr ? xh * g[i] + b[i] : xh * g[i];
    }
    mean[r] = m;
    rstd[r] = rs;
  }
}

/// dx += norm_bwd(dy); dg/db += (gradyanlar).
void norm_backward(bool rms, const float* x, const float* g, const float* mean, const float* rstd, const float* dy,
                   float* dx, float* dg, float* db, std::int64_t rows, int c) {
  std::vector<float> dxh(static_cast<std::size_t>(c));
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * c;
    const float* dyr = dy + r * c;
    float* dxr = dx + r * c;
    const float m = mean[r];
    const float rs = rstd[r];
    double s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < c; ++i) {
      const float xh = (xr[i] - m) * rs;
      dxh[static_cast<std::size_t>(i)] = dyr[i] * g[i];
      s1 += dxh[static_cast<std::size_t>(i)];
      s2 += static_cast<double>(dxh[static_cast<std::size_t>(i)]) * xh;
      dg[i] += dyr[i] * xh;
      if (db != nullptr) db[i] += dyr[i];
    }
    const float a1 = rms ? 0.0f : static_cast<float>(s1 / c);
    const float a2 = static_cast<float>(s2 / c);
    for (int i = 0; i < c; ++i) {
      const float xh = (xr[i] - m) * rs;
      dxr[i] += rs * (dxh[static_cast<std::size_t>(i)] - a1 - xh * a2);
    }
  }
}

void add_bias(float* y, const float* b, std::int64_t rows, int n) {
  for (std::int64_t r = 0; r < rows; ++r) {
    float* yr = y + r * n;
    for (int i = 0; i < n; ++i) yr[i] += b[i];
  }
}

void col_sum(const float* dy, float* db, std::int64_t rows, int n) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* d = dy + r * n;
    for (int i = 0; i < n; ++i) db[i] += d[i];
  }
}

std::uint64_t vec_bytes(const std::vector<float>& v) { return v.size() * sizeof(float); }

}  // namespace

// ==========================================================================================
// Impl
// ==========================================================================================

struct LayerIdx {
  int n1w = -1, n1b = -1, qkvw = -1, qkvb = -1, ow = -1, ob = -1, n2w = -1, n2b = -1, upw = -1, upb = -1, dw = -1,
      db = -1;
};

struct ChunkDef {
  int g = 0;
  int l0 = 0, l1 = 0;
  bool embed = false, final = false;
  int wte = -1, wpe = -1, nfw = -1, nfb = -1, head = -1;
};

struct LayerAct {
  std::vector<float> x, m1, s1, hfull, q, k, v, o, lse, r, m2, s2, h2full, up, actv;
  std::vector<float> ddown, dup, dofull, dqkv;  // W (agirlik gradyani) icin
};

struct ChunkAct {
  std::vector<LayerAct> layers;
  std::vector<float> xf, mf, sf, hffull, dlogits;
  std::vector<float> demb_rows, demb_full;
  [[nodiscard]] std::uint64_t bytes() const {
    std::uint64_t b = vec_bytes(xf) + vec_bytes(mf) + vec_bytes(sf) + vec_bytes(hffull) + vec_bytes(dlogits) +
                      vec_bytes(demb_rows) + vec_bytes(demb_full);
    for (const LayerAct& l : layers) {
      for (const std::vector<float>* v : {&l.x, &l.m1, &l.s1, &l.hfull, &l.q, &l.k, &l.v, &l.o, &l.lse, &l.r, &l.m2,
                                          &l.s2, &l.h2full, &l.up, &l.actv, &l.ddown, &l.dup, &l.dofull, &l.dqkv}) {
        b += vec_bytes(*v);
      }
    }
    return b;
  }
};

struct MbInfo {
  std::vector<std::int32_t> ids, targets;
  std::vector<int> pos;
  std::int64_t valid = 0;  // tum mikro yigindaki gecerli hedef sayisi (tum cp rank'leri)
  float weight = 0.0f;
};

struct ParallelLm::Impl {
  ParallelLm* self = nullptr;
  // koordinatlar
  int t = 0, T = 1, ppr = 0, PP = 1, c = 0, C = 1, V = 1;
  GroupPtr tp, pp, cp, dp, dpcp, tpp, tied, kv;
  bool kv_first = true;
  bool kv_replicated = false;
  std::vector<GroupPtr> fwd_edge, bwd_edge;  // a -> a+1 / a -> a-1 (pp grubunda 2'lik)
  // boyutlar
  int d = 0, H = 0, Hk = 0, D = 0, F = 0, Hl = 0, Hkl = 0, Fl = 0, upw = 0, qkvw = 0, Vp = 0, Vs = 0, v0 = 0, vocab = 0;
  bool rms = false, gated = false, silu = false, learned_pos = false, rope = false, tie = false;
  float eps = 1e-5f;
  std::shared_ptr<lm::RopeCache> rope_cache;
  // parcalar
  std::vector<ChunkDef> chunks;  // tum global parcalar (pp * v)
  std::vector<int> my_chunks;    // global kimlik
  std::map<int, LayerIdx> layers;
  // calisma sekli (prepare)
  int B = 0, Tseq = 0, Tc = 0, nch = 1;
  std::int64_t N = 0, R = 0, Ns = 0;
  std::vector<int> chunk_ids;
  // duz gradyan / optimizer
  std::vector<std::size_t> offset;
  std::size_t total = 0, padded = 0, shard = 0, shard_lo = 0;
  int G = 1, gr = 0;  // dp x cp grup boyu ve rank'i
  std::vector<float> gflat, gshard;
  std::vector<float> m, v;  // zero 0: total; zero 1: shard
  std::vector<std::uint8_t> norm_mask;  // duz indeks basina norma katilim
  std::vector<float> losses;
  std::map<std::pair<int, int>, ChunkAct> acts;
  std::uint64_t act_bytes = 0, act_peak = 0;

  [[nodiscard]] int owner(int g) const noexcept { return g % PP; }
  [[nodiscard]] int last_chunk() const noexcept { return PP * V - 1; }
  [[nodiscard]] Param& P(int i) { return self->params_[static_cast<std::size_t>(i)]; }
  [[nodiscard]] float* W(int i) { return i < 0 ? nullptr : self->params_[static_cast<std::size_t>(i)].value.data(); }
  [[nodiscard]] float* Gd(int i) { return i < 0 ? nullptr : self->params_[static_cast<std::size_t>(i)].grad.data(); }
  [[nodiscard]] bool sp() const noexcept { return self->cfg_.sp && T > 1; }
  [[nodiscard]] bool async_tp() const noexcept { return sp() && self->cfg_.async_tp; }

  // ---- tp iletisim yardimcilari ---------------------------------------------------------
  void tp_allreduce(float* p, std::size_t n) {
    if (T > 1 && n > 0) tp->all_reduce(p, n, DType::F32, ReduceOp::Sum)->wait();
  }

  /// shard [R, w] -> full [N, w]; compute(src, satirlar) her parca icin (async: halka ortusmeli).
  void gather_rows(const float* part, int w, std::vector<float>& full,
                   const std::function<void(int src, const float* rows)>& compute) {
    full.resize(static_cast<std::size_t>(N) * w);
    const std::size_t chunk = static_cast<std::size_t>(Ns) * w;
    if (!sp()) {
      std::memcpy(full.data(), part, static_cast<std::size_t>(N) * w * sizeof(float));
      if (compute) compute(-1, full.data());
      return;
    }
    if (!async_tp()) {
      tp->all_gather(part, full.data(), chunk, DType::F32)->wait();
      if (compute) compute(-1, full.data());
      return;
    }
    std::vector<float> cur(part, part + chunk), next(chunk);
    for (int s = 0; s < T; ++s) {
      const int src = (t - s + T) % T;
      WorkPtr xw;  // parca s+1 halkada ilerlerken parca s hesaplanir
      if (s + 1 < T) xw = tp->send_recv(cur.data(), chunk, (t + 1) % T, next.data(), chunk, (t - 1 + T) % T, DType::F32, kTagAg);
      if (compute) compute(src, cur.data());
      std::memcpy(full.data() + static_cast<std::size_t>(src) * chunk, cur.data(), chunk * sizeof(float));
      if (xw) {
        xw->wait();
        cur.swap(next);
      }
    }
  }

  /// partial(row0, rows, out): [rows, w] kismi toplam (out sifirlanmis gelir). Sonuc [R, w].
  void reduce_rows(int w, float* out, const std::function<void(std::int64_t row0, std::int64_t rows, float* dst)>& partial) {
    if (!sp()) {
      std::fill(out, out + static_cast<std::size_t>(N) * w, 0.0f);
      partial(0, N, out);
      tp_allreduce(out, static_cast<std::size_t>(N) * w);
      return;
    }
    const std::size_t chunk = static_cast<std::size_t>(Ns) * w;
    if (!async_tp()) {
      std::vector<float> full(static_cast<std::size_t>(N) * w, 0.0f);
      partial(0, N, full.data());
      tp->reduce_scatter(full.data(), out, chunk, DType::F32, ReduceOp::Sum)->wait();
      return;
    }
    // Halka: adim s'de parca c = t - 1 - s'nin kismi toplami hesaplanirken onceki adimin biriktiricisi
    // (sol komsudan) gelir; toplanip saga gecer. Son adimda c = t (kendi parcam).
    std::vector<float> tmp(chunk), acc(chunk), in(chunk);
    WorkPtr xw;
    for (int s = 0; s < T; ++s) {
      const int cidx = ((t - 1 - s) % T + 2 * T) % T;
      std::fill(tmp.begin(), tmp.end(), 0.0f);
      partial(static_cast<std::int64_t>(cidx) * Ns, Ns, tmp.data());
      if (xw) {
        xw->wait();  // in: sol komsunun biriktiricisi; acc gonderimi bitti
        for (std::size_t e = 0; e < chunk; ++e) acc[e] = in[e] + tmp[e];
      } else {
        acc = tmp;
      }
      if (s + 1 < T) {
        xw = tp->send_recv(acc.data(), chunk, (t + 1) % T, in.data(), chunk, (t - 1 + T) % T, DType::F32, kTagRs);
      } else {
        std::memcpy(out, acc.data(), chunk * sizeof(float));
      }
    }
  }

  // ---- hazirlik ----------------------------------------------------------------------------
  void prepare(int batch, int seq) {
    if (batch == B && seq == Tseq) return;
    const int n = cp_num_chunks(C, self->cfg_.cp_layout);
    if (C > 1 && seq % n != 0) {
      throw DistError("Nexus ParallelLm: T = " + std::to_string(seq) + " cp parca sayisina (" + std::to_string(n) +
                      ") bolunmeli");
    }
    B = batch;
    Tseq = seq;
    chunk_ids = C > 1 ? cp_chunks(C, c, self->cfg_.cp_layout) : std::vector<int>{0};
    nch = static_cast<int>(chunk_ids.size());
    Tc = C > 1 ? seq / n : seq;
    N = static_cast<std::int64_t>(nch) * B * Tc;
    if (sp() && N % T != 0) {
      throw DistError("Nexus ParallelLm: SP icin yerel token sayisi (" + std::to_string(N) + ") tp'ye bolunmeli");
    }
    Ns = sp() ? N / T : N;
    R = Ns;
    if (rope) rope_cache->ensure(seq);
    if (learned_pos && seq > self->lmc_.max_position) throw DistError("Nexus ParallelLm: T > max_position");
  }

  MbInfo mb_info(const std::int32_t* tokens, float weight) const {
    MbInfo mb;
    mb.weight = weight;
    mb.ids.resize(static_cast<std::size_t>(N));
    mb.targets.resize(static_cast<std::size_t>(N));
    mb.pos.resize(static_cast<std::size_t>(N));
    std::size_t row = 0;
    for (int ch : chunk_ids) {
      for (int b = 0; b < B; ++b) {
        for (int tt = 0; tt < Tc; ++tt, ++row) {
          const int p = ch * Tc + tt;
          const std::int32_t id = tokens[static_cast<std::size_t>(b) * Tseq + p];
          if (id < 0 || id >= vocab) throw DistError("Nexus ParallelLm: token sozluk disinda");
          mb.ids[row] = id;
          mb.pos[row] = p;
          mb.targets[row] = p + 1 < Tseq ? tokens[static_cast<std::size_t>(b) * Tseq + p + 1] : kIgnore;
        }
      }
    }
    for (int b = 0; b < B; ++b) {
      for (int p = 1; p < Tseq; ++p) mb.valid += tokens[static_cast<std::size_t>(b) * Tseq + p] != kIgnore ? 1 : 0;
    }
    return mb;
  }

  /// SP'de rank'in satiri i -> yerel N satir indeksi.
  [[nodiscard]] std::int64_t row_of(std::int64_t i) const noexcept { return sp() ? static_cast<std::int64_t>(t) * Ns + i : i; }

  // ---- dikkat (yerel / CP) -----------------------------------------------------------------
  void attention_forward(const float* q, const float* k, const float* vv, float* o, std::vector<float>& lse) {
    if (C == 1) {
      AttentionParams p;
      p.batch = B;
      p.heads = Hl;
      p.kv_heads = Hkl;
      p.tq = Tseq;
      p.tk = Tseq;
      p.head_dim = D;
      p.dtype = DType::F32;
      p.causal = true;
      p.set_contiguous_bthd();
      lse.resize(static_cast<std::size_t>(B) * Hl * Tseq);
      attention::flash_attention_forward_cpu(o, lse.data(), q, k, vv, p, self->cfg_.attention_threads);
      return;
    }
    CpAttention a{B, Tseq, Hl, Hkl, D, 0.0f, self->cfg_.cp_layout, self->cfg_.attention_threads};
    if (self->cfg_.cp_algorithm == CpAlgorithm::Ring) {
      lse.resize(static_cast<std::size_t>(nch) * B * Hl * Tc);
      ring_attention_forward(*cp, a, q, k, vv, o, lse.data());
    } else {
      lse.resize(ulysses_lse_size(a, C));
      ulysses_attention_forward(*cp, a, q, k, vv, o, lse.data());
    }
  }

  void attention_backward(const LayerAct& s, const float* dout, float* dq, float* dk, float* dv) {
    if (C == 1) {
      AttentionParams p;
      p.batch = B;
      p.heads = Hl;
      p.kv_heads = Hkl;
      p.tq = Tseq;
      p.tk = Tseq;
      p.head_dim = D;
      p.dtype = DType::F32;
      p.causal = true;
      p.set_contiguous_bthd();
      attention::flash_attention_backward_cpu(dq, dk, dv, dout, s.q.data(), s.k.data(), s.v.data(), s.o.data(),
                                              s.lse.data(), p, self->cfg_.attention_threads);
      return;
    }
    CpAttention a{B, Tseq, Hl, Hkl, D, 0.0f, self->cfg_.cp_layout, self->cfg_.attention_threads};
    if (self->cfg_.cp_algorithm == CpAlgorithm::Ring) {
      ring_attention_backward(*cp, a, s.q.data(), s.k.data(), s.v.data(), s.o.data(), s.lse.data(), dout, dq, dk, dv);
    } else {
      ulysses_attention_backward(*cp, a, s.q.data(), s.k.data(), s.v.data(), s.o.data(), s.lse.data(), dout, dq, dk, dv);
    }
  }

  // ---- katman ileri / geri -----------------------------------------------------------------

  /// Column-parallel: x [R, in] -> y [N, out] (+ b), xfull [N, in] saklanir.
  void col_forward(const float* x, int in, std::vector<float>& xfull, const float* w, const float* b, int out,
                   std::vector<float>& y) {
    y.assign(static_cast<std::size_t>(N) * out, 0.0f);
    gather_rows(x, in, xfull, [&](int src, const float* rows) {
      if (src < 0) {
        nexus::cpu::gemm_nn(rows, w, y.data(), static_cast<int>(N), in, out);
      } else {
        nexus::cpu::gemm_nn(rows, w, y.data() + static_cast<std::size_t>(src) * Ns * out, static_cast<int>(Ns), in, out);
      }
    });
    if (b != nullptr) add_bias(y.data(), b, N, out);
  }

  /// Row-parallel: a [N, in] -> y [R, out] (indirgenmis) + b.
  void row_forward(const float* a, int in, const float* w, const float* b, int out, float* y) {
    reduce_rows(out, y, [&](std::int64_t row0, std::int64_t rows, float* dst) {
      nexus::cpu::gemm_nn(a + row0 * in, w, dst, static_cast<int>(rows), in, out);
    });
    if (b != nullptr) add_bias(y, b, R, out);
  }

  /// Column-parallel geri (giris yolu): dy [N, out] -> dx [R, in] (indirgenmis).
  void col_backward_input(const float* dy, int out, const float* w, int in, float* dx) {
    reduce_rows(in, dx, [&](std::int64_t row0, std::int64_t rows, float* dst) {
      nexus::cpu::gemm_nt_accum(dy + row0 * out, w, dst, static_cast<int>(rows), out, in);
    });
  }

  /// Row-parallel geri (giris yolu): dz [R, out] -> dzfull [N, out] (saklanir), da [N, in].
  void row_backward_input(const float* dz, int out, std::vector<float>& dzfull, const float* w, int in,
                          std::vector<float>& da) {
    da.assign(static_cast<std::size_t>(N) * in, 0.0f);
    gather_rows(dz, out, dzfull, [&](int src, const float* rows) {
      if (src < 0) {
        nexus::cpu::gemm_nt_accum(rows, w, da.data(), static_cast<int>(N), out, in);
      } else {
        nexus::cpu::gemm_nt_accum(rows, w, da.data() + static_cast<std::size_t>(src) * Ns * in, static_cast<int>(Ns), out, in);
      }
    });
  }

  void layer_forward(int l, std::vector<float>& x, LayerAct& s, const MbInfo& mb) {
    const LayerIdx& L = layers.at(l);
    s.x = x;
    std::vector<float> h(static_cast<std::size_t>(R) * d);
    s.m1.resize(static_cast<std::size_t>(R));
    s.s1.resize(static_cast<std::size_t>(R));
    norm_forward(rms, x.data(), W(L.n1w), W(L.n1b), h.data(), s.m1.data(), s.s1.data(), R, d, eps);
    std::vector<float> qkv;
    col_forward(h.data(), d, s.hfull, W(L.qkvw), W(L.qkvb), qkvw, qkv);
    s.q.resize(static_cast<std::size_t>(N) * Hl * D);
    s.k.resize(static_cast<std::size_t>(N) * Hkl * D);
    s.v.resize(static_cast<std::size_t>(N) * Hkl * D);
    for (std::int64_t r = 0; r < N; ++r) {
      const float* src = qkv.data() + r * qkvw;
      std::memcpy(s.q.data() + r * Hl * D, src, static_cast<std::size_t>(Hl) * D * sizeof(float));
      std::memcpy(s.k.data() + r * Hkl * D, src + Hl * D, static_cast<std::size_t>(Hkl) * D * sizeof(float));
      std::memcpy(s.v.data() + r * Hkl * D, src + (Hl + Hkl) * D, static_cast<std::size_t>(Hkl) * D * sizeof(float));
    }
    if (rope) {
      lm::rope_apply(s.q.data(), static_cast<int>(N), Hl, D, mb.pos.data(), *rope_cache, false);
      lm::rope_apply(s.k.data(), static_cast<int>(N), Hkl, D, mb.pos.data(), *rope_cache, false);
    }
    s.o.resize(s.q.size());
    attention_forward(s.q.data(), s.k.data(), s.v.data(), s.o.data(), s.lse);
    std::vector<float> ao(static_cast<std::size_t>(R) * d);
    row_forward(s.o.data(), Hl * D, W(L.ow), W(L.ob), d, ao.data());
    s.r.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) s.r[i] = x[i] + ao[i];
    std::vector<float> h2(s.r.size());
    s.m2.resize(static_cast<std::size_t>(R));
    s.s2.resize(static_cast<std::size_t>(R));
    norm_forward(rms, s.r.data(), W(L.n2w), W(L.n2b), h2.data(), s.m2.data(), s.s2.data(), R, d, eps);
    col_forward(h2.data(), d, s.h2full, W(L.upw), W(L.upb), upw, s.up);
    s.actv.resize(static_cast<std::size_t>(N) * Fl);
    for (std::int64_t r = 0; r < N; ++r) {
      const float* a = s.up.data() + r * upw;
      float* y = s.actv.data() + r * Fl;
      for (int i = 0; i < Fl; ++i) {
        float f = 0.0f, df = 0.0f;
        if (silu) {
          act<true>(a[i], &f, &df);
        } else {
          act<false>(a[i], &f, &df);
        }
        y[i] = gated ? f * a[Fl + i] : f;
      }
    }
    std::vector<float> dn(static_cast<std::size_t>(R) * d);
    row_forward(s.actv.data(), Fl, W(L.dw), W(L.db), d, dn.data());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = s.r[i] + dn[i];
  }

  /// dx [R, d]: giris = cikis gradyani, donus = giris gradyani (yerinde).
  void layer_backward_input(int l, std::vector<float>& dx, LayerAct& s) {
    const LayerIdx& L = layers.at(l);
    // MLP
    std::vector<float> dact;
    row_backward_input(dx.data(), d, s.ddown, W(L.dw), Fl, dact);
    s.dup.resize(static_cast<std::size_t>(N) * upw);
    for (std::int64_t r = 0; r < N; ++r) {
      const float* a = s.up.data() + r * upw;
      const float* gy = dact.data() + r * Fl;
      float* da = s.dup.data() + r * upw;
      for (int i = 0; i < Fl; ++i) {
        float f = 0.0f, df = 0.0f;
        if (silu) {
          act<true>(a[i], &f, &df);
        } else {
          act<false>(a[i], &f, &df);
        }
        if (gated) {
          da[i] = gy[i] * a[Fl + i] * df;
          da[Fl + i] = gy[i] * f;
        } else {
          da[i] = gy[i] * df;
        }
      }
    }
    std::vector<float> dh2(static_cast<std::size_t>(R) * d);
    col_backward_input(s.dup.data(), upw, W(L.upw), d, dh2.data());
    std::vector<float> dr = dx;
    norm_backward(rms, s.r.data(), W(L.n2w), s.m2.data(), s.s2.data(), dh2.data(), dr.data(), Gd(L.n2w), Gd(L.n2b), R, d);
    // dikkat
    std::vector<float> dov;
    row_backward_input(dr.data(), d, s.dofull, W(L.ow), Hl * D, dov);
    std::vector<float> dq(s.q.size()), dk(s.k.size()), dv(s.v.size());
    attention_backward(s, dov.data(), dq.data(), dk.data(), dv.data());
    if (rope) {
      lm::rope_apply(dq.data(), static_cast<int>(N), Hl, D, cur_pos, *rope_cache, true);
      lm::rope_apply(dk.data(), static_cast<int>(N), Hkl, D, cur_pos, *rope_cache, true);
    }
    s.dqkv.resize(static_cast<std::size_t>(N) * qkvw);
    for (std::int64_t r = 0; r < N; ++r) {
      float* dst = s.dqkv.data() + r * qkvw;
      std::memcpy(dst, dq.data() + r * Hl * D, static_cast<std::size_t>(Hl) * D * sizeof(float));
      std::memcpy(dst + Hl * D, dk.data() + r * Hkl * D, static_cast<std::size_t>(Hkl) * D * sizeof(float));
      std::memcpy(dst + (Hl + Hkl) * D, dv.data() + r * Hkl * D, static_cast<std::size_t>(Hkl) * D * sizeof(float));
    }
    std::vector<float> dh(static_cast<std::size_t>(R) * d);
    col_backward_input(s.dqkv.data(), qkvw, W(L.qkvw), d, dh.data());
    dx = dr;
    norm_backward(rms, s.x.data(), W(L.n1w), s.m1.data(), s.s1.data(), dh.data(), dx.data(), Gd(L.n1w), Gd(L.n1b), R, d);
    // W icin gereksizler
    s.x.clear();
    s.r.clear();
    s.q.clear();
    s.k.clear();
    s.v.clear();
    s.lse.clear();
    s.up.clear();
    s.m1.clear();
    s.s1.clear();
    s.m2.clear();
    s.s2.clear();
  }

  void layer_backward_weight(int l, LayerAct& s) {
    const LayerIdx& L = layers.at(l);
    nexus::cpu::gemm_tn_accum(s.actv.data(), s.ddown.data(), Gd(L.dw), Fl, static_cast<int>(N), d);
    if (L.db >= 0) col_sum(s.ddown.data() + row_of(0) * d, Gd(L.db), R, d);
    nexus::cpu::gemm_tn_accum(s.h2full.data(), s.dup.data(), Gd(L.upw), d, static_cast<int>(N), upw);
    if (L.upb >= 0) col_sum(s.dup.data(), Gd(L.upb), N, upw);
    nexus::cpu::gemm_tn_accum(s.o.data(), s.dofull.data(), Gd(L.ow), Hl * D, static_cast<int>(N), d);
    if (L.ob >= 0) col_sum(s.dofull.data() + row_of(0) * d, Gd(L.ob), R, d);
    nexus::cpu::gemm_tn_accum(s.hfull.data(), s.dqkv.data(), Gd(L.qkvw), d, static_cast<int>(N), qkvw);
    if (L.qkvb >= 0) col_sum(s.dqkv.data(), Gd(L.qkvb), N, qkvw);
    s = LayerAct{};
  }

  // ---- gomme / son ---------------------------------------------------------------------------
  void embed_forward(const ChunkDef& ch, const MbInfo& mb, std::vector<float>& x) {
    x.assign(static_cast<std::size_t>(R) * d, 0.0f);
    reduce_rows(d, x.data(), [&](std::int64_t row0, std::int64_t rows, float* dst) {
      const float* wte = W(ch.wte);
      for (std::int64_t i = 0; i < rows; ++i) {
        const std::int64_t id = mb.ids[static_cast<std::size_t>(row0 + i)] - v0;
        if (id < 0 || id >= Vs) continue;  // sozluk parcasi disi: 0 (Partial)
        std::memcpy(dst + i * d, wte + id * d, static_cast<std::size_t>(d) * sizeof(float));
      }
    });
    if (ch.wpe >= 0) {
      const float* wpe = W(ch.wpe);
      for (std::int64_t i = 0; i < R; ++i) {
        const int p = mb.pos[static_cast<std::size_t>(row_of(i))];
        float* xr = x.data() + i * d;
        for (int e = 0; e < d; ++e) xr[e] += wpe[static_cast<std::size_t>(p) * d + e];
      }
    }
  }

  void embed_backward_input(const std::vector<float>& dx, ChunkAct& a) {
    a.demb_rows = dx;
    gather_rows(dx.data(), d, a.demb_full, nullptr);  // W yerel kalsin diye toplama burada
  }

  void embed_backward_weight(const ChunkDef& ch, const MbInfo& mb, ChunkAct& a) {
    if (ch.wpe >= 0) {
      float* g = Gd(ch.wpe);
      for (std::int64_t i = 0; i < R; ++i) {
        const int p = mb.pos[static_cast<std::size_t>(row_of(i))];
        const float* dr = a.demb_rows.data() + i * d;
        for (int e = 0; e < d; ++e) g[static_cast<std::size_t>(p) * d + e] += dr[e];
      }
    }
    float* g = Gd(ch.wte);
    for (std::int64_t r = 0; r < N; ++r) {
      const std::int64_t id = mb.ids[static_cast<std::size_t>(r)] - v0;
      if (id < 0 || id >= Vs) continue;
      const float* dr = a.demb_full.data() + r * d;
      float* grow = g + id * d;
      for (int e = 0; e < d; ++e) grow[e] += dr[e];
    }
    a.demb_rows.clear();
    a.demb_full.clear();
  }

  /// Son norm + sozluk-parcali kafa + CE. dlogits saklanir. Donus: yerel satirlarin kayip toplami.
  double final_forward(const ChunkDef& ch, const std::vector<float>& x, ChunkAct& a, const MbInfo& mb, bool grad) {
    a.xf = x;
    std::vector<float> hf(x.size());
    a.mf.resize(static_cast<std::size_t>(R));
    a.sf.resize(static_cast<std::size_t>(R));
    norm_forward(rms, x.data(), W(ch.nfw), W(ch.nfb), hf.data(), a.mf.data(), a.sf.data(), R, d, eps);
    std::vector<float> logits(static_cast<std::size_t>(N) * Vs, 0.0f);
    const float* head = W(ch.head);
    gather_rows(hf.data(), d, a.hffull, [&](int src, const float* rows) {
      if (src < 0) {
        nexus::cpu::gemm_nt_accum(rows, head, logits.data(), static_cast<int>(N), d, Vs);
      } else {
        nexus::cpu::gemm_nt_accum(rows, head, logits.data() + static_cast<std::size_t>(src) * Ns * Vs, static_cast<int>(Ns), d, Vs);
      }
    });
    const int valid_cols = std::clamp(vocab - v0, 0, Vs);  // dolgu sutunlari -inf (yok sayilir)
    std::vector<float> mx(static_cast<std::size_t>(N), -std::numeric_limits<float>::infinity());
    for (std::int64_t r = 0; r < N; ++r) {
      const float* lr = logits.data() + r * Vs;
      float mrow = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < valid_cols; ++j) mrow = std::max(mrow, lr[j]);
      mx[static_cast<std::size_t>(r)] = mrow;
    }
    if (T > 1) tp->all_reduce(mx.data(), mx.size(), DType::F32, ReduceOp::Max)->wait();
    std::vector<float> st(2 * static_cast<std::size_t>(N), 0.0f);  // [sum-exp | hedef logit]
    for (std::int64_t r = 0; r < N; ++r) {
      const float* lr = logits.data() + r * Vs;
      const float mrow = mx[static_cast<std::size_t>(r)];
      float s = 0.0f;
      for (int j = 0; j < valid_cols; ++j) s += std::exp(lr[j] - mrow);
      st[static_cast<std::size_t>(r)] = s;
      const std::int64_t tg = mb.targets[static_cast<std::size_t>(r)] - v0;
      if (mb.targets[static_cast<std::size_t>(r)] != kIgnore && tg >= 0 && tg < valid_cols) {
        st[static_cast<std::size_t>(N + r)] = lr[tg];
      }
    }
    tp_allreduce(st.data(), st.size());
    double loss = 0.0;
    const float scale = mb.valid > 0 ? mb.weight / static_cast<float>(mb.valid) : 0.0f;
    if (grad) a.dlogits.assign(logits.size(), 0.0f);
    for (std::int64_t r = 0; r < N; ++r) {
      if (mb.targets[static_cast<std::size_t>(r)] == kIgnore) continue;
      const float mrow = mx[static_cast<std::size_t>(r)];
      const float s = st[static_cast<std::size_t>(r)];
      loss += static_cast<double>(std::log(s) + mrow - st[static_cast<std::size_t>(N + r)]);
      if (!grad) continue;
      const float* lr = logits.data() + r * Vs;
      float* dl = a.dlogits.data() + r * Vs;
      const float inv = 1.0f / s;
      for (int j = 0; j < valid_cols; ++j) dl[j] = std::exp(lr[j] - mrow) * inv * scale;
      const std::int64_t tg = mb.targets[static_cast<std::size_t>(r)] - v0;
      if (tg >= 0 && tg < valid_cols) dl[tg] -= scale;
    }
    return loss;
  }

  void final_backward_input(const ChunkDef& ch, std::vector<float>& dx, ChunkAct& a) {
    std::vector<float> dhf(static_cast<std::size_t>(R) * d);
    const float* head = W(ch.head);
    reduce_rows(d, dhf.data(), [&](std::int64_t row0, std::int64_t rows, float* dst) {
      nexus::cpu::gemm_nn(a.dlogits.data() + row0 * Vs, head, dst, static_cast<int>(rows), Vs, d);
    });
    dx.assign(static_cast<std::size_t>(R) * d, 0.0f);
    norm_backward(rms, a.xf.data(), W(ch.nfw), a.mf.data(), a.sf.data(), dhf.data(), dx.data(), Gd(ch.nfw), Gd(ch.nfb), R, d);
    a.xf.clear();
  }

  void final_backward_weight(const ChunkDef& ch, ChunkAct& a) {
    nexus::cpu::gemm_tn_accum(a.dlogits.data(), a.hffull.data(), Gd(ch.head), Vs, static_cast<int>(N), d);
    a.dlogits.clear();
    a.hffull.clear();
  }

  // ---- parca ileri / geri ------------------------------------------------------------------
  const int* cur_pos = nullptr;  // geri gecis RoPE konumlari (aktif mikro yigin)

  /// x: giris (parca 0 haric) -> cikis (son parca haric). Son parcada kayip toplami doner.
  double chunk_forward(int mb_i, int g, std::vector<float>& x, const MbInfo& mb, bool grad) {
    const ChunkDef& ch = chunks[static_cast<std::size_t>(g)];
    ChunkAct& a = acts[{mb_i, g}];
    a.layers.resize(static_cast<std::size_t>(ch.l1 - ch.l0));
    if (ch.embed) embed_forward(ch, mb, x);
    for (int l = ch.l0; l < ch.l1; ++l) layer_forward(l, x, a.layers[static_cast<std::size_t>(l - ch.l0)], mb);
    double loss = 0.0;
    if (ch.final) loss = final_forward(ch, x, a, mb, grad);
    track();
    return loss;
  }

  void chunk_backward_input(int mb_i, int g, std::vector<float>& dx, const MbInfo& mb) {
    const ChunkDef& ch = chunks[static_cast<std::size_t>(g)];
    ChunkAct& a = acts.at({mb_i, g});
    cur_pos = mb.pos.data();
    if (ch.final) final_backward_input(ch, dx, a);
    for (int l = ch.l1 - 1; l >= ch.l0; --l) layer_backward_input(l, dx, a.layers[static_cast<std::size_t>(l - ch.l0)]);
    if (ch.embed) embed_backward_input(dx, a);
    track();
  }

  void chunk_backward_weight(int mb_i, int g, const MbInfo& mb) {
    const ChunkDef& ch = chunks[static_cast<std::size_t>(g)];
    auto it = acts.find({mb_i, g});
    ChunkAct& a = it->second;
    if (ch.final) final_backward_weight(ch, a);
    for (int l = ch.l1 - 1; l >= ch.l0; --l) layer_backward_weight(l, a.layers[static_cast<std::size_t>(l - ch.l0)]);
    if (ch.embed) embed_backward_weight(ch, mb, a);
    acts.erase(it);
    track();
  }

  void track() {
    std::uint64_t b = 0;
    for (const auto& entry : acts) b += entry.second.bytes();
    act_bytes = b;
    act_peak = std::max(act_peak, b);
  }
};

// ==========================================================================================
// Kurulum
// ==========================================================================================

namespace {

void check_supported(const lm::LmConfig& c) {
  auto no = [](const char* what) { throw DistError(std::string("Nexus ParallelLm: desteklenmiyor: ") + what); };
  if (c.init == lm::InitKind::MuP) no("muP");
  if (!c.prenorm) no("son-normalizasyon (prenorm = false)");
  if (c.parallel_block) no("paralel blok");
  if (c.window > 0) no("kayan pencere");
  if (c.attn_softcap > 0.0f) no("dikkat softcap");
  if (c.resid_dropout > 0.0f || c.attn_dropout > 0.0f) no("dropout");
  if (c.qk_norm) no("qk_norm");
  if (c.pos == lm::PosKind::Alibi || c.pos == lm::PosKind::Sinusoidal) no("ALiBi / sinuzoidal konum");
  if (c.recompute != lm::RecomputeKind::None) no("yeniden hesaplama (recompute)");
  if (c.moe_experts > 0) no("MoE (uzman paralelligi: dist::MoeExpertParallel)");
}

}  // namespace

ParallelLm::ParallelLm(const lm::LanguageModel& reference, GroupPtr world, ParallelConfig config)
    : cfg_(config), lmc_(reference.config()), impl_(std::make_unique<Impl>()) {
  check_supported(lmc_);
  Impl& I = *impl_;
  I.self = this;
  const lm::LmConfig& c = lmc_;
  if (cfg_.dp * cfg_.tp * cfg_.pp * cfg_.cp != world->size()) {
    throw DistError("Nexus ParallelLm: dp * tp * pp * cp = " + std::to_string(cfg_.dp * cfg_.tp * cfg_.pp * cfg_.cp) +
                    " != world " + std::to_string(world->size()));
  }
  if (cfg_.zero != 0 && cfg_.zero != 1) {
    throw DistError("Nexus ParallelLm: TP/PP/CP ile zero 0 ya da 1 (ZeRO-2/3 saf DP'de FsdpEngine ile)");
  }
  if (cfg_.sp && cfg_.tp < 2) throw DistError("Nexus ParallelLm: sp = true tp > 1 ister");
  if (cfg_.async_tp && !cfg_.sp) throw DistError("Nexus ParallelLm: async_tp sp = true ister");
  if (cfg_.virtual_stages < 1) throw DistError("Nexus ParallelLm: virtual_stages >= 1");
  if (cfg_.virtual_stages > 1 && cfg_.pp_schedule != PipeSchedule::Interleaved1F1B) {
    throw DistError("Nexus ParallelLm: virtual_stages > 1 yalniz interleaved_1f1b ile");
  }
  mesh_ = std::make_unique<DeviceMesh>(world, std::vector<int>{cfg_.dp, cfg_.pp, cfg_.cp, cfg_.tp},
                                       std::vector<std::string>{"dp", "pp", "cp", "tp"});
  I.t = mesh_->coordinate("tp");
  I.T = cfg_.tp;
  I.ppr = mesh_->coordinate("pp");
  I.PP = cfg_.pp;
  I.c = mesh_->coordinate("cp");
  I.C = cfg_.cp;
  I.V = cfg_.virtual_stages;
  I.tp = mesh_->group("tp");
  I.pp = mesh_->group("pp");
  I.cp = mesh_->group("cp");
  I.dp = mesh_->group("dp");
  I.dpcp = mesh_->flatten({"dp", "cp"});
  I.tpp = mesh_->flatten({"pp", "tp"});
  I.G = I.dpcp->size();
  I.gr = I.dpcp->rank();

  // boyutlar
  I.d = c.d_model;
  I.H = c.n_head;
  I.Hk = c.kv_heads();
  I.D = c.hdim();
  I.F = c.d_ff;
  I.vocab = c.vocab_size;
  I.Vp = c.padded_vocab();
  I.rms = c.norm == lm::NormKind::RmsNorm;
  I.gated = c.gated_mlp();
  I.silu = c.mlp == lm::MlpKind::SwiGlu;
  I.learned_pos = c.pos == lm::PosKind::Learned;
  I.rope = c.pos == lm::PosKind::Rope;
  I.tie = c.tie_embeddings;
  I.eps = c.norm_eps;
  const int T = I.T;
  if (I.H % T != 0) throw DistError("Nexus TP: n_head (" + std::to_string(I.H) + ") tp'ye bolunmeli");
  if (I.F % T != 0) throw DistError("Nexus TP: d_ff tp'ye bolunmeli");
  if (I.Vp % T != 0) throw DistError("Nexus TP: (dolgulu) sozluk tp'ye bolunmeli (vocab_pad_multiple kullanin)");
  if (I.Hk % T != 0 && T % I.Hk != 0) {
    throw DistError("Nexus TP: GQA KV bas sayisi tp'yi bolmeli ya da tp'ye bolunmeli");
  }
  I.Hl = I.H / T;
  I.kv_replicated = I.Hk % T != 0;
  I.Hkl = I.kv_replicated ? 1 : I.Hk / T;
  const int group = I.H / I.Hk;  // q basi / kv basi
  const int kv0 = I.kv_replicated ? (I.t * I.Hl) / group : I.t * I.Hkl;
  I.Fl = I.F / T;
  I.upw = I.gated ? 2 * I.Fl : I.Fl;
  I.qkvw = (I.Hl + 2 * I.Hkl) * I.D;
  I.Vs = I.Vp / T;
  I.v0 = I.t * I.Vs;
  if (cfg_.cp > 1 && cfg_.cp_algorithm == CpAlgorithm::Ulysses && (I.Hl % cfg_.cp != 0 || I.Hkl % cfg_.cp != 0)) {
    throw DistError("Nexus CP Ulysses: yerel bas sayilari cp'ye bolunmeli");
  }
  if (I.rope) I.rope_cache = std::make_shared<lm::RopeCache>(c.rot_dim(), c.rope_theta, c.rope_scaling, c.max_position);

  // kv gruplari (tp icinde, KV cogaltildiysa): ayni kv basini paylasan ardisik tp rank'leri
  if (I.kv_replicated && T > 1) {
    for (int j = 0; j < I.Hk; ++j) {
      std::vector<int> ranks;
      for (int r = 0; r < T; ++r) {
        if ((r * I.Hl) / group == j) ranks.push_back(r);
      }
      GroupPtr g = I.tp->new_group(ranks);
      if (g) {
        I.kv = g;
        I.kv_first = ranks.front() == I.t;
      }
    }
  }
  // pp kenar gruplari (her yonlu kenar ayri grup: alicinin is parcacigi yalniz recv, gonderenin yalniz send yurutur)
  if (I.PP > 1) {
    I.fwd_edge.resize(static_cast<std::size_t>(I.PP));
    I.bwd_edge.resize(static_cast<std::size_t>(I.PP));
    for (int kind = 0; kind < 2; ++kind) {
      for (int a = 0; a < I.PP; ++a) {
        const int b = kind == 0 ? (a + 1) % I.PP : (a - 1 + I.PP) % I.PP;
        GroupPtr g = I.pp->new_group({std::min(a, b), std::max(a, b)});
        (kind == 0 ? I.fwd_edge : I.bwd_edge)[static_cast<std::size_t>(a)] = g;
      }
    }
    if (I.tie) I.tied = I.pp->new_group({0, I.PP - 1});
  }

  // katmanlari pp * v parcaya maliyet dengeli bol (kafa maliyeti son katmana eklenir)
  const int parts = I.PP * I.V;
  if (c.n_layer < parts) {
    throw DistError("Nexus PP: katman sayisi (" + std::to_string(c.n_layer) + ") pp * v'den (" +
                    std::to_string(parts) + ") az olamaz");
  }
  const double layer_cost = static_cast<double>(I.d) * (I.H + 2 * I.Hk) * I.D + static_cast<double>(I.H) * I.D * I.d +
                            static_cast<double>(I.d) * (I.gated ? 2 * I.F : I.F) + static_cast<double>(I.F) * I.d;
  std::vector<double> costs(static_cast<std::size_t>(c.n_layer), layer_cost);
  costs.back() += static_cast<double>(I.Vp) * I.d;
  const std::vector<std::pair<int, int>> ranges = balance_partition(costs, parts);
  I.chunks.resize(static_cast<std::size_t>(parts));
  for (int g = 0; g < parts; ++g) {
    ChunkDef& ch = I.chunks[static_cast<std::size_t>(g)];
    ch.g = g;
    ch.l0 = ranges[static_cast<std::size_t>(g)].first;
    ch.l1 = ranges[static_cast<std::size_t>(g)].second;
    ch.embed = g == 0;
    ch.final = g == parts - 1;
    if (I.owner(g) == I.ppr) I.my_chunks.push_back(g);
  }

  // parametreler (sahip olunan parcalar, artan sirayla)
  auto add = [&](const std::string& name, Param::Slice slice, std::int64_t row_begin, std::int64_t rows,
                 std::vector<std::pair<std::int64_t, std::int64_t>> cols, bool tp_rep) -> int {
    const lm::NamedParam* np = reference.find(name);
    if (np == nullptr) throw DistError("Nexus ParallelLm: referans parametre yok: " + name);
    Param p;
    p.name = name;
    p.full_shape = np->value.shape().dims;
    p.decay = np->decay;
    p.wd = np->decay ? 1.0f : 0.0f;
    p.slice = slice;
    p.row_begin = row_begin;
    p.rows = rows;
    p.cols = std::move(cols);
    p.tp_replicated = tp_rep;
    params_.push_back(std::move(p));
    const std::size_t i = params_.size() - 1;
    const std::vector<std::int64_t> idx = full_index(i);
    const std::vector<float>& full = np->value.values();
    params_[i].value.resize(idx.size());
    for (std::size_t e = 0; e < idx.size(); ++e) params_[i].value[e] = full[static_cast<std::size_t>(idx[e])];
    params_[i].grad.assign(idx.size(), 0.0f);
    return static_cast<int>(i);
  };
  using S = Param::Slice;
  const std::int64_t D = I.D;
  const std::vector<std::pair<std::int64_t, std::int64_t>> qkv_cols = {
      {static_cast<std::int64_t>(I.t) * I.Hl * D, static_cast<std::int64_t>(I.Hl) * D},
      {static_cast<std::int64_t>(I.H) * D + kv0 * D, static_cast<std::int64_t>(I.Hkl) * D},
      {static_cast<std::int64_t>(I.H + I.Hk) * D + kv0 * D, static_cast<std::int64_t>(I.Hkl) * D}};
  std::vector<std::pair<std::int64_t, std::int64_t>> up_cols = {{static_cast<std::int64_t>(I.t) * I.Fl, I.Fl}};
  if (I.gated) up_cols.push_back({static_cast<std::int64_t>(I.F) + static_cast<std::int64_t>(I.t) * I.Fl, I.Fl});
  for (int g : I.my_chunks) {
    ChunkDef& ch = I.chunks[static_cast<std::size_t>(g)];
    if (ch.embed) {
      ch.wte = add("wte", S::Rows, I.v0, I.Vs, {}, false);
      if (I.learned_pos) ch.wpe = add("wpe", S::Full, 0, 0, {}, true);
    }
    for (int l = ch.l0; l < ch.l1; ++l) {
      const std::string p = "h" + std::to_string(l) + ".";
      LayerIdx L;
      const bool ln_bias = !I.rms && c.bias;
      L.n1w = add(p + "norm1.weight", S::Full, 0, 0, {}, true);
      if (ln_bias) L.n1b = add(p + "norm1.bias", S::Full, 0, 0, {}, true);
      L.qkvw = add(p + "attn.qkv.weight", S::Cols, 0, 0, qkv_cols, false);
      if (c.bias || c.qkv_bias) L.qkvb = add(p + "attn.qkv.bias", S::Cols, 0, 0, qkv_cols, false);
      if (I.kv_replicated && T > 1) {
        params_[static_cast<std::size_t>(L.qkvw)].kv_offset = I.Hl * I.D;
        if (L.qkvb >= 0) params_[static_cast<std::size_t>(L.qkvb)].kv_offset = I.Hl * I.D;
      }
      L.ow = add(p + "attn.o.weight", S::Rows, static_cast<std::int64_t>(I.t) * I.Hl * D, static_cast<std::int64_t>(I.Hl) * D,
                 {}, false);
      if (c.bias) L.ob = add(p + "attn.o.bias", S::Full, 0, 0, {}, true);
      L.n2w = add(p + "norm2.weight", S::Full, 0, 0, {}, true);
      if (ln_bias) L.n2b = add(p + "norm2.bias", S::Full, 0, 0, {}, true);
      L.upw = add(p + "mlp.up.weight", S::Cols, 0, 0, up_cols, false);
      if (c.bias) L.upb = add(p + "mlp.up.bias", S::Cols, 0, 0, up_cols, false);
      L.dw = add(p + "mlp.down.weight", S::Rows, static_cast<std::int64_t>(I.t) * I.Fl, I.Fl, {}, false);
      if (c.bias) L.db = add(p + "mlp.down.bias", S::Full, 0, 0, {}, true);
      I.layers[l] = L;
    }
    if (ch.final) {
      ch.nfw = add("norm_f.weight", S::Full, 0, 0, {}, true);
      if (!I.rms && c.bias) ch.nfb = add("norm_f.bias", S::Full, 0, 0, {}, true);
      if (I.tie) {
        const int first = I.chunks[0].wte;  // ayni rank parca 0'a da sahipse ortak parametre
        if (first >= 0) {
          ch.head = first;
        } else {
          ch.head = add("wte", S::Rows, I.v0, I.Vs, {}, false);
          params_[static_cast<std::size_t>(ch.head)].tied_copy = true;
        }
      } else {
        ch.head = add("lm_head.weight", S::Rows, I.v0, I.Vs, {}, false);
      }
    }
  }

  // duz tampon (dp x cp indirgemesi + ZeRO-1) ve norm maskesi
  I.offset.resize(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i) {
    I.offset[i] = I.total;
    I.total += params_[i].value.size();
  }
  // Tum dp x cp rank'leri ayni yerel parametre kumesine sahip (ayni tp/pp koordinati) -> ayni boy.
  I.shard = (I.total + static_cast<std::size_t>(I.G) - 1) / static_cast<std::size_t>(I.G);
  I.padded = I.shard * static_cast<std::size_t>(I.G);
  I.shard_lo = I.shard * static_cast<std::size_t>(I.gr);
  I.norm_mask.assign(I.padded, 0);
  for (std::size_t i = 0; i < params_.size(); ++i) {
    const Param& p = params_[i];
    if (p.tied_copy) continue;
    if (p.tp_replicated && I.t != 0) continue;
    const std::size_t n = p.value.size();
    if (p.kv_offset >= 0 && !I.kv_first) {
      // yalniz q sutunlari (KV kopyasi grubun ilk rank'inde sayilir)
      const std::size_t width = static_cast<std::size_t>(I.qkvw);
      for (std::size_t e = 0; e < n; ++e) {
        if (e % width < static_cast<std::size_t>(p.kv_offset)) I.norm_mask[I.offset[i] + e] = 1;
      }
      continue;
    }
    std::fill(I.norm_mask.begin() + static_cast<std::ptrdiff_t>(I.offset[i]),
              I.norm_mask.begin() + static_cast<std::ptrdiff_t>(I.offset[i] + n), std::uint8_t{1});
  }
  I.gflat.assign(I.padded, 0.0f);
  I.gshard.assign(I.shard, 0.0f);
  const std::size_t state = cfg_.zero == 1 ? I.shard : I.total;
  I.m.assign(state, 0.0f);
  I.v.assign(state, 0.0f);
}

ParallelLm::~ParallelLm() = default;

const GroupPtr& ParallelLm::data_parallel_group() const { return impl_->dp; }

std::vector<std::int64_t> ParallelLm::full_index(std::size_t i) const {
  const Param& p = params_.at(i);
  std::vector<std::int64_t> out;
  const std::int64_t numel = std::accumulate(p.full_shape.begin(), p.full_shape.end(), std::int64_t{1},
                                             std::multiplies<std::int64_t>());
  const std::int64_t cols = p.full_shape.size() == 2 ? p.full_shape[1] : numel;
  const std::int64_t rows = p.full_shape.size() == 2 ? p.full_shape[0] : 1;
  switch (p.slice) {
    case Param::Slice::Full:
      out.resize(static_cast<std::size_t>(numel));
      std::iota(out.begin(), out.end(), std::int64_t{0});
      break;
    case Param::Slice::Rows:
      for (std::int64_t r = p.row_begin; r < p.row_begin + p.rows; ++r) {
        for (std::int64_t j = 0; j < cols; ++j) out.push_back(r * cols + j);
      }
      break;
    case Param::Slice::Cols:
      for (std::int64_t r = 0; r < rows; ++r) {
        for (const auto& cr : p.cols) {
          for (std::int64_t j = 0; j < cr.second; ++j) out.push_back(r * cols + cr.first + j);
        }
      }
      break;
  }
  return out;
}

void ParallelLm::set_param_scales(const std::vector<float>& weight_decay, const std::vector<float>& lr) {
  if (weight_decay.size() != params_.size() || lr.size() != params_.size()) {
    throw DistError("Nexus ParallelLm::set_param_scales: boy parametre sayisina esit olmali");
  }
  for (std::size_t i = 0; i < params_.size(); ++i) {
    params_[i].wd = weight_decay[i];
    params_[i].lr_scale = lr[i];
  }
}

std::uint64_t ParallelLm::persistent_bytes() const {
  std::uint64_t b = 0;
  for (const Param& p : params_) b += (p.value.size() + p.grad.size()) * sizeof(float);
  b += (impl_->m.size() + impl_->v.size() + impl_->gflat.size() + impl_->gshard.size()) * sizeof(float);
  return b;
}

std::string ParallelLm::layout_summary() const {
  const Impl& I = *impl_;
  std::string s = "dp" + std::to_string(mesh_->coordinate("dp")) + " pp" + std::to_string(I.ppr) + " cp" +
                  std::to_string(I.c) + " tp" + std::to_string(I.t) + ":";
  for (int g : I.my_chunks) {
    const ChunkDef& ch = I.chunks[static_cast<std::size_t>(g)];
    s += " c" + std::to_string(g) + "[" + (ch.embed ? "emb," : "") + "L" + std::to_string(ch.l0) + "-" +
         std::to_string(ch.l1 - 1) + (ch.final ? ",head" : "") + "]";
  }
  return s;
}

// ==========================================================================================
// Adim
// ==========================================================================================

std::vector<float> ParallelLm::forward_backward(const std::vector<const std::int32_t*>& micro,
                                                const std::vector<float>& weights, int batch, int seq) {
  Impl& I = *impl_;
  if (micro.empty() || micro.size() != weights.size()) throw DistError("Nexus ParallelLm: mikro yigin / agirlik boyu");
  I.prepare(batch, seq);
  const int M = static_cast<int>(micro.size());
  std::vector<MbInfo> mbs;
  mbs.reserve(micro.size());
  for (int i = 0; i < M; ++i) mbs.push_back(I.mb_info(micro[static_cast<std::size_t>(i)], weights[static_cast<std::size_t>(i)]));
  const std::vector<PipeAction> actions = make_schedule(cfg_.pp_schedule, I.PP, I.ppr, M, I.V);
  const bool zb = cfg_.pp_schedule == PipeSchedule::ZeroBubbleH1;
  stats_ = ParallelStepStats{};
  I.act_peak = 0;
  I.losses.assign(static_cast<std::size_t>(M), 0.0f);
  const int last = I.last_chunk();
  const std::size_t act_elems = static_cast<std::size_t>(I.R) * I.d;

  // Alici tamponlari ve recv'ler tuketim sirasiyla kuyruga (kenar grubu basina tek alici).
  struct Inbox {
    std::vector<float> buf;
    WorkPtr work;
  };
  std::map<std::pair<int, int>, Inbox> act_in, grad_in;
  auto peer_index = [](int me, int peer) { return peer < me ? 0 : 1; };
  for (const PipeAction& a : actions) {
    if (a.kind == PipeAction::Kind::F && a.chunk > 0 && I.owner(a.chunk - 1) != I.ppr) {
      Inbox& in = act_in[{a.mb, a.chunk}];
      in.buf.resize(act_elems);
      const int src = I.owner(a.chunk - 1);
      ProcessGroup& g = *I.fwd_edge[static_cast<std::size_t>(src)];
      in.work = g.recv(in.buf.data(), act_elems, DType::F32, peer_index(I.ppr, src), 0);
    }
    if (a.kind == PipeAction::Kind::B && a.chunk < last && I.owner(a.chunk + 1) != I.ppr) {
      Inbox& in = grad_in[{a.mb, a.chunk}];
      in.buf.resize(act_elems);
      const int src = I.owner(a.chunk + 1);
      ProcessGroup& g = *I.bwd_edge[static_cast<std::size_t>(src)];
      in.work = g.recv(in.buf.data(), act_elems, DType::F32, peer_index(I.ppr, src), 0);
    }
  }
  std::map<std::pair<int, int>, std::vector<float>> local_act, local_grad;  // ayni rank'teki parca devri
  std::vector<std::pair<std::shared_ptr<std::vector<float>>, WorkPtr>> sends;
  std::deque<std::pair<int, int>> wq;  // ZB-H1 ertelenmis W
  const std::size_t w_limit = static_cast<std::size_t>(std::max(1, I.PP - I.ppr));

  auto ready = [&](const PipeAction& a) {
    if (a.kind == PipeAction::Kind::F && a.chunk > 0) {
      auto it = act_in.find({a.mb, a.chunk});
      return it == act_in.end() || it->second.work->is_completed();
    }
    if (a.kind == PipeAction::Kind::B && a.chunk < last) {
      auto it = grad_in.find({a.mb, a.chunk});
      return it == grad_in.end() || it->second.work->is_completed();
    }
    return true;
  };
  auto run_w = [&]() {
    const auto [mb, g] = wq.front();
    wq.pop_front();
    const auto t0 = Clock::now();
    I.chunk_backward_weight(mb, g, mbs[static_cast<std::size_t>(mb)]);
    stats_.weight_ms += ms_since(t0);
    stats_.weights += 1;
    if (cfg_.trace) stats_.trace.push_back("W" + std::to_string(mb) + "c" + std::to_string(g));
  };
  auto send_to = [&](std::vector<float>&& data, bool forward, int g_next, int mb) {
    const int dst = I.owner(g_next);
    if (dst == I.ppr) {
      (forward ? local_act : local_grad)[{mb, g_next}] = std::move(data);
      return;
    }
    auto buf = std::make_shared<std::vector<float>>(std::move(data));
    ProcessGroup& g = forward ? *I.fwd_edge[static_cast<std::size_t>(I.ppr)] : *I.bwd_edge[static_cast<std::size_t>(I.ppr)];
    sends.emplace_back(buf, g.send(buf->data(), buf->size(), DType::F32, peer_index(I.ppr, dst), 0));
  };

  const auto step0 = Clock::now();
  for (const PipeAction& a : actions) {
    if (zb) {
      while (!wq.empty() && (wq.size() > w_limit || !ready(a))) run_w();
    }
    const MbInfo& mb = mbs[static_cast<std::size_t>(a.mb)];
    if (a.kind == PipeAction::Kind::F) {
      std::vector<float> x;
      if (a.chunk > 0) {
        auto it = act_in.find({a.mb, a.chunk});
        if (it != act_in.end()) {
          const auto tw = Clock::now();
          it->second.work->wait();
          stats_.idle_ms += ms_since(tw);
          x = std::move(it->second.buf);
          act_in.erase(it);
        } else {
          x = std::move(local_act.at({a.mb, a.chunk}));
          local_act.erase({a.mb, a.chunk});
        }
      }
      const auto t0 = Clock::now();
      const double loss = I.chunk_forward(a.mb, a.chunk, x, mb, true);
      stats_.forward_ms += ms_since(t0);
      stats_.forwards += 1;
      if (a.chunk == last) {
        I.losses[static_cast<std::size_t>(a.mb)] = static_cast<float>(loss);
      } else {
        send_to(std::move(x), true, a.chunk + 1, a.mb);
      }
    } else {
      std::vector<float> dx;
      if (a.chunk < last) {
        auto it = grad_in.find({a.mb, a.chunk});
        if (it != grad_in.end()) {
          const auto tw = Clock::now();
          it->second.work->wait();
          stats_.idle_ms += ms_since(tw);
          dx = std::move(it->second.buf);
          grad_in.erase(it);
        } else {
          dx = std::move(local_grad.at({a.mb, a.chunk}));
          local_grad.erase({a.mb, a.chunk});
        }
      }
      const auto t0 = Clock::now();
      I.chunk_backward_input(a.mb, a.chunk, dx, mb);
      if (!zb) I.chunk_backward_weight(a.mb, a.chunk, mb);
      stats_.backward_ms += ms_since(t0);
      stats_.backwards += 1;
      if (a.chunk > 0) send_to(std::move(dx), false, a.chunk - 1, a.mb);
      if (zb) {
        wq.emplace_back(a.mb, a.chunk);
        stats_.deferred_w_peak = std::max(stats_.deferred_w_peak, static_cast<int>(wq.size()));
      }
    }
    if (cfg_.trace) stats_.trace.push_back(a.str());
  }
  while (!wq.empty()) run_w();
  for (auto& s : sends) s.second->wait();
  stats_.step_ms = ms_since(step0);
  stats_.peak_activation_bytes = I.act_peak;
  if (!I.acts.empty()) throw DistError("Nexus ParallelLm: zamanlama sonunda aktivasyon kaldi (ic hata)");

  // ---- gradyan indirgemeleri ---------------------------------------------------------------
  if (I.sp()) {
    std::vector<float> buf;
    for (const Param& p : params_) {
      if (p.tp_replicated) buf.insert(buf.end(), p.grad.begin(), p.grad.end());
    }
    I.tp_allreduce(buf.data(), buf.size());
    std::size_t o = 0;
    for (Param& p : params_) {
      if (!p.tp_replicated) continue;
      std::copy(buf.begin() + static_cast<std::ptrdiff_t>(o), buf.begin() + static_cast<std::ptrdiff_t>(o + p.grad.size()),
                p.grad.begin());
      o += p.grad.size();
    }
  }
  if (I.kv && I.kv->size() > 1) {
    std::vector<float> buf;
    const std::size_t width = static_cast<std::size_t>(I.qkvw);
    for (const Param& p : params_) {
      if (p.kv_offset < 0) continue;
      for (std::size_t e = 0; e < p.grad.size(); ++e) {
        if (e % width >= static_cast<std::size_t>(p.kv_offset)) buf.push_back(p.grad[e]);
      }
    }
    I.kv->all_reduce(buf.data(), buf.size(), DType::F32, ReduceOp::Sum)->wait();
    std::size_t o = 0;
    for (Param& p : params_) {
      if (p.kv_offset < 0) continue;
      for (std::size_t e = 0; e < p.grad.size(); ++e) {
        if (e % width >= static_cast<std::size_t>(p.kv_offset)) p.grad[e] = buf[o++];
      }
    }
  }
  if (I.tied) {
    for (Param& p : params_) {
      if (p.name == "wte") I.tied->all_reduce(p.grad.data(), p.grad.size(), DType::F32, ReduceOp::Sum)->wait();
    }
  }
  // dp x cp: duz tampon reduce-scatter (zero 0: + all-gather). Ayni RS deseni -> zero 0/1 bit bit ayni.
  std::fill(I.gflat.begin(), I.gflat.end(), 0.0f);
  for (std::size_t i = 0; i < params_.size(); ++i) {
    std::copy(params_[i].grad.begin(), params_[i].grad.end(), I.gflat.begin() + static_cast<std::ptrdiff_t>(I.offset[i]));
  }
  if (I.G > 1) {
    I.dpcp->reduce_scatter(I.gflat.data(), I.gshard.data(), I.shard, DType::F32, ReduceOp::Sum)->wait();
    if (cfg_.zero == 0) {
      I.dpcp->all_gather(I.gshard.data(), I.gflat.data(), I.shard, DType::F32)->wait();
      for (std::size_t i = 0; i < params_.size(); ++i) {
        std::copy(I.gflat.begin() + static_cast<std::ptrdiff_t>(I.offset[i]),
                  I.gflat.begin() + static_cast<std::ptrdiff_t>(I.offset[i] + params_[i].grad.size()), params_[i].grad.begin());
      }
    }
  } else {
    std::copy(I.gflat.begin(), I.gflat.begin() + static_cast<std::ptrdiff_t>(I.shard), I.gshard.begin());
  }
  // kayiplar: cp toplami / gecerli; son asamadan yayin
  std::vector<float> out(static_cast<std::size_t>(M));
  if (I.C > 1) I.cp->all_reduce(I.losses.data(), I.losses.size(), DType::F32, ReduceOp::Sum)->wait();
  for (int i = 0; i < M; ++i) {
    const std::int64_t v = mbs[static_cast<std::size_t>(i)].valid;
    out[static_cast<std::size_t>(i)] = v > 0 ? I.losses[static_cast<std::size_t>(i)] / static_cast<float>(v) : 0.0f;
  }
  if (I.PP > 1) I.pp->broadcast(out.data(), out.size(), DType::F32, I.owner(last))->wait();
  return out;
}

ParallelLm::StepResult ParallelLm::step(const FsdpAdamW& opt, double lr, double clip) {
  Impl& I = *impl_;
  // Norm: parca uzerinden (zero 0/1 ayni toplam sirasi), sonra dp x cp ve pp x tp toplami.
  double sq = 0.0;
  for (std::size_t e = 0; e < I.shard; ++e) {
    if (I.norm_mask[I.shard_lo + e]) sq += static_cast<double>(I.gshard[e]) * I.gshard[e];
  }
  if (I.G > 1) I.dpcp->all_reduce(&sq, 1, DType::F64, ReduceOp::Sum)->wait();
  if (I.tpp->size() > 1) I.tpp->all_reduce(&sq, 1, DType::F64, ReduceOp::Sum)->wait();
  StepResult res;
  res.grad_norm = std::sqrt(sq);
  res.skipped = !std::isfinite(res.grad_norm);
  if (res.skipped) {
    for (Param& p : params_) std::fill(p.grad.begin(), p.grad.end(), 0.0f);
    return res;
  }
  float coef = 1.0f;
  if (clip > 0.0) {
    const double c = clip / (res.grad_norm + 1e-6);
    if (c < 1.0) coef = static_cast<float>(c);
  }
  step_ += 1;
  const double bc1 = 1.0 - std::pow(opt.beta1, static_cast<double>(step_));
  const double bc2 = 1.0 - std::pow(opt.beta2, static_cast<double>(step_));
  const float b1 = static_cast<float>(opt.beta1);
  const float b2 = static_cast<float>(opt.beta2);
  const float eps = static_cast<float>(opt.eps);
  const float sbc2 = static_cast<float>(std::sqrt(bc2));
  // duz [lo, hi) araligindaki ogeler: zero 0 tumu (tam durum), zero 1 yalniz parca.
  const std::size_t lo = cfg_.zero == 1 ? I.shard_lo : 0;
  const std::size_t hi = cfg_.zero == 1 ? std::min(I.total, I.shard_lo + I.shard) : I.total;
  std::vector<float> pshard;
  if (cfg_.zero == 1) pshard.assign(I.shard, 0.0f);
  for (std::size_t i = 0; i < params_.size(); ++i) {
    Param& p = params_[i];
    const std::size_t a = std::max(lo, I.offset[i]);
    const std::size_t b = std::min(hi, I.offset[i] + p.value.size());
    const float decay = static_cast<float>(lr * p.lr_scale * opt.weight_decay * p.wd);
    const float step_size = static_cast<float>(lr * p.lr_scale / bc1);
    for (std::size_t f = a; f < b; ++f) {
      const std::size_t e = f - I.offset[i];
      const std::size_t s = f - lo;
      const float g = (cfg_.zero == 1 ? I.gshard[s] : p.grad[e]) * coef;
      float w = p.value[e];
      w -= decay * w;
      I.m[s] = b1 * I.m[s] + (1.0f - b1) * g;
      I.v[s] = b2 * I.v[s] + (1.0f - b2) * g * g;
      w -= step_size * I.m[s] / (std::sqrt(I.v[s]) / sbc2 + eps);
      if (cfg_.zero == 1) {
        pshard[s] = w;
      } else {
        p.value[e] = w;
      }
    }
  }
  if (cfg_.zero == 1) {
    std::vector<float> full(I.padded);
    if (I.G > 1) {
      I.dpcp->all_gather(pshard.data(), full.data(), I.shard, DType::F32)->wait();
    } else {
      std::copy(pshard.begin(), pshard.end(), full.begin());
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
      std::copy(full.begin() + static_cast<std::ptrdiff_t>(I.offset[i]),
                full.begin() + static_cast<std::ptrdiff_t>(I.offset[i] + params_[i].value.size()), params_[i].value.begin());
    }
  }
  for (Param& p : params_) std::fill(p.grad.begin(), p.grad.end(), 0.0f);
  return res;
}

double ParallelLm::eval_loss(const std::int32_t* tokens, int batch, int seq) {
  Impl& I = *impl_;
  I.prepare(batch, seq);
  const MbInfo mb = I.mb_info(tokens, 0.0f);
  const int last = I.last_chunk();
  const std::size_t act_elems = static_cast<std::size_t>(I.R) * I.d;
  auto peer_index = [](int me, int peer) { return peer < me ? 0 : 1; };
  double loss = 0.0;
  std::vector<float> x;
  for (int g = 0; g <= last; ++g) {
    if (I.owner(g) != I.ppr) continue;
    if (g > 0 && I.owner(g - 1) != I.ppr) {
      x.resize(act_elems);
      I.fwd_edge[static_cast<std::size_t>(I.owner(g - 1))]
          ->recv(x.data(), act_elems, DType::F32, peer_index(I.ppr, I.owner(g - 1)), 0)
          ->wait();
    }
    loss = I.chunk_forward(0, g, x, mb, false);
    I.acts.erase({0, g});
    if (g < last && I.owner(g + 1) != I.ppr) {
      I.fwd_edge[static_cast<std::size_t>(I.ppr)]->send(x.data(), x.size(), DType::F32, peer_index(I.ppr, I.owner(g + 1)), 0)->wait();
    }
  }
  float l = static_cast<float>(loss);
  if (I.C > 1) I.cp->all_reduce(&l, 1, DType::F32, ReduceOp::Sum)->wait();
  l = mb.valid > 0 ? l / static_cast<float>(mb.valid) : 0.0f;
  if (I.PP > 1) I.pp->broadcast(&l, 1, DType::F32, I.owner(last))->wait();
  return l;
}

namespace {

void put_floats(std::vector<unsigned char>& out, const std::vector<float>& v) {
  const std::uint64_t n = v.size();
  const auto* p = reinterpret_cast<const unsigned char*>(&n);
  out.insert(out.end(), p, p + sizeof(n));
  const auto* d = reinterpret_cast<const unsigned char*>(v.data());
  out.insert(out.end(), d, d + n * sizeof(float));
}

void get_floats(const std::vector<unsigned char>& in, std::size_t& at, std::vector<float>& v) {
  std::uint64_t n = 0;
  if (at + sizeof(n) > in.size()) throw DistError("Nexus ParallelLm::load_state: kayit kisa");
  std::memcpy(&n, in.data() + at, sizeof(n));
  at += sizeof(n);
  if (n != v.size() || at + n * sizeof(float) > in.size()) {
    throw DistError("Nexus ParallelLm::load_state: boy uyusmuyor (farkli paralel yerlesim?)");
  }
  std::memcpy(v.data(), in.data() + at, n * sizeof(float));
  at += n * sizeof(float);
}

}  // namespace

std::vector<unsigned char> ParallelLm::save_state() const {
  std::vector<unsigned char> out;
  const std::int64_t s = step_;
  const auto* p = reinterpret_cast<const unsigned char*>(&s);
  out.insert(out.end(), p, p + sizeof(s));
  for (const Param& q : params_) put_floats(out, q.value);
  put_floats(out, impl_->m);
  put_floats(out, impl_->v);
  return out;
}

void ParallelLm::load_state(const std::vector<unsigned char>& blob) {
  std::size_t at = sizeof(std::int64_t);
  if (blob.size() < at) throw DistError("Nexus ParallelLm::load_state: kayit kisa");
  std::memcpy(&step_, blob.data(), sizeof(std::int64_t));
  for (Param& q : params_) {
    get_floats(blob, at, q.value);
    std::fill(q.grad.begin(), q.grad.end(), 0.0f);
  }
  get_floats(blob, at, impl_->m);
  get_floats(blob, at, impl_->v);
}

std::vector<float> assemble_full(const std::vector<const ParallelLm*>& ranks, const std::string& name, int which,
                                 double* replica_diff) {
  std::vector<float> full;
  std::vector<std::uint8_t> seen;
  double diff = 0.0;
  for (const ParallelLm* r : ranks) {
    const auto& ps = r->parameters();
    for (std::size_t i = 0; i < ps.size(); ++i) {
      const ParallelLm::Param& p = ps[i];
      if (p.name != name) continue;
      if (full.empty()) {
        const std::int64_t n = std::accumulate(p.full_shape.begin(), p.full_shape.end(), std::int64_t{1},
                                               std::multiplies<std::int64_t>());
        full.assign(static_cast<std::size_t>(n), 0.0f);
        seen.assign(static_cast<std::size_t>(n), 0);
      }
      const std::vector<float>& src = which == 0 ? p.value : p.grad;
      const std::vector<std::int64_t> idx = r->full_index(i);
      for (std::size_t e = 0; e < idx.size(); ++e) {
        const std::size_t f = static_cast<std::size_t>(idx[e]);
        if (seen[f]) {
          diff = std::max(diff, static_cast<double>(std::fabs(full[f] - src[e])));
        } else {
          full[f] = src[e];
          seen[f] = 1;
        }
      }
    }
  }
  if (full.empty()) throw DistError("Nexus assemble_full: parametre yok: " + name);
  for (std::uint8_t s : seen) {
    if (!s) throw DistError("Nexus assemble_full: " + name + " tum ogeleri kapsanmadi");
  }
  if (replica_diff != nullptr) *replica_diff = diff;
  return full;
}

}  // namespace dist
}  // namespace nexus
