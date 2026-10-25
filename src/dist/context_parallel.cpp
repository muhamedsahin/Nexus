/**
 * @file context_parallel.cpp
 * @brief FAZ 12.F: Ring Attention (contiguous/zigzag) ve Ulysses; CPU FlashAttention + LSE birlesimi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/context_parallel.hpp"

#include "nexus/attention/cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nexus {
namespace dist {

namespace {
constexpr int kTagKv = 101;
constexpr int kTagDkv = 102;
}  // namespace

const char* to_string(CpLayout layout) noexcept { return layout == CpLayout::Zigzag ? "zigzag" : "contiguous"; }
const char* to_string(CpAlgorithm algorithm) noexcept { return algorithm == CpAlgorithm::Ring ? "ring" : "ulysses"; }

int cp_num_chunks(int cp, CpLayout layout) noexcept { return layout == CpLayout::Zigzag ? 2 * cp : cp; }

std::vector<int> cp_chunks(int cp, int rank, CpLayout layout) {
  if (rank < 0 || rank >= cp) throw DistError("Nexus CP: rank aralik disinda");
  if (layout == CpLayout::Contiguous) return {rank};
  return {rank, 2 * cp - 1 - rank};
}

int CpAttention::chunk_len(int cp) const {
  const int n = cp_num_chunks(cp, layout);
  if (seq % n != 0) {
    throw DistError("Nexus CP: T = " + std::to_string(seq) + " parca sayisina (" + std::to_string(n) +
                    ") bolunmeli (" + to_string(layout) + ")");
  }
  return seq / n;
}

int ring_attention_blocks(int cp, int rank, CpLayout layout) {
  int blocks = 0;
  for (int gi : cp_chunks(cp, rank, layout)) {
    for (int s = 0; s < cp; ++s) {
      for (int gj : cp_chunks(cp, s, layout)) blocks += gj <= gi ? 1 : 0;
    }
  }
  return blocks;
}

namespace {

AttentionParams chunk_params(const CpAttention& a, int tq, int tk, int heads, int kv_heads, bool causal) {
  AttentionParams p;
  p.batch = a.batch;
  p.heads = heads;
  p.kv_heads = kv_heads;
  p.tq = tq;
  p.tk = tk;
  p.head_dim = a.head_dim;
  p.dtype = DType::F32;
  p.scale = a.scale;
  p.causal = causal;
  p.set_contiguous_bthd();
  return p;
}

/// (o, lse) <- birlesim((o, lse), (ob, lb)). o [B, Tc, H, D], lse [B, H, Tc].
void merge(float* o, float* lse, const float* ob, const float* lb, int batch, int tc, int heads, int d) {
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int t = 0; t < tc; ++t) {
        float& l1 = lse[(static_cast<std::size_t>(b) * heads + h) * tc + t];
        const float l2 = lb[(static_cast<std::size_t>(b) * heads + h) * tc + t];
        const float m = std::max(l1, l2);
        const float l = m + std::log(std::exp(l1 - m) + std::exp(l2 - m));
        const float w1 = std::exp(l1 - l);
        const float w2 = std::exp(l2 - l);
        float* or_ = o + ((static_cast<std::size_t>(b) * tc + t) * heads + h) * d;
        const float* br = ob + ((static_cast<std::size_t>(b) * tc + t) * heads + h) * d;
        for (int i = 0; i < d; ++i) or_[i] = or_[i] * w1 + br[i] * w2;
        l1 = l;
      }
    }
  }
}

}  // namespace

void ring_attention_forward(ProcessGroup& cp, const CpAttention& a, const float* q, const float* k, const float* v,
                            float* o, float* lse) {
  const int C = cp.size();
  const int r = cp.rank();
  const int tc = a.chunk_len(C);
  const std::vector<int> mine = cp_chunks(C, r, a.layout);
  const int nch = static_cast<int>(mine.size());
  const std::size_t qchunk = static_cast<std::size_t>(a.batch) * tc * a.heads * a.head_dim;
  const std::size_t kchunk = static_cast<std::size_t>(a.batch) * tc * a.kv_heads * a.head_dim;
  const std::size_t lchunk = static_cast<std::size_t>(a.batch) * a.heads * tc;
  const std::size_t kv_elems = 2 * static_cast<std::size_t>(nch) * kchunk;
  std::vector<float> cur(kv_elems), next(kv_elems);
  std::memcpy(cur.data(), k, nch * kchunk * sizeof(float));
  std::memcpy(cur.data() + nch * kchunk, v, nch * kchunk * sizeof(float));
  std::vector<float> ob(qchunk), lb(lchunk);
  std::vector<bool> started(static_cast<std::size_t>(nch), false);
  const int right = (r + 1) % C;
  const int left = (r - 1 + C) % C;
  for (int s = 0; s < C; ++s) {
    const int src = (r - s + C) % C;
    WorkPtr xw;  // K/V halka adimi (hesapla ortusur)
    if (s + 1 < C) xw = cp.send_recv(cur.data(), kv_elems, right, next.data(), kv_elems, left, DType::F32, kTagKv);
    const std::vector<int> theirs = cp_chunks(C, src, a.layout);
    for (int i = 0; i < nch; ++i) {
      for (std::size_t j = 0; j < theirs.size(); ++j) {
        const int gi = mine[static_cast<std::size_t>(i)];
        const int gj = theirs[j];
        if (gj > gi) continue;
        const AttentionParams p = chunk_params(a, tc, tc, a.heads, a.kv_heads, gj == gi);
        const float* kb = cur.data() + j * kchunk;
        const float* vb = cur.data() + nch * kchunk + j * kchunk;
        float* oi = o + static_cast<std::size_t>(i) * qchunk;
        float* li = lse + static_cast<std::size_t>(i) * lchunk;
        const float* qi = q + static_cast<std::size_t>(i) * qchunk;
        if (!started[static_cast<std::size_t>(i)]) {
          attention::flash_attention_forward_cpu(oi, li, qi, kb, vb, p, a.threads);
          started[static_cast<std::size_t>(i)] = true;
        } else {
          attention::flash_attention_forward_cpu(ob.data(), lb.data(), qi, kb, vb, p, a.threads);
          merge(oi, li, ob.data(), lb.data(), a.batch, tc, a.heads, a.head_dim);
        }
      }
    }
    if (xw) {
      xw->wait();
      cur.swap(next);
    }
  }
}

void ring_attention_backward(ProcessGroup& cp, const CpAttention& a, const float* q, const float* k, const float* v,
                             const float* o, const float* lse, const float* dout, float* dq, float* dk, float* dv) {
  const int C = cp.size();
  const int r = cp.rank();
  const int tc = a.chunk_len(C);
  const std::vector<int> mine = cp_chunks(C, r, a.layout);
  const int nch = static_cast<int>(mine.size());
  const std::size_t qchunk = static_cast<std::size_t>(a.batch) * tc * a.heads * a.head_dim;
  const std::size_t kchunk = static_cast<std::size_t>(a.batch) * tc * a.kv_heads * a.head_dim;
  const std::size_t lchunk = static_cast<std::size_t>(a.batch) * a.heads * tc;
  const std::size_t kv_elems = 2 * static_cast<std::size_t>(nch) * kchunk;
  std::vector<float> cur(kv_elems), next(kv_elems);
  std::memcpy(cur.data(), k, nch * kchunk * sizeof(float));
  std::memcpy(cur.data() + nch * kchunk, v, nch * kchunk * sizeof(float));
  std::vector<float> dkv_local(kv_elems), dkv_in(kv_elems, 0.0f), dkv_out(kv_elems);
  std::vector<float> tq(qchunk), tk(kchunk), tv(kchunk);
  std::fill(dq, dq + nch * qchunk, 0.0f);
  const int right = (r + 1) % C;
  const int left = (r - 1 + C) % C;
  WorkPtr dkv_x;  // dK/dV biriktiricisi halka adimi: dkv_out -> sag, dkv_in <- sol
  for (int s = 0; s < C; ++s) {
    const int src = (r - s + C) % C;
    WorkPtr xw;
    if (s + 1 < C) xw = cp.send_recv(cur.data(), kv_elems, right, next.data(), kv_elems, left, DType::F32, kTagKv);
    std::fill(dkv_local.begin(), dkv_local.end(), 0.0f);
    const std::vector<int> theirs = cp_chunks(C, src, a.layout);
    for (int i = 0; i < nch; ++i) {
      for (std::size_t j = 0; j < theirs.size(); ++j) {
        const int gi = mine[static_cast<std::size_t>(i)];
        const int gj = theirs[j];
        if (gj > gi) continue;
        const AttentionParams p = chunk_params(a, tc, tc, a.heads, a.kv_heads, gj == gi);
        const float* kb = cur.data() + j * kchunk;
        const float* vb = cur.data() + nch * kchunk + j * kchunk;
        const std::size_t qo = static_cast<std::size_t>(i) * qchunk;
        attention::flash_attention_backward_cpu(tq.data(), tk.data(), tv.data(), dout + qo, q + qo, kb, vb, o + qo,
                                                lse + static_cast<std::size_t>(i) * lchunk, p, a.threads);
        float* dqi = dq + qo;
        for (std::size_t e = 0; e < qchunk; ++e) dqi[e] += tq[e];
        float* dkj = dkv_local.data() + j * kchunk;
        float* dvj = dkv_local.data() + nch * kchunk + j * kchunk;
        for (std::size_t e = 0; e < kchunk; ++e) {
          dkj[e] += tk[e];
          dvj[e] += tv[e];
        }
      }
    }
    if (dkv_x) dkv_x->wait();  // onceki adimin biriktiricisi geldi (dkv_in), dkv_out serbest
    for (std::size_t e = 0; e < kv_elems; ++e) dkv_out[e] = (s > 0 ? dkv_in[e] : 0.0f) + dkv_local[e];
    if (C > 1) {
      dkv_x = cp.send_recv(dkv_out.data(), kv_elems, right, dkv_in.data(), kv_elems, left, DType::F32, kTagDkv);
    }
    if (xw) {
      xw->wait();
      cur.swap(next);
    }
  }
  if (C > 1) {
    dkv_x->wait();  // dkv_in: kendi K/V'min son biriktiricisi (sol komsu son ziyaretci)
  } else {
    dkv_in = dkv_out;
  }
  std::memcpy(dk, dkv_in.data(), nch * kchunk * sizeof(float));
  std::memcpy(dv, dkv_in.data() + nch * kchunk, nch * kchunk * sizeof(float));
}

// ---- Dizi <-> parca-ana yerlesim ----------------------------------------------------------

void cp_scatter_sequence(const float* full, float* local, int batch, int seq, int width, int cp, int rank,
                         CpLayout layout) {
  const int n = cp_num_chunks(cp, layout);
  const int tc = seq / n;
  const std::vector<int> mine = cp_chunks(cp, rank, layout);
  for (std::size_t i = 0; i < mine.size(); ++i) {
    for (int b = 0; b < batch; ++b) {
      std::memcpy(local + ((i * batch + b) * static_cast<std::size_t>(tc)) * width,
                  full + (static_cast<std::size_t>(b) * seq + static_cast<std::size_t>(mine[i]) * tc) * width,
                  static_cast<std::size_t>(tc) * width * sizeof(float));
    }
  }
}

void cp_gather_sequence(const float* local, float* full, int batch, int seq, int width, int cp, int rank,
                        CpLayout layout) {
  const int n = cp_num_chunks(cp, layout);
  const int tc = seq / n;
  const std::vector<int> mine = cp_chunks(cp, rank, layout);
  for (std::size_t i = 0; i < mine.size(); ++i) {
    for (int b = 0; b < batch; ++b) {
      std::memcpy(full + (static_cast<std::size_t>(b) * seq + static_cast<std::size_t>(mine[i]) * tc) * width,
                  local + ((i * batch + b) * static_cast<std::size_t>(tc)) * width,
                  static_cast<std::size_t>(tc) * width * sizeof(float));
    }
  }
}

// ---- Ulysses ------------------------------------------------------------------------------

namespace {

/// Yerel [nch,B,Tc,h,D] (tum h bas) -> tam dizi, bas parcasi [B,T,h/C,D]. all-to-all.
void seq_to_heads(ProcessGroup& cp, const CpAttention& a, int h, const float* local, float* full) {
  const int C = cp.size();
  const int r = cp.rank();
  const int tc = a.chunk_len(C);
  const int hs = h / C;
  const int D = a.head_dim;
  const int nch = static_cast<int>(cp_chunks(C, r, a.layout).size());
  const std::size_t rows = static_cast<std::size_t>(nch) * a.batch * tc;
  const std::size_t piece = rows * hs * D;
  std::vector<float> send(piece * C), recv(piece * C);
  for (int s = 0; s < C; ++s) {
    for (std::size_t row = 0; row < rows; ++row) {
      std::memcpy(send.data() + s * piece + row * hs * D, local + (row * h + static_cast<std::size_t>(s) * hs) * D,
                  static_cast<std::size_t>(hs) * D * sizeof(float));
    }
  }
  std::vector<std::size_t> counts(static_cast<std::size_t>(C), piece);
  cp.all_to_all_v(send.data(), counts, recv.data(), counts, DType::F32)->wait();
  for (int s = 0; s < C; ++s) {
    const std::vector<int> ch = cp_chunks(C, s, a.layout);
    for (std::size_t i = 0; i < ch.size(); ++i) {
      for (int b = 0; b < a.batch; ++b) {
        std::memcpy(full + (static_cast<std::size_t>(b) * a.seq + static_cast<std::size_t>(ch[i]) * tc) * hs * D,
                    recv.data() + s * piece + ((i * a.batch + b) * static_cast<std::size_t>(tc)) * hs * D,
                    static_cast<std::size_t>(tc) * hs * D * sizeof(float));
      }
    }
  }
}

/// Ters yon: tam dizi bas parcasi [B,T,h/C,D] -> yerel [nch,B,Tc,h,D].
void heads_to_seq(ProcessGroup& cp, const CpAttention& a, int h, const float* full, float* local) {
  const int C = cp.size();
  const int r = cp.rank();
  const int tc = a.chunk_len(C);
  const int hs = h / C;
  const int D = a.head_dim;
  const int nch = static_cast<int>(cp_chunks(C, r, a.layout).size());
  const std::size_t rows = static_cast<std::size_t>(nch) * a.batch * tc;
  const std::size_t piece = rows * hs * D;
  std::vector<float> send(piece * C), recv(piece * C);
  for (int s = 0; s < C; ++s) {
    const std::vector<int> ch = cp_chunks(C, s, a.layout);
    for (std::size_t i = 0; i < ch.size(); ++i) {
      for (int b = 0; b < a.batch; ++b) {
        std::memcpy(send.data() + s * piece + ((i * a.batch + b) * static_cast<std::size_t>(tc)) * hs * D,
                    full + (static_cast<std::size_t>(b) * a.seq + static_cast<std::size_t>(ch[i]) * tc) * hs * D,
                    static_cast<std::size_t>(tc) * hs * D * sizeof(float));
      }
    }
  }
  std::vector<std::size_t> counts(static_cast<std::size_t>(C), piece);
  cp.all_to_all_v(send.data(), counts, recv.data(), counts, DType::F32)->wait();
  for (int s = 0; s < C; ++s) {
    for (std::size_t row = 0; row < rows; ++row) {
      std::memcpy(local + (row * h + static_cast<std::size_t>(s) * hs) * D, recv.data() + s * piece + row * hs * D,
                  static_cast<std::size_t>(hs) * D * sizeof(float));
    }
  }
}

void check_ulysses(const CpAttention& a, int C) {
  if (a.heads % C != 0 || a.kv_heads % C != 0) {
    throw DistError("Nexus Ulysses: bas sayilari (" + std::to_string(a.heads) + ", kv " + std::to_string(a.kv_heads) +
                    ") cp = " + std::to_string(C) + "'ye bolunmeli (degilse ring kullanin)");
  }
}

}  // namespace

std::size_t ulysses_lse_size(const CpAttention& a, int cp) noexcept {
  return static_cast<std::size_t>(a.batch) * (a.heads / cp) * a.seq;
}

void ulysses_attention_forward(ProcessGroup& cp, const CpAttention& a, const float* q, const float* k, const float* v,
                               float* o, float* lse) {
  const int C = cp.size();
  check_ulysses(a, C);
  const std::size_t qn = static_cast<std::size_t>(a.batch) * a.seq * (a.heads / C) * a.head_dim;
  const std::size_t kn = static_cast<std::size_t>(a.batch) * a.seq * (a.kv_heads / C) * a.head_dim;
  std::vector<float> fq(qn), fk(kn), fv(kn), fo(qn);
  seq_to_heads(cp, a, a.heads, q, fq.data());
  seq_to_heads(cp, a, a.kv_heads, k, fk.data());
  seq_to_heads(cp, a, a.kv_heads, v, fv.data());
  const AttentionParams p = chunk_params(a, a.seq, a.seq, a.heads / C, a.kv_heads / C, true);
  attention::flash_attention_forward_cpu(fo.data(), lse, fq.data(), fk.data(), fv.data(), p, a.threads);
  heads_to_seq(cp, a, a.heads, fo.data(), o);
}

void ulysses_attention_backward(ProcessGroup& cp, const CpAttention& a, const float* q, const float* k,
                                const float* v, const float* o, const float* lse, const float* dout, float* dq,
                                float* dk, float* dv) {
  const int C = cp.size();
  check_ulysses(a, C);
  const std::size_t qn = static_cast<std::size_t>(a.batch) * a.seq * (a.heads / C) * a.head_dim;
  const std::size_t kn = static_cast<std::size_t>(a.batch) * a.seq * (a.kv_heads / C) * a.head_dim;
  std::vector<float> fq(qn), fk(kn), fv(kn), fo(qn), fd(qn), gq(qn), gk(kn), gv(kn);
  seq_to_heads(cp, a, a.heads, q, fq.data());
  seq_to_heads(cp, a, a.kv_heads, k, fk.data());
  seq_to_heads(cp, a, a.kv_heads, v, fv.data());
  seq_to_heads(cp, a, a.heads, o, fo.data());
  seq_to_heads(cp, a, a.heads, dout, fd.data());
  const AttentionParams p = chunk_params(a, a.seq, a.seq, a.heads / C, a.kv_heads / C, true);
  attention::flash_attention_backward_cpu(gq.data(), gk.data(), gv.data(), fd.data(), fq.data(), fk.data(), fv.data(),
                                          fo.data(), lse, p, a.threads);
  heads_to_seq(cp, a, a.heads, gq.data(), dq);
  heads_to_seq(cp, a, a.kv_heads, gk.data(), dk);
  heads_to_seq(cp, a, a.kv_heads, gv.data(), dv);
}

}  // namespace dist
}  // namespace nexus
