/**
 * @file passes.cpp
 * @brief FAZ 13.2-13.3: dce, sabit katlama, cse, desen eslemesi, autodiff, eleman eleman fuzyon,
 *        min-cut yeniden hesaplama, bellek planlama, iletisim zamanlama.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/passes.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nexus {
namespace ir {

namespace {

[[noreturn]] void bad(const std::string& what) { throw std::runtime_error("Nexus derleyici: " + what); }

/// Eski grafigi esleme ile kopyalamak icin yardimci.
struct Rebuild {
  const Graph& old;
  Graph out;
  std::vector<int> map;
  explicit Rebuild(const Graph& g) : old(g), map(g.nodes.size(), -1) { out.symbols = g.symbols; }
  int copy(int i) {
    Node n = old.nodes[static_cast<std::size_t>(i)];
    for (int& x : n.in) {
      x = map[static_cast<std::size_t>(x)];
      if (x < 0) bad("yeniden kurma: eslenmemis girdi");
    }
    map[static_cast<std::size_t>(i)] = out.add(std::move(n));
    return map[static_cast<std::size_t>(i)];
  }
  int add(Node n) { return out.add(std::move(n)); }
  void finish() {
    out.outputs.clear();
    for (int o : old.outputs) out.outputs.push_back(map[static_cast<std::size_t>(o)]);
    out.output_names = old.output_names;
  }
};

Node make(Op op, std::vector<int> in, Shape s, double a = 0.0, std::vector<std::int64_t> ints = {}, DType t = DType::F32) {
  Node n;
  n.op = op;
  n.in = std::move(in);
  n.shape = std::move(s);
  n.a = a;
  n.ints = std::move(ints);
  n.dtype = t;
  return n;
}

Node output_of(int parent, int index, Shape s, DType t = DType::F32) {
  Node n;
  n.op = Op::Output;
  n.in = {parent};
  n.index = index;
  n.shape = std::move(s);
  n.dtype = t;
  return n;
}

bool expensive(Op op) {
  switch (op) {
    case Op::Input: case Op::Param: case Op::MatMul: case Op::MatMulNT: case Op::MatMulTN: case Op::Attention:
    case Op::Embedding: case Op::CrossEntropy: case Op::FusedLinearCE: case Op::LinearBiasGelu: case Op::BMM:
    case Op::MatMulBias: case Op::AttentionQKV:
    case Op::BMMNT: case Op::AllGather: case Op::ReduceScatter: case Op::AllReduce:
      return true;
    default:
      return false;
  }
}

std::int64_t value_bytes(const Node& n, const Bindings& b) {
  const std::int64_t e = numel(n.shape, b) * 4;
  return (e + 63) / 64 * 64;
}

}  // namespace

// ==========================================================================================
// Temizlik
// ==========================================================================================

Graph dce(const Graph& g) {
  std::vector<char> live(g.nodes.size(), 0);
  for (int o : g.outputs) live[static_cast<std::size_t>(o)] = 1;
  for (std::size_t k = g.nodes.size(); k-- > 0;) {
    if (!live[k]) continue;
    for (int i : g.nodes[k].in) live[static_cast<std::size_t>(i)] = 1;
  }
  Rebuild r(g);
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    if (live[i]) r.copy(static_cast<int>(i));
  }
  r.finish();
  return r.out;
}

Graph constant_fold(const Graph& g, int* folded) {
  Rebuild r(g);
  int count = 0;
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const Node& n = g.nodes[i];
    auto cval = [&](int x, double* v) {
      const Node& m = r.out.nodes[static_cast<std::size_t>(r.map[static_cast<std::size_t>(x)])];
      if (m.op != Op::Const) return false;
      *v = m.a;
      return true;
    };
    double a = 0.0, b = 0.0, v = 0.0;
    bool f = false;
    if ((n.op == Op::Scale || n.op == Op::AddScalar || n.op == Op::Neg) && cval(n.in[0], &a)) {
      v = n.op == Op::Scale ? a * n.a : n.op == Op::AddScalar ? a + n.a : -a;
      f = true;
    } else if ((n.op == Op::Add || n.op == Op::Mul || n.op == Op::Sub) && cval(n.in[0], &a) && cval(n.in[1], &b)) {
      v = n.op == Op::Add ? a + b : n.op == Op::Mul ? a * b : a - b;
      f = true;
    }
    if (f) {
      r.map[i] = r.add(make(Op::Const, {}, n.shape, v));
      count += 1;
    } else {
      r.copy(static_cast<int>(i));
    }
  }
  r.finish();
  if (folded) *folded = count;
  return dce(r.out);
}

Graph cse(const Graph& g, int* merged) {
  Rebuild r(g);
  std::map<std::string, int> seen;
  int count = 0;
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const Node& n = g.nodes[i];
    if (n.op == Op::Param || n.op == Op::Input || n.op == Op::Fused || n.op == Op::AllGather ||
        n.op == Op::ReduceScatter || n.op == Op::AllReduce) {
      r.copy(static_cast<int>(i));
      continue;
    }
    std::ostringstream k;
    k << static_cast<int>(n.op) << "|" << n.index << "|" << n.a << "|";
    for (int x : n.in) k << r.map[static_cast<std::size_t>(x)] << ",";
    k << "|";
    for (auto v : n.ints) k << v << ",";
    k << "|" << shape_str(n.shape, g) << (n.dtype == DType::I32 ? "i" : "f") << "|" << n.name;  // "recompute" kopyalari ayri
    auto it = seen.find(k.str());
    if (it != seen.end()) {
      r.map[i] = it->second;
      count += 1;
      continue;
    }
    seen[k.str()] = r.copy(static_cast<int>(i));
  }
  r.finish();
  if (merged) *merged = count;
  return dce(r.out);
}

// ==========================================================================================
// Desen eslemesi
// ==========================================================================================

Graph rewrite_patterns(const Graph& g, PatternStats* stats, int ce_chunk_rows, unsigned mask) {
  PatternStats st;
  const std::vector<int> uses = g.use_counts();
  Rebuild r(g);
  auto N = [&](int i) -> const Node& { return g.nodes[static_cast<std::size_t>(i)]; };
  auto M = [&](int i) { return r.map[static_cast<std::size_t>(i)]; };
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const Node& n = g.nodes[i];
    // (1) softmax(Q K^T * s + mask) V -> Attention
    if ((mask & kPatAttention) && n.op == Op::MergeHeads) {
      const Node& bm = N(n.in[0]);
      if (bm.op == Op::BMM && N(bm.in[0]).op == Op::Softmax && N(N(bm.in[0]).in[0]).op == Op::CausalMask &&
          N(N(N(bm.in[0]).in[0]).in[0]).op == Op::Scale) {
        const Node& sc = N(N(N(bm.in[0]).in[0]).in[0]);
        const Node& s = N(sc.in[0]);
        auto unrep = [&](int x) { return N(x).op == Op::RepeatKV ? N(x).in[0] : x; };
        if (s.op == Op::BMMNT && N(s.in[0]).op == Op::SplitHeads && N(unrep(s.in[1])).op == Op::SplitHeads &&
            N(unrep(bm.in[1])).op == Op::SplitHeads) {
          const Node& qh = N(s.in[0]);
          const Node& kh = N(unrep(s.in[1]));
          const Node& vh = N(unrep(bm.in[1]));
          const std::int64_t H = qh.ints[0], D = qh.ints[1], Hk = kh.ints[0];
          if (vh.ints[0] == Hk && n.ints[0] == H && kh.ints[1] == D) {
            const double scale = std::fabs(sc.a - 1.0 / std::sqrt(static_cast<double>(D))) < 1e-12 ? 0.0 : sc.a;
            const Node& q = N(qh.in[0]);
            const Node& k = N(kh.in[0]);
            const Node& v = N(vh.in[0]);
            // Paketli QKV: ayni kaynaktan (0, HD), (HD, HkD), ((H + Hk) D, HkD) dilimleri -> dilimsiz okuma.
            if ((mask & kPatPackedQKV) && q.op == Op::SliceCols && k.op == Op::SliceCols && v.op == Op::SliceCols &&
                q.in[0] == k.in[0] && q.in[0] == v.in[0] && q.ints[0] == 0 && k.ints[0] == H * D &&
                v.ints[0] == (H + Hk) * D && N(q.in[0]).shape[1] == Dim::constant((H + 2 * Hk) * D)) {
              r.map[i] = r.add(make(Op::AttentionQKV, {M(q.in[0])}, n.shape, scale, {H, Hk, D, n.ints[2], n.ints[3]}));
              st.attention += 1;
              st.packed_qkv += 1;
              continue;
            }
            r.map[i] = r.add(make(Op::Attention, {M(qh.in[0]), M(kh.in[0]), M(vh.in[0])}, n.shape, scale,
                                  {H, Hk, D, n.ints[2], n.ints[3]}));
            st.attention += 1;
            continue;
          }
        }
      }
    }
    // (2) silu(u[:, :F]) * u[:, F:2F] -> SwiGlu
    if ((mask & kPatSwiGlu) && n.op == Op::Mul && N(n.in[0]).op == Op::Silu && N(N(n.in[0]).in[0]).op == Op::SliceCols &&
        N(n.in[1]).op == Op::SliceCols) {
      const Node& s0 = N(N(n.in[0]).in[0]);
      const Node& s1 = N(n.in[1]);
      const std::int64_t F = s0.ints[1];
      if (s0.in[0] == s1.in[0] && s0.ints[0] == 0 && s1.ints[0] == F && s1.ints[1] == F &&
          N(s0.in[0]).shape[1] == Dim::constant(2 * F)) {
        r.map[i] = r.add(make(Op::SwiGlu, {M(s0.in[0])}, n.shape));
        st.swiglu += 1;
        continue;
      }
    }
    // (3) gelu(x W + b) -> LinearBiasGelu (cuBLASLt GELU epilogu karsiligi)
    if ((mask & kPatLinearGelu) && n.op == Op::Gelu && N(n.in[0]).op == Op::BiasAdd && N(N(n.in[0]).in[0]).op == Op::MatMul &&
        uses[static_cast<std::size_t>(n.in[0])] == 1 && uses[static_cast<std::size_t>(N(n.in[0]).in[0])] == 1) {
      const Node& ba = N(n.in[0]);
      const Node& mm = N(ba.in[0]);
      r.map[i] = r.add(make(Op::LinearBiasGelu, {M(mm.in[0]), M(mm.in[1]), M(ba.in[1])}, n.shape));
      st.linear_gelu += 1;
      continue;
    }
    // (3b) x W + b -> MatMulBias (cuBLASLt bias epilogu)
    if ((mask & kPatLinearBias) && n.op == Op::BiasAdd && N(n.in[0]).op == Op::MatMul &&
        uses[static_cast<std::size_t>(n.in[0])] == 1) {
      const Node& mm = N(n.in[0]);
      r.map[i] = r.add(make(Op::MatMulBias, {M(mm.in[0]), M(mm.in[1]), M(n.in[1])}, n.shape));
      st.linear_bias += 1;
      continue;
    }
    // (4) norm(a + b) -> AddNorm; Add'in diger kullanicilari artik ciktisini (Output 1) kullanir
    if ((mask & kPatAddNorm) && (n.op == Op::LayerNorm || n.op == Op::RmsNorm) && N(n.in[0]).op == Op::Add) {
      const Node& ad = N(n.in[0]);
      std::vector<int> in{M(ad.in[0]), M(ad.in[1]), M(n.in[1])};
      const bool has_bias = n.op == Op::LayerNorm && n.ints[0] == 1;
      if (has_bias) in.push_back(M(n.in[2]));
      const int y = r.add(make(Op::AddNorm, in, n.shape, n.a, {n.op == Op::LayerNorm ? 0 : 1, has_bias ? 1 : 0}));
      const int res = r.add(output_of(y, 1, ad.shape));
      r.map[i] = y;
      r.map[static_cast<std::size_t>(n.in[0])] = res;
      st.add_norm += 1;
      continue;
    }
    // (5) CE(h W^T) -> FusedLinearCE (logit tamponu yok)
    if ((mask & kPatLinearCE) && n.op == Op::CrossEntropy && N(n.in[0]).op == Op::MatMulNT && uses[static_cast<std::size_t>(n.in[0])] == 1) {
      const Node& mm = N(n.in[0]);
      r.map[i] = r.add(make(Op::FusedLinearCE, {M(mm.in[0]), M(mm.in[1]), M(n.in[1])}, n.shape, 0.0,
                            {n.ints[0], n.ints[1], ce_chunk_rows}));
      st.linear_ce += 1;
      continue;
    }
    r.copy(static_cast<int>(i));
  }
  r.finish();
  if (stats) *stats = st;
  return dce(r.out);
}

// ==========================================================================================
// Autodiff
// ==========================================================================================

Joint build_joint(const Graph& fwd) {
  Joint j;
  j.graph = fwd;
  Graph& g = j.graph;
  j.backward_begin = static_cast<int>(g.nodes.size());
  if (fwd.outputs.size() != 1 || !fwd.nodes[static_cast<std::size_t>(fwd.outputs[0])].shape.empty()) {
    bad("build_joint: tek skaler cikti (kayip) bekler");
  }
  const int loss = fwd.outputs[0];
  const std::size_t F = fwd.nodes.size();
  std::vector<int> grad(F, -1);
  std::map<int, std::map<int, int>> aux_grad;                              // ust -> (indeks -> gradyan)
  std::map<int, std::vector<std::tuple<std::int64_t, std::int64_t, int>>> slices;  // kaynak -> (bas, boy, grad)
  std::map<std::pair<int, int>, int> outputs_of;                           // (ust, indeks) -> Output dugumu
  for (std::size_t i = 0; i < F; ++i) {
    if (fwd.nodes[i].op == Op::Output) outputs_of[{fwd.nodes[i].in[0], fwd.nodes[i].index}] = static_cast<int>(i);
  }
  auto add = [&](Node n) { return g.add(std::move(n)); };
  auto S = [&](int v) -> Shape { return g.nodes[static_cast<std::size_t>(v)].shape; };
  auto out_of = [&](int parent, int idx, const Shape& s) {
    auto it = outputs_of.find({parent, idx});
    if (it != outputs_of.end()) return it->second;
    const int o = add(output_of(parent, idx, s));
    outputs_of[{parent, idx}] = o;
    return o;
  };
  auto acc = [&](int v, int gv) {
    const Node& n = g.nodes[static_cast<std::size_t>(v)];
    if (n.dtype == DType::I32 || n.op == Op::Const || n.op == Op::Input || n.op == Op::Positions) return;
    if (static_cast<std::size_t>(v) >= F) bad("autodiff: ileri disi deger");
    int& slot = grad[static_cast<std::size_t>(v)];
    slot = slot < 0 ? gv : add(make(Op::Add, {slot, gv}, S(v)));
  };
  grad[static_cast<std::size_t>(loss)] = add(make(Op::Const, {}, {}, 1.0));
  for (std::size_t ii = F; ii-- > 0;) {
    const int i = static_cast<int>(ii);
    const Node n = g.nodes[ii];  // kopya: g buyuyor
    if (!slices[i].empty()) {
      auto& v = slices[i];
      std::sort(v.begin(), v.end());
      const std::int64_t cols = n.shape[1].coef;
      std::vector<int> parts;
      std::int64_t at = 0;
      for (const auto& [b0, len, gs] : v) {
        if (b0 < at) bad("autodiff: ust uste binen SliceCols gradyanlari");
        if (b0 > at) parts.push_back(add(make(Op::Const, {}, {n.shape[0], Dim::constant(b0 - at)}, 0.0)));
        parts.push_back(gs);
        at = b0 + len;
      }
      if (at < cols) parts.push_back(add(make(Op::Const, {}, {n.shape[0], Dim::constant(cols - at)}, 0.0)));
      acc(i, parts.size() == 1 ? parts[0] : add(make(Op::ConcatCols, parts, n.shape)));
    }
    const int gr = grad[ii];
    if (n.op == Op::Output) {
      if (gr >= 0) aux_grad[n.in[0]][n.index] = gr;
      continue;
    }
    const bool has_aux = aux_grad.count(i) != 0;
    if (gr < 0 && !has_aux) continue;
    auto in = [&](int k) { return n.in[static_cast<std::size_t>(k)]; };
    switch (n.op) {
      case Op::Param: case Op::Input: case Op::Const: case Op::Positions:
        break;
      case Op::Add:
        acc(in(0), gr);
        acc(in(1), gr);
        break;
      case Op::Sub:
        acc(in(0), gr);
        acc(in(1), add(make(Op::Neg, {gr}, n.shape)));
        break;
      case Op::Mul:
        acc(in(0), add(make(Op::Mul, {gr, in(1)}, n.shape)));
        acc(in(1), add(make(Op::Mul, {gr, in(0)}, n.shape)));
        break;
      case Op::Scale:
        acc(in(0), add(make(Op::Scale, {gr}, n.shape, n.a)));
        break;
      case Op::AddScalar:
        acc(in(0), gr);
        break;
      case Op::Neg:
        acc(in(0), add(make(Op::Neg, {gr}, n.shape)));
        break;
      case Op::Exp:
        acc(in(0), add(make(Op::Mul, {gr, i}, n.shape)));
        break;
      case Op::Tanh:
        acc(in(0), add(make(Op::TanhBwd, {gr, i}, n.shape)));
        break;
      case Op::Sigmoid:
        acc(in(0), add(make(Op::SigmoidBwd, {gr, i}, n.shape)));
        break;
      case Op::Gelu:
        acc(in(0), add(make(Op::GeluBwd, {gr, in(0)}, n.shape)));
        break;
      case Op::GeluTanh:
        acc(in(0), add(make(Op::GeluTanhBwd, {gr, in(0)}, n.shape)));
        break;
      case Op::Silu:
        acc(in(0), add(make(Op::SiluBwd, {gr, in(0)}, n.shape)));
        break;
      case Op::BiasAdd:
        acc(in(0), gr);
        acc(in(1), add(make(Op::SumRows, {gr}, S(in(1)))));
        break;
      case Op::MatMul:
        acc(in(0), add(make(Op::MatMulNT, {gr, in(1)}, S(in(0)))));
        acc(in(1), add(make(Op::MatMulTN, {in(0), gr}, S(in(1)))));
        break;
      case Op::MatMulNT:
        acc(in(0), add(make(Op::MatMul, {gr, in(1)}, S(in(0)))));
        acc(in(1), add(make(Op::MatMulTN, {gr, in(0)}, S(in(1)))));
        break;
      case Op::MatMulBias:
        acc(in(0), add(make(Op::MatMulNT, {gr, in(1)}, S(in(0)))));
        acc(in(1), add(make(Op::MatMulTN, {in(0), gr}, S(in(1)))));
        acc(in(2), add(make(Op::SumRows, {gr}, S(in(2)))));
        break;
      case Op::AttentionQKV: {
        const int lse = out_of(i, 1, {n.shape[0], Dim::constant(n.ints[0])});
        acc(in(0), add(make(Op::AttentionQKVBwd, {gr, in(0), i, lse}, S(in(0)), n.a, n.ints)));
        break;
      }
      case Op::SliceCols:
        slices[in(0)].emplace_back(n.ints[0], n.ints[1], gr);
        break;
      case Op::ConcatCols: {
        std::int64_t at = 0;
        for (int x : n.in) {
          const std::int64_t len = S(x)[1].coef;
          acc(x, add(make(Op::SliceCols, {gr}, S(x), 0.0, {at, len})));
          at += len;
        }
        break;
      }
      case Op::LayerNorm: {
        acc(in(0), add(make(Op::LayerNormBwdX, {gr, in(0), in(1)}, n.shape, n.a)));
        const int dw = add(make(Op::LayerNormBwdW, {gr, in(0)}, S(in(1)), n.a, {n.ints[0]}));
        acc(in(1), dw);
        if (n.ints[0] == 1) acc(in(2), add(output_of(dw, 1, S(in(2)))));
        break;
      }
      case Op::RmsNorm:
        acc(in(0), add(make(Op::RmsNormBwdX, {gr, in(0), in(1)}, n.shape, n.a)));
        acc(in(1), add(make(Op::RmsNormBwdW, {gr, in(0)}, S(in(1)), n.a)));
        break;
      case Op::AddNorm: {
        const bool rms = n.ints[0] == 1;
        const bool has_bias = n.ints[1] == 1;
        const int res = out_of(i, 1, S(in(0)));
        int dr = -1;
        if (gr >= 0) {
          dr = add(make(rms ? Op::RmsNormBwdX : Op::LayerNormBwdX, {gr, res, in(2)}, n.shape, n.a));
          if (rms) {
            acc(in(2), add(make(Op::RmsNormBwdW, {gr, res}, S(in(2)), n.a)));
          } else {
            const int dw = add(make(Op::LayerNormBwdW, {gr, res}, S(in(2)), n.a, {has_bias ? 1 : 0}));
            acc(in(2), dw);
            if (has_bias) acc(in(3), add(output_of(dw, 1, S(in(3)))));
          }
        }
        auto it = aux_grad.find(i);
        if (it != aux_grad.end() && it->second.count(1)) {
          const int gres = it->second[1];
          dr = dr < 0 ? gres : add(make(Op::Add, {dr, gres}, n.shape));
        }
        if (dr >= 0) {
          acc(in(0), dr);
          acc(in(1), dr);
        }
        break;
      }
      case Op::Rope:
        acc(in(0), add(make(Op::RopeBwd, {gr}, n.shape, n.a, n.ints)));
        break;
      case Op::Attention: {
        const int lse = out_of(i, 1, {n.shape[0], Dim::constant(n.ints[0])});
        const int dq = add(make(Op::AttentionBwd, {gr, in(0), in(1), in(2), i, lse}, S(in(0)), n.a, n.ints));
        acc(in(0), dq);
        acc(in(1), add(output_of(dq, 1, S(in(1)))));
        acc(in(2), add(output_of(dq, 2, S(in(2)))));
        break;
      }
      case Op::Embedding:
        acc(in(0), add(make(Op::EmbeddingBwd, {gr, in(1)}, S(in(0)))));
        break;
      case Op::CrossEntropy:
        acc(in(0), add(make(Op::CrossEntropyBwd, {in(0), in(1), gr}, S(in(0)), 0.0, n.ints)));
        break;
      case Op::FusedLinearCE: {
        const int dh = add(make(Op::FusedLinearCEBwd, {in(0), in(1), in(2), gr}, S(in(0)), 0.0, n.ints));
        acc(in(0), dh);
        acc(in(1), add(output_of(dh, 1, S(in(1)))));
        break;
      }
      case Op::SwiGlu:
        acc(in(0), add(make(Op::SwiGluBwd, {gr, in(0)}, S(in(0)))));
        break;
      case Op::LinearBiasGelu: {
        const int pre = out_of(i, 1, n.shape);
        const int dpre = add(make(Op::GeluBwd, {gr, pre}, n.shape));
        acc(in(0), add(make(Op::MatMulNT, {dpre, in(1)}, S(in(0)))));
        acc(in(1), add(make(Op::MatMulTN, {in(0), dpre}, S(in(1)))));
        acc(in(2), add(make(Op::SumRows, {dpre}, S(in(2)))));
        break;
      }
      default:
        bad(std::string("autodiff: VJP kurali yok: ") + to_string(n.op) + " (desen eslemesi kapali mi?)");
    }
  }
  g.outputs = {loss};
  g.output_names = {"loss"};
  for (std::size_t i = 0; i < F; ++i) {
    const Node& n = fwd.nodes[i];
    if (n.op != Op::Param) continue;
    int gi = grad[i];
    if (gi < 0) gi = add(make(Op::Const, {}, n.shape, 0.0));
    g.outputs.push_back(gi);
    g.output_names.push_back("grad:" + n.name);
    j.param_names.push_back(n.name);
  }
  return j;
}

// ==========================================================================================
// Eleman eleman fuzyon
// ==========================================================================================

Graph fuse_elementwise(const Graph& g, FusionStats* stats) {
  const std::size_t n = g.nodes.size();
  std::vector<int> cl(n, -1);
  struct Cluster {
    std::vector<int> members;
    int start = 0;
    Shape shape;
  };
  std::vector<Cluster> cls;
  std::vector<std::vector<int>> users(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (int x : g.nodes[i].in) users[static_cast<std::size_t>(x)].push_back(static_cast<int>(i));
  }
  auto ok_with = [&](std::size_t i, int c) {
    const Node& nd = g.nodes[i];
    if (nd.shape != cls[static_cast<std::size_t>(c)].shape) return false;
    for (int x : nd.in) {
      if (cl[static_cast<std::size_t>(x)] == c) continue;
      if (x >= cls[static_cast<std::size_t>(c)].start) return false;
    }
    return true;
  };
  for (std::size_t i = 0; i < n; ++i) {
    const Node& nd = g.nodes[i];
    if (!is_elementwise(nd.op) || nd.dtype != DType::F32) continue;
    std::vector<int> cand;
    for (int x : nd.in) {  // dikey: uretici kumesi
      if (cl[static_cast<std::size_t>(x)] >= 0) cand.push_back(cl[static_cast<std::size_t>(x)]);
    }
    for (int x : nd.in) {  // yatay: ayni girdiyi okuyan kardeslerin kumesi
      for (int u : users[static_cast<std::size_t>(x)]) {
        if (u < static_cast<int>(i) && cl[static_cast<std::size_t>(u)] >= 0) cand.push_back(cl[static_cast<std::size_t>(u)]);
      }
    }
    int chosen = -1;
    for (int c : cand) {
      if (ok_with(i, c)) {
        chosen = c;
        break;
      }
    }
    if (chosen < 0) {
      cls.push_back({{static_cast<int>(i)}, static_cast<int>(i), nd.shape});
      chosen = static_cast<int>(cls.size()) - 1;
    } else {
      cls[static_cast<std::size_t>(chosen)].members.push_back(static_cast<int>(i));
    }
    cl[i] = chosen;
  }
  FusionStats fs;
  std::vector<char> is_out(n, 0);
  for (int o : g.outputs) is_out[static_cast<std::size_t>(o)] = 1;
  Rebuild r(g);
  for (std::size_t i = 0; i < n; ++i) {
    const int c = cl[i];
    if (c < 0 || cls[static_cast<std::size_t>(c)].members.size() < 2) {
      r.copy(static_cast<int>(i));
      continue;
    }
    const Cluster& C = cls[static_cast<std::size_t>(c)];
    if (C.start != static_cast<int>(i)) continue;  // ilk uyede uretildi
    auto f = std::make_shared<FusedOp>();
    std::vector<int> ext;
    std::map<int, int> local;
    std::vector<int> outs;
    for (int m : C.members) {
      const Node& nd = g.nodes[static_cast<std::size_t>(m)];
      FusedOp::Inst inst;
      inst.op = nd.op;
      inst.a = nd.a;
      for (std::size_t k = 0; k < nd.in.size(); ++k) {
        const int x = nd.in[k];
        if (cl[static_cast<std::size_t>(x)] == c) {
          inst.args.push_back(local.at(x));
          continue;
        }
        auto it = std::find(ext.begin(), ext.end(), x);
        int e = static_cast<int>(it - ext.begin());
        if (it == ext.end()) {
          ext.push_back(x);
          f->input_is_row.push_back(nd.op == Op::BiasAdd && k == 1);
        }
        inst.args.push_back(-(e + 1));
      }
      local[m] = static_cast<int>(f->insts.size());
      f->insts.push_back(inst);
      bool outside = is_out[static_cast<std::size_t>(m)] != 0;
      for (int u : users[static_cast<std::size_t>(m)]) outside = outside || cl[static_cast<std::size_t>(u)] != c;
      if (outside) {
        outs.push_back(m);
        f->outputs.push_back(local[m]);
      }
    }
    std::ostringstream key;
    for (const auto& in : f->insts) {
      key << static_cast<int>(in.op) << "(";
      for (int a : in.args) key << a << ",";
      key << in.a << ")";
    }
    key << "o";
    for (int o : f->outputs) key << o << ",";
    key << "r";
    for (bool b : f->input_is_row) key << (b ? 1 : 0);
    f->key = key.str();
    Node fn;
    fn.op = Op::Fused;
    for (int x : ext) fn.in.push_back(r.map[static_cast<std::size_t>(x)]);
    fn.shape = C.shape;
    fn.fused = f;
    const int fid = r.add(std::move(fn));
    r.map[static_cast<std::size_t>(outs[0])] = fid;
    for (std::size_t k = 1; k < outs.size(); ++k) r.map[static_cast<std::size_t>(outs[k])] = r.add(output_of(fid, static_cast<int>(k), C.shape));
    fs.clusters += 1;
    fs.fused_nodes += static_cast<int>(C.members.size());
  }
  r.finish();
  if (stats) *stats = fs;
  return r.out;
}

// ==========================================================================================
// Geri gecis epilog fuzyonlari
// ==========================================================================================

namespace {

std::int64_t flags_of(const Node& n) { return n.ints.empty() ? 0 : n.ints[0]; }
bool is_mm(Op op) { return op == Op::MatMul || op == Op::MatMulNT || op == Op::MatMulTN; }

}  // namespace

Graph fuse_epilogues(const Graph& g0, EpilogueStats* stats) {
  EpilogueStats es;
  Graph g = g0;
  // (1) DGELU: GeluBwd(MatMulNT(a, b), pre), matmul tek kullanimli
  {
    const std::vector<int> uses = g.use_counts();
    Rebuild r(g);
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      const Node& n = g.nodes[i];
      if (n.op == Op::GeluBwd && g.nodes[static_cast<std::size_t>(n.in[0])].op == Op::MatMulNT &&
          flags_of(g.nodes[static_cast<std::size_t>(n.in[0])]) == 0 && uses[static_cast<std::size_t>(n.in[0])] == 1) {
        const Node& mm = g.nodes[static_cast<std::size_t>(n.in[0])];
        r.map[i] = r.add(make(Op::MatMulNT, {r.map[static_cast<std::size_t>(mm.in[0])], r.map[static_cast<std::size_t>(mm.in[1])],
                                             r.map[static_cast<std::size_t>(n.in[1])]},
                              n.shape, 0.0, {4}));
        es.dgelu += 1;
        continue;
      }
      r.copy(static_cast<int>(i));
    }
    r.finish();
    g = dce(r.out);
  }
  // (2) BGRAD: MatMulTN(x, gy) ve SumRows(gy) -> MatMulTN{2} + Output(1)
  {
    std::vector<std::vector<int>> users(g.nodes.size());
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      for (int x : g.nodes[i].in) users[static_cast<std::size_t>(x)].push_back(static_cast<int>(i));
    }
    std::map<int, int> sum_for_mm;  // MatMulTN -> SumRows
    std::set<int> taken;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      const Node& n = g.nodes[i];
      if (n.op != Op::SumRows) continue;
      for (int u : users[static_cast<std::size_t>(n.in[0])]) {
        const Node& m = g.nodes[static_cast<std::size_t>(u)];
        if (m.op != Op::MatMulTN || flags_of(m) != 0 || m.in[1] != n.in[0] || taken.count(u)) continue;
        bool late = true;  // SumRows kullanicilari matmul'dan sonra olmali (Output matmul'dan sonra gelir)
        for (int w : users[i]) late = late && w > u;
        bool out = false;
        for (int o : g.outputs) out = out || o == static_cast<int>(i);
        if (!late && !out) continue;
        sum_for_mm[u] = static_cast<int>(i);
        taken.insert(u);
        break;
      }
    }
    std::map<int, int> sum_to_mm;
    for (const auto& kv : sum_for_mm) sum_to_mm[kv.second] = kv.first;
    Rebuild r(g);
    std::map<int, int> pending_out;  // SumRows (eski) -> yeni Output
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      const Node& n = g.nodes[i];
      auto it = sum_for_mm.find(static_cast<int>(i));
      if (it != sum_for_mm.end()) {
        Node m = n;
        m.ints = {2};
        for (int& x : m.in) x = r.map[static_cast<std::size_t>(x)];
        const int id = r.add(std::move(m));
        r.map[i] = id;
        const int s = it->second;
        const int o = r.add(output_of(id, 1, g.nodes[static_cast<std::size_t>(s)].shape));
        if (s < static_cast<int>(i)) {
          pending_out[s] = o;  // daha once yer tutucu kopyalandi; kullanicilar (hep sonra) yeniden eslenir
          r.map[static_cast<std::size_t>(s)] = o;
        } else {
          pending_out[s] = o;
        }
        es.bgrad += 1;
        continue;
      }
      if (sum_to_mm.count(static_cast<int>(i))) {
        auto po = pending_out.find(static_cast<int>(i));
        if (po != pending_out.end()) {
          r.map[i] = po->second;  // matmul once geldi
        } else {
          r.copy(static_cast<int>(i));  // matmul sonra gelecek; o zaman esleme Output'a doner
        }
        continue;
      }
      r.copy(static_cast<int>(i));
    }
    r.finish();
    g = dce(r.out);
  }
  // (3) beta = 1 biriktirme: Add(c, matmul) / Add(matmul, c), matmul tek kullanimli, Output cocugu yok
  {
    const std::vector<int> uses = g.use_counts();
    std::vector<char> has_child(g.nodes.size(), 0);
    for (const Node& n : g.nodes) {
      if (n.op == Op::Output) has_child[static_cast<std::size_t>(n.in[0])] = 1;
    }
    Rebuild r(g);
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      const Node& n = g.nodes[i];
      if (n.op == Op::Add) {
        int mm = -1, other = -1;
        for (int k = 1; k >= 0; --k) {
          const int x = n.in[static_cast<std::size_t>(k)];
          const Node& m = g.nodes[static_cast<std::size_t>(x)];
          if (mm < 0 && is_mm(m.op) && (flags_of(m) & 1) == 0 && (flags_of(m) & 4) == 0 && uses[static_cast<std::size_t>(x)] == 1 &&
              !has_child[static_cast<std::size_t>(x)] && m.shape == n.shape) {
            mm = x;
            other = n.in[static_cast<std::size_t>(1 - k)];
          }
        }
        if (mm >= 0) {
          Node m = g.nodes[static_cast<std::size_t>(mm)];
          m.ints = {flags_of(m) | 1};
          for (int& x : m.in) x = r.map[static_cast<std::size_t>(x)];
          m.in.push_back(r.map[static_cast<std::size_t>(other)]);
          r.map[i] = r.add(std::move(m));
          es.accumulate += 1;
          continue;
        }
      }
      r.copy(static_cast<int>(i));
    }
    r.finish();
    g = dce(r.out);
  }
  // (4) Norm geri fuzyonu: NormBwdX(g, r, w) + NormBwdW(g, r) (+ db) + Add(dx, artik) -> tek cekirdek
  //     (ints[0]: 1 = agirlik gradyanlari Output 1/2, 2 = son girdi biriktirilir).
  {
    const std::vector<int> uses = g.use_counts();
    std::vector<std::vector<int>> users(g.nodes.size());
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      for (int x : g.nodes[i].in) users[static_cast<std::size_t>(x)].push_back(static_cast<int>(i));
    }
    struct Plan {
      int w = -1, db = -1, add = -1, other = -1;
    };
    std::map<int, Plan> plans;
    std::set<int> absorbed;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      const Node& x = g.nodes[i];
      if (x.op != Op::LayerNormBwdX && x.op != Op::RmsNormBwdX) continue;
      const Op wop = x.op == Op::LayerNormBwdX ? Op::LayerNormBwdW : Op::RmsNormBwdW;
      Plan p;
      for (int u : users[static_cast<std::size_t>(x.in[0])]) {
        const Node& w = g.nodes[static_cast<std::size_t>(u)];
        if (w.op == wop && w.in[1] == x.in[1] && u > static_cast<int>(i) && !absorbed.count(u)) {
          p.w = u;
          break;
        }
      }
      if (p.w < 0) continue;
      for (int u : users[static_cast<std::size_t>(p.w)]) {
        if (g.nodes[static_cast<std::size_t>(u)].op == Op::Output) p.db = u;
      }
      if (uses[i] == 1 && users[i].size() == 1 && g.nodes[static_cast<std::size_t>(users[i][0])].op == Op::Add) {
        const Node& ad = g.nodes[static_cast<std::size_t>(users[i][0])];
        const int other = ad.in[0] == static_cast<int>(i) ? ad.in[1] : ad.in[0];
        if (other < static_cast<int>(i) && ad.shape == x.shape) {
          p.add = users[i][0];
          p.other = other;
        }
      }
      absorbed.insert(p.w);
      plans[static_cast<int>(i)] = p;
    }
    Rebuild r(g);
    std::set<int> skip;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      if (skip.count(static_cast<int>(i))) continue;
      auto it = plans.find(static_cast<int>(i));
      if (it == plans.end()) {
        r.copy(static_cast<int>(i));
        continue;
      }
      const Plan& p = it->second;
      Node n = g.nodes[i];
      for (int& x : n.in) x = r.map[static_cast<std::size_t>(x)];
      n.ints = {1 | (p.add >= 0 ? 2 : 0)};
      if (p.add >= 0) n.in.push_back(r.map[static_cast<std::size_t>(p.other)]);
      const int id = r.add(std::move(n));
      r.map[i] = id;
      const Node& wn = g.nodes[static_cast<std::size_t>(p.w)];
      r.map[static_cast<std::size_t>(p.w)] = r.add(output_of(id, 1, wn.shape));
      skip.insert(p.w);
      if (p.db >= 0) {
        r.map[static_cast<std::size_t>(p.db)] = r.add(output_of(id, 2, g.nodes[static_cast<std::size_t>(p.db)].shape));
        skip.insert(p.db);
      }
      if (p.add >= 0) {
        r.map[static_cast<std::size_t>(p.add)] = id;
        skip.insert(p.add);
      }
      es.norm_backward += 1;
    }
    r.finish();
    g = dce(r.out);
  }
  if (stats) *stats = es;
  return g;
}

// ==========================================================================================
// Min-cut yeniden hesaplama
// ==========================================================================================

namespace {

struct Dinic {
  struct E {
    int to;
    std::int64_t cap;
  };
  std::vector<E> e;
  std::vector<std::vector<int>> adj;
  std::vector<int> level, it;
  explicit Dinic(int n) : adj(static_cast<std::size_t>(n)), level(static_cast<std::size_t>(n)), it(static_cast<std::size_t>(n)) {}
  void edge(int a, int b, std::int64_t c) {
    adj[static_cast<std::size_t>(a)].push_back(static_cast<int>(e.size()));
    e.push_back({b, c});
    adj[static_cast<std::size_t>(b)].push_back(static_cast<int>(e.size()));
    e.push_back({a, 0});
  }
  bool bfs(int s, int t) {
    std::fill(level.begin(), level.end(), -1);
    std::queue<int> q;
    level[static_cast<std::size_t>(s)] = 0;
    q.push(s);
    while (!q.empty()) {
      const int v = q.front();
      q.pop();
      for (int id : adj[static_cast<std::size_t>(v)]) {
        if (e[static_cast<std::size_t>(id)].cap > 0 && level[static_cast<std::size_t>(e[static_cast<std::size_t>(id)].to)] < 0) {
          level[static_cast<std::size_t>(e[static_cast<std::size_t>(id)].to)] = level[static_cast<std::size_t>(v)] + 1;
          q.push(e[static_cast<std::size_t>(id)].to);
        }
      }
    }
    return level[static_cast<std::size_t>(t)] >= 0;
  }
  std::int64_t dfs(int v, int t, std::int64_t f) {
    if (v == t) return f;
    for (int& k = it[static_cast<std::size_t>(v)]; k < static_cast<int>(adj[static_cast<std::size_t>(v)].size()); ++k) {
      const int id = adj[static_cast<std::size_t>(v)][static_cast<std::size_t>(k)];
      E& ed = e[static_cast<std::size_t>(id)];
      if (ed.cap > 0 && level[static_cast<std::size_t>(ed.to)] == level[static_cast<std::size_t>(v)] + 1) {
        const std::int64_t d = dfs(ed.to, t, std::min(f, ed.cap));
        if (d > 0) {
          ed.cap -= d;
          e[static_cast<std::size_t>(id ^ 1)].cap += d;
          return d;
        }
      }
    }
    return 0;
  }
  std::int64_t maxflow(int s, int t) {
    std::int64_t flow = 0;
    while (bfs(s, t)) {
      std::fill(it.begin(), it.end(), 0);
      while (std::int64_t f = dfs(s, t, std::numeric_limits<std::int64_t>::max())) flow += f;
    }
    return flow;
  }
  std::vector<char> reachable(int s) {
    std::vector<char> seen(adj.size(), 0);
    std::vector<int> st{s};
    seen[static_cast<std::size_t>(s)] = 1;
    while (!st.empty()) {
      const int v = st.back();
      st.pop_back();
      for (int id : adj[static_cast<std::size_t>(v)]) {
        const E& ed = e[static_cast<std::size_t>(id)];
        if (ed.cap > 0 && !seen[static_cast<std::size_t>(ed.to)]) {
          seen[static_cast<std::size_t>(ed.to)] = 1;
          st.push_back(ed.to);
        }
      }
    }
    return seen;
  }
};

}  // namespace

Joint min_cut_recompute(const Joint& joint, const Bindings& ex, RecomputeStats* stats) {
  const Graph& g = joint.graph;
  const int F = joint.backward_begin;
  const std::int64_t INF = std::numeric_limits<std::int64_t>::max() / 4;
  std::vector<char> needed(static_cast<std::size_t>(F), 0);
  for (std::size_t j = static_cast<std::size_t>(F); j < g.nodes.size(); ++j) {
    for (int x : g.nodes[j].in) {
      if (x < F) needed[static_cast<std::size_t>(x)] = 1;
    }
  }
  for (int o : g.outputs) {
    if (o < F) needed[static_cast<std::size_t>(o)] = 1;
  }
  // Output dugumu ust ile birlikte: ust pahaliysa Output da kaynak-baglidir.
  auto must = [&](int v) {
    const Node& n = g.nodes[static_cast<std::size_t>(v)];
    if (n.op == Op::Output) return expensive(g.nodes[static_cast<std::size_t>(n.in[0])].op);
    return expensive(n.op);
  };
  const int S = 2 * F, T = 2 * F + 1;
  Dinic net(2 * F + 2);
  RecomputeStats rs;
  for (int v = 0; v < F; ++v) {
    const Node& n = g.nodes[static_cast<std::size_t>(v)];
    const bool free_save = n.op == Op::Param || n.op == Op::Input;
    net.edge(2 * v, 2 * v + 1, free_save ? 0 : value_bytes(n, ex));
    if (must(v)) net.edge(S, 2 * v, INF);
    // Output(i) ust dugum yeniden kosulmadan uretilemez: girdileri ustun girdileridir.
    const std::vector<int>& deps = n.op == Op::Output ? g.nodes[static_cast<std::size_t>(n.in[0])].in : n.in;
    for (int x : deps) {
      if (x < F) net.edge(2 * x + 1, 2 * v, INF);
    }
    if (needed[static_cast<std::size_t>(v)]) {
      net.edge(2 * v + 1, T, INF);
      if (!free_save) rs.save_all_bytes += value_bytes(n, ex);
    }
  }
  (void)net.maxflow(S, T);
  const std::vector<char> reach = net.reachable(S);
  std::vector<char> saved(static_cast<std::size_t>(F), 0);
  for (int v = 0; v < F; ++v) {
    const Node& n = g.nodes[static_cast<std::size_t>(v)];
    const bool cut = reach[static_cast<std::size_t>(2 * v)] && !reach[static_cast<std::size_t>(2 * v + 1)];
    saved[static_cast<std::size_t>(v)] = cut || n.op == Op::Param || n.op == Op::Input ? 1 : 0;
    if (cut && n.op != Op::Param && n.op != Op::Input && needed[static_cast<std::size_t>(v)]) {
      rs.saved += 1;
      rs.saved_bytes += value_bytes(n, ex);
    } else if (cut && n.op != Op::Param && n.op != Op::Input) {
      rs.saved += 1;
      rs.saved_bytes += value_bytes(n, ex);
    }
  }
  // Yeniden kur: ileri ayni; geri dugumlerin ihtiyac duydugu saklanmayan ileri degerler kopyalanir.
  Rebuild r(g);
  for (int v = 0; v < F; ++v) r.copy(v);
  std::map<int, int> clone;  // ileri dugum -> geri bolumdeki kopya
  std::function<int(int)> avail;
  std::function<int(int)> make_clone = [&](int v) -> int {
    auto it = clone.find(v);
    if (it != clone.end()) return it->second;
    Node n = g.nodes[static_cast<std::size_t>(v)];
    if (must(v)) {
      // Kesim kaynagi pahali dugumleri her zaman saklar; buraya gelinmemeli.
      bad("min-cut: pahali dugum saklanmadi (ic hata)");
    }
    if (n.op == Op::Output) {
      n.in[0] = make_clone(n.in[0]);  // ek cikti icin ust dugum her zaman yeniden kosulur
    } else {
      for (int& x : n.in) x = avail(x);
    }
    n.name = "recompute";  // cse ileri asliyla birlestirmesin
    const int id = r.add(std::move(n));
    clone[v] = id;
    rs.recomputed += 1;
    return id;
  };
  avail = [&](int v) -> int {
    if (v >= F || saved[static_cast<std::size_t>(v)]) return r.map[static_cast<std::size_t>(v)];
    return make_clone(v);
  };
  for (std::size_t j = static_cast<std::size_t>(F); j < g.nodes.size(); ++j) {
    Node n = g.nodes[j];
    for (int& x : n.in) x = avail(x);
    r.map[j] = r.add(std::move(n));
  }
  r.out.outputs.clear();
  for (int o : g.outputs) r.out.outputs.push_back(avail(o));
  r.out.output_names = g.output_names;
  Joint out;
  out.graph = dce(r.out);
  // dce ileri bolumu kucultebilir: geri baslangicini ilk kopya/geri dugumu olarak yeniden bul (yalniz bilgi).
  out.backward_begin = -1;  // dce sonrasi sinir tutulmaz (yurutucu gerektirmez)
  out.param_names = joint.param_names;
  if (stats) *stats = rs;
  return out;
}

// ==========================================================================================
// Bellek planlama
// ==========================================================================================

MemoryPlan plan_memory(const Graph& g, const Bindings& b) {
  const std::size_t n = g.nodes.size();
  MemoryPlan p;
  p.offset.assign(n, -1);
  p.bytes.assign(n, 0);
  p.aux_offset.assign(n, -1);
  std::vector<int> def(n, 0), last(n, -1);
  const int end = static_cast<int>(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Node& nd = g.nodes[i];
    def[i] = nd.op == Op::Output ? nd.in[0] : static_cast<int>(i);
    last[i] = def[i];
    for (int x : nd.in) last[static_cast<std::size_t>(x)] = std::max(last[static_cast<std::size_t>(x)], static_cast<int>(i));
  }
  for (int o : g.outputs) last[static_cast<std::size_t>(o)] = end;
  // Output(i) kullanicisi, ust dugumun degil kendi tamponunun yasamini uzatir; ust tampon Output'lardan bagimsiz.
  struct Buf {
    std::size_t id;
    std::int64_t bytes;
    int a, z;
  };
  std::vector<Buf> bufs;
  for (std::size_t i = 0; i < n; ++i) {
    const Node& nd = g.nodes[i];
    if (nd.op == Op::Param || nd.op == Op::Input) continue;
    p.bytes[i] = value_bytes(nd, b);
    p.naive_bytes += p.bytes[i];
    bufs.push_back({i, p.bytes[i], def[i], last[i]});
  }
  std::sort(bufs.begin(), bufs.end(), [](const Buf& x, const Buf& y) {
    return x.bytes != y.bytes ? x.bytes > y.bytes : x.id < y.id;
  });
  std::vector<Buf> placed;
  std::vector<std::int64_t> placed_off;
  for (const Buf& bf : bufs) {
    // cakisan (zamanda) yerlesmislerin araliklarini topla, en dusuk bos ofseti bul
    std::vector<std::pair<std::int64_t, std::int64_t>> busy;
    for (std::size_t k = 0; k < placed.size(); ++k) {
      if (placed[k].a <= bf.z && bf.a <= placed[k].z) busy.emplace_back(placed_off[k], placed_off[k] + placed[k].bytes);
    }
    std::sort(busy.begin(), busy.end());
    std::int64_t off = 0;
    for (const auto& [s0, s1] : busy) {
      if (off + bf.bytes <= s0) break;
      off = std::max(off, s1);
    }
    p.offset[bf.id] = off;
    p.arena_bytes = std::max(p.arena_bytes, off + bf.bytes);
    placed.push_back(bf);
    placed_off.push_back(off);
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (g.nodes[i].op == Op::Output) p.aux_offset[i] = p.offset[i];
  }
  return p;
}

// ==========================================================================================
// Iletisim zamanlama
// ==========================================================================================

Graph schedule_communication(const Graph& g, int* moved) {
  const std::size_t n = g.nodes.size();
  std::vector<int> pending(n, 0);
  std::vector<std::vector<int>> users(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (int x : g.nodes[i].in) {
      pending[i] += 1;
      users[static_cast<std::size_t>(x)].push_back(static_cast<int>(i));
    }
  }
  auto comm = [&](std::size_t i) {
    const Op op = g.nodes[i].op;
    return op == Op::AllGather || op == Op::ReduceScatter || op == Op::AllReduce;
  };
  std::set<int> ready_comm, ready;
  for (std::size_t i = 0; i < n; ++i) {
    if (pending[i] == 0) (comm(i) ? ready_comm : ready).insert(static_cast<int>(i));
  }
  std::vector<int> order;
  while (!ready.empty() || !ready_comm.empty()) {
    int v;
    if (!ready_comm.empty()) {
      v = *ready_comm.begin();
      ready_comm.erase(ready_comm.begin());
    } else {
      v = *ready.begin();
      ready.erase(ready.begin());
    }
    order.push_back(v);
    for (int u : users[static_cast<std::size_t>(v)]) {
      if (--pending[static_cast<std::size_t>(u)] == 0) (comm(static_cast<std::size_t>(u)) ? ready_comm : ready).insert(u);
    }
  }
  int mv = 0;
  for (std::size_t k = 0; k < order.size(); ++k) {
    if (comm(static_cast<std::size_t>(order[k])) && static_cast<int>(k) < order[k]) mv += 1;
  }
  Rebuild r(g);
  for (int v : order) r.copy(v);
  r.finish();
  if (moved) *moved = mv;
  return r.out;
}

}  // namespace ir
}  // namespace nexus
