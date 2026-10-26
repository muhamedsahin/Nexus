/**
 * @file trace.cpp
 * @brief FAZ 13.2: Builder (sekil cikarimli meta tensor islemleri) ve lm::LmConfig ileri gecis izlemesi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/trace.hpp"

#include <cmath>
#include <stdexcept>

namespace nexus {
namespace ir {

namespace {

[[noreturn]] void bad(const std::string& what) { throw std::runtime_error("Nexus IR izleme: " + what); }

void same(const Shape& a, const Shape& b, const char* op) {
  if (a != b) bad(std::string(op) + ": sekiller ayni olmali");
}

}  // namespace

int Builder::node(Op op, std::vector<int> in, Shape s, DType t, double a, std::vector<std::int64_t> ints) {
  Node n;
  n.op = op;
  n.in = std::move(in);
  n.shape = std::move(s);
  n.dtype = t;
  n.a = a;
  n.ints = std::move(ints);
  return g_.add(std::move(n));
}

int Builder::output(int of, int index, Shape s, DType t) {
  Node n;
  n.op = Op::Output;
  n.in = {of};
  n.index = index;
  n.shape = std::move(s);
  n.dtype = t;
  return g_.add(std::move(n));
}

int Builder::input(const std::string& name, Shape s, DType t) {
  Node n;
  n.op = Op::Input;
  n.name = name;
  n.shape = std::move(s);
  n.dtype = t;
  return g_.add(std::move(n));
}

int Builder::param(const std::string& name, Shape s) {
  Node n;
  n.op = Op::Param;
  n.name = name;
  n.shape = std::move(s);
  return g_.add(std::move(n));
}

int Builder::constant(double value, Shape s) { return node(Op::Const, {}, std::move(s), DType::F32, value); }

int Builder::positions(int sb, int st) {
  return node(Op::Positions, {}, {Dim::symbol(sb) * Dim::symbol(st)}, DType::I32, 0.0, {sb, st});
}

int Builder::unary(Op op, int x, double a) { return node(op, {x}, shape(x), DType::F32, a); }

int Builder::binary(Op op, int x, int y) {
  same(shape(x), shape(y), to_string(op));
  return node(op, {x, y}, shape(x));
}

int Builder::bias_add(int x, int b) {
  const Shape& s = shape(x);
  if (s.size() != 2 || shape(b).size() != 1 || shape(b)[0] != s[1]) bad("BiasAdd: x [R, C], b [C]");
  return node(Op::BiasAdd, {x, b}, s);
}

int Builder::sum_rows(int x) {
  const Shape& s = shape(x);
  if (s.size() != 2) bad("SumRows: [R, C]");
  return node(Op::SumRows, {x}, {s[1]});
}

int Builder::matmul(int a, int b) {
  const Shape& sa = shape(a);
  const Shape& sb = shape(b);
  if (sa.size() != 2 || sb.size() != 2 || sa[1] != sb[0]) bad("MatMul: [M, K] @ [K, N]");
  return node(Op::MatMul, {a, b}, {sa[0], sb[1]});
}

int Builder::matmul_nt(int a, int b) {
  const Shape& sa = shape(a);
  const Shape& sb = shape(b);
  if (sa.size() != 2 || sb.size() != 2 || sa[1] != sb[1]) bad("MatMulNT: [M, K] @ [N, K]^T");
  return node(Op::MatMulNT, {a, b}, {sa[0], sb[0]});
}

int Builder::matmul_tn(int a, int b) {
  const Shape& sa = shape(a);
  const Shape& sb = shape(b);
  if (sa.size() != 2 || sb.size() != 2 || sa[0] != sb[0]) bad("MatMulTN: [K, M]^T @ [K, N]");
  return node(Op::MatMulTN, {a, b}, {sa[1], sb[1]});
}

int Builder::slice_cols(int x, std::int64_t begin, std::int64_t len) {
  const Shape& s = shape(x);
  if (s.size() != 2 || !s[1].is_constant() || begin < 0 || begin + len > s[1].coef) bad("SliceCols: aralik");
  return node(Op::SliceCols, {x}, {s[0], Dim::constant(len)}, DType::F32, 0.0, {begin, len});
}

int Builder::concat_cols(const std::vector<int>& xs) {
  std::int64_t cols = 0;
  for (int x : xs) {
    if (shape(x).size() != 2 || shape(x)[0] != shape(xs[0])[0] || !shape(x)[1].is_constant()) bad("ConcatCols");
    cols += shape(x)[1].coef;
  }
  return node(Op::ConcatCols, xs, {shape(xs[0])[0], Dim::constant(cols)});
}

int Builder::split_heads(int x, std::int64_t heads, std::int64_t d, int sb, int st) {
  const Shape& s = shape(x);
  if (s.size() != 2 || s[1] != Dim::constant(heads * d)) bad("SplitHeads: [N, h*D]");
  return node(Op::SplitHeads, {x}, {Dim::symbol(sb) * Dim::constant(heads), Dim::symbol(st), Dim::constant(d)},
              DType::F32, 0.0, {heads, d, sb, st});
}

int Builder::repeat_kv(int x, std::int64_t groups) {
  const Shape& s = shape(x);
  if (s.size() != 3) bad("RepeatKV: [G, T, D]");
  if (groups == 1) return x;
  return node(Op::RepeatKV, {x}, {s[0] * Dim::constant(groups), s[1], s[2]}, DType::F32, 0.0, {groups});
}

int Builder::bmm_nt(int a, int b) {
  const Shape& sa = shape(a);
  const Shape& sb = shape(b);
  if (sa.size() != 3 || sb.size() != 3 || sa[0] != sb[0] || sa[2] != sb[2]) bad("BMMNT");
  return node(Op::BMMNT, {a, b}, {sa[0], sa[1], sb[1]});
}

int Builder::bmm(int a, int b) {
  const Shape& sa = shape(a);
  const Shape& sb = shape(b);
  if (sa.size() != 3 || sb.size() != 3 || sa[0] != sb[0] || sa[2] != sb[1]) bad("BMM");
  return node(Op::BMM, {a, b}, {sa[0], sa[1], sb[2]});
}

int Builder::causal_mask(int x) { return node(Op::CausalMask, {x}, shape(x)); }
int Builder::softmax(int x) { return node(Op::Softmax, {x}, shape(x)); }

int Builder::merge_heads(int x, std::int64_t heads, std::int64_t d, int sb, int st) {
  const Shape& s = shape(x);
  if (s.size() != 3) bad("MergeHeads: [B*h, T, D]");
  return node(Op::MergeHeads, {x}, {Dim::symbol(sb) * Dim::symbol(st), Dim::constant(heads * d)}, DType::F32, 0.0,
              {heads, d, sb, st});
}

int Builder::layer_norm(int x, int w, int b, double eps) {
  const Shape& s = shape(x);
  if (s.size() != 2 || shape(w) != Shape{s[1]}) bad("LayerNorm: x [R, C], w [C]");
  std::vector<int> in{x, w};
  if (b >= 0) in.push_back(b);
  return node(Op::LayerNorm, in, s, DType::F32, eps, {b >= 0 ? 1 : 0});
}

int Builder::rms_norm(int x, int w, double eps) {
  const Shape& s = shape(x);
  if (s.size() != 2 || shape(w) != Shape{s[1]}) bad("RmsNorm: x [R, C], w [C]");
  return node(Op::RmsNorm, {x, w}, s, DType::F32, eps);
}

int Builder::rope(int x, std::int64_t heads, std::int64_t d, std::int64_t rot, double theta, int st) {
  return node(Op::Rope, {x}, shape(x), DType::F32, theta, {heads, d, rot, st});
}

int Builder::embedding(int table, int ids) {
  const Shape& t = shape(table);
  if (t.size() != 2 || shape(ids).size() != 1) bad("Embedding: tablo [V, C], kimlik [N]");
  return node(Op::Embedding, {table, ids}, {shape(ids)[0], t[1]});
}

int Builder::cross_entropy(int logits, int tokens, std::int64_t valid_vocab, int st) {
  return node(Op::CrossEntropy, {logits, tokens}, {}, DType::F32, 0.0, {valid_vocab, st});
}

// ==========================================================================================
// lm izleme
// ==========================================================================================

TracedLm trace_lm(const lm::LmConfig& c) {
  auto no = [](const char* what) { bad(std::string("lm alani desteklenmiyor: ") + what); };
  if (c.init == lm::InitKind::MuP) no("muP");
  if (!c.prenorm) no("son-normalizasyon");
  if (c.parallel_block) no("paralel blok");
  if (c.window > 0) no("kayan pencere");
  if (c.attn_softcap > 0.0f) no("softcap");
  if (c.resid_dropout > 0.0f || c.attn_dropout > 0.0f) no("dropout");
  if (c.qk_norm) no("qk_norm");
  if (c.pos == lm::PosKind::Alibi || c.pos == lm::PosKind::Sinusoidal) no("ALiBi / sinuzoidal");
  if (c.pos == lm::PosKind::Rope && c.rope_scaling.type != lm::RopeScaling::Type::None) no("RoPE olcekleme");
  if (c.recompute != lm::RecomputeKind::None) no("recompute (derleyici min-cut kullanir)");
  if (c.moe_experts > 0) no("MoE (derleyici IR'de MoE op yok)");

  TracedLm tr;
  Graph& g = tr.graph;
  Builder b(g);
  const int sb = g.symbol("B");
  const int st = g.symbol("T");
  tr.sym_b = sb;
  tr.sym_t = st;
  const Dim N = Dim::symbol(sb) * Dim::symbol(st);
  const std::int64_t d = c.d_model, H = c.n_head, Hk = c.kv_heads(), D = c.hdim(), F = c.d_ff;
  const std::int64_t Vp = c.padded_vocab();
  const bool ln = c.norm == lm::NormKind::LayerNorm;
  const bool ln_bias = ln && c.bias;
  auto C = [](std::int64_t v) { return Dim::constant(v); };

  tr.tokens = b.input("tokens", {N}, DType::I32);
  const int wte = b.param("wte", {C(Vp), C(d)});
  int x = b.embedding(wte, tr.tokens);
  if (c.pos == lm::PosKind::Learned) {
    const int wpe = b.param("wpe", {C(c.max_position), C(d)});
    x = b.binary(Op::Add, x, b.embedding(wpe, b.positions(sb, st)));
  }
  auto norm = [&](int v, const std::string& wn, const std::string& bn) {
    const int w = b.param(wn, {C(d)});
    if (!ln) return b.rms_norm(v, w, c.norm_eps);
    return b.layer_norm(v, w, ln_bias ? b.param(bn, {C(d)}) : -1, c.norm_eps);
  };
  for (int l = 0; l < c.n_layer; ++l) {
    const std::string p = "h" + std::to_string(l) + ".";
    const int h = norm(x, p + "norm1.weight", p + "norm1.bias");
    const std::int64_t qkvw = (H + 2 * Hk) * D;
    int qkv = b.matmul(h, b.param(p + "attn.qkv.weight", {C(d), C(qkvw)}));
    if (c.bias || c.qkv_bias) qkv = b.bias_add(qkv, b.param(p + "attn.qkv.bias", {C(qkvw)}));
    int q = b.slice_cols(qkv, 0, H * D);
    int k = b.slice_cols(qkv, H * D, Hk * D);
    const int v = b.slice_cols(qkv, (H + Hk) * D, Hk * D);
    if (c.pos == lm::PosKind::Rope) {
      q = b.rope(q, H, D, c.rot_dim(), c.rope_theta, st);
      k = b.rope(k, Hk, D, c.rot_dim(), c.rope_theta, st);
    }
    const int qh = b.split_heads(q, H, D, sb, st);
    const int kh = b.repeat_kv(b.split_heads(k, Hk, D, sb, st), H / Hk);
    const int vh = b.repeat_kv(b.split_heads(v, Hk, D, sb, st), H / Hk);
    const int s = b.causal_mask(b.unary(Op::Scale, b.bmm_nt(qh, kh), 1.0 / std::sqrt(static_cast<double>(D))));
    const int att = b.merge_heads(b.bmm(b.softmax(s), vh), H, D, sb, st);
    int o = b.matmul(att, b.param(p + "attn.o.weight", {C(H * D), C(d)}));
    if (c.bias) o = b.bias_add(o, b.param(p + "attn.o.bias", {C(d)}));
    const int r = b.binary(Op::Add, x, o);
    const int h2 = norm(r, p + "norm2.weight", p + "norm2.bias");
    const std::int64_t upw = c.gated_mlp() ? 2 * F : F;
    int up = b.matmul(h2, b.param(p + "mlp.up.weight", {C(d), C(upw)}));
    if (c.bias) up = b.bias_add(up, b.param(p + "mlp.up.bias", {C(upw)}));
    int act = 0;
    switch (c.mlp) {
      case lm::MlpKind::Gelu:
        act = b.unary(Op::Gelu, up);
        break;
      case lm::MlpKind::SwiGlu:
        act = b.binary(Op::Mul, b.unary(Op::Silu, b.slice_cols(up, 0, F)), b.slice_cols(up, F, F));
        break;
      case lm::MlpKind::GeGlu:
        act = b.binary(Op::Mul, b.unary(Op::Gelu, b.slice_cols(up, 0, F)), b.slice_cols(up, F, F));
        break;
    }
    int dn = b.matmul(act, b.param(p + "mlp.down.weight", {C(F), C(d)}));
    if (c.bias) dn = b.bias_add(dn, b.param(p + "mlp.down.bias", {C(d)}));
    x = b.binary(Op::Add, r, dn);
  }
  const int hf = norm(x, "norm_f.weight", "norm_f.bias");
  const int head = c.tie_embeddings ? wte : b.param("lm_head.weight", {C(Vp), C(d)});
  const int logits = b.matmul_nt(hf, head);
  tr.loss = b.cross_entropy(logits, tr.tokens, c.vocab_size, st);
  g.outputs = {tr.loss};
  g.output_names = {"loss"};
  return tr;
}

}  // namespace ir
}  // namespace nexus
