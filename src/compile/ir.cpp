/**
 * @file ir.cpp
 * @brief FAZ 13.1: IR yardimcilari — sembolik boyutlar, metin/Graphviz dokumu, parmak izi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/ir.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace nexus {
namespace ir {

Dim operator*(const Dim& a, const Dim& b) {
  Dim r;
  r.coef = a.coef * b.coef;
  r.syms = a.syms;
  r.syms.insert(r.syms.end(), b.syms.begin(), b.syms.end());
  std::sort(r.syms.begin(), r.syms.end());
  return r;
}

std::int64_t eval(const Dim& d, const Bindings& b) {
  std::int64_t v = d.coef;
  for (int s : d.syms) {
    auto it = b.find(s);
    if (it == b.end()) throw std::runtime_error("Nexus IR: sembol baglanmamis (" + std::to_string(s) + ")");
    v *= it->second;
  }
  return v;
}

std::int64_t numel(const Shape& s, const Bindings& b) {
  std::int64_t n = 1;
  for (const Dim& d : s) n *= eval(d, b);
  return n;
}

const char* to_string(Op op) noexcept {
  switch (op) {
#define NX_OP(x) \
  case Op::x:    \
    return #x;
    NX_OP(Input) NX_OP(Param) NX_OP(Const) NX_OP(Positions) NX_OP(Add) NX_OP(Sub) NX_OP(Mul) NX_OP(Div) NX_OP(Neg)
    NX_OP(Exp) NX_OP(Log) NX_OP(Tanh) NX_OP(Sigmoid) NX_OP(Gelu) NX_OP(GeluTanh) NX_OP(Silu) NX_OP(Relu) NX_OP(Square)
    NX_OP(Sqrt) NX_OP(Rsqrt) NX_OP(Scale) NX_OP(AddScalar) NX_OP(GeluBwd) NX_OP(GeluTanhBwd) NX_OP(SiluBwd)
    NX_OP(SigmoidBwd) NX_OP(TanhBwd) NX_OP(BiasAdd) NX_OP(SumRows) NX_OP(MatMul) NX_OP(MatMulNT) NX_OP(MatMulTN) NX_OP(MatMulBias)
    NX_OP(SliceCols) NX_OP(ConcatCols) NX_OP(SplitHeads) NX_OP(RepeatKV) NX_OP(BMMNT) NX_OP(BMM) NX_OP(CausalMask)
    NX_OP(Softmax) NX_OP(MergeHeads) NX_OP(LayerNorm) NX_OP(LayerNormBwdX) NX_OP(LayerNormBwdW) NX_OP(RmsNorm)
    NX_OP(RmsNormBwdX) NX_OP(RmsNormBwdW) NX_OP(AddNorm) NX_OP(Rope) NX_OP(RopeBwd) NX_OP(Attention)
    NX_OP(AttentionBwd) NX_OP(Embedding) NX_OP(EmbeddingBwd) NX_OP(CrossEntropy) NX_OP(CrossEntropyBwd)
    NX_OP(FusedLinearCE) NX_OP(FusedLinearCEBwd) NX_OP(SwiGlu) NX_OP(SwiGluBwd) NX_OP(LinearBiasGelu) NX_OP(AttentionQKV) NX_OP(AttentionQKVBwd)
    NX_OP(AllGather) NX_OP(ReduceScatter) NX_OP(AllReduce) NX_OP(Output) NX_OP(Fused)
#undef NX_OP
  }
  return "?";
}

bool is_elementwise(Op op) noexcept {
  switch (op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Neg: case Op::Exp: case Op::Log:
    case Op::Tanh: case Op::Sigmoid: case Op::Gelu: case Op::GeluTanh: case Op::Silu: case Op::Relu:
    case Op::Square: case Op::Sqrt: case Op::Rsqrt: case Op::Scale: case Op::AddScalar: case Op::GeluBwd:
    case Op::GeluTanhBwd: case Op::SiluBwd: case Op::SigmoidBwd: case Op::TanhBwd: case Op::BiasAdd:
      return true;
    default:
      return false;
  }
}

int Graph::add(Node n) {
  for (int i : n.in) {
    if (i < 0 || i >= static_cast<int>(nodes.size())) throw std::runtime_error("Nexus IR: gecersiz girdi (topolojik degil)");
  }
  nodes.push_back(std::move(n));
  return static_cast<int>(nodes.size()) - 1;
}

int Graph::symbol(const std::string& name) {
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    if (symbols[i] == name) return static_cast<int>(i);
  }
  symbols.push_back(name);
  return static_cast<int>(symbols.size()) - 1;
}

int Graph::find_param(const std::string& name) const {
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].op == Op::Param && nodes[i].name == name) return static_cast<int>(i);
  }
  return -1;
}

std::vector<int> Graph::use_counts() const {
  std::vector<int> uses(nodes.size(), 0);
  for (const Node& n : nodes) {
    for (int i : n.in) uses[static_cast<std::size_t>(i)] += 1;
  }
  for (int o : outputs) uses[static_cast<std::size_t>(o)] += 1;
  return uses;
}

std::string shape_str(const Shape& s, const Graph& g) {
  std::string r = "[";
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (i) r += ", ";
    const Dim& d = s[i];
    std::string t;
    if (d.coef != 1 || d.syms.empty()) t = std::to_string(d.coef);
    for (int sym : d.syms) {
      if (!t.empty()) t += "*";
      t += sym < static_cast<int>(g.symbols.size()) ? g.symbols[static_cast<std::size_t>(sym)] : "s" + std::to_string(sym);
    }
    r += t;
  }
  return r + "]";
}

std::string Graph::to_text() const {
  std::ostringstream o;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const Node& n = nodes[i];
    o << "%" << i << " = " << to_string(n.op);
    if (n.op == Op::Output) o << "." << n.index;
    o << "(";
    for (std::size_t j = 0; j < n.in.size(); ++j) o << (j ? ", " : "") << "%" << n.in[j];
    o << ")";
    if (!n.ints.empty()) {
      o << " {";
      for (std::size_t j = 0; j < n.ints.size(); ++j) o << (j ? "," : "") << n.ints[j];
      o << "}";
    }
    if (n.a != 0.0) o << " a=" << n.a;
    o << " : " << (n.dtype == DType::I32 ? "i32" : "f32") << shape_str(n.shape, *this);
    if (!n.name.empty()) o << "  // " << n.name;
    if (n.op == Op::Fused && n.fused) o << "  // " << n.fused->insts.size() << " op, " << n.fused->outputs.size() << " cikti";
    o << "\n";
  }
  o << "outputs:";
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    o << " %" << outputs[i];
    if (i < output_names.size()) o << "(" << output_names[i] << ")";
  }
  o << "\n";
  return o.str();
}

std::string Graph::to_dot() const {
  std::ostringstream o;
  o << "digraph nexus_ir {\n  node [shape=box, fontsize=10];\n";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const Node& n = nodes[i];
    const char* color = n.op == Op::Fused ? "lightblue" : (n.op == Op::Param || n.op == Op::Input) ? "lightgrey" : "white";
    o << "  n" << i << " [label=\"%" << i << " " << to_string(n.op) << "\\n" << shape_str(n.shape, *this)
      << "\", style=filled, fillcolor=" << color << "];\n";
    for (int j : n.in) o << "  n" << j << " -> n" << i << ";\n";
  }
  o << "}\n";
  return o.str();
}

std::string Graph::fingerprint() const {
  // FNV-1a 64 bit, op + girdiler + oznitelikler + sembolik sekil.
  std::uint64_t h = 1469598103934665603ULL;
  auto mix = [&](std::uint64_t v) {
    for (int k = 0; k < 8; ++k) {
      h ^= (v >> (8 * k)) & 0xFF;
      h *= 1099511628211ULL;
    }
  };
  for (const Node& n : nodes) {
    mix(static_cast<std::uint64_t>(n.op));
    for (int i : n.in) mix(static_cast<std::uint64_t>(i));
    for (std::int64_t v : n.ints) mix(static_cast<std::uint64_t>(v));
    std::uint64_t bits = 0;
    static_assert(sizeof(double) == 8);
    std::memcpy(&bits, &n.a, 8);
    mix(bits);
    mix(static_cast<std::uint64_t>(n.index));
    for (const Dim& d : n.shape) {
      mix(static_cast<std::uint64_t>(d.coef));
      for (int s : d.syms) mix(static_cast<std::uint64_t>(s) + 77);
    }
    if (n.fused) {
      for (const FusedOp::Inst& in : n.fused->insts) {
        mix(static_cast<std::uint64_t>(in.op) + 1000);
        for (int x : in.args) mix(static_cast<std::uint64_t>(x + 100000));
      }
    }
    for (char c : n.name) mix(static_cast<std::uint64_t>(c));
  }
  for (int o : outputs) mix(static_cast<std::uint64_t>(o) + 555);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

int Graph::kernel_count() const {
  int k = 0;
  for (const Node& n : nodes) {
    if (n.op != Op::Input && n.op != Op::Param && n.op != Op::Const && n.op != Op::Output) k += 1;
  }
  return k;
}

}  // namespace ir
}  // namespace nexus
