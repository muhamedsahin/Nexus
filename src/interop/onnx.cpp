/**
 * @file onnx.cpp
 * @brief FAZ 17.4: IR -> ONNX disa aktarma (bkz. nexus/interop/onnx.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/onnx.hpp"

#include "nexus/compile/trace.hpp"
#include "nexus/interop/json.hpp"
#include "nexus/lm/rope.hpp"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>

namespace nexus {
namespace interop {

namespace {

// ---- protobuf kodlayici ----------------------------------------------------------------
struct Pb {
  std::string b;
  void varint(std::uint64_t v) {
    while (v >= 0x80) {
      b += static_cast<char>((v & 0x7F) | 0x80);
      v >>= 7;
    }
    b += static_cast<char>(v);
  }
  void key(int field, int wire) { varint((static_cast<std::uint64_t>(field) << 3) | static_cast<std::uint64_t>(wire)); }
  void i64(int field, std::int64_t v) {
    key(field, 0);
    varint(static_cast<std::uint64_t>(v));
  }
  void bytes(int field, const std::string& s) {
    key(field, 2);
    varint(s.size());
    b += s;
  }
  void msg(int field, const Pb& m) { bytes(field, m.b); }
  void f32(int field, float v) {
    key(field, 5);
    char c[4];
    std::memcpy(c, &v, 4);
    b.append(c, 4);
  }
};

enum : int { FLOAT = 1, INT64 = 7, BOOL_T = 9 };

struct Builder {
  Pb graph;
  std::vector<Pb> inits;
  int nodes = 0, counter = 0;
  OnnxOptions opt;
  std::string data_name;
  std::ofstream* data = nullptr;
  std::uint64_t data_bytes = 0;

  std::string fresh(const std::string& hint) { return hint + "_" + std::to_string(counter++); }

  void tensor(const std::string& name, int type, const std::vector<std::int64_t>& dims, const void* raw, std::size_t bytes) {
    Pb t;
    for (auto d : dims) t.i64(1, d);
    t.i64(2, type);
    t.bytes(8, name);
    if (opt.external_data && data != nullptr && bytes >= opt.external_threshold) {
      const std::uint64_t pad = (4096 - data_bytes % 4096) % 4096;
      static const char zeros[4096] = {};
      data->write(zeros, static_cast<std::streamsize>(pad));
      data_bytes += pad;
      const std::uint64_t off = data_bytes;
      data->write(static_cast<const char*>(raw), static_cast<std::streamsize>(bytes));
      data_bytes += bytes;
      auto kv = [&](const char* k, const std::string& v) {
        Pb e;
        e.bytes(1, k);
        e.bytes(2, v);
        t.msg(13, e);
      };
      kv("location", data_name);
      kv("offset", std::to_string(off));
      kv("length", std::to_string(bytes));
      t.i64(14, 1);  // EXTERNAL
    } else {
      t.bytes(9, std::string(static_cast<const char*>(raw), bytes));
    }
    inits.push_back(std::move(t));
  }
  std::string ints(const std::vector<std::int64_t>& v, const std::string& hint = "c") {
    const std::string n = fresh(hint);
    tensor(n, INT64, {static_cast<std::int64_t>(v.size())}, v.data(), v.size() * 8);
    return n;
  }
  std::string scalar_f(float v) {
    const std::string n = fresh("f");
    tensor(n, FLOAT, {}, &v, 4);
    return n;
  }
  std::string scalar_i(std::int64_t v) {
    const std::string n = fresh("i");
    tensor(n, INT64, {}, &v, 8);
    return n;
  }
  struct Attr {
    std::string name;
    int kind;  // 1 float, 2 int, 7 ints
    float f = 0;
    std::int64_t i = 0;
    std::vector<std::int64_t> is;
  };
  std::string node(const std::string& op, const std::vector<std::string>& in, std::vector<Attr> attrs = {}, std::string out = "") {
    if (out.empty()) out = fresh(op);
    Pb n;
    for (const auto& s : in) n.bytes(1, s);
    n.bytes(2, out);
    n.bytes(3, "n" + std::to_string(nodes++));
    n.bytes(4, op);
    for (const Attr& a : attrs) {
      Pb at;
      at.bytes(1, a.name);
      if (a.kind == 1) at.f32(2, a.f);
      if (a.kind == 2) at.i64(3, a.i);
      if (a.kind == 7)
        for (auto v : a.is) at.i64(8, v);
      at.i64(20, a.kind == 1 ? 1 : a.kind == 2 ? 2 : 7);
      n.msg(5, at);
    }
    graph.msg(1, n);
    return out;
  }
  void value_info(int field, const std::string& name, int type, const std::vector<std::string>& dims) {
    Pb shape;
    for (const auto& d : dims) {
      Pb dim;
      if (!d.empty() && std::isdigit(static_cast<unsigned char>(d[0]))) dim.i64(1, std::stoll(d));
      else dim.bytes(2, d);
      shape.msg(1, dim);
    }
    Pb tt;
    tt.i64(1, type);
    tt.msg(2, shape);
    Pb tp;
    tp.msg(1, tt);
    Pb vi;
    vi.bytes(1, name);
    vi.msg(2, tp);
    graph.msg(field, vi);
  }
};

[[noreturn]] void bad(const std::string& w) { throw InteropError("ONNX: " + w); }

}  // namespace

void export_onnx(const lm::LanguageModel& m, const std::string& path, const OnnxOptions& opt, OnnxStats* stats) {
  const auto t0 = std::chrono::steady_clock::now();
  if (opt.opset != 17 && opt.opset != 18) bad("opset 17 ya da 18");
  const lm::LmConfig& c = m.config();
  ir::TracedLm tr = ir::trace_lm(c);
  const ir::Graph& g = tr.graph;
  Builder b;
  b.opt = opt;
  std::ofstream data;
  if (opt.external_data) {
    b.data_name = std::filesystem::path(path).filename().string() + ".data";
    data.open(path + ".data", std::ios::binary | std::ios::trunc);
    if (!data) bad("dis veri dosyasi acilamadi");
    b.data = &data;
  }
  // logit dugumu: CrossEntropy'nin girdisi
  int logits = -1;
  for (const auto& n : g.nodes)
    if (n.op == ir::Op::CrossEntropy) logits = n.in.at(0);
  if (logits < 0) bad("izlenen grafikte logit bulunamadi");
  std::vector<std::string> name(g.nodes.size());
  std::map<std::string, bool> emitted;
  // ortak sekil degerleri
  const std::string tok_shape = b.node("Shape", {"tokens"});
  const std::string Bv = b.node("Slice", {tok_shape, b.ints({0}), b.ints({1})});
  const std::string Tv = b.node("Slice", {tok_shape, b.ints({1}), b.ints({2})});
  const std::string flat = b.node("Reshape", {"tokens", b.ints({-1})});
  std::string positions;
  std::string causal;  // [T, T] bool: j > i
  const std::string Ts = b.node("Squeeze", {Tv, b.ints({0})});
  auto get_positions = [&] {
    if (!positions.empty()) return positions;
    const std::string N = b.node("Mul", {b.node("Squeeze", {Bv, b.ints({0})}), Ts});
    const std::string r = b.node("Range", {b.scalar_i(0), N, b.scalar_i(1)});
    positions = b.node("Mod", {r, Ts});
    return positions;
  };
  auto get_causal = [&] {
    if (!causal.empty()) return causal;
    const std::string r = b.node("Range", {b.scalar_i(0), Ts, b.scalar_i(1)});
    causal = b.node("Greater", {b.node("Unsqueeze", {r, b.ints({0})}), b.node("Unsqueeze", {r, b.ints({1})})});
    return causal;
  };
  auto shape_of = [&](std::vector<std::string> parts) { return b.node("Concat", parts, {{"axis", 2, 0, 0, {}}}); };
  auto reduce_mean = [&](const std::string& x) {
    if (opt.opset >= 18) return b.node("ReduceMean", {x, b.ints({-1})}, {{"keepdims", 2, 0, 1, {}}});
    return b.node("ReduceMean", {x}, {{"axes", 7, 0, 0, {-1}}, {"keepdims", 2, 0, 1, {}}});
  };
  int initializers_params = 0;
  auto param = [&](const std::string& pname) {
    if (!emitted[pname]) {
      const lm::NamedParam* p = m.find(pname);
      if (p == nullptr) bad("parametre yok " + pname);
      std::vector<std::int64_t> dims(p->value.shape().dims.begin(), p->value.shape().dims.end());
      b.tensor(pname, FLOAT, dims, p->value.values().data(), p->value.values().size() * 4);
      emitted[pname] = true;
      ++initializers_params;
    }
    return pname;
  };
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const ir::Node& n = g.nodes[i];
    auto in = [&](int k) { return name.at(static_cast<std::size_t>(n.in.at(static_cast<std::size_t>(k)))); };
    using ir::Op;
    switch (n.op) {
      case Op::Input:
        name[i] = flat;
        break;
      case Op::Param:
        name[i] = param(n.name);
        break;
      case Op::Positions:
        name[i] = get_positions();
        break;
      case Op::Embedding:
        name[i] = b.node("Gather", {in(0), in(1)}, {{"axis", 2, 0, 0, {}}});
        break;
      case Op::Add:
      case Op::BiasAdd:
        name[i] = b.node("Add", {in(0), in(1)});
        break;
      case Op::Mul:
        name[i] = b.node("Mul", {in(0), in(1)});
        break;
      case Op::Sub:
        name[i] = b.node("Sub", {in(0), in(1)});
        break;
      case Op::MatMul:
        name[i] = b.node("MatMul", {in(0), in(1)});
        break;
      case Op::MatMulNT:
        name[i] = b.node("Gemm", {in(0), in(1)}, {{"transB", 2, 0, 1, {}}});
        break;
      case Op::SliceCols:
        name[i] = b.node("Slice", {in(0), b.ints({n.ints[0]}), b.ints({n.ints[0] + n.ints[1]}), b.ints({1})});
        break;
      case Op::ConcatCols: {
        std::vector<std::string> xs;
        for (std::size_t k = 0; k < n.in.size(); ++k) xs.push_back(in(static_cast<int>(k)));
        name[i] = b.node("Concat", xs, {{"axis", 2, 0, 1, {}}});
        break;
      }
      case Op::RmsNorm: {
        const std::string x = in(0);
        const std::string ms = reduce_mean(b.node("Mul", {x, x}));
        const std::string den = b.node("Sqrt", {b.node("Add", {ms, b.scalar_f(static_cast<float>(n.a))})});
        name[i] = b.node("Mul", {b.node("Div", {x, den}), in(1)});
        break;
      }
      case Op::LayerNorm: {
        std::string bias;
        if (n.in.size() > 2) {
          bias = in(2);
        } else {
          const std::int64_t C = n.shape[1].coef;
          std::vector<float> z(static_cast<std::size_t>(C), 0.0f);
          bias = b.fresh("ln_zero_bias");
          b.tensor(bias, FLOAT, {C}, z.data(), z.size() * 4);
        }
        name[i] = b.node("LayerNormalization", {in(0), in(1), bias}, {{"axis", 2, 0, -1, {}}, {"epsilon", 1, static_cast<float>(n.a), 0, {}}});
        break;
      }
      case Op::Gelu: {
        const std::string x = in(0);
        const std::string e = b.node("Erf", {b.node("Mul", {x, b.scalar_f(0.70710678118654752f)})});
        name[i] = b.node("Mul", {b.node("Mul", {x, b.scalar_f(0.5f)}), b.node("Add", {e, b.scalar_f(1.0f)})});
        break;
      }
      case Op::GeluTanh: {
        const std::string x = in(0);
        const std::string x3 = b.node("Mul", {b.node("Mul", {x, x}), x});
        const std::string inner = b.node("Mul", {b.node("Add", {x, b.node("Mul", {x3, b.scalar_f(0.044715f)})}), b.scalar_f(0.7978845608028654f)});
        name[i] = b.node("Mul", {b.node("Mul", {x, b.scalar_f(0.5f)}), b.node("Add", {b.node("Tanh", {inner}), b.scalar_f(1.0f)})});
        break;
      }
      case Op::Silu:
        name[i] = b.node("Mul", {in(0), b.node("Sigmoid", {in(0)})});
        break;
      case Op::Scale:
        name[i] = b.node("Mul", {in(0), b.scalar_f(static_cast<float>(n.a))});
        break;
      case Op::Rope: {
        const std::int64_t h = n.ints[0], D = n.ints[1], rot = n.ints[2], half = rot / 2;
        lm::RopeCache cache(static_cast<int>(rot), static_cast<float>(n.a), c.rope_scaling, c.max_position);
        cache.ensure(c.max_position);
        std::vector<float> cs(static_cast<std::size_t>(c.max_position * half)), sn(cs.size());
        for (int p = 0; p < c.max_position; ++p) {
          std::copy_n(cache.cos(p), half, cs.begin() + static_cast<std::ptrdiff_t>(p) * half);
          std::copy_n(cache.sin(p), half, sn.begin() + static_cast<std::ptrdiff_t>(p) * half);
        }
        const std::string ct = "rope_cos_" + std::to_string(rot) + "_" + std::to_string(static_cast<long long>(n.a));
        const std::string st = "rope_sin_" + std::to_string(rot) + "_" + std::to_string(static_cast<long long>(n.a));
        if (!emitted[ct]) {
          b.tensor(ct, FLOAT, {c.max_position, half}, cs.data(), cs.size() * 4);
          b.tensor(st, FLOAT, {c.max_position, half}, sn.data(), sn.size() * 4);
          emitted[ct] = emitted[st] = true;
        }
        const std::string pos = get_positions();
        const std::string cg = b.node("Unsqueeze", {b.node("Gather", {ct, pos}, {{"axis", 2, 0, 0, {}}}), b.ints({1})});
        const std::string sg = b.node("Unsqueeze", {b.node("Gather", {st, pos}, {{"axis", 2, 0, 0, {}}}), b.ints({1})});
        const std::string x3 = b.node("Reshape", {in(0), b.ints({-1, h, D})});
        const std::string x1 = b.node("Slice", {x3, b.ints({0}), b.ints({half}), b.ints({2})});
        const std::string x2 = b.node("Slice", {x3, b.ints({half}), b.ints({rot}), b.ints({2})});
        const std::string o1 = b.node("Sub", {b.node("Mul", {x1, cg}), b.node("Mul", {x2, sg})});
        const std::string o2 = b.node("Add", {b.node("Mul", {x2, cg}), b.node("Mul", {x1, sg})});
        std::vector<std::string> parts = {o1, o2};
        if (rot < D) parts.push_back(b.node("Slice", {x3, b.ints({rot}), b.ints({D}), b.ints({2})}));
        name[i] = b.node("Reshape", {b.node("Concat", parts, {{"axis", 2, 0, 2, {}}}), b.ints({-1, h * D})});
        break;
      }
      case Op::SplitHeads: {
        const std::int64_t h = n.ints[0], D = n.ints[1];
        const std::string x4 = b.node("Reshape", {in(0), shape_of({b.ints({-1}), Tv, b.ints({h, D})})});
        const std::string tr4 = b.node("Transpose", {x4}, {{"perm", 7, 0, 0, {0, 2, 1, 3}}});
        name[i] = b.node("Reshape", {tr4, shape_of({b.ints({-1}), Tv, b.ints({D})})});
        break;
      }
      case Op::MergeHeads: {
        const std::int64_t h = n.ints[0], D = n.ints[1];
        const std::string x4 = b.node("Reshape", {in(0), shape_of({b.ints({-1, h}), Tv, b.ints({D})})});
        const std::string tr4 = b.node("Transpose", {x4}, {{"perm", 7, 0, 0, {0, 2, 1, 3}}});
        name[i] = b.node("Reshape", {tr4, b.ints({-1, h * D})});
        break;
      }
      case Op::RepeatKV: {
        const std::int64_t groups = n.ints[0];
        if (groups == 1) {
          name[i] = in(0);
          break;
        }
        const std::int64_t H = n.shape[0].coef, Hk = H / groups, D = n.shape[2].coef;
        const std::string x4 = b.node("Reshape", {in(0), shape_of({b.ints({-1, Hk}), Tv, b.ints({D})})});
        const std::string x5 = b.node("Unsqueeze", {x4, b.ints({2})});
        const std::string e = b.node("Expand", {x5, shape_of({Bv, b.ints({Hk, groups}), Tv, b.ints({D})})});
        name[i] = b.node("Reshape", {e, shape_of({b.ints({-1}), Tv, b.ints({D})})});
        break;
      }
      case Op::BMMNT:
        name[i] = b.node("MatMul", {in(0), b.node("Transpose", {in(1)}, {{"perm", 7, 0, 0, {0, 2, 1}}})});
        break;
      case Op::BMM:
        name[i] = b.node("MatMul", {in(0), in(1)});
        break;
      case Op::CausalMask:
        name[i] = b.node("Where", {get_causal(), b.scalar_f(-std::numeric_limits<float>::infinity()), in(0)});
        break;
      case Op::Softmax:
        name[i] = b.node("Softmax", {in(0)}, {{"axis", 2, 0, -1, {}}});
        break;
      case Op::CrossEntropy:
        break;  // disa aktarilmaz (cikti logit)
      default:
        bad(std::string("IR islemi desteklenmiyor: ") + ir::to_string(n.op));
    }
    if (static_cast<int>(i) == logits) break;
  }
  // [N, Vp] -> [B, T, V]
  const std::string sl = b.node("Slice", {name[static_cast<std::size_t>(logits)], b.ints({0}), b.ints({c.vocab_size}), b.ints({1})});
  b.node("Reshape", {sl, shape_of({Bv, Tv, b.ints({c.vocab_size})})}, {}, "logits");
  // grafik
  Pb graph;
  graph.b = b.graph.b;
  graph.bytes(2, "nexus_" + c.name);
  for (const Pb& t : b.inits) graph.msg(5, t);
  b.graph.b.clear();
  b.value_info(11, "tokens", INT64, {"batch", "seq"});
  b.value_info(12, "logits", FLOAT, {"batch", "seq", std::to_string(c.vocab_size)});
  graph.b += b.graph.b;
  Pb model;
  model.i64(1, 8);  // IR surumu 8 (opset 17/18)
  model.bytes(2, "nexus");
  model.bytes(3, "1.0");
  Pb ops;
  ops.bytes(1, "");
  ops.i64(2, opt.opset);
  model.msg(8, ops);
  model.msg(7, graph);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) bad("yazilamadi " + path);
  f.write(model.b.data(), static_cast<std::streamsize>(model.b.size()));
  if (!f) bad("yazma hatasi " + path);
  if (stats) {
    stats->model_bytes = model.b.size();
    stats->data_bytes = b.data_bytes;
    stats->nodes = b.nodes;
    stats->initializers = static_cast<int>(b.inits.size());
    stats->ir_nodes = logits + 1;
    stats->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  (void)initializers_params;
}

}  // namespace interop
}  // namespace nexus
