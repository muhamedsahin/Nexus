/**
 * @file compile.cpp
 * @brief FAZ 13.8: CompiledLm boru hatti, koruyucular (sembolik B/T), dokum; 13.7 StableHLO / HLO disa aktarma.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/compile.hpp"

#include "nexus/compile/trace.hpp"
#include "nexus/lm/model.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nexus {
namespace compile {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void dump(const CompileOptions& o, const std::string& stage, const ir::Graph& g) {
  if (o.dump_dir.empty()) return;
  std::filesystem::create_directories(o.dump_dir);
  std::ofstream(std::filesystem::path(o.dump_dir) / (stage + ".txt")) << g.to_text();
  std::ofstream(std::filesystem::path(o.dump_dir) / (stage + ".dot")) << g.to_dot();
}

}  // namespace

CompiledLm::CompiledLm(const lm::LanguageModel& model, CompileOptions options)
    : model_(&model), options_(std::move(options)), jit_(std::make_unique<ir::CpuJit>(options_.cache_dir)) {
  if (!options_.dump_dir.empty()) jit_->set_dump_dir(options_.dump_dir);
  if (options_.mode != "default" && options_.mode != "reduce-overhead" && options_.mode != "max-autotune") {
    throw std::invalid_argument("nexus::compile: mode 'default' | 'reduce-overhead' | 'max-autotune'");
  }
  for (const lm::NamedParam& p : model.parameters()) param_names_.push_back(p.name);
}

CompiledLm::~CompiledLm() = default;
CompiledLm::CompiledLm(CompiledLm&&) noexcept = default;

const ir::Graph& CompiledLm::joint_graph() const {
  if (!train_) throw std::logic_error("nexus::compile: once forward_backward cagrilmali");
  return train_->graph();
}

void CompiledLm::build(int batch, int time, bool train) {
  const bool have = train ? train_ != nullptr : eval_ != nullptr;
  if (have && (options_.dynamic || (batch == built_b_ && time == built_t_))) return;
  if (have) {
    stats_.recompiles += 1;  // dynamic = false koruyucu: yeni (B, T) -> yeniden izleme
    train_.reset();
    eval_.reset();
  }
  const lm::LmConfig& cfg = model_->config();
  auto t0 = Clock::now();
  ir::TracedLm tr = ir::trace_lm(cfg);
  forward_ = tr.graph;
  stats_.trace_ms += ms_since(t0);
  dump(options_, "00_trace", tr.graph);
  const ir::Bindings ex{{tr.sym_b, batch}, {tr.sym_t, time}};
  t0 = Clock::now();
  ir::Graph fwd = tr.graph;
  if (options_.patterns) {
    fwd = ir::rewrite_patterns(fwd, &stats_.patterns, options_.ce_chunk_rows, options_.pattern_mask);
    dump(options_, "01_patterns", fwd);
  }
  fwd = ir::constant_fold(ir::cse(fwd));
  dump(options_, "02_cse_fold", fwd);
  std::unique_ptr<ir::Graph> g;
  if (train) {
    // Eager karsiligi (IR cekirdek sayimi): yalniz dikkat SDPA cekirdegine (PyTorch eager da tek SDPA cagirir).
    stats_.kernels_eager = ir::build_joint(ir::rewrite_patterns(tr.graph, nullptr, options_.ce_chunk_rows, ir::kPatAttention))
                               .graph.kernel_count();
    ir::Joint joint = ir::build_joint(fwd);
    dump(options_, "03_joint", joint.graph);
    if (options_.recompute) {
      joint = ir::min_cut_recompute(joint, ex, &stats_.recompute);
      dump(options_, "04_min_cut", joint.graph);
    }
    g = std::make_unique<ir::Graph>(ir::cse(joint.graph));
    if (options_.patterns && options_.pattern_mask == ir::kPatAll) {
      *g = ir::fuse_epilogues(*g, &stats_.epilogues);
      dump(options_, "045_epilogues", *g);
    }
    ir::FusionStats fs;
    if (options_.fusion) {
      *g = ir::fuse_elementwise(*g, &fs);
      dump(options_, "05_fusion", *g);
    }
    *g = ir::schedule_communication(*g);
    stats_.fused_clusters = fs.clusters;
    stats_.kernels_compiled = g->kernel_count();
  } else {
    g = std::make_unique<ir::Graph>(options_.fusion ? ir::fuse_elementwise(fwd) : fwd);
  }
  stats_.passes_ms += ms_since(t0);
  t0 = Clock::now();
  const ir::JitStats before = jit_->stats();
  (train ? train_ : eval_) = std::make_unique<Executable>(std::move(*g), *jit_, options_);
  stats_.codegen_ms += ms_since(t0);
  stats_.compiler_invocations += jit_->stats().compiler_invocations - before.compiler_invocations;
  stats_.cache_hits += jit_->stats().cache_hits - before.cache_hits;
  built_b_ = batch;
  built_t_ = time;
}

float CompiledLm::forward_backward(const std::vector<std::int32_t>& tokens, int batch, int time,
                                   std::vector<std::vector<float>>* grads) {
  if (static_cast<int>(tokens.size()) != batch * time) throw std::invalid_argument("nexus::compile: tokens boyu B*T olmali");
  build(batch, time, true);
  std::map<std::string, const float*> params;
  for (const lm::NamedParam& p : model_->parameters()) params[p.name] = p.value.values().data();
  const ir::Bindings b{{0, batch}, {1, time}};
  std::vector<std::vector<float>> out = train_->run(b, params, {{"tokens", tokens.data()}});
  stats_.arena_bytes = train_->arena_bytes();
  stats_.naive_bytes = train_->naive_bytes();
  stats_.autotuned = train_->autotuned();
  if (grads) {
    grads->assign(param_names_.size(), {});
    const ir::Graph& g = train_->graph();
    for (std::size_t k = 1; k < g.output_names.size(); ++k) {
      const std::string name = g.output_names[k].substr(5);  // "grad:"
      for (std::size_t p = 0; p < param_names_.size(); ++p) {
        if (param_names_[p] == name) (*grads)[p] = std::move(out[k]);
      }
    }
  }
  return out[0][0];
}

float CompiledLm::loss(const std::vector<std::int32_t>& tokens, int batch, int time) {
  build(batch, time, false);
  std::map<std::string, const float*> params;
  for (const lm::NamedParam& p : model_->parameters()) params[p.name] = p.value.values().data();
  return eval_->run({{0, batch}, {1, time}}, params, {{"tokens", tokens.data()}})[0][0];
}

std::string CompiledLm::stablehlo() const { return export_stablehlo(forward_, {}); }

CompiledLm compile(const lm::LanguageModel& model, CompileOptions options) { return CompiledLm(model, std::move(options)); }

// ==========================================================================================
// 13.7 StableHLO / HLO
// ==========================================================================================

namespace {

std::string tensor_type(const ir::Node& n, const ir::Bindings& b) {
  std::string s = "tensor<";
  for (const ir::Dim& d : n.shape) {
    if (d.is_constant()) {
      s += std::to_string(d.coef);
    } else {
      bool bound = true;
      for (int sym : d.syms) bound = bound && b.count(sym);
      s += bound ? std::to_string(ir::eval(d, b)) : "?";
    }
    s += "x";
  }
  return s + (n.dtype == ir::DType::I32 ? "i32>" : "f32>");
}

}  // namespace

std::string export_stablehlo(const ir::Graph& g, const ir::Bindings& b) {
  std::ostringstream o;
  std::vector<std::string> name(g.nodes.size());
  std::vector<int> args;
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    if (g.nodes[i].op == ir::Op::Param || g.nodes[i].op == ir::Op::Input) {
      name[i] = "%arg" + std::to_string(args.size());
      args.push_back(static_cast<int>(i));
    } else {
      name[i] = "%v" + std::to_string(i);
    }
  }
  o << "// Nexus FAZ 13.7 StableHLO disa aktarimi (ozel cekirdekler stablehlo.custom_call)\n";
  o << "module @nexus_lm {\n  func.func @main(";
  for (std::size_t k = 0; k < args.size(); ++k) {
    const ir::Node& n = g.nodes[static_cast<std::size_t>(args[k])];
    o << (k ? ", " : "") << name[static_cast<std::size_t>(args[k])] << ": " << tensor_type(n, b) << " {nexus.name = \"" << n.name
      << "\"}";
  }
  o << ") -> (";
  for (std::size_t k = 0; k < g.outputs.size(); ++k) o << (k ? ", " : "") << tensor_type(g.nodes[static_cast<std::size_t>(g.outputs[k])], b);
  o << ") {\n";
  auto types_in = [&](const ir::Node& n) {
    std::string s = "(";
    for (std::size_t k = 0; k < n.in.size(); ++k) s += (k ? ", " : "") + tensor_type(g.nodes[static_cast<std::size_t>(n.in[k])], b);
    return s + ")";
  };
  auto args_of = [&](const ir::Node& n) {
    std::string s;
    for (std::size_t k = 0; k < n.in.size(); ++k) s += (k ? ", " : "") + name[static_cast<std::size_t>(n.in[k])];
    return s;
  };
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const ir::Node& n = g.nodes[i];
    const std::string ty = tensor_type(n, b);
    using ir::Op;
    auto elem = [&](const char* op) { o << "    " << name[i] << " = stablehlo." << op << " " << args_of(n) << " : " << ty << "\n"; };
    switch (n.op) {
      case Op::Param: case Op::Input:
        break;
      case Op::Const:
        o << "    " << name[i] << " = stablehlo.constant dense<" << n.a << "> : " << ty << "\n";
        break;
      case Op::Add: elem("add"); break;
      case Op::Sub: elem("subtract"); break;
      case Op::Mul: elem("multiply"); break;
      case Op::Div: elem("divide"); break;
      case Op::Neg: elem("negate"); break;
      case Op::Exp: elem("exponential"); break;
      case Op::Log: elem("log"); break;
      case Op::Tanh: elem("tanh"); break;
      case Op::Sigmoid: elem("logistic"); break;
      case Op::Sqrt: elem("sqrt"); break;
      case Op::Rsqrt: elem("rsqrt"); break;
      case Op::MatMul: case Op::MatMulNT:
        o << "    " << name[i] << " = \"stablehlo.dot_general\"(" << args_of(n)
          << ") {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = ["
          << (n.op == Op::MatMul ? 0 : 1) << "]>} : " << types_in(n) << " -> " << ty << "\n";
        break;
      case Op::Scale: {
        o << "    " << name[i] << "_c = stablehlo.constant dense<" << n.a << "> : " << ty << "\n";
        o << "    " << name[i] << " = stablehlo.multiply " << args_of(n) << ", " << name[i] << "_c : " << ty << "\n";
        break;
      }
      case Op::BiasAdd:
        o << "    " << name[i] << "_b = stablehlo.broadcast_in_dim " << name[static_cast<std::size_t>(n.in[1])]
          << ", dims = [1] : (" << tensor_type(g.nodes[static_cast<std::size_t>(n.in[1])], b) << ") -> " << ty << "\n";
        o << "    " << name[i] << " = stablehlo.add " << name[static_cast<std::size_t>(n.in[0])] << ", " << name[i] << "_b : " << ty
          << "\n";
        break;
      default: {
        std::string cfg;
        for (std::size_t k = 0; k < n.ints.size(); ++k) cfg += (k ? "," : "") + std::to_string(n.ints[k]);
        o << "    " << name[i] << " = stablehlo.custom_call @nexus." << ir::to_string(n.op) << "(" << args_of(n)
          << ") {backend_config = \"ints=" << cfg << ";a=" << n.a << ";index=" << n.index << "\"} : " << types_in(n) << " -> "
          << ty << "\n";
      }
    }
  }
  o << "    return ";
  for (std::size_t k = 0; k < g.outputs.size(); ++k) o << (k ? ", " : "") << name[static_cast<std::size_t>(g.outputs[k])];
  o << " : ";
  for (std::size_t k = 0; k < g.outputs.size(); ++k) o << (k ? ", " : "") << tensor_type(g.nodes[static_cast<std::size_t>(g.outputs[k])], b);
  o << "\n  }\n}\n";
  return o.str();
}

std::string export_hlo_text(const ir::Graph& g, const ir::Bindings& b) {
  std::ostringstream o;
  int param = 0;
  auto shape = [&](const ir::Node& n) {
    std::string s = "f32[";
    for (std::size_t k = 0; k < n.shape.size(); ++k) s += (k ? "," : "") + std::to_string(ir::eval(n.shape[k], b));
    return s + "]";
  };
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const ir::Node& n = g.nodes[i];
    const bool root = !g.outputs.empty() && g.outputs[0] == static_cast<int>(i);
    const std::string lhs = std::string(root ? "ROOT " : "") + "v" + std::to_string(i) + " = " + shape(n);
    switch (n.op) {
      case ir::Op::Param: case ir::Op::Input:
        o << lhs << " parameter(" << param++ << ")\n";
        break;
      case ir::Op::MatMul:
        o << lhs << " dot(v" << n.in[0] << ", v" << n.in[1] << "), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n";
        break;
      case ir::Op::Add:
        o << lhs << " add(v" << n.in[0] << ", v" << n.in[1] << ")\n";
        break;
      case ir::Op::Mul:
        o << lhs << " multiply(v" << n.in[0] << ", v" << n.in[1] << ")\n";
        break;
      default:
        throw std::runtime_error(std::string("Nexus HLO: desteklenmeyen op (StableHLO kullanin): ") + ir::to_string(n.op));
    }
  }
  return o.str();
}

}  // namespace compile
}  // namespace nexus
