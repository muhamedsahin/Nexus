/**
 * @file pjrt.cpp
 * @brief FAZ 19.E: Nexus StableHLO CPU reference interpreter (not OpenXLA),
 *        DeviceMesh -> GSPMD sharding export and legacy PjrtDriver.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/backend/pjrt.hpp"
#include "pjrt_plugin.hpp"

#include "nexus/attention/cpu.hpp"
#include "nexus/backend/backend.hpp"
#include "nexus/compile/compile.hpp"
#include "nexus/core/error.hpp"
#include "nexus/cpu/gemm.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nexus::backend::pjrt {

// ============================================================================
//  1) DeviceMesh + DTensor -> GSPMD mhlo.sharding ve StableHLO SPMD ihraci
// ============================================================================

std::string to_mhlo_sharding(const std::vector<std::int64_t>& tensor_shape,
                             const dist::DeviceMesh& mesh,
                             const dist::Placements& placements) {
  dist::DTensor::validate(tensor_shape, mesh, placements);
  bool all_rep = true;
  bool any_partial = false;
  for (const auto& p : placements) {
    if (!p.is_replicate()) all_rep = false;
    if (p.is_partial()) any_partial = true;
  }
  if (all_rep) {
    return "{replicated}";
  }
  if (any_partial) {
    throw TensorError("Nexus StableHLO: Partial(sum) must be reduced before sharding export");
  }
  const int rank = static_cast<int>(tensor_shape.size());
  std::vector<int> tile_dims(static_cast<std::size_t>(rank), 1);
  int replicated_way = 1;
  for (int m = 0; m < mesh.ndim(); ++m) {
    const int msize = mesh.size(m);
    if (placements[static_cast<std::size_t>(m)].is_shard()) {
      const int d = placements[static_cast<std::size_t>(m)].dim;
      tile_dims[static_cast<std::size_t>(d)] *= msize;
    } else {
      replicated_way *= msize;
    }
  }
  std::ostringstream o;
  o << "{devices=[";
  for (int d = 0; d < rank; ++d) {
    if (d > 0) o << ",";
    o << tile_dims[static_cast<std::size_t>(d)];
  }
  if (replicated_way > 1) {
    o << "," << replicated_way;
  }
  o << "]";
  const int total_devices = mesh.world()->size();
  // XLA orders tile coordinates by tensor dimension, then the replication axis.
  // Mesh dimensions may be ordered differently (e.g. TP before DP).
  std::vector<std::pair<std::vector<int>, int>> ordered;
  for (int r = 0; r < total_devices; ++r) {
    int remaining = r;
    std::vector<int> coord(static_cast<std::size_t>(mesh.ndim()));
    for (int m = mesh.ndim() - 1; m >= 0; --m) {
      coord[static_cast<std::size_t>(m)] = remaining % mesh.size(m);
      remaining /= mesh.size(m);
    }
    std::vector<int> key(static_cast<std::size_t>(rank) + 1, 0);
    for (int m = 0; m < mesh.ndim(); ++m) {
      const auto& p = placements[static_cast<std::size_t>(m)];
      if (p.is_shard()) key[static_cast<std::size_t>(p.dim)] = coord[static_cast<std::size_t>(m)];
      else key.back() = key.back() * mesh.size(m) + coord[static_cast<std::size_t>(m)];
    }
    ordered.emplace_back(std::move(key), r);
  }
  std::sort(ordered.begin(), ordered.end());
  for (int r = 0; r < total_devices; ++r) o << (r == 0 ? "" : ",") << ordered[static_cast<std::size_t>(r)].second;
  if (replicated_way > 1) {
    o << " last_tile_dim_replicate";
  }
  o << "}";
  return o.str();
}

std::string export_stablehlo_spmd(const ir::Graph& g, const ir::Bindings& b,
                                  const std::map<std::string, std::string>& arg_shardings,
                                  const std::vector<std::string>& result_shardings) {
  auto dim_str = [&](const ir::Dim& d) -> std::string {
    if (d.is_constant()) return std::to_string(d.coef);
    if (!b.empty()) {
      bool all_bound = true;
      for (int s : d.syms) {
        if (b.find(s) == b.end()) all_bound = false;
      }
      if (all_bound) return std::to_string(ir::eval(d, b));
    }
    return "?";
  };
  auto tensor_type = [&](const ir::Node& n) {
    std::string s = "tensor<";
    for (std::size_t k = 0; k < n.shape.size(); ++k) s += dim_str(n.shape[k]) + "x";
    s += (n.dtype == ir::DType::I32 ? "i32>" : "f32>");
    return s;
  };

  std::ostringstream o;
  o << std::setprecision(std::numeric_limits<float>::max_digits10);
  o << "module @nexus_spmd {\n  func.func @main(";
  std::vector<std::string> name(g.nodes.size());
  int arg_idx = 0;
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const ir::Node& n = g.nodes[i];
    if (n.op == ir::Op::Param || n.op == ir::Op::Input) {
      name[i] = "%arg" + std::to_string(arg_idx);
      if (arg_idx++ > 0) o << ", ";
      o << name[i] << ": " << tensor_type(n);
      auto it = arg_shardings.find(n.name);
      if (it != arg_shardings.end() && !it->second.empty()) {
        o << " {mhlo.sharding = \"" << it->second << "\"}";
      }
    } else {
      // MLIR numeric SSA ids cannot have suffixes (e.g. %4_b is invalid).
      name[i] = "%v" + std::to_string(i);
    }
  }
  o << ") -> (";
  for (std::size_t k = 0; k < g.outputs.size(); ++k) {
    if (k > 0) o << ", ";
    o << tensor_type(g.nodes[static_cast<std::size_t>(g.outputs[k])]);
    if (k < result_shardings.size() && !result_shardings[k].empty()) {
      o << " {mhlo.sharding = \"" << result_shardings[k] << "\"}";
    }
  }
  o << ") {\n";

  auto types_in = [&](const ir::Node& n) {
    std::string s = "(";
    for (std::size_t k = 0; k < n.in.size(); ++k) {
      s += (k ? ", " : "") + tensor_type(g.nodes[static_cast<std::size_t>(n.in[k])]);
    }
    return s + ")";
  };
  auto args_of = [&](const ir::Node& n) {
    std::string s;
    for (std::size_t k = 0; k < n.in.size(); ++k) s += (k ? ", " : "") + name[static_cast<std::size_t>(n.in[k])];
    return s;
  };

  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const ir::Node& n = g.nodes[i];
    const std::string ty = tensor_type(n);
    using ir::Op;
    auto elem = [&](const char* op) { o << "    " << name[i] << " = stablehlo." << op << " " << args_of(n) << " : " << ty << "\n"; };
    switch (n.op) {
      case Op::Param:
      case Op::Input:
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
      case Op::MatMul:
      case Op::MatMulNT:
        o << "    " << name[i] << " = \"stablehlo.dot_general\"(" << args_of(n)
          << ") {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = ["
          << (n.op == Op::MatMul ? 0 : 1) << "]>} : " << types_in(n) << " -> " << ty << "\n";
        break;
      case Op::Scale:
        o << "    " << name[i] << "_c = stablehlo.constant dense<" << n.a << "> : " << ty << "\n";
        o << "    " << name[i] << " = stablehlo.multiply " << args_of(n) << ", " << name[i] << "_c : " << ty << "\n";
        break;
      case Op::BiasAdd:
        o << "    " << name[i] << "_b = stablehlo.broadcast_in_dim " << name[static_cast<std::size_t>(n.in[1])]
          << ", dims = [1] : (" << tensor_type(g.nodes[static_cast<std::size_t>(n.in[1])]) << ") -> " << ty << "\n";
        o << "    " << name[i] << " = stablehlo.add " << name[static_cast<std::size_t>(n.in[0])] << ", " << name[i] << "_b : " << ty << "\n";
        break;
      default:
        throw TensorError(std::string("Nexus StableHLO: unsupported export operation: ") + ir::to_string(n.op));
    }
  }
  o << "    return ";
  for (std::size_t k = 0; k < g.outputs.size(); ++k) o << (k ? ", " : "") << name[static_cast<std::size_t>(g.outputs[k])];
  o << " : ";
  for (std::size_t k = 0; k < g.outputs.size(); ++k) o << (k ? ", " : "") << tensor_type(g.nodes[static_cast<std::size_t>(g.outputs[k])]);
  o << "\n  }\n}\n";
  return o.str();
}

// ============================================================================
//  2) Yerlesik PJRT CPU Eklentisi (StableHLO MLIR Derleyici ve Yurutucu)
// ============================================================================

struct PJRT_Error {
  std::string message;
};

struct PJRT_Device {
  int id = 0;
  std::string debug_str = "PJRT_CPU_Device(id=0, host=avx2)";
};

struct PJRT_Client {
  std::vector<PJRT_Device> devices;
};

struct PJRT_Buffer {
  std::vector<std::int64_t> dims;
  std::vector<float> data;
  PJRT_Device* device = nullptr;
};

namespace {

enum class HloOp {
  Constant,
  Add,
  Sub,
  Mul,
  Div,
  Neg,
  Exp,
  Log,
  Tanh,
  Logistic,
  Sqrt,
  Rsqrt,
  DotGeneral,
  BroadcastInDim,
  ReduceSum,
  CustomRmsNorm,
  CustomLayerNorm,
  CustomSwiGlu,
  CustomSoftmax,
  CustomRope,
  CustomAttention,
  Identity,
};

struct HloInst {
  HloOp op = HloOp::Identity;
  std::string result_ssa;
  std::vector<std::string> args;
  std::vector<std::int64_t> out_shape;
  float const_val = 0.0f;
  int rhs_contract_dim = 0;
};

}  // namespace

struct PJRT_LoadedExecutable {
  std::vector<std::string> param_names;
  std::vector<std::vector<std::int64_t>> param_shapes;
  std::vector<std::string> param_shardings;
  std::vector<HloInst> instructions;
  std::vector<std::string> return_names;
};

namespace {

std::string trim(std::string_view sv) {
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) sv.remove_prefix(1);
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) sv.remove_suffix(1);
  return std::string(sv);
}

std::vector<std::int64_t> parse_tensor_shape(std::string_view type_str) {
  std::vector<std::int64_t> dims;
  const std::size_t pos = type_str.rfind("tensor<");
  if (pos == std::string_view::npos) return dims;
  std::string_view inner = type_str.substr(pos + 7);
  const std::size_t gt = inner.find('>');
  if (gt != std::string_view::npos) inner = inner.substr(0, gt);
  if (inner.size() < 3 || inner.substr(inner.size() - 3) != "f32")
    throw TensorError("Nexus StableHLO reference: only FP32 tensor types are supported");
  // ornek: 8x4xf32 veya f32
  std::size_t start = 0;
  while (start < inner.size()) {
    const std::size_t x = inner.find('x', start);
    if (x == std::string_view::npos) break;
    const std::string_view tok = inner.substr(start, x - start);
    if (tok == "?") {
      dims.push_back(-1);
    } else {
      dims.push_back(std::stoll(std::string(tok)));
    }
    start = x + 1;
  }
  return dims;
}

std::int64_t shape_numel(const std::vector<std::int64_t>& dims) {
  if (dims.empty()) return 1;
  std::int64_t p = 1;
  for (std::int64_t d : dims) {
    if (d < 0 || (d != 0 && p > std::numeric_limits<std::int64_t>::max() / d))
      throw TensorError("Nexus StableHLO reference: invalid or overflowing tensor shape");
    p *= d;
  }
  return p;
}

PJRT_Error* make_err(std::string msg) { return new PJRT_Error{std::move(msg)}; }

const char* builtin_platform_name() { return "nexus-stablehlo-reference-cpu"; }
const char* builtin_platform_version() { return "Nexus-reference-1 (not OpenXLA/PJRT)"; }

PJRT_Error* builtin_client_create(PJRT_Client** out_client) {
  if (out_client == nullptr) return make_err("out_client is null");
  auto* c = new PJRT_Client();
  c->devices.push_back(PJRT_Device{0, "Nexus CPU reference interpreter (not TPU/XLA)"});
  *out_client = c;
  return nullptr;
}

void builtin_client_destroy(PJRT_Client* client) { delete client; }
int builtin_client_device_count(PJRT_Client* client) {
  return client ? static_cast<int>(client->devices.size()) : 0;
}
PJRT_Device* builtin_client_addressable_device(PJRT_Client* client, int index) {
  if (!client || index < 0 || static_cast<std::size_t>(index) >= client->devices.size()) return nullptr;
  return &client->devices[static_cast<std::size_t>(index)];
}
const char* builtin_device_debug_string(PJRT_Device* device) {
  return device ? device->debug_str.c_str() : "invalid";
}

PJRT_Error* builtin_client_compile(PJRT_Client* /*client*/, const char* mlir_module, std::size_t mlir_size,
                                   PJRT_LoadedExecutable** out_exec) {
  if (!mlir_module || mlir_size == 0 || !out_exec) {
    return make_err("PJRT_Client_Compile: bos StableHLO modulu");
  }
  const std::string text(mlir_module, mlir_size);
  if (text.find("func.func @main") == std::string::npos) {
    return make_err("PJRT_Client_Compile: StableHLO icinde 'func.func @main' bulunamadi");
  }
  auto exec = std::make_unique<PJRT_LoadedExecutable>();

  // 1) @main argumanlarini cozumle
  const std::size_t fn_pos = text.find("func.func @main(");
  const std::size_t fn_eol = text.find('\n', fn_pos);
  const std::size_t close_args = text.find(") ->", fn_pos);
  if (close_args != std::string::npos && (fn_eol == std::string::npos || close_args < fn_eol)) {
    std::string args_hdr = text.substr(fn_pos + 16, close_args - (fn_pos + 16));
    std::size_t cursor = 0;
    while (cursor < args_hdr.size()) {
      const std::size_t pct = args_hdr.find('%', cursor);
      if (pct == std::string::npos) break;
      const std::size_t colon = args_hdr.find(':', pct);
      if (colon == std::string::npos) break;
      const std::string arg_name = trim(std::string_view(args_hdr).substr(pct, colon - pct));
      const std::size_t t_end = args_hdr.find('>', colon);
      if (t_end == std::string::npos) break;
      const std::string type_part = args_hdr.substr(colon + 1, t_end - colon);
      exec->param_names.push_back(arg_name);
      exec->param_shapes.push_back(parse_tensor_shape(type_part));
      cursor = t_end + 1;
    }
  }

  // 2) GOVDE satirlarini cozumle
  std::istringstream lines(fn_eol != std::string::npos ? text.substr(fn_eol + 1) : std::string{});
  std::string raw_line;
  while (std::getline(lines, raw_line)) {
    std::string line = trim(raw_line);
    if (line.empty() || line == "}" || line.rfind("//", 0) == 0) continue;
    if (line.rfind("return ", 0) == 0) {
      const std::size_t colon = line.find(':');
      std::string vars = trim(line.substr(7, colon == std::string::npos ? std::string::npos : colon - 7));
      std::stringstream vs(vars);
      std::string item;
      while (std::getline(vs, item, ',')) {
        exec->return_names.push_back(trim(item));
      }
      break;
    }
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    HloInst inst;
    inst.result_ssa = trim(std::string_view(line).substr(0, eq));
    const std::string rhs = trim(std::string_view(line).substr(eq + 1));
    inst.out_shape = parse_tensor_shape(rhs);

    auto extract_ssa_vars = [](std::string_view s) {
      std::vector<std::string> vars;
      for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%') {
          std::size_t j = i + 1;
          while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
          vars.emplace_back(s.substr(i, j - i));
          i = j;
        }
      }
      return vars;
    };

    const std::size_t colon = rhs.find(':');
    const std::string_view lhs_of_type = colon == std::string::npos ? std::string_view(rhs) : std::string_view(rhs).substr(0, colon);

    if (rhs.find("stablehlo.constant") != std::string::npos) {
      inst.op = HloOp::Constant;
      const std::size_t d = rhs.find("dense<");
      const std::size_t g = rhs.find('>', d);
      if (d != std::string::npos && g != std::string::npos) {
        inst.const_val = std::stof(rhs.substr(d + 6, g - (d + 6)));
      }
    } else if (rhs.find("stablehlo.dot_general") != std::string::npos) {
      inst.op = HloOp::DotGeneral;
      inst.args = extract_ssa_vars(lhs_of_type);
      inst.rhs_contract_dim = (rhs.find("rhs_contracting_dimensions = [1]") != std::string::npos) ? 1 : 0;
    } else if (rhs.find("stablehlo.broadcast_in_dim") != std::string::npos) {
      inst.op = HloOp::BroadcastInDim;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.reduce") != std::string::npos) {
      inst.op = HloOp::ReduceSum;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.add") != std::string::npos) {
      inst.op = HloOp::Add;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.subtract") != std::string::npos) {
      inst.op = HloOp::Sub;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.multiply") != std::string::npos) {
      inst.op = HloOp::Mul;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.divide") != std::string::npos) {
      inst.op = HloOp::Div;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.negate") != std::string::npos) {
      inst.op = HloOp::Neg;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.exponential") != std::string::npos) {
      inst.op = HloOp::Exp;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.log ") != std::string::npos) {
      inst.op = HloOp::Log;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.tanh") != std::string::npos) {
      inst.op = HloOp::Tanh;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.logistic") != std::string::npos) {
      inst.op = HloOp::Logistic;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.sqrt") != std::string::npos) {
      inst.op = HloOp::Sqrt;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("stablehlo.rsqrt") != std::string::npos) {
      inst.op = HloOp::Rsqrt;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("custom_call @nexus.RmsNorm") != std::string::npos) {
      inst.op = HloOp::CustomRmsNorm;
      inst.args = extract_ssa_vars(lhs_of_type);
      const std::size_t apos = rhs.find(";a=");
      inst.const_val = apos != std::string::npos ? std::stof(rhs.substr(apos + 3)) : 1e-5f;
    } else if (rhs.find("custom_call @nexus.LayerNorm") != std::string::npos) {
      inst.op = HloOp::CustomLayerNorm;
      inst.args = extract_ssa_vars(lhs_of_type);
      const std::size_t apos = rhs.find(";a=");
      inst.const_val = apos != std::string::npos ? std::stof(rhs.substr(apos + 3)) : 1e-5f;
    } else if (rhs.find("custom_call @nexus.SwiGlu") != std::string::npos) {
      inst.op = HloOp::CustomSwiGlu;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else if (rhs.find("custom_call @nexus.Softmax") != std::string::npos) {
      inst.op = HloOp::CustomSoftmax;
      inst.args = extract_ssa_vars(lhs_of_type);
    } else {
      return make_err("PJRT_Client_Compile: desteklenmeyen StableHLO talimati: " + line);
    }
    exec->instructions.push_back(std::move(inst));
  }

  *out_exec = exec.release();
  return nullptr;
}

void builtin_executable_destroy(PJRT_LoadedExecutable* exec) { delete exec; }

PJRT_Error* builtin_buffer_from_host(PJRT_Client* client, const float* host_data,
                                     const std::int64_t* dims, std::size_t num_dims,
                                     PJRT_Device* device, PJRT_Buffer** out_buffer) {
  if (!client || !out_buffer) return make_err("PJRT_Buffer_FromHostBuffer: gecersiz arguman");
  auto buf = std::make_unique<PJRT_Buffer>();
  buf->device = device;
  buf->dims.assign(dims, dims + num_dims);
  const std::int64_t n = shape_numel(buf->dims);
  buf->data.resize(static_cast<std::size_t>(n), 0.0f);
  if (host_data && n > 0) {
    std::memcpy(buf->data.data(), host_data, static_cast<std::size_t>(n) * sizeof(float));
  }
  *out_buffer = buf.release();
  return nullptr;
}

PJRT_Error* builtin_buffer_to_host(PJRT_Buffer* buffer, float* dst, std::size_t dst_bytes) {
  if (!buffer || !dst) return make_err("PJRT_Buffer_ToHostBuffer: gecersiz tampon");
  const std::size_t avail = buffer->data.size() * sizeof(float);
  if (dst_bytes > avail) return make_err("PJRT_Buffer_ToHostBuffer: hedef boyutu tampon boyutunu asiyor");
  std::memcpy(dst, buffer->data.data(), dst_bytes);
  return nullptr;
}

void builtin_buffer_destroy(PJRT_Buffer* buffer) { delete buffer; }

PJRT_Error* builtin_executable_execute(PJRT_LoadedExecutable* exec,
                                       PJRT_Buffer* const* argument_lists, std::size_t num_args,
                                       PJRT_Buffer** output_lists, std::size_t num_outputs) {
  if (!exec) return make_err("PJRT_LoadedExecutable_Execute: bos executable");
  if (num_args != exec->param_names.size()) {
    return make_err("PJRT_LoadedExecutable_Execute: arguman sayisi uyusmuyor");
  }
  std::unordered_map<std::string, PJRT_Buffer> env;
  for (std::size_t i = 0; i < num_args; ++i) {
    if (!argument_lists[i] || argument_lists[i]->dims != exec->param_shapes[i])
      return make_err("Nexus StableHLO reference: argument shape differs from module signature");
    env[exec->param_names[i]] = *argument_lists[i];
  }
  Driver& cpu = Registry::instance().require(Kind::Cpu);
  for (const HloInst& inst : exec->instructions) {
    PJRT_Buffer out;
    out.dims = inst.out_shape;
    const std::size_t n = static_cast<std::size_t>(shape_numel(out.dims));
    out.data.resize(n, 0.0f);
    switch (inst.op) {
      case HloOp::Constant:
        std::fill(out.data.begin(), out.data.end(), inst.const_val);
        break;
      case HloOp::Add:
      case HloOp::Sub:
      case HloOp::Mul:
      case HloOp::Div: {
        const auto& a = env.at(inst.args.at(0)).data;
        const auto& b = env.at(inst.args.at(1)).data;
        if (a.size() != n || b.size() != n)
          return make_err("StableHLO reference: binary operand/output size mismatch");
        const Operation op = inst.op == HloOp::Add   ? Operation::Add
                             : inst.op == HloOp::Sub ? Operation::Sub
                             : inst.op == HloOp::Mul ? Operation::Mul
                                                     : Operation::Div;
        cpu.elementwise(op, DeviceRef::cpu(), a.data(), b.data(), out.data.data(), n);
        break;
      }
      case HloOp::Neg:
      case HloOp::Exp:
      case HloOp::Log:
      case HloOp::Tanh:
      case HloOp::Logistic:
      case HloOp::Sqrt:
      case HloOp::Rsqrt: {
        const auto& a = env.at(inst.args.at(0)).data;
        if (a.size() != n) return make_err("StableHLO reference: unary operand/output size mismatch");
        const Operation op = inst.op == HloOp::Neg        ? Operation::Neg
                             : inst.op == HloOp::Exp      ? Operation::Exp
                             : inst.op == HloOp::Log      ? Operation::Log
                             : inst.op == HloOp::Tanh     ? Operation::Tanh
                             : inst.op == HloOp::Logistic ? Operation::Sigmoid
                             : inst.op == HloOp::Sqrt     ? Operation::Sqrt
                                                          : Operation::Rsqrt;
        cpu.elementwise(op, DeviceRef::cpu(), a.data(), nullptr, out.data.data(), n);
        break;
      }
      case HloOp::DotGeneral: {
        const auto& lhs = env.at(inst.args.at(0));
        const auto& rhs = env.at(inst.args.at(1));
        if (lhs.dims.size() != 2 || rhs.dims.size() != 2)
          return make_err("StableHLO reference: dot_general requires rank-2 inputs");
        for (const auto dim : {lhs.dims[0], lhs.dims[1], rhs.dims[0], rhs.dims[1]}) {
          if (dim <= 0 || dim > std::numeric_limits<int>::max())
            return make_err("StableHLO reference: dot_general dimension outside supported range");
        }
        const int m = static_cast<int>(lhs.dims.at(0));
        const int k = static_cast<int>(lhs.dims.at(1));
        const bool b_trans = (inst.rhs_contract_dim == 1);
        const int ncols = static_cast<int>(b_trans ? rhs.dims.at(0) : rhs.dims.at(1));
        if (lhs.dims[1] != rhs.dims[static_cast<std::size_t>(inst.rhs_contract_dim)] ||
            out.dims != std::vector<std::int64_t>{m, ncols})
          return make_err("StableHLO reference: dot_general contracting/output shape mismatch");
        out.dims = {m, ncols};
        out.data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(ncols), 0.0f);
        MatMulArgs mm;
        mm.m = m;
        mm.n = ncols;
        mm.k = k;
        mm.a = lhs.data.data();
        mm.lda = k;
        mm.a_trans = false;
        mm.b = rhs.data.data();
        mm.ldb = static_cast<int>(rhs.dims.at(1));
        mm.b_trans = b_trans;
        mm.c = out.data.data();
        mm.ldc = ncols;
        mm.alpha = 1.0f;
        mm.beta = 0.0f;
        cpu.matmul(DeviceRef::cpu(), mm);
        break;
      }
      case HloOp::BroadcastInDim: {
        const auto& src = env.at(inst.args.at(0));
        if (out.dims.size() != 2 || src.dims.size() != 1 || src.dims[0] != out.dims[1])
          return make_err("StableHLO reference: broadcast requires a matching row bias");
        const int rows = static_cast<int>(out.dims.at(0));
        const int cols = static_cast<int>(out.dims.at(1));
        for (int r = 0; r < rows; ++r) {
          std::memcpy(out.data.data() + static_cast<std::size_t>(r) * cols, src.data.data(),
                      static_cast<std::size_t>(cols) * sizeof(float));
        }
        break;
      }
      case HloOp::ReduceSum: {
        const auto& src = env.at(inst.args.at(0));
        float s = 0.0f;
        cpu.reduce_sum(DeviceRef::cpu(), src.data.data(), &s, src.data.size());
        out.data.assign(1, s);
        break;
      }
      case HloOp::CustomRmsNorm: {
        const auto& x = env.at(inst.args.at(0));
        const auto& w = env.at(inst.args.at(1));
        if (x.dims.size() != 2 || out.dims != x.dims || w.data.size() != static_cast<std::size_t>(x.dims[1]))
          return make_err("StableHLO reference: invalid RMSNorm shapes");
        const int rows = static_cast<int>(x.dims.at(0));
        const int cols = static_cast<int>(x.dims.at(1));
        cpu.rms_norm(DeviceRef::cpu(), x.data.data(), w.data.data(), out.data.data(), rows, cols, inst.const_val);
        break;
      }
      case HloOp::CustomLayerNorm: {
        const auto& x = env.at(inst.args.at(0));
        const auto& w = env.at(inst.args.at(1));
        if (x.dims.size() != 2 || out.dims != x.dims || w.data.size() != static_cast<std::size_t>(x.dims[1]) ||
            (inst.args.size() >= 3 && env.at(inst.args[2]).data.size() != w.data.size()))
          return make_err("StableHLO reference: invalid LayerNorm shapes");
        const float* b_ptr = inst.args.size() >= 3 ? env.at(inst.args.at(2)).data.data() : nullptr;
        const int rows = static_cast<int>(x.dims.at(0));
        const int cols = static_cast<int>(x.dims.at(1));
        cpu.layer_norm(DeviceRef::cpu(), x.data.data(), w.data.data(), b_ptr, out.data.data(), rows, cols, inst.const_val);
        break;
      }
      case HloOp::CustomSwiGlu: {
        const auto& x = env.at(inst.args.at(0));
        if (out.dims.size() != 2 || x.data.size() != 2 * n)
          return make_err("StableHLO reference: invalid SwiGLU shapes");
        const int rows = static_cast<int>(out.dims.at(0));
        const int hidden = static_cast<int>(out.dims.at(1));
        cpu.swiglu(DeviceRef::cpu(), x.data.data(), out.data.data(), rows, hidden);
        break;
      }
      case HloOp::CustomSoftmax: {
        const auto& x = env.at(inst.args.at(0));
        if (x.dims.size() != 2 || out.dims != x.dims || x.dims[1] <= 0)
          return make_err("StableHLO reference: invalid Softmax shapes");
        const int rows = static_cast<int>(x.dims.at(0));
        const int cols = static_cast<int>(x.dims.at(1));
        cpu.softmax(DeviceRef::cpu(), x.data.data(), out.data.data(), rows, cols);
        break;
      }
      default:
        return make_err("StableHLO reference: unsupported instruction");
    }
    env[inst.result_ssa] = std::move(out);
  }
  const std::size_t ret_n = std::min(num_outputs, exec->return_names.size());
  for (std::size_t i = 0; i < ret_n; ++i) {
    output_lists[i] = new PJRT_Buffer(env.at(exec->return_names[i]));
  }
  return nullptr;
}

const char* builtin_error_message(const PJRT_Error* error) {
  return error ? error->message.c_str() : "";
}
void builtin_error_destroy(PJRT_Error* error) { delete error; }

}  // namespace

const PJRT_Api* GetBuiltinCpuPjrtApi() noexcept {
  static const PJRT_Api kApi = {
      sizeof(PJRT_Api),
      {sizeof(PJRT_Api_Version), 0, 54},
      &builtin_platform_name,
      &builtin_platform_version,
      &builtin_client_create,
      &builtin_client_destroy,
      &builtin_client_device_count,
      &builtin_client_addressable_device,
      &builtin_device_debug_string,
      &builtin_client_compile,
      &builtin_executable_destroy,
      &builtin_buffer_from_host,
      &builtin_buffer_to_host,
      &builtin_buffer_destroy,
      &builtin_executable_execute,
      &builtin_error_message,
      &builtin_error_destroy,
  };
  return &kApi;
}

// ============================================================================
//  3) pjrt::Client
// ============================================================================

Client::Client(std::string_view plugin_path) {
  std::string path(plugin_path);
  if (path.empty()) {
    if (const char* env = std::getenv("NEXUS_PJRT_PLUGIN_PATH"); env != nullptr && *env != '\0') {
      path = env;
    }
  }
  if (!path.empty()) {
    external_ = std::make_unique<detail::PluginClient>(path);
    return;
  }
  api_ = GetBuiltinCpuPjrtApi();
  if (PJRT_Error* err = api_->Client_Create(&client_); err != nullptr) {
    std::string msg = api_->Error_Message(err);
    api_->Error_Destroy(err);
    throw TensorError("Nexus PJRT: Client_Create basarisiz: " + msg);
  }
}

Client::~Client() {
  if (api_ && client_) {
    api_->Client_Destroy(client_);
    client_ = nullptr;
  }
}

std::string_view Client::platform_name() const noexcept { return external_ ? external_->platform() : (api_ ? api_->PlatformName() : "unknown"); }
std::string_view Client::platform_version() const noexcept { return external_ ? external_->version() : (api_ ? api_->PlatformVersion() : "0"); }
int Client::device_count() const noexcept { return external_ ? external_->device_count() : ((api_ && client_) ? api_->Client_DeviceCount(client_) : 0); }
std::string Client::device_name(int index) const {
  if (external_) return external_->device_name(index);
  if (!api_ || !client_) return {};
  PJRT_Device* dev = api_->Client_AddressableDevice(client_, index);
  return dev ? api_->Device_DebugString(dev) : std::string{};
}

std::vector<std::vector<float>> Client::compile_and_execute(
    std::string_view stablehlo_mlir,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<std::int64_t>>& input_shapes) {
  if (inputs.size() != input_shapes.size()) {
    throw TensorError("Nexus PJRT: inputs ve input_shapes boyutu esit olmali");
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto n = shape_numel(input_shapes[i]);
    if (static_cast<std::uint64_t>(n) != inputs[i].size())
      throw TensorError("Nexus StableHLO reference: input shape does not match buffer size");
  }
  if (external_) return external_->run(stablehlo_mlir, inputs, input_shapes);
  PJRT_LoadedExecutable* exec = nullptr;
  if (PJRT_Error* err = api_->Client_Compile(client_, stablehlo_mlir.data(), stablehlo_mlir.size(), &exec);
      err != nullptr) {
    std::string msg = api_->Error_Message(err);
    api_->Error_Destroy(err);
    throw TensorError("Nexus PJRT Derleme Hatasi: " + msg);
  }
  PJRT_Device* dev = api_->Client_AddressableDevice(client_, 0);
  std::vector<PJRT_Buffer*> in_bufs(inputs.size(), nullptr);
  std::vector<PJRT_Buffer*> out_bufs;
  auto cleanup = [&] {
    for (auto* b : out_bufs) if (b) api_->Buffer_Destroy(b);
    for (auto* b : in_bufs) if (b) api_->Buffer_Destroy(b);
    if (exec) api_->LoadedExecutable_Destroy(exec);
  };
  try {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (PJRT_Error* err = api_->Buffer_FromHostBuffer(client_, inputs[i].data(), input_shapes[i].data(),
                                                        input_shapes[i].size(), dev, &in_bufs[i]);
          err != nullptr) {
        std::string msg = api_->Error_Message(err);
        api_->Error_Destroy(err);
        throw TensorError("Nexus PJRT Buffer_FromHostBuffer Hatasi: " + msg);
      }
    }
    const std::size_t num_out = std::max<std::size_t>(1, exec->return_names.size());
    out_bufs.resize(num_out, nullptr);
    if (PJRT_Error* err = api_->LoadedExecutable_Execute(exec, in_bufs.data(), in_bufs.size(),
                                                         out_bufs.data(), out_bufs.size());
        err != nullptr) {
      std::string msg = api_->Error_Message(err);
      api_->Error_Destroy(err);
      throw TensorError("Nexus PJRT Execute Hatasi: " + msg);
    }
    std::vector<std::vector<float>> results(num_out);
    for (std::size_t i = 0; i < num_out; ++i) {
      if (out_bufs[i] != nullptr) {
        results[i].resize(out_bufs[i]->data.size());
        if (PJRT_Error* err = api_->Buffer_ToHostBuffer(out_bufs[i], results[i].data(),
                                                        results[i].size() * sizeof(float));
            err != nullptr) {
          std::string msg = api_->Error_Message(err);
          api_->Error_Destroy(err);
          throw TensorError("Nexus PJRT Buffer_ToHostBuffer Hatasi: " + msg);
        }
      }
    }
    cleanup();
    return results;
  } catch (...) {
    cleanup();
    throw;
  }
}

}  // namespace nexus::backend::pjrt

// ============================================================================
//  4) PjrtDriver (Backend fabrikasina kayitli surucu)
// ============================================================================

namespace nexus::backend {
namespace {

struct PjrtAllocation {
  std::vector<float> storage;
  std::size_t bytes = 0;
};

class PjrtDriver final : public Driver {
 public:
  PjrtDriver() : client_("") {
    // This host-staging facade executes on the client's first addressable device.
    const int n = std::min(1, client_.device_count());
    for (int i = 0; i < n; ++i) {
      DeviceInfo info;
      info.device = DeviceRef::tpu(i);
      info.name = client_.device_name(i);
      info.vendor = client_.uses_external_plugin() ? "OpenXLA/PJRT plugin" : "Nexus CPU reference interpreter";
      info.api_version = std::string(client_.platform_version());
      info.driver_version = std::string(client_.platform_name());
      info.memory_bytes = 0;  // Unknown host capacity; never invent TPU memory.
      info.subgroup_size = 8;
      info.unified_memory = false;  // Allocations are opaque handles, not host pointers.
      devices_.push_back(std::move(info));
    }
  }
  ~PjrtDriver() override {
    for (PjrtAllocation* b : all_) delete b;
  }

  Kind kind() const noexcept override { return Kind::Pjrt; }
  std::string_view name() const noexcept override { return client_.uses_external_plugin() ? "pjrt-plugin" : "stablehlo-reference-cpu"; }
  std::string_view version() const noexcept override { return client_.platform_version(); }
  const std::vector<DeviceInfo>& devices() const noexcept override { return devices_; }
  std::string unavailable_reason() const override {
    return devices_.empty() ? "PJRT istemcisi olusturulamadi" : std::string{};
  }
  bool supports(Operation op) const noexcept override {
    return !devices_.empty() && (client_.uses_external_plugin() ? op == Operation::MatMul : op <= Operation::ReduceSum);
  }

  void* allocate(int device, std::size_t bytes) override {
    if (device != 0) throw TensorError("Nexus PJRT facade: only addressable device 0 is exposed");
    if (bytes == 0) return nullptr;
    if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(float))
      throw TensorError("Nexus StableHLO reference: allocation size overflow");
    std::lock_guard lock(mutex_);
    auto owned = std::make_unique<PjrtAllocation>();
    auto* blk = owned.get();
    blk->bytes = bytes;
    blk->storage.resize((bytes + sizeof(float) - 1) / sizeof(float), 0.0f);
    all_.insert(blk);
    ++stats_.allocations;
    ++stats_.live_blocks;
    stats_.current_bytes += bytes;
    stats_.reserved_bytes += bytes;
    return owned.release();
  }

  void deallocate(int /*device*/, void* pointer) noexcept override {
    if (!pointer) return;
    std::lock_guard lock(mutex_);
    auto* blk = static_cast<PjrtAllocation*>(pointer);
    auto it = all_.find(blk);
    if (it == all_.end()) return;
    const std::size_t b = blk->bytes;
    stats_.current_bytes = stats_.current_bytes >= b ? stats_.current_bytes - b : 0;
    stats_.reserved_bytes = stats_.reserved_bytes >= b ? stats_.reserved_bytes - b : 0;
    if (stats_.live_blocks > 0) --stats_.live_blocks;
    all_.erase(it);
    delete blk;
  }

  void copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src, std::size_t bytes) override {
    if (bytes == 0) return;
    if ((!dst_device.is_host() && dst_device != DeviceRef::tpu(0)) ||
        (!src_device.is_host() && src_device != DeviceRef::tpu(0)))
      throw TensorError("Nexus PJRT: copy requires host or the first PJRT device");
    if (!dst || !src) throw TensorError("Nexus StableHLO reference: null copy buffer");
    std::lock_guard lock(mutex_);
    const bool dst_pjrt = (dst_device.kind == DeviceKind::Tpu);
    const bool src_pjrt = (src_device.kind == DeviceKind::Tpu);
    auto validate = [&](const void* pointer) {
      auto* block = const_cast<PjrtAllocation*>(static_cast<const PjrtAllocation*>(pointer));
      if (all_.find(block) == all_.end() || bytes > block->bytes)
        throw TensorError("Nexus StableHLO reference: copy exceeds allocation or uses an invalid handle");
    };
    if (dst_pjrt) validate(dst);
    if (src_pjrt) validate(src);
    const void* src_ptr = src_pjrt ? static_cast<const void*>(static_cast<const PjrtAllocation*>(src)->storage.data()) : src;
    void* dst_ptr = dst_pjrt ? static_cast<void*>(static_cast<PjrtAllocation*>(dst)->storage.data()) : dst;
    std::memcpy(dst_ptr, src_ptr, bytes);
  }

  void synchronize(int device) override {
    if (device != 0) throw TensorError("Nexus PJRT facade: invalid device index");
  }

  void elementwise(Operation op, DeviceRef device, const float* a, const float* b, float* out,
                   std::size_t count, float scalar) override {
    if (device != DeviceRef::tpu(0)) throw TensorError("Nexus PJRT facade: invalid device index");
    if (count == 0) return;
    if (client_.uses_external_plugin()) require_cpu_fallback(*this, op, count);
    const auto* ab = static_cast<const PjrtAllocation*>(static_cast<const void*>(a));
    const auto* bb = b ? static_cast<const PjrtAllocation*>(static_cast<const void*>(b)) : nullptr;
    auto* ob = static_cast<PjrtAllocation*>(static_cast<void*>(out));
    Registry::instance().require(Kind::Cpu).elementwise(
        op, DeviceRef::cpu(), ab->storage.data(), bb ? bb->storage.data() : nullptr,
        ob->storage.data(), count, scalar);
  }

  void matmul(DeviceRef device, const MatMulArgs& args) override {
    if (device != DeviceRef::tpu(0)) throw TensorError("Nexus PJRT facade: invalid device index");
    const auto* ab = static_cast<const PjrtAllocation*>(static_cast<const void*>(args.a));
    const auto* bb = static_cast<const PjrtAllocation*>(static_cast<const void*>(args.b));
    auto* cb = static_cast<PjrtAllocation*>(static_cast<void*>(args.c));
    if (!args.a_trans && !args.b_trans && args.lda == args.k && args.ldb == args.n &&
        args.ldc == args.n && args.alpha == 1.0f && args.beta == 0.0f) {
      std::ostringstream mlir;
      mlir << "module @pjrt_gemm {\n"
           << "  func.func @main(%arg0: tensor<" << args.m << "x" << args.k << "xf32>, "
           << "%arg1: tensor<" << args.k << "x" << args.n << "xf32>) -> (tensor<" << args.m << "x" << args.n << "xf32>) {\n"
           << "    %0 = \"stablehlo.dot_general\"(%arg0, %arg1) {dot_dimension_numbers = "
           << "#stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : "
           << "(tensor<" << args.m << "x" << args.k << "xf32>, tensor<" << args.k << "x" << args.n << "xf32>) -> "
           << "tensor<" << args.m << "x" << args.n << "xf32>\n"
           << "    return %0 : tensor<" << args.m << "x" << args.n << "xf32>\n"
           << "  }\n}\n";
      std::vector<float> a_vec(ab->storage.begin(), ab->storage.begin() + static_cast<std::ptrdiff_t>(args.m) * args.k);
      std::vector<float> b_vec(bb->storage.begin(), bb->storage.begin() + static_cast<std::ptrdiff_t>(args.k) * args.n);
      auto res = client_.compile_and_execute(mlir.str(), {std::move(a_vec), std::move(b_vec)},
                                             {{args.m, args.k}, {args.k, args.n}});
      std::memcpy(cb->storage.data(), res[0].data(), res[0].size() * sizeof(float));
      return;
    }
    MatMulArgs host_args = args;
    if (client_.uses_external_plugin())
      require_cpu_fallback(*this, Operation::MatMul, static_cast<std::size_t>(args.m) * args.n);
    host_args.a = ab->storage.data();
    host_args.b = bb->storage.data();
    host_args.c = cb->storage.data();
    Registry::instance().require(Kind::Cpu).matmul(DeviceRef::cpu(), host_args);
  }

  void reduce_sum(DeviceRef device, const float* input, float* output, std::size_t count) override {
    if (device != DeviceRef::tpu(0)) throw TensorError("Nexus PJRT facade: invalid device index");
    if (client_.uses_external_plugin()) require_cpu_fallback(*this, Operation::ReduceSum, count);
    const auto* ib = static_cast<const PjrtAllocation*>(static_cast<const void*>(input));
    auto* ob = static_cast<PjrtAllocation*>(static_cast<void*>(output));
    Registry::instance().require(Kind::Cpu).reduce_sum(DeviceRef::cpu(), ib->storage.data(), ob->storage.data(), count);
  }

  AllocatorStats allocator_stats(int /*device*/) const override {
    std::lock_guard lock(mutex_);
    return stats_;
  }

 private:
  pjrt::Client client_;
  std::vector<DeviceInfo> devices_;
  mutable std::mutex mutex_;
  std::unordered_set<PjrtAllocation*> all_;
  AllocatorStats stats_{};
};

}  // namespace

std::unique_ptr<Driver> make_pjrt_driver() { return std::make_unique<PjrtDriver>(); }

}  // namespace nexus::backend
