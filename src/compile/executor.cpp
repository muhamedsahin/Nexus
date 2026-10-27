/**
 * @file executor.cpp
 * @brief FAZ 13 & FAZ 18.6: IR grafik yurutucusu (CPU) — statik arena (plan_memory), AVX2 + paralel
 *        op cekirdekleri, cok is parcacikli JIT fuzyon cagrilari, kopyasiz AttentionQKV, asenkron iletisim.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/attention/cpu.hpp"
#include "nexus/compile/compile.hpp"
#include "nexus/cpu/gemm.hpp"
#include "nexus/cpu/threads.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus/fused/training.hpp"
#include "nexus/lm/rope.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nexus {
namespace compile {

namespace K = nexus_model::kernels;
using ir::Bindings;
using ir::Node;
using ir::Op;

namespace {

[[noreturn]] void bad(const std::string& w) { throw std::runtime_error("Nexus yurutucu: " + w); }

constexpr float kInvSqrt2 = 0.70710678118654752f;
constexpr float kInvSqrt2Pi = 0.39894228040143268f;

#if defined(__AVX2__)
inline float hsum256_ps(__m256 v) noexcept {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

inline __m256 fast_exp256_ps(__m256 x) noexcept {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.33654f)), _mm256_set1_ps(88.37626f));
  __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
  const __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}
#endif

float gelu_tanh_bwd(float dy, float x) {
  const float k = 0.7978845608028654f, u = k * (x + 0.044715f * x * x * x), t = std::tanh(u);
  return dy * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * k * (1.0f + 3.0f * 0.044715f * x * x));
}

float scalar_op(Op op, float x, float y, double a) {
  switch (op) {
    case Op::Add: case Op::BiasAdd: return x + y;
    case Op::Sub: return x - y;
    case Op::Mul: return x * y;
    case Op::Div: return x / y;
    case Op::Neg: return -x;
    case Op::Exp: return std::exp(x);
    case Op::Log: return std::log(x);
    case Op::Tanh: return std::tanh(x);
    case Op::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
    case Op::Gelu: return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
    case Op::GeluTanh: return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    case Op::Silu: return x / (1.0f + std::exp(-x));
    case Op::Relu: return x > 0.0f ? x : 0.0f;
    case Op::Square: return x * x;
    case Op::Sqrt: return std::sqrt(x);
    case Op::Rsqrt: return 1.0f / std::sqrt(x);
    case Op::Scale: return x * static_cast<float>(a);
    case Op::AddScalar: return x + static_cast<float>(a);
    case Op::GeluBwd: return x * (0.5f * (1.0f + std::erf(y * kInvSqrt2)) + y * kInvSqrt2Pi * std::exp(-0.5f * y * y));
    case Op::GeluTanhBwd: return gelu_tanh_bwd(x, y);
    case Op::SiluBwd: {
      const float s = 1.0f / (1.0f + std::exp(-y));
      return x * s * (1.0f + y * (1.0f - s));
    }
    case Op::SigmoidBwd: return x * y * (1.0f - y);
    case Op::TanhBwd: return x * (1.0f - y * y);
    default:
      bad(std::string("eleman eleman degil: ") + ir::to_string(op));
  }
}

void norm_fwd_rows(bool rms, const float* x, const float* w, const float* b, float* y, std::int64_t r0, std::int64_t r1,
                   std::int64_t c, double eps) {
  for (std::int64_t r = r0; r < r1; ++r) {
    const float* xr = x + r * c;
    float* yr = y + r * c;
    double mu = 0.0;
    if (!rms) {
      for (std::int64_t i = 0; i < c; ++i) mu += xr[i];
      mu /= static_cast<double>(c);
    }
    double var = 0.0;
    for (std::int64_t i = 0; i < c; ++i) {
      const double d = xr[i] - mu;
      var += d * d;
    }
    var /= static_cast<double>(c);
    const float rs = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mu);
    std::int64_t i = 0;
#if defined(__AVX2__)
    const __m256 vm = _mm256_set1_ps(m);
    const __m256 vrs = _mm256_set1_ps(rs);
    for (; i + 8 <= c; i += 8) {
      const __m256 xh = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + i), vm), vrs);
      const __m256 vw = _mm256_loadu_ps(w + i);
      const __m256 vo = b ? _mm256_fmadd_ps(xh, vw, _mm256_loadu_ps(b + i)) : _mm256_mul_ps(xh, vw);
      _mm256_storeu_ps(yr + i, vo);
    }
#endif
    for (; i < c; ++i) {
      const float xh = (xr[i] - m) * rs;
      yr[i] = b ? xh * w[i] + b[i] : xh * w[i];
    }
  }
}

void norm_fwd(bool rms, const float* x, const float* w, const float* b, float* y, std::int64_t rows, std::int64_t c,
              double eps) {
  if (rows * c >= 16384 && rows >= 4) {
    cpu::Pool::instance().run(static_cast<std::size_t>(rows), [&](int, std::size_t r0, std::size_t r1) {
      norm_fwd_rows(rms, x, w, b, y, static_cast<std::int64_t>(r0), static_cast<std::int64_t>(r1), c, eps);
    });
  } else {
    norm_fwd_rows(rms, x, w, b, y, 0, rows, c, eps);
  }
}

/// dx (yazilir) ve/veya dw, db (yazilir).
void norm_bwd(bool rms, const float* dy, const float* x, const float* w, float* dx, float* dw, float* db,
              std::int64_t rows, std::int64_t c, double eps) {
  if (dw) std::memset(dw, 0, static_cast<std::size_t>(c) * sizeof(float));
  if (db) std::memset(db, 0, static_cast<std::size_t>(c) * sizeof(float));
  thread_local std::vector<float> dxh;
  if (dx && dxh.size() < static_cast<std::size_t>(c)) dxh.resize(static_cast<std::size_t>(c));
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * c;
    const float* dyr = dy + r * c;
    double mu = 0.0;
    if (!rms) {
      for (std::int64_t i = 0; i < c; ++i) mu += xr[i];
      mu /= static_cast<double>(c);
    }
    double var = 0.0;
    for (std::int64_t i = 0; i < c; ++i) {
      const double d = xr[i] - mu;
      var += d * d;
    }
    var /= static_cast<double>(c);
    const float rs = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mu);
    double s1 = 0.0, s2 = 0.0;
    for (std::int64_t i = 0; i < c; ++i) {
      const float xh = (xr[i] - m) * rs;
      if (dw) dw[i] += dyr[i] * xh;
      if (db) db[i] += dyr[i];
      if (dx) {
        const float v = dyr[i] * w[i];
        dxh[static_cast<std::size_t>(i)] = v;
        s1 += v;
        s2 += static_cast<double>(v) * xh;
      }
    }
    if (!dx) continue;
    const float a1 = rms ? 0.0f : static_cast<float>(s1 / static_cast<double>(c));
    const float a2 = static_cast<float>(s2 / static_cast<double>(c));
    float* dxr = dx + r * c;
    std::int64_t i = 0;
#if defined(__AVX2__)
    const __m256 vm = _mm256_set1_ps(m);
    const __m256 vrs = _mm256_set1_ps(rs);
    const __m256 va1 = _mm256_set1_ps(a1);
    const __m256 va2 = _mm256_set1_ps(a2);
    for (; i + 8 <= c; i += 8) {
      const __m256 xh = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + i), vm), vrs);
      const __m256 d = _mm256_fnmadd_ps(xh, va2, _mm256_sub_ps(_mm256_loadu_ps(dxh.data() + i), va1));
      _mm256_storeu_ps(dxr + i, _mm256_mul_ps(vrs, d));
    }
#endif
    for (; i < c; ++i) {
      const float xh = (xr[i] - m) * rs;
      dxr[i] = rs * (dxh[static_cast<std::size_t>(i)] - a1 - xh * a2);
    }
  }
}

void add_arrays_cpu(float* dst, const float* a, const float* b, std::int64_t cnt) noexcept {
  std::int64_t e = 0;
#if defined(__AVX2__)
  for (; e + 16 <= cnt; e += 16) {
    _mm256_storeu_ps(dst + e, _mm256_add_ps(_mm256_loadu_ps(a + e), _mm256_loadu_ps(b + e)));
    _mm256_storeu_ps(dst + e + 8, _mm256_add_ps(_mm256_loadu_ps(a + e + 8), _mm256_loadu_ps(b + e + 8)));
  }
  for (; e + 8 <= cnt; e += 8) {
    _mm256_storeu_ps(dst + e, _mm256_add_ps(_mm256_loadu_ps(a + e), _mm256_loadu_ps(b + e)));
  }
#endif
  for (; e < cnt; ++e) dst[e] = a[e] + b[e];
}

void sum_rows_cpu(const float* x, float* out, std::int64_t rows, std::int64_t cols) noexcept {
  std::memset(out, 0, static_cast<std::size_t>(cols) * sizeof(float));
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    std::int64_t k = 0;
#if defined(__AVX2__)
    for (; k + 16 <= cols; k += 16) {
      _mm256_storeu_ps(out + k, _mm256_add_ps(_mm256_loadu_ps(out + k), _mm256_loadu_ps(xr + k)));
      _mm256_storeu_ps(out + k + 8, _mm256_add_ps(_mm256_loadu_ps(out + k + 8), _mm256_loadu_ps(xr + k + 8)));
    }
    for (; k + 8 <= cols; k += 8) {
      _mm256_storeu_ps(out + k, _mm256_add_ps(_mm256_loadu_ps(out + k), _mm256_loadu_ps(xr + k)));
    }
#endif
    for (; k < cols; ++k) out[k] += xr[k];
  }
}

AttentionParams attn_params(std::int64_t B, std::int64_t T, std::int64_t H, std::int64_t Hk, std::int64_t D, double scale) {
  AttentionParams p;
  p.batch = static_cast<int>(B);
  p.heads = static_cast<int>(H);
  p.kv_heads = static_cast<int>(Hk);
  p.tq = static_cast<int>(T);
  p.tk = static_cast<int>(T);
  p.head_dim = static_cast<int>(D);
  p.dtype = DType::F32;
  p.scale = static_cast<float>(scale);
  p.causal = true;
  p.set_contiguous_bthd();
  return p;
}

void dispatch_fused_cpu(ir::CpuKernelFn fn, const ir::FusedOp& f, const std::vector<const float*>& ins,
                        const std::vector<float*>& outs, long long rows, long long cols) {
  const long long total = rows * cols;
  if (total < 16384 || cpu::Pool::instance().size() <= 1) {
    fn(ins.data(), outs.data(), rows, cols);
    return;
  }
  const std::size_t nin = ins.size();
  const std::size_t nout = outs.size();
  if (rows >= static_cast<long long>(cpu::Pool::instance().size())) {
    cpu::Pool::instance().run(static_cast<std::size_t>(rows), [&](int, std::size_t r0, std::size_t r1) {
      const float* l_ins[16];
      float* l_outs[8];
      std::vector<const float*> v_ins;
      std::vector<float*> v_outs;
      const float** p_in = nin <= 16 ? l_ins : (v_ins.resize(nin), v_ins.data());
      float** p_out = nout <= 8 ? l_outs : (v_outs.resize(nout), v_outs.data());
      const long long off = static_cast<long long>(r0) * cols;
      for (std::size_t e = 0; e < nin; ++e) {
        p_in[e] = f.input_is_row[e] ? ins[e] : (ins[e] + off);
      }
      for (std::size_t k = 0; k < nout; ++k) {
        p_out[k] = outs[k] + off;
      }
      fn(p_in, p_out, static_cast<long long>(r1 - r0), cols);
    });
    return;
  }
  if (rows == 1) {
    // Tek satirlik (yayinsiz) duz vektor: 64-elemanlik bloklar halinde paralel parcala.
    const long long blocks = (cols + 63) / 64;
    cpu::Pool::instance().run(static_cast<std::size_t>(blocks), [&](int, std::size_t b0, std::size_t b1) {
      const long long c0 = static_cast<long long>(b0) * 64;
      const long long c1 = std::min(cols, static_cast<long long>(b1) * 64);
      if (c0 >= c1) return;
      const float* l_ins[16];
      float* l_outs[8];
      std::vector<const float*> v_ins;
      std::vector<float*> v_outs;
      const float** p_in = nin <= 16 ? l_ins : (v_ins.resize(nin), v_ins.data());
      float** p_out = nout <= 8 ? l_outs : (v_outs.resize(nout), v_outs.data());
      for (std::size_t e = 0; e < nin; ++e) p_in[e] = ins[e] + c0;
      for (std::size_t k = 0; k < nout; ++k) p_out[k] = outs[k] + c0;
      fn(p_in, p_out, 1, c1 - c0);
    });
    return;
  }
  fn(ins.data(), outs.data(), rows, cols);
}

}  // namespace

struct Executable::Impl {
  const ir::Graph* g = nullptr;
  CompileOptions opt;
  std::map<std::string, ir::CpuKernelFn> kernels;  // ad -> fonksiyon (vektor ve "_s")
  std::map<std::string, bool> scalar_choice;       // autotune: kume anahtari -> skaler daha hizli
  std::vector<std::vector<int>> aux;               // ust -> indeks -> Output dugumu
  std::map<std::string, ir::MemoryPlan> plans;
  std::vector<float> arena;
  std::shared_ptr<dist::ProcessGroup> group;
};

Executable::Executable(ir::Graph graph, ir::CpuJit& jit, const CompileOptions& options)
    : graph_(std::move(graph)), impl_(std::make_unique<Impl>()) {
  impl_->g = &graph_;
  impl_->opt = options;
  impl_->aux.resize(graph_.nodes.size());
  for (std::size_t i = 0; i < graph_.nodes.size(); ++i) {
    const Node& n = graph_.nodes[i];
    if (n.op == Op::Output) {
      auto& v = impl_->aux[static_cast<std::size_t>(n.in[0])];
      if (static_cast<int>(v.size()) <= n.index) v.resize(static_cast<std::size_t>(n.index) + 1, -1);
      if (v[static_cast<std::size_t>(n.index)] >= 0) bad("ayni ek cikti (Output) iki dugumle temsil ediliyor (gecis hatasi)");
      v[static_cast<std::size_t>(n.index)] = static_cast<int>(i);
    }
  }
  std::vector<const ir::FusedOp*> uniq;
  std::set<std::string> seen;
  std::vector<std::string> names;
  for (const Node& n : graph_.nodes) {
    if (n.op != Op::Fused) continue;
    const std::string nm = ir::kernel_name(*n.fused);
    if (seen.insert(nm).second) {
      uniq.push_back(n.fused.get());
      names.push_back(nm);
      names.push_back(nm + "_s");
    }
  }
  if (!uniq.empty()) impl_->kernels = jit.load(ir::cpu_source(uniq), names);
}

Executable::~Executable() = default;

void Executable::set_group(std::shared_ptr<dist::ProcessGroup> group) { impl_->group = std::move(group); }

std::vector<std::vector<float>> Executable::run(const Bindings& b, const std::map<std::string, const float*>& params,
                                                const std::map<std::string, const void*>& inputs) {
  Impl& I = *impl_;
  const ir::Graph& g = graph_;
  std::string key;
  for (const auto& kv : b) key += std::to_string(kv.first) + "=" + std::to_string(kv.second) + ";";
  auto pit = I.plans.find(key);
  if (pit == I.plans.end()) pit = I.plans.emplace(key, ir::plan_memory(g, b)).first;
  const ir::MemoryPlan& plan = pit->second;
  last_arena_ = plan.arena_bytes;
  last_naive_ = plan.naive_bytes;
  if (I.arena.size() * 4 < static_cast<std::size_t>(plan.arena_bytes)) I.arena.assign(static_cast<std::size_t>(plan.arena_bytes / 4 + 16), 0.0f);
  std::vector<void*> ptr(g.nodes.size(), nullptr);
  auto P = [&](int v) -> float* { return static_cast<float*>(ptr[static_cast<std::size_t>(v)]); };
  auto I32 = [&](int v) -> const std::int32_t* { return static_cast<const std::int32_t*>(ptr[static_cast<std::size_t>(v)]); };
  auto E = [&](const ir::Dim& d) { return ir::eval(d, b); };
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const Node& n = g.nodes[i];
    if (n.op == Op::Param) {
      auto it = params.find(n.name);
      if (it == params.end()) bad("parametre verilmedi: " + n.name);
      ptr[i] = const_cast<float*>(it->second);
    } else if (n.op == Op::Input) {
      auto it = inputs.find(n.name);
      if (it == inputs.end()) bad("girdi verilmedi: " + n.name);
      ptr[i] = const_cast<void*>(it->second);
    } else {
      ptr[i] = reinterpret_cast<char*>(I.arena.data()) + plan.offset[i];
    }
  }
  std::map<int, dist::WorkPtr> pending;
  std::vector<std::vector<float>> scratch;  // kullanilmayan ek ciktilar
  auto aux_ptr = [&](std::size_t i, int k, std::size_t elems) -> float* {
    const auto& v = I.aux[i];
    if (static_cast<int>(v.size()) > k && v[static_cast<std::size_t>(k)] >= 0) return P(v[static_cast<std::size_t>(k)]);
    scratch.emplace_back(elems);
    return scratch.back().data();
  };
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const Node& n = g.nodes[i];
    for (int x : n.in) {
      auto it = pending.find(x);
      if (it != pending.end()) {
        it->second->wait();
        pending.erase(it);
      }
    }
    float* out = P(static_cast<int>(i));
    const std::int64_t cnt = ir::numel(n.shape, b);
    auto in = [&](int k) { return P(n.in[static_cast<std::size_t>(k)]); };
    auto shp = [&](int k) -> const ir::Shape& { return g.nodes[static_cast<std::size_t>(n.in[static_cast<std::size_t>(k)])].shape; };
    switch (n.op) {
      case Op::Param: case Op::Input: case Op::Output:
        break;
      case Op::Const:
        std::fill(out, out + cnt, static_cast<float>(n.a));
        break;
      case Op::Positions: {
        auto* o = reinterpret_cast<std::int32_t*>(out);
        const std::int64_t T = b.at(static_cast<int>(n.ints[1]));
        for (std::int64_t r = 0; r < cnt; ++r) o[r] = static_cast<std::int32_t>(r % T);
        break;
      }
      case Op::BiasAdd: {
        const std::int64_t c = E(n.shape[1]);
        const std::int64_t rows = cnt / c;
        const float* x = in(0);
        const float* bb = in(1);
        for (std::int64_t r = 0; r < rows; ++r) {
          add_arrays_cpu(out + r * c, x + r * c, bb, c);
        }
        break;
      }
      case Op::SumRows: {
        const std::int64_t c = E(n.shape[0]);
        const std::int64_t rows = ir::numel(shp(0), b) / c;
        sum_rows_cpu(in(0), out, rows, c);
        break;
      }
      case Op::MatMul: case Op::MatMulNT: case Op::MatMulTN: {
        // Epilog bayraklari: 1 biriktirme (son girdi), 2 BGRAD (MatMulTN, Output 1), 4 DGELU (MatMulNT, girdi 2).
        const std::int64_t fl = n.ints.empty() ? 0 : n.ints[0];
        const float* accv = (fl & 1) ? in(static_cast<int>(n.in.size()) - 1) : nullptr;
        if (n.op == Op::MatMul) {
          const std::int64_t M = E(shp(0)[0]), Kd = E(shp(0)[1]), N = E(shp(1)[1]);
          if (accv && accv != out) std::memcpy(out, accv, static_cast<std::size_t>(cnt) * 4);
          cpu::GemmArgs ga;
          ga.M = static_cast<int>(M);
          ga.N = static_cast<int>(N);
          ga.K = static_cast<int>(Kd);
          ga.A = in(0);
          ga.lda = static_cast<int>(Kd);
          ga.B = in(1);
          ga.ldb = static_cast<int>(N);
          ga.C = out;
          ga.ldc = static_cast<int>(N);
          ga.beta = accv ? 1.0f : 0.0f;
          cpu::gemm(ga);
        } else if (n.op == Op::MatMulNT) {
          const std::int64_t M = E(shp(0)[0]), Kd = E(shp(0)[1]), N = E(shp(1)[0]);
          if (accv && accv != out) std::memcpy(out, accv, static_cast<std::size_t>(cnt) * 4);
          cpu::GemmArgs ga;
          ga.M = static_cast<int>(M);
          ga.N = static_cast<int>(N);
          ga.K = static_cast<int>(Kd);
          ga.A = in(0);
          ga.lda = static_cast<int>(Kd);
          ga.B = in(1);
          ga.ldb = static_cast<int>(Kd);
          ga.b_trans = true;
          ga.C = out;
          ga.ldc = static_cast<int>(N);
          ga.beta = accv ? 1.0f : 0.0f;
          cpu::gemm(ga);
          if (fl & 4) {
            const float* pre = in(2);
            for (std::int64_t e = 0; e < cnt; ++e) out[e] = scalar_op(Op::GeluBwd, out[e], pre[e], 0.0);
          }
        } else {
          const std::int64_t Kd = E(shp(0)[0]), M = E(shp(0)[1]), N = E(shp(1)[1]);
          if (accv && accv != out) std::memcpy(out, accv, static_cast<std::size_t>(cnt) * 4);
          cpu::GemmArgs ga;
          ga.M = static_cast<int>(M);
          ga.N = static_cast<int>(N);
          ga.K = static_cast<int>(Kd);
          ga.A = in(0);
          ga.lda = static_cast<int>(M);
          ga.a_trans = true;
          ga.B = in(1);
          ga.ldb = static_cast<int>(N);
          ga.C = out;
          ga.ldc = static_cast<int>(N);
          ga.beta = accv ? 1.0f : 0.0f;
          cpu::gemm(ga);
          if (fl & 2) {
            float* db = aux_ptr(i, 1, static_cast<std::size_t>(N));
            sum_rows_cpu(in(1), db, Kd, N);
          }
        }
        break;
      }
      case Op::MatMulBias: {
        const std::int64_t M = E(shp(0)[0]), Kd = E(shp(0)[1]), N = E(shp(1)[1]);
        cpu::GemmArgs ga;
        ga.M = static_cast<int>(M);
        ga.N = static_cast<int>(N);
        ga.K = static_cast<int>(Kd);
        ga.A = in(0);
        ga.lda = static_cast<int>(Kd);
        ga.B = in(1);
        ga.ldb = static_cast<int>(N);
        ga.C = out;
        ga.ldc = static_cast<int>(N);
        ga.bias = in(2);
        cpu::gemm(ga);
        break;
      }
      case Op::AttentionQKV: case Op::AttentionQKVBwd: {
        const std::int64_t H = n.ints[0], Hk = n.ints[1], D = n.ints[2], B = b.at(static_cast<int>(n.ints[3])), T = b.at(static_cast<int>(n.ints[4]));
        const bool bwd = n.op == Op::AttentionQKVBwd;
        const float* qkv = in(bwd ? 1 : 0);
        const std::int64_t rows = B * T, w = (H + 2 * Hk) * D;
        if (!bwd) {
          // FAZ 18.6: ileri geciste qkv tamponunu kopyalamadan dogrudan adimli (strided) oku.
          AttentionParams p = attn_params(B, T, H, Hk, D, n.a);
          p.q_row_stride = w;
          p.q_head_stride = D;
          p.q_batch_stride = T * w;
          p.k_row_stride = w;
          p.k_head_stride = D;
          p.k_batch_stride = T * w;
          p.o_row_stride = H * D;
          p.o_head_stride = D;
          p.o_batch_stride = T * H * D;
          float* lse = aux_ptr(i, 1, static_cast<std::size_t>(B * H * T));
          attention::flash_attention_forward_cpu(out, lse, qkv, qkv + H * D, qkv + (H + Hk) * D, p, I.opt.attention_threads);
        } else {
          const AttentionParams p = attn_params(B, T, H, Hk, D, n.a);
          std::vector<float> q(static_cast<std::size_t>(rows * H * D)), k(static_cast<std::size_t>(rows * Hk * D)), v(k.size());
          for (std::int64_t r = 0; r < rows; ++r) {
            std::memcpy(q.data() + r * H * D, qkv + r * w, static_cast<std::size_t>(H * D) * 4);
            std::memcpy(k.data() + r * Hk * D, qkv + r * w + H * D, static_cast<std::size_t>(Hk * D) * 4);
            std::memcpy(v.data() + r * Hk * D, qkv + r * w + (H + Hk) * D, static_cast<std::size_t>(Hk * D) * 4);
          }
          std::vector<float> dq(q.size()), dk(k.size()), dv(v.size());
          attention::flash_attention_backward_cpu(dq.data(), dk.data(), dv.data(), in(0), q.data(), k.data(), v.data(), in(2), in(3),
                                                  p, I.opt.attention_threads);
          for (std::int64_t r = 0; r < rows; ++r) {
            std::memcpy(out + r * w, dq.data() + r * H * D, static_cast<std::size_t>(H * D) * 4);
            std::memcpy(out + r * w + H * D, dk.data() + r * Hk * D, static_cast<std::size_t>(Hk * D) * 4);
            std::memcpy(out + r * w + (H + Hk) * D, dv.data() + r * Hk * D, static_cast<std::size_t>(Hk * D) * 4);
          }
        }
        break;
      }
      case Op::SliceCols: {
        const std::int64_t rows = E(n.shape[0]), src = E(shp(0)[1]), b0 = n.ints[0], len = n.ints[1];
        for (std::int64_t r = 0; r < rows; ++r) std::memcpy(out + r * len, in(0) + r * src + b0, static_cast<std::size_t>(len) * 4);
        break;
      }
      case Op::ConcatCols: {
        const std::int64_t rows = E(n.shape[0]), cols = E(n.shape[1]);
        std::int64_t at = 0;
        for (std::size_t k = 0; k < n.in.size(); ++k) {
          const std::int64_t w = E(shp(static_cast<int>(k))[1]);
          for (std::int64_t r = 0; r < rows; ++r) std::memcpy(out + r * cols + at, in(static_cast<int>(k)) + r * w, static_cast<std::size_t>(w) * 4);
          at += w;
        }
        break;
      }
      case Op::SplitHeads: case Op::MergeHeads: {
        const std::int64_t h = n.ints[0], D = n.ints[1], B = b.at(static_cast<int>(n.ints[2])), T = b.at(static_cast<int>(n.ints[3]));
        const float* x = in(0);
        for (std::int64_t bb = 0; bb < B; ++bb)
          for (std::int64_t hh = 0; hh < h; ++hh)
            for (std::int64_t t = 0; t < T; ++t) {
              const std::int64_t flat = ((bb * T + t) * h + hh) * D, split = ((bb * h + hh) * T + t) * D;
              if (n.op == Op::SplitHeads) {
                std::memcpy(out + split, x + flat, static_cast<std::size_t>(D) * 4);
              } else {
                std::memcpy(out + flat, x + split, static_cast<std::size_t>(D) * 4);
              }
            }
        break;
      }
      case Op::RepeatKV: {
        const std::int64_t groups = n.ints[0], H = n.shape[0].coef, Hk = H / groups;
        const std::int64_t T = E(n.shape[1]), D = E(n.shape[2]), G = E(n.shape[0]), B = G / H;
        for (std::int64_t bb = 0; bb < B; ++bb)
          for (std::int64_t hh = 0; hh < H; ++hh)
            std::memcpy(out + (bb * H + hh) * T * D, in(0) + (bb * Hk + hh / groups) * T * D, static_cast<std::size_t>(T * D) * 4);
        break;
      }
      case Op::BMMNT: {
        const std::int64_t G = E(shp(0)[0]), T = E(shp(0)[1]), D = E(shp(0)[2]), T2 = E(shp(1)[1]);
        for (std::int64_t q = 0; q < G; ++q) {
          cpu::GemmArgs ga;
          ga.M = static_cast<int>(T);
          ga.N = static_cast<int>(T2);
          ga.K = static_cast<int>(D);
          ga.A = in(0) + q * T * D;
          ga.lda = static_cast<int>(D);
          ga.B = in(1) + q * T2 * D;
          ga.ldb = static_cast<int>(D);
          ga.b_trans = true;
          ga.C = out + q * T * T2;
          ga.ldc = static_cast<int>(T2);
          cpu::gemm(ga);
        }
        break;
      }
      case Op::BMM: {
        const std::int64_t G = E(shp(0)[0]), T = E(shp(0)[1]), T2 = E(shp(0)[2]), D = E(shp(1)[2]);
        for (std::int64_t q = 0; q < G; ++q) {
          nexus::cpu::gemm_nn(in(0) + q * T * T2, in(1) + q * T2 * D, out + q * T * D, static_cast<int>(T), static_cast<int>(T2),
                              static_cast<int>(D));
        }
        break;
      }
      case Op::CausalMask: {
        const std::int64_t G = E(n.shape[0]), T = E(n.shape[1]), T2 = E(n.shape[2]);
        const float* x = in(0);
        for (std::int64_t q = 0; q < G; ++q)
          for (std::int64_t r = 0; r < T; ++r)
            for (std::int64_t c = 0; c < T2; ++c) {
              const std::int64_t at = (q * T + r) * T2 + c;
              out[at] = c > r ? -std::numeric_limits<float>::infinity() : x[at];
            }
        break;
      }
      case Op::Softmax: {
        const std::int64_t c = E(n.shape.back()), rows = cnt / c;
        const float* x = in(0);
        for (std::int64_t r = 0; r < rows; ++r) {
          float m = -std::numeric_limits<float>::infinity();
          for (std::int64_t k = 0; k < c; ++k) m = std::max(m, x[r * c + k]);
          double s = 0.0;
          for (std::int64_t k = 0; k < c; ++k) s += std::exp(x[r * c + k] - m);
          for (std::int64_t k = 0; k < c; ++k) out[r * c + k] = static_cast<float>(std::exp(x[r * c + k] - m) / s);
        }
        break;
      }
      case Op::LayerNorm: case Op::RmsNorm: {
        const std::int64_t c = E(n.shape[1]);
        norm_fwd(n.op == Op::RmsNorm, in(0), in(1), n.op == Op::LayerNorm && n.ints[0] == 1 ? in(2) : nullptr, out, cnt / c, c, n.a);
        break;
      }
      case Op::LayerNormBwdX: case Op::RmsNormBwdX: {
        // ints[0]: 1 = dw (Output 1) ve LN'de db (Output 2) ayni geciste; 2 = son girdi (artik gradyani) eklenir.
        const std::int64_t c = E(n.shape[1]);
        const std::int64_t fl = n.ints.empty() ? 0 : n.ints[0];
        float* dw = (fl & 1) ? aux_ptr(i, 1, static_cast<std::size_t>(c)) : nullptr;
        float* db = (fl & 1) && n.op == Op::LayerNormBwdX && I.aux[i].size() > 2 && I.aux[i][2] >= 0 ? aux_ptr(i, 2, static_cast<std::size_t>(c)) : nullptr;
        norm_bwd(n.op == Op::RmsNormBwdX, in(0), in(1), in(2), out, dw, db, cnt / c, c, n.a);
        if (fl & 2) {
          add_arrays_cpu(out, out, in(static_cast<int>(n.in.size()) - 1), cnt);
        }
        break;
      }
      case Op::LayerNormBwdW: case Op::RmsNormBwdW: {
        const std::int64_t c = E(n.shape[0]);
        const std::int64_t rows = ir::numel(shp(0), b) / c;
        float* db = n.op == Op::LayerNormBwdW && n.ints[0] == 1 ? aux_ptr(i, 1, static_cast<std::size_t>(c)) : nullptr;
        norm_bwd(n.op == Op::RmsNormBwdW, in(0), in(1), nullptr, nullptr, out, db, rows, c, n.a);
        break;
      }
      case Op::AddNorm: {
        const std::int64_t c = E(n.shape[1]);
        float* r = aux_ptr(i, 1, static_cast<std::size_t>(cnt));
        add_arrays_cpu(r, in(0), in(1), cnt);
        norm_fwd(n.ints[0] == 1, r, in(2), n.ints[1] == 1 ? in(3) : nullptr, out, cnt / c, c, n.a);
        break;
      }
      case Op::Rope: case Op::RopeBwd: {
        const std::int64_t heads = n.ints[0], D = n.ints[1], rot = n.ints[2], T = b.at(static_cast<int>(n.ints[3]));
        const std::int64_t rows = E(n.shape[0]);
        lm::RopeCache cache(static_cast<int>(rot), static_cast<float>(n.a), lm::RopeScaling{}, static_cast<int>(std::max<std::int64_t>(T, 1)));
        cache.ensure(static_cast<int>(T));
        std::vector<int> pos(static_cast<std::size_t>(rows));
        for (std::int64_t r = 0; r < rows; ++r) pos[static_cast<std::size_t>(r)] = static_cast<int>(r % T);
        std::memcpy(out, in(0), static_cast<std::size_t>(cnt) * 4);
        lm::rope_apply(out, static_cast<int>(rows), static_cast<int>(heads), static_cast<int>(D), pos.data(), cache, n.op == Op::RopeBwd);
        break;
      }
      case Op::Attention: {
        const std::int64_t H = n.ints[0], Hk = n.ints[1], D = n.ints[2], B = b.at(static_cast<int>(n.ints[3])), T = b.at(static_cast<int>(n.ints[4]));
        const AttentionParams p = attn_params(B, T, H, Hk, D, n.a);
        float* lse = aux_ptr(i, 1, static_cast<std::size_t>(B * H * T));
        attention::flash_attention_forward_cpu(out, lse, in(0), in(1), in(2), p, I.opt.attention_threads);
        break;
      }
      case Op::AttentionBwd: {
        const std::int64_t H = n.ints[0], Hk = n.ints[1], D = n.ints[2], B = b.at(static_cast<int>(n.ints[3])), T = b.at(static_cast<int>(n.ints[4]));
        const AttentionParams p = attn_params(B, T, H, Hk, D, n.a);
        const std::size_t kv = static_cast<std::size_t>(ir::numel(shp(2), b));
        float* dk = aux_ptr(i, 1, kv);
        float* dv = aux_ptr(i, 2, kv);
        attention::flash_attention_backward_cpu(out, dk, dv, in(0), in(1), in(2), in(3), in(4), in(5), p, I.opt.attention_threads);
        break;
      }
      case Op::Embedding: {
        const std::int64_t c = E(n.shape[1]), rows = E(n.shape[0]), V = E(shp(0)[0]);
        const std::int32_t* ids = I32(n.in[1]);
        for (std::int64_t r = 0; r < rows; ++r) {
          if (ids[r] < 0 || ids[r] >= V) bad("Embedding: kimlik tablo disinda");
          std::memcpy(out + r * c, in(0) + static_cast<std::int64_t>(ids[r]) * c, static_cast<std::size_t>(c) * 4);
        }
        break;
      }
      case Op::EmbeddingBwd: {
        const std::int64_t c = E(n.shape[1]);
        const std::int64_t rows = E(shp(0)[0]);
        const std::int32_t* ids = I32(n.in[1]);
        std::memset(out, 0, static_cast<std::size_t>(cnt) * sizeof(float));
        for (std::int64_t r = 0; r < rows; ++r) {
          const float* gr = in(0) + r * c;
          float* o = out + static_cast<std::int64_t>(ids[r]) * c;
          add_arrays_cpu(o, o, gr, c);
        }
        break;
      }
      case Op::CrossEntropy: case Op::CrossEntropyBwd: {
        const std::int64_t rows = E(shp(0)[0]), Vp = E(shp(0)[1]), V = n.ints[0], T = b.at(static_cast<int>(n.ints[1]));
        std::vector<float> lg(in(0), in(0) + rows * Vp);
        for (std::int64_t r = 0; r < rows; ++r)
          for (std::int64_t k = V; k < Vp; ++k) lg[static_cast<std::size_t>(r * Vp + k)] = -std::numeric_limits<float>::infinity();
        const fused::CrossEntropyOptions co;
        if (n.op == Op::CrossEntropy) {
          out[0] = fused::cross_entropy_shifted(lg.data(), I32(n.in[1]), static_cast<int>(rows / T), static_cast<int>(T),
                                                static_cast<int>(Vp), co, nullptr).loss;
        } else {
          (void)fused::cross_entropy_shifted(lg.data(), I32(n.in[1]), static_cast<int>(rows / T), static_cast<int>(T),
                                             static_cast<int>(Vp), co, out);
          const float gsc = in(2)[0];
          for (std::int64_t e = 0; e < cnt; ++e) out[e] *= gsc;
        }
        break;
      }
      case Op::FusedLinearCE: case Op::FusedLinearCEBwd: {
        const float* h = in(0);
        const float* w = in(1);
        const std::int32_t* tok = I32(n.in[2]);
        const std::int64_t rows = E(shp(0)[0]), C = E(shp(0)[1]), Vp = E(shp(1)[0]), V = n.ints[0];
        const std::int64_t T = b.at(static_cast<int>(n.ints[1]));
        const std::int64_t chunk = std::max<std::int64_t>(1, n.ints[2]);
        auto target = [&](std::int64_t r) -> std::int32_t { return (r % T) + 1 < T ? tok[r + 1] : -100; };
        std::int64_t valid = 0;
        for (std::int64_t r = 0; r < rows; ++r) valid += target(r) != -100 ? 1 : 0;
        const bool bwd = n.op == Op::FusedLinearCEBwd;
        float* dw = bwd ? aux_ptr(i, 1, static_cast<std::size_t>(Vp * C)) : nullptr;
        if (bwd) {
          std::memset(dw, 0, static_cast<std::size_t>(Vp * C) * sizeof(float));
          std::memset(out, 0, static_cast<std::size_t>(rows * C) * sizeof(float));
        }
        const float scale = bwd && valid > 0 ? in(3)[0] / static_cast<float>(valid) : 0.0f;
        std::vector<float> lg(static_cast<std::size_t>(chunk * Vp));
        double loss = 0.0;
        for (std::int64_t r0 = 0; r0 < rows; r0 += chunk) {
          const std::int64_t R = std::min(chunk, rows - r0);
          cpu::GemmArgs ga;
          ga.M = static_cast<int>(R);
          ga.N = static_cast<int>(Vp);
          ga.K = static_cast<int>(C);
          ga.A = h + r0 * C;
          ga.lda = static_cast<int>(C);
          ga.B = w;
          ga.ldb = static_cast<int>(C);
          ga.b_trans = true;
          ga.C = lg.data();
          ga.ldc = static_cast<int>(Vp);
          cpu::gemm(ga);
          for (std::int64_t r = 0; r < R; ++r) {
            float* x = lg.data() + r * Vp;
            const std::int32_t t = target(r0 + r);
            if (t == -100) {
              std::memset(x, 0, static_cast<std::size_t>(Vp) * sizeof(float));
              continue;
            }
            float m = -std::numeric_limits<float>::infinity();
            for (std::int64_t k = 0; k < V; ++k) m = std::max(m, x[k]);
            double s = 0.0;
            for (std::int64_t k = 0; k < V; ++k) s += std::exp(static_cast<double>(x[k] - m));
            const double lse = m + std::log(s);
            loss += lse - x[t];
            if (!bwd) continue;
            for (std::int64_t k = 0; k < Vp; ++k) {
              x[k] = k < V ? static_cast<float>(std::exp(x[k] - lse)) * scale : 0.0f;
            }
            x[t] -= scale;
          }
          if (bwd) {
            nexus::cpu::gemm_nn(lg.data(), w, out + r0 * C, static_cast<int>(R), static_cast<int>(Vp), static_cast<int>(C));
            nexus::cpu::gemm_tn_accum(lg.data(), h + r0 * C, dw, static_cast<int>(Vp), static_cast<int>(R), static_cast<int>(C));
          }
        }
        if (!bwd) out[0] = valid > 0 ? static_cast<float>(loss / static_cast<double>(valid)) : 0.0f;
        break;
      }
      case Op::SwiGlu: case Op::SwiGluBwd: {
        const bool bwd = n.op == Op::SwiGluBwd;
        const float* u = bwd ? in(1) : in(0);
        const std::int64_t F = E(n.shape[1]) / (bwd ? 2 : 1);
        const std::int64_t rows = E(n.shape[0]);
        auto run_rows = [&](std::int64_t r0, std::int64_t r1) {
          for (std::int64_t r = r0; r < r1; ++r) {
            const float* a = u + r * 2 * F;
            const float* bb = a + F;
            std::int64_t k = 0;
#if defined(__AVX2__)
            const __m256 one = _mm256_set1_ps(1.0f);
            const __m256 zero = _mm256_setzero_ps();
            for (; k + 8 <= F; k += 8) {
              const __m256 va = _mm256_loadu_ps(a + k);
              const __m256 vbb = _mm256_loadu_ps(bb + k);
              const __m256 sg = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp256_ps(_mm256_sub_ps(zero, va))));
              const __m256 vf = _mm256_mul_ps(va, sg);
              if (!bwd) {
                _mm256_storeu_ps(out + r * F + k, _mm256_mul_ps(vf, vbb));
              } else {
                const __m256 vgy = _mm256_loadu_ps(in(0) + r * F + k);
                const __m256 vdf = _mm256_mul_ps(sg, _mm256_fmadd_ps(va, _mm256_sub_ps(one, sg), one));
                _mm256_storeu_ps(out + r * 2 * F + k, _mm256_mul_ps(_mm256_mul_ps(vgy, vbb), vdf));
                _mm256_storeu_ps(out + r * 2 * F + F + k, _mm256_mul_ps(vgy, vf));
              }
            }
#endif
            for (; k < F; ++k) {
              const float sg = 1.0f / (1.0f + std::exp(-a[k]));
              const float f = a[k] * sg;
              if (!bwd) {
                out[r * F + k] = f * bb[k];
              } else {
                const float gy = in(0)[r * F + k];
                const float df = sg * (1.0f + a[k] * (1.0f - sg));
                out[r * 2 * F + k] = gy * bb[k] * df;
                out[r * 2 * F + F + k] = gy * f;
              }
            }
          }
        };
        if (rows * F >= 8192 && rows >= 2) {
          cpu::Pool::instance().run(static_cast<std::size_t>(rows), [&](int, std::size_t r0, std::size_t r1) {
            run_rows(static_cast<std::int64_t>(r0), static_cast<std::int64_t>(r1));
          });
        } else {
          run_rows(0, rows);
        }
        break;
      }
      case Op::LinearBiasGelu: {
        const std::int64_t M = E(shp(0)[0]), Kd = E(shp(0)[1]), N = E(shp(1)[1]);
        float* pre = aux_ptr(i, 1, static_cast<std::size_t>(cnt));
        cpu::GemmArgs ga;
        ga.M = static_cast<int>(M);
        ga.N = static_cast<int>(N);
        ga.K = static_cast<int>(Kd);
        ga.A = in(0);
        ga.lda = static_cast<int>(Kd);
        ga.B = in(1);
        ga.ldb = static_cast<int>(N);
        ga.C = pre;
        ga.ldc = static_cast<int>(N);
        ga.bias = in(2);
        cpu::gemm(ga);
        for (std::int64_t e = 0; e < cnt; ++e) {
          out[e] = 0.5f * pre[e] * (1.0f + std::erf(pre[e] * kInvSqrt2));
        }
        break;
      }
      case Op::AllGather: case Op::ReduceScatter: case Op::AllReduce: {
        if (!I.group) bad("iletisim dugumu icin ProcessGroup yok (set_group)");
        const std::int64_t src = ir::numel(shp(0), b);
        if (n.op == Op::AllGather) {
          pending[static_cast<int>(i)] = I.group->all_gather(in(0), out, static_cast<std::size_t>(src), DType::F32);
        } else if (n.op == Op::ReduceScatter) {
          pending[static_cast<int>(i)] = I.group->reduce_scatter(in(0), out, static_cast<std::size_t>(cnt), DType::F32, dist::ReduceOp::Sum);
        } else {
          std::memcpy(out, in(0), static_cast<std::size_t>(cnt) * 4);
          pending[static_cast<int>(i)] = I.group->all_reduce(out, static_cast<std::size_t>(cnt), DType::F32, dist::ReduceOp::Sum);
        }
        break;
      }
      case Op::Fused: {
        const ir::FusedOp& f = *n.fused;
        std::vector<const float*> ins;
        for (int x : n.in) ins.push_back(P(x));
        std::vector<float*> outs{out};
        for (std::size_t k = 1; k < f.outputs.size(); ++k) outs.push_back(aux_ptr(i, static_cast<int>(k), static_cast<std::size_t>(cnt)));
        bool row = false;
        for (bool r : f.input_is_row) row = row || r;
        const long long cols = row ? static_cast<long long>(E(n.shape.back())) : static_cast<long long>(cnt);
        const long long rows = cols > 0 ? static_cast<long long>(cnt) / cols : 0;
        const std::string nm = ir::kernel_name(f);
        auto pick = I.scalar_choice.find(f.key);
        if (I.opt.mode == "max-autotune" && pick == I.scalar_choice.end()) {
          double best[2] = {0, 0};
          for (int v = 0; v < 2; ++v) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 3; ++rep) {
              dispatch_fused_cpu(I.kernels.at(v ? nm + "_s" : nm), f, ins, outs, rows, cols);
            }
            best[v] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          }
          pick = I.scalar_choice.emplace(f.key, best[1] < best[0]).first;
          autotuned_ += 1;
        }
        const bool scalar = pick != I.scalar_choice.end() && pick->second;
        dispatch_fused_cpu(I.kernels.at(scalar ? nm + "_s" : nm), f, ins, outs, rows, cols);
        break;
      }
      default: {
        if (!ir::is_elementwise(n.op)) bad(std::string("yurutulemeyen op: ") + ir::to_string(n.op));
        const float* x = in(0);
        const float* y = n.in.size() > 1 ? in(1) : nullptr;
        for (std::int64_t e = 0; e < cnt; ++e) out[e] = scalar_op(n.op, x[e], y ? y[e] : 0.0f, n.a);
      }
    }
  }
  for (auto& kv : pending) kv.second->wait();
  std::vector<std::vector<float>> outs;
  for (int o : g.outputs) {
    const std::int64_t c = ir::numel(g.nodes[static_cast<std::size_t>(o)].shape, b);
    const float* p = P(o);
    outs.emplace_back(p, p + c);
  }
  return outs;
}

}  // namespace compile
}  // namespace nexus
