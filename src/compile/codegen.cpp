/**
 * @file codegen.cpp
 * @brief FAZ 13.4: fuzyon kumeleri -> CPU (AVX2/SVML C++) ve CUDA kaynagi; CPU JIT (cl.exe -> DLL, disk onbellegi).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/codegen.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace nexus {
namespace ir {

namespace fs = std::filesystem;

std::string source_hash(const std::string& text) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : text) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

std::string kernel_name(const FusedOp& f) { return "nxk_" + source_hash(f.key); }

namespace {

std::string lit(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9gf", v);
  std::string s = buf;
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find("inf") == std::string::npos &&
      s.find("nan") == std::string::npos) {
    s.insert(s.size() - 1, ".0");
  }
  return s;
}

/// Skaler ifade (CPU ve CUDA ortak: expf/erff/tanhf/logf/sqrtf).
std::string scalar_expr(const FusedOp::Inst& in, const std::vector<std::string>& a) {
  const std::string& x = a.empty() ? std::string() : a[0];
  const std::string y = a.size() > 1 ? a[1] : std::string();
  switch (in.op) {
    case Op::Add: case Op::BiasAdd: return x + " + " + y;
    case Op::Sub: return x + " - " + y;
    case Op::Mul: return x + " * " + y;
    case Op::Div: return x + " / " + y;
    case Op::Neg: return "-" + x;
    case Op::Exp: return "expf(" + x + ")";
    case Op::Log: return "logf(" + x + ")";
    case Op::Tanh: return "tanhf(" + x + ")";
    case Op::Sigmoid: return "1.0f / (1.0f + expf(-" + x + "))";
    case Op::Gelu: return "0.5f * " + x + " * (1.0f + erff(" + x + " * 0.70710678118654752f))";
    case Op::GeluTanh:
      return "0.5f * " + x + " * (1.0f + tanhf(0.7978845608028654f * (" + x + " + 0.044715f * " + x + " * " + x + " * " + x + ")))";
    case Op::Silu: return x + " / (1.0f + expf(-" + x + "))";
    case Op::Relu: return "(" + x + " > 0.0f ? " + x + " : 0.0f)";
    case Op::Square: return x + " * " + x;
    case Op::Sqrt: return "sqrtf(" + x + ")";
    case Op::Rsqrt: return "1.0f / sqrtf(" + x + ")";
    case Op::Scale: return x + " * " + lit(in.a);
    case Op::AddScalar: return x + " + " + lit(in.a);
    case Op::GeluBwd:  // dy = x, girdi = y
      return x + " * (0.5f * (1.0f + erff(" + y + " * 0.70710678118654752f)) + " + y + " * 0.39894228040143268f * expf(-0.5f * " + y +
             " * " + y + "))";
    case Op::GeluTanhBwd:
      return "nx_gelu_tanh_bwd(" + x + ", " + y + ")";
    case Op::SiluBwd:
      return "nx_silu_bwd(" + x + ", " + y + ")";
    case Op::SigmoidBwd: return x + " * " + y + " * (1.0f - " + y + ")";
    case Op::TanhBwd: return x + " * (1.0f - " + y + " * " + y + ")";
    default:
      throw std::runtime_error(std::string("Nexus codegen: fuzyonda desteklenmeyen op ") + to_string(in.op));
  }
}

/// AVX2 ifadesi (MSVC SVML).
std::string vector_expr(const FusedOp::Inst& in, const std::vector<std::string>& a) {
  const std::string& x = a.empty() ? std::string() : a[0];
  const std::string y = a.size() > 1 ? a[1] : std::string();
  auto c = [](double v) { return "_mm256_set1_ps(" + lit(v) + ")"; };
  switch (in.op) {
    case Op::Add: case Op::BiasAdd: return "_mm256_add_ps(" + x + ", " + y + ")";
    case Op::Sub: return "_mm256_sub_ps(" + x + ", " + y + ")";
    case Op::Mul: return "_mm256_mul_ps(" + x + ", " + y + ")";
    case Op::Div: return "_mm256_div_ps(" + x + ", " + y + ")";
    case Op::Neg: return "_mm256_sub_ps(_mm256_setzero_ps(), " + x + ")";
    case Op::Exp: return "_mm256_exp_ps(" + x + ")";
    case Op::Log: return "_mm256_log_ps(" + x + ")";
    case Op::Tanh: return "_mm256_tanh_ps(" + x + ")";
    case Op::Sigmoid: return "nxv_sigmoid(" + x + ")";
    case Op::Gelu: return "nxv_gelu(" + x + ")";
    case Op::GeluTanh: return "nxv_gelu_tanh(" + x + ")";
    case Op::Silu: return "_mm256_mul_ps(" + x + ", nxv_sigmoid(" + x + "))";
    case Op::Relu: return "_mm256_max_ps(" + x + ", _mm256_setzero_ps())";
    case Op::Square: return "_mm256_mul_ps(" + x + ", " + x + ")";
    case Op::Sqrt: return "_mm256_sqrt_ps(" + x + ")";
    case Op::Rsqrt: return "_mm256_div_ps(" + c(1.0) + ", _mm256_sqrt_ps(" + x + "))";
    case Op::Scale: return "_mm256_mul_ps(" + x + ", " + c(in.a) + ")";
    case Op::AddScalar: return "_mm256_add_ps(" + x + ", " + c(in.a) + ")";
    case Op::GeluBwd: return "nxv_gelu_bwd(" + x + ", " + y + ")";
    case Op::GeluTanhBwd: return "nxv_gelu_tanh_bwd(" + x + ", " + y + ")";
    case Op::SiluBwd: return "nxv_silu_bwd(" + x + ", " + y + ")";
    case Op::SigmoidBwd: return "_mm256_mul_ps(_mm256_mul_ps(" + x + ", " + y + "), _mm256_sub_ps(" + c(1.0) + ", " + y + "))";
    case Op::TanhBwd: return "_mm256_mul_ps(" + x + ", _mm256_sub_ps(" + c(1.0) + ", _mm256_mul_ps(" + y + ", " + y + ")))";
    default:
      throw std::runtime_error(std::string("Nexus codegen: fuzyonda desteklenmeyen op ") + to_string(in.op));
  }
}

const char* kScalarHelpers = R"(
static NX_INLINE float nx_silu_bwd(float dy, float x) { const float s = 1.0f / (1.0f + expf(-x)); return dy * s * (1.0f + x * (1.0f - s)); }
static NX_INLINE float nx_gelu_tanh_bwd(float dy, float x) {
  const float k = 0.7978845608028654f, u = k * (x + 0.044715f * x * x * x), t = tanhf(u);
  return dy * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * k * (1.0f + 3.0f * 0.044715f * x * x));
}
)";

}  // namespace

std::string cpu_source(const std::vector<const FusedOp*>& kernels) {
  std::ostringstream o;
  o << "// Nexus FAZ 13 uretilmis CPU cekirdekleri (AVX2 + SVML)\n#include <immintrin.h>\n#include <math.h>\n";
  o << "#define NX_INLINE __forceinline\n" << kScalarHelpers;
  o << R"(
static NX_INLINE __m256 nxv_sigmoid(__m256 x) { const __m256 one = _mm256_set1_ps(1.0f);
  return _mm256_div_ps(one, _mm256_add_ps(one, _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x)))); }
static NX_INLINE __m256 nxv_gelu(__m256 x) { const __m256 h = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
  return _mm256_mul_ps(_mm256_mul_ps(h, x), _mm256_add_ps(one, _mm256_erf_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.70710678118654752f))))); }
static NX_INLINE __m256 nxv_gelu_tanh(__m256 x) { const __m256 h = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
  const __m256 u = _mm256_mul_ps(_mm256_set1_ps(0.7978845608028654f), _mm256_add_ps(x, _mm256_mul_ps(_mm256_set1_ps(0.044715f), _mm256_mul_ps(x, _mm256_mul_ps(x, x)))));
  return _mm256_mul_ps(_mm256_mul_ps(h, x), _mm256_add_ps(one, _mm256_tanh_ps(u))); }
static NX_INLINE __m256 nxv_gelu_bwd(__m256 dy, __m256 x) { const __m256 h = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
  const __m256 cdf = _mm256_mul_ps(h, _mm256_add_ps(one, _mm256_erf_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.70710678118654752f)))));
  const __m256 pdf = _mm256_exp_ps(_mm256_mul_ps(_mm256_set1_ps(-0.5f), _mm256_mul_ps(x, x)));
  return _mm256_mul_ps(dy, _mm256_add_ps(cdf, _mm256_mul_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.39894228040143268f)), pdf))); }
static NX_INLINE __m256 nxv_silu_bwd(__m256 dy, __m256 x) { const __m256 one = _mm256_set1_ps(1.0f), s = nxv_sigmoid(x);
  return _mm256_mul_ps(dy, _mm256_mul_ps(s, _mm256_add_ps(one, _mm256_mul_ps(x, _mm256_sub_ps(one, s))))); }
static NX_INLINE __m256 nxv_gelu_tanh_bwd(__m256 dy, __m256 x) { float a[8], b[8]; _mm256_storeu_ps(a, dy); _mm256_storeu_ps(b, x);
  for (int i = 0; i < 8; ++i) a[i] = nx_gelu_tanh_bwd(a[i], b[i]); return _mm256_loadu_ps(a); }
)";
  for (const FusedOp* f : kernels) {
   for (int variant = 0; variant < 2; ++variant) {  // 0: AVX2 + skaler kuyruk, 1: yalniz skaler ("_s", autotune)
    const std::size_t nin = f->input_is_row.size();
    o << "\nextern \"C\" __declspec(dllexport) void " << kernel_name(*f) << (variant ? "_s" : "")
      << "(const float* const* in, float* const* out, long long rows, long long cols) {\n";
    o << "  for (long long r = 0; r < rows; ++r) {\n    long long c = 0;\n    for (; " << (variant ? "false" : "c + 8 <= cols")
      << "; c += 8) {\n";
    for (std::size_t e = 0; e < nin; ++e) {
      o << "      const __m256 x" << e << " = _mm256_loadu_ps(in[" << e << "] + " << (f->input_is_row[e] ? "c" : "r * cols + c")
        << ");\n";
    }
    auto arg = [&](int a, const char* pre) { return a < 0 ? std::string(pre) + std::to_string(-a - 1) : "t" + std::to_string(a); };
    for (std::size_t k = 0; k < f->insts.size(); ++k) {
      std::vector<std::string> a;
      for (int x : f->insts[k].args) a.push_back(arg(x, "x"));
      o << "      const __m256 t" << k << " = " << vector_expr(f->insts[k], a) << ";\n";
    }
    for (std::size_t k = 0; k < f->outputs.size(); ++k) {
      o << "      _mm256_storeu_ps(out[" << k << "] + r * cols + c, t" << f->outputs[k] << ");\n";
    }
    o << "    }\n    for (; c < cols; ++c) {\n";
    for (std::size_t e = 0; e < nin; ++e) {
      o << "      const float x" << e << " = in[" << e << "][" << (f->input_is_row[e] ? "c" : "r * cols + c") << "];\n";
    }
    for (std::size_t k = 0; k < f->insts.size(); ++k) {
      std::vector<std::string> a;
      for (int x : f->insts[k].args) a.push_back(arg(x, "x"));
      o << "      const float t" << k << " = " << scalar_expr(f->insts[k], a) << ";\n";
    }
    for (std::size_t k = 0; k < f->outputs.size(); ++k) o << "      out[" << k << "][r * cols + c] = t" << f->outputs[k] << ";\n";
    o << "    }\n  }\n}\n";
   }
  }
  return o.str();
}

std::string cuda_source(const std::vector<const FusedOp*>& kernels) {
  std::ostringstream o;
  o << "// Nexus FAZ 13 uretilmis CUDA cekirdekleri (NVRTC)\n#define NX_INLINE __device__ __forceinline__\n" << kScalarHelpers;
  for (const FusedOp* f : kernels) {
    const std::size_t nin = f->input_is_row.size();
    o << "\nextern \"C\" __global__ void " << kernel_name(*f)
      << "(const float* const* in, float* const* out, long long rows, long long cols) {\n";
    o << "  const long long n = rows * cols;\n";
    o << "  if ((cols & 3) == 0) {\n    const long long n4 = n / 4;\n";
    o << "    for (long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x; q < n4; q += (long long)gridDim.x * blockDim.x) {\n";
    o << "      const long long i = q * 4, c = i % cols;\n";
    for (std::size_t e = 0; e < nin; ++e) {
      o << "      const float4 x" << e << "v = *reinterpret_cast<const float4*>(in[" << e << "] + " << (f->input_is_row[e] ? "c" : "i")
        << ");\n";
    }
    auto arg = [&](int a, const std::string& suf) {
      return a < 0 ? "x" + std::to_string(-a - 1) + suf : "t" + std::to_string(a) + suf;
    };
    o << "      float4 r4;\n";
    for (const char* lane : {"x", "y", "z", "w"}) {
      const std::string suf = std::string("v.") + lane;
      o << "      {\n";
      for (std::size_t k = 0; k < f->insts.size(); ++k) {
        std::vector<std::string> a;
        for (int x : f->insts[k].args) a.push_back(x < 0 ? arg(x, suf) : "t" + std::to_string(x));
        o << "        const float t" << k << " = " << scalar_expr(f->insts[k], a) << ";\n";
      }
      for (std::size_t k = 0; k < f->outputs.size(); ++k) {
        o << "        out[" << k << "][i + " << (lane[0] == 'x' ? 0 : lane[0] == 'y' ? 1 : lane[0] == 'z' ? 2 : 3) << "] = t"
          << f->outputs[k] << ";\n";
      }
      o << "      }\n";
    }
    o << "    }\n    return;\n  }\n";
    o << "  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {\n";
    o << "    const long long c = i % cols;\n";
    for (std::size_t e = 0; e < nin; ++e) {
      o << "    const float x" << e << " = in[" << e << "][" << (f->input_is_row[e] ? "c" : "i") << "];\n";
    }
    for (std::size_t k = 0; k < f->insts.size(); ++k) {
      std::vector<std::string> a;
      for (int x : f->insts[k].args) a.push_back(arg(x, ""));
      o << "    const float t" << k << " = " << scalar_expr(f->insts[k], a) << ";\n";
    }
    for (std::size_t k = 0; k < f->outputs.size(); ++k) o << "    out[" << k << "][i] = t" << f->outputs[k] << ";\n";
    o << "  }\n}\n";
  }
  return o.str();
}

// ==========================================================================================
// CPU JIT
// ==========================================================================================

namespace {

std::string env(const char* key) {
#ifdef _MSC_VER
  char* v = nullptr;
  std::size_t n = 0;
  if (_dupenv_s(&v, &n, key) == 0 && v != nullptr) {
    std::string s = v;
    std::free(v);
    return s;
  }
  return {};
#else
  const char* v = std::getenv(key);
  return v ? v : "";
#endif
}

}  // namespace

std::string default_cache_dir() {
  const std::string e = env("NEXUS_COMPILE_CACHE");
  if (!e.empty()) return e;
  const std::string base = env("LOCALAPPDATA");
  return (fs::path(base.empty() ? "." : base) / "nexus" / "compile_cache").string();
}

namespace {

std::string find_vcvars() {
  const std::string e = env("NEXUS_VCVARS");
  if (!e.empty()) return e;
#ifdef _WIN32
  const char* vswhere = "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe";
  if (fs::exists(vswhere)) {
    std::string cmd = std::string("\"\"") + vswhere + "\" -latest -prerelease -products * -property installationPath\"";
    if (FILE* p = _popen(cmd.c_str(), "r")) {
      char buf[1024];
      std::string out;
      while (std::fgets(buf, sizeof(buf), p)) out += buf;
      _pclose(p);
      while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
      const fs::path bat = fs::path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat";
      if (!out.empty() && fs::exists(bat)) return bat.string();
    }
  }
#endif
  throw std::runtime_error("Nexus CPU JIT: vcvars64.bat bulunamadi (NEXUS_VCVARS ayarlayin); yorumlayici yedegi yok");
}

}  // namespace

CpuJit::CpuJit(std::string cache_dir) : dir_(cache_dir.empty() ? default_cache_dir() : std::move(cache_dir)) {
  fs::create_directories(dir_);
}

CpuJit::~CpuJit() {
#ifdef _WIN32
  for (auto& kv : modules_) FreeLibrary(static_cast<HMODULE>(kv.second));
#endif
}

std::map<std::string, CpuKernelFn> CpuJit::load(const std::string& source, const std::vector<std::string>& names) {
  std::map<std::string, CpuKernelFn> out;
  if (names.empty()) return out;
#ifndef _WIN32
  throw std::runtime_error("Nexus CPU JIT: yalniz Windows (cl.exe) bu derlemede");
#else
  const std::string h = source_hash(source + "|cl /O2 /arch:AVX2 /LD v1");
  if (!dump_.empty()) {
    fs::create_directories(dump_);
    std::ofstream(fs::path(dump_) / ("cpu_" + h + ".cpp")) << source;
  }
  void* mod = nullptr;
  auto it = modules_.find(h);
  if (it != modules_.end()) {
    mod = it->second;
    stats_.cache_hits += 1;
  } else {
    const fs::path dll = fs::path(dir_) / (h + ".dll");
    if (!fs::exists(dll)) {
      const fs::path src = fs::path(dir_) / (h + ".cpp");
      const fs::path obj = fs::path(dir_) / (h + ".obj");
      const fs::path log = fs::path(dir_) / (h + ".log");
      const fs::path bat = fs::path(dir_) / (h + ".bat");
      std::ofstream(src) << source;
      // vcvars ortami bir kez yakalanir (vcvars_env.bat); sonraki derlemeler vcvars maliyetini odemez.
      const fs::path envbat = fs::path(dir_) / "vcvars_env.bat";
      if (!fs::exists(envbat)) {
        const fs::path cap = fs::path(dir_) / "vcvars_capture.bat";
        const fs::path txt = fs::path(dir_) / "vcvars_env.txt";
        std::ofstream(cap) << "@echo off\r\ncall \"" << find_vcvars() << "\" >nul 2>&1\r\nset > \"" << txt.string() << "\"\r\n";
        const auto tc = std::chrono::steady_clock::now();
        if (std::system(("\"" + cap.string() + "\"").c_str()) != 0 || !fs::exists(txt)) {
          throw std::runtime_error("Nexus CPU JIT: vcvars ortami yakalanamadi");
        }
        stats_.compile_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
        std::ifstream in(txt);
        std::ofstream out(envbat);
        std::string line;
        while (std::getline(in, line)) {
          if (!line.empty() && line.back() == '\r') line.pop_back();
          if (line.find('=') != std::string::npos && line.find('"') == std::string::npos) out << "set \"" << line << "\"\r\n";
        }
      }
      {
        std::ofstream b(bat);
        b << "@echo off\r\ncall \"" << envbat.string() << "\"\r\n"
          << "cl /nologo /O2 /arch:AVX2 /fp:precise /LD \"" << src.string() << "\" /Fe:\"" << dll.string() << "\" /Fo:\""
          << obj.string() << "\" > \"" << log.string() << "\" 2>&1\r\n";
      }
      const auto t0 = std::chrono::steady_clock::now();
      const int rc = std::system(("\"" + bat.string() + "\"").c_str());
      stats_.compile_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      stats_.compiler_invocations += 1;
      if (rc != 0 || !fs::exists(dll)) {
        std::ifstream l(log);
        std::stringstream ss;
        ss << l.rdbuf();
        throw std::runtime_error("Nexus CPU JIT: cl derlemesi basarisiz (" + std::to_string(rc) + "):\n" + ss.str());
      }
    } else {
      stats_.cache_hits += 1;
    }
    mod = LoadLibraryA(dll.string().c_str());
    if (mod == nullptr) throw std::runtime_error("Nexus CPU JIT: DLL yuklenemedi: " + dll.string());
    modules_[h] = mod;
  }
  for (const std::string& n : names) {
    auto fn = reinterpret_cast<CpuKernelFn>(GetProcAddress(static_cast<HMODULE>(mod), n.c_str()));
    if (fn == nullptr) throw std::runtime_error("Nexus CPU JIT: sembol yok: " + n);
    out[n] = fn;
  }
  return out;
#endif
}

}  // namespace ir
}  // namespace nexus
