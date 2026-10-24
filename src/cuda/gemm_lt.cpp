/**
 * @file gemm_lt.cpp
 * @brief FAZ 3.1 cuBLASLt GEMM: plan onbellegi, epilog fuzyonu, TF32 politikasi.
 *
 * Satir-ana C[m,n] = op(A) op(B) cagrisi sutun-ana cuBLASLt'ye kopyasiz eslenir:
 * C^T[n,m] = op(B)^T op(A)^T. Yani cuBLASLt'nin A'si bizim B'miz, B'si bizim A'miz;
 * M_lt = n, N_lt = m, K = k. Bias uzunlugu M_lt = n (cikis ozellik sayisi).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gemm.hpp"

#include "nexus/core/error.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/runtime.hpp"

#include <cublasLt.h>

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace nexus {
namespace cuda {
namespace {

void check_lt(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw TensorError(std::string("Nexus cuBLASLt: ") + what + " basarisiz (durum " +
                      std::to_string(static_cast<int>(status)) + ")");
  }
}

#define NEXUS_LT_CHECK(expr) check_lt((expr), #expr)

constexpr std::size_t kWorkspaceBytes = 32u << 20;
constexpr int kMaxDevices = 16;

std::atomic<int> g_precision{static_cast<int>(MatmulPrecision::Highest)};
std::atomic<bool> g_autotune{false};

struct PlanKey {
  int device, m, n, k, lda, ldb, ldc, batch;
  long long sa, sb, sc;
  bool ta, tb;
  int epilogue, precision;
  bool has_aux;
  int ldaux;
  bool beta_zero;
  auto tie() const {
    return std::tie(device, m, n, k, lda, ldb, ldc, batch, sa, sb, sc, ta, tb, epilogue, precision, has_aux, ldaux,
                    beta_zero);
  }
  bool operator<(const PlanKey& other) const { return tie() < other.tie(); }
};

struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr;  // lt-A = bizim B
  cublasLtMatrixLayout_t b = nullptr;  // lt-B = bizim A
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

struct DeviceContext {
  cublasLtHandle_t handle = nullptr;
  DeviceBuffer workspace;
};

struct State {
  std::mutex mutex;
  std::array<DeviceContext, kMaxDevices> devices;
  std::map<PlanKey, Plan> plans;
  GemmCacheStats stats;
};

State& state() {
  static auto* s = new State();  // CUDA baglamindan sonra yikilmasin diye sizdirilir
  return *s;
}

cublasLtEpilogue_t to_lt(Epilogue epilogue, bool has_aux) {
  switch (epilogue) {
    case Epilogue::None:
      return CUBLASLT_EPILOGUE_DEFAULT;
    case Epilogue::Bias:
      return CUBLASLT_EPILOGUE_BIAS;
    case Epilogue::Relu:
      return CUBLASLT_EPILOGUE_RELU;
    case Epilogue::BiasRelu:
      return CUBLASLT_EPILOGUE_RELU_BIAS;
    case Epilogue::Gelu:
      return has_aux ? CUBLASLT_EPILOGUE_GELU_AUX : CUBLASLT_EPILOGUE_GELU;
    case Epilogue::BiasGelu:
      return has_aux ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_BIAS;
    case Epilogue::DGelu:
      return CUBLASLT_EPILOGUE_DGELU;
    case Epilogue::BGradA:
      return CUBLASLT_EPILOGUE_BGRADB;  // bizim A = lt-B
    case Epilogue::BGradB:
      return CUBLASLT_EPILOGUE_BGRADA;  // bizim B = lt-A
  }
  return CUBLASLT_EPILOGUE_DEFAULT;
}

cublasLtMatrixLayout_t make_layout(int rows, int cols, int ld, int batch, long long stride) {
  cublasLtMatrixLayout_t layout = nullptr;
  NEXUS_LT_CHECK(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, static_cast<std::uint64_t>(rows),
                                            static_cast<std::uint64_t>(cols), ld));
  if (batch > 1) {
    NEXUS_LT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    NEXUS_LT_CHECK(
        cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
  }
  return layout;
}

void validate(const GemmArgs& g) {
  if (g.m <= 0 || g.n <= 0 || g.k <= 0 || g.a == nullptr || g.b == nullptr || g.c == nullptr) {
    throw TensorError("Nexus gemm: m, n, k pozitif ve A/B/C bos olmamali");
  }
  const int a_cols = g.trans_a ? g.m : g.k;
  const int b_cols = g.trans_b ? g.k : g.n;
  if (g.lda < a_cols || g.ldb < b_cols || g.ldc < g.n) {
    throw TensorError("Nexus gemm: satir adimi (ld) satir genisliginden kucuk");
  }
  const bool needs_bias = g.epilogue == Epilogue::Bias || g.epilogue == Epilogue::BiasRelu ||
                          g.epilogue == Epilogue::BiasGelu;
  if (needs_bias && g.bias == nullptr) {
    throw TensorError(std::string("Nexus gemm: ") + to_string(g.epilogue) + " bias isaretcisi ister");
  }
  if ((g.epilogue == Epilogue::BGradA || g.epilogue == Epilogue::BGradB) && g.bias_grad == nullptr) {
    throw TensorError("Nexus gemm: BGrad epilogu bias_grad isaretcisi ister");
  }
  if (g.epilogue == Epilogue::DGelu && g.aux == nullptr) {
    throw TensorError("Nexus gemm: DGelu aux (on-aktivasyon) ister");
  }
  if (g.batch > 1 && g.epilogue != Epilogue::None) {
    throw TensorError("Nexus gemm: batched GEMM'de epilog desteklenmez");
  }
}

Plan build_plan(const PlanKey& key, const GemmArgs& g, DeviceContext& ctx, cudaStream_t stream) {
  Plan plan;
  const auto precision = static_cast<MatmulPrecision>(key.precision);
  const cublasComputeType_t compute =
      precision == MatmulPrecision::Highest ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
  NEXUS_LT_CHECK(cublasLtMatmulDescCreate(&plan.desc, compute, CUDA_R_32F));
  const cublasOperation_t op_lt_a = g.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_lt_b = g.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  NEXUS_LT_CHECK(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_lt_a, sizeof(op_lt_a)));
  NEXUS_LT_CHECK(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_lt_b, sizeof(op_lt_b)));
  const cublasLtEpilogue_t epi = to_lt(g.epilogue, g.aux != nullptr);
  NEXUS_LT_CHECK(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  if (g.aux != nullptr) {
    const std::int64_t ld_aux = g.ldc;
    NEXUS_LT_CHECK(
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld_aux, sizeof(ld_aux)));
  }
  const int M = g.n;
  const int N = g.m;
  const int K = g.k;
  plan.a = g.trans_b ? make_layout(K, M, g.ldb, g.batch, g.stride_b) : make_layout(M, K, g.ldb, g.batch, g.stride_b);
  plan.b = g.trans_a ? make_layout(N, K, g.lda, g.batch, g.stride_a) : make_layout(K, N, g.lda, g.batch, g.stride_a);
  plan.c = make_layout(M, N, g.ldc, g.batch, g.stride_c);

  // Gecici: isaretci ozellikleri sezgiye dahil (hizalama), bu yuzden once kurulur.
  if (g.bias != nullptr) {
    NEXUS_LT_CHECK(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &g.bias, sizeof(g.bias)));
  }
  if (g.bias_grad != nullptr) {
    NEXUS_LT_CHECK(
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &g.bias_grad, sizeof(g.bias_grad)));
  }
  if (g.aux != nullptr) {
    NEXUS_LT_CHECK(
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &g.aux, sizeof(g.aux)));
  }

  cublasLtMatmulPreference_t pref = nullptr;
  NEXUS_LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  const std::uint64_t ws = kWorkspaceBytes;
  NEXUS_LT_CHECK(
      cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  constexpr int kCandidates = 8;
  std::array<cublasLtMatmulHeuristicResult_t, kCandidates> results{};
  int returned = 0;
  const cublasStatus_t heuristic = cublasLtMatmulAlgoGetHeuristic(ctx.handle, plan.desc, plan.a, plan.b, plan.c, plan.c,
                                                                  pref, kCandidates, results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  if (heuristic != CUBLAS_STATUS_SUCCESS || returned == 0) {
    throw TensorError(std::string("Nexus gemm: cuBLASLt bu sekil/epilog icin algoritma bulamadi (epilog ") +
                      to_string(g.epilogue) + ", m=" + std::to_string(g.m) + " n=" + std::to_string(g.n) +
                      " k=" + std::to_string(g.k) + ")");
  }
  plan.algo = results[0].algo;

  if (g_autotune.load() && key.beta_zero && returned > 1) {
    // Her aday 3 kez kosulur; en kisa medyanli olan secilir. beta=0 oldugundan C uzerine
    // yazmak zararsizdir (gercek cagri hemen ardindan tekrar yazar).
    float best = 1e30f;
    Event start;
    Event stop;
    const Stream s(stream, key.device);
    for (int i = 0; i < returned; ++i) {
      float total = 0.0f;
      bool ok = true;
      for (int r = 0; r < 3 && ok; ++r) {
        start.record(s);
        ok = cublasLtMatmul(ctx.handle, plan.desc, &g.alpha, g.b, plan.a, g.a, plan.b, &g.beta, g.c, plan.c, g.c,
                            plan.c, &results[static_cast<std::size_t>(i)].algo, ctx.workspace.data(), kWorkspaceBytes,
                            stream) == CUBLAS_STATUS_SUCCESS;
        stop.record(s);
        stop.synchronize();
        total += start.elapsed_ms(stop);
      }
      if (ok && total < best) {
        best = total;
        plan.algo = results[static_cast<std::size_t>(i)].algo;
      }
    }
    state().stats.autotuned += 1;
  }
  return plan;
}

}  // namespace

const char* to_string(Epilogue epilogue) noexcept {
  switch (epilogue) {
    case Epilogue::None:
      return "none";
    case Epilogue::Bias:
      return "bias";
    case Epilogue::Relu:
      return "relu";
    case Epilogue::BiasRelu:
      return "bias_relu";
    case Epilogue::Gelu:
      return "gelu";
    case Epilogue::BiasGelu:
      return "bias_gelu";
    case Epilogue::DGelu:
      return "dgelu";
    case Epilogue::BGradA:
      return "bgrad_a";
    case Epilogue::BGradB:
      return "bgrad_b";
  }
  return "unknown";
}

void gemm(const GemmArgs& g, cudaStream_t stream) {
  validate(g);
  State& st = state();
  std::lock_guard<std::mutex> lock(st.mutex);
  const int device = current_device();
  if (device >= kMaxDevices) {
    throw TensorError("Nexus gemm: aygit indeksi sinir disinda");
  }
  DeviceContext& ctx = st.devices[static_cast<std::size_t>(device)];
  if (ctx.handle == nullptr) {
    NEXUS_LT_CHECK(cublasLtCreate(&ctx.handle));
    ctx.workspace = DeviceBuffer(kWorkspaceBytes, Stream(nullptr, device));
  }
  const PlanKey key{device, g.m, g.n, g.k, g.lda, g.ldb, g.ldc, g.batch, g.stride_a, g.stride_b, g.stride_c,
                    g.trans_a, g.trans_b, static_cast<int>(g.epilogue), g_precision.load(), g.aux != nullptr,
                    g.ldc, g.beta == 0.0f};
  auto it = st.plans.find(key);
  if (it == st.plans.end()) {
    st.stats.misses += 1;
    it = st.plans.emplace(key, build_plan(key, g, ctx, stream)).first;
    st.stats.plans = st.plans.size();
  } else {
    st.stats.hits += 1;
  }
  Plan& plan = it->second;
  // Isaretci ozellikleri her cagrida degisebilir.
  if (g.bias != nullptr) {
    NEXUS_LT_CHECK(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &g.bias, sizeof(g.bias)));
  }
  if (g.bias_grad != nullptr) {
    NEXUS_LT_CHECK(
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &g.bias_grad, sizeof(g.bias_grad)));
  }
  if (g.aux != nullptr) {
    NEXUS_LT_CHECK(
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &g.aux, sizeof(g.aux)));
  }
  // Calisma alani aygit basina tektir ve serbest birakilmaz. Iki farkli akistan es zamanli
  // GEMM ayni calisma alanini yarisa sokabilir; bu surumde Nexus GPU yolu tek hesap akisi
  // kullanir (cok akisli calisma alani havuzu FAZ 10 kapsaminda).
  NEXUS_LT_CHECK(cublasLtMatmul(ctx.handle, plan.desc, &g.alpha, g.b, plan.a, g.a, plan.b, &g.beta, g.c, plan.c, g.c,
                                plan.c, &plan.algo, ctx.workspace.data(), kWorkspaceBytes, stream));
}

void set_float32_matmul_precision(MatmulPrecision precision) noexcept { g_precision.store(static_cast<int>(precision)); }

MatmulPrecision float32_matmul_precision() noexcept { return static_cast<MatmulPrecision>(g_precision.load()); }

void set_float32_matmul_precision(const std::string& name) {
  if (name == "highest") {
    set_float32_matmul_precision(MatmulPrecision::Highest);
  } else if (name == "high") {
    set_float32_matmul_precision(MatmulPrecision::High);
  } else if (name == "medium") {
    set_float32_matmul_precision(MatmulPrecision::Medium);
  } else {
    throw TensorError("Nexus: matmul hassasiyeti 'highest', 'high' veya 'medium' olmali: " + name);
  }
}

void set_gemm_autotune(bool enabled) noexcept { g_autotune.store(enabled); }

GemmCacheStats gemm_cache_stats() noexcept {
  State& st = state();
  std::lock_guard<std::mutex> lock(st.mutex);
  return st.stats;
}

void clear_gemm_cache() {
  State& st = state();
  std::lock_guard<std::mutex> lock(st.mutex);
  for (auto& [key, plan] : st.plans) {
    cublasLtMatrixLayoutDestroy(plan.a);
    cublasLtMatrixLayoutDestroy(plan.b);
    cublasLtMatrixLayoutDestroy(plan.c);
    cublasLtMatmulDescDestroy(plan.desc);
    (void)key;
  }
  st.plans.clear();
  st.stats = {};
}

}  // namespace cuda
}  // namespace nexus
