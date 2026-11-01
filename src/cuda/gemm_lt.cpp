/**
 * @file gemm_lt.cpp
 * @brief FAZ 3.1 cuBLASLt GEMM: plan onbellegi, epilog fuzyonu, TF32 politikasi.
 *
 * Satir-ana C[m,n] = op(A) op(B) cagrisi sutun-ana cuBLASLt'ye kopyasiz eslenir:
 * C^T[n,m] = op(B)^T op(A)^T. Yani cuBLASLt'nin A'si bizim B'miz, B'si bizim A'miz;
 * M_lt = n, N_lt = m, K = k. Bias uzunlugu M_lt = n (cikis ozellik sayisi).
 *
 * Dalga 3.1 eszamanlilik: plan onbellegi std::shared_mutex ile okunur (paylasimli kilit: okuyucular
 * birbirini beklemez); yalniz yeni plan eklemek ozel kilit alir. Isaretci ozelligi (bias / aux) olan
 * planlarda tanimlayici cagri basina yazildigindan "ozellik yaz + baslat" o plana ait mutex altinda.
 * Calisma alani (aygit, akis) basina: ayni akistaki cagrilar sirali, farkli akislar ayri tampon.
 * Autotune: sezgiden en cok 16 aday, her biri 3 x olculur; secim istege bagli diske yazilir
 * (set_gemm_tune_cache), dosya cuBLASLt surumu + aygit adi + SM sayisiyla damgali; yuklenen algoritma
 * cublasLtMatmulAlgoCheck'ten gecmezse yeniden secilir (sayac: disk_rejected).
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

#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
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

// FAZ 4.4: FP32 matmul varsayilani "high" (TF32 Tensor Core); tam FP32 icin "highest".
std::atomic<int> g_precision{static_cast<int>(MatmulPrecision::High)};
std::atomic<bool> g_autotune{false};

cudaDataType_t to_cuda(DType t) {
  switch (t) {
    case DType::F32:
      return CUDA_R_32F;
    case DType::BF16:
      return CUDA_R_16BF;
    case DType::F16:
      return CUDA_R_16F;
    default:
      throw TensorError(std::string("Nexus gemm: desteklenmeyen veri tipi ") + std::string(nexus::to_string(t)));
  }
}

struct PlanKey {
  int device, m, n, k, lda, ldb, ldc, batch;
  int a_type, c_type, bias_type;
  long long sa, sb, sc;
  bool ta, tb;
  int epilogue, precision;
  bool has_aux;
  int ldaux;
  bool beta_zero;
  auto tie() const {
    return std::tie(device, m, n, k, lda, ldb, ldc, batch, a_type, c_type, bias_type, sa, sb, sc, ta, tb, epilogue,
                    precision, has_aux, ldaux, beta_zero);
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

struct Entry {
  Plan plan;
  bool pointer_attrs = false;  // bias / aux isaretcisi cagri basina tanimlayiciya yazilir
  std::mutex mutex;            // pointer_attrs ise "yaz + baslat" bolumunu korur
};

struct State {
  std::shared_mutex mutex;  // plans / handles / workspaces: okuma paylasimli, ekleme ozel
  std::array<cublasLtHandle_t, kMaxDevices> handles{};
  std::map<PlanKey, std::unique_ptr<Entry>> plans;
  std::map<std::pair<int, cudaStream_t>, DeviceBuffer> workspaces;
  std::atomic<std::uint64_t> hits{0}, misses{0}, autotuned{0}, disk_hits{0}, disk_rejected{0};
  std::string tune_path;                         // bos: disk onbellegi kapali
  std::map<PlanKey, cublasLtMatmulAlgo_t> disk;  // yuklenen + yeni olculen secimler
  bool disk_loaded = false;
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

cublasLtMatrixLayout_t make_layout(cudaDataType_t type, int rows, int cols, int ld, int batch, long long stride) {
  cublasLtMatrixLayout_t layout = nullptr;
  NEXUS_LT_CHECK(cublasLtMatrixLayoutCreate(&layout, type, static_cast<std::uint64_t>(rows),
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
  const bool narrow = g.a_type == DType::BF16 || g.a_type == DType::F16;
  if (g.a_type != g.b_type || !(g.a_type == DType::F32 || narrow)) {
    throw TensorError("Nexus gemm: A ve B ayni tipte olmali (F32, BF16 veya F16)");
  }
  if (!(g.c_type == DType::F32 || g.c_type == g.a_type)) {
    throw TensorError("Nexus gemm: C tipi F32 ya da A ile ayni dar tip olmali");
  }
}

// ---- disk onbellegi (autotune secimleri) ----
std::string tune_stamp(int device) {
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  std::ostringstream o;
  o << "nexus-gemm-tune v1 cublasLt=" << cublasLtGetVersion() << " sm=" << prop.major << prop.minor
    << " sms=" << prop.multiProcessorCount << " name=" << prop.name;
  return o.str();
}

constexpr int kKeyInts = 21;
std::array<long long, kKeyInts> key_ints(const PlanKey& k) {
  return {k.device, k.m,       k.n,          k.k,      k.lda,       k.ldb,     k.ldc,
          k.batch,  k.a_type,  k.c_type,     k.bias_type, k.sa,     k.sb,      k.sc,
          k.ta,     k.tb,      k.epilogue,   k.precision, k.has_aux, k.ldaux,  k.beta_zero};
}
PlanKey key_from(const std::array<long long, kKeyInts>& v) {
  PlanKey k{};
  k.device = int(v[0]);
  k.m = int(v[1]);
  k.n = int(v[2]);
  k.k = int(v[3]);
  k.lda = int(v[4]);
  k.ldb = int(v[5]);
  k.ldc = int(v[6]);
  k.batch = int(v[7]);
  k.a_type = int(v[8]);
  k.c_type = int(v[9]);
  k.bias_type = int(v[10]);
  k.sa = v[11];
  k.sb = v[12];
  k.sc = v[13];
  k.ta = v[14] != 0;
  k.tb = v[15] != 0;
  k.epilogue = int(v[16]);
  k.precision = int(v[17]);
  k.has_aux = v[18] != 0;
  k.ldaux = int(v[19]);
  k.beta_zero = v[20] != 0;
  return k;
}

// Ozel kilit altinda cagrilir. Damga uyusmazsa dosya yok sayilir (ilk kayitta ustune yazilir).
void load_disk(State& st, int device) {
  st.disk_loaded = true;
  std::ifstream in(st.tune_path);
  if (!in) return;
  std::string stamp;
  std::getline(in, stamp);
  if (stamp != tune_stamp(device)) return;
  std::array<long long, kKeyInts> v{};
  cublasLtMatmulAlgo_t algo{};
  while (true) {
    for (auto& x : v) in >> x;
    for (auto& w : algo.data) in >> w;
    if (!in) break;
    st.disk[key_from(v)] = algo;
  }
}

void save_disk(const State& st, int device) {
  std::ofstream out(st.tune_path, std::ios::trunc);
  if (!out) throw TensorError("Nexus gemm: autotune onbellek dosyasi yazilamadi: " + st.tune_path);
  out << tune_stamp(device) << '\n';
  for (const auto& [key, algo] : st.disk) {
    for (long long x : key_ints(key)) out << x << ' ';
    for (std::uint64_t w : algo.data) out << w << ' ';
    out << '\n';
  }
}

Plan build_plan(const PlanKey& key, const GemmArgs& g, cublasLtHandle_t handle, void* workspace,
                cudaStream_t stream) {
  Plan plan;
  const auto precision = static_cast<MatmulPrecision>(key.precision);
  // Dar girislerde (BF16/F16) FP32 biriktirici + Tensor Core; FP32 girislerde TF32 politikasi.
  const bool narrow = g.a_type != DType::F32;
  const cublasComputeType_t compute = (narrow || precision == MatmulPrecision::Highest) ? CUBLAS_COMPUTE_32F
                                                                                        : CUBLAS_COMPUTE_32F_FAST_TF32;
  NEXUS_LT_CHECK(cublasLtMatmulDescCreate(&plan.desc, compute, CUDA_R_32F));
  const bool has_bias_ptr = g.bias != nullptr || g.bias_grad != nullptr;
  if (has_bias_ptr) {
    const cudaDataType_t bias_type = to_cuda(g.c_type == DType::F32 ? DType::F32 : g.bias_type);
    NEXUS_LT_CHECK(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                                  sizeof(bias_type)));
  }
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
  const cudaDataType_t ab = to_cuda(g.a_type);
  const cudaDataType_t cd = to_cuda(g.c_type);
  plan.a = g.trans_b ? make_layout(ab, K, M, g.ldb, g.batch, g.stride_b) : make_layout(ab, M, K, g.ldb, g.batch, g.stride_b);
  plan.b = g.trans_a ? make_layout(ab, N, K, g.lda, g.batch, g.stride_a) : make_layout(ab, K, N, g.lda, g.batch, g.stride_a);
  plan.c = make_layout(cd, M, N, g.ldc, g.batch, g.stride_c);

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
  constexpr int kCandidates = 16;
  std::array<cublasLtMatmulHeuristicResult_t, kCandidates> results{};
  int returned = 0;
  const cublasStatus_t heuristic = cublasLtMatmulAlgoGetHeuristic(handle, plan.desc, plan.a, plan.b, plan.c, plan.c,
                                                                  pref, kCandidates, results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  if (heuristic != CUBLAS_STATUS_SUCCESS || returned == 0) {
    throw TensorError(std::string("Nexus gemm: cuBLASLt bu sekil/epilog icin algoritma bulamadi (epilog ") +
                      to_string(g.epilogue) + ", m=" + std::to_string(g.m) + " n=" + std::to_string(g.n) +
                      " k=" + std::to_string(g.k) + ")");
  }
  plan.algo = results[0].algo;

  State& st = state();
  if (!st.tune_path.empty()) {
    if (!st.disk_loaded) load_disk(st, key.device);
    const auto it = st.disk.find(key);
    if (it != st.disk.end()) {
      cublasLtMatmulHeuristicResult_t checked{};
      if (cublasLtMatmulAlgoCheck(handle, plan.desc, plan.a, plan.b, plan.c, plan.c, &it->second, &checked) ==
              CUBLAS_STATUS_SUCCESS &&
          checked.workspaceSize <= kWorkspaceBytes) {
        plan.algo = it->second;
        st.disk_hits += 1;
        return plan;
      }
      st.disk_rejected += 1;
    }
  }
  if (g_autotune.load() && key.beta_zero && returned > 1) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
    if (capture != cudaStreamCaptureStatusNone) {
      throw TensorError("Nexus gemm: autotune acikken CUDA graf yakalamasi sirasinda yeni GEMM sekli (m=" +
                        std::to_string(g.m) + " n=" + std::to_string(g.n) + " k=" + std::to_string(g.k) +
                        "); sekli yakalamadan once bir kez kosun");
    }
    // Adaylar 5 tur boyunca ic ice (tur basina her aday 8 ardisik cagri) olculur: GPU saat kaymasi
    // tum adaylara esit dagilir. Aday basina tur medyani; sezgi secimi (aday 0) ancak bir aday ondan
    // >= %2 hizliysa degistirilir (olcum gurultusuyle daha yavas algoritma secilmesin). beta=0
    // oldugundan C / aux / bias_grad uzerine yazmak zararsizdir (gercek cagri hemen ardindan yazar).
    constexpr int kRounds = 5, kCalls = 8;
    const Stream s(stream, key.device);
    Event start;
    Event stop;
    auto run = [&](int i) {
      return cublasLtMatmul(handle, plan.desc, &g.alpha, g.b, plan.a, g.a, plan.b, &g.beta, g.c, plan.c, g.c, plan.c,
                            &results[static_cast<std::size_t>(i)].algo, workspace, kWorkspaceBytes,
                            stream) == CUBLAS_STATUS_SUCCESS;
    };
    std::vector<bool> ok(static_cast<std::size_t>(returned), true);
    std::vector<std::vector<float>> times(static_cast<std::size_t>(returned));
    for (int i = 0; i < returned; ++i) ok[static_cast<std::size_t>(i)] = run(i) && run(i);  // isinma
    for (int r = 0; r < kRounds; ++r) {
      for (int i = 0; i < returned; ++i) {
        if (!ok[static_cast<std::size_t>(i)]) continue;
        start.record(s);
        for (int c = 0; c < kCalls && ok[static_cast<std::size_t>(i)]; ++c) ok[static_cast<std::size_t>(i)] = run(i);
        stop.record(s);
        stop.synchronize();
        times[static_cast<std::size_t>(i)].push_back(start.elapsed_ms(stop));
      }
    }
    auto median = [](std::vector<float> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    if (ok[0]) {
      const float base = median(times[0]);
      float best = base;
      int pick = 0;
      for (int i = 1; i < returned; ++i) {
        if (!ok[static_cast<std::size_t>(i)]) continue;
        const float t = median(times[static_cast<std::size_t>(i)]);
        if (t < best) {
          best = t;
          pick = i;
        }
      }
      if (pick != 0 && best <= 0.98f * base) plan.algo = results[static_cast<std::size_t>(pick)].algo;
    }
    st.autotuned += 1;
    if (!st.tune_path.empty()) {
      st.disk[key] = plan.algo;
      save_disk(st, key.device);
    }
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

namespace {

void set_pointers(const Plan& plan, const GemmArgs& g) {
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
}

void launch(const Plan& plan, const GemmArgs& g, cublasLtHandle_t handle, void* workspace, cudaStream_t stream) {
  NEXUS_LT_CHECK(cublasLtMatmul(handle, plan.desc, &g.alpha, g.b, plan.a, g.a, plan.b, &g.beta, g.c, plan.c, g.c,
                                plan.c, &plan.algo, workspace, kWorkspaceBytes, stream));
}

}  // namespace

void gemm(const GemmArgs& g, cudaStream_t stream) {
  validate(g);
  State& st = state();
  const int device = current_device();
  if (device >= kMaxDevices) {
    throw TensorError("Nexus gemm: aygit indeksi sinir disinda");
  }
  const PlanKey key{device, g.m, g.n, g.k, g.lda, g.ldb, g.ldc, g.batch, static_cast<int>(g.a_type),
                    static_cast<int>(g.c_type), static_cast<int>(g.bias_type), g.stride_a, g.stride_b, g.stride_c,
                    g.trans_a, g.trans_b, static_cast<int>(g.epilogue), g_precision.load(), g.aux != nullptr,
                    g.ldc, g.beta == 0.0f};
  const std::pair<int, cudaStream_t> ws_key{device, stream};
  {
    // Sicak yol: paylasimli kilit (okuyucular birbirini beklemez); clear_gemm_cache ozel kilitle bekler.
    std::shared_lock<std::shared_mutex> read(st.mutex);
    const auto it = st.plans.find(key);
    const auto ws = st.workspaces.find(ws_key);
    if (it != st.plans.end() && ws != st.workspaces.end()) {
      st.hits += 1;
      Entry& e = *it->second;
      const cublasLtHandle_t handle = st.handles[static_cast<std::size_t>(device)];
      if (e.pointer_attrs) {
        std::lock_guard<std::mutex> lock(e.mutex);
        set_pointers(e.plan, g);
        launch(e.plan, g, handle, ws->second.data(), stream);
      } else {
        launch(e.plan, g, handle, ws->second.data(), stream);
      }
      return;
    }
  }
  std::unique_lock<std::shared_mutex> write(st.mutex);
  cublasLtHandle_t& handle = st.handles[static_cast<std::size_t>(device)];
  if (handle == nullptr) NEXUS_LT_CHECK(cublasLtCreate(&handle));
  auto ws = st.workspaces.find(ws_key);
  if (ws == st.workspaces.end()) {
    // Akisa bagli ayirma: tampon yalniz bu akista kullanilir, sira akis sirasiyla korunur.
    ws = st.workspaces.emplace(ws_key, DeviceBuffer(kWorkspaceBytes, Stream(stream, device))).first;
  }
  auto it = st.plans.find(key);
  if (it == st.plans.end()) {
    st.misses += 1;
    auto e = std::make_unique<Entry>();
    e->plan = build_plan(key, g, handle, ws->second.data(), stream);
    e->pointer_attrs = g.bias != nullptr || g.bias_grad != nullptr || g.aux != nullptr;
    it = st.plans.emplace(key, std::move(e)).first;
  } else {
    st.hits += 1;
  }
  Entry& e = *it->second;
  set_pointers(e.plan, g);  // ozel kilit altinda: plan mutex'i gerekmez
  launch(e.plan, g, handle, ws->second.data(), stream);
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

void set_gemm_tune_cache(const std::string& path) {
  State& st = state();
  std::unique_lock<std::shared_mutex> lock(st.mutex);
  st.tune_path = path;
  st.disk.clear();
  st.disk_loaded = false;
}

GemmCacheStats gemm_cache_stats() noexcept {
  State& st = state();
  std::shared_lock<std::shared_mutex> lock(st.mutex);
  GemmCacheStats out;
  out.plans = st.plans.size();
  out.hits = st.hits.load();
  out.misses = st.misses.load();
  out.autotuned = st.autotuned.load();
  out.disk_hits = st.disk_hits.load();
  out.disk_rejected = st.disk_rejected.load();
  out.workspaces = st.workspaces.size();
  return out;
}

void clear_gemm_cache() {
  State& st = state();
  std::unique_lock<std::shared_mutex> lock(st.mutex);
  for (auto& [key, e] : st.plans) {
    cublasLtMatrixLayoutDestroy(e->plan.a);
    cublasLtMatrixLayoutDestroy(e->plan.b);
    cublasLtMatrixLayoutDestroy(e->plan.c);
    cublasLtMatmulDescDestroy(e->plan.desc);
    (void)key;
  }
  st.plans.clear();
  st.hits = 0;
  st.misses = 0;
  st.autotuned = 0;
  st.disk_hits = 0;
  st.disk_rejected = 0;
}

}  // namespace cuda
}  // namespace nexus
