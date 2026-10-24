/** FAZ 4.5: sm_89+ cuBLASLt FP8 matmul (bu sm_86 makinede derleme kapsamli). */
#include "nexus/cuda/fp8.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/runtime.hpp"
#include <cublasLt.h>
#include <array>
#include <cmath>
#include <mutex>
#include <string>

namespace nexus::cuda {
namespace {
void check_lt(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw TensorError(std::string("Nexus FP8 cuBLASLt: ") + what + " basarisiz (durum " +
                      std::to_string(static_cast<int>(status)) + ")");
  }
}
struct LtDescriptors {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  ~LtDescriptors() {
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};
std::mutex& handle_mutex() { static std::mutex m; return m; }
std::array<cublasLtHandle_t, 16>& handles() {
  static auto* h = new std::array<cublasLtHandle_t, 16>{};
  return *h;
}
}  // namespace

void fp8_gemm_native(float* c, const std::uint8_t* a, const std::uint8_t* b, int m, int n, int k,
                     float scale_a, float scale_b, Fp8Format format, cudaStream_t stream) {
  require_fp8_native(current_device());
  if (c == nullptr || a == nullptr || b == nullptr || m <= 0 || n <= 0 || k <= 0 || !(scale_a > 0.0f) ||
      !(scale_b > 0.0f) || !std::isfinite(scale_a) || !std::isfinite(scale_b)) {
    throw TensorError("Nexus FP8 native GEMM: gecersiz isaretci, sekil veya olcek");
  }
  const cudaDataType_t fp8 = format == Fp8Format::E4M3 ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
  LtDescriptors d;
  check_lt(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
  const cublasOperation_t no = CUBLAS_OP_N;
  check_lt(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &no, sizeof(no)), "TRANSA");
  check_lt(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &no, sizeof(no)), "TRANSB");
  // Satir-ana C=A*B, sutun-ana cuBLASLt'de C^T=B^T*A^T olarak cagrilir.
  check_lt(cublasLtMatrixLayoutCreate(&d.a, fp8, n, k, n), "A layout");
  check_lt(cublasLtMatrixLayoutCreate(&d.b, fp8, k, m, k), "B layout");
  check_lt(cublasLtMatrixLayoutCreate(&d.c, CUDA_R_32F, n, m, n), "C layout");
  const float alpha = 1.0f / (scale_a * scale_b), beta = 0.0f;
  const int device = current_device();
  if (device < 0 || device >= static_cast<int>(handles().size())) throw TensorError("Nexus FP8: aygit indeksi sinir disi");
  DeviceBuffer workspace(4u << 20, Stream(stream, device));
  std::lock_guard<std::mutex> lock(handle_mutex());
  cublasLtHandle_t& handle = handles()[static_cast<std::size_t>(device)];
  if (handle == nullptr) check_lt(cublasLtCreate(&handle), "cublasLtCreate");
  check_lt(cublasLtMatmul(handle, d.op, &alpha, b, d.a, a, d.b, &beta, c, d.c, c, d.c, nullptr,
                          workspace.data(), workspace.bytes(), stream), "FP8 matmul");
}
}  // namespace nexus::cuda
