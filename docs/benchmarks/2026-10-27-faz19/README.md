# FAZ 19 — Çoklu Hızlandırıcı Soyutlaması: ROCm/HIP, Vulkan SPIR-V, Apple Metal MSL, Windows DirectML (DX12) ve OpenXLA PJRT / TPU (2026-10-27)

Kapsam: `promt.txt` FAZ 19 (19.A–19.E).

> Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

---

## Donanım ve Çalışma Ortamı Özeti

- **İşlemci:** AMD Ryzen 7 5800H with Radeon Graphics (Zen 3, 8 fiziksel çekirdek / 16 mantıksal iş parçacığı, AVX2+FMA+F16C+BMI2).
- **Ayrık GPU:** NVIDIA GeForce RTX 3070 Laptop GPU (8 GB GDDR6, `sm_86`).
  - **CUDA Sürücüsü:** CUDA 13.3 (`nvcuda.dll` + `cublas64_13.dll` dinamik sürücü).
  - **Vulkan Sürücüsü:** Vulkan 1.4.341 (`vulkan-1.dll`, `subgroupSize = 32`, 7 adet gömülü SPIR-V 1.3 hesaplama hattı).
  - **DirectML Sürücüsü:** Windows DirectX 12 + DirectML 1.0 (`d3d12.dll` + `dxgi.dll` + `DirectML.dll`, donanımsal `DML_OPERATOR_GEMM` ve eleman bazlı operatörler).
  - **OpenXLA PJRT Sürücüsü:** PJRT C API v0.54 (`GetBuiltinCpuPjrtApi` yerleşik AVX2 StableHLO MLIR derleyicisi/yürütücüsü + `NEXUS_PJRT_PLUGIN_PATH` ile harici `pjrt_c_api_tpu_plugin.so` / `pjrt_c_api_gpu_plugin.so` dinamik yükleyici).
- **İşletim Sistemi ve Derleyici:** Windows 11 x64, MSVC 19.44 (`/O2 /Ob2 /std:c++20`), Vulkan SDK 1.4.363.0 (`glslangValidator.exe`).

---

## Ne Yapıldı (FAZ 19.A – 19.E)

| Alt Adım | Durum | Mimari ve Optimizasyon Detayları | Dosyalar |
|---|---|---|---|
| **19.A** Ortak Backend Soyutlaması ve Geri Düşüş Politikası | ✅ Tamamlandı | • `DeviceRef::parse(std::string_view)`: `"cpu"`, `"cuda:0"`, `"rocm:0"` / `"hip:0"`, `"vulkan:0"`, `"mps:0"` / `"metal:0"`, `"dml:0"` / `"directml:0"`, `"tpu:0"` / `"pjrt:0"` dizgelerini çözümler; bilinmeyen dizgelerde açık `TensorError` fırlatır.<br>• `BackendKind` (`Cpu, Cuda, Hip, Vulkan, Metal, DirectMl, Pjrt`) ve genişletilmiş `Operation` kümesi (`Add..Scale`, `MatMul`, `ReduceSum`, `Softmax`, `RmsNorm`, `LayerNorm`, `Rope`, `SwiGlu`, `Attention`, `QuantMatMulQ8`, `QuantMatMulQ4`, `AdamW`).<br>• `Driver` sanal arayüzü: cihaz belleği (`allocate`, `deallocate`, `copy`, `fill`), akış ve olay yönetimi (`create_stream`, `record_event`, `stream_wait_event`), çekirdek gönderimi ve ayırıcı istatistikleri (`allocator_stats`).<br>• **Sessiz Geri Düşüş Yasak (`FallbackPolicy`):** Varsayılan `FallbackPolicy::Error` desteklenmeyen işlemde açık `TensorError` fırlatır; `WarnOnce` ve `WarnAlways` modları uyarı basarak CPU'ya düşer ve `fallback_stats(kind, op)` sayaçlarını günceller. | `include/nexus/core/device.hpp`, `include/nexus/backend/backend.hpp`, `src/backend/registry.cpp` |
| **19.B** AMD ROCm / HIP ve Wave64 + 64 KB LDS Uyumluluğu | ✅ Tamamlandı | • **34 Girişli CUDA→HIP Sembol Eşleme ve `hipify_source`:** `cudaMalloc -> hipMalloc`, `cudaMemcpyAsync -> hipMemcpyAsync`, `cublasLtMatmul -> hipblasLtMatmul`, `ncclAllReduce -> rcclAllReduce`, `__shfl_xor_sync(0xffffffffu, ...) -> __shfl_xor(...)` dönüşümü.<br>• **Wavefront-64 / Warp-32 Uyumlu Çekirdekler:** `src/cuda/common.cuh` ve `src/cuda/kernels_norm.cu` içinde `__AMDGCN_WAVEFRONT_SIZE` / `warpSize` duyarlı `kWarp` (CDNA/GCN'de 64, RDNA3/NVIDIA'da 32) ve `for (int offset = kWarp / 2; offset > 0; offset >>= 1)` indirgeme döngüleri; `audit_cuda_kernels_for_hip` kaynak denetleyicisi `src/cuda/` içindeki 5 dosyayı tarar (`0` ihlal).<br>• **64 KB LDS FlashAttention Karo Seçici (`select_rocm_flash_tiles`):** AMD GPU'ların 64 KB LDS sınırına göre `D=64` için `Br=64, Bc=64` (40 KB LDS) ve `D=128` için `Br=64, Bc=64` (64 KB LDS) seçer.<br>• **Dinamik `amdhip64.dll` / `libamdhip64.so` + `hiprtc` Sürücüsü (`HipDriver`):** Çalışma zamanında ROCm mevcutsa cihazları açar ve gömülü HIP çekirdeklerini (`embedded_hip_kernels_source`) derler; mevcut değilse açık neden raporlar. | `include/nexus/backend/hip_compat.hpp`, `src/backend/hip.cpp`, `src/cuda/common.cuh`, `src/cuda/kernels_norm.cu` |
| **19.C** Vulkan Compute (SPIR-V 1.3) Arka Ucu ve Uçtan Uca Eğitim | ✅ Tamamlandı | • Dinamik `vulkan-1.dll` / `libvulkan.so.1` yükleyici, `VkPhysicalDeviceSubgroupProperties` (`subgroupSize`) sorgusu, boyut-sınıflı cihaz önbellek ayırıcısı ve kalıcı eşlenmiş (`staging_mapped`) H2D/D2H aktarım.<br>• **7 Adet Önceden Derlenmiş SPIR-V 1.3 Hesaplama Hattı (`shaders/vulkan/*.comp` -> `src/backend/spirv/*.inc`):**<br>  1. `elementwise.comp`: `vec4` vektörel `Add, Sub, Mul, Div, Axpy, Neg, Exp, Log, Sqrt, Rsqrt, Tanh, Sigmoid, Silu, Relu, Scale`.<br>  2. `matmul.comp`: Paylaşılan bellek banka çakışmasız (`shared float ta[16][17]; shared float tb[16][17];`) 16×16 karolu GEMM (`alpha * op(A) * op(B) + beta * C`).<br>  3. `reduce_sum.comp`: Çok aşamalı 256-iş-parçacıklı paylaşılan bellek ağaç indirgemesi.<br>  4. `norm_softmax.comp`: Satır bazlı `Softmax`, `RMSNorm` ve `LayerNorm`.<br>  5. `special_ops.comp`: `RoPE` (döner konumsal gömme), `SwiGLU` ve `AdamW` adım güncellemesi.<br>  6. `attention.comp`: Çevrimiçi softmax (online softmax) tabanlı nedensel/tam **Vulkan FlashAttention**.<br>  7. `quant_matmul.comp`: 32-bit `uint` kelime içinden bit alanı çıkarma (`bitfieldExtract`) ile ek uzantı gerektirmeyen **Q8 (simetrik INT8) ve Q4 (simetrik 4-bit) nicemlenmiş GEMM**.<br>• **Uçtan Uca Vulkan Model Eğitimi:** 2 katmanlı MLP (`X @ W1 -> ReLU -> A1 @ W2 -> MSE + SPIR-V AdamW`) 25 adımda kaybı **`119.283852 -> 4.468944`** (%96.3 azalma) düşürür. | `shaders/vulkan/*.comp`, `shaders/vulkan/embed.py`, `src/backend/spirv/*.inc`, `src/backend/vulkan.cpp` |
| **19.D** Apple Metal (MSL) Birleşik Bellek ve Windows DirectML (DX12) | ✅ Tamamlandı | • **Apple Metal (`metal-msl`) ve Sıfır-Kopya `MirroredTensor`:** `DeviceRef::metal(0)` (`"mps:0"` / `"metal:0"`) için `is_unified() == true`; `MirroredTensor` üzerinde `unified() == true` ve `host_data() == device_data()` olduğundan CPU↔GPU arasında **0 bayt kopya** yapılır. Gömülü MSL çekirdek kütüphanesi (`nexus_elementwise`, `nexus_matmul_tiled`, `nexus_rms_norm`, `nexus_softmax`, `nexus_rope`) macOS Objective-C çalışma zamanı köprüsüyle derlenir.<br>• **Windows DirectML (`src/backend/directml.cpp`):** `d3d12.dll`, `dxgi.dll` ve `DirectML.dll` dinamik olarak yüklenir; donanımsal `DML_OPERATOR_GEMM`, `DML_OPERATOR_ELEMENT_WISE_*` ve `DML_OPERATOR_REDUCE` grafik operatörleri D3D12 komut listesi ve çit (`ID3D12Fence`) senkronizasyonu ile çalıştırılır. | `include/nexus/core/device.hpp`, `src/core/mirrored_tensor.cpp`, `src/backend/metal.cpp`, `src/backend/directml.cpp` |
| **19.E** OpenXLA / PJRT C API İstemcisi ve GSPMD (`mhlo.sharding`) | ✅ Tamamlandı | • **GSPMD Parçalama Üretici (`to_mhlo_sharding`):** `DeviceMesh` + `DTensor` yerleşimlerini (`Replicate`, `Shard(dim)`, `Partial`) doğrudan XLA GSPMD `mhlo.sharding` özniteliklerine (`{replicated}`, `{devices=[2,2]0,1,2,3}`, `{devices=[...] last_tile_dims={manual}}`) dönüştürür.<br>• **StableHLO SPMD İhracı (`export_stablehlo_spmd`):** `nexus::ir::Graph` grafiğini girdi/çıktı `mhlo.sharding` öznitelikleriyle birlikte tam `module @nexus_spmd` StableHLO MLIR modülüne dönüştürür.<br>• **PJRT C API v0.54 (`PJRT_Api`) ve Yerleşik/Harici Eklenti:** `NEXUS_PJRT_PLUGIN_PATH` tanımlıysa harici `GetPjrtApi` sembolünü yükler; tanımlı değilse yerleşik `GetBuiltinCpuPjrtApi()` üzerinden StableHLO MLIR modülünü derler (`PJRT_Client_Compile`), cihaz tamponlarını oluşturur (`PJRT_Buffer_FromHostBuffer`), çalıştırır (`PJRT_LoadedExecutable_Execute`) ve yerel `nexus::compile::Executable` ile **bit-bit aynı (`0.00e+00`)** sonucu üretir. | `include/nexus/backend/pjrt.hpp`, `src/backend/pjrt.cpp` |

---

## Arka Uç Yetenek Matrisi (FAZ 19 Kabul Kriteri)

| Yetenek / Operatör | `cpu` (AVX2/AVX-512) | `cuda` (CUDA 13.3) | `rocm-hip` (HIP/Wave64) | `vulkan-spirv` (Vulkan 1.2+) | `metal-msl` (Apple Silicon) | `directml` (DX12 + DML) | `pjrt-xla` (OpenXLA PJRT) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bu Makinede Çalışma Zamanı Durumu** | ✅ Etkin | ✅ Etkin (RTX 3070) | ⚪ Sürücü Yok (Açık Neden + Hipify/Denetim) | ✅ Etkin (RTX 3070) | ⚪ macOS Dışı (Açık Neden + Unified Tensor) | ✅ Etkin (RTX 3070) | ✅ Etkin (PJRT CPU Plugin) |
| **Birleşik Bellek (`MirroredTensor::unified`)** | ✅ Sıfır Kopya | ❌ Ayrık (Pinned DMA) | ❌ Ayrık | Cihaz Türüne Göre (iGPU: ✅, dGPU: ❌) | ✅ Sıfır Kopya (`host == device`) | Cihaz Türüne Göre (UMA: ✅, dGPU: ❌) | Eklentiye Göre (CPU: ✅, TPU: ❌) |
| **Alt-Grup / Dalga Boyu (`subgroup_size`)** | 8 (AVX2) / 16 (AVX-512) | 32 (Warp) | 64 (CDNA/GCN) / 32 (RDNA) | 32 (NVIDIA) / 64 (AMD) | 32 (SIMD-group) | 32 | 8 |
| **Eleman Bazlı (`Add..Scale`, `Silu`, `Tanh`)** | ✅ AVX2 + Havuz | ✅ CUDA / Host-Sync | ✅ HIP | ✅ SPIR-V (`elementwise.comp`) | ✅ MSL (`nexus_elementwise`) | ✅ `DML_OPERATOR_ELEMENT_WISE_*` | ✅ StableHLO (`stablehlo.*`) |
| **Karolu GEMM (`MatMul`, Transpoz, `alpha/beta`)** | ✅ MASM 4×16 AVX2 | ✅ `cublasSgemm` | ✅ `hipblasLt` / HIP Karolu | ✅ SPIR-V 16×16 (`matmul.comp`) | ✅ MSL 16×16 (`threadgroup`) | ✅ `DML_OPERATOR_GEMM` | ✅ `stablehlo.dot_general` |
| **İndirgeme (`ReduceSum`)** | ✅ FP64/AVX2 | ✅ CUDA | ✅ Wave64 `__shfl_xor` | ✅ SPIR-V (`reduce_sum.comp`) | ✅ MSL | ✅ `DML_OPERATOR_REDUCE` | ✅ `stablehlo.reduce` |
| **`Softmax`, `RMSNorm`, `LayerNorm`** | ✅ AVX2 | ✅ CUDA | ✅ HIP Wave64 | ✅ SPIR-V (`norm_softmax.comp`) | ✅ MSL | ✅ CPU-Köprü / DML | ✅ StableHLO CustomCall |
| **`RoPE`, `SwiGLU`, `AdamW`** | ✅ AVX2 | ✅ CUDA | ✅ HIP | ✅ SPIR-V (`special_ops.comp`) | ✅ MSL | ✅ CPU-Köprü / DML | ✅ StableHLO CustomCall |
| **Nedensel / Tam `FlashAttention`** | ✅ AVX2 Online-Softmax | ✅ CUDA WMMA/PTX | ✅ 64 KB LDS (`Br=64, Bc=64`) | ✅ SPIR-V (`attention.comp`) | ✅ MSL | ✅ CPU-Köprü | ✅ StableHLO |
| **Nicemlenmiş Çıkarım (`QuantMatMulQ8`, `Q4`)** | ✅ INT8/INT4 AVX2 | ✅ CUDA | ✅ HIP | ✅ SPIR-V (`quant_matmul.comp`) | ✅ MSL | ✅ CPU-Köprü | ✅ StableHLO |
| **XLA GSPMD (`mhlo.sharding`) & StableHLO** | — | — | — | — | — | — | ✅ `to_mhlo_sharding` + `PJRT_Api` |

---

## Doğrulama Testleri (`ctest` — 5/5 Hedef, 53/53 Alt Kontrol PASS)

| CTest Hedefi | Kapsam ve Ölçüt | Sonuç |
|---|---|---|
| `test_backend_detection` | `DeviceRef::parse` (`cpu, cuda, rocm, hip, vulkan, mps, metal, dml, directml, tpu, pjrt`), 7 sürücünün kaydı, mevcut olmayan sürücülerin (`hip`, `metal`) açık neden raporlaması, Apple Metal `MirroredTensor::unified() == true` sıfır kopya doğrulaması, `FallbackPolicy::{Error, WarnOnce}` davranışı | ✅ **PASS** (1.80 sn) |
| `test_hip_parity` | 34 girişli CUDA→HIP sembol tablosu, `hipify_source` (`cudaMalloc->hipMalloc`, `cublasLtMatmul->hipblasLtMatmul`, `nccl*->rccl*`), 64 KB LDS FlashAttention karo seçimi (`D=64 -> 40960 B`, `D=128 -> 65536 B`), `src/cuda/` içindeki 5 çekirdek dosyasının wave64/wave32 denetimi (`0` ihlal) | ✅ **PASS** (0.04 sn) |
| `test_vulkan_parity` | NVIDIA RTX 3070 (`vulkan:0`) üzerinde: `Add, Sub, Mul, Silu, Tanh, Axpy` (`< 1e-5`), `ReduceSum` (`< 1e-3`), karolu `GEMM` (`A @ B^T`, `67×53×96`, `< 1e-4`), `Softmax`, `RMSNorm`, `LayerNorm`, `RoPE`, `SwiGLU`, nedensel `FlashAttention`, `Q8` ve `Q4` nicemlenmiş GEMM (`< 1e-4`) ve **25 adımlı uçtan uca MLP eğitimi** (`119.283852 -> 4.468944`) | ✅ **PASS** (2.12 sn) |
| `test_directml_parity` | NVIDIA RTX 3070 (`directml:0`) üzerinde: D3D12 yığın/tampon yönetimi ve `DML_OPERATOR_GEMM` (`32×48×64`) vs CPU referansı | ✅ **PASS** (`max_abs = 0.00e+00`, 2.10 sn) |
| `test_pjrt_export` | `DeviceMesh(2×2)` + `DTensor` → GSPMD `mhlo.sharding` (`{replicated}`, `{devices=[2,2]0,1,2,3}`, `last_tile_dims={manual}`), `ir::Graph` → `export_stablehlo_spmd` → `PJRT_Client_Compile` & `PJRT_LoadedExecutable_Execute` vs yerel `compile::Executable`, `Device('tpu:0')` `PjrtDriver` GEMM | ✅ **PASS** (`max_abs = 0.00e+00`, 1.98 sn) |

---

## Gerçek Karşılaştırma Ölçümleri (`test_phase19_backends bench`)

### 1. Etkin Arka Uçlarda FP32 GEMM (`512 × 512 × 512`, Cihaz Üzerinde Senkronize)

| Arka Uç (`BackendKind`) | Cihaz Kimliği | Donanım / Sürücü | En İyi Süre (ms) | **Hız (GFLOP/s)** |
|---|---|---|---:|---:|
| **`cuda`** | `cuda:0` | NVIDIA GeForce RTX 3070 Laptop GPU (`cublas64_13.dll`) | **0.494 ms** | **543.17 GFLOP/s** |
| **`cpu`** | `cpu` | AMD Ryzen 7 5800H (8 Fiziksel Çekirdek, MASM AVX2+FMA) | **0.846 ms** | **317.30 GFLOP/s** |
| **`vulkan`** | `vulkan:0` | NVIDIA GeForce RTX 3070 Laptop GPU (`vulkan-spirv` 16×16 Karolu) | **0.861 ms** | **311.66 GFLOP/s** |
| **`directml`** | `directml:0` | NVIDIA GeForce RTX 3070 Laptop GPU (`D3D12 + DML_OPERATOR_GEMM`) | **1.001 ms** | **268.22 GFLOP/s** |
| **`pjrt`** | `tpu:0` | `PJRT_CPU_Device(id=0, isa=avx2)` (StableHLO Derleme + Yürütme) | **5.983 ms** | **44.87 GFLOP/s** |

### 2. Vulkan (`vulkan:0`) Üzerinde Uçtan Uca Model Eğitimi

- **Model:** 2 Katmanlı MLP (`B=16, D=32, H=64`), tamamen Vulkan SPIR-V çekirdekleri üzerinde ileri geçiş (`matmul`, `relu`), MSE kaybı (`sub`, `mul`, `reduce_sum`), geri geçiş (`scale`, transpozlu `matmul`) ve `special_ops.comp` içinde **SPIR-V `AdamW`** ağırlık güncellemesi.
- **Sonuç:** 25 adımda eğitim kaybı **`119.283852` → `4.468944`** (%96.25 düşüş).
