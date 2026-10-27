# FAZ 18 — CPU Arka Ucu Derin Optimizasyonu: Çok-Yollu GEMM/GEMV, Sabitli Havuz/NUMA, CPU FlashAttention, ZeRO-Offload Adam, Kuantize Çıkarım ve CPU Füzyon Derleyicisi (2026-10-27)

Kapsam: `promt.txt` FAZ 18 (18.A–18.F / 18.1–18.6).

**Donanım ve Mimari Özeti:**
- **İşlemci:** AMD Ryzen 7 5800H (Zen 3, 8 fiziksel çekirdek / 16 mantıksal iş parçacığı, 1 NUMA düğümü, L1d 32 KB/çekirdek, L2 512 KB/çekirdek, paylaşılan L3 16 MB).
- **Komut Seti (ISA):** Donanımda **AVX2 + FMA + F16C + BMI2** mevcut; donanımsal AVX-512 / VNNI / AMX bu işlemcide bulunmuyor (`CPUID + XGETBV` ile çalışma zamanında algılanıyor).
- **KURAL 7 (Özellik Eşdeğerliği ve Çok-Yollu Tasarım):**
  - Çalışma zamanı `active_gemm_isa()` yönlendiricisi donanımda mevcut olduğunda `AmxTile` / `Avx512Bf16` / `Avx512Vnni` / `Avx512F` / `Avx2Fma` yollarını seçer.
  - AVX-512 BF16 (`vdpbf16ps`) ve AMX (`tdpbf16ps`) çiftli nokta çarpımı veri düzeni (`[ceil(N/16)][ceil(K/2)][16][2]`) ve biriktirme semantiği `PackedBf16Vnni` / `pack_bf16_vnni` / `gemm_bf16_vnni` ile gerçeklenmiş ve `test_cpu_gemm_avx512_vs_ref` ile FP64 referansa karşı doğrulanmıştır (`1.61e-7`).
  - `nexus::compile` CPU JIT kod üreticisi (`src/compile/codegen.cpp`) her füzyon kümesi için hem `#if defined(__AVX512F__)` (`__m512` 16-genişlikli FMA + `_mm512_reduce_add_ps`) hem `#elif defined(__AVX2__)` (`__m256` 8-genişlikli FMA) hem de skaler kuyruk kodunu üretir ve `CpuJit::load` içinde CPUID'ye göre `/arch:AVX512` veya `/arch:AVX2` seçer.
- **Bellek:** 32 GB DDR4.
- **GPU (ZeRO-Offload doğrulaması için):** NVIDIA GeForce RTX 3070 Laptop GPU (8 GB, sm_86).
- **Yazılım:** Windows 11, MSVC 19.44 (`/O2 /Ob2 /arch:AVX2` + Win64 MASM `ml64.exe`), PyTorch `2.14.1+cpu`.
- Tüm zaman ölçümleri **fişte** (`Win32_Battery` = 2) alınmıştır.

---

## Ne Yapıldı (FAZ 18.1 – 18.6)

| Alt Adım | Durum | Mimari ve Optimizasyon Detayları | Dosyalar |
|---|---|---|---|
| **18.A / 18.1** CPUID + Topoloji | ✅ Tamamlandı | • `CPUID` (yaprak 0, 1, 7.0, 7.1, 0x1D/0x1E) + `XGETBV` (XCR0 bit 1–2 YMM, 5–7 ZMM, 17–18 AMX XTILE) ile gerçek OS-destekli ISA algılama.<br>• `GetLogicalProcessorInformationEx` ile fiziksel çekirdek, SMT kardeşleri, L1d/L2/L3 önbellek boyutları ve NUMA düğüm haritası.<br>• `active_gemm_isa()` ve `isa()` çalışma zamanı çekirdek seçimi; AVX2+FMA yoksa sessizce yavaş yola düşmez, açık hata verir. | `include/nexus/cpu/features.hpp`, `src/cpu/threads.cpp` |
| **18.B / 18.1** Çok-Katmanlı GEMM & GEMV | ✅ Tamamlandı | • **Ortak Paralel B Paneli Paketleme (`kKc = 512`):** Eski yolda her işçi kendi `B` dilimini tekrar tekrar paketliyordu; yeni mimaride `B` paneli tüm havuz işçileri tarafından paralel olarak **tek kez** 64-bayt hizalı ortak tampona (`t_panel_b`) paketlenir (`pack_b_panel_16`).<br>• **Win64 MASM 4×16 Mikroçekirdek (`nexus_avx2_kernel_4x16`):** C++ derleyicisinin 12 YMM yazmacında yaptığı yığın dökümünü (register spill) tamamen ortadan kaldırmak için `matrix_pro_cpu` içindeki el yazımı MASM `nexus_avx2_kernel_4x16` (8 YMM biriktirici `ymm0..ymm7`, 8× K-açma, sıfır yığın dökümü, L1/L2 prefetch) doğrudan bağlandı.<br>• **AVX2 8×8 / 4×8 Yazmaç-İçi Devrik Paketleme:** `b_trans = true` (`[N, K]`) için `transpose8x8_store`, `a_trans = false` için `pack_a_4` içinde 4×8 SIMD devrik paketleme; BF16 (`_mm256_slli_epi32`) ve FP16 (`_mm256_cvtph_ps` F16C) ağırlıklar paketleme sırasında vektörel olarak FP32'ye açılır.<br>• **Yüksek Bant Genişlikli GEMV (`M <= 4`):** Paketlemesiz doğrudan akış; `b_trans = false` (`[K, N]`) için 64/32/16 sütunluk YMM yazmaç biriktiricili tek geçişli `gemv_kn`, `b_trans = true` (`[N, K]`) için 4-sütun×2-YMM `gemv_block`.<br>• **INT8 (`PackedS8`) & INT4 (`PackedQ4`) GEMM/GEMV:** Kanal başına simetrik INT8 (`[ceil(N/16)][Kp/4][16][4]`) ve optimal MSE kırpmalı 4-bit nibble-şeritli INT4 (`[ceil(N/16)][Kp/8][16][4]`); AVX2 `_mm256_maddubs_epi16` + `_mm256_madd_epi16` ile 4×16 mikroçekirdek.<br>• **Belirlenimcilik:** Sonuç iş parçacığı sayısından (`1, 3, 8`) bağımsız olarak **bit-bit aynıdır**. | `include/nexus/cpu/gemm.hpp`, `src/cpu/gemm_avx2.cpp` |
| **18.C / 18.2** BF16 VNNI / AMX Karo Yolu | ✅ Tamamlandı | • AVX-512 BF16 (`vdpbf16ps`) ve AMX (`tdpbf16ps` 16×16 karo) için çiftli VNNI ağırlık paketleme (`PackedBf16Vnni`: `[ceil(N/16)][ceil(K/2)][16][2]`) ve çiftli nokta çarpımı yürütücüsü (`gemm_bf16_vnni`).<br>• Kenar sütunlarda (`N % 16 != 0`) güvenli kuyruk yazımı ve ASan doğrulaması. | `include/nexus/cpu/gemm.hpp`, `src/cpu/gemm_avx2.cpp` |
| **18.D / 18.3** NUMA & Sabitli İş Parçacığı Havuzu | ✅ Tamamlandı | • Fiziksel çekirdek başına 1 işçi (Zen 3'te mantıksal `0, 2, 4, 6, 8, 10, 12, 14`; SMT kardeş çakışması yok).<br>• **Sıfır Sistem Çağrılı Ana İş Parçacığı Sabitleme:** Her `Pool::each` çağrısında 2 kez yapılan `SetThreadAffinityMask` sistem çağrısı kaldırıldı; ana iş parçacığı ilk kullanımda `cpus[0]`'a bir kez sabitlenir.<br>• **64-Bayt Hizalı Melez Spin-Wait Bariyer:** `alignas(64)` atomik sayaçlar (`generation`, `remaining`, `sleeping`, `stop`) + uyku öncesi 8000 iterasyonluk `_mm_pause()` spin penceresi ile ardışık `Pool::each` / `Pool::run` gecikmesi ~40 µs'den **< 0.3 µs**'ye düşürüldü; kayıp uyandırma (lost wakeup) tasarım gereği imkânsız.<br>• `VirtualAllocExNuma` + `MEM_LARGE_PAGES` (2 MB büyük sayfa isteği ve açık tanı raporu) + paralel ilk dokunuş (`first-touch`) sayfa yerleştirme. | `include/nexus/cpu/threads.hpp`, `src/cpu/threads.cpp` |
| **18.E / 18.4** CPU FlashAttention-2 (FP32 / BF16 / FP16) | ✅ Tamamlandı | • **Sıfır Yığın Ayırmalı (`thread_local FlashScratch`) Mimari:** İleri ve geri geçişte görev başına yapılan `std::vector` ayırmaları tamamen kaldırıldı.<br>• **Sabitli Havuz + Atomik İş Çalma:** Nedensel (causal) dikkat üçgenlerindeki yük dengesizliğini gidermek için `cpu::Pool::instance()` üzerinde atomik indeks (`next.fetch_add(1)`) ile dinamik iş paylaşımı.<br>• **AVX2 8×8 Yazmaç-İçi Devrik K/V Paketleme (`transpose8x8_to_block`):** `[D][64]` K ve V blok paketlemesi SIMD matris devriği ile hızlandırıldı.<br>• **Vektörel Online Softmax + Birleşik `exp_shift_sum`:** İlk KV bloğunda (`l_prev == 0`) ve maksimum değişmediğinde (`m_new == m_prev`) gereksiz `acc *= alpha` ölçeklemesi atlanır.<br>• **Tek Geçişli (Single-Pass) Geri Geçiş:** `batch * kv_heads >= threads` olduğunda `tile()` (`Q K^T`, `exp`, `dO V^T`) iki kez yerine **tek kez** hesaplanarak `dV`, `dK` ve `dQ` tek geçişte biriktirilir.<br>• **BF16 ve FP16 Desteği:** FP32 yanında BF16 ve FP16 girdi/çıktı desteği (yüklemede vektörel FP32 açma, FP32 online softmax/biriktirme, kayıtta RNE daraltma). | `include/nexus/attention/cpu.hpp`, `src/attention/cpu.cpp` |
| **18.F / 18.5** ZeRO-Offload CPU Adam & Kuantize LLM Çıkarımı | ✅ Tamamlandı | • **Tek Geçişli AVX2 AdamW (`nexus/cpu/adam.hpp`):** FP32/BF16 gradyan açma + kırpma + AdamW güncellemesi + RNE BF16 gölge ağırlık üretimi tek bellek geçişinde; GPU `fused::adamw_multi_tensor` ile **bit-bit aynı**.<br>• **Sıfır-Ayırmalı Hızlı CPU Çıkarım Motoru (`LanguageModel::forward_cpu_fast`):** `CpuWeightMode::{Auto, F32, BF16, INT8, INT4}` ve `prepare_cpu_inference()` ile model ağırlıklarını `[N, K]` devrik FP32/BF16, `PackedS8` (INT8) veya `PackedQ4` (INT4, `lm_head` Q8_0 korumalı) olarak hazırlar.<br>• `thread_local FastInferenceScratch`, yerinde RoPE, birleşik RMSNorm/LayerNorm ve **tek-token AVX2 GQA decode dikkat çekirdeği (`decode_attention_1tok_cpu`)** ile TinyLlama-1.1B üzerinde **41.15 token/sn** (INT4) ve **23.06 token/sn** (INT8) gerçek çıkarım hızı. | `include/nexus/cpu/adam.hpp`, `include/nexus/lm/model.hpp`, `src/lm/model.cpp`, `src/lm/ops.cpp` |
| **18.6** `nexus::compile` CPU Arka Ucu ve Füzyon Derleyicisi | ✅ Tamamlandı | • `src/compile/codegen.cpp` (`cpu_source`): Füzyon kümeleri için hem **AVX-512 (`__m512`)** hem **AVX2 (`__m256`)** vektör döngülerini (FMA, hızlı polinom `exp`/`erf`/`tanh`/`rsqrt`, yatay indirgeme) üretir.<br>• `src/compile/executor.cpp`: Füzyon kümelerini `cpu::Pool::instance()` üzerinde paralel yürütür (`dispatch_fused_cpu`); `AttentionQKV` ileri geçişini birleşik QKV tamponu üzerinde **sıfır kopya (strided view)** ile çalıştırır; `MatMulBias` ve `LinearBiasGelu` içinde bias'ı doğrudan `cpu::gemm` içine kaynaştırır; `LayerNorm`/`RMSNorm`/`AddNorm`/`SwiGLU`/`SumRows` işlemlerini AVX2 + paralel havuz ile çalıştırır. | `src/compile/codegen.cpp`, `src/compile/executor.cpp` |

---

## Doğrulama ve Eşdeğerlik Testleri (`ctest`)

| Test Adı | Kapsam ve Ölçüt | Sonuç |
|---|---|---|
| `test_cpu_features` | CPUID + XGETBV + Windows topoloji tutarlılığı (`8 fiziksel / 16 mantıksal`, `1 NUMA`, `AVX2+FMA+F16C`) | ✅ **PASS** |
| `test_cpu_gemm_vs_ref` / `test_cpu_gemm_avx2_vs_ref` | 24 durum (`FP32 / BF16 / FP16` × `[K,N] / [N,K]` × kenar boyları / KC aşımı / GEMV / `beta + bias`) vs FP64 referans; `1 / 3 / 8` işçide bit-bit belirlenimcilik; INT8 (`PackedS8`) ve INT4 (`PackedQ4`) doğruluğu | ✅ **PASS** (FP32/BF16/FP16 en kötü göreli hata **1.34e-6** ≤ 1e-5; INT8 tam sayı ref **1.1e-7**; INT4 kosinüs benzerliği **0.99387** > 0.985) |
| `test_cpu_gemm_avx512_vs_ref` | `PackedBf16Vnni` (`vdpbf16ps` / `tdpbf16ps` çiftli BF16 nokta çarpımı düzeni ve biriktirme semantiği, kenar boyutu `M=29, N=67, K=195` + bias) vs referans | ✅ **PASS** (en kötü göreli hata **1.61e-7** ≤ 1e-5) |
| `test_numa_affinity` / `test_cpu_numa_and_pinning` | İşçilerin fiziksel çekirdeklere (`0, 2, 4, 6, 8, 10, 12, 14`) sabitlenmesi (SMT çakışmasız), NUMA düğüm eşlemesi, `VirtualAllocExNuma` ilk dokunuş ve büyük sayfa tanı raporu | ✅ **PASS** |
| `test_offload_adam_bitwise` / `test_cpu_adam_step` / `test_zero_offload_step` | AVX2 tek-geçişli AdamW vs skaler `fused::adamw` (2 grad tipi × 3 boyut × 2 weight decay × 5 adım): `p`, `m`, `v`, `BF16` gölge ağırlıklar ve FP64 `sq_norm` | ✅ **PASS** (farklı öğe sayısı **0** — **bit-bit aynı**, `sq_norm` bağıl fark **1.47e-13**) |
| `test_zero_offload_equals_gpu_adam` (CUDA) | 200 eğitim adımında GPU+CPU ZeRO-Offload ile saf GPU AdamW karşılaştırması | ✅ **PASS** (parametre farkı **0** — **bit-bit aynı**) |
| `test_cpu_flash_vs_math` / `test_cpu_flash_attn_vs_ref` | 8 FP32 durumu (tam, causal, GQA 4/2, `Tq != Tk`, tek/çift yönlü pencere, softcap+ALiBi, `D=40`, `T=1`) + BF16 ileri/geri geçiş vs FP64/FP32 matematiksel kahin | ✅ **PASS** (FP32 O/LSE/dQ/dK/dV en kötü bağıl hata **1.09e-6** ≤ 1e-4; BF16 ileri+geri **2.77e-4** ≤ 2e-2) |
| `test_cpu_train_parity` | 4 katman `d=256, T=128, B=4` LM, 60 eğitim adımı: yeni AVX2 GEMM vs eski referans GEMM kayıp eğrisi | ✅ **PASS** (kayıp `5.5891 -> 1.7430`, en kötü adım bağıl fark **1.18e-6** ≤ 1e-3, **4.33× daha hızlı**) |
| `test_cpu_quantized_inference` | 4 katmanlı LLaMA-tarzı GQA+RoPE+SwiGLU dil modelinde KV-önbellekli artımlı çıkarım (`F32`, `BF16`, `INT8`, `INT4`) vs tam geçiş FP32 referans logit kosinüs benzerliği ve bellek ayak izi | ✅ **PASS** (F32 `1.000000` [12.00 MB], BF16 `0.999991` [6.00 MB], INT8 `0.999732` [3.04 MB], INT4 `0.960585` [1.60 MB — **7.5× küçülme**]) |
| `test_cpu_compile_fusion` | `nexus::compile` (18.6): üretilen CPU JIT kaynağında hem `_mm512_` hem `_mm256_` vektör dallarının varlığı, istekli (eager) vs derlenmiş ileri+geri kayıp eşdeğerliği ve statik bellek planı | ✅ **PASS** (`eager loss = 5.557344`, `compiled loss = 5.557344`, fark **0.00e+00**; çekirdek sayısı `90 -> 79`; arena **5.63 MB** vs naif **16.63 MB**) |

---

## Karşılaştırma Ölçümleri (Benchmarklar)

### 1. CPU GEMM GFLOP/s (Kare `n × n × n`, 8 İş Parçacığı)

| Boyut `n` | Nexus FP32 (1 i.p.) | **Nexus FP32 (8 i.p.)** | **Nexus BF16 B** | **Nexus FP16 B** | **Nexus INT8 (GOP/s)** | **Nexus INT4 (GOP/s)** | Eski `gemm_nn` | Hızlanma (FP32 vs Eski) | PyTorch FP32 (oneDNN) | PyTorch BF16 (AVX2) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **256** | 44.5 | **258.9** | **254.4** | **263.0** | 223.1 | **330.3** | 40.2 | **6.44×** | 316 | 0.96 |
| **512** | 39.6 | **311.7** | **316.4** | **321.6** | 232.5 | **344.0** | 39.8 | **7.83×** | 362 | 0.97 |
| **1024** | 37.6 | **299.9** | **322.4** | **317.6** | 311.8 | **358.0** | 25.8 | **11.62×** | 358 | 0.75 |
| **2048** | 36.9 | **304.9** | **277.0** | **301.6** | 313.2 | **393.0** | 21.6 | **14.12×** | 363 | — |
| **4096** | — | **292.3** | **301.3** | **299.7** | 327.4 | **396.0** | — | — | 396 | — |

- Win64 MASM 4×16 mikroçekirdek (`nexus_avx2_kernel_4x16`) + ortak paralel `B` panel paketleme (`kKc = 512`) + AVX2 8×8 yazmaç-içi devrik paketleme sayesinde:
  - FP32 GEMM hızı `258.9 – 311.7 GFLOP/s` seviyesine ulaşarak eski `gemm_nn` çekirdeğine göre **6.4× – 14.1×** hızlanma sağladı (1→8 çekirdek ölçeklenmesi `2048³`'te **8.26×** — süper-doğrusal L3/panel paylaşım etkisi).
  - BF16 ağırlıklı GEMM (`322.4 GFLOP/s`'e kadar), AVX2 üzerinde BF16 hızlı yolu olmayan PyTorch'tan (`0.75 – 0.97 GFLOP/s`) **>330×** hızlıdır.
  - Yeni 4-bit nibble-şeritli `gemm_q4` (INT4) çekirdeği **396.0 GOP/s**'e ulaşmaktadır.

### 2. CPU GEMV (`M = 1`, Tek-Token Çıkarım), 8 İş Parçacığı

| Şekil `[N, K]` | **Nexus FP32** | **Nexus BF16** | **Nexus INT8** | **Nexus INT4** | PyTorch FP32 | PyTorch BF16 |
|---|---:|---:|---:|---:|---:|---:|
| **4096 × 2048** | **0.780 ms (43.0 GB/s)** | **0.153 ms (109.6 GB/s*)** | **0.059 ms** | **0.069 ms** | 23.8 GB/s | 41.0 GB/s |
| **8192 × 4096** | **3.558 ms (37.7 GB/s)** | **1.815 ms (37.0 GB/s)** | **0.912 ms** | **0.285 ms** | 25.9 GB/s | 38.4 GB/s |

\* `4096 × 2048` BF16 (16 MB), INT8 (8 MB) ve INT4 (4 MB) matrisleri 16 MB L3 önbelleğe sığdığından L3 bant genişliğinde (~110 GB/s) çalışır; `8192 × 4096` gerçek DDR4 DRAM bant genişliğini (~37.7 GB/s) doyurur ve INT4'te FP32'ye göre **12.5×** daha düşük gecikme (`0.285 ms` vs `3.558 ms`) sağlar.

### 3. CPU FlashAttention-2 vs PyTorch SDPA (`FP32`)

**8 Fiziksel Çekirdek (`test_phase18_cpu bench_attn`, `B=2, H=8, D=64`):**

| `T` | Maske | 1 Çekirdek İleri (ms) | **8 Çekirdek İleri (ms)** | İleri Ölçeklenme | 1 Çekirdek İleri+Geri (ms) | **8 Çekirdek İleri+Geri (ms)** | İleri+Geri Ölçeklenme |
|---:|---|---:|---:|---:|---:|---:|---:|
| **512** | tam | 16.50 | **2.28** | 7.24× | 55.16 | **8.65** | 6.38× |
| **512** | causal | 9.72 | **1.50** | 6.47× | 32.66 | **5.35** | 6.11× |
| **1024** | tam | 69.86 | **8.86** | 7.89× | 229.57 | **32.79** | 7.00× |
| **1024** | causal | 37.80 | **5.52** | 6.84× | 132.36 | **20.41** | 6.48× |
| **2048** | tam | 277.17 | **36.49** | 7.60× | 921.72 | **126.06** | 7.31× |
| **2048** | causal | 148.97 | **18.36** | 8.11× | 478.75 | **76.35** | 6.27× |
| **4096** | tam | 1140.42 | **144.87** | 7.87× | 3857.96 | **545.72** | 7.07× |
| **4096** | causal | 580.73 | **75.53** | 7.69× | 1913.62 | **268.79** | 7.12× |

- Tek geçişli (single-pass) geri geçiş, `thread_local FlashScratch` ve AVX2 8×8 devrik K/V paketleme sayesinde ileri+geri geçiş süresi önceki FAZ 18 taslağına göre **1.62× – 1.81×** daha hızlandı (`T=4096` causal fwd+bwd `459.4 ms -> 268.8 ms`), 1→8 çekirdek ölçeklenmesi nedensel maskede dahi **6.1× – 8.1×** aralığına ulaştı.
- `B=1, H=8, D=64` (16 iş parçacığı) karşılaştırmasında Nexus CPU FlashAttention, 16 durumun 15'inde PyTorch SDPA'dan öndedir ve FAZ 5 çekirdeğinden **2.0× – 3.2×** hızlıdır.

### 4. ZeRO-Offload CPU AdamW (`bench_adam` ve GPU+CPU `gpt2-350m`)

- **130M Parametre CPU AdamW Mikro-Ölçümü (`test_phase18_cpu bench_adam`):**
  - FP32 gradyan: eski yol `281.5 ms` → **yeni AVX2 sabitli havuz `191.5 ms` (1.47× hızlanma, 23.1 GB/s)**.
  - BF16 gradyan: eski yol `282.9 ms` → **yeni AVX2 sabitli havuz `181.2 ms` (1.56× hızlanma, 21.5 GB/s)**.
- **Uçtan Uca GPU+CPU ZeRO-Offload (`gpt2-350m`, BF16, `B=1, T=1024`):**
  - Adım süresi `1116–1235 ms` → **`825–881 ms` (%23–31 azalma)**, GPU AdamW ile **bit-bit aynı** ağırlıklar.

### 5. CPU Eğitimi (`test_cpu_train_parity`, 4 Katman `d=256, T=128, B=4`, 60 Adım)

- Eski skaler GEMM: **938.7 ms/adım**
- **Yeni FAZ 18 AVX2 GEMM: 216.9 ms/adım (4.33× hızlanma)**
- Kayıp eğrisi: `5.5891 -> 1.7430` (en kötü adım bağıl farkı **1.18e-6**).

### 6. Gerçek Model CPU Çıkarımı: TinyLlama-1.1B (`test_phase18_cpu bench_decode`, 22 Katman, `d=2048, F=5632, V=32000`)

İstem: 16 token prefill + 31 yeni token KV-önbellekli greedy üretim (`LanguageModel::forward_cpu_fast` + `prepare_cpu_inference`):

| Ağırlık Modu | Ağırlık Belleği | Prefill (16 token) | **Decode (ms/token)** | **Üretim Hızı (token/sn)** | FP32'ye Göre Hızlanma | Etkin Ağırlık Bant Genişliği |
|---|---:|---:|---:|---:|---:|---:|
| Eski Skaler GEMM (FP32) | 3946.0 MB | 1702.0 ms | 429.0 ms | 2.33 tok/s | 0.44× | 10.3 GB/s |
| **FAZ 18 `F32`** | **3946.0 MB** | **760.1 ms** | **188.1 ms** | **5.32 tok/s** | **1.00×** | **22.0 GB/s** |
| **FAZ 18 `BF16`** | **1973.0 MB** | **691.5 ms** | **86.2 ms** | **11.60 tok/s** | **2.18×** | **24.0 GB/s** |
| **FAZ 18 `INT8` (`PackedS8`)** | **988.1 MB** | **164.1 ms** | **43.4 ms** | **23.06 tok/s** | **4.33×** | **23.9 GB/s** |
| **FAZ 18 `INT4` (`PackedQ4`)** | **526.1 MB** | **210.3 ms** | **24.3 ms** | **41.15 tok/s** | **7.73×** | **22.7 GB/s** |

- Tüm modlarda (`F32`, `BF16`, `INT8`, `INT4`) DDR4 bellek bant genişliği (~22–24 GB/s uçtan uca, dikkat + norm + aktivasyon dahil) doyurulmaktadır.
- `INT4` modu ağırlık belleğini **3.95 GB'tan 526 MB'a (7.5× küçülme)** indirirken tek bir dizüstü CPU'da **41.15 token/sn** gerçek üretim hızına ulaşmaktadır (eski skaler yola göre **17.7×** hızlı).

### 7. FAZ 18.6 `nexus::compile` CPU Füzyon Derleyicisi (`test_phase13_compile bench`)

| Model | Mod | İstekli (Eager) İleri+Geri | **Derlenmiş (Compiled) İleri+Geri** | **Hızlanma** | Çekirdek Sayısı (Eager → Compiled) | Statik Arena vs Naif Bellek |
|---|---|---:|---:|---:|---:|---:|
| `gpt2-124m` | `default` | 260.47 ms | **196.59 ms** | **1.325×** | 167 → 92 | **36.9 MB** vs 92.0 MB (**2.49× tasarruf**) |
| `llama2-7b` (vekil) | `default` | 232.18 ms | **136.01 ms** | **1.707×** | 170 → 151 | **40.0 MB** vs 141.4 MB (**3.54× tasarruf**) |

---

## Sanitizer (`build-asan`, MSVC `/fsanitize=address`)

`build-asan` dizininde `/fsanitize=address` ile derlenen tüm FAZ 18 ve ilgili MoE testleri çalıştırıldı:

- İlk ASan koşusunda `gemm_bf16_vnni` içinde `N % 16 != 0` kenar sütunlarında (`N=67, n0=64, nn=3`) `bias` vektör yüklemesinin (`_mm256_loadu_ps(bias + n0)`) 20 bayt tampon dışına taştığı tespit edildi ve `if (nn == 16)` dalı içine alınarak düzeltildi.
- Düzeltme sonrası çalıştırılan 15 ASan testinin tamamı (`test_moe_k1_equals_dense`, `test_cpu_features`, `test_cpu_gemm_vs_ref`, `test_cpu_gemm_avx2_vs_ref`, `test_cpu_gemm_avx512_vs_ref`, `test_numa_affinity`, `test_cpu_numa_and_pinning`, `test_offload_adam_bitwise`, `test_cpu_adam_step`, `test_zero_offload_step`, `test_cpu_flash_vs_math`, `test_cpu_flash_attn_vs_ref`, `test_cpu_train_parity`, `test_cpu_quantized_inference`, `test_cpu_compile_fusion`) **15/15 PASS, 0 ASan hatası** ile geçti.

---

## Regresyon (KURAL 9)

- `build-phase2-cpu` üzerindeki 152 CTest hedefinin tamamı (**152/152 PASS**) ve `build-phase1` (CUDA Release) üzerindeki ZeRO-Offload + FAZ 18 testlerinin tamamı (**13/13 PASS**) sıfır hatayla geçti.
- KURAL 9 regresyon koşucuları (`bench_phase0`, `bench_phase1`, `bench_phase2`, `bench_phase3_fused`, `bench_phase3_training`, `bench_attention_cpu`) **6/6 PASS** ile doğrulandı; önceki fazlarda hiçbir performans veya doğruluk gerilemesi yoktur.

---

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
