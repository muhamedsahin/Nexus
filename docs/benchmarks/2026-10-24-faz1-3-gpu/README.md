# FAZ 1 ve FAZ 3 GPU Açıklarının Kapatılması (2026-10-24)

FAZ 0–3 denetiminde açık kalan GPU işleri: FAZ 1'in CUDA ayırıcısı ve akış/olay
semantiği, FAZ 3'ün cuBLASLt katmanı, CUDA eğitim çekirdekleri ve tek kartta GPT-2
124M eğitimi. Bu klasör o işin raporu ve ham ölçümleridir.

Makine: NVIDIA GeForce RTX 3070 Laptop (GA104, sm_86, 8 GB GDDR6, 448 GB/s teorik),
sürücü 610.88, CUDA 13.4, MSVC 19.44, Release, Ninja (`build-phase1`, `NEXUS_ENABLE_CUDA=ON`).
Kart aynı anda ekranı da sürüyor (~1.6 GB başka süreçlerde).

## 1. Eklenenler

| Spec | Ne | Dosya |
|---|---|---|
| 1.3 | `Stream`, `Event` (record/wait/query/elapsed), `StreamGuard`, `DeviceGuard`, iş parçacığı başına geçerli akış | `include/nexus/cuda/runtime.hpp` |
| 1.4 | Önbellekli aygıt ayırıcısı: küçük/büyük havuz, 512 B yuvarlama, 2/20 MiB segment, blok bölme + birleştirme, akış başına serbest liste, `record_stream` ile olay tabanlı ertelenmiş serbest bırakma, OOM'da boşalt-tekrar dene + ayrıntılı hata, istatistikler, değiştirilebilir `DeviceAllocator` arayüzü | `include/nexus/cuda/caching_allocator.hpp` |
| 1.4 | Pinned host havuzu (`cudaHostAlloc`, olayla güvenli yeniden kullanım), RAII `DeviceBuffer` (ledger etiketli) | aynı |
| 1.G | `nexus::allocate_device` artık önbellekli ayırıcıdan geçer | `src/core/memory.cpp` |
| 3.1 | cuBLASLt GEMM: satır-ana sözleşme, kopyasız transpoze, plan önbelleği, sezgisel + isteğe bağlı otomatik ayar, 32 MiB çalışma alanı, epiloglar (bias, ReLU, GELU+aux, DGELU, BGRAD), strided batched, TF32 politikası (`highest/high/medium`) | `include/nexus/cuda/gemm.hpp` |
| 3.2 | LayerNorm/RMSNorm ileri (warp/satır, Welford) ve geri (≤2048 sütunda tek geçiş, yazmaçta satır; daha genişte 4 çekirdek; ikisi de atomiksiz), add&norm füzyonu | `src/cuda/kernels_norm.cu` |
| 3.3 | Nedensel softmax ileri/geri, maske çekirdek içinde (maske tensörü yok) | `src/cuda/kernels_elementwise.cu` |
| 3.4 | Gömme + konum füzyonu; geri: atomik (hızlı) ve cub radix sort + segment toplamı (belirlenimci, padding_idx) | `src/cuda/kernels_train.cu` |
| 3.5 | Yerinde füzyonlu kaydırmalı CE (ignore_index, label smoothing, z-loss; sözlük dışı hedefte kayıp NaN) | aynı |
| 3.6 | Foreach AdamW + aygıtta global norm (FP64) ve kırpma; host senkronu yok; ağırlık sönümü maskesi | aynı |
| 3.7 | Philox4x32-10 dropout, CPU maskesiyle bit-bit aynı | aynı |
| 3.8 / 3.I | `GptModel` / `GptTrainer`: GPT-2 tamamen kartta; pinned çift tampon token kopyası, N adımda bir asenkron kayıp okuma | `include/nexus/cuda/gpt.hpp` |

## 2. Testler

| Test | Kapsam | Sonuç |
|---|---|---|
| `test_cuda_allocator` | akış/olay, bölme/birleştirme, 10⁶ rastgele ayırma (sızıntı yok), akışlar arası `record_stream`, OOM yolu, pinned havuz | **30/30** |
| `test_cuda_kernels` | GEMM (4 transpoze × 5 şekil, V=50257 dahil, tüm epiloglar, batched, TF32), LN/RMS/add&norm (1×1 … 4096×768, 2×50257), softmax (T=1…1024), gömme (atomik ve belirlenimci), CE (6 yapılandırma), dropout, AdamW 1000 adım | **122/122** |
| `test_cuda_gpt_small` | L=2, C=64, V=257 GPT-2, 10 adım PyTorch ile | kayıp farkı ≤ **9.5e-7**, ilk adım gradyanları ≤ **9.4e-7** |
| `test_gpt2s_parity_fp32` | GPT-2 124M, T=1024, 10 adım AdamW + kırpma | FP64 referansa **7.9e-6** (eşik 1e-4), gradyanlar **3.7e-6** |

Ayrıntılar:

- GPU AdamW, Nexus CPU foreach AdamW ile 1000 adımda **bit-bit aynı** (kırpma dahil);
  güncelleme çekirdeği `__fmul_rn/__fadd_rn` ile FMA birleştirmesini engeller.
- GPU dropout maskesi CPU `philox_uniform` ile 1 000 003 elemanda **0 fark**.
- LayerNorm geri ve belirlenimci gömme geri iki koşuda **bit-bit aynı**.

### GPT-2 124M parity: neden FP64 referansı?

İlk koşuda Nexus GPU ile PyTorch CPU FP32 arasındaki fark ilk 3 adımda ~1e-8 idi, sonra
büyüyüp 9. adımda 1.2e-3'e çıktı (spec eşiği 1e-4). Aynı eğitimi PyTorch **FP64** ile de
koşturunca durum netleşti:

| Adım | \|Nexus − FP64\| | \|PyTorch FP32 − FP64\| |
|---|---|---|
| 0 | 2.1e-07 | 1.6e-07 |
| 3 | 4.0e-08 | 5.7e-06 |
| 6 | 1.4e-07 | 5.9e-05 |
| 9 | **7.9e-06** | **1.25e-03** |

Sapma PyTorch CPU FP32 tarafındadır. Başlangıçta gradyanı yuvarlama gürültüsü
düzeyinde olan ağırlıkları AdamW ±lr adımına normalize eder; bu yüzden toplama sırası
farkları adım adım büyür. Test, spec eşiğini (1e-4) doğru referansa (FP64) karşı uygular;
PyTorch FP32 farkı bilgi olarak raporlanır. Ham veri: `gpt2s_parity_fp32.json`.

### compute-sanitizer

Üç CUDA testi × dört araç (memcheck, racecheck, initcheck, synccheck): **0 bellek hatası,
0 yarış, 0 başlatılmamış okuma, 0 senkron hatası.** Ayırıcı testinde varsayılan memcheck'in
gösterdiği 4 kayıt, OOM testinde kasıtlı tetiklenen `cudaMalloc` yetersiz bellek
dönüşleridir; `--report-api-errors no` ile 0. Tam çıktı: `sanitizer.txt`.

## 3. Benchmark'lar (BÖLÜM G: ısınma 3 + 7 tekrar, CUDA olayları, medyan)

Ham veri: `bench_cuda_phase3.json`.

| Ölçüm | Nexus | Hedef (promt.txt) | Durum |
|---|---|---|---|
| Ayırıcı önbellek isabeti (1 MiB al+bırak) | 458 ns | < 1 µs | ✅ |
| cudaMalloc+cudaFree'ye göre | 916× hızlı | ≥ 100× | ✅ |
| add 2²⁴ FP32 | 360 GB/s (%80) | ≥ %85 tepe | ❌ (sürücü D2D kopyası 337 GB/s, %75) |
| sum 2²⁴ FP32 | 394 GB/s (%88) | ≥ %80 tepe | ✅ |
| Linear fwd+bwd 8192×4096×4096 FP32 | 8.79 TFLOPS | cuBLAS'a göre ek yük | **%0.0** (−0.005) |
| Linear fwd+bwd TF32 | 16.2 TFLOPS | aynı | **%1.5** |
| LayerNorm fwd+bwd 8192×768 | 290 GB/s (%65) | ≥ %80 tepe | ❌ |
| RMSNorm fwd+bwd 8192×768 | 302 GB/s (%67) | ≥ %80 tepe | ❌ |
| CE B·T=8192, V=50304 | 17.7 ms, ek tepe **0 B** (logit 1572 MiB) | PyTorch eager'dan ≥ 2× az bellek | ölçülmedi (PyTorch CUDA yok) |
| AdamW 124M + global norm | 12.1 ms (330 GB/s, %74) | ≤ PyTorch fused=True | ölçülmedi (PyTorch CUDA yok) |
| GPT-2 124M, T=1024, B=1 FP32 | 6 127 token/s, tepe 3.5 GB | ≥ PyTorch eager | ölçülmedi |
| GPT-2 124M, T=1024, B=2 FP32 | 6 618 token/s, tepe 5.1 GB | aynı | ölçülmedi |
| GPT-2 124M, T=1024, B=1 TF32 | 9 058 token/s, tepe 3.5 GB | aynı | ölçülmedi |
| GPT-2 124M, T=1024, B=2 TF32 | **10 212 token/s**, tepe 5.1 GB | aynı | ölçülmedi |

Dürüst notlar:

- **add %80:** Aynı koşuda sürücünün aygıt-aygıt `cudaMemcpy`'si 337 GB/s (%75) ölçüldü;
  add (float4 yükleme) bunun üstünde. Spec'in %85 hedefi teorik 448 GB/s'e göredir ve bu
  kartta tutmadı.
- **LayerNorm/RMSNorm %66–71:** Tek geçişli geri çekirdeği %44'ten buraya çıkardı. İleri
  geçiş satırı iki kez okuyor (ikincisi önbellekten). %80 için ileri yolu da yazmaçta
  tutmak gerekiyor; açık iş.
- **PyTorch GPU karşılaştırması yok:** `.bench-venv` içinde `torch 2.14.1+cpu` kurulu.
  GPU satırları bu yüzden doğrudan cuBLAS çağrısına karşıdır. CUDA'lı PyTorch kurulursa
  aynı protokolle eklenir.
- **Dikkat skoru yazılıyor:** [B,H,T,T] olasılıkları katman başına saklanır
  (GPT-2 124M, B=1'de ~600 MB). FlashAttention FAZ 5'in işidir.

## 4. Hâlâ açık olanlar

- Genel `nn::Sequential` / `nn::Graph` katmanları GPU'da çalışmıyor; `ExecutionPolicy::cuda`
  onlar için matmul'u CPU'da bırakıyor. Uçtan uca GPU yolu bugün `nexus::cuda::GptModel`.
- Birleşik `Storage/TensorImpl` (stride, view, sürüm sayacı), `TensorIterator`, DLPack (FAZ 1.B/1.E).
- CUDA VMM genişletilebilir segmentler (cuMemCreate/cuMemMap).
- Çok akışlı cuBLASLt çalışma alanı havuzu (bugün tek hesap akışı varsayılıyor).
- Gradyan biriktirme (FAZ 9) ve aktivasyon yeniden hesaplama (FAZ 7).

## Çalıştırma

```
cmake --build build-phase1
ctest --test-dir build-phase1 -R "test_cuda|test_gpt2s" --output-on-failure
build-phase1/benchmarks/runner/bench_cuda_phase3
compute-sanitizer --tool memcheck build-phase1/tests/cuda/test_cuda_kernels
```

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
