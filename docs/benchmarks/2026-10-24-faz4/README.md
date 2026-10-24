# FAZ 4 — Karışık Hassasiyet (2026-10-24)

Kapsam: `promt.txt` FAZ 4 — BF16/FP16 autocast, FP32 ana kopya, dinamik kayıp ölçeği,
TF32 politikası, FP8 altyapısı. Makine: RTX 3070 Laptop (sm_86, 8 GB), CUDA 13.4, MSVC 19.44.

> Not: Bu klasörün ilk taslağı, GPU başka işle meşgulken alınmış ölçümler içeriyordu
> (GPT-2 FP32 1286 ms/adım). Aynı benchmark temiz koşulda yeniden çalıştırıldı; aşağıdaki
> sayılar o koşudandır. FAZ 3 benchmark'ı da aynı oturumda tekrarlandı (FP32 181 ms/adım,
> FAZ 3 raporundaki 167 ms ile uyumlu), yani FP32 yolunda gerileme yoktur.

## Uygulanan

| Spec | Ne | Dosya |
|---|---|---|
| 4.1 | `AutocastGuard`, op politika tablosu (Lower / Fp32 / Promote / Keep), GPT yolu tipleri tablodan alır | `include/nexus/amp/autocast.hpp` |
| 4.1 | Ağırlık cast önbelleği (kaynak, tip, boyut, akış, sürüm) | `include/nexus/cuda/cast_cache.hpp` |
| 4.A | GEMM A/B BF16/F16, C F32 veya dar; FP32 biriktirici. LN, softmax, kafa permütasyonları, gömme depolama tipine şablonlu (FP32 yolu aynı çekirdek örnekleri, bit-bit aynı) | `src/cuda/*` |
| 4.2 | Ana kopya modları A (cast/forward), B (FP32 ana kopya, varsayılan), C (ana kopyasız, BF16 stokastik yuvarlama) | `src/cuda/gpt.cpp` |
| 4.3 | Tek kayıp ölçeği politikası (`nexus_train::LossScalePolicy`); `nexus_train::LossScaler`, `nexus::detail::LossScaler` ve aygıt kodu aynı kuralları kullanır. FP16'da unscale + isfinite AdamW'in norm geçişinde, tek bayrak, host senkronu yok | `lib/NexusTrain/.../loss_scale_policy.hpp`, `src/cuda/kernels_train.cu` |
| 4.4 | FP32 matmul varsayılanı "high" (TF32) | `src/cuda/gemm_lt.cpp` |
| 4.5 | E4M3/E5M2 host türleri (RNE, doymalı), gecikmeli ölçekleme, blok ölçek hazırlığı, aygıt niceleme + amax, emüle FP8 GEMM; yerel FP8 sm_89+ ister, bu kartta açık hata | `include/nexus/core/fp8.hpp`, `src/cuda/kernels_fp8.cu`, `src/cuda/fp8_native.cpp` |
| 4.6 | Softmax/CE/norm FP32 biriktirir; kafa çıktısı (logit) FP32'ye yükseltilerek CE'ye girer | `src/cuda/gpt.cpp` |

## Testler

| Test | Sonuç |
|---|---|
| `test_autocast_policy` | tablo tüm op sınıflarında doğru |
| `test_phase4_narrow_gemm`, `test_phase4_cast_cache` | geçti |
| `test_fp8_emulation`, `test_phase4_host_fp8` | 256 desen gidiş-dönüş, aygıt/host bit-bit, amax geçmişi |
| `test_fp16_loss_scale` | kasıtlı inf → adım atlanır, parametre değişmez, ölçek 1024 → 512; sonraki temiz adım uygulanır |
| `test_master_weight_modes` (tiny, 500 adım) | **A ve B bit-bit aynı**; C, B'den %4.0 düşük kayıp (deneysel), %25 daha az durum belleği |
| `test_bf16_training_converges` (tiny, 2000 adım) | 2000. adımda **%0.56** ✅ · 500. adımda **%2.7** ❌ (spec %1) |

Tiny = `configs/tiny.json` şekli (L=6, d=384, H=6, T=256, V=8192, B=4). Veri: her token
için 4 ardıllı Markov zinciri (teorik taban 1.208 nat), ezber yok; iki hassasiyet aynı
batch'leri görür; karşılaştırma 50 adımlık pencere ortalamasıdır.

BF16 eğrisi FP32'nin sürekli biraz gerisinde (300–500. adım %2.6–3.0, sonra %0.5–1.3).
500. adım kaybın hızlı düştüğü bölgede. Bunun BF16'nın doğal maliyeti mi yoksa Nexus'a özgü
bir kayıp mı olduğunu ayırmak için PyTorch BF16 autocast ile aynı kurulumda karşılaştırma
düzeneği yazıldı (`tests/cuda/reference_amp_torch.py`, `test_phase4_cuda_training dump`),
ama koşu oturum kesilince tamamlanmadı. **Açık iş.**

## Benchmark (BÖLÜM G: 2 ısınma + 7 tekrar, CUDA olayları, medyan)

Ham veri: `bench_cuda_phase4.json`. GPT-2 124M, T=1024, B=1.

| Mod | ms/adım | token/s | Tepe bellek |
|---|---:|---:|---:|
| FP32 | 176.1 | 5 815 | 3 526 MiB |
| TF32 | 120.7 | 8 484 | 3 526 MiB |
| BF16, B (FP32 ana kopya) | 71.5 | **14 313** | 3 209 MiB |
| FP16, B + dinamik ölçek | 78.0 | 13 122 | 3 209 MiB |
| BF16, A (cast/forward) | 72.3 | 14 155 | 3 209 MiB |
| BF16, C (ana kopyasız, deneysel) | 62.7 | **16 320** | **2 734 MiB** |

BF16 B, FP32'den 2.46×, TF32'den 1.69× hızlı (GPT satırları koşudan koşuya ±%7 oynar).

Linear fwd+bwd 8192×4096×4096 (3 ısınma + 15 tekrar): BF16 **31.1 TFLOPS**, FP16 **30.9 TFLOPS**.
Dense tepe = 40 SM × 512 FLOP/saat × saat. CUDA'nın bildirdiği saat (1560 MHz) ile tepe 31.9
TFLOPS ve oran %97 çıkar; ama kart yük altında 2100 MHz'e kadar çıkar. Azami saatle tepe 43.0
TFLOPS ve oran **%72** — spec hedefi (≥ %70) bu daha sıkı ölçüte göre de sağlanır.

## Açık kalanlar

- BF16 500. adım %1 ölçütü (yukarıda) ve PyTorch BF16 autocast eğri karşılaştırması.
- PyTorch GPU token/s ve bellek karşılaştırması: kurulu PyTorch yalnız CPU.
- Yerel FP8 matmul: sm_89+ gerekir, bu makinede çalıştırılamaz (yalnız derleme).
- CPU autograd yolu FP32 kalır; autocast tablosu bugün GPU GPT yolunda kullanılır.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
