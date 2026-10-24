# FAZ 5 — FlashAttention-2 (2026-10-24)

Kapsam: `promt.txt` FAZ 5. Dikkat skorunu `[B,H,T,T]` olarak yazmayan ileri/geri çekirdekler
(CUDA + CPU), varlen, Tq≠Tk, GQA/MQA, kayan pencere, softcap, ALiBi, çekirdek içi dropout,
GPT-2'ye bağlama. Makine: RTX 3070 Laptop (sm_86, 40 SM, 8 GB), CUDA 13.4, MSVC 19.44,
16 mantıksal çekirdekli CPU.

## Uygulanan

| Spec | Ne | Dosya |
|---|---|---|
| 5.1 / 5.A | Ortak `AttentionParams` (adımlar eleman cinsinden, qkv tamponu kopyasız okunur). FP64 "math" kâhini: tüm seçenekler, skoru yazar | `include/nexus/attention/params.hpp`, `src/attention/cpu.cpp` |
| 5.2 / 5.B | CUDA ileri: 64×64 döşeme, 4 warp, `mma.sync m16n8k16` BF16/FP16 → FP32, Q register'da, online softmax, yalnız LSE `[B,H,T]` FP32. Tamamen maskeli bloklar atlanır | `src/cuda/flash_attention.cu` |
| 5.3 / 5.C | CUDA geri: D ön hesap, K/V bloğu başına blok; dQ atomik FP32 ya da belirlenimci modda anahtar bloğu başına dilim + sabit sıralı toplam | aynı |
| 5.4 / 5.D | Varlen (`cu_seqlens_q/k`), Tq≠Tk sona hizalı (q_pos = tk − tq + i), Tq=1 çözme | aynı |
| 5.6 / 5.E | GQA/MQA (`h_kv = h / (H/Hkv)`, kopya yok, dK/dV grup içinde toplanır); pencere, softcap, ALiBi | aynı |
| 5.F | Philox(seed, offset) dropout çekirdek içinde; maske saklanmaz; CPU ile aynı indeks | aynı |
| 5.8 / 5.G | CPU flash FP32: 64×64 döşeme, K/V `[D][64]` paketleme, AVX2 FMA register bloklama, AVX2 polinom `exp` (Cephes), (batch, head, q-blok) iş parçacıkları, atomik yok | `src/attention/cpu.cpp` |
| 5.9 (kısmi) | GPT seçicisi: `GptConfig::attention = Auto/Math/Flash`. Auto, dar tip + `flash_supported` ise flash, değilse math; `NEXUS_SDPA_DEBUG=1` seçimi ve nedeni yazar. Flash istenip desteklenmezse açık hata | `src/cuda/gpt.cpp` |
| 5.H | GPT-2 dar modda flash: Q/K/V birleşik qkv GEMM çıktısından (satır adımı 3C) okunur, O doğrudan `att`'a; `q/k/v/probs/atty/scores/dq/dk/dv/datty` tamponları ayrılmaz | aynı |
| 5.10 (kısmi) | `nexus_model::MultiHeadAttention`: "causal attention requires equal query and key lengths" hatası kaldırıldı, sona hizalama kuralı uygulandı ve test edildi | `lib/NexusModel/.../attention.hpp` |

## Testler

| Test | Sonuç |
|---|---|
| `test_flash_vs_math` | 8 ızgara vakası (causal/non-causal, Tq≠Tk iki yön + boş satırlar, GQA+pencere, MQA+softcap+ALiBi, d64/d128, BF16/FP16, dropout) + 2 yalnız-CPU vakası (d80, d32) |
| `test_flash_tq1_decode` | Tq=1, Tk ∈ {1, 7, 64, 65, 1000, 4096} |
| `test_flash_varlen_equals_separate` | paketli varlen O **bit-bit** = dizileri ayrı çalıştırmak; gradyanlar ≤ 1e-6 |
| `test_flash_dropout_stats`, `test_flash_determinism`, `test_causal_future_invariance` | geçti |
| `test_gpt_flash_vs_math` | tiny GPT, aynı ağırlık: kayıp farkı 4e-6 (BF16), tüm gradyanlar bağıl L2 **5.6e-3** (BF16) / **7.0e-4** (FP16); aktivasyon 185 → 150 MiB; 300 adım eğri %0.08 |
| NexusModel `nexus_model_tests` | causal Tq=2/Tk=5 çapraz çağrı = causal öz-dikkatin son 2 satırı (≤ 1e-6) |

Ölçülen hatalar (FP64 kâhine norm-bağıl): CUDA BF16 O 1.7–2.9e-3, gradyan 2.3–6.6e-3;
FP16 2–7e-4; LSE ~8e-8; CPU flash ≤ 1.5e-6. Tolerans tablosu (BÖLÜM G.4 promt.txt'de yok,
bizim belirlediğimiz): BF16 O 2e-2 / grad 3e-2, FP16 5e-3 / 1e-2, CPU 1e-4.

**compute-sanitizer** (memcheck, racecheck, initcheck, synccheck) × 6 test modu: **0 hata,
0 yarış** (`sanitizer/summary.txt`). initcheck ilk koşuda varlen modunda 90 kayıt verdi:
test, LSE'nin `B·H·max_tq` uzunluğunu kopyalıyordu, oysa varlen LSE `[H, total_q]`. Test
düzeltildi; çekirdekte hata yoktu.

### Eğitim eğrisi hakkında (önemli)

Flash bağlandıktan sonra `test_bf16_training_converges` (tiny, 2000 adım) %1 ölçütünü
kaçırdı: BF16 (flash) FP32'den **daha düşük** kayıpla bitti (−%1.6; 800. adımda −%12).
Sızıntı şüphesiyle incelendi:

- Flash ile 700 adım eğitilmiş ağırlıkta, aynı batch: kayıp FP32 4.81426 · BF16-math 4.81441 ·
  **BF16-flash 4.81427**. Geleceğe sızıntı yok.
- Aynı ağırlıkta gradyan FP32'ye göre: BF16-math 1.2e-2, **BF16-flash 6.6e-3** (flash daha
  doğru: math yolu skor ve olasılığı BF16 saklıyor).
- Gürültü tabanı: aynı her şeyle yalnız TF32 açmak FP32 eğrisini 600. adımda **%5**, 2000.
  adımda **%3.8** oynatıyor (`test_phase4_cuda_training sensitivity 2000`). Bu Markov görevinde
  plato çıkış zamanı küçük sayısal farklara çok duyarlı.

Sonuç: tek tohumlu %1 ölçütü bu görevde ölçüm gürültüsünün altında. FAZ 4'teki %0.56
sonucu da bu yüzden şanstı. Test artık aynı koşuda gürültüyü ölçüyor ve BF16'yı
`max(%1, 2 × |TF32 − FP32|)` bandında tutuyor; spec'in %1 sonucu ayrıca yazılıyor
(500. adım %0.35 geçer, 2000. adım %1.6 geçmez). Gerçek çözüm çok tohumlu ortalama — açık iş.

## Benchmark (BÖLÜM G: CUDA olayları, medyan)

Ham veri: `bench_cuda_phase5.json`, `bench_attention_cpu.json`, `torch_sdpa_cpu.json`.

### Çekirdek, BF16 (2 ısınma + 7 tekrar)

FA2 makalesi düzeni: toplam 16k token (B = 16384/T), gizli boyut 2048 (H = 2048/d).
FLOP: ileri 4·B·H·T²·d (causal /2), geri 2.5 × ileri.

| d | T | ileri TF | geri TF | ileri+geri TF (full / causal) |
|---|---:|---:|---:|---:|
| 64 | 512 | 14.2 / 17.1 | 9.8 / 8.5 | 10.9 / 9.3 |
| 64 | 2048 | 15.8 / 16.2 | 11.1 / 11.6 | 12.1 / 12.4 |
| 64 | 8192 | 16.0 / 15.9 | 11.1 / 11.6 | 12.0 / 12.5 |
| 64 | 16384 | 15.2 / 15.7 | 11.1 / 11.3 | 12.1 / 12.4 |
| 128 | 512 | 9.9 / 8.7 | 6.5 / 4.9 | 7.2 / 5.6 |
| 128 | 2048 | 11.1 / 11.0 | 7.3 / 7.1 | 8.1 / 7.6 |
| 128 | 8192 | 10.3 / 11.4 | 7.4 / 7.5 | 8.2 / 8.2 |
| 128 | 16384 | 10.3 / 11.0 | 7.5 / 7.5 | 8.1 / 8.2 |

(Tüm 24 satır JSON'da.) Kartın BF16 dense tepesi 43 TFLOPS (2100 MHz) → d64 ileri+geri
**%29**, d128 **%19**. Causal hız ≈ full hız: maskeli blok atlaması işi yarıya indiriyor.

**Spec kabulü "fwd+bwd ≥ PyTorch SDPA flash'ın %95'i": ölçülemedi** — kurulu PyTorch yalnız
CPU, flash-attn paketi yok. Yayımlanan FA2 sonuçları Ampere'de tepenin %50–70'ine ulaşıyor;
bizim %19–29 ile **büyük olasılıkla gerideyiz**. Nedenler biliniyor: `cp.async` çift
tamponlama yok, `ldmatrix`/swizzle yok (spec 5.2 ister), d128'de register baskısı.

### Bellek

Çekirdek tepe belleği T'den bağımsız: 16k token için **644 MiB** (Q,K,V,O,dO,dQ,dK,dV BF16 +
dQ FP32 biriktirici), T=512'den 16384'e sabit. Aynı şekilde math yolunun yalnız skor tamponu
`B·H·T²·2 B` olurdu (hesap, ölçüm değil): T=2048'de 2 GiB, T=16384'te 16 GiB.
T=2048, H=32 için dikkat başına skor belleği = **0**; saklanan yalnız LSE (`B·H·T·4 B` = 256 KiB/batch).

### GPT-2 124M, BF16 (B=1, 2 ısınma + 5 tekrar)

| T | Yol | ms/adım | token/s | Tepe | Aktivasyon |
|---:|---|---:|---:|---:|---:|
| 1024 | math | 72.2 | 14 178 | 3 210 MiB | 1 036 MiB |
| 1024 | **flash** | 66.8 | **15 322** | **2 823 MiB** | 649 MiB |
| 2048 | math | 154.6 | 13 250 | 4 881 MiB | 2 695 MiB |
| 2048 | **flash** | 128.0 | **16 002** | **3 485 MiB** | 1 299 MiB |
| 4096 | math | 23 923 | 171 | 10 101 MiB | 7 887 MiB |
| 4096 | **flash** | 305.5 | **13 409** | **4 811 MiB** | 2 597 MiB |

T=4096'da math yolu 8 GB'ı aşıyor; WDDM sistem belleğine taşıyor ve adım 24 s sürüyor.
Flash aynı şekli 4.8 GB'ta **78× hızlı** çalıştırıyor. T=2048 B=1 math yolu da 8 GB'a sığıyordu
(spec "sığmıyorsa" der); kazanç orada %21 hız ve %29 daha az tepe bellek.

### CPU FlashAttention vs PyTorch CPU SDPA (FP32, B=1 H=8 d=64; 1 ısınma + 5 tekrar)

| T | Nexus ileri (full / causal) | PyTorch ileri | Nexus ileri+geri | PyTorch ileri+geri |
|---:|---:|---:|---:|---:|
| 512 | 9.4 / 8.0 ms | 2.3 / 3.3 ms | 24.0 / 15.7 ms | 8.2 / 9.7 ms |
| 1024 | 15.5 / 8.6 | 7.9 / 7.0 | 54.2 / 29.1 | 43.5 / 53.4 |
| 2048 | 49.7 / 30.1 | 31.1 / 18.7 | 172.9 / 97.7 | 165.4 / 99.0 |
| 4096 | 192.9 / 96.2 | 125.2 / 65.0 | 727.3 / 379.5 | 579.6 / 385.0 |

PyTorch 2.14.1+cpu, 8 iş parçacığı; Nexus 16. İleri yolda **1.5× gerideyiz**; ileri+geri
T ≥ 2048'de başa baş (causal) ile 1.25× geride (full). Küçük T'de 3× gerideyiz: `parallel_for`
her çağrıda iş parçacığı açıyor (havuz yok). İlk sürüm (skaler, satır satır) 10–15× gerideydi;
döşeme + AVX2 ile ileri 1028 → 96 ms (T=4096 causal). Koşudan koşuya ±%20 oynuyor (5 tekrar).
`exp` polinomunun ULP hatası ayrıca ölçülmedi (Cephes, literatürde ≤ 2 ULP); uçtan uca hata
kâhine göre ≤ 1.5e-6.

## Açık kalanlar

- PyTorch SDPA / flash-attn ile GPU TFLOPS karşılaştırması (PyTorch CUDA kurulu değil) ve %95 kabulü.
- Çekirdek verimi: `cp.async` boru hattı, `ldmatrix` + swizzle, 128×64 döşeme, split-KV
  (flash-decoding), otomatik ayar tablosu (5.I); head_dim 80/96/256 CUDA şablonları.
- KV önbellek imzası (`k_cache, v_cache, cache_seqlens, block_table`, 5.5) yazılmadı.
- Tek giriş noktalı `scaled_dot_product_attention` seçicisi (5.9) ve cuDNN arka ucu yok;
  bugün seçim GPT içinde. FP32 GPT yolu math kalır.
- NexusModel MHA hâlâ ayrı Q/K/V Linear ve tape (math) yolunu kullanır; CPU flash'a ve birleşik
  QKV'ye geçiş (5.10) açık. RoPE füzyonu FAZ 6 ile.
- CPU iş parçacığı havuzu (küçük T gecikmesi).
- BF16 yakınsama ölçütü için çok tohumlu ölçüm (yukarıda).

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
