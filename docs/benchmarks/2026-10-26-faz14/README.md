# FAZ 14 — Uzman Karışımı (MoE) ve Uzman Paralelliği (2026-10-26)

Kapsam: `promt.txt` FAZ 14 (DUVAR 16). MoE çekirdeği CPU'da (ham FP32 tamponlar, elle yazılmış geri geçiş). EP, thread
ProcessGroup üzerinden all-to-all(v) ile çalışıyor. GPU tarafında yalnız uzman GEMM ölçümü var.

Makine: RTX 3070 Laptop 8 GB, Ryzen 7 5800H (8 çekirdek / 16 iş parçacığı), Windows 11, **fişte** (`Win32_Battery` 2).
Pilde alınan ilk GPU ölçümü (4.3 TFLOPS) atıldı.
Veri: TinyStoriesV2-GPT4, bayt düzeyinde (sözlük 256, tokenizer yok).

## Ne yapıldı

| Alt adım | Durum | Nerede |
|---|---|---|
| 14.A yönlendirici (FP32), top-k, kapı = seçilen k üzerinde softmax, aux kaybı, z-kaybı | ✅ | `moe::route`, `router_backward` |
| 14.A gürültülü top-k (Shazeer): `H = logits + eps · softplus(x W_noise)`, tohumlu; çıkarımda kapalı | ✅ | `Spec::noisy` |
| 14.2 kapasite: GShard grup başına `C = max(min_cap, round(k · N_g / E · cf))`, drop ya da reroute | ✅ | `Overflow::{Drop,Reroute}` |
| 14.B sayım sıralı permütasyon (histogram + cumsum), belirlenimci; geri geçiş aynı sırayı tersine kullanır | ✅ | `Routing::offsets / perm_token / row` |
| 14.B gruplu GEMM, CPU: uzmanlar iş parçacıklarına dağılır, yarış yok, sonuç iş parçacığı sayısından bağımsız | ✅ | `experts_forward/backward` |
| 14.B gruplu GEMM, GPU: uzman başına cuBLASLt döngüsü ve dolgulu strided batched | ✅ yalnız ölçüm | `tests/cuda/test_phase14_moe_cuda.cpp` |
| 14.C `lm` yapılandırıcısı: `moe_*` alanları, `moe_every`, `moe-124m` ön tanımlısı, JSON, NXS tür 36 `MoEFFN` | ✅ | `lm::LmConfig`, `lm::moe_ffn` |
| 14.D EP: uzmanlar rank'lere bitişik bloklar halinde, I64 sayım a2a, satır a2a(v), yerel GEMM, geri a2a; aux/z küresel | ✅ CPU | `dist::MoeExpertParallel` |
| 14.E paylaşımlı uzman (DeepSeek tarzı yoğun MLP + MoE) | ✅ | `moe_shared_expert` |
| 14.F BF16 / FP8-E4M3 uzman ağırlığı (uzman başına ölçek, yalnız ileri) | ✅ | `quantize_experts` |
| 14.F uzman offload (yalnız aktif uzman kartta) | ⛔ yok | — |
| 14.4 EP + TP + DP birleşimi | ⛔ yok (EP tek başına) | — |

## Testler (spec adlarıyla)

| Test | Sonuç |
|---|---|
| `test_moe_router_deterministic` | ✅ Aynı giriş → aynı top-k, kapı, kapasite ataması; gürültülü modda da (aynı tohum). Grup başına uzman ≤ C. Düşen sayısı = atama − kabul. Reroute düşmeyi 58'den 48'e indirdi (29 yeniden yönlendirme). Çıkarımda gürültü tohumdan bağımsız. |
| `test_moe_k1_equals_dense` | ✅ E=1, k=1 `lm` modeli yoğun SwiGLU modeline eşit: kayıp ve **tüm gradyanlar bit bit aynı**. Yönlendirici gradyanı tam 0 (kapı = 1). |
| `test_moe_grad_numerics` | ✅ Optimize yolu, bağımsız double referansla karşılaştırıldı (token başına, permütasyon yok). Göreli hatalar: y 2.4e-7, dx 1.9e-7, drouter 2.9e-7, dup 2.4e-7, ddown 1.0e-7, aux 1.3e-8, z 4.1e-8 (sınır 1e-5). Drop ve reroute ile ayrı ayrı. PyTorch MoE ile karşılaştırılmadı; spec "veya elle formül" diyor. |
| `test_grouped_gemm_vs_loop` | ✅ Paralel gruplu GEMM = sıralı uzman döngüsü (**bit bit**), **3.11× hızlı**. Token başına döngüyle de fark 0. |
| `test_load_balance_in_range` | ✅ 300 adım, bayt TinyStories, son 20 adım. aux kaybıyla uzman payı **0.066–0.164** (alt sınır 0.5/E = 0.0625). aux'suz: **min 0.000** (ölü uzman var), max 0.294. |
| `test_ep_equivalence` (1 rank ile 2 / 4 / 8 rank) | ✅ y, dx, dup, ddown **bit bit aynı**. drouter ≤ 3.2e-7 (rank'ler arası all-reduce toplama sırası). aux+z aynı. |
| `test_moe_quantized_experts` | ✅ BF16: bellek 2.00× az, çıkış hatası 2.8e-3. FP8-E4M3: bellek 4.00× az, çıkış hatası 4.0e-2. |
| `test_moe_config_roundtrip` | ✅ `moe-124m`: toplam 123.5M, aktif 38.0M (%31). JSON gidiş-dönüş tamam. Paylaşımlı uzman + gürültü + `moe_every=2` ile `count_parameters` doğru. NXS kayıt/yükleme sonrası kayıp aynı. |
| `test_moe_grouped_gemm_cuda` | ✅ Döngü = batched (fark 1.5e-5, BF16 girdi, FP32 çıkış). CPU double referansına göre 8.4e-7. compute-sanitizer 0 hata. |

Eşdeğerliğin dayanağı: ep rank'li EP, rank başına 1 grup. Tek rankta aynı tokenlar arka arkaya ve `groups = ep`.
Kapasite ve satır sırası (uzman, grup, kabul) iki durumda birebir aynı. Bu yüzden toplama sırası değişmez.

## Benchmark'lar

### MoE-124M ve aynı aktif FLOP'lu yoğun model (CPU, bayt TinyStories)

Ayar: d 512, 5 katman, 8 kafa, sözlük 256. MoE: E=8, k=2, uzman F 1856, aux α 0.01. Yoğun model: d_ff = k · F = 3712, yani token
başına aynı MLP FLOP'u. AdamW lr 1e-3 (β2 0.95, wd 0.1, kırpma 1.0). B=8, T=128, 150 adım, aynı veri sırası ve tohum.

| Model | Toplam param | Aktif param | Kayıp ilk → son 20 ort. | ms/adım (temiz koşu) | tok/s |
|---|---|---|---|---|---|
| **MoE-124M** | 119.4M | 33.9M | 5.70 → **1.931** | 3 547 | 289 |
| Yoğun, aynı aktif | 33.9M | 33.9M | 5.66 → 2.037 | 9 024 | 113 |

- **Kayıp:** Aynı aktif hesapla MoE 150 adımda %5 daha düşük kayıp verdi. Tek tohum, kısa koşu; istatistiksel güven aralığı yok.
- **Uzman kullanımı (150 adım, son 20):** pay 0.062–0.163, 5 katmanda boş uzman yok.
- **Hız:** MoE CPU'da 2.5× hızlı. Bu **MoE'nin değil, çekirdeklerin farkıdır.** Yoğun MLP genel autograd matmul ile çalışıyor
  (~23 GFLOP/s). MoE uzmanları kendi paralel gruplu çekirdeğiyle çalışıyor (~59 GFLOP/s). Aktif FLOP aynı.
  Adil bir hız karşılaştırması için aynı GEMM'in iki yolda da kullanılması gerekiyor. Bu yapılmadı.
- **Aynı toplam parametreli yoğun modele göre FLOP oranı:** 119M yoğun modelin token başına FLOP'u, MoE'nin 3.5 katı
  (aktif / toplam = %28). Bu hesap muhasebeden geliyor, ölçülmedi.
- 150 adımlık koşu ve bazı testler bir süre aynı anda çalıştı. Bu yüzden eğri o koşudan, ms/adım ayrı 12 adımlık temiz koşudan alındı.

### Gruplu GEMM

| Yol | Süre | Verim |
|---|---|---|
| CPU, sıralı uzman döngüsü (E=8, 8192 satır, d 256, F 512) | 135.5 ms | 47.5 GFLOP/s |
| **CPU, paralel gruplu** | **43.5 ms** | **148.1 GFLOP/s (3.11×)** |
| CPU, token başına döngü (tahmini, 256 satırdan) | ~860 ms | — |
| GPU, uzman başına cuBLASLt döngüsü (16 çağrı; moe-124m boyutları, 16 384 satır, BF16) | 4.376 ms | 21.35 TFLOPS |
| **GPU, dolgulu strided batched (2 çağrı)** | **4.240 ms** | **22.03 TFLOPS** yararlı (23.16 dolgu dahil), 1.03× |
| GPU, CUTLASS grouped | — | ⛔ ölçülmedi (CUTLASS bu depoda yok) |

- GPU dolgusu: uzman başına satır 1 918–2 153. Cmax'a dolgu yapınca yararlı / dolgulu FLOP oranı 0.95.
  Dolgu kopyası süreye dahil değil (host'ta yapıldı). Dengesiz yükte batched yolun kaybı büyür.
- Uçtan uca GPU MoE eğitimi yok. `cuda::gpt_backend` MoE desteklemiyor; açık hata veriyor.

### Uzman paralelliği: all-to-all (CPU, thread ProcessGroup, tek makine)

Ayar: moe-124m katmanı (d 512, E 8, F 1856), rank başına 4 096 token, ileri + geri, 3 tekrarın en iyisi.

| ep | Adım | a2a | Hesap | a2a payı | Gönderilen | a2a bant genişliği |
|---|---|---|---|---|---|---|
| 2 | 3 493 ms | 82 ms | 3 344 ms | %2.4 | 66.7 MB | 0.81 GB/s |
| 4 | 4 567 ms | 117 ms | 4 340 ms | %2.6 | 201.5 MB | 1.72 GB/s |
| 8 | 8 266 ms | 514 ms | 7 953 ms | %6.2 | 470.0 MB | 0.91 GB/s |

- **Hesapla örtüşme yok.** a2a ve hesap sırayla çalışıyor; örtüşme oranı 0. Spec bunu istiyor, yapılmadı.
- Tüm rank'ler aynı 16 iş parçacıklı CPU'yu paylaşıyor. Bu yüzden ep arttıkça adım süresi uzuyor. Gerçek çok cihazlı
  ölçekleme değil, iletişim yolunun doğruluk ve maliyet ölçümü.
- Bant genişliği bellek kopyası (thread PG) sınırında. NCCL / ağ ölçülemedi (tek GPU, Windows).

## Regresyon

- CPU ctest: **123/123** (FAZ 14 öncesi 115).
- ASan: FAZ 14'ün 8 testi temiz (`asan_ctest.txt`).
- GPU: 39 testin 38'i ilk koşuda geçti. `test_overlap_occurs_cuda` (FAZ 10 DDP) tam koşuda yük altında kova zaman aşımı
  verdi. Tek başına tekrar koşulunca geçti (1.7 s). FAZ 14 bu yola dokunmuyor, ama test zamanlamaya duyarlı; not edildi.
- `test_moe_grouped_gemm_cuda` compute-sanitizer memcheck: 0 hata.

## Gerideyiz / yapılmadı

- **Mixtral 8×7B ve yoğun 7B karşılaştırması ⛔:** donanım yetmiyor. Yerine küçük ölçekte MoE ve aynı aktif yoğun model ölçüldü.
- **GPU MoE eğitimi yok.** GPU'da yalnız uzman GEMM'leri ölçüldü. Permütasyon, yönlendirici ve a2a çekirdekleri GPU'da yok.
- **CUTLASS grouped GEMM ölçülmedi.** Batched yol dolgu istiyor (dolgusuz gerçek grouped yok).
- **a2a / hesap örtüşmesi yok.** EP + TP + DP birleşimi yok. EP, DeviceMesh ekseni olarak değil, doğrudan bir `GroupPtr` ile
  kuruluyor.
- **Uzman offload yok.** BF16/FP8 uzman ağırlığı yalnız ileri geçişte (çıkarım). Dar ağırlıkla eğitim yok.
- **CPU hız karşılaştırması adil değil:** yoğun yol daha yavaş bir GEMM kullanıyor (yukarıda).
- Nsight zaman çizelgesiyle boşta uzman süresi ölçülmedi. Kullanım histogramı ve boşta uzman sayısı metrik olarak var
  (`MoeLayerStats`, `Routing::idle_experts`).
- z-kaybı: spec formülü `(1/E) Σ_i log(Σ_j exp)²`. Uygulanan, ST-MoE makalesindeki gibi token ortalaması: `mean_t lse_t²`.
- MoE, `trainer.parallel` (FAZ 12) ve `nexus::compile` (FAZ 13) ile birlikte kullanılamıyor; ikisi de açık hata veriyor.

## Dosyalar

- Kod:
  - `include/nexus/moe/moe.hpp`, `src/moe/moe.cpp`: yönlendirici, kapasite, permütasyon, gruplu GEMM, kayıplar, geri geçiş, referans, nicemleme.
  - `include/nexus/lm/moe_layer.hpp`, `src/lm/moe_layer.cpp`: autograd düğümleri.
  - `lm/config`, `lm/model`: MoE blokları, `moe-124m`, NXS `MoEFFN`.
  - `include/nexus/dist/moe_ep.hpp`, `src/dist/moe_ep.cpp`: uzman paralelliği.
- Testler: `tests/parity/test_phase14_moe.cpp`, `tests/cuda/test_phase14_moe_cuda.cpp`.
- Ham çıktılar:
  - `train_moe_vs_dense.log`, `train_speed_clean.jsonl`;
  - `grouped_gemm_cpu.log`, `grouped_gemm_cuda.log`;
  - `ep_bench.jsonl`;
  - `tests_cpu.log`, `ctest_cpu.txt`, `ctest_cuda.txt`, `asan_ctest.txt`, `compute_sanitizer.txt`.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
