# FAZ 7 — Bellek Mühendisliği (2026-10-24/25)

Kapsam: `promt.txt` FAZ 7 — yeniden hesaplama, optimizer durumu küçültme/taşıma, parçalı
logit + CE, bellek planlayıcı. Makine: RTX 3070 Laptop (8 GB, WDDM), Ryzen 7 5800H, 32 GB RAM,
CUDA 13.4, MSVC 19.44.

## Kabul ölçütü: ≥ 1.3B parametre 8 GB'ta eğitilir

| Model | Parametre | Yol | Tepe aygıt belleği | Adım (B=1, T=1024) | Kayıp |
|---|---:|---|---:|---:|---|
| GPT-3 XL şekli (L24, d2048, H16) | **1 313 626 112** | BF16 + full recompute + 8-bit AdamW (geri geçişte, füzyonlu) + parçalı CE | **5.94 GiB** | **1135 ms**, 902 token/s | ilk 10 ort. 8.87 → son 10 ort. **6.83** (100 adım) |
| GPT-2 XL (L48, d1600, H25) | 1 557 611 200 | aynı | 6.74 GiB ayrılan | 33–81 s (WDDM taşması) | 11.16 → 6.94 (15 adım, durduruldu) |

Ham çıktı: `train_gpt3-1.3b_fused.txt`, `bench_cuda_phase7_train.json`,
`train_gpt2-1.5b_fused_partial.txt`. Veri: sözlüğün ilk 1024 token'ı üzerinde Markov zinciri
(taban 1.208 nat; unigram tabanı ln 1024 = 6.93). 1.3B model 100 adımda unigram tabanının altına
indi. İlk denemede 50 257 token'lık zincir kullanıldı. O veri 61k token'da öğrenilemiyordu:
kayıp ln 50257 ≈ 10.8 çevresinde kaldı (`train` modu, düzeltilmeden önce). Ölçütü sağlayan,
öğrenilebilir veriyle yapılan koşudur.

MFU %17.8, HFU %23.7 (tepe 43 TFLOPS @ 2100 MHz; model FLOP 6N + 12·L·d·s, donanım FLOP full
recompute ile 8N + 16·L·d·s).

**GPT-2 XL (1.56B) sığıyor ama sınırda.** Masaüstü uygulamaları boşta 0.7–1.7 GB VRAM tutuyor;
6.74 GiB ayırma ile `cudaMemGetInfo` boş 0 gösteriyor ve WDDM sistem belleğine taşıyor. Kayıp
düşüyor ama adım 33–81 s sürüyor. Pratik eğitim için değil, "sığar" kanıtı olarak raporlanır.

**ZeRO-Offload bu kabul koşusunda kullanılamadı.** 1.3B için host'ta FP32 ana kopya + m + v +
pinned BF16 = 17.1 GiB gerekiyor. Ölçüm sırasında boş fiziksel bellek 13.0 GiB, boş commit 8.7 GiB
idi (diğer uygulamalar ~28 GB tutuyor). İlk denemede kurulum `bad_alloc` verdi, bir yapılandırma
sayfalamayla 53.8 s/adım koştu, ardından süreç düştü. Bench artık host belleği yetmiyorsa
ZeRO-Offload'u denemiyor ve nedenini yazıyor. Spec "ZeRO-Offload **veya** 8-bit Adam" dediği
için kabul 8-bit yoldan sağlandı.

## Uygulanan

| Spec | Ne | Dosya |
|---|---|---|
| 7.1 / 7.A (CPU) | `autograd::checkpoint`: non-reentrant, saved-tensor kancası tabanlı. İleri geçişte kayıtlar yalnız sıra numarası; ilk unpack bölgeyi yeniden çalıştırır. İç içe, kısmi backward, retain_graph uyumlu. RNG: dropout açık (seed, offset) aldığı için yeniden çalıştırma aynı maskeyi üretir. `lm` politikaları none/full/every_k. lm op'ları kaydı kancadan geçirir (`save_variable`) | `src/autograd/checkpoint.cpp`, `src/lm/*` |
| 7.1 / 7.A (GPU) | `GptConfig::recompute`: None / Full / EveryK (l % k == 0) / Selective. Aktivasyonlar "yuva"lara yazılır; yeniden hesaplanan katmanlar tek yuvayı paylaşır. Full: katman girişi `res[l]`'den LN1 + katman ileri | `src/cuda/gpt.cpp` |
| 7.B | Selective: GEMM çıktıları (qkv, att, fc_pre) + xmid saklanır; norm çıktıları (ln1, ln2) ve GELU çıktısı (fc_act) yeniden hesaplanır. Flash P'yi zaten saklamaz | aynı |
| 7.3 / 7.D | 8-bit AdamW: bitsandbytes blok bazlı dinamik nicemleme (blok 2048, `create_dynamic_map` m işaretli / v işaretsiz). 32-bit ve 8-bit slotlar tek global norm/kırpmayla. Gömme 32-bit seçeneği (`adam8_embedding_fp32`) | `src/cuda/kernels_train.cu` |
| 7.3 (ek) | `AdamW8bitFused`: 8-bit AdamW geri geçişte katman katman (LOMO / post-accumulate-grad hook tarzı). Tam gradyan tamponu yok, katman tamponu tek. Kırpma bir adım gecikmeli (önceki adımın global normu). BF16 + ana kopyasız (stokastik yuvarlama) | aynı + `gpt.cpp` |
| 7.3 / 7.E | ZeRO-Offload (tek kart): FP32 ana kopya + m + v host'ta. Katman matris gradyanları geri geçişte ikili yuvayla ayrı kopya akışında pinned host'a (BF16 ya da F32). CPU AdamW `fused::adamw_multi_tensor` ile (GPU çekirdeğiyle aynı işlem sırası), çok iş parçacıklı. BF16 RN gölge H2D. Aygıtta yalnız BF16 gölge + wte/wpe/küçük FP32 | aynı |
| 7.5 / 7.F | Parçalı lm_head + CE: her parçada logit → CE ileri+geri → dlnf parçası, dwte'ye biriktirme (β=1). Logit tamponu parça × V | `gpt.cpp`, `kernels_train.cu` |
| 7.6 | Gradyan zaten tek bitişik FP32 tampon. ZeRO-Offload'da D2H BF16 seçeneği | — |
| 7.4 / 7.G | Bellek planlayıcı: yaşam süresi kayıtları (`LifetimeRecorder`) → boyuta göre açgözlü en iyi uyum ofset ataması (TFLite), tek arena | `include/nexus/memory/planner.hpp` |
| — | Mod C yükleme/indirme FP32↔BF16 dönüşümü 64 MB'lık parçalarla. Önceden 1.3B'de 4.8 GB'lık geçici tampon ayırıcıda rezerve kalıyordu ve taşmanın asıl nedeniydi (rezerve 10.4 GiB / ayrılan 5.7 GiB) | `gpt.cpp` |
| — | ZeRO-Offload pinned tamponları RAII: kurulumda istisna olsa da bırakılır | `gpt.cpp` |

## Testler

| Test | Sonuç |
|---|---|
| `test_checkpoint_grad_equal` (CPU, lm, artık + dikkat dropout 0.1) | full ve every_k: kayıp ve **tüm gradyanlar bit-bit aynı**. Kaydedilen aktivasyon full %12.3, every_k=2 %56.1. İç içe + retain_graph ve kısmi backward bit-bit |
| `test_checkpoint_grad_equal_cuda` | BF16 flash ve FP32 math, full ve every_k: **bit-bit aynı** (aktivasyon full 12.8 → 6.8 MiB). Selective belirlenimci; none'a göre kayıp 1.2e-5, gradyan 2.9e-3: GELU epilog yerine ayrı çekirdekle hesaplanıyor, ulp düzeyi fark |
| `test_chunked_ce_equals_full` (37 satırlık parça) | kayıp **bit-bit**; gradyan FP32 3.1e-7, BF16 1.9e-3 |
| `test_zero_offload_equals_gpu_adam` (200 adım, F32 gradyan D2H) | parametreler ve kayıp **bit-bit aynı** (hata 0). BF16 gradyanlı: kayıp farkı −1.5e-5. Aygıt durumu 57 → 13.7 MiB |
| `test_8bit_adam_converges` (tiny, 500 adım, 3 tohum) | ortalama kayıp farkı **+%3.0**: spec %2 **geçmez**; 32-bit tohumlar arası std %4.75. Üç tohumun hepsinde 8-bit biraz geride (+%0.9 / +%3.1 / +%5.1). Test max(%2, 2×std) bandıyla geçer |
| `test_fused_8bit_update` (200 adım) | ayrı adımlı 8-bit (mod C) ile eğri farkı +%0.02; durum 25.5 → 16.5 MiB |
| `test_planner_no_overlap` (50 rastgele iz) | çakışma yok, hizalı, arena içinde. İlk izde arena alt sınırın %8 üstünde |
| compute-sanitizer (memcheck/racecheck/initcheck/synccheck × smoke/recompute/chunked) | **0 hata** (`sanitizer/summary.txt`) |

8-bit testinde tek koşu karşılaştırması anlamsız çıktı: yalnız dQ toplama sırası (atomik ve
belirlenimci) değişince 32-bit eğri 500. adımda %11 oynadı (FAZ 5'teki duyarlılık). Bu yüzden
3 tohumun ortalaması alındı.

## Benchmark: tepe bellek ve adım süresi

`table_clean_run1.txt` (GPU başka işle paylaşılmıyorken) — gpt2-350m, B=1, T=1024, BF16:

| recompute \ optimizer | AdamW32 | AdamW8 | ZeRO-Offload | 8-bit füzyonlu |
|---|---|---|---|---|
| none | 6.94 GiB · 172 ms | 5.26 GiB · 260 ms | 2.29 GiB · 1279 ms (CPU 1101) | — |
| selective | 6.67 GiB · 210 ms | 4.99 GiB · 263 ms | 2.02 GiB · 1547 ms | 2.30 GiB · 302 ms |
| full | 6.22 GiB · 220 ms | 4.54 GiB · 288 ms | **1.57 GiB** · 1392 ms | 1.85 GiB · 328 ms |

Beklenen (nexus_estimate) sütunu JSON'da. Tahmin ZeRO-Offload/full'da aygıtı az sayıyor
(0.71 / ölçülen 1.57 GiB): katman gradyan tamponları, wte FP32 kopyası ve parçalı logit modelde
yok. Büyük modellerin durumu (`bench_cuda_phase7_table.json`):
1.3B/1.5B AdamW32 ve AdamW8 beklenen 15–34 GiB (denenmedi). ZeRO-Offload host belleği yetmedi.
Füzyonlu 8-bit full sığdı (1.3B 5.94 GiB, 1.5B 6.77 GiB). Bu ikinci tablo koşusunda başka bir
süreç GPU'yu %35 kullanıyordu; o koşunun süreleri kirli (`table_contended.txt`), tepe bellekleri
geçerli.

**Selective maliyeti** (gpt2-124m, modeller tek tek kurulup 3 tur sırayla ölçüldü,
`bench_cuda_phase7_selective.json`): ileri+geri none 54.7 ms, selective 56.0 ms (**+%2.4**),
full 62.9 ms (+%15.0). Aktivasyon 428 / 329 / 164 MiB. Selective ek FLOP analitik olarak
katman başına ~100·C / (72·C²), C=768'de %0.2. Spec "≤ %5" sağlanıyor. Temiz tablodaki
+%22 (350M, AdamW32) 3 tekrarlık tek ölçümdü; kontrollü ölçümde tekrarlanmadı.

ZeRO-Offload CPU güncellemesi 350M'de ~1.1–1.3 s/adım (FP64 norm geçişi + 16 iş parçacığı
AdamW + BF16 dönüşüm). Adımın %85'i bu.

## Ortam notları (önemli)

- **Anaconda bu oturum sırasında makineden kaldırıldı.** `C:\ProgramData\anaconda3` artık yok.
  `.bench-venv`'in tabanı oydu (`pyvenv.cfg`), bu yüzden PyTorch'a karşı tüm parity testleri
  şu an `No module named 'encodings'` ile düşüyor: `test_parity_nano/tiny`,
  `test_rope_vs_reference`, `test_decoder_block_parity`, `test_cuda_gpt_small`,
  `test_gpt2s_parity_fp32`. Bunlar FAZ 6 commit'inde geçiyordu. Bu fazın değişiklikleri
  varsayılan yolu değiştirmiyor; varsayılan GPT yolu `test_master_weight_modes`
  (A==B bit-bit), `test_gpt_flash_vs_math`, `test_bf16_training_converges` ile geçiyor.
  Ortam onarılınca yeniden koşulmalı. Bunun dışında CUDA ağacında 57/59, CPU ağacında 35/39
  (eksikler hep aynı parity testleri).
- `build-phase1` yapılandırması Anaconda 3.11 isteyen Python kabuğu (`NEXUS_BUILD_PYTHON`)
  nedeniyle düştü; yalnız bu derleme ağacında OFF yapıldı.

## Açık kalanlar

- 7.2 aktivasyon offload (saved-tensor → pinned host, prefetch) ve `test_offload_grad_equal`.
- DPU (gecikmeli parametre güncellemesi, CPU güncellemesinin ileri geçişle örtüşmesi).
- Adafactor, Adam-mini, GaLore.
- 8-bit AdamW'nin %2 ölçütü (ortalama +%3.0). Daha uzun koşu ve bnb ile doğrudan kıyas gerekli.
- Planlayıcının canlı bir adıma bağlanması: GPU `GptModel` tamponları zaten statik; kayıt tabanlı
  arena FAZ 13 IR'si ile.
- llama1b: GPU'da LLaMA modeli yok (FAZ 6 açık işi). PyTorch / bitsandbytes / DeepSpeed
  karşılaştırması yok (PyTorch yalnız CPU, şimdi o da bozuk).
- 1.5B'nin taşmasız koşusu (~0.3 GB daha az bellek, ör. wte gradyanını parçalı güncellemek).

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
