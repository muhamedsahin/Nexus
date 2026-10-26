# FAZ 16 — Çıkarım: Sayfalı KV, Artımlı Çözme, Örnekleme, Spekülatif Çözme, Nicemleme, Sürekli Batch (2026-10-27)

Kapsam: `promt.txt` FAZ 16 (DUVAR 5, DUVAR 14). Kullanıcı kararıyla kapsam:
- **GPU:** GPT-2 124M (HF ağırlıkları) için tam çıkarım motoru (`nexus::cuda::infer`) ve sunucu (`nexus::cuda::serve`).
- **CPU:** `lm::LanguageModel` (LLaMA/GPT tipi tüm varyantlar) için KV önbellekli artımlı çözme.
- 7B, llama.cpp ve vLLM indirilmedi; karşılaştırmaları yok (⛔, aşağıda).

Makine: RTX 3070 Laptop 8 GB (sm_86), Ryzen 7 5800H, Windows 11, **fişte** (`Win32_Battery` 2). Pilde yapılan ön ölçümler atıldı.
Referans: PyTorch 2.14.1+cu132, aynı ağırlıklar, FP16.

## Ne yapıldı

| Alt adım | Durum | Nerede |
|---|---|---|
| 16.A Sayfalı KV önbelleği | ✅ Katman başına `[blok, 16, H, D]`. Blok tahsis edici + serbest liste + referans sayacı. **Copy-on-write** dal (`fork_sequence`): paylaşılan son bloğa yazılacağında kopyalanır. Biten sıra blokları geri verir | `infer::Engine` |
| 16.A KV INT8 | ✅ (yuva, baş) başına simetrik ölçek amax/127. Token başına 36.9 → 19.6 KB | `KvType::INT8` |
| 16.B Artımlı çözme | ✅ Tek adımda karışık yığın: taze istemler FlashAttention-2 varlen (FP16) ile; decode / devam / spekülatif doğrulama yeni **sayfalı dikkat çekirdeği** ile (her sorgu kendi konumuna kadar nedensel) | `forward(chunks)` |
| 16.B Decode CUDA Graph | ✅ Statik arena, boyut kovaları 1, 2, 4, …, max_seqs. Blok tablosu / konum / yuva sabit aygıt tamponlarında; her adım yalnız kopyala + oynat. Dolgu satırları ayrılmış blok 0'a yazar | `EngineConfig::cuda_graphs` |
| 16.C Örnekleme | ✅ Tek çekirdek: ceza → sıcaklık → top-k / top-p / min_p eşikleri (ikiye bölme araması) → çok terimli çekiliş (Philox4x32-10) ya da greedy. Cezalar: tekrar (HF), frekans + varlık (OpenAI; yalnız çıktı tokenları). CPU senkronu yalnız örnek tokenlar için | `Engine::sample` |
| 16.D Spekülatif çözme | ✅ Greedy doğrulama. Taslak = aynı modelin ilk k katmanı (aynı sözlük, ek model yok). γ aday tek hedef geçişinde doğrulanır; ilk uyuşmazlıkta durulur; reddedilen KV atılır | `speculative_generate` |
| 16.E Nicemleme | ✅ W8A8 (kanal başına ağırlık, token başına dinamik aktivasyon, cuBLASLt INT8, **SmoothQuant** α=0.5). W4A16 (grup 128, asimetrik; **AWQ** ölçek araması, decode için kendi GEMV çekirdeği). KV INT8 | `QuantOptions` |
| 16.E FP8 | ⛔ sm_89+ gerekiyor; bu kart sm_86 | — |
| 16.F Sürekli batch sunucusu | ✅ İstek kuyruğu + zamanlayıcı. Prefill ve decode aynı forward'da. Token / blok bütçesi. KV yetmezse **preempt** (yeniden prefill). Metrikler: TTFT, TPOT, toplam token/s, eşzamanlı istek | `serve::Server` |
| CPU artımlı çözme | ✅ `ForwardOptions::cache` (`lm::KvCache`). Desteklenenler: RoPE (kısmi), öğrenilen / sinüzoidal konum, ALiBi, kayan pencere, GQA, qk_norm, paralel blok, softcap, MoE | `lm::LanguageModel` |

**Doğruluk için iki tasarım kararı:**
- **FP32 doğrulama kipi.** FP16'da artımlı ve tam yol farklı çekirdeklerden geçtiği için fark ~4e-4 oluyor. Spec'in ≤1e-4 sınırı bu yüzden FP32 kipinde ölçülüyor (her şey FP32, TF32 kapalı). FP16 hatası ayrıca raporlanıyor.
- **İnce GEMM çekirdeği (m ≤ 16 satır).** Decode ve doğrulama için:
  - Ağırlık `[n, k]` düzeninde (cuBLASLt prefill'de aynı kopyayı `trans_b` ile kullanıyor).
  - Çıkış başına bir warp, 16 baytlık okumalar, sabit indirgeme sırası.
  - Sonuç: satır sonucu m'den **bağımsız**. Spekülatif çözme greedy ile bu sayede birebir aynı.

## Testler

| Test | Sonuç |
|---|---|
| `test_kv_cache_matches_full` (GPU) | ✅ FP32: artımlı (8 + 42 decode) vs tam ileri **6.4e-6**. PyTorch FP64'e göre tam 6.1e-6, artımlı 7.1e-7. FP16: referansa göre 3.9e-4 / 5.5e-4, artımlı vs tam 4.2e-4; argmax 50/50 aynı. |
| `test_kv_cache_matches_full_cpu` | ✅ 7 varyantın hepsinde artımlı (5 + 7'lik parça + tek tek) = tam ileri, **fark 0**: GPT-2 tipi, LLaMA GQA, ALiBi + pencere, qk_norm + paralel blok, sinüzoidal, kısmi RoPE + softcap, MoE. Önbellekli greedy = tam yeniden hesap (48 token), 3.6× hızlı. |
| `test_paged_cache_equivalence` | ✅ Sayfalı (16'lık bloklar, iki sıra iç içe büyüyor) = bitişik (tek blok) **bit bit**. CoW dal = bağımsız sıra, bit bit. Tüm bloklar geri dönüyor. compute-sanitizer 0 hata. |
| `test_sampling_distribution` | ✅ 7 ayar (T=1, T=0.7, top-k 40, top-p 0.9, min_p 0.05, karma, tekrar cezası 1.3) × 20 000 örnek. Kesin filtreli dağılıma göre ki-kare z ∈ [−1.1, 1.6]; destek dışı örnek 0; greedy = argmax. |
| `test_speculative_exact` | ✅ 9 ayarın (taslak 4/6/8 katman × γ 2/4/6) hepsinde 128 token greedy ile **birebir aynı**. |
| `test_quant_ppl` | ✅ Aşağıdaki tablo. |
| `test_decode_cuda_graph` | ✅ Graph = eager (96 token aynı). 5 sıralık decode kova 8 ile çalışıyor. compute-sanitizer 0 hata. |
| `test_continuous_batching` | ✅ 16 istek sunucuda = tek tek üretim (**16/16 birebir**). Toplam süre sıralıdan 7.0× kısa. 39 blokluk havuzda preempt + yeniden prefill ile 8/8 istek tamamlandı, bloklar geri döndü. |

### Nicemleme perplexity (TinyStories valid, 30 208 token, pencere 512)

| Yöntem | ppl | Değişim | Ağırlık |
|---|---|---|---|
| FP16 | 11.389 | — | 248.7 MB |
| W8A8 (ölçeksiz) | 11.466 | +%0.67 | 164.1 MB |
| **W8A8 SmoothQuant** | 11.436 | **+%0.41** (eşik %5) | 164.1 MB |
| W4A16 g128 RTN | 12.524 | +%9.96 | 124.6 MB |
| **W4A16 g128 AWQ** | 12.110 | **+%6.33** (eşik %15) | 124.6 MB |
| KV INT8 (64 prefill + 192 decode protokolü) | 14.479 | **−%0.01** (eşik %1) | KV 36.9 → 19.6 KB/token |

- Ağırlık boyutuna FP16 gömme / lm_head (77 MB) dahil; o kısım nicemlenmiyor.
- AWQ kalibrasyonu (512 token, α ızgarası) CPU'da 24 s sürüyor (fişte; pilde 49 s).

## Benchmark'lar

### Tek istek TTFT / TPOT (128 token üretim, greedy, istemi rastgele token)

| Yol | İstem 32: TTFT / TPOT | İstem 128 | İstem 512 |
|---|---|---|---|
| **Nexus FP16** (decode CUDA Graph) | 3.2 ms / **1.51 ms** | 2.5 / **1.70** | 5.5 / **2.32** |
| Nexus W8A8 | 3.7 / 1.63 | 3.9 / 1.68 | 4.5 / 2.63 |
| Nexus W4A16 | 6.4 / **1.26** | 7.4 / 1.56 | 11.2 / 2.37 |
| PyTorch eager (KV önbellekli kendi generate) | 12.8 / 12.78 | 15.5 / 13.64 | 12.2 / 11.24 |
| PyTorch `torch.compile(reduce-overhead)` decode, statik KV | 13.8 / 2.00 | 12.3 / 2.17 | 16.5 / 3.77 |

- **TPOT:** Nexus FP16, PyTorch compile'dan %25–38 hızlı (1.51 vs 2.00, 1.70 vs 2.17, 2.32 vs 3.77 ms).
  - PyTorch statik önbelleği maskeyle tümüyle okuyor (gpt-fast deseni).
  - Nexus yalnız geçerli uzunluğu okuyor; uzun istemde farkın bir kısmı bu.
- **Decode CUDA Graph:** TPOT 2.12 → 1.52 ms (%28).
- **W4A16** decode'da %17 hızlı (bellek bant genişliği sınırlı GEMV). Prefill'de yavaş: ağırlık her adımda FP16'ya açılıyor.
- **W8A8 decode'u hızlandırmıyor ⛔.** INT8 için ince GEMM yok; küçük m'de nicemleme + cuBLASLt INT8 + açma ek yükü kazancı yiyor.
- **Bant genişliği sınırına uzaklık:** FP16 TPOT 1.5 ms. Teorik alt sınır yaklaşık 0.6 ms (250 MB ağırlık / ~400 GB/s); arada 2.5× var.

### Spekülatif çözme (128 token, γ=4; ısınmış ölçüm)

| Taslak | Kabul | Adım başına token | Süre | Greedy |
|---|---|---|---|---|
| İlk 4 katman | %5 | 1.21 | 652 ms | 201 ms |
| İlk 6 katman | %44 | 2.70 | 339 ms | 201 ms |

- **⛔ Spekülatif çözme greedy'den yavaş.** Adım başına >1 token hedefi tutuyor (2.70), ama:
  - Taslak, hedefin ilk yarısı kadar maliyetli (ayrı küçük model yok).
  - Taslak adımları CUDA Graph'sız ve her token için senkron.

  Hızlanma için ayrı, çok küçük bir taslak model ve graph'lı taslak döngüsü gerekiyor.

### Sürekli batch: eşzamanlı istek → toplam token/s (istem 128, üretim 128, greedy)

| Eşzamanlı | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Toplam tok/s | 601 | 1 153 | 2 228 | 3 653 | 4 301 | 10 332 | **14 448** |
| TPOT (ms) | 1.63 | 1.67 | 1.77 | 2.11 | 3.55 | 2.83 | 3.97 |

- **Kabul ölçütü:** sürekli batch toplam token/s'i ayrı isteklerin toplamından yüksek. ✅ 64 istekte 24×.
- **16 istek ayrıksı:** m=16'da ince GEMM satır başına ağırlığı yeniden okuyor. cuBLASLt m>16'da devreye giriyor ve daha hızlı. Eşik ayarlanmadı.

## Gerideyiz / yapılmadı

- **7B BF16/INT4, llama.cpp ve vLLM karşılaştırması ⛔:** indirme yapılmadı (kullanıcı kararı). vLLM Windows'ta resmi olarak yok.
  "7B INT4 TPOT ≤ llama.cpp" kabul ölçütü **ölçülmedi**.
- **GPU'da LLaMA tipi model yolu yok.** GPU motoru yalnız GPT-2 mimarisini (LayerNorm, GELU, öğrenilen konum, head_dim 64) destekliyor; CPU yolu tüm varyantları destekliyor.
- **FP8 ⛔** (sm_86).
- **Spekülatif:**
  - Ağaç doğrulama yok.
  - Örneklemeli spekülatif (reddetme örneklemesi) yok; yalnız greedy.
  - Greedy'den yavaş.
- **W8A8 decode hızı kazancı yok.**
- **Sunucu:**
  - Önek (prefix) önbelleği ve parçalı (chunked) prefill yok.
  - Preempt sonrası yeniden prefill edilen üretilmiş tokenlar frekans/varlık cezasına "istem" olarak sayılıyor (küçük sapma).
- **KV INT8:** taze prefill'in kendi dikkat hesabı FP16 K/V ile yapılıyor; nicemleme yalnız sonraki okumalarda etkili.
- **TTFT:** istem prefill'i graph'sız. FlashAttention + cuBLASLt kullanıyor; boyut kovası yok.

## Dosyalar

- GPU:
  - `include/nexus/cuda/infer.hpp`, `src/cuda/infer.cpp`;
  - `src/cuda/infer_kernels.{hpp,cu}`: sayfalı dikkat, ince GEMM, örnekleme, nicemleme çekirdekleri;
  - `include/nexus/cuda/serve.hpp`, `src/cuda/serve.cpp`.
- CPU: `lm::KvCache`, `ForwardOptions::cache`, `AttentionSpec::kv_time` (`include/nexus/lm/{model,ops}.hpp`, `src/lm/{model,ops}.cpp`).
- Araç: `benchmarks/runner/faz16_gpt2.py`. Komutları:
  - `export`: ağırlık + sözlük;
  - `logits`: FP64 referans;
  - `bench` / `bench_compiled`: PyTorch ölçümü.
- Testler: `tests/cuda/test_phase16_infer_cuda.cpp`, `tests/parity/test_phase16_infer.cpp`.
- Ham çıktılar: `nexus_bench.jsonl`, `torch_bench.jsonl`, `tests_cuda.log`, `tests_quant.log`, `tests_cpu.log`, `ctest_cpu.txt`, `ctest_cuda.txt`, `asan_ctest.txt`.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
