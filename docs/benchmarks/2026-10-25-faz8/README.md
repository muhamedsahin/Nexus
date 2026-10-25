# FAZ 8 — Veri Hattı (2026-10-25)

Kapsam: `promt.txt` FAZ 8. Byte düzeyinde BPE (kodlama + eğitim), NXTK token parçaları ve
`nexus_tokenize`, mmap okuyucu, Feistel karıştırmalı devam edebilen örnekleyici, paketleme
(cu_seqlens + kayıp maskesi), karışım, rank dilimi, pinned halkalı asenkron GPU yükleyici.
Makine: RTX 3070 Laptop (8 GB, WDDM), Ryzen 7 5800H (8 çekirdek / 16 iş parçacığı), 32 GB RAM,
CUDA 13.4, MSVC 19.44. Referanslar: tiktoken 0.14, HF tokenizers (pip, `.bench-venv-py314`).

**Ölçüm koşulu uyarısı.** Ölçümlerin bir kısmı laptop pildeyken alındı (deşarj, %36,
"Legion Balance"). CPU 1.9 GHz'de, GPU tepe saatin altında çalıştı. Bu yüzden aynı ölçüm iki
oturumda yaklaşık 2 kat fark ediyor. Karşılaştırmalar hep **aynı oturumda, art arda**
koşuldu. Oranlar geçerli, mutlak sayılar makinenin en iyi durumu değil.

## Kabul ölçütleri

| Ölçüt | Sonuç | Kanıt |
|---|---|---|
| GPT-2 tokenizasyonu tiktoken ile bayt-bayt aynı | ✅ GPT-2 **ve** cl100k: 10 000 üretilmiş + 400 uzun parçalı belge ve **200 MB gerçek derlem (24 598 dosya)**, 0 farklı belge | `test_bpe_encode_matches_tiktoken`, `verify_dir` modu |
| Kodlama hızı ≥ tiktoken (aynı iş parçacığı sayısında) | ✅ 1 / 8 / 16 iş parçacığında GPT-2'de 4.7–5.0×, 6.5×, 7.3–7.8×; cl100k'de 5.1×, 6.1–6.3×, 7.2–7.4× | `bench_encode_interleaved.txt` |
| Korpus boyutu RAM'den bağımsız | 🟡 Kısmen. 19 GB'lık mantıksal veri kümesinde (192 parça) 3.07 milyar token akıtıldı: özel bellek **1.1 MB**, tepe çalışma kümesi **196 MB**. Bu değerler veri boyundan bağımsız. Ancak parçalar 8 dosyanın tekrarı ve diskte RAM'den büyük farklı veri yok (boş disk 42 GB). | `bench_stream.txt` |
| Devam testi birebir | ✅ 120. örnekte durum kaydedildi (`v1,0,3,12,120,99`, 16 bayt). **Yeni süreçte** kalan 180 örnek kesintisiz akışla birebir aynı ve epoch sınırını geçiyor. Karışım durumu da birebir. | `test_sampler_resume_exact` |

## Kodlama hızı (MB/s, 200 MB derlem, en iyi 3 koşu)

Derlem: yerel Go kaynakları, Python 3.14 `Lib`, Git ve JetBrains kurulum dosyaları (çoğu kod, bir
kısmı HTML/MD). Ayrıntı `encode_corpus_info.json`'da. Doğal dil metni değil, kod ağırlıklı.

Oturum B (pilde), art arda, iki tur (tur 1 / tur 2):

| Tokenizer | İş parçacığı | Nexus | tiktoken 0.14 | Oran |
|---|---:|---:|---:|---:|
| GPT-2 | 1 | 13.1 / 10.9 | 2.80 / 2.18 | 4.7× / 5.0× |
| GPT-2 | 8 | 94.0 / 85.5 | 14.4 / 14.5 | 6.5× / 5.9× |
| GPT-2 | 16 | 150.0 / 137.9 | 19.2 / 18.8 | 7.8× / 7.3× |
| cl100k | 1 | 11.6 / 11.7 | 2.27 / 2.28 | 5.1× / 5.1× |
| cl100k | 8 | 94.4 / 92.7 | 15.1 / 15.3 | 6.3× / 6.1× |
| cl100k | 16 | 151.1 / 143.5 | 20.5 / 20.0 | 7.4× / 7.2× |

Oturum A (fişte, ilk ölçüm; `bench_encode_session_a.txt`): tiktoken GPT-2 5.7 / 32.1 / 39.1, cl100k
5.7 / 33.7 / 40.8. Nexus GPT-2 24.0 / 175.4 / 266.7, cl100k 27.9 / 207.8 / 320.4 (bu cl100k
satırı yığın yolundan önceki sürümle alındı).

HF tokenizers (GPT-2, tiktoken sözlüğünden TikTokenConverter yöntemiyle kuruldu, 2000 belgede
tiktoken ile 0 fark): 16 iş parçacığında **0.92 MB/s**, 1 iş parçacığında 0.78 MB/s. Bu derlemde
uzun ön-token parçaları onu da çok yavaşlatıyor. 8 iş parçacığı ölçülmedi (bir koşu 10 dakikayı
aşıyor). İlk HF ölçümü yanlış kurulmuş sözlükle alınmıştı (2000 belgenin 1991'i farklıydı).
Atıldı, raporda yok.

**Uzun parça düzeltmesi.** İlk ölçümde GPT-2, 16 iş parçacığında tiktoken'in **gerisindeydi**
(37.3'e karşı 39.1 MB/s) ve 8'den sonra ölçeklenmiyordu. Neden: tiktoken'in doğrusal taramalı
birleştirmesi parça uzunluğunda O(n²). Kod derlemindeki uzun boşluk/sembol dizileri tek bir
belgeyi dakikalarca kodlatıyordu. Çözüm: 256 bayttan uzun parçalarda bağlı liste ve tembel
geçersizleştirmeli min-yığın. Seçim kuralı aynı: en küçük rank, eşitlikte en soldaki. Sonuç
tiktoken ile birebir kaldı (yukarıdaki 200 MB doğrulaması bu sürümle yapıldı). Spec hedefi
"≥ 50 MB/s/çekirdek": **sağlanmadı**. Tek iş parçacığında fişte 24–28, pilde 11–13 MB/s.

## BPE eğitimi (960 MB, 32k sözlük)

Derlem: 65 009 dosya, 959.8 MB (yukarıdaki kaynaklar + Windows SDK, CUDA ve MSVC başlıkları,
site-packages; `corpus_info.json`). Ön-bölme GPT-2, `min_frequency=0`, özel `<|endoftext|>`.

| | Süre | Birleşim |
|---|---:|---|
| HF tokenizers `BpeTrainer` (tüm çekirdekler) | **150.3 s** | 31 743 |
| Nexus `train_bpe` (16 iş parçacığı) | **19.8 s** (7.6×) | 31 743, **HF ile birebir aynı sıra** |

1 389 402 farklı kelime. Küçük derlemde (`test_bpe_train_matches_hf`) sözlük de birebir aynı.
Spec "10 GB metinde dakika ölçeği" diyor. 10 GB ölçülmedi. 1 GB'tan doğrusal tahmin ~3.5 dk,
ölçülmüş bir sonuç değil.

## NXTK ve nexus_tokenize

960 MB JSONL → `--dedup --threads 16`: 59 990 belge (5 019 tam kopya atıldı), 918 MB,
**395 925 902 token**, 8 parça (~50 M token, ~100 MB). Süre 58.0 s, **15.8 MB/s**. Araç
`encode_batch`'in (140–270 MB/s) çok gerisinde. JSON ayrıştırma, SHA-256 dedup ve yazma tek iş
parçacığında, kodlamayla örtüşmüyor. Açık kalan iş.

Biçim: 64 baytlık başlık (`NXTK`, sürüm 1, dtype 2/4 bayt, token/belge sayısı, tokenizer
SHA-256), token dizisi, 8'e hizalı dolgu, u64 belge ofsetleri. Her belgenin sonunda EOS var.
Manifest JSON her parçanın SHA-256'sını taşıyor. `test_shard_roundtrip`: 1500 belge birden çok
parçada token-token geri okunuyor, tek baytlık bozulma SHA ile yakalanıyor.

## Akış ve RAM bağımsızlığı

`TokenStream`, T=1024, 300 000 örnek. Parçalar manifest'te tekrarlanarak mantıksal veri büyütüldü:

| Mantıksal veri | Parça | Örnek/s | Özel bellek | Tepe çalışma kümesi |
|---:|---:|---:|---:|---:|
| 0.79 GB | 8 | 319 731 | 1.1 MB | 195.7 MB |
| 6.34 GB | 64 | 337 427 | 1.1 MB | 195.5 MB |
| 19.02 GB | 192 | 354 419 | 1.1 MB | 195.5 MB |
| 19.02 GB, 3 M örnek (3.07 G token) | 192 | 285 899 | 1.1 MB | 196.1 MB |

İlk sürüm tüm parçaları açılışta eşliyordu. Özel bellek parça sayısıyla doğrusal büyüdü:
2.2 → 12.6 → 36.3 MB. Neden: Windows dokunulan her 2 MB eşleme için 4 KB sayfa tablosu
tutuyor (19 GB'ta ≈ 38 MB). Örnekleyici aynı anda tek parçada gezdiği için parçalar artık tembel
eşleniyor, en fazla 2'si açık. Örnek dizisi değişmedi (kontrol toplamları iki sürümde aynı).
Tepe çalışma kümesi ≈ 2 açık parça. Bu sayfa önbelleği, geri alınabilir bellek.

## Paketleme, rank, karışım

- `test_pack_mask`: 2000 pencerede (390 belge sınırı) `cu_seqlens` ve kayıp maskesi kaba kuvvetle
  aynı. EOS'tan sonraki belgenin ilk token'ı hedef olarak maskeli.
- best-fit decreasing: 800 belge 376 kutuya (1024 token, doluluk %99.1), her belge bir kez, kesilmeden.
- `test_no_overlap_across_ranks`: 4 rank bir epoch'ta ortak pencere almıyor, birleşimleri
  epoch'un tamamı. Feistel n=1 000 003'te gerçek permütasyon.
- Karışım .70/.20/.10 hedefine karşı 4000 örnekte 0.705 / 0.193 / 0.102.

## GPU yükleyici (8.F)

`nexus::cuda::TokenLoader`: arka plan iş parçacığı üreticiyi çağırıyor ve N yuvalı pinned halkayı
dolduruyor. Kopya ayrı akışta `cudaMemcpyAsync` ile gidiyor, `h2d_done` olayıyla hesap akışına
devrediliyor. Yuva, hesap akışının `consumed` olayı tamamlanmadan yeniden yazılmıyor. Üretici
istisnası `next()`'te yeniden fırlatılıyor. `GptTrainer::step_device` aygıt token'ı alıyor.

| Test / ölçüm | Sonuç |
|---|---|
| `test_loader_order` (2 ve 4 yuva, her adımda 3 ms yapay gecikmeli tüketici, 200 yığın) | her yığın doğru ve sıralı |
| `test_loader_error` | 5 geçerli yığından sonra üretici hatası tüketiciye ulaşıyor |
| GPT-2 124M BF16 flash, B=4 T=1024, gerçek NXTK verisi, 100 adım | veri bekleme **0.0003 ms/adım (%0.000)** |

Mutlak adım süresi 1072 ms (3.8k token/s). FAZ 4'teki 14k token/s'in çok altında, çünkü GPU pilde
kısılmıştı (P8, 210 MHz boşta). Senkron besleme de aynı süreyi verdi (1071 ms): üretim yalnız
0.07 ms/adım, bu modelde zaten darboğaz değil. Yükleyici darboğazı kaldırıyor. Ama bu ölçüm
hızlı bir adımda kazancı göstermiyor. Spec "Nsight zaman çizelgesi" istiyor: alınmadı. Kanıt
host taraflı bekleme süresi.

## Sanitizer

AddressSanitizer (`build-asan`, `NEXUS_ENABLE_ASAN=ON`) altında 7 FAZ 8 CPU testinin hepsi geçti
(`asan_ctest.txt`). Tam CPU paketi 67/67. CUDA: `test_loader_order`, `test_loader_error`,
`test_cuda_gpt_small`, `test_bf16_training_converges`, `test_master_weight_modes` geçti.
TokenLoader için ThreadSanitizer koşulmadı (MSVC'de TSan yok).

## Açık kalanlar (KURAL 3)

- o200k ön-bölme deseni yok (gpt2, cl100k, Llama-3 var). Unigram/SentencePiece ve HF
  `tokenizer.json` okuma yok (FAZ 17 ile).
- SIMD ön-bölme yok. 50 MB/s/çekirdek hedefi sağlanmadı.
- `nexus_tokenize`: Parquet / WebDataset / uzak URL girdisi ve MinHash yakın kopya yok. Hattın
  kendisi tek iş parçacıklı (15.8 MB/s).
- Okuyucu: uzak depolama önceden getirme ve `FILE_FLAG_SEQUENTIAL_SCAN` ipucu yok (örnekleyici
  rastgele erişiyor).
- GPU GPT modeli `cu_seqlens`'i flash varlen'e, kayıp maskesini CE'ye henüz bağlamıyor. Örnekler
  üretiliyor ve test ediliyor ama GPU eğitimi hâlâ tam nedensel pencere ile eğitiyor (FAZ 9:
  LanguageTrainer v2).
- `NexusTrain async_prefetcher` ile birleştirme ve `TokenRows` → `TokenStream` geçişi FAZ 9'a kaldı.
- 8.7 / 8.G Python bağlamaları yapılmadı (`NEXUS_BUILD_PYTHON=OFF`, sistem Anaconda'sı kaldırılmış).
- RAM'den büyük farklı veriyle (≥ 33 GB) gerçek akış testi yapılmadı (disk).

## Dosyalar

| Spec | Dosya |
|---|---|
| 8.1 / 8.A / 8.B | `lib/NexusData/include/nexusdata/text/bpe.hpp`, `lib/NexusData/src/bpe.cpp`, `lib/NexusData/tools/gen_unicode_tables.py` (Unicode 16 \p{L}/\p{N} tabloları) |
| 8.2 / 8.C | `lib/NexusData/include/nexusdata/text/token_shard.hpp`, `lib/NexusData/src/token_shard.cpp`, `tools/nexus_tokenize/main.cpp` |
| 8.3–8.5 / 8.D–8.E | `token_shard.*`: `ShardReader`, `Manifest`, `FeistelPermutation`, `TokenStream`, `MixtureStream`, `best_fit_pack`, `SamplerState` |
| 8.6 / 8.F | `include/nexus/cuda/token_loader.hpp`, `src/cuda/token_loader.cpp`, `GptTrainer::step_device` |
| Testler | `tests/parity/test_phase8_data.cpp`, `tests/parity/reference_tokenizer.py`, `tests/cuda/test_phase8_loader.cpp` |

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
