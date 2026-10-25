# FAZ 11 — ZeRO-1/2/3, FSDP, HSDP, NVMe Offload (2026-10-25)

Kapsam: `promt.txt` FAZ 11 (DUVAR 8). Bu fazda gelen parçalar:
- düz birim + parçalama (FlatParameter);
- ZeRO-0/1/2/3 (NO_SHARD, SHARD_GRAD_OP, FULL_SHARD) ve HYBRID_SHARD;
- ileri/geri önceden getirme ve all-gather sınırlayıcısı;
- BF16 iletişim politikası;
- NVMe takası (ZeRO-Infinity G/Ç);
- parçalı kayıt ve farklı dp'ye yeniden parçalama;
- eğitmen v2'de `.parallel({zero})` / `.fsdp(...)`.

Hepsi **CPU `nn::Sequential` modellerinde** (`dist::FsdpEngine`). GPU `GptModel` yolunda ZeRO/FSDP yok (aşağıda).
Makine: Ryzen 7 5800H, 32 GB, RTX 3070 Laptop 8 GB, Windows 11, fişte. Rank'ler thread arka ucunda
(tek süreç) ya da `nexus-run` ile tcp süreçleri. Referans: PyTorch 2.14.1 FSDP2 (`fully_shard`, CPU aygıt ağı, gloo).

## Kabul ölçütleri

| Ölçüt | Sonuç | Kanıt |
|---|---|---|
| Eşdeğerlik testleri geçer | ✅ ZeRO-0/1/2/3, dp=4, 100 adım, FP32: ağırlık, m ve v **bit bit aynı** (fark 0). Kayıp 3.60414 → 2.65204, her aşamada aynı. 4 tcp süreciyle ZeRO-3 = ZeRO-0 (50 adım, bit bit). FSDP ZeRO-0 ile FAZ 10 DDP arasında kayıp farkı 1.8e-7. Ağırlıkta anahtar bias'ında 1.5e-3 fark var; FAZ 10'daki aynı sıfır-gradyan gürültüsü. | `test_zero_stages_equivalence` (+ `_tcp`) |
| 8 kartta llama7b BF16 FSDP, MFU ≥ PyTorch FSDP2 | ⛔ Ölçülemedi: tek GPU. GPU yolunda FSDP de yok. | — |
| 8 GB tek kartta 7B'nin birkaç adımı ZeRO-Infinity ile | ⛔ **Yapılmadı.** (1) GPT tipi 6.7B için durum: BF16 parametre 13.4 GB + FP32 ana kopya 26.8 GB + m, v 53.6 GB ≈ **94 GB**. 8-bit Adam + ana kopyasız modda bile ≈ 27 GB. Diskte boş alan **26 GB**. (2) GPU `GptModel` katman katman parametre akıtmayı desteklemiyor (tüm parametreler aygıtta); NVMe takası yalnız CPU motorunda. CPU'da 7B'nin bir adımı ~6·7e9·token FLOP ister, gösterim süresine sığmaz. NVMe takasının doğruluğu küçük modelde gösterildi (aşağıda). | — |

## Kart (rank) başına kalıcı bellek ve adım süresi

Model (iki çerçevede de aynı şekil, **3 168 267 parametre**):
- Embedding(11, 256) → 6 × [LN, 2 başlı dikkat, LN, FFN 512] → Linear;
- rank başına 4 × 64 token, AdamW, kırpma 1.0;
- blok başına bir birim (per-block wrap).

Kalıcı bellek, adımlar arasında tutulan parametre + gradyan + optimizer baytlarıdır (geçici AG/RS tamponları hariç).
PyTorch tarafında aynı kapsam ölçüldü (DTensor yerel tensörleri). Adım süresi 6 adımın ortalaması; rank'ler aynı 8 çekirdeği paylaşıyor.

| dp | Nexus ZeRO-0 | ZeRO-1 | ZeRO-2 | **ZeRO-3** | **PyTorch FSDP2** | Nexus ZeRO-3 ms | FSDP2 ms |
|---|---|---|---|---|---|---|---|
| 1 | 63.4 MB | 76.0 | 63.4 | **50.7** | **50.7** | 119 | 97 |
| 2 | 57.0 | 50.7 | 38.0 | **25.3** | **25.4** | 129 | 355 |
| 4 | 53.9 | 38.0 | 25.3 | **12.7** | **12.7** | 166 | 305 |
| 8 | 52.3 | 31.7 | 19.0 | **6.34** | **6.34** | 279 | 651 |

- **ZeRO-3:** Tam 1/dp ölçekleniyor (test: dp=2/4/8 oranı 0.500 / 0.250 / 0.125, tolerans %2). PyTorch FSDP2 ile aynı.
- **Adım süresi:** dp ≥ 2'de Nexus 2.3–2.8× hızlı. dp=1'de PyTorch %18 hızlı (farklı CPU çekirdekleri).
- **ZeRO-0/1/2 fazlalığı (gerideyiz):**
  - Ortak iletişim deseni (RS + AG) için her aşamada ayrı bir parça gradyanı tutuluyor.
  - ZeRO-1/2'de FP32 ana kopya parçası parametre tensörlerinden ayrı.
  - Sonuç: dp=1'de ZeRO-0 63.4 MB, ideal 50.7; ZeRO-1 76.0 MB. Bu, eşdeğerliği bit bit yapan tasarımın bedeli.
    FP32 eğitimde ana kopya yerinde güncellenebilir; bu yapılmadı.

Ham çıktılar: `bench_mem_nexus.txt`, `bench_mem_torch_fsdp2.txt`.

## Testler

| Test | Ne denetlenir | Sonuç |
|---|---|---|
| `test_zero_stages_equivalence` (+ `_tcp`) | yukarıdaki kabul satırı | ✅ |
| `test_fsdp_memory_scaling` | ZeRO-1/2/3, dp=1, 2, 4, 8; ZeRO-3 kalıcı bellek / dp1 = 1/dp (±%2) | ✅ |
| `test_fsdp_prefetch_order` | ZeRO-3 dp=2, 6 birim, limit 2: (a) ilk adımda ileri sıra kaydedilir; (b) 2. adımda her k için AG(k+1) birim k'nin ileri hesabından önce verilir; (c) geri geçişte AG(k−1) B(k)'den önce verilir; (d) uçuşta en çok 2 all-gather. Örnek iz: `AG 0 · AG 2 · AG 3 · F 0 · FREE 0 · AG 4 · F 2 · …` | ✅ |
| `test_nvme_offload_roundtrip` | Hizalı, asenkron yazma/okuma (1 B, 4 KB, 10 MB) bit bit; yaz+oku 1.13 GB/s. ZeRO-3, optimizer ve parametre parçası NVMe'de: 20 adım sonra RAM'de çalışan sürümle **bit bit aynı**. RAM'deki optimizer belleği 0. Kalıcı RAM 142.4 → 35.6 KB, NVMe'de 106.8 KB. Adım 0.69 → 36.9 ms (yavaş ama doğru). | ✅ |
| `test_sharded_checkpoint_reshard` | dp=4'te 4 parça dosyası → dp=2 dünyasında `load_resharded`: tam parametre ve m durumu **bit bit aynı**; ardından ileri geçiş çalışıyor. | ✅ |
| `test_hsdp_and_bf16_comm` | HSDP (2 düğüm × 2): kayıplar FULL_SHARD ile 1.2e-7 içinde. Kalıcı bellek düğüm içi parça kadar (20.7 vs 10.4 KB). BF16 all-gather + reduce-scatter ile son kayıp 2.6580 (FP32: 2.6578). | ✅ |

Regresyon: CPU paketi **97/97** (`ctest_cpu.txt`), GPU paketi **32/32** (`ctest_cuda.txt`). Başlık bölünmesinden (`fsdp_options.hpp`) sonra FAZ 9–11 CPU testleri yeniden koşuldu: 12/12. ASan: FAZ 9–11 CPU testleri **30/30**,
tcp çok süreçli testler dahil (`asan_ctest.txt`).

## Tasarım özeti

- **11.1 Düz birim.**
  - Birim: parametreli her Sequential katmanı (`Wrap::PerBlock`) ya da tüm model (`Wrap::Whole`).
  - Birimin parametreleri duz [0, N) uzayında ardışık; S = ⌈N/dp⌉, dolgu sıfır; rank r [rS, (r+1)S) parçasının sahibi.
  - nn katmanları parametreyi kendi `Tensor`'ında tuttuğu için **görünüm değil paketle/aç kopyası** kullanılıyor.
    Spec "kopyasız görünüm" istiyor; bu sağlanmadı.
- **11.2 Aşamalar.**
  - Ortak desen: hepsi reduce-scatter (+ gerekiyorsa all-gather) kullanıyor. Toplama sırası öğe başına aşamadan
    bağımsız; AdamW öğe başına çalışıyor; global norm parça gradyanlarından hesaplanıyor. Bu yüzden ağırlıklar
    aşamalar arası bit bit aynı.
  - ZeRO-2/3: tam gradyan paketlenince serbest kalır.
  - ZeRO-3: birimden önce AG → hesap → parametre tensörleri serbest; geri geçişte tekrar AG.
  - HSDP: düğüm içi FULL_SHARD + parça gradyanının düğümler arası all-reduce'u.
- **11.3 Önceden getirme.** İlk ileri geçişte birim sırası kaydedilir. Sonraki geçişlerde birim k hesaplanırken
  sıradaki birimlerin AG'si uçuştadır (`limit_all_gathers`); geri geçişte ters sırayla. RS'ler asenkron başlar,
  adımda beklenir. İz (`trace()`) testte doğrulanıyor. Tüm iletişim ProcessGroup iş parçacığında, hesapla eş zamanlı.
- **11.4 Karışık hassasiyet.** Parça FP32 ana kopya; AG `param` tipinde (BF16: iletişim yarı, hesap BF16'ya yuvarlanmış
  ağırlıkla), RS `reduce` tipinde.
- **11.5 Offload.**
  - CPU modelinde parametreler zaten host'ta, "cpu" offload burada anlamsız.
  - `optimizer_nvme` / `params_nvme`: m, v ve FULL_SHARD'da FP32 parametre parçası kullanım dışında NVMe
    dosyasında; kullanımdan önce okunur, sonra yazılıp RAM'den bırakılır.
  - Windows: `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, 4096 bayta hizalı bounce tamponu, m ve v okuması
    aynı anda uçuşta. IOCP portu yok (olay başına bekleme). Linux io_uring / libaio yok (iş parçacığında pread/pwrite).
  - Bir sonraki adımın okumasını önceden başlatma yok.
- **11.6 Kayıt.** `save_shard`: rank'in parçaları (FP32 parametre, m, v, adım) ve kapsadığı düz aralık.
  `load_resharded`: eski dünyanın tüm dosyalarından birim uzayını kurar, yeni parçayı alır.
  `full_parameters` / `full_optimizer_state`: tam durum (küçük modeller). Eğitmen `save/resume` FSDP'de rank başına
  dosya yazar. Tam FAZ 15 kayıt sistemi değil.
- **11.7 API.** `train::LanguageTrainerBuilder::fsdp(group, FsdpOptions{sharding, wrap, mixed_precision, offload,
  limit_all_gathers, ranks_per_node})` ve `.parallel(group, {zero = 0..3})`. Token ağırlığı, kayıp toplama ve
  rank 0 günlüğü FAZ 10'daki gibi.

## Açık kalanlar / gerideyiz

- **GPU'da ZeRO/FSDP yok.** `GptModel` tüm parametre, gradyan ve optimizer durumunu aygıtta tek tamponda tutuyor.
  Parçalama için katman katman parametre akıtma gerekiyor (gelecek iş). VRAM tasarrufu ölçülmedi.
- **7B ZeRO-Infinity gösterimi yapılmadı** (disk 26 GB boş, ≥27–94 GB gerekiyor; GPU akıtma yok).
- **Çok GPU ölçümü yok;** PyTorch FSDP2 / DeepSpeed GPU karşılaştırması yok (tek GPU; DeepSpeed Windows'ta kurulu değil).
- **Kopyasız görünüm yok** (paketle/aç kopyası).
- **ZeRO-0/1/2'de fazladan parça gradyanı ve ana kopya** (tablo).
- **Kayıt ve izleme eksikleri:**
  - Katman normu histogramı FSDP'de kapalı.
  - NaN teşhisinde katman adı yerine "model" düzeyi raporlanıyor.
- **NVMe'nin kendi sınırları:**
  - Önceden getirme yok.
  - IOCP portu yok.
  - io_uring yok.
  - Ölçülen 1.13 GB/s tek dosyalı tek koşu.

## Dosyalar

| Spec | Dosya |
|---|---|
| 11.1–11.4, 11.6 | `include/nexus/dist/fsdp.hpp`, `src/dist/fsdp.cpp` |
| 11.5 NVMe | `include/nexus/dist/nvme.hpp`, `src/dist/nvme.cpp` |
| 11.7 | `include/nexus/train/lm_trainer.hpp` (`fsdp`, `parallel`), `src/train/lm_trainer.cpp` (CPU arka ucu) |
| Testler / araçlar | `tests/parity/test_phase11_fsdp.cpp`, `benchmarks/runner/faz11_torch_fsdp.py` |

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
