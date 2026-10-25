# FAZ 12 — Tensör, Dizi, Ardışık ve Bağlam Paralelliği; DeviceMesh / DTensor (2026-10-25)

Kapsam: `promt.txt` FAZ 12 (DUVAR 9, model paralelliği kısmı). Alt adımlar 12.A–12.H'nin hepsi
**CPU üzerinde** yapıldı: rank'ler thread arka ucunda (tek süreç) ya da `nexus-run` ile tcp süreçleri.
Model `lm::LanguageModel` (FAZ 6) yapılandırmasının paralel sürümü (`dist::ParallelLm`); GPU `GptModel`
yolunda TP/PP/CP yok (aşağıda).

Makine: Ryzen 7 5800H (8 çekirdek), 32 GB, Windows 11. Güç: koşunun başında pil, sonunda fiş
(`Win32_Battery` 1 → 2). Her yapılandırma 3 kez koşuldu; tablolarda medyan, tekrarlar arası en büyük sapma %12.
Referans: PyTorch 2.14.1+cpu `torch.distributed.tensor.parallel` (TorchTitan'ın TP planı), gloo.

## Kabul ölçütleri

| Ölçüt | Sonuç | Kanıt |
|---|---|---|
| Tüm eşdeğerlik testleri geçer | ✅ Aşağıdaki 11 testin hepsi geçti. tp=1 modeli `lm::LanguageModel` otogradına karşı kayıp ≤ 4.8e-7, gradyan ≤ 3.8e-7 (göreli). TP/SP/async TP, PP'nin dört zamanlaması, CP (ring + Ulysses), 3B ve 4B (16 rank) tek ranka göre gradyan ≤ 1e-6, 3 adım kayıp ≤ 3.6e-6. | `test_*` (aşağıda) |
| Erişilen kümede 7B için en iyi yapılandırmada MFU ≥ TorchTitan/Megatron | ⛔ **Ölçülemedi.** Tek GPU var, GPU yolunda TP/PP/CP yok, 7B 8 GB'a sığmıyor (FAZ 11). CPU'da küçük modelle PyTorch TP'ye karşı ölçüldü: Nexus TP ölçekleniyor, PyTorch CPU TP ölçeklenmiyor; ama tek iş parçacığı çekirdek hızında PyTorch önde ve en iyi PyTorch ayarı (TP'siz, 8 iş parçacığı) Nexus'un en iyisinden **1.5× hızlı**. | `torch_tp.jsonl`, `nexus_tp_*.jsonl` |

## 12.A DeviceMesh + DTensor

- `DeviceMesh(world, shape, names)`: satır-ana koordinat; her boyut için alt grup; `flatten({"dp","cp"})`.
- `DTensor`: yerel parça + yerleşim listesi `Shard(d) | Replicate | Partial(sum)`; torch.chunk kuralıyla eşit olmayan
  parçalar (6 satır / 4 rank → 2,2,2,0).
- `redistribute`: S→R all-gather, P→R all-reduce, P→S reduce-scatter, S(i)→S(j) all-to-all, R→S yerel dilim,
  R→P (koordinat 0 tutar), S→P. İç içe parçalama (iki mesh boyutu aynı tensör boyutunu) açık hata.
- Yayılım kuralları: matmul, add/mul, norm, softmax, embedding (`propagate` + `dt_*` işlemleri).

`test_dtensor_redistribute_all_pairs`: 1B mesh (4 rank) **16/16**, 2B mesh (2×2) **196/196** (kaynak, hedef)
çifti; tam değer, yerel parça ve yerel şekil doğru (tamsayı değerlerle tam eşitlik). Op yayılımı: 60 (1B) + 630 (2B)
denetim, tam hesapla aynı ve çıkış yerleşimi kuralla aynı.

## 12.B TP, 12.C SP, 12.D Async TP

- QKV column-parallel. GQA'da `Hkv % tp == 0` ise KV bölünür, `tp % Hkv == 0` ise KV kopyalanır ve gradyanı
  "kv grubu"nda toplanır. Çıkış projeksiyonu row-parallel. MLP up (SwiGLU'da eşleşen W1|W3 dilimleri) column,
  down row.
- Gömme sözlük-parçalı; lm_head sözlük-parçalı. CE logit toplanmadan: yerel max → all-reduce(Max), sum-exp ve
  hedef logit → all-reduce(Sum). Dolgu sütunları (61 → 64) −inf sayılır.
- SP: norm/artık satır ekseninde parçalı `[N/tp, C]`; bölge girişinde all-gather, çıkışında reduce-scatter.
- Async TP: SP'de AG+GEMM ve GEMM+RS halka üzerinde parça parça. Parça k hesaplanırken k+1 halkada
  (`ProcessGroup::send_recv`, bu fazda eklendi: tcp'de karşılıklı gönderim kilitlenmesini önler).

| Test | Sonuç |
|---|---|
| `test_tp_equivalence` | 3 model (GPT tipi LN+GELU+öğrenilen konum+bias+bağlı; LLaMA tipi RMS+SwiGLU+RoPE+GQA kv=2; kv=1 kopyalama) × tp=2/4 × {TP, +SP, +SP+async}: kayıp farkı **0**, gradyan göreli ≤ 6.9e-7, TP kopyaları bit bit aynı |
| `test_vocab_parallel_ce_equals_full` | sözlük 61 (dolgulu 64) ve 250 (256), tp=1/2/4: CE farkı **0**, wte/head gradyanı ≤ 3.4e-7 |
| `test_tp_equivalence_tcp` | 4 tcp süreci, dp=2 × tp=2+SP+async: 3 adım kayıp tek ranka göre 2.4e-7 |

### Adım süresi (CPU, LLaMA tipi 4.3M parametre, d 256, 8 baş, T 256, B 2)

Nexus rank başına tek iş parçacığı (seri GEMM). PyTorch: rank başına süreç, `threads` iş parçacığı.

| Yapılandırma | Nexus thread | Nexus tcp | PyTorch TP (1 iş p./rank) | PyTorch (çok iş p.) |
|---|---|---|---|---|
| tp=1 | 536 ms | — | **283** | **105** (8 iş p.) |
| tp=2 | 280 | — | 362 | — |
| tp=2 + SP | 282 | 288 | 416 | 303 (4 iş p.) |
| tp=2 + SP + async | 260 | 263 | — | — |
| tp=4 | 173 | — | 333 | — |
| tp=4 + SP | 178 | 186 | 420 | 367 (2 iş p.) |
| tp=4 + SP + async | **157** | **164** | — | — |

- Nexus TP ölçekleniyor: tp=4'te tp=1'e göre 3.4×. PyTorch CPU TP'de tp > 1 her durumda tp=1'den yavaş
  (DTensor dağıtım + gloo ek yükü; gloo all-reduce'u FAZ 10'da da yavaştı).
- Async TP: SP'ye göre tp=2'de %8, tp=4'te %12 kısa (tcp'de %9 / %11).
  GEMM satır bloklaması tek başına hız kazandırmıyor (ayrı ölçüm: 512 satır ile 4×128 satır aynı ~45 GFLOP/s).
  Kazanç örtüşmeden ve toplu senkronizasyon noktasının kalkmasından geliyor.
- **Gerideyiz:** tek iş parçacığında Nexus 536 ms, PyTorch 283 ms (1.9× yavaş). Nexus'un `gemm_nt_accum` ve dikkat
  çekirdekleri MKL kadar hızlı değil. En iyi PyTorch ayarı (TP'siz, 8 iş parçacığı) 105 ms; en iyi Nexus (tp=4 async,
  4 çekirdek) 157 ms.

## 12.E PP

Zamanlamalar: GPipe, 1F1B, interleaved 1F1B (sanal aşama v), ZB-H1. Model `pp × v` parçaya maliyet dengeli bölünür
(DP ile en büyük parça maliyeti en küçük; kafa maliyeti son katmana eklenir). Aşamalar arası aktivasyon/gradyan:
yönlü kenar grupları, asenkron send/recv; alıcı tamponları adım başında ayrılır ve tüketim sırasıyla kuyruğa alınır.
Bağlı gömmede ilk/son aşama kopyasının gradyanı {ilk, son} grubunda toplanır. ZB-H1: B yalnız giriş gradyanı;
W'ler FIFO ertelenir, sonraki F/B'nin girişi gelmemişse (kabarcık) ya da bekleyen W sayısı pp − r'yi geçerse
çalışır. Bu karar çalışma anında verilir (statik ZB-H1 tablosu yok).

`test_pp_schedules_equivalence` (pp=4, 8 katman, m=8): dört zamanlamanın gradyanı **GPipe ile bit bit aynı** (fark 0;
mikro yığın birikim sırası hepsinde aynı) ve otograda göre 9.5e-7.

### Kabarcık oranı (pp=4, 8 katman, d 256, T 128, B 2; ölçülen = recv beklemesi / adım, rank ortalaması)

| m | GPipe ölç. / teori | 1F1B | Interleaved v=2 | ZB-H1 | 1F1B tepe akt. | GPipe tepe akt. |
|---|---|---|---|---|---|---|
| 4 | 0.303 / 0.429 | 0.290 / 0.429 | 0.185 / 0.273 | 0.237 / 0.329 | 32.6 MB | 34.4 MB |
| 8 | 0.183 / 0.273 | 0.190 / 0.273 | 0.110 / 0.158 | 0.137 / 0.199 | 32.6 | 68.7 |
| 16 | 0.100 / 0.158 | 0.110 / 0.158 | 0.063 / 0.086 | 0.070 / 0.110 | 32.6 | 137.4 |

- Sıralama teoriyle aynı: GPipe ≈ 1F1B > ZB-H1 > interleaved. m arttıkça hepsi düşüyor.
- 1F1B aktivasyon belleği m'den bağımsız; GPipe m ile doğrusal artıyor.
- Ölçülen değerler teorinin altında (%30–40). Teori aşama başına eşit (tF + tB) ve sıfır iletişim varsayıyor.
  Ölçüm yalnız recv beklemesini sayıyor: ilk rank'in ısınma sonundaki kısa beklemeleri ve son rank'in ilk girişten
  önceki hazırlığı tam yakalanmıyor. Bu yüzden ölçüm alt sınır olarak okunmalı.
- ZB-H1 teorisi ölçülen tF/tB/tW ile: tB (giriş) 67 ms, tW 19.5 ms, 1F1B'nin tam B'si 84 ms. W ucuz; burada
  dikkat geri geçişi giriş gradyanında. Bu yüzden ZB-H1 kazancı interleaved'den az.

## 12.F CP

- Ring Attention: K/V halkada (`send_recv`, hesapla örtüşür); blok başına yerel Flash + LSE birleşimi.
  Geri geçişte global O/LSE ile blok Flash geri; dK/dV biriktiricisi K/V ile dolaşıp sahibine döner.
- Zigzag ataması: 2cp parça, rank r {r, 2cp−1−r}.
- Ulysses: all-to-all ile [B, T, H/cp, D].

`test_ring_attention_equals_flash`: ring (ardışık, zigzag; cp=4, GQA 4/2; cp=8, 2/1) ve Ulysses (cp=2, cp=4): ileri ve
geri tam Flash'a göre ≤ 9.5e-7 (Ulysses tam 0). Nedensel blok sayısı rank başına: ardışıkta 1…4 (cp=4), zigzag'de
hepsi 9 (dengeli).

### Bağlam taraması (ring zigzag, cp=8 thread rank, 1 baş, D 64, B 1, ileri + geri)

| T | ileri | geri | rank başına tampon | cp=1 tam Flash (ileri / geri / tampon) |
|---|---|---|---|---|
| 8k | 73 ms | 136 ms | 4.75 MB | 252 / 708 ms / 18.0 MB |
| 16k | 191 | 530 | 9.51 | 1043 / 2852 / 36.1 |
| 32k | 798 | 2060 | 19.0 | 4292 / 11702 / 72.1 |
| 64k | 3089 | 8794 | 38.0 | — |
| **128k** | 13212 | 34736 | 76.1 | — |

- 128k bağlam ileri + geri çalışıyor. Rank başına bellek T ile doğrusal (T/cp + halka tamponları). cp=1'e göre
  3.8× az: 8× değil, çünkü halka için 10 ek blok tutuluyor. Süre T² ile ölçekleniyor (nedensel).
- Ardışık atama (32k): ileri 1255 ms, zigzag 798 ms. Son rank 8 blok, ilk rank 1 blok hesaplıyor; zigzag'de her rank 17.
- Tampon belleği analitik (ayrılan dizilerin boyu); süreç tepe belleği rank başına ayrılamadığı için ölçülmedi.
- Hedef "128k+ bağlam eğitimi": dikkat çekirdeği düzeyinde gösterildi. Tam model eğitimi 128k'da koşulmadı
  (CPU'da adım dakikalar sürer). Model içi CP `test_4d_equivalence`'ta T=32 ile doğrulandı.

## 12.G Otomatik yapılandırma (`nexus_estimate --auto-parallel`)

(dp, tp, pp, cp, mikro-yığın, recompute, zero) kombinasyonlarını kapalı biçim tahmini adım süresi ve bellekle sıralar,
en iyi 3'ü verir. Bellek: FAZ 0 ZeRO kuralı + Korthikanti aktivasyon tablosu (TP/SP/recompute/CP).
Zaman: 6N + 12LhT, halka iletişim maliyetleri, 1F1B kabarcığı, DP örtüşme kesri, parça boyu verim çarpanı.
**Bu bir tahmin; gerçek kümede kalibre edilmedi** (küme yok). Varsayımlar `tools/nexus_estimate/autoparallel.hpp`'de.

`test_auto_parallel_config`:
- statik bellek FAZ 0 `estimate_memory` ile aynı;
- tp, SP, cp ve recompute belleği azaltıyor;
- 7B @ 8×80 GB: 3 öneri, sığan ve sıralı (#1 dp=8 ZeRO-3 mb=2, 73.6 GiB, tahmini MFU %47);
- 70B @ 64×80 GB: TP düğüm içi (#1 dp=1 tp=2 pp=8 cp=4 ZeRO-3);
- 7B @ 1×8 GB: sığan yok, açık not.

## 12.H 3B / 4B birleşik eşdeğerlik ve 12.8 API

| Test | Yapılandırma | Gradyan (otograd) | 3 adım kayıp | Ağırlık farkı / güncelleme |
|---|---|---|---|---|
| `test_3d_equivalence` | dp=2 × tp=2+SP × pp=2 1F1B, 8 rank (GPT tipi) | 6.0e-7 | 4.8e-7 | 2.1e-4 |
| | dp=2 × tp=2+SP+async × pp=2 interleaved v=2 (LLaMA tipi) | 4.4e-7 | 4.8e-7 | 3.3e-6 |
| | aynı + ZeRO-1 | — | 4.8e-7 | 3.3e-6 |
| `test_4d_equivalence` | dp=2 × pp=2 × cp=2 ring zigzag × tp=2+SP, **16 rank** | 6.9e-7 | 3.6e-6 | 2.1e-4 |
| | dp=2 × pp=2 ZB-H1 × cp=2 Ulysses × tp=2+SP, 16 rank | 4.5e-7 | 2.4e-7 | 3.0e-5 |
| `test_parallel_zero1_equivalence` | ZeRO-1 vs ZeRO-0 (dp=4; dp=2×tp=2), 5 adım | — | **bit bit** | **bit bit** |
| `test_parallel_trainer_api` | `LanguageTrainerBuilder(lm).parallel(world, {dp=2, tp=2, pp=2, sp, pp_schedule="1f1b"})`, 4 adım | — | tek rank eğitmene göre 1.2e-7 | — |

- Ağırlık ölçütü ‖Δw‖ / ‖w − w₀‖ (parametre başına en büyük). Max-abs fark (≤ 1.2e-4) gradyanı gürültü tabanında olan
  öğelerde Adam'ın işaret çevirmesinden geliyor (FAZ 10'daki anahtar bias'ı durumu; GPT tipinde en kötü `qkv.bias`).
- ZeRO-1'de `Param::grad` dp × cp indirgemesinden önce yereldir (indirgenmiş gradyan düz parçada); bu yüzden
  gradyan sütunu "—".
- API: model kodu değişmiyor. `lm::LanguageModel` aynı; `.parallel({dp, tp, pp, cp, zero, sp, async_tp, pp_schedule,
  virtual_stages, cp_algorithm, cp_layout})` paralel katmanları seçiyor. Biriktirme mikro adımları pipeline mikro
  yığınları. Desteklenmeyenler açık hata: reshape, token_losses (score_choices), NaN teşhisi, katman histogramı.

## Regresyon

- CPU paketi **108/108** (`ctest_cpu.txt`).
- GPU paketi **36/36** (`ctest_cuda.txt`; `lm_trainer.hpp` değişikliğinden sonra yeniden derlendi).
- ASan: FAZ 12'nin 11 testi (16 rank thread ve 4 süreç tcp dahil) + FAZ 9–11 seçkisi **20/20** (`asan_ctest.txt`).

## Gerideyiz / yapılmadı

- **GPU:** TP/PP/CP yalnız CPU `ParallelLm`'de. GPU `GptModel`'de yok; NCCL yok. MFU kabul ölçütü ölçülemedi.
- **Çekirdek hızı:** tek iş parçacığında PyTorch 1.9× hızlı; en iyi PyTorch ayarı en iyi Nexus'tan 1.5× hızlı.
- **ZeRO:** TP/PP/CP ile yalnız ZeRO-0/1. ZeRO-2/3 saf DP'de FAZ 11 `FsdpEngine`'de.
- **Kapsam dışı `lm` alanları:** µP, son-norm, paralel blok, kayan pencere, softcap, dropout, qk_norm, ALiBi,
  sinüzoidal konum, recompute. Hepsi açık hata veriyor.
- **ZB-H1:** statik tablo değil, çalışma anı doldurma. ZB-V / ZB-H2 yok.
- **Kabarcık:** ölçüm yöntemi alt sınır (yukarıda). P2P iletişim süresi ayrıca ölçülmedi.
- **Async TP:** thread/tcp üzerinde CPU memcpy/loopback; gerçek NVLink örtüşmesi ölçülmedi.
- **CP:** model içinde 128k eğitim koşulmadı; rank başına süreç belleği ölçülmedi (analitik tampon).
- **Otomatik yapılandırma:** kalibre edilmemiş tahmin; mikro-yığının GEMM verimine etkisi modelde yok.
- Kopyasız parametre görünümü: parametreler rank başına ayrı dizilerde. dp × cp indirgemesi için düz tampona
  paketleme kopyası var.

## Dosyalar

- Kod:
  - `include/nexus/dist/dtensor.hpp`, `context_parallel.hpp`, `pipeline.hpp`, `parallel_lm.hpp`;
  - `src/dist/dtensor.cpp`, `context_parallel.cpp`, `pipeline.cpp`, `parallel_lm.cpp`;
  - `src/train/parallel_backend.cpp`;
  - `tools/nexus_estimate/autoparallel.{hpp,cpp}`;
  - `ProcessGroup::send_recv` ve 63 alt grup sınırı denetimi.
- Testler: `tests/parity/test_phase12_parallel.cpp`, `test_phase12_autoconfig.cpp`.
- Ölçüm: `benchmarks/runner/faz12_bench.py`, `faz12_torch_tp.py`.
- Ham çıktılar: `nexus_tp_thread.jsonl`, `nexus_tp_tcp.jsonl`, `torch_tp.jsonl`, `nexus_pp.jsonl`, `nexus_cp.jsonl`,
  `ctest_cpu.txt`, `ctest_cuda.txt`, `asan_ctest.txt`.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
