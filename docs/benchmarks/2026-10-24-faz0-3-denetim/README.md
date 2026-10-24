# FAZ 0–3 Denetimi ve Test Tamamlama (2026-10-24)

FAZ 4'e geçmeden önce `promt.txt` içindeki FAZ 0, 1, 2 ve 3 tanımları kodla tek tek
karşılaştırıldı. Bu klasör o denetimin raporu ve ham ölçümleridir.

Makine: AMD Ryzen 7 5800H (AVX2+FMA, AVX-512 yok), Windows, MSVC 19.44, Release, Ninja,
CPU derlemesi (`build-phase2-cpu`, `NEXUS_ENABLE_CUDA=OFF`). PyTorch 2.14.1+cpu (`.bench-venv`).

## 1. Denetimde bulunan hatalar

Önceki oturumda FAZ 0–3 "tamamlandı" diye raporlanmıştı, ama kod **derlenmiyordu** ve
faz testleri **hiç koşulmamıştı**. Bulunan ve düzeltilen hatalar:

| # | Yer | Hata | Düzeltme |
|---|---|---|---|
| 1 | `src/autograd/engine.cpp` | `Variable::backward` içinden serbest `backward()` çağrısı üye adı tarafından gölgeleniyordu; `Node::apply` tekil `Variable` ile çağrılıyordu → **derleme hatası** | `autograd::backward` ile nitelendi, `apply` vektörle çağrılıyor |
| 2 | `Variable::zeros/ones/full` | Argüman değerlendirme sırası: MSVC `std::move(shape)`'i önce çalıştırınca boş şekilden 1 elemanlı vektör üretiliyordu → her `sum`/`mean` geri geçişi çöküyordu | `numel` önce hesaplanıyor |
| 3 | Motor | Bir kök başka bir kökün atası ise iki kez çalışıyordu; ikinci `backward` sessizce hiçbir şey yapmıyordu | Bağımlılık sayımı tüm köklerden; serbest bırakılmış grafikte açık hata |
| 4 | `saved_tensors_hooks` | Kancalar kaydediliyor ama **hiç uygulanmıyordu** ("FAZ 7'de uygulanacak" — KURAL 2 ihlali) | Gerçek pack/unpack; paketli yolda özgün veriye güçlü referans yok, sürüm denetimi korunuyor |
| 5 | Anomali modu | Bayrak vardı, davranış yoktu | NaN/Inf üreten düğüm adı + ileri geçiş bağlamıyla hata |
| 6 | Ara tensör kancaları, `retain_grad` | Yaprak olmayan tensörde hiç ateşlenmiyordu | Düğüm çıktı üst-verisini tutuyor; motor kancaları ve `retain_grad`'ı uyguluyor |
| 7 | `apps/lm_baseline` | Var olmayan `nn::TransformerDecoderLayer` kullanıyordu; `psapi.h`, `windows.h`'den önce → **derleme hatası** | Gerçek tarif (Embedding + sinüs konum + CausalTransformerBlock + Linear) |
| 8 | `apps/lm_baseline` | token/s adım sayısıyla fazladan çarpılıyordu (9 M yerine 47 k) | Düzeltildi |
| 9 | `test_phase0_prof`, `test_phase1_allocator` | Monoton `allocation_count()` canlı sayaç sanılmıştı; `totals()` kopyalarının iteratörleri karşılaştırılıyordu (UB) | `live_allocation_count()` eklendi; tek kopya üzerinde arama |
| 10 | `test_estimate_parity` | Windows `cmd /c` tırnak soyması: CLI hiç çalışmıyordu; "GPT-2 124M 8 GB'a sığmaz" iddiası yanlıştı (~4.6 GiB, FAZ 3 kabul kriteri sığmasını gerektirir) | Komut bir kat daha tırnaklanıyor; iddia düzeltildi, sığmayan örnek GPT-2 XL |
| 11 | `benchmarks/conv_gate` | `matrix_pro::Tensor` (CUDA yolu) CPU derlemesinde bağlanmıyordu | `NEXUS_ENABLE_CUDA` arkasında |

## 2. Eklenen kod ve testler

| Faz | Spec maddesi | Dosya | Test |
|---|---|---|---|
| 0.4 | `configs/*.json` (nano, tiny, gpt2s, gpt2m, gpt2xl, llama1b, llama7b) | `configs/` | `lm_baseline`, parity aynı dosyayı okur |
| 0.5 | Parity düzeneği | `tests/parity/{test_parity_nano.cpp, reference_torch.py, compare.py}` | `test_parity_nano`, `test_parity_tiny` |
| 1.1 | `DType`, `promote`, `nexus::half`, `nexus::bfloat16` | `include/nexus/core/{dtype,half}.hpp` | `test_phase1_dtype` |
| 1.4 | 10⁶ rastgele ayırma stres testi, canlı sayaç | `memory.{hpp,cpp}` | `test_phase1_dtype` |
| 2.3 | Gerçek `saved_tensors_hooks` + `SavedTensorsHooksGuard` | `autograd.hpp`, `engine.cpp` | `test_phase2_engine` |
| 2.4 | Anomali modu, `AnomalyModeGuard`, `detach` | `engine.cpp` | `test_phase2_engine` |
| 2.5 | `log_softmax`, `embedding`(padding_idx), `narrow`, `cat` + genel `gradcheck` (FP64 fark) | `ops.cpp`, `engine.cpp` | 26 op gradcheck |
| 2.6 | `custom_function` (autograd.Function) + `module_function` (klasik katman köprüsü) | `ops.cpp`, `module_function.hpp` | Klasik `Linear` ile bit-bit |
| 3.2 | RMSNorm fwd/bwd | `src/fused/training.cpp` | gradcheck + FP64 |
| 3.4 | Gömme + konum füzyonu, deterministik geri | aynı | gradcheck |
| 3.5 | Füzyonlu kaydırmalı CE: indeks ofseti, yerinde, ignore_index, label smoothing, z-loss | aynı | `test_ce_shift_no_copy` + FP64 şekil taraması |
| 3.5 | `LanguageTrainer` iki kopyası kaldırıldı | `include/nexus/train/language.hpp` | `test_nxs` (kayıp düşer) |
| 3.6 | Foreach AdamW + global norm kırpma (tek çağrı) | `training.cpp` | 1000 adım NexusOptim ile |
| 3.7 | Philox4x32-10 dropout, maske saklanmaz | `training.cpp` | Random123 KAT + yeniden üretim |

## 3. Sonuçlar

**ctest (CPU, Release): 26/26 geçti.** ASan (`build-asan`, RelWithDebInfo): 10 faz testi temiz.

| Ölçüt | Hedef (promt.txt) | Ölçülen |
|---|---|---|
| Parity nano — kayıp | max_rel < 1e-5 | **6.4e-08** |
| Parity nano — gradyan | max_rel < 1e-4 | **7.3e-07** (en kötü `h1.wv`), min kosinüs 0.9999999 |
| Parity tiny (L=6, d=384, V=8192; B=1, T=64) — kayıp / gradyan | aynı | **1.3e-07 / 4.9e-06** |
| 65 536 FP16 deseni gidiş-dönüş | 0 hata | 0; F16C donanımıyla ~93 M örnekte bit-bit |
| 65 536 BF16 deseni + RNE tablo | 0 hata | 0; NexusOptim `float16/bfloat16` ile bit-bit aynı |
| 10⁶ rastgele ayırma | sızıntı yok | canlı sayaç başa döndü, %50 önbellek isabeti |
| Foreach AdamW, 1000 adım | ≤ 1e-6 | NexusOptim skaler: **0 (bit-bit)**; AVX2 yolu: 4.6e-07 |
| CE yerinde, ikinci logit tamponu | yok | ledger tepe artışı 0, heap ayırma 0 |

Parity'de 5–7 tensör "gürültü tabanında" işaretlenir: dikkat key biası (`bk`) analitik
olarak sıfırdır (softmax satır kaymasına değişmez), 0.02 başlatmada ilk katman Q/K
gradyanları da ~1e-13'tür. Bunlarda bölüm anlamsızdır; `compare.py` mutlak hatayı global
gradyan ölçeğine böler ve `parity_report.json`'da `"noise_floor": true` yazar.

### Benchmark'lar (CPU, BÖLÜM G: ısınma + 5–7 tekrar, medyan)

Ham veri: `../2026-10-23-faz3/bench_phase3_training.json`.

| Ölçüm | Nexus | Karşılaştırma | Sonuç |
|---|---|---|---|
| Kaydırmalı CE, B·T=2048, V=50304 | 1066 ms, **0 MiB** ek heap | eski LanguageTrainer yolu: 1619 ms, **1178 MiB** ek heap | 1.52× hızlı, 3 logit kopyası yok |
| AdamW 124M parametre, ms/adım | foreach: 260 ms (+ global norm) | NexusOptim AVX2: 164 ms | **Nexus foreach 1.6× yavaş** (kütüphane SSE2 derleniyor; global norm ayrı geçiş) |
| RMSNorm fwd+bwd [4096,768] | 17.6 ms (3.6 GB/s) | LayerNorm: 33.5 ms (1.9 GB/s) | — |

### KURAL 9 regresyonu (21 Ekim benchmark'ları)

15 koşu medyanı vs 21 Ekim'in 3 koşu medyanı (`kural9_regresyon.json`):

| Ölçüm | 21 Ekim | Bugün | Fark |
|---|---|---|---|
| GEMM N=2048 | 361.9 GFLOPS | 391.1 | +8.1% |
| Conv2d | 5.43 GFLOPS | 6.68 | +23.0% |
| Depthwise | 10.35 GFLOPS | 11.20 | +8.2% |
| Attention | 9483 çağrı/s | 9447 | −0.4% |
| Adam | 24.8 GB/s | 34.1 | +37.3% |

Hiçbiri %3'ten fazla gerilemedi. Koşular arası yayılım büyüktür (attention tek koşuda
7281–10228); 5 koşuluk ilk turda attention −3.6% göründü, bu yüzden 15 koşuya çıkıldı.

### FAZ 0 temel çizgisi (`lm_baseline`, CPU, bugünkü klasik eğitmen)

| Yapılandırma | Parametre | Adım | token/s | Kayıp |
|---|---|---|---|---|
| nano (L=2, d=64, T=64, V=256, B=4) | 132 992 | 5.4 ms | 47 294 | 6.49 → 0.0023 (200 adım) |
| tiny (L=6, d=384, T=256, V=8192, B=2) | 16 946 432 | 866 ms | 591 | 10.02 → 5.75 (20 adım) |

## 4. Açık kalanlar (dürüst liste)

Bunlar promt.txt'de FAZ 0–3 altında yazılıdır ve bu denetimde **yapılmadı**:

- **FAZ 0:** gpt2s CPU temel çizgisi ve aynı modellerin PyTorch CPU/CUDA tablosu (0.F);
  ledger ile `nexus_estimate` arasında ±%5 uyumu — ledger yalnız `nexus` kapısını sayıyor,
  `nexus_model` tensörleri kendi ayırıcısını kullanıyor, bu yüzden ölçülemez.
- **FAZ 1:** Birleşik `Storage/TensorImpl` (stride, offset, view işlemleri, sürüm sayacı),
  `Stream/Event/Guard`, CUDA önbellekli ayırıcı ve pinned havuz, `TensorIterator`
  eleman-eleman üretici, DLPack, sıfır kopya köprüleri. Bugün yalnız host k64 havuzu ve
  dtype/half/bf16 katmanı var.
- **FAZ 2:** Cihaz başına iş parçacığı kuyruğu ve çok akışlı backward (CUDA), reentrant
  backward, `create_graph` (çift backward), Python `nexus.autograd.Function`, her katman için
  ayrı CustomFunction sarmalayıcısı (bugün genel `module_function` var), 2.G klasik ↔ autograd
  kayıp eğrisi eşitliği (yerine PyTorch parity yapıldı), motor ek yükünün PyTorch ile karşılaştırması.
- **FAZ 3:** GPU yolunun tamamı: cuBLASLt epilog/plan cache, CUDA LayerNorm/RMSNorm/softmax/
  embedding/CE/dropout/AdamW çekirdekleri, `ExecutionPolicy::cuda` uçtan uca, gpt2s'nin RTX 3070'te
  eğitimi ve `test_gpt2s_parity_fp32`. Bu denetimdeki FAZ 3 kodu CPU referans + üretim yoludur.

FAZ 4 (karışık hassasiyet) bu listedeki FAZ 1 dtype katmanını kullanabilir. GPU autocast ise
FAZ 1/3'ün GPU açıklarına bağlıdır.

## Çalıştırma

```
cmake --build build-phase2-cpu
ctest --test-dir build-phase2-cpu -E "^bench|nexus_master|competitor" --output-on-failure
ctest --test-dir build-phase2-cpu -R "bench_phase3_training" -V
build-phase2-cpu/tests/parity/test_parity_nano configs/nano.json
build-phase2-cpu/apps/lm_baseline/lm_baseline --config configs/nano.json --json out.json
```

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
