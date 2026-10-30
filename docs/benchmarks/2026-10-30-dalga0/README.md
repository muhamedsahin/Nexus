# Dalga 0 — Ağaç Doğrulama + Referans Ortamı (2026-10-30 / 31)

Kapsam: `plan.txt` Dalga 0 (0.1–0.4).

- **Ağaç:** `git` HEAD `261cc06` artı çalışma ağacındaki **commit'lenmemiş** değişiklikler (workflow, faz20/21 Python, testler, `tools/nexus_run`; 18 dosya). Doğrulanan, bu ikisinin birleşimi.
- **Makine:** Ryzen 7 5800H, RTX 3070 Laptop (sürücü 610.88), CUDA 13.4, MSVC 2022.
- **Güç:** testler **pilde** koştu (`Win32_Battery` = 1). Bu dalgada zaman ölçümü **yok**; aşağıdaki süreler yalnız test süresidir.

## 0.1 Tam test paketi

| Derleme | Sonuç | Süre | Kanıt |
|---|---|---|---|
| Release + CUDA (`build-phase1`) | ✅ **213 / 213 geçti**. Atlanan 2 test (donanım / eklenti yok, SKIP 77): `test_hip_parity`, `test_pjrt_vendor` | 57.6 dk | `gpu_ctest.{log,xml}` |
| Release CPU (`build-phase2-cpu`) | ✅ **163 / 163 geçti** (aynı 2 test atlandı) | 13.8 dk | `cpu_ctest.{log,xml}` |
| Debug + ASan (`build-asan`) | ✅ **AddressSanitizer hatası 0**. 162 testten 160 geçti, 2'si zaman aşımı; düzeltmeden sonra ikisi de temiz (aşağıda) | 112.6 dk | `asan_ctest.{log,xml}`, `asan_rerun.log` |

27 Ekim regresyonundaki zaman aşımları güncel ağaçta **tekrar etmedi**:

- `test_cpu_train_parity`: 38 s;
- `bench_cuda_phase5`: 94 s.

O koşu eski ağaçtaydı.

## 0.2 Zaman aşımları (ASan)

| Test | Neden | Düzeltme |
|---|---|---|
| `test_deep_stack_stability` | 32/48 katman × 500 adım eğitimi Debug + ASan'da 3600 s'yi aşıyor. Bellek hatası değil. | ASan derlemesinde `deep 120 asan` modu çalışıyor: bellek ve NaN denetimi tam yapılıyor, öğrenme ölçütü **SKIP** yazdırılıyor. Neden: 120 / 200 / 300 adım Release'te denendi; L=48, lr=3e-3 için öğrenme ölçütü 500 adımın altında tutmuyor. Ölçütü gevşetmek yerine Release'te 500 adımla bırakıldı. `tests/parity/CMakeLists.txt`, `test_phase6_lm.cpp` |
| `bench_attention_cpu` | Benchmark, ASan altında 600 s'yi aşıyor. Ayrıca bench'ler `docs/benchmarks` altındaki rapor JSON'larının üzerine yazıyor. | ASan derlemesinde `benchmarks/runner` altındaki tüm ctest girdileri `DISABLED` (20 bench). Çalıştırılabilir dosyalar yine derleniyor. `benchmarks/runner/CMakeLists.txt` |

ASan yeniden koşusu (yalnız bu iki nokta, `asan_rerun.log`):

- `test_deep_stack_stability` ✅ **715.7 s**: 4 NaN'sız ölçütü geçti, 4 öğrenme ölçütü SKIP, AddressSanitizer hatası 0.
- `bench_*` girdileri "Not Run (Disabled)".

Böylece ASan paketi 162 / 162 temiz. Tam ASan paketi bu değişiklikten sonra baştan koşulmadı: değişiklik yalnız bu iki kaydı etkiliyor.

ASan yapılandırmasının `NEXUS_VALIDATION_OUTPUT_DIR` değeri kullanıcının `2026-10-29-plan` dizinine yazıyordu. Bu koşuda `2026-10-30-dalga0/asan-generated`'a yönlendirildi; önceki kanıtların üzerine yazılmadı. Release ctest'lerinin ezdiği FAZ 0–6 bench JSON'ları `git checkout` ile geri alındı.

## 0.3 Referans ortamı (kullanıcı onaylı indirmeler)

| Referans | Sürüm | Yer | Not |
|---|---|---|---|
| PyTorch CUDA | **2.15.0+cu134**, CUDA 13.4, cuDNN 9.26 | `.bench-venv-cu134` | Nexus'la **aynı CUDA toolkit**. FAZ 21'deki "13.4 vs 13.2 → no_data" engeli kalktı. Eski `.bench-venv-cuda` (cu132) dokunulmadan duruyor. |
| TensorFlow | **2.21.0**, CPU (oneDNN) | `.bench-venv-tf` (Python **3.12.10**, `py install 3.12`) | TF Windows'ta Python 3.14 tekerleği sunmuyor. Windows'ta TF GPU yok (TF ≥ 2.11); GPU TF için WSL2 gerekir → GPU TF kıyası şimdilik ⛔. |
| JAX | **0.12.0.1**, CPU (XLA) | `.bench-venv-jax` | |
| llama.cpp | commit **792fcf4** (2026-10-30), MSVC 19.44 | `~/nexus-data/llama.cpp`, `build-cpu` (AVX2 native) ve `build-cuda` (sm_86) | `llama-bench` iki derlemede de çalışıyor, CUDA0 = RTX 3070 Laptop. |

Bağımlılık kilitleri: `requirements-{cu134,tf,jax}.lock`.

Duman denemesi: harness worker'ları her yorumlayıcıda koştu ve doğruluk kapısından geçti.

- TF 2.21.0: CPU GEMM;
- JAX 0.12.0.1: CPU GEMM;
- torch 2.15.0+cu134: CPU GEMM, CUDA GEMM, CUDA attention.

Süreler **geçersiz** (pil + arka planda ASan); rapora girmez.

## 0.4 Harness (`python/src/nexus_bench`)

Kullanıcının commit'lenmemiş işinde zaten var olanlar:

- süreç içi CV, süreç medyanlarının IQR'ı, 2000 örnekli bootstrap güven aralığı;
- zamanlama ısınması ile scheduler ısınmasının ayrımı;
- ilk adım süresi;
- GPU saat ve sıcaklık telemetrisi.

Bu dalgada eklenenler (`cli.py`, yalnız ekleme):

| Değişiklik | Neden |
|---|---|
| `--tensorflow-python` (varsayılan `.bench-venv-tf`) ve `--jax-python` için `.bench-venv-jax` varsayılanı. Her framework kendi yorumlayıcısında koşuyor. | TF ve JAX, torch ile aynı venv'de kurulamıyor (Python sürümü). |
| Varsayılan `--reference-python`: `.bench-venv-cu134` varsa o, yoksa eski cu132 | K2: aynı CUDA sürümü. |
| `metadata.json` → `power` (BatteryStatus, `on_ac`) | K5: pilde alınan ölçüm rapordan ayırt edilebilsin. |
| TF ortam kilidi `tensorflow-environment-lock.json` | Tekrarlanabilirlik |

`test_phase21_bench` (harness birim testi) değişiklikten sonra ✅ geçti.

**Yapılmadı:** pilde ölçümü harness'ın otomatik olarak `no_data`'ya çevirmesi. Şimdilik yalnız kaydediliyor; kapı mantığı Dalga 1'de `claims()` içine eklenecek.

## Sonuç ve sıradaki adım

Dalga 0 kapısı:

- tam paket yeşil ✅;
- ASan bellek hatası 0 ✅;
- referanslar aynı sürümde kurulu ve koşuyor ✅;
- CV ≤ %10 kapısı ölçüm gerektiriyor → fişte FAZ 21 matrisi tekrar koşulunca doğrulanacak (Dalga 1'in ilk adımı).

Dalga 1: LR=6e-4 parity kök neden analizi + 124M parity ≤ 1e-3, sonra cu134 torch'la aynı sürüm kıyası.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
