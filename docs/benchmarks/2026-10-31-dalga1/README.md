# Dalga 1 — Doğruluk Kapıları: LR=6e-4 Parity Kök Nedeni ve BF16 Parity (2026-10-31)

Kapsam: `plan.txt` Dalga 1 (1.1 LR=6e-4 kök neden, 1.2 FAZ 21 124M parity ≤ 1e-3, 1.3 1.3B eşdeğer optimizer referansı).

**Ortam:**

- Model: 124M GPT-2. Başlangıç SHA `8b9a7bb0…`, gerçek token dosyası SHA `0ccef04d…`.
- Nexus: `bench_cuda_phase20.exe` SHA `51dd5959…`.
- Referans: nanoGPT (`model.py` SHA `7c017032…`), PyTorch **2.15.0+cu134**, Nexus ile aynı CUDA 13.4 (Dalga 0). Deterministik algoritmalar açık, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Tolerans değiştirilmedi.** Eski noktasal kapı (`compare`, ≤ 1e-3) aynen duruyor; yeni kapılar onun yanına eklendi.

## 1.1 LR=6e-4: kök neden = kaotik büyüme, Nexus hatası değil

Soru şuydu: FAZ 20'deki 0.0437'lik fark bir hata mı? Önceki denetimde iki taraf da kendi tekrarında bit bit aynıydı, ama fark sürüyordu.

**Deney** (`benchmarks/runner/dalga1_noise_envelope.py`):

- PyTorch aynı tarifle 13 kez koşuldu: 1 kez bozmasız, 12 kez her parametresi rastgele işaretli **1 ULP** oynatılarak (`faz9_nanogpt.py --perturb-ulp`).
- Bu, referansın kendi FP32 yuvarlama hassasiyetini ölçüyor.

| LR = 6e-4, 100 adım | PyTorch ↔ PyTorch (78 çift) | Nexus ↔ PyTorch |
|---|---|---|
| En büyük kayıp farkı | 0.0125 – **0.0682** (medyan 0.024) | 0.031 – 0.068 |
| Noktasal ≤ 1e-3 geçen | **0 / 78** | 0 |
| İlk 1e-3 aşımı | 33–35. adım | 33. adım |
| İlk 10 adım | ≤ 9.5e-7 | 9.5e-7 |
| Son 20 adım ort. kayıp | 5.0737 ± 0.0045 (aralık 5.066–5.085) | 5.0691 (z = −1.04) |

LR = 1e-4'te topluluk zarfı 9.5e-7, Nexus da 9.5e-7.

**Sonuç:**

- LR = 6e-4'te eğitim kaotik. Ağırlıklardaki 1 ULP'lik herhangi bir fark 33–35. adımda 1e-3'ü aşıyor.
- Noktasal ≤ 1e-3 ölçütü bu LR'de **PyTorch'un kendisine karşı bile** sağlanamıyor (0/78).
- Nexus, PyTorch'un kendi yuvarlama topluluğunun içinde kalıyor.
- Nexus'un yeniden koşusu bit bit aynı çıktı. Başlangıç meta dosyası aynı SHA ile yeniden üretildi.
- Ek kontrol: ağırlık azaltma grupları (2B → decay, bias / LN → yok) nanoGPT ile birebir.

**Dürüst not:** Nexus'un üye skoru (0.0677) 13 PyTorch üyesinden 11'inin skorundan büyük, yani topluluğun üst ucunda. Bu beklenen bir durum: Nexus'un çekirdek toplama sıraları PyTorch'tan farklı olduğu için her adımda 1 ULP'lik tek seferlik bozmadan daha büyük bir yuvarlama farkı üretiyor. Hata kanıtı değil, ama "PyTorch'la aynı" da denemez.

### Yeni kapı: `faz20_validate.ensemble_compare` (v2)

Üç şartın hepsi gerekiyor:

1. İlk 10 adım (kaos öncesi): |Nexus − ref0| ≤ 1e-5.
2. **Üye skoru:** Nexus'un 100 adımdaki en büyük sapması (her referansa göre), referans üyelerinin en büyük skorunu aşmamalı. Taban: 2 FP32 ULP.
3. Son 20 adımın ortalama kaybı topluluğa göre |z| ≤ 3. Standart sapma tabanı: 2 ULP.

| LR | Noktasal (eski) | Topluluk v2 |
|---|---|---|
| 6e-4 | ❌ 0.0437 | ✅ ilk 10 adım 9.5e-7 · skor 0.0677 ≤ 0.0682 · z = −1.04 |
| 1e-4 | ✅ 9.5e-7 | ✅ skor 9.5e-7 ≤ 1.9e-6 · z = 0.09 |

**v1 tasarımı reddedildi** (kanıt `*-v1-rejected-design.json` olarak duruyor). v1 tabansız, adım adım bir zarf kullanıyordu ve **LR = 1e-4'te bile başarısız oldu**. İki nedeni var:

- FP32 kaybın 1 ULP'lik farkını, zarf 0 olduğu için hata saydı.
- Nexus'u tek başına 13 üyenin en uzağına karşı adım adım sınadı. Topluluğa 14. üye olarak eklenen herhangi bir PyTorch koşusu da bu sınamayı sık sık aşar.

v2 bu iki hatayı gidermek için tasarlandı. Tasarım verilere bakıldıktan sonra değiştirildiği için iki sürüm de raporda tutuluyor. Birim testleri `test_phase20_evidence.py` içinde (9 test):

- geçen durum;
- erken sapma, zarf dışı ve kuyruk sapması reddi;
- 1 ULP kuantizasyon;
- fail-closed (eksik referans, NaN, kısa eğri).

## 1.2 FAZ 21 BF16 parity (0.0021): kök neden = PyTorch BF16'nın kendi hatası

FAZ 21 tarifi: 15 adım, LR = 1e-4, warmup 2. Aynı başlangıç ağırlıkları kullanıldı (`dalga1_bf16_accuracy.py`).

| Kıyas | en büyük kayıp farkı |
|---|---|
| Nexus FP32 ↔ PyTorch FP32 | 9.5e-7 |
| **Nexus BF16 ↔ FP32 gerçeği** | **4.36e-4** |
| PyTorch `torch.compile` max-autotune BF16 ↔ FP32 | 1.83e-3 |
| PyTorch eager autocast BF16 ↔ FP32 | 6.23e-3 |
| PyTorch BF16 kendi 1 ULP topluluğu | 9.8e-4 |
| Nexus BF16 ↔ PyTorch compile BF16 (eski FAZ 21 ölçüsü) | 2.09e-3 (FAZ 21'deki 0.0021 tekrar üretildi) |

**Sonuç:**

- Eski kapı BF16'yı BF16'yla kıyaslıyordu. Böylece PyTorch'un kendi BF16 hatası Nexus'a yükleniyordu.
- Nexus BF16, FP32'ye derlenmiş PyTorch'tan **4.2×**, eager autocast'tan **14×** yakın.
- Olası neden (ölçülmedi, kod okumasından): Nexus artığa eklenen projeksiyon GEMM'lerini doğrudan FP32 çıktıyla yazıyor (`gpt.cpp` tip kuralları). PyTorch autocast'ta linear çıktısı önce BF16'ya yuvarlanıyor.

**Harness değişikliği** (`nexus_bench/cli.py`, `evidence.py`):

- 124M eğitim parity'si artık aynı ağırlıklı deterministik PyTorch FP32 eğrisine karşı ölçülüyor; Nexus BF16 ≤ 1e-3 olmalı.
- BF16 ↔ BF16 farkı bilgi sütunu olarak kalıyor.
- Yeni sütunlar yalnız yeni kayıtlarda çıkıyor: yayımlanmış FAZ 21 raporu eski kayıttan **bayt bayt aynı** yeniden üretiliyor (test 14/14, Python 3.14 ve 3.12).

**Uçtan uca harness koşusu** (`harness-training-e2e/`, fişte, 1 tekrar):

| | Sonuç |
|---|---|
| Nexus BF16 ↔ FP32 gerçeği | **4.53e-4** → parity ✅ |
| PyTorch compile BF16 ↔ FP32 | 1.90e-3 |
| CUDA sürümü | iki taraf da 13040 → FAZ 21'deki `no_data` gerekçesi kalktı |
| 124M hız | Nexus **68.6 ms**, torch.compile **65.4 ms** → **%4.9 gerideyiz** → iddia `failed` |
| Bellek | ✅ Nexus ≤ PyTorch |

Tek tekrar olduğu için hız sonucu istatistiksel değil. 5 tekrarlı koşu ve hızın kendisi Dalga 3'ün işi. Artık eğitim iddiasını tutan engel **ölçüm kuralı değil, hız**.

## 1.3 1.3B eşdeğer optimizer referansı

**Sorun:**

- FAZ 21'in 1.3B ölçümü `fused8` kullanıyordu (8-bit AdamW, FP32 ana kopya yok, gecikmeli kırpma). PyTorch'ta bunun karşılığı yok → `no_data`.
- 8 GB'ta PyTorch'un standart AdamW'si 1.3B'yi taşıyamıyor.
- Windows'ta DeepSpeed yok.

**Çözüm: iki tarafta aynı algoritma.** FP32 ana kopya + AdamW host'ta, gradyanlar katman katman host'a, BF16 ağırlık H2D.

- **Nexus:** `--optimizer offload` (FAZ 7 ZeRO-Offload, FAZ 18 AVX2 Adam). 1.3B için host'ta 17.1 GiB gerekiyor; bu ölçümde 23.3 GiB boştu (FAZ 7'de yetmemişti).
- **PyTorch:** `benchmarks/runner/torch_offload_ref.py` (yeni, elle ZeRO-Offload):
  - `register_post_accumulate_grad_hook` ile gradyan oluşur oluşmaz ayrı akışta pinned CPU'ya, GPU kopyası hemen bırakılıyor;
  - CPU'da FP32 ana kopya + `torch.optim.AdamW(fused=True)`, nanoGPT grupları;
  - blok başına checkpointing;
  - kayıp FP32 logit üzerinden;
  - BF16 modunda parametreler GPU'da saf BF16 (8 GB'ta autocast + FP32 parametre sığmıyor), pinned tampon gradyan ve ağırlık için ortak (Nexus gibi);
  - FP32 modu 1.3B'nin FP32 "gerçek" eğrisi (TF32 kapalı).

### Referansın doğrulanması (124M FP32, LR = 1e-4, 15 adım)

| Referans sürümü | Standart nanoGPT FP32'ye fark |
|---|---|
| İlk sürüm (CPU'da `torch.nn.utils.clip_grad_norm_`) | 1.18e-4 ❌ |
| foreach CPU AdamW | 1.18e-4 (aynı) |
| checkpoint kapalı | 1.18e-4 (aynı) |
| kanca yok, CPU kırpma + CPU AdamW | 1.18e-4 (aynı) |
| **GPU'da kırpma + CPU AdamW** | **0** |
| GPU fused / GPU foreach AdamW (offload yok) | 0 / 9.5e-7 |
| **Son sürüm (FP64 normlu kırpma, kancalar, checkpoint, CPU AdamW)** | **0** (Nexus FP32'ye 9.5e-7) |

**Yan bulgu:** PyTorch'un CPU'daki `clip_grad_norm_`'u 124M'de CUDA'daki aynı çağrıdan 1.2e-4 kayıp farkı üretti.

- CPU fused AdamW, CUDA fused AdamW ile kayıp düzeyinde bit bit aynı çıktı. Farkın tamamı kırpma normundan geliyor.
- Olası neden: büyük tensörlerde (wte 38M öge) CPU FP32 norm indirgemesi. Bu, kod incelemesi değil, A/B'den çıkan bir çıkarım.
- Referans normu FP64'te hesaplıyor ve PyTorch'un FP32 katsayı ifadesini aynen uyguluyor. Nexus offload da normu FP64'te hesaplıyor.
- Yani DeepSpeed'siz, saf bir PyTorch CPU offload'u ek bir hata kaynağı taşıyor.

### 1.3B sonuçları (15 adım, LR = 1e-4, warmup 2, aynı başlangıç SHA `5686cace…`)

Doğruluk ilk elle koşudan (`offload-1.3b/`), hız fişteki uçtan uca harness koşusundan (`harness-1.3b-e2e/`, `BatteryStatus` = 2, **1 tekrar**):

| | Nexus BF16 offload | PyTorch BF16 offload | PyTorch FP32 offload (gerçek) |
|---|---|---|---|
| FP32 gerçeğine en büyük kayıp farkı | **2.24e-3** | 2.12e-2 (harness tekrarında 1.86e-2, deterministik değil) | — |
| ms / adım (medyan, ilk 5 adım hariç) | **2955** | 5621 | 10975 (pilde) |
| GPU tepe rezerv | **4.92 GB** | 6.58 GB | 9.72 GB (8 GB VRAM'i aştı, sistem belleğine taştı; yalnız doğruluk referansı) |
| CPU güncellemesi / adım | ölçülmedi (bu koşuda ayrı yazılmıyor) | ~5.6–6.0 s (torch CPU AdamW + kopyalar) | ~5.7 s |

**Karar (harness `1.3b_claim`, yeni):** `passed`.

- **Hız:** Nexus 1.90× hızlı.
- **Bellek:** Nexus daha az GPU belleği kullanıyor.
- **Parity:** Nexus BF16, PyTorch BF16'dan FP32 gerçeğine 8–9× daha yakın.

**Dürüst sınırlar:**

- **Mutlak ≤ 1e-3 1.3B'de geçmiyor** (2.24e-3; `strict_1e3_passed: false` olarak ayrıca yazılıyor). Fark daha 0. adımda, yalnız ileri geçişte 8e-4. 1.3B'nin BF16 ileri geçişi FP32'den bu kadar ayrışıyor. Bu yüzden 1.3B kapısı "aynı algoritmayı koşan referanstan kötü değil" olarak tanımlandı; mutlak eşik 124M için kalıyor.
- Hız sonucu **1 tekrar**: 5 tekrarlı tam matris koşulmadı, CV ölçülmedi.
- PyTorch referansı DeepSpeed değil, bu depoda yazılmış elle bir offload. Daha iyi optimize edilmiş bir PyTorch offload'u (örneğin GPU-CPU örtüşmeli) daha hızlı olabilir.
- `fused8` (Nexus'un en hızlı 1.3B yolu, FAZ 21'de 906 ms/adım) hâlâ referanssız; bilgi amaçlı ölçülmeye devam ediyor.

**Harness:** `nexus_bench/cli.py` → `offload_claim()`:

- 5.25 GB başlangıç dosyası kanıt dizinine yazılmıyor, SHA kaydediliyor;
- FP32 gerçeği tarif başına bir kez koşuluyor;
- 124M iddiasının PyTorch koşu süzgeci `model == '124m'` ile sınırlandı.

## Açık kalanlar

- BF16 farkının kod düzeyinde nedeni (projeksiyon çıktı tipi) ayrıca izole edilip ölçülmedi.
- LR = 6e-4 BF16 topluluk kıyası yapılmadı (yalnız FP32 ve LR = 1e-4 BF16).
- 124M hızında torch.compile'dan %4.9 gerideyiz (1 tekrar).

## Dosyalar

- `noise-envelope/`: 13 + 5 PyTorch eğrisi, Nexus eğrisi ve yeniden koşu. `gate-summary-v2.json`, `ensemble-parity-v2.json`, `pointwise-parity.json`, reddedilen `*-v1-*`.
- `bf16-accuracy/`: `bf16-accuracy.json` (derlenmiş dahil), `bf16-accuracy-eager-only.json` (triton kurulmadan önceki koşu).
- `harness-training-e2e/`: `training.json` ve ham eğriler. 475 MB başlangıç dosyası kopyalanmadı, SHA'sı `initial.bin.sha256` içinde.
- `offload-1.3b/`: 1.3B elle koşular (Nexus BF16 offload, PyTorch FP32 / BF16 offload) ve 124M referans doğrulama A/B'leri (`check124m-*`, `adamw-impl-check*`). 5.25 GB başlangıç kopyalanmadı (SHA `initial.bin.sha256`).
- `harness-1.3b-e2e/`: `offload-claim.json` (fişte, 1 tekrar).
- Betikler: `benchmarks/runner/dalga1_noise_envelope.py`, `dalga1_bf16_accuracy.py`, `torch_offload_ref.py`, `faz9_nanogpt.py --perturb-ulp`.

Ek indirme (kullanıcı onaylı): `triton-windows 3.9.0.post30` `.bench-venv-cu134` içine kuruldu (torch.compile için). Kilit dosyası `../2026-10-30-dalga0/requirements-cu134.lock` güncellendi.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
