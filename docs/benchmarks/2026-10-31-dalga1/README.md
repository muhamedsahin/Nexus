# Dalga 1 — Doğruluk Kapıları: LR=6e-4 Parity Kök Nedeni ve BF16 Parity (2026-10-31)

Kapsam: `plan.txt` Dalga 1 (1.1 LR=6e-4 kök neden, 1.2 FAZ 21 124M parity ≤ 1e-3). 1.3 (1.3B eşdeğer optimizer) bu dalgada **yapılmadı**.

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

## Açık kalanlar

- 1.3: 1.3B için PyTorch eşdeğeri optimizer (fused8 gecikmeli kırpmanın referansı yok) → `no_data` sürüyor.
- BF16 farkının kod düzeyinde nedeni (projeksiyon çıktı tipi) ayrıca izole edilip ölçülmedi.
- LR = 6e-4 BF16 topluluk kıyası yapılmadı (yalnız FP32 ve LR = 1e-4 BF16).
- 124M hızında torch.compile'dan %4.9 gerideyiz (1 tekrar).

## Dosyalar

- `noise-envelope/`: 13 + 5 PyTorch eğrisi, Nexus eğrisi ve yeniden koşu. `gate-summary-v2.json`, `ensemble-parity-v2.json`, `pointwise-parity.json`, reddedilen `*-v1-*`.
- `bf16-accuracy/`: `bf16-accuracy.json` (derlenmiş dahil), `bf16-accuracy-eager-only.json` (triton kurulmadan önceki koşu).
- `harness-training-e2e/`: `training.json` ve ham eğriler. 475 MB başlangıç dosyası kopyalanmadı, SHA'sı `initial.bin.sha256` içinde.
- Betikler: `benchmarks/runner/dalga1_noise_envelope.py`, `dalga1_bf16_accuracy.py`, `faz9_nanogpt.py --perturb-ulp`.

Ek indirme (kullanıcı onaylı): `triton-windows 3.9.0.post30` `.bench-venv-cu134` içine kuruldu (torch.compile için). Kilit dosyası `../2026-10-30-dalga0/requirements-cu134.lock` güncellendi.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
