# Faz 20 — Ölçek doğrulama (28 Ekim 2026)

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

## Sonuç ve kapsam

**Faz 20 tamamen kapanmadı.** Gerçek 124M modelde 100 adım nanoGPT parity ve tek RTX 3070 Laptop
8 GiB üzerinde 124M/350M/1.3B eğitim ölçümleri elde edildi. >24 saat eğitim, litgpt 1.3B parity,
7B küme ölçümü ve bilimsel ölçekleme yasası taraması henüz doğrulanmadı. Faz 21'e geçilmedi.
Faz 19'daki HIP/Metal ve gerçek satıcı PJRT donanım doğrulama eksikleri bu çalışma ile kapanmış sayılmaz.

## Mimari

`benchmarks/runner/bench_cuda_phase20.cpp` mevcut `GptModel`, `LanguageTrainerBuilder`, CUDA ayırıcı
ve NXS3 kayıt yöneticisini kullanır. Yeni bir eğitmen veya çekirdek kopyası eklenmedi.
Standart AdamW yolunda mevcut NaN durdurma politikası, deterministik dikkat, tam aktivasyon yeniden
hesaplama, 256 satırlık parçalı CE ve güncel adımın global normuyla kırpma kullanılır.

1.3B yolunda mevcut `AdamW8bitFused`, BF16 ağırlık, stokastik yuvarlama ve önceki adımın global
normuyla gecikmeli kırpma kullanılır. FP32 ana ağırlık kopyası yoktur. Bu yol standart AdamW ile aynı
optimizer algoritması değildir; nanoGPT/litgpt parity kanıtı olarak sunulmaz. Ölçülen 1.3B yolu
**ZeRO-Offload kullanmaz**. Birleşik offload + 8-bit optimizer desteği iddia edilmiyor.

`BinarySource` yalnız bir yığın okur; token aralığı ve kısa okuma denetlenir. Veri SHA-256'sı ve imleci
kayda girer. Devamda farklı veri, model, dtype, optimizer, LR, seed veya zamanlayıcı tarifi açık hata
verir. Çıktılar mevcut kanıtların üzerine yazılmaz. Periyodik kayıtlar mevcut atomik NXS3 protokolünü kullanır.

## 124M parity

TinyStoriesV2-GPT4 korpusunun mevcut GPT-2 BPE/NXTK verisinden ardışık gerçek tokenlar kullanıldı.
Vocab 50257, 12 katman, d=768, 12 kafa, 124439808 parametre, mikro yığın 1, T=1024;
1023 tahmin hedefi/adım. İki uygulama aynı başlangıç ağırlıklarını ve token dosyasını okur.
AdamW beta=(0.9,0.95), eps=1e-8, wd=0.1; global norm kırpma 1; warmup 10, cosine 100,
minimum LR oranı 0.1. FP32 ve TF32 kapalı.

Referans: nanoGPT `model.py` SHA-256
`7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`,
PyTorch 2.14.1+cu132. Nexus ile eşlemek için GELU açıkça tanh'a uyarlanır ve vocab dolgusu kapatılır.
Öğrenilen konum tablosu 1024 satırıyla korunur. Dolayısıyla bu uyarlanmış nanoGPT tarifinin parity'sidir.
[nanoGPT kaynak tanımı](https://github.com/karpathy/nanoGPT/blob/3adf61e/model.py).

| Tarif | 100 adım en büyük mutlak kayıp farkı | ≤1e-3 |
|---|---:|---|
| LR=6e-4 ilk deneme | 0.0428066 | Geçemedi |
| LR=1e-4 kontrollü tarif | 0.000000954 | Geçti |

İlk başarısız sonuç [parity-124m.json](parity-124m.json) içinde korunur. Daha düşük LR sonucu
[parity-124m-lr1e4.json](parity-124m-lr1e4.json) içindedir. Yüksek LR'daki ayrışmanın kök nedeni
kanıtlanmadı; sonuç tüm öğrenme hızlarına genellenmez. Tolerans değiştirilmedi.
Son kaynak/binary ile tek komut üzerinden yeniden yapılan 100 adımlık doğrulama da geçti:
[final-parity.json](final-parity.json), [komutlar ve SHA kayıtları](final-parity-metadata.json),
[Nexus ham eğrisi](final-nexus.jsonl), [nanoGPT ham eğrisi](final-nanogpt.json).

## Tek GPU ölçek tablosu

Her hücre beş bağımsız süreç; süreç başına 30 eğitim adımı, ilk 5 adım ısınma dışı bırakılır.
Tablo beş süreç medyanının medyanıdır; ham süreler, min/max/std, loss, kaynak/binary SHA ve
donanım kayıtları [scale/matrix.json](scale/matrix.json) ile [scale/metadata.json](scale/metadata.json) içindedir.
Veri aynı gerçek token dosyası, B=1, T=1024, BF16, tam recompute, LR=6e-4, warmup=5.

| Model | Optimizer | Medyan adım ms | Hedef token/s | Tepe rezerv GiB | İlk → son kayıp |
|---|---|---:|---:|---:|---:|
| 124M | AdamW FP32 durum | 457.635 | 2235.4 | 2.348 | 11.001 → 5.949 |
| 350M | AdamW FP32 durum | 1121.045 | 912.5 | 6.303 | 10.977 → 5.989 |
| 1.3B | Füzyonlu 8-bit Adam | 862.814 | 1185.7 | 6.084 | 11.165 → 6.224 |

1.3B'nin optimizer ve bellek yolu farklıdır; tablodan model büyüdükçe daha hızlı olduğu sonucu çıkarılmaz.
Bu oturumda rakip BF16 throughput ölçümü yapılmadı; en hızlı veya PyTorch'tan hızlı iddiası yoktur.
Laptop güç/saat durumu hızları etkileyebilir; koşu sonu telemetri her süreç kaydına eklenmiştir.

MFU tahmini, konum parametreleri hariç `6N + 12LdT` model FLOP formülü ve ölçülen adım süresiyle
hesaplanır. BF16 dense tepe NVML azami SM saati üzerinden mevcut donanım tablosundan gelir.
Strict FP32'de TF32 tepesi kullanılmaz: MFU bilinmiyor (`null`). Tam recompute HFU değeri
`4/3 * MFU` yaklaşımıdır; donanım sayacı ölçümü değildir ve optimizer FLOP'larını içermez.
Parçalanma, `inactive_split_bytes / reserved_bytes` olarak tanımlanır; tepe rezerv, tahsis ve host
durumu ayrı tutulur. Bunlar toplam sistem/GPU süreç belleğiyle aynı ölçüt değildir.

## Doğrulama

- Tam Release/CUDA ve mevcut ASan yapılandırması derlendi.
- [Release testleri](release-tests.log): 7/7 geçti; yeni kanıt testleri, ayrı süreçte BF16 kayıt/devam,
  NaN, füzyonlu 8-bit güncelleme, recompute ve parçalı CE regresyonları.
- [ASan son koşu](asan-final.log): 7/7 geçti. İlk denemede MSVC ASan DLL'i PATH'te olmadığından
  süreçler 0xc0000135 ile açılmadı; runtime yolu eklenerek testler yeniden çalıştırıldı.
- [Kısa dayanıklılık](endurance-smoke/endurance.json): periyodik NXS3 sonrası yeni süreçte kayıp/LR
  birebir; `passed_24h=false`. Bu bir >24 saat koşusu değildir.
- [Kayıt/devam testi](resume-short.log): kesintisiz 8 adım = 4 + kayıt + yeni süreç + 4;
  değişmiş tarif ve değişmiş veri reddedildi.
- CUDA Compute Sanitizer günlükleri `sanitizer-*.log`: küçük BF16 eğitim yolu üzerinde
  memcheck, racecheck, initcheck ve synccheck; tam 1.3B model sanitizer altında çalıştırılmadı.

## Yeniden üretim

Depo kökünde, mevcut MSVC/CUDA ortamıyla `cmake --build build-phase1 --target bench_cuda_phase20`.
Bu makinede CMake tam yolu `C:/Program Files/CMake/bin/cmake.exe`.
`Python3_EXECUTABLE` çalışır bir Python'a yöneltilmelidir.

```powershell
$py = '.bench-venv-py314/Scripts/python.exe'
$exe = 'build-phase1/benchmarks/runner/bench_cuda_phase20.exe'
& $py benchmarks/runner/faz20_validate.py prepare --manifest C:/Users/muham/nexus-data/tinystories/nxtk-train/manifest.json --count 1054720 --out build-phase1/faz20/new-tokens.bin
& $py benchmarks/runner/faz20_parity.py --exe $exe --tokens build-phase1/faz20/new-tokens.bin --python .bench-venv-cuda/Scripts/python.exe --nanogpt C:/Users/muham/nexus-data/nanogpt --out build-phase1/faz20/new-parity
& $py benchmarks/runner/faz20_scale.py --exe $exe --tokens build-phase1/faz20/new-tokens.bin --out build-phase1/faz20/new-scale
```

Veri hazırlama her kullanılan NXTK parçasının SHA-256'sını ve tokenizer başlığını doğrular.
Sıra/EOS bilgisi korunur; belge sınırında dikkat sıfırlanmaz. Bu, iki tarafta aynı ardışık LM tarifidir.
Veri manifesti ve dilim özeti `tokens-provenance.json` içindedir. Ağırlık ve token ikilileri boyut
nedeniyle ignored build dizinindedir; komutlar bunları yeniden üretir.

Uzun koşu için önce yeterli gerçek veri hazırlanmalıdır (varsayılan 200000×1024 token).
Yeni çıktı diziniyle aşağıdaki komut ön planda çalışır; gerçek eğitim süresi >86400 saniye ve
başarılı ayrı-süreç devam karşılaştırmaları olmadan `passed_24h` üretilmez:

```powershell
& $py benchmarks/runner/faz20_longrun.py --exe $exe --tokens LARGE_REAL_TOKENS.bin --out build-phase1/faz20/endurance-24h --min-seconds 86401
```

## Açık kabul koşulları

1. >24 saat gerçek eğitim ve uzun dönem sayısal dayanıklılık ölçülmedi.
2. litgpt ile LLaMA tarzı 1.3B parity ölçülmedi; GPT tabanlı 1.3B kapasite testi bunun yerine geçmez.
3. 7B FSDP/TP küme erişimi yok; ≥%50 MFU kanıtı yok.
4. Model×veri boyutu ve sabit doğrulama kümesi taraması yapılmadı. Bu 30 adımlık eğitim kayıplarından
   Chinchilla katsayısı veya ölçekleme yasası çıkarılmaz. Bilimsel kapsam için
   [Hoffmann ve arkadaşları](https://arxiv.org/abs/2203.15556) referans alınır.
5. Linux derlemesi, eski 21 Ekim çekirdek matrisinde ≤%3 regresyon sınırı ve tam depo test paketi
   bu fazda yeniden doğrulanmadı. Çalıştırılan kapsam yukarıda açıkça belirtilmiştir.
