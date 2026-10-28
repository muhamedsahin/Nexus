# Faz 21 — Tekrarlanabilir performans karşılaştırması

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

## Kapsam

`nexus bench --matrix all` mevcut Nexus çalışma motorlarını çağırır; yeni bir GEMM,
attention veya eğitmen kopyası içermez. Çekirdek ölçümleri, tarihsel CPU matrisi,
gerçek tokenlarla 124M/1.3B eğitim ve hazır GPT-2 ağırlıklarıyla çıkarım aynı komutla
çalışır. Kullanılamayan yollar açık gerekçeyle kaydedilir.

**Bu rapor “dünyanın en hızlısı” iddiası değildir.** Faz 20'den >24 saat eğitim,
litgpt parity, bilimsel ölçekleme taraması ve küme kanıtları açık kalır.
Buradaki kısa eğitim koşuları bu ön koşulları kapatmaz.

## Mimari

- `benchmarks/runner/bench_phase21.cpp`: mevcut CPU GEMM, CUDA cuBLASLt,
  FlashAttention ileri/geri ve GPT-2 çıkarım motorlarını ölçer. GEMM çıktılarının
  tamamı, attention çıktısı ve üç gradyanın tamamı analitik referansla denetlenir.
- `python/src/nexus_bench/evidence.py`: örnek doğrulama, istatistik, karşılaştırma
  koşulları, %3 regresyon kontrolü, Markdown ve ayrı ölçekli SVG grafikleri.
- `python/src/nexus_bench/workers.py`: ayrı süreçlerde PyTorch ve isteğe bağlı
  TensorFlow/XLA ile JAX CPU referansları. Framework bulunmazsa 77 ile atlanır.
- `python/src/nexus_bench/cli.py`: sıralı çalıştırma, kaynak/ikili SHA-256, sürüm ve
  telemetri kaydı; mevcut Faz 20 eğitim aracını ve nanoGPT referansını kullanır.
- `windows_job.py`: çocuk süreç askıda oluşturulur, kill-on-close Windows Job'a
  atanır, ardından başlatılır. Timeout'ta başlatıcıyla birlikte bütün alt süreçler
  kapanır. POSIX'te ayrı süreç grubu aynı amaçla kullanılır.

Python ölçüm paketi `_core` uzantısını import etmez; rapor üretimi için yalnız
Python 3.11+ standart kütüphanesi gerekir. PyTorch/TensorFlow/JAX ayrı referans
yorumlayıcılarında isteğe bağlıdır. Mevcut Python API korunur.

## Protokol ve dürüst karşılaştırma

Ana çekirdek tablosu: beş bağımsız süreç × iki ısınma × yedi ölçüm. Süre, host
saatinden çağrı ve tamamlanma senkronizasyonuna kadar ölçülür. Medyan, ortalama,
min/max, nüfus standart sapması/varyansı ve CV saklanır. Çekirdek girdileri sabittir:
GEMM A=1/32, B=1/64; attention Q=K=0, V=1/4, dO=1/32. Bu sayede bütün çıktılar
analitik hesapla doğrulanır; sonuç genel rastgele matris dağılımlarına genellenmez.
PyTorch GEMM mikro referansı `torch.mm(out=...)` kullanır; compiled eğitim
referansıyla karıştırılmaz. Mikro GEMM için max-autotune üstünlüğü kanıtlanmaz.

Eğitim: aynı gerçek TinyStories/GPT-2 token dosyası, aynı Nexus başlangıç ağırlıkları,
tohum 1337, AdamW beta=(0.9,0.95), eps=1e-8, wd=0.1, LR=1e-4 ve aynı cosine tarifi.
Nexus 124M: CUDA Graph + mevcut GEMM autotune + füzyonlu CE/cast, tam recompute;
PyTorch: `torch.compile(mode="max-autotune")`, SDPA, BF16 ve fused AdamW.
GPU kartının SM sayısına göre PyTorch autotune seçimleri kısıtlanabilir; uyarılar
ham stderr kaydında tutulur. GELU tanh uyarlaması açıkça belirtilen nanoGPT tarifidir.

Her süreç 15 adım eğitir; ilk beş adım sıcak süre hesabından çıkarılır. İlk adım ve
toplam süreç süresi ayrıca kaydedilir: ilk adım derleme, capture ve gerçek hesabı
birlikte içerir; saf derleme süresi olarak sunulmaz. Kaynak satırı 1024 token,
geçerli hedef sayısı 1023'tür. Nexus 1024 konum işler, nanoGPT kaydırılmış girdide
1023 konum işler. Gerçek hedef token/s bu 1023 hedef üzerinden hesaplanır.

1.3B: BF16 ağırlık, FP32 ana kopya olmadan mevcut füzyonlu 8-bit Adam ve gecikmeli
global norm kırpma. Bu yolun eşdeğer bir PyTorch optimizer referansı yoktur;
1.3B token/s üstünlüğü **veri yok** olarak kalır. ZeRO-Offload kullanıldığı iddia edilmez.

MFU/HFU mevcut Faz 20 formülleriyle **tahmin** olarak saklanır. Donanım sayacı değildir.
Tepe tahsis/rezerv ve `cudaMemGetInfo` ayrı kavramlardır. Kernel bellek ölçümleri
doğruluk kontrolünden sonra başlatılır; ayırıcı rezervi daha önceki doğrulamanın
önbelleğini de içerebilir. Toplam GPU kullanımına masaüstü uygulamaları dahildir.

Çıkarım: gerçek GPT-2 124M, F16, 32 token istem + 32 token greedy üretim; iki ısınma,
yedi örnek, beş süreç. Her süreçte ve süreçler arasında token dizisi birebir denetlenir.
TTFT/TPOT sonucu 7B INT4 llama.cpp hedefinin kanıtı değildir.

Tarihsel CPU aracı ayrıca beş kez çalışır; içindeki eski 3/4-çağrı ortalamaları
aynen korunur ve bu protokol ayrı etiketlenir. Eski JSON'un “8 threads” etiketi
gerçek MatrixFlash pool sayısıyla uyuşmayabilir. Gerçek pool sayısı rapora girer;
bu tablodan eş işçi sayısı sağlanmadan rakip hız oranı üretilmez.

Karar kapıları:

- Eksik tekrar, NaN/Inf, negatif/sıfır süre, hatalı çıktı veya tutarsız veri reddedilir.
- Farklı shape/dtype/cihaz, CUDA derleme sürümü veya CPU işçi sayısı eşdeğer sayılmaz.
- Bağımsız süreç medyanlarında CV >%10 ise hız kararı verilmez.
- Baseline farklı makine/sürücü, bağımlılık sürümü veya işçi sayısındaysa karşılaştırma reddedilir.
- Gerçek karşılaştırılabilir ve kararlı baseline'da süre artışı >%3 ise komut başarısız olur.
- Eğitimde hız ve bellek koşullarına ek olarak `1e-3` mutlak kayıp eğrisi kapısı uygulanır.
  BF16 bu eşiği geçemezse sonuç korunur; tolerans genişletilmez.

Bu makinede Nexus CUDA 13.4, PyTorch paketi CUDA 13.2 ile derlenmiştir. D.2'nin
tam CUDA sürümü eşliği sağlanmaz. Ayrıca kurulu Windows PyTorch wheel'i gerçek
FlashAttention çekirdeği içermiyor. Bu hücre başka bir SDPA yoluyla sessizce
doldurulmaz. JAX CPU için sekiz işçi isteği aktarılır; iç XLA havuz sayısı doğrudan
denetlenmediğinden JAX sonucundan kesin eş-işçi üstünlüğü çıkarılmaz.

## Çalıştırma

Önce mevcut MSVC 2022 x64 / CUDA ortamında tam build:

```powershell
& 'C:/Program Files/CMake/bin/cmake.exe' --build build-phase1 --parallel 4
$env:PYTHONPATH=(Resolve-Path python/src).Path
& .bench-venv-py314/Scripts/python.exe -m nexus_bench bench --matrix all --out docs/benchmarks/2026-10-28-faz21/my-run
```

Kurulu `nexus` Python paketinin console entry point'i aynı işi şu komutla yapar:

```text
nexus bench --matrix all
```

Yeni çıktı dizini otomatik seçilir; mevcut sonuçların üzerine yazılmaz. CWD depo
kökü olmalıdır veya `--repo`, `--native` verilmelidir. Mevcut `.bench-venv-cuda`,
Faz 19 JAX venv'i ve Faz 20 `build-phase1/faz20/tokens.bin` varsa bulunur. Kullanıcının
`~/nexus-data/nanogpt` ve `~/nexus-data/gpt2/gpt2.nxg` dosyaları varsa kullanılır.
Başka veri veya ortam için açık seçenekler:

```text
nexus bench --matrix all --repo REPO --native NATIVE --reference-python TORCH_PYTHON --jax-python JAX_PYTHON --tokens REAL_INT32_TOKENS --nanogpt NANOGPT_DIR --inference-weights GPT2.nxg --out NEW_DIRECTORY
nexus bench --matrix kernels --baseline SAME_MACHINE_PREVIOUS_RUN --out NEW_DIRECTORY
nexus report --out EXISTING_RUN
```

`report` yeni ölçüm yapmaz; ham örneklerden aynı istatistikleri yeniden hesaplar ve
özetle uyuşmazlığı reddeder. `reference-requirements.lock` / `jax-requirements.lock`
kurulu bağımlılıkları pinler; kaynak ve ikili özetleri ilgili manifestlerde bulunur.
Derleme timeout'u varsayılan 600 saniye, genel worker timeout'u 180 saniyedir.

## Kanıt dizinleri

- `run-01` (yalnız yerel): geliştirme/tanılama. İlk Windows FlashAttention isteği açık hata verdi;
  doğru davranış sonraki sürümde “kullanılamıyor” olarak düzeltildi. Bu koşunun
  süreleri son sürüm üstünlük kararı için kullanılmaz.
- `run-02` (yalnız yerel): kısa tanılama, tamamlanmadan durduruldu; rapor değildir.
- `run-03`: tam matris; yayımlanan rapor ve ham veriler burada tutulur.
- `release-portability.log/xml`, `asan-portability.log/xml`: son odaklı host testleri.
- `sanitizer-*.log`: cuBLAS ve gerçek FlashAttention ileri/geri yolunun dört CUDA taraması.

Kaynak testleri: deterministik istatistik/rapor üretimi, eksik ve bozuk kanıtların
reddi, yanlış karşılaştırmalar, %3 regresyon mantığı ve alt süreç timeout temizliği.
`test_bench_deterministic` donanım saatinin sabitliğini iddia etmez; gerçek zamanlama
değişkenliği ayrıca CV kapısıyla değerlendirilir.

## Kamuya açıklık ve CI

Depo: [muhamedsahin/Nexus](https://github.com/muhamedsahin/Nexus).
`.github/workflows/benchmark.yml`, main kod güncellemeleri, PR, günlük program ve
manuel çalıştırmada kanıt testleri ve taşınabilir gerçek CPU benchmark'ı çalıştırır;
ham çıktıyı artifact olarak saklar. Paylaşılan GitHub runner'ları yerel GPU performans
kanıtı veya kararlı %3 baseline olarak kullanılmaz. CI durumu ayrıca doğrulanır;
yalnız workflow dosyası bulunması “CI yeşil” kabul edilmez.

**Doğrulanmış CI: [başarılı Linux çalışması](https://github.com/muhamedsahin/Nexus/actions/runs/40747929946)**
(`119c5d4` kaynak commit'i). Derleme, sekiz Python testi, gerçek CPU ölçümü ve
artifact yükleme geçti. API iş/artefakt kanıtları `ci-latest-*.json` ve
`ci-artifacts.json` dosyalarındadır. Rapor/ham veri yayını etiketi:
[`bench-faz21-2026-10-28`](https://github.com/muhamedsahin/Nexus/tree/bench-faz21-2026-10-28).
Bu etiket ölçüm anlık görüntüsüdür; tüm bilimsel kabul koşullarının geçtiği
anlamına gelmez.

## 28 Ekim ölçüm sonucu ve kabul durumu

Tam matris `run-03/REPORT.md` içindedir. RTX 3070 Laptop / Windows 11 / sürücü
610.88 üzerinde beş bağımsız süreç ölçüldü. Medyanlar:

| İş | Nexus | Referans | Karar |
|---|---:|---:|---|
| CPU FP32 GEMM 2048 | 49,6055 ms | PyTorch 46,8450 ms | Nexus CV %24; kararlı üstünlük yok |
| CUDA BF16 GEMM 2048 | 0,6572 ms | PyTorch 0,7241 ms | CUDA build farklı; geçerli üstünlük kanıtı yok |
| FlashAttention ileri + geri | 1,6474 ms | Kullanılamıyor | Windows PyTorch paketi FlashAttention içermiyor |
| 124M BF16 eğitim adımı | 66,5091 ms | compiled PyTorch 68,1469 ms | Sürüm ve parity koşulları geçmedi |
| 1.3B BF16 eğitim adımı | 906,4792 ms | Eşdeğer optimizer yok | Referansa üstünlük doğrulanamadı |
| GPT-2 124M F16 TTFT / TPOT | 2,526 / 1,3034 ms | Ölçülmedi | 7B INT4 karşılaştırması değildir |

124M kayıp eğrisinin en büyük mutlak farkı **0,002137184**, istenen 0,001
toleransını aşar. Tolerans gevşetilmedi; ham `parity_passed=false` korunur.
Nexus CUDA 13.4, PyTorch CUDA 13.2 kullanır. 1.3B optimizer önceki-adım clipping
kullanır; standart AdamW ile eşdeğer sayılmaz. TensorFlow bu ortamda kurulu değil;
JAX CPU gerçek ölçümü raporda yer alır. TTFT CV %10,36 olduğundan kararlı bir
gecikme üstünlüğü iddia edilmez.

**Faz 21 bütünüyle kapanmadı.** Doğrulama kaybına erişme süresi ölçümü, eşlenmiş
sürüm/referanslar, 1.3B eşdeğer optimizer, Faz 20 uzun eğitim/ölçek ön koşulları,
7B INT4/küme hedefleri ve sabit makinede CI performans baseline'ı açık kalır.
Paylaşılan CI runner'ındaki CPU ölçümü bu son koşulun yerine geçmez.

Son Release testleri **9/9**, ASan testleri **7/7** geçti. Faz 21 Python paketi
sekiz test içerir. CUDA'nın dört Compute Sanitizer aracı sıfır hata bildirdi;
racecheck sıfır hazard bildirdi. Bunlar odaklı doğrulamadır; bütün deponun test
paketinin yeniden çalıştırıldığı iddia edilmez. `verification.json` kaynak ve
ikili SHA-256 eşleşmesini, raporun byte-identical yeniden üretimini ve açık
kabul koşullarını kaydeder.

Ölçüm sırasında HEAD `77b7f0c` ve çalışma ağacı değişikti. Ölçülen sekiz kaynağın
SHA-256 değerleri yayımlanan **7f587f8** commit'iyle (LF/CRLF normalizasyonuyla)
eşleşir. Sonraki değişiklikler rapor sütunları, girdi doğrulama, CI ve standart
başlık/derleme bayrağı taşınabilirliğidir; performans sayıları sonradan yeniden adlandırılmadı.
PJRT'nin upstream C API 0.117 başlığına yalnız C++ global tip nitelemesi eklendi:
`::fn_type* fn_type`. ABI alan sırası, boyutları ve C tanımı değişmez. Linux GCC
bunu gerektirir; Release ve ASan eklenti ABI/cleanup testleri yeniden geçti.
