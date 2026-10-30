# Plan — ilk doğrulama ve ölçüm düzeltmeleri

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

## Kapsam ve durum

`plan.txt` bütünüyle okundu. Bu çalışma Aşama 0 doğrulamasını ve Aşama 1'in ilk
ölçüm hatalarını ele alır. Model matematiği, optimizer veya GPU çekirdekleri
değiştirilmedi. Planın bütün aşamalarının kapatıldığı iddia edilmez.

29–30 Ekim koşuları tamamlandı. **Release: 211 geçti, 2 atlandı, 0 hata
(213 hedef). Debug + ASan: 158 geçti, 2 atlandı, 4 başarısız (164 hedef).**
İki hata dar kapsamlı düzeltmelerden sonra odak testlerde geçti; iki zaman aşımı
açık kaldı. Dolayısıyla Aşama 0 veya bütün plan tamamlandı denmez.
Makine tarafından okunabilir son durum: [verification.json](verification.json).
Tam paketler başlangıç doğrulamasıdır; son değişikliklerin tamamını içeren yeni
bir tam paket koşusu değildir. Değişen hedefler ayrıca derlenip sınandı.

## Düzeltilen hatalar

1. **Süreç medyanları jitter'ı gizliyordu.** Her süreçte `[1,1,1,1,1,1,100]`
   ölçüldüğünde medyanlar aynı olduğundan eski kapı CV=0 görüp PyTorch'a karşı
   `passed` üretiyordu. Gerçek süreç içi CV yaklaşık 2,2877'dir. Yeni kapı hem
   süreç içi hem süreçler arası CV için %10 sınırını uygular; bu örnek `no_data`
   olur. Önceki hata `01-baseline/variance-gate-reproduction.json` içinde saklandı.
2. **Aynı tekrar bağımsız sayılabiliyordu.** Beş adet `repeat=0` eskiden kabul
   ediliyordu. Artık framework/tekrar kimlikleri benzersiz ve tam sayı olmalıdır;
   eksik, yinelenmiş veya aralık dışı kayıtlar reddedilir.
3. **Tahmin testi yapılandırılmış Python'u kullanmıyordu.** PATH'teki yönetici
   ilk koşuda 180 saniyede zaman aşımına uğradı. Yapılandırılmış yorumlayıcıyla
   aynı test 1,8 saniyede geçti. CMake artık bu yolu teste verir; bozuk bir açık
   yapılandırma sessizce başka yorumlayıcıya düşmez. Python PATH'ten çıkarılmışken
   Release ve ASan'da üç gerçek C++/Python parity örneği geçti; test başına 38
   denetim başarılıdır (`04-patch-validation/interpreter-*.json/log`).
4. **Launcher ilk gözlenen hatayı sonradan değiştirebiliyordu.** Rank 1'in
   kod 3 hatası gözlendikten sonra kapanan rank 0'ın kod 4 sonucu, işletim sistemi
   zaman damgası yüzünden dönüş kodunu değiştiriyordu. İlk gözlenen hata grubu
   artık sabitlenir. Test, logdaki ilk hata ile dönüş kodunun eşitliğini ve hızlı
   kapanışı doğrular; zaman damgasından nedensel kök hata çıkarmaz. Release ve
   ASan'da ayrı ayrı **20/20 tekrar** ve elastic restart geçti
   ([kanıt](09-launcher-stress/verification.json)).
5. **CTest 3.27 global sınırı uzun testleri kesiyordu.** Derin ağ testinin
   3600 saniyelik özelliği global 1500 saniye ile sınırlandı. Yeni yapılandırma
   varsayılan global üst sınırı 7200 yapar; özel test sınırlarını ve kullanıcının
   cache tercihini korur. Bu bir test başarısı değildir: derin ağın bütün ASan
   konfigürasyonları henüz bitmedi. Mekanizma ve resmî kaynak SHA'sı
   [timeout-analysis.json](04-patch-validation/timeout-analysis.json) içindedir.
6. **ASan zamanı üretim performansı gibi sınanıyordu.** Gruplu GEMM'in bit
   düzeyinde iki doğruluk kontrolü korunur; ASan altında hız sonucu açıkça
   `performance_status=no_data` olur. Release hız eşiği korunur. Yeni ASan
   ikilisiyle iki sayısal kontrol geçti
   ([çıktı](04-patch-validation/asan-grouped-execution.json)).
7. **VRAM dışına taşan test geçerli GPU ölçümü üretiyordu.** Önceki math
   T=4096 koşusu 8 GiB kartta 10.59 GB ayırmış ve adım medyanı 144457.5 ms
   olmuştu. Aynı BF16/FP32-master AdamW tarifi için muhafazakâr bellek alt sınırı
   artık model kurulmadan, boş VRAM'in %90'ı ile karşılaştırılır. Sığmayan örnek
   `hardware_blocked` olur; diğer aktivasyon/workspace maliyetlerini dışarıda
   bırakan bu alt sınır, tam bellek tahmini diye sunulmaz. Gerçek kontrol
   koşusunda beş desteklenen örnek ölçüldü, bir örnek ayrılmadan engellendi
   ([kanıt](11-budget-validation/verification.json)). Bu, çekirdek hızlanması
   iddiası değildir.
8. **Parity referansı kendi tekrarında farklı sonuç üretiyordu.** Aynı veri,
   ağırlık ve tohumla PyTorch'un iki yüksek-LR koşusu arasındaki en büyük fark
   0.019011 oldu. `faz9_nanogpt.py --deterministic 1`, CUDA başlatılmadan sabit
   cuBLAS workspace ve deterministik algoritmaları etkinleştirir; seçim çıktıya
   yazılır. Hız referansının varsayılanı hâlâ 0'dır. Parity sürücüsü varsayılan
   olarak deterministik çalışır ve her iki tarafa aynı `--lr` değerini verir.
   Sayısal tolerans değiştirilmedi.

## Kanıt sözleşmesi

Yeni kayıtlar **şema 2** kullanır. Her sürecin medyan/ortalama/min/max/std/varyans/CV
ve Q1/Q3/IQR değerleri saklanır. Bağımsız süreç medyanlarının IQR'si ve %95
percentile bootstrap aralığı ayrıca yazılır: 2000 örnekleme, sabit tohum 1337.
Süreç içi iterasyonlar bağımsız süreç gibi havuzlanmaz. Küçük süreç sayısı güven
aralığının yorumunu sınırlar; aralık jitter kapısının yerini almaz.

Rapor ve `--baseline` girdisi aynı doğrulayıcıdan geçer; özetlerin ham kayıtlarla
uyuşması gerekir. Eksik süreç içi varyansa sahip eski baseline hız/regresyon
iddiasını kapatmaz. NaN/Inf, yanlış türler ve değiştirilmiş varyans alanları
reddedilir. Eski şema 1 raporları okunur: yayımlanmış Faz 21 raporu, karar JSON'u
ve SVG'si byte düzeyinde aynı yeniden üretildi. Buna karşılık yeni karşılaştırma
kapıları, eski kayıtlarda bulunmayan veriyi var saymaz.

Eğitim kontrolcüsünde LR scheduler ısınması (`scheduler_warmup_steps=2`) ile
zamanlamadan çıkarılan adımlar (`timing_discard_steps=5`) ayrı kaydedilir. Sayısal
tarif değişmedi. İlk adım hâlâ derleme/capture/hesabı birlikte içerir; saf compile
süresi olarak sunulmaz.

## Değiştirilmeyen ham ölçümlerin yeniden analizi

[Rapor](05-dispersion-audit/REPORT.md), [ham sonuçlar](05-dispersion-audit/results.json),
[analiz kökeni](05-dispersion-audit/analysis.json).

Bu **yeni bir zamanlama koşusu değildir**. 28 Ekim `run-03` ham kayıtları ve ölçüm
metadata'sı aynen korunarak yeni istatistikler hesaplandı. Örneğin Nexus CUDA
GEMM'de süreç medyanları CV'si %3,64 iken süreç içi CV'nin en büyüğü %22,03'tür.
FlashAttention'da bu değerler %1,38 ve %14,35'tir. Dolayısıyla yalnız medyanların
kararlı görünmesi yeterli değildir. Bu analiz yeni hız üstünlüğü kanıtı üretmez.

## Doğrulama ve ortam

- Değişiklik öncesi eski Python paketi: 8/8 geçti.
- Değişiklik sonrası Python 3.14 ve 3.12: **14/14** geçti. Ham çıktılar
  `04-patch-validation/unit-python*.log` içindedir.
- Release ve ASan'da değişen C++ test hedefi yeniden derlendi. PATH'te Python
  bulunmadığı halde yapılandırılmış yorumlayıcıyla parity bölümü atlanmadan geçti.
- Tam paket başlangıç kapsamı: Release **213**, ASan CPU **164** CTest hedefi.
  Tam sonuçlar: [Release JUnit](02-unsandboxed-baseline/release-ctest.xml),
  [ASan JUnit](02-unsandboxed-baseline/asan-o1-ctest.xml). İki pakette atlananlar
  HIP parity ve gerçek satıcı PJRT testidir; ölçülmüş başarı sayılmazlar.
- ASan'ın ilk dört başarısızlığı: `test_deep_stack_stability` (1500.22 s
  timeout), `test_rank_failure_clean_exit`, `test_grouped_gemm_vs_loop` ve
  `bench_attention_cpu` (600.04 s timeout). Rank ve grouped düzeltmelerinin
  sonraki odak sonuçları yukarıdadır. CPU attention en son T=2048 örneklerine
  ulaşmıştır; bütün dizinin tamamlandığı veya bellek hatasız olduğu iddia edilmez.
- ASan yapılandırması: `Debug`, `/Zi /O1 /Ob1`, debug CRT ve AddressSanitizer açık,
  CUDA kapalı. Optimizasyon yalnız test derleme ayarıdır; test/assert veya ASan
  kontrolü çıkarılmadı. GPU kanıtı ayrı CUDA sanitizer çalışması gerektirir.
- İlk kısıtlı deneme Python yöneticisi, tiktoken referansının DNS erişimi ve yerel
  TCP erişimi yüzünden kesildi. Ayrı portta izinli TCP deneyi 0,3 saniyede geçti.
  Kesilen ASan `/Od` denemesinde süreçler tarafımızca durduruldu; o sonuç gerçek
  bir model hatası veya başarılı tam koşu sayılmaz. Ham kesinti kayıtları korunur.
- Son Python paketi iki yorumlayıcıda 14/14 geçti; iki referans betiği
  `py_compile` kontrolünden geçti. Son kısıtlı TEMP ACL hatası da ayrı
  `unit-final-python314-sandbox-temp-error.log` olarak saklandı; erişilebilir TEMP
  ile izinli tekrar başarılıdır. Hata uygulama hatası diye sınıflandırılmadı.
- CUDA memcheck/initcheck/synccheck sıfır hata ile geçti
  (`06-cuda-sanitizers/`). İlk racecheck 180 saniyede zaman aşımına uğradı;
  aynı GEMM + FlashAttention ileri/geri kapsamı 600 saniyelik izinli tekrarda
  sıfır hazard/hata ile geçti ([kanıt](07-racecheck-retry/verification.json)).
  Sanitizer süreleri hız karşılaştırması değildir.
- Son uçtan uca şema 2 kontrolünde Nexus/PyTorch/JAX toplam **15 bağımsız
  süreçte**, altı özet satırıyla ve sıfır worker hatasıyla tamamlandı
  ([ham kayıt](14-schema2-smoke/results.json),
  [rapor](14-schema2-smoke/REPORT.md)). CPU varyansı, eşleşmeyen CUDA sürümü ve
  eksik FlashAttention referansı ilgili iddiaları `no_data` bıraktı. Bu koşu
  ölçüm altyapısının doğrulamasıdır; rekabetçi performans veya regresyon kapanışı
  değildir.

Eski benchmark'lar sabit tarihli dizinlere yazdığı için yalnız doğrulama
yapılandırmasına `01-baseline/evidence-hook.cmake` eklendi; yeni çıktılar bu
çalışmanın dizinine gider. Ürün kaynaklarında çıktı yolu değişikliği yapılmadı.
Önceki 404 izlenen kanıt dosyasının SHA-256 anlık görüntüsü
`01-baseline/previous-evidence-sha256.json` içindedir. Önceden kirli dosyalar ve
alt depolar temizlenmedi. Başka bir mevcut CTest ağacına müdahale edilmedi;
eşzamanlı yük altındaki bu koşular hız veya ≤%3 regresyon kanıtı değildir.

## Açık koşullar

Faz 21'in önceki BF16 kayıp eğrisinde ilk güncellemeden önce de yaklaşık 0,000211
fark vardır (`01-baseline/parity-trajectory-analysis.json`). Bu, LR'nin tek başına
kanıtlanmış neden olmadığını gösterir; FP32/BF16 çıktıları ve gradyanlar izole
edilmeden head/optimizer kök nedeni ilan edilmez. Faz 20 LR=6e-4 ayrışması bu
gözlemle çözülmüş sayılmaz. Tolerans gevşetilmedi.

Yüksek LR FP32 incelemesinde prefix ve tam giriş geometrisi ayrı sınandı;
geometri değişikliği ayrışmayı kapatmadı (`08-parity-layout/`). Nexus kendi
tekrarında ve deterministik PyTorch kendi tekrarında bit düzeyinde aynı sonucu
verdi. Ancak iki uygulama arasındaki LR=6e-4 farkı hâlâ 0.043724'tür
([karşılaştırma](12-determinism-probe/comparison.json)). Bu koşul **failed** kalır.

Güncel gerçek parity sürücüsüyle **124M FP32, 100 adım, LR=1e-4** kontrolü geçti:
en büyük fark **9.5367431640625e-7**, tolerans **0.001**
([sonuç](13-strict-parity/parity.json), [komutlar](13-strict-parity/metadata.json)).
Bu yalnız düşük-LR FP32 doğrulamasıdır; yüksek LR veya BF16 hedefini kapatmaz.
Yaklaşık 498 MB başlangıç ağırlığı rapora kopyalanmadı; SHA-256 ve boyutu
[provenance.json](13-strict-parity/provenance.json) içinde saklandı.

CUDA 13.2 toolkit bu makinede bulunmuyor; yerel toolkit'ler 12.2/13.4, kurulu
PyTorch paketi 2.14.1+cu132. Eş sürüm karşılaştırması, BF16 124M parity, 1.3B eşdeğer
optimizer, time-to-validation-loss, gerçek JAX worker sayısı, bütün fazları
kapsayan matris ve sabit GPU CI baseline'ı açık kalır. Faz 20 uzun eğitim/ölçek,
çok GPU, ROCm/Metal/PJRT donanımı ve sonraki performans aşamaları da kapanmadı.

Önceki Linux CPU CI başarısı Faz 21'in `119c5d4` commit'ine aittir; bu çalışma
için yeni Linux sonucu olmadan o kanıt yeni bir commit'in CI sonucu diye sunulmaz.
Workflow'a estimate parity ve tekrarlı launcher hata aktarımı eklendi; yeni
workflow henüz sunucuda çalıştırılmadı. Bu çalışma için commit/push yapılmadı.

## Tekrarlama

Depo kökünde, yapılandırılmış derleyici/ASan ortamıyla:

```powershell
& .bench-venv-py314/Scripts/python.exe tests/parity/test_phase21_bench.py
& 'C:/Program Files/CMake/bin/ctest.exe' --test-dir build-phase1 -C Release --output-on-failure
& 'C:/Program Files/CMake/bin/ctest.exe' --test-dir build-asan -C Debug --output-on-failure
& .bench-venv-py314/Scripts/python.exe benchmarks/runner/faz20_parity.py --exe build-phase1/benchmarks/runner/bench_cuda_phase20.exe --tokens build-phase1/faz20/tokens.bin --python .bench-venv-cuda/Scripts/python.exe --nanogpt C:/Users/muham/nexus-data/nanogpt --out build-phase1/NEW-UNUSED-parity --lr 0.0001 --deterministic 1
```

Tam paketleri tekrar çalıştırmadan önce eski kanıtları koruyan çıktı yönlendirmesi
ve yeni, kullanılmamış kayıt dizini seçilmelidir. Bu çalışmaya özel yardımcılar
`build-phase1/` altındadır; sayısal sonuçların komutları, ikili/betik SHA'ları ve
ham çıktıları ilgili kanıt dizinlerinde tutulur. `03-dispersion-audit/` erken
prototiptir; geçerli yeniden analiz `05-dispersion-audit/` dizinidir.
