# FAZ 15 — Dayanıklılık: Tam Durum Kaydı, Parçalı/Eşzamansız Kayıt, Yeniden Parçalama, Bekçi, Elastik Yeniden Başlatma (2026-10-26)

Kapsam: `promt.txt` FAZ 15 (DUVAR 13). Her şey CPU eğitmeni (`nn::Sequential` arka ucu, DDP / ZeRO / FSDP) üzerinde
çalışıyor ve sınandı. Elastik test gerçek süreçlerle (nexus-run, tcp ProcessGroup) yapıldı.

Makine: Ryzen 7 5800H (8 çekirdek / 16 iş parçacığı, SHA-NI var), NVMe SSD, Windows 11, **fişte** (`Win32_Battery` 2).
Pilde başlayan ilk ölçüm atıldı.

## Ne yapıldı

| Alt adım | Durum | Nerede |
|---|---|---|
| 15.A `TrainerState` | ✅ Alanlar: adım, epoch, küresel örnek indeksi, veri imleçleri, LR çarpanı / zamanlayıcı adımı, loss scale, RNG (tohum, ofset), biriktirme sayacı, yığın şekli, dünya | `nexus/ckpt/checkpoint.hpp` |
| 15.A NXS3 arşivi | ✅ Sihir, sürüm 3, adlı girdiler, sonda içeriğin SHA-256'sı | `encode_archive` / `decode_archive` |
| 15.A/B Atomik kayıt | ✅ Her rank dosyasını `.part` olarak yazar → yeniden adlandırır → `.done`. Rank 0 manifesti yazar ve geçici dizini **tek yeniden adlandırmayla** `step-N` yapar. `latest` atomik, son-N saklama var | `CheckpointManager` |
| 15.B Yükleme doğrulaması | ✅ Manifest biçimi/sürümü, dosya boyutu, SHA-256 kontrol edilir; uyumsuzsa açık hata (sessiz yedek yok) | `load_latest` |
| 15.C Parçalı kayıt | ✅ ZeRO/FSDP'de her rank yalnız kendi parçasını yazar; tek rankta toplanmaz | `FsdpEngine::save_shard` / `load_shard` |
| 15.D Yeniden parçalama | ✅ Rank sayısı değişince tüm eski parçalardan yeni düzen kurulur (FAZ 11 `load_resharded`). Veri imleci küresel olmalı (`TokenBatchSource::global_cursor`) | `resume_latest` |
| 15.E Eşzamansız kayıt | ✅ Çift tampon: eğitim iş parçacığı yalnız durumu kopyalar; kodlama + SHA + yazma + tamamlama arka planda. Bir yazılıyor + bir bekliyor; üçüncü kayıt bekler | `CheckpointOptions::async` |
| 15.F Bekçi | ✅ Rank kalp atışı (adım, süre), takılan rank tespiti, straggler tespiti (adıma varış gecikmesi). Panolar: `LocalBoard` (thread), `StoreBoard` (TcpStore) | `dist::Watchdog` |
| 15.F Elastik yeniden başlatma | ✅ "Temiz yeniden başlatma": `nexus-run --max-restarts K --restart-nproc M`. Bir süreç ölürse tüm grup sonlandırılır, M süreçle yeniden başlar, program `resume_latest` ile devam eder (gerekirse yeniden parçalar) | `tools/nexus_run` |
| 15.F Kalan rank'lerle iletişim dünyasını süreç içinde yeniden kurma | ⛔ yok (spec'in risk notundaki varsayılan yol seçildi) | — |
| Sıkıştırma seçeneği | ⛔ yok (zlib/zstd derlemede kapalı) | — |

Trainer API'si:

```cpp
builder.checkpointing({dir, interval, keep, async}).watchdog(wd);
trainer->resume_latest();
```

## Testler (spec adlarıyla)

| Test | Sonuç |
|---|---|
| `test_save_load_exact` | ✅ 6 adım + kayıt, **yeni süreçte** yükleme + 6 adım = kesintisiz 12 adım. Kayıp, ağırlık ve gradyan **bit bit aynı**. Dropout 0.1, senkron ve eşzamansız kip ayrı ayrı. |
| `test_rng_restore` | ✅ Dropout maskesi `(tohum, ofset, çağrı)`'dan birebir geliyor. **Negatif kontrol:** FAZ 9 kaydı (RNG sayacı yok) devamda farklı maske veriyor (kayıp farkı 6.7e-2). FAZ 15 kaydında fark 0. |
| `test_sharded_save_load` | ✅ ZeRO-3, 4 ve 8 rank. Her rank yalnız kendi parçasını yazıyor (4 rank: 8.3 KB/rank, 8 rank: 4.3 KB/rank; tam durum 31.8 KB). Yükleme + devam = kesintisiz (bit bit). |
| `test_reshard_2_to_4` (+ 4 → 2) | ✅ Devam kaybı kesintisiz eğriye göre en çok 3.7e-8 göreli fark. Ağırlık farkı 5.2e-4 / 6.5e-4; kesintisiz 2 rank ile 4 rank arasındaki fark 1.6e-3 / 1.2e-3. Yani fark yalnız indirgeme sırasından (AdamW sıfıra yakın gradyanlarda bunu büyütüyor). |
| `test_crash_injection` | ✅ Süreç kaydın 3 noktasında öldürüldü: yarım rank dosyası, manifestten önce, son yeniden adlandırmadan önce. Üçünde de son geçerli kayıt (adım 2) sağlam; yarım dizin yok sayılıyor; devam kesintisizle bit bit aynı. Bozuk dosya (1 bayt) ve yanlış sürüm açık hata veriyor. |
| `test_watchdog_timeout` | ✅ Rank 2, 3.5 s takıldı. Bekçi onu **509–512 ms** sonra raporladı (iki koşu; `stall_timeout` 500 ms) ("rank 2 takıldı, son adım 2, en ileri 3"); başka rank suçlanmadı. Diğer rank'ler kollektif zaman aşımıyla çıktı (toplam 5.6 s, asılı kalma yok). Straggler: her adımda 60 ms geciken rank 1 tespit edildi (ortalama 67 ms geç varış). |
| `test_elastic_restart` | ✅ 4 tcp süreci, ZeRO-3, kayıt aralığı 2. Rank 3 adım 5'te `_Exit(9)` ile öldü. nexus-run 2 süreçle yeniden başlattı; adım 4 kaydından 4 → 2 yeniden parçalamayla devam etti. Kayıp adım 1 (≤ aralık). Devam kaybı kesintisiz 4 rank eğrisine göre 6.0e-8. |
| SHA-256 | ✅ SHA-NI yolu = skaler yol (0–300 bayt, tüm dolgu sınırları, 1 MiB). FIPS 180-2 "abc" vektörü doğru. |

### Kabul kriterleri

| Kriter | Sonuç |
|---|---|
| `test_save_load_exact` birebir | ✅ bit bit (yeni süreç, senkron + eşzamansız) |
| 8 rank parçalı kayıt tek rankta toplanmadan | ✅ her rank kendi dosyası (~1/8); kayıt yolunda `full_parameters` / all-gather yok |
| Kurtarma, aralıktan fazla adım kaybettirmez; devam eğrisi kesintisizle aynı | ✅ 1 adım kayıp (aralık 2); eğri 6e-8 içinde |

## Benchmark'lar

### Kayıt hızı (sentetik yük; kodlama + SHA-256 + yazma + tamamlama)

| Dünya | Toplam | Süre | Hız |
|---|---|---|---|
| 1 rank | 64 MB | 145 ms | 0.46 GB/s |
| 1 rank | 512 MB | 1 028 ms | 0.52 GB/s |
| 4 rank (iş parçacığı) | 64 MB | 60 ms | 1.13 GB/s |
| 4 rank (iş parçacığı) | 512 MB | 342 ms | **1.57 GB/s** |

- **Skaler SHA-256 çok yavaştı.** İlk ölçümde tek rank 0.10 GB/s idi. Skaler SHA (NexusData) tek başına 0.24 GB/s ölçüldü ve
  dosya başına iki kez hesaplanıyordu.
- **İki düzeltme:**
  - Manifest artık arşivin kendi özetini taşıyor (tek SHA geçişi).
  - CPUID ile seçilen bir SHA-NI yolu eklendi (yoksa skaler yola düşüyor).

  Sonuç: tek rank 0.10 → 0.52 GB/s, 4 rank 0.34 → 1.57 GB/s.
- Tek rankta hâlâ disk sınırına (NVMe) ulaşılmıyor. Kodlama tamponu kopyası ve tek iş parçacıklı SHA sınırlıyor.

### Adım süresine etkisi (CPU eğitimi, 1.06M parametre, durum 12.7 MB, 21 adım)

| Kayıt | Adım | Ek yük | Eğitim iş parçacığının beklediği süre |
|---|---|---|---|
| yok | 17.7 ms | — | — |
| senkron, her 2 adımda | 40.6 ms | +%129 | 38.6 ms / kayıt |
| **eşzamansız**, her 2 adımda | 29.0 ms | +%63 | **0.0 ms** |
| senkron, her 10 adımda | 23.4 ms | +%32 | 27.7 ms / kayıt |
| **eşzamansız**, her 10 adımda | 21.2 ms | +%20 | **0.0 ms** |

- **Spec "≈ 0" hedefi tutmadı (⛔).** Eşzamansız kayıt eğitimi hiç bloklamıyor; kayıt çağrısı yalnız durum kopyası kadar sürüyor.
  Ama CPU eğitimi tüm çekirdekleri kullandığı için arka plan yazıcısı (SHA, kopya, disk) onlarla yarışıyor. Ek yük senkron
  yolun yaklaşık yarısı ile üçte ikisi arasında kalıyor.
- GPU eğitiminde CPU çoğunlukla boşta olduğundan ek yükün çok daha küçük olması beklenir. Bu **ölçülmedi**.
- Ölçüm ayarı bilerek sert: 18 ms'lik adımlar ve her 2–10 adımda bir kayıt. Gerçek eğitimde (dakikalar arayla kayıt) ek yük
  oranı bu tablodakinden çok daha küçük olur.

### Kayıt boyutu

- 1.06M parametre → 12.73 MB dosya, yani **12.0 bayt/parametre**: FP32 ağırlık + Adam m + v. Arşiv başlığı ve TrainerState ihmal edilebilir.
- Sıkıştırma ⛔ yok: zlib/zstd derlemede kapalı.

### MTTR / RPO (elastik test)

- **RPO:** kayıt aralığı 2 adım. Ölüm adım 5'te, devam adım 4'ten; 1 adım kayıp.
- **MTTR = 87–132 ms** (iki koşu): ölümden ilk devam adımına kadar.
  - İçerdikleri: grubun sonlandırılması, 2 sürecin başlatılması, tcp store + grup, kaydın yüklenmesi ve doğrulanması, 4 → 2 yeniden parçalama, ilk ileri + geri.
  - Çok küçük model ve yerel makine; büyük modelde süreye yükleme baskın olur.

## Yol boyunca bulunan ve düzeltilen hatalar

1. **NexusOptim `ThreadPool` kilitlenmesi (kayıp uyandırma).**
   - **Yarış:** `parallel()` `epoch_`'u muteks almadan artırıp `notify_all` yapıyordu. Bu sırada uyumak üzere olan işçi
     koşulu kilitle kontrol edip `cv.wait`'e girerken bildirim kayboluyordu. İşçi sonsuza kadar uyuyor, çağıran
     `remaining_ != 0` döngüsünde dönüyordu.
   - **Belirti:** CPU AdamW eğitimi ara sıra (bu modelde 22–65 adımda bir) tek çekirdek %100 dönerek takılıyordu.
   - **Teşhis:** takılan süreçte DbgHelp ile tüm iş parçacığı yığınları alındı.
   - **Düzeltme:** artırım muteks altında (alt modül `lib/NexusOptim`, `src/core/thread_pool.cpp`).
   - **Doğrulama:** düzeltmeden sonra 8 × 2000 adım takılmadan bitti.
   - FAZ 15'ten bağımsız bir hata. Önceki fazlardaki nadir "takılan test" görülmelerini açıklayabilir.
2. **`FsdpEngine::load_shard` aynı düzende çalışmıyordu (FAZ 11).** Yalnız kendi parçasıyla `load_resharded({blob})`
   çağırıyor ve tam kapsama istiyordu; dp > 1'de hep hata veriyordu. Artık kendi parçasını doğrudan yerine koyuyor. ZeRO-1/2'de
   parametreler parçalardan all-gather ile kuruluyor.
3. **`nn::Dropout` RNG'si kayda girmiyordu.** Modül içinde artan tohum vardı; devamda maske dizisi baştan başlıyordu
   (negatif kontrol bunu gösteriyor). Artık maske, trainer'ın verdiği `(tohum, ofset)` sayacından türetiliyor.

## Gerideyiz / yapılmadı

- **Eşzamansız kayıt ek yükü ≈ 0 değil (CPU):** +%20 (her 10 adım) / +%63 (her 2 adım). Eğitim iş parçacığı bloklanmıyor; kayıp
  CPU paylaşımından geliyor.
- **GPU arka ucu (`cuda::gpt_backend`) ile FAZ 15 kaydı sınanmadı.** Aynı `save_state` yolunu kullanıyor, ama:
  - aygıttan kopya senkron;
  - pinned çift tampon / ayrı akış yok.
- **FAZ 12 paralel arka ucu (TP/PP/CP) için yeniden parçalama yok.** Aynı dünyada kayıt/yükleme var; farklı dünya açık hata
  veriyor. DTensor yerleşimi kayıtta tutulmuyor.
- **NCCL zaman aşımı / iletişim dünyasını süreç içinde yeniden kurma yok:** NCCL yok, Windows. Yalnız temiz yeniden başlatma var.
- **Bekçinin TcpStore panosu (`StoreBoard`) testte kullanılmadı.** Bekçi testleri thread dünyasında `LocalBoard` ile yapıldı.
- **Rank'ler arası tamamlama paylaşımlı dosya sistemi varsayıyor:** tek düğüm ya da ortak FS. Çok düğümlü yerel disk desteklenmiyor.
- Sıkıştırma yok.
- Yeniden parçalamada her rank tüm eski parçaları okuyor (bellek: tam durum / rank). Büyük modelde aralık bazlı okuma gerekir.

## Dosyalar

- Kod:
  - `include/nexus/ckpt/checkpoint.hpp`, `src/ckpt/checkpoint.cpp`: NXS3, TrainerState, CheckpointManager.
  - `src/ckpt/sha256.cpp`: SHA-NI + skaler.
  - `include/nexus/dist/watchdog.hpp`, `src/dist/watchdog.cpp`: bekçi.
  - `lm_trainer` entegrasyonu: `checkpointing`, `watchdog`, `resume_latest`, `set_rng`, `load_resharded_state`.
  - `nn::Dropout`: sayaç tabanlı RNG.
  - `FsdpEngine::load_shard`.
  - `tools/nexus_run`: `--max-restarts`, `--restart-nproc`.
  - `lib/NexusOptim/src/core/thread_pool.cpp`: kilitlenme düzeltmesi.
- Test: `tests/parity/test_phase15_durability.cpp`.
- Ham çıktılar: `bench.jsonl`, `tests_cpu.log`, `ctest_cpu.txt`, `ctest_cuda.txt`, `asan_ctest.txt`.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
