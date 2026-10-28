# Faz 19 denetimi — 27–28 Ekim 2026

## 28 Ekim yayın kontrolü

Kullanıcı, ROCm/macOS makinesi veya üretici eklentisi sağlamadan mevcut Windows makinesinde
doğrulanabilen kapsamın tamamlanıp push edilmesini seçti. HIP kaynak testi
`test_hip_source_compat` adıyla ayrıldı; gerçek `test_hip_parity` uygulanmadığı için atlanır.
Son Release grubunda **8 geçti, 2 atlandı, 0 hata**: [günlük](publish-release.log),
[JUnit](publish-release.xml). Atlananlar gerçek HIP yürütmesi ve satıcı PJRT eklentisidir.
Bu yayın HIP/Metal motorlarının tamamlandığı anlamına gelmez. Faz 20 kanıtları ayrı dizindedir.
Son ASan grubunda **7 geçti, 2 atlandı, 0 hata**: [günlük](publish-asan.log),
[JUnit](publish-asan.xml). Tam Release/CUDA ve ASan derlemeleri tamamlandı.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Faz 19'un mevcut kodu, testleri ve rapordaki iddiaları karşılaştırıldı. Güvenlik ve doğruluk hataları düzeltildi. **Faz bütünüyle kapatılmadı:** HIP/Metal yürütme motorları tamamlanmış değildir. Kullanıcı onayıyla resmî harici PJRT C ABI adaptörü ayrıca eklendi. Faz 20'ye geçilmedi.

## Doğrulanan nedenler ve düzeltmeler

1. **Yanlış yetenek bildirimi ve sessiz CPU hesabı.** CUDA/DirectML registry sürücüleri, host'a kopyalayıp CPU'da hesapladıkları işlemleri yerel GPU desteği olarak bildiriyordu. Artık bu sürücüler yalnız gerçek GEMM yolunu yerel destek olarak bildirir. Desteklenmeyen işlemler `Error` politikasında hata verir; `WarnOnce`/`WarnAlways` altında sayılır ve görünür biçimde CPU'ya taşınır. Doğrudan `Driver` çağrıları da aynı politikayı uygular. Bu değişiklik ayrı `nexus_cuda` eğitim motorundaki mevcut CUDA çekirdeklerini kaldırmaz.
2. **Cihaz işaretçisinin CPU'da kullanılması.** Softmax, RMSNorm, LayerNorm, RoPE, SwiGLU, attention, Q8/Q4 GEMM ve AdamW geri dönüş yolları artık mevcut staging uygulamalarını kullanır. AdamW'de parametre ile iki moment de cihaza geri yazılır. Opaque allocation handle kullanan CPU referans sürücüsüyle regresyon testi, bu hatayı gerçek bir GPU gerektirmeden yakalar.
3. **Sahte Metal birleşik belleği.** `DeviceRef::metal()` etiketi, normal host ayırmasını GPU'nun kullanabildiği paylaşılan tampon haline getirmez. `is_unified()` gerçek bir accelerator allocator uygulanana kadar yalnız CPU için doğrudur. Olmayan Metal sürücüsünde `device_data()` açık hata verir.
4. **PJRT ABI uyuşmazlığı.** Nexus'un eski elle tanımlanmış tablosu ile resmî OpenXLA ABI'si ayrıldı. Kullanıcının açık onayı sonrası `src/backend/pjrt_plugin.cpp` eklendi: mutlak/canonical eklenti yolu, Windows'ta kontrollü DLL dependency araması, ABI boyutu/ana sürüm ve gereken fonksiyonların denetimi, paylaşılan eklentide tek initialization, client'tan uzun yaşayan library sahipliği, opaque executable/buffer/event arayüzleri kullanılır. Çoklu çıktı ve row-major host layout desteklenir; bütün girişler non-donatable işaretlenir. Hata ile birlikte dönen olay da sahiplenilip beklenir; host belleği erken bırakılmaz. Eklenti isteği başarısızsa referans yorumlayıcısına sessiz geçiş yapılmaz. Resmî [OpenXLA başlığı](https://github.com/openxla/xla/blob/61c4e245c954ecdb7755b0127e62dcc1fe2984af/xla/pjrt/c/pjrt_c_api.h) değiştirilmeden saklandı (C API 0.117). Derleme seçenekleri tek replica/tek partition içerir; boş protobuf'un sıfır replica istemesi önlendi ve aynı seçenekler gerçek XLA testiyle doğrulandı.
5. **Yorumlayıcı ile XLA'nın karıştırılması.** Yerleşik yol artık `nexus-stablehlo-reference-cpu` olarak tanımlanır; OpenXLA sürümü veya 16 GB TPU belleği uydurmaz. Otomatik seçim normal CPU'yu bu referans yolundan önce tercih eder. `tpu:0` eski API etiketi korunmuştur; gerçek TPU bulunduğu anlamına gelmez.
6. **Geçersiz StableHLO.** `%4_b` gibi geçersiz sayısal SSA isimleri `%v4_b` biçimine dönüştürüldü. Float sabitler FP32'nin geri dönüş hassasiyetiyle yazılır. Bağlanmamış custom-call hedefi üretmek yerine desteklenmeyen dışa aktarma açık hata verir. Nexus IR → gerçek XLA testi bu SSA hatasını yakaladı.
7. **Yanlış GSPMD cihaz sırası.** Mesh eksenleri tensör eksenleriyle farklı sıradaysa tile cihazları yeniden sıralanır: örneğin `Shard(1), Shard(0)` için `0,2,1,3`. Replikasyon ekseni tile eksenlerinden sonra gelir. `Partial(sum)` manuel sharding gibi gösterilmez; önce reduction yapılması istenir.
8. **Tampon boyutu ve yaşam süresi.** Girdi şekli/eleman sayısı, modül imzası, unary/binary operand boyutları, GEMM contraction ve broadcast boyutları denetlenir. Taşan/negatif şekiller, FP32 dışındaki referans girdiler ve allocation sınırını aşan kopyalar reddedilir. Başarısız yürütmede giriş/çıkış tamponları temizlenir. Float'a hizalı olmayan allocation için sayaç hatası düzeltildi. `stablehlo.logistic` yanlışlıkla `stablehlo.log` olarak ayrıştırılmaz.
9. **Ölçüm ve test dürüstlüğü.** Farklı arka uçların opak tamponları birbirinin sürücüsüne verilmez; cross-backend kopya önce kaynak sürücüsüyle host’a, sonra hedef sürücüsüne taşınır. Bunun Vulkan ↔ referans opaque-handle gidiş-dönüş testi eklendi. Benchmark 3 ısınma + 7 ham örnek, medyan/min/max/standart sapma, sürücü ve gerçek implementation kimliğini üretir. Karşılaştırma yardımcıları boyut uyuşmazlığını veya NaN'ı başarı saymaz. Vulkan/DirectML bulunmadığında ilgili donanım testi CTest'e 77 (`Skipped`) döndürür. HIP testi bir statik uyumluluk denetimidir; gerçek HIP parity olarak yorumlanmaz.

## Gerçek destek kapsamı

| Yol | Doğrulanan durum | Kalan sınır |
|---|---|---|
| CPU | AVX2/FMA ve mevcut CPU işlemleri | AVX-512/AMX bu makinede ölçülmedi |
| CUDA registry sürücüsü | Gerçek cuBLAS GEMM; politika kontrollü host staging | Registry'nin diğer işlemleri yerel CUDA çekirdeği değildir; ayrı eğitim motorunun CUDA testleri ayrıca çalıştırılır |
| Vulkan | RTX 3070 üzerinde SPIR-V işlemleri ve 25 adımlı MLP eğitimi | MLP başarısı büyük dil modeli token/s kanıtı değildir |
| DirectML | RTX 3070 üzerinde gerçek DML GEMM | Diğer registry işlemleri izinli CPU fallback kullanır |
| HIP/ROCm | Sembol dönüşümü, wave/LDS kaynak denetimi | Çalışan allocator/kernel/rocBLAS/RCCL yürütme motoru yok; gerçek HIP derleme/parity kanıtı yok |
| Metal | Örnek MSL kaynakları ve açık unavailable davranışı | Objective-C++ allocator/command queue/yürütme motoru yok; macOS derleme kanıtı yok |
| Yerleşik StableHLO | Açıkça etiketlenen FP32 CPU referans yorumlayıcısı | XLA, resmî PJRT eklentisi veya TPU değildir |
| Harici C++ PJRT adaptörü | Resmî C ABI ile yükleme/compile/execute, çoklu çıktı ve 14 hata senaryosunda cleanup testi | Gerçek üretici eklentisi bu oturumda temin edilmedi; sözleşme testi bilinçli olarak adlandırılmış test double kullanır. Registry yalnız ilk addressable cihazı açar; desteklenen doğrudan GEMM şekli dışındaki CPU geçişi politikaya tabidir |
| Bağımsız XLA/PJRT CPU | Nexus'un gerçek IR dışa aktarımı JAX/jaxlib 0.10.2 ile derlenip çalıştırıldı | C++ adaptörden bağımsız gerçek XLA derleme/hesap kanıtı; gerçek TPU/GSPMD küme yürütmesi ölçülmedi |

## Nihai doğrulama ve ölçüm

Adaptör öncesi geniş regresyon koşusunda **207 hedefin 206’sı geçti** (atlanan yok). `test_collectives_tcp_4` ilk koşuda ve ilk tekrarda 29624 portuna bağlanamadı. Aynı test 29724 portunda geçti; **son kontrolde asıl CTest hedefi 29624 portunda da geçti**. Hiçbir port ayarı değiştirilmedi. [Asıl hedefin son başarılı koşusu](tcp4-final.log); önceki başarısız günlükler korunur.

Son adaptör ve kopya güvenliği değişikliklerinden sonra tam Release/CUDA ve ASan derlemeleri yeniden tamamlandı. Faz 19 grubunda **8 test geçti, 1 üretici eklentisi testi atlandı**; ASan grubunda **9 test geçti, aynı üretici testi atlandı**. Gerçek üretici eklentisi verilmediğinde `test_pjrt_vendor` 77 döndürür. Bu hedef `-DNEXUS_PJRT_TEST_PLUGIN_PATH=<güvenilir DLL/so yolu>` ile gerçek eklentide çalıştırılabilir. Uzun 207 hedeflik paket adaptör eklemesinden sonra baştan tekrar koşulmadı. [Makine özeti](verification.json), [34 ABI/cleanup denetiminin tam ham çıktısı](pjrt-abi-contract.log).

Yeni ölçüm: 512×512×512 FP32 GEMM, 8 CPU işçisi, 3 ısınma, 7 tekrar, çağrı sonunda sürücü senkronizasyonu. Diğer derleme/test yükleri bittikten sonra ölçüldü; saatler kilitlenmedi. Ham örnekler ve kaynak/ikili SHA-256 kimlikleri [backend-bench.json](backend-bench.json) içindedir. Tablo gerçek harici XLA eklentisi ölçümü içermez.

| Arka uç / implementation | Medyan ms | Min ms | Max ms | Std ms | GFLOP/s |
|---|---:|---:|---:|---:|---:|
| cpu | 0.4614 | 0.4498 | 0.5790 | 0.0428 | 581.78 |
| cuda | 0.1162 | 0.1095 | 0.2761 | 0.0560 | 2310.12 |
| vulkan-spirv | 0.4611 | 0.4517 | 0.5680 | 0.0418 | 582.16 |
| directml | 0.4069 | 0.3744 | 1.2180 | 0.2795 | 659.71 |
| stablehlo-reference-cpu | 3.8963 | 3.4168 | 4.8346 | 0.4682 | 68.89 |

Mevcut Faz 7 regresyonunda 1.313.626.112 parametreli model, BF16, B=1, T=1024, 8-bit fused AdamW ile 60 adım tamamladı: medyan **947,30 ms/adım**, ölçülen tepe **5,941 GiB**. [Ham koşu](regression-outputs/2026-10-24-faz7/bench_cuda_phase7_all.json). Bu mevcut altyapının doğrulanmasıdır; yeni bir çekirdek hızlanması iddiası değildir. Faz 20 veri/parity çalışması yapılmadı.

## Kanıt dosyaları

- [Arka uç envanteri](inventory.log)
- [Nexus'un dışa aktardığı MLIR](nexus.mlir), [aynı girişler ve yerel referans çıktı](nexus.mlir.json)
- [Gerçek XLA çıktı karşılaştırması](xla-cpu.json): `max_abs = 1.430511474609375e-6`, `module_origin = nexus_ir_export`
- [Nihai ASan test raporu](asan-final.log): Faz 19 + eklenti ABI + küçük model eğitimi, **9 geçti, 1 üretici testi atlandı**; [JUnit](asan-final.xml)
- [CUDA sanitizer çıkış kodları](sanitizers.json): dört araç da `0`; [memcheck](sanitizer-memcheck.log), [racecheck](sanitizer-racecheck.log), [initcheck](sanitizer-initcheck.log), [synccheck](sanitizer-synccheck.log)
- [Nihai Faz 19 Release testleri](release-final-phase19.log): **8 geçti, 1 üretici testi atlandı**; [JUnit](release-final-phase19.xml). ABI test DLL’i XLA değildir; çoklu opaque çıktı, boş çıktı, initialization paylaşımı, 14 hata enjeksiyonu ve her seferinde sıfır canlı test nesnesi denetlenir.
- [Tam Release test günlüğü](release-tests.log), [JUnit](release-tests.xml)
- [29624 portunda aynı testin tekrarı](tcp4-retry.log), [29724 portunda aynı dört rank testinin başarısı](tcp4-alternate-port.log)

Compute Sanitizer bu denetimde registry CUDA/cuBLAS yolunu tarar; Vulkan SPIR-V, DirectML veya Metal bellek yarışlarının kanıtı değildir. ASan host belleği ve C++ sınır kontrollerini tarar. Linux, AMD GPU, Apple Silicon ve gerçek TPU donanımı bu oturumda doğrulanmadı.

## Tekrarlama

Windows'ta önce Visual Studio 2022 x64 geliştirme ortamını yükleyin. PATH'teki başka bir CMake yerine `C:\Program Files\CMake\bin` kullanın.

```bat
"C:\Program Files\CMake\bin\cmake.exe" --build build-phase1 --parallel 6
"C:\Program Files\CMake\bin\ctest.exe" --test-dir build-phase1 -C Release --output-on-failure
"C:\Program Files\CMake\bin\ctest.exe" --test-dir build-phase1 -C Release --output-on-failure -R "test_(backend_detection|backend_fallback|hip_parity|vulkan_parity|directml_parity|pjrt_export|pjrt_xla_cpu)$"
```

Bağımsız XLA testi için ayrı test ortamında `jax==0.10.2` ve `jaxlib==0.10.2` kullanıldı. Python yolu `-DNEXUS_XLA_PYTHON=<python.exe>` ile yapılandırılır. Bağımlılık eksikse test 77 ile atlanır; başarı sayılmaz. Kullanıcının eski `.bench-venv` ortamı değiştirilmedi.

```bat
<python.exe> benchmarks/runner/faz19_pjrt.py --producer build-phase1/tests/parity/test_phase19_backends.exe --output docs/benchmarks/2026-10-27-faz19/audit/xla-cpu.json
<python.exe> benchmarks/runner/faz19_audit.py --executable build-phase1/tests/parity/test_phase19_backends.exe --threads 8 --output docs/benchmarks/2026-10-27-faz19/audit/backend-bench.json
```

## Fazı kapatmak için kalan işler

Resmî PJRT adaptörü önce otomatik onay denetiminde harici DLL/so yükleme ve yaşam döngüsü riski nedeniyle reddedildi. Kullanıcının açık “Evet, gerçek PJRT adaptörünü uygula” onayı sonrası uygulandı ve sözleşme/hata testleri eklendi. Bu kısıt artık kullanıcı onayı bekleyen bir engel değildir.

C++ adaptör için üreticinin gerçek `GetPjrtApi` eklentisi üzerinde uçtan uca çalışma bu oturumda doğrulanmadı; `test_pjrt_vendor` bu yüzden açıkça atlandı. Mevcut JAX paketinde bu sembolü dışa aktaran bağımsız PJRT DLL'i bulunmadı. Test double, resmî ABI çağrılarını ve kaynak sahipliğini doğrular; XLA veya TPU performansının yerine geçmez. Gerçek XLA CPU sonucu ayrı JAX/jaxlib koşusundan gelir. Bu ayrım özellikle korunur.

HIP/Metal yürütme yolları, uygun toolchain ile gerçek derleme ve ilgili donanım üzerinde test gerektiriyor. Her arka uç için dil modeli token/s ölçümü ve %3 regresyon kabulü henüz kanıtlanmış değildir. Önceki en-iyi-örnek GEMM tabloları ile yeni medyan ölçümleri farklı protokollerdedir; doğrudan hızlanma yüzdesi çıkarılmaz. Bu rapor PyTorch/TensorFlow'u geçme iddiasında bulunmaz.

`promt.txt` içindekilerinde G/H/I/J bölümleri anılıyor, fakat verilen dosya Faz 22 sonunda bitiyor; bu bölümlerin gövdeleri bulunmuyor. Denetim raporu mevcut açık kabul koşullarını esas alır.
