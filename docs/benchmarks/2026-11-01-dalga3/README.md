# Dalga 3 — GPU P0 çekirdekler: 3.1 GEMM (ilk parça)

Tarih: 2026-11-01.

Ortam:
- RTX 3070 Laptop 8 GB, AC güçte.
- Nexus cuBLASLt, CUDA 13.4.
- PyTorch 2.15.0+cu134, aynı GPU.

Bu dalga plan.txt'te 2–3 hafta. Bu rapor yalnız 3.1'in ilk parçası. Diğer parçalar yapılmadı:
- epilog füzyonunun genişletilmesi (residual / dgelu yolu);
- split-K;
- 3.2 FlashAttention-2;
- 3.3 füzyon çekirdekleri;
- 3.4 CUDA Graph işi.

## Yapılan (src/cuda/gemm_lt.cpp, include/nexus/cuda/gemm.hpp)

| Madde | Önce | Şimdi |
|---|---|---|
| Kilit | Her `gemm()` çağrısı global `std::mutex` alıyordu. | `std::shared_mutex`: plan bulunan sıcak yol paylaşımlı kilitle çalışır, okuyucular birbirini beklemez. Yalnız yeni plan, tanıtıcı veya çalışma alanı eklemek özel kilit alır. |
| İşaretçi yarışı | Bias / aux işaretçisi paylaşılan tanımlayıcıya her çağrıda yazılıyordu. Kilit kalkınca iki iş parçacığı birbirinin bias'ını kullanabilirdi. | İşaretçi özelliği olan planlarda "yaz + başlat" bölümü o plana ait mutex altında. İşaretçisiz planlar kilitsiz. |
| Çalışma alanı | Aygıt başına tek tampon. Kendi yorumu "iki akış yarışır" diyordu. | (aygıt, akış) başına ayrı tampon, o akışa bağlı ayrılır. |
| Autotune | 8 aday, her biri 3 tekil ölçüm, en kısa toplam seçiliyordu (gürültüye açık). | 16 aday. 5 tur iç içe ölçüm, her turda aday başına 8 çağrı; aday başına medyan alınır. Sezgisel seçim ancak bir aday ondan ≥ %2 hızlıysa değişir. |
| Kalıcı seçim | Yok. | `set_gemm_tune_cache(path)`. Dosya cuBLASLt sürümü, SM, SM sayısı ve aygıt adıyla damgalı; damga uyuşmazsa dosya yok sayılır. Yüklenen algoritma `cublasLtMatmulAlgoCheck`'ten geçmezse yeniden seçilir (sayaç: `disk_rejected`). |
| Graf yakalama | Autotune açıkken yakalama sırasında yeni şekil gelirse `stop.synchronize()` belirsiz bir CUDA hatası veriyordu. | Açık `TensorError`, mesajı "şekli yakalamadan önce bir kez koşun". |
| İstatistik | — | Yeni sayaçlar: `disk_hits`, `disk_rejected`, `workspaces`. |

## Test

`test_cuda_kernels` 134/134. Yeni "Dalga 3.1" bölümü şunları sınar:

- **Eşzamanlılık:** 4 iş parçacığı, her biri kendi akışında, aynı iki şekli Bias epilogüyle 40'ar kez çalıştırıyor. Her iş parçacığının bias'ı farklı ölçekte. Çıktılar tek iş parçacıklı koşuya bit bit eşit; bias karışsaydı fark görünürdü. Plan sayısı 2 (paylaşıldı), en az 5 çalışma alanı açıldı.
- **Disk önbelleği:** ölç → yaz → önbelleği temizle → autotune kapalıyken yükle. İkinci koşu diskten gelir (`disk_hits` 1, `autotuned` 0) ve sonuç bit bit aynı. Damga bozulunca dosya kullanılmıyor.

GPU tam regresyon (bench dışı) **202/202 geçti**: [regression/gpu_ctest.log](regression/gpu_ctest.log).

## Ölçüm: GPT-2 124M eğitimindeki her GEMM şekli

Yöntem:
- **Kod:** `benchmarks/runner/dalga3_gemm.cpp` (Nexus), `dalga3_gemm_torch.py` (PyTorch), `dalga3_gemm_compare.py` (sürücü).
- **İş:** BF16 giriş, BF16 çıkış, epilogsuz; iki taraf aynı işi yapıyor.
- **Tekrar:** 5 tekrar, süreçler dönüşümlü çalışıyor (sıra her tekrarda yer değiştiriyor). Her süreçte 10 ısınma, sonra 7 × 50 çağrı, CUDA event, medyan.
- **PyTorch iki biçimde:** eager ve CUDA graph. Graph biçimi Python ek yükünü ölçümden çıkarıyor.
- **Güç:** her süreçten önce AC kontrolü yapılıyor; AC değilse koşu duruyor.

Oran = PyTorch ms / Nexus ms (> 1 ise Nexus hızlı).

| BT (token) | Nexus sezgisel toplam | PyTorch eager | PyTorch graph | Oran eager | Oran graph | Nexus ≥ graph olan şekil |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 10.413 ms | 10.322 | 10.494 | 0.991 | 1.008 | 12 / 15 |
| 4096 | 40.495 ms | 40.416 | 40.388 | 0.998 | 0.997 | 7 / 15 |

Tablolar şekil şekil: [table-bt1024.md](table-bt1024.md), [table-bt4096.md](table-bt4096.md). Ham veri: [3.1-gemm/](3.1-gemm/).

**Sonuç: GEMM'de Nexus = PyTorch.** Üstünlük iddiası yok.
- İki taraf da cuBLAS(Lt) çağırıyor ve tepeye yakın: büyük şekillerde 25–28 TFLOP/s. Bu GPU'nun BF16 (FP32 biriktirmeli) tepesi yaklaşık 32 TFLOP/s.
- Şekil bazında tekrarlar arası yayılım BT=1024'te %37'ye, BT=4096'da %16'ya kadar çıkıyor. Toplamdaki ±%1 fark bu gürültünün içinde.
- Plan hedefi "her şekilde ≥ 1.00×" **karşılanmadı**. Tek tek şekillerde 0.91–1.11× aralığı ölçüldü.

**Autotune toplamda kazandırmıyor.** BT=1024'te 10.18 vs 10.41 ms sezgisel, BT=4096'da 40.49 vs 40.50 ms. Bazı şekillerde kazandırıyor, bazılarında kaybettiriyor; süreç içi ölçüm, yeni süreçteki ölçümü tam yansıtmıyor. Bu yüzden varsayılan olarak kapalı kalıyor.

### Epilog füzyonu: fwd.fc (bias + GELU, ön-aktivasyon aux olarak saklanıyor)

| BT | Nexus BiasGelu+aux (tek çağrı) | Nexus Bias GEMM + ayrı GELU | PyTorch addmm + gelu | Oran (füzyonlu) |
|---|---:|---:|---:|---:|
| 1024 | 0.2199 ms | 0.2660 | 0.2630 | **1.196×** |
| 4096 | 0.9595 ms | 0.9582 | 0.8707 | **0.907×** (yavaş) |

- BT=1024'te plan hedefi "epilog dahil ≥ 1.10×" karşılandı.
- BT=4096'da Nexus'un iki yolu da PyTorch'tan yavaş.

### BT=4096 bias + GELU yavaşlığı: kök neden

nsys ile BT=4096'da çekirdek dizisi okundu (süreler çekirdek süresi):

| Yol | Çekirdekler | Toplam |
|---|---|---:|
| Nexus füzyonlu (Lt BiasGelu + aux) | GEMM 0.627 ms + ayrı `cublasLt::epilogue::impl::globalKernel` 0.191 ms | ≈ 0.818 ms |
| Nexus ayrı yol, eski skaler GELU | GEMM 0.621 ms + GELU 0.173 ms | ≈ 0.794 ms |
| PyTorch addmm + gelu | GEMM 0.66 ms + vektörlü GELU 0.125 ms | ≈ 0.785 ms |

- **Füzyonlu yol:** BT=2048 ve 4096'da cuBLASLt GELU_AUX_BIAS epilogu GEMM içinde değil, ayrı bir çekirdekte çalışıyor. "Tek çağrı" iki çekirdek oluyor. BT=1024'te GELU cutlass çekirdeğinin içinde, o yüzden orada hızlı.
- **Ayrı yol:** Nexus GELU çekirdeği eleman başına skaler yükleme yapıyordu. PyTorch 16 baytlık vektörlü yükleme yapıyor.

**Yapılan:** `gelu_forward` artık 16 bayt hizalı işaretçilerde vektörlü çekirdek kullanıyor (src/cuda/kernels_elementwise.cu). Bir iş parçacığı 8 BF16/F16 ya da 4 F32 eleman işliyor; formül aynı. Hizasız ya da kısa girişte eski skaler çekirdek. Yeni test `test_gelu_vectorized`: n = 1000003, üç tip, vektörlü yol skaler yola **bit bit eşit**. `test_cuda_kernels` 137/137.

BT=4096'da GELU çekirdeği 0.173 → 0.124 ms. Ayrı yol (Bias GEMM + vektörlü GELU), süreç içi ölçüm, AC:

| BT | Ayrı yol | Füzyonlu yol |
|---|---:|---:|
| 1024 | 0.210 ms | 0.225 ms |
| 2048 | 0.393 ms | 0.405 ms |
| 4096 | 0.745 ms | 0.818 ms |

**GptModel'de füzyon korundu.** fc ileri geçişi ayrı yola çevrilip denendi, sonra geri alındı. Nedenler:
- Ayrı yol sayısal sonucu değiştiriyor. Füzyonlu epilog GELU'yu FP32 birikimden hesaplıyor; ayrı yol önce BF16'ya yuvarlıyor (PyTorch gibi). tiny modelde adım 0 kaybı 9.08886 → 9.08883, 500 adımda B kaybı 6.759 → 6.866. `test_master_weight_modes` deneysel bandı düştü: C − B farkı −%1.5 → −%7.5.
- 124M uçtan uca (BT=1024, AC, eski/yeni dönüşümlü 2'şer koşu): eski 64.08 / 64.17 ms, yeni 64.21 / 63.88 ms. **Fark ölçülemedi.**

Sonuç:
- BT=4096 füzyonlu fc hâlâ PyTorch'un 0.907×'i. Bu yavaşlık cuBLASLt'nin ayrı epilog çekirdeğinden geliyor ve çözülmedi.
- Vektörlü GELU, `gelu_forward` kullanan yollarda (seçici geri hesaplama) devrede.
- `dalga3_gemm_compare.py` ile 5 tekrarlı yeniden karşılaştırma **yapılmadı**: ölçüm sırasında laptop pile geçti.

## Ölçüm notu (geçersiz sayılan koşu)

Bir koşu sırasında laptop pile geçti (BatteryStatus 1, GPU 210 MHz). O koşu ve AC kontrolü yalnız başta yapılan önceki BT=1024 tekrarı **atıldı**. Bu tablodaki sayılar, her süreçten önce AC kontrolü eklenen yeni koşudan geliyor.

## Açık kalanlar (Dalga 3)

- 3.1:
  - BT ≥ 2048 füzyonlu fc'de cuBLASLt'nin ayrı epilog çekirdeği (kök bulundu, çözülmedi); FP32 ön-aktivasyonu koruyan ayrı yol ya da kendi epilogumuz;
  - residual epilogu (ayrı C girişi, D çıkışı);
  - dgelu yolunun ölçümü;
  - decode için split-K.
- 3.2 FlashAttention-2 yeniden yazımı (cp.async, ldmatrix + mma, swizzle).
- 3.3 füzyon çekirdekleri.
- 3.4 graf içi dinamik LR / ölçek.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
