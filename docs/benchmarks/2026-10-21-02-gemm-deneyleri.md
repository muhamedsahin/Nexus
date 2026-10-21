# 21 Ekim 2026 — GEMM deneyleri

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Seçili çekirdek ölçüm sonunda yine `avx2-4x16`. İki yeni algoritma derlendi, aynı protokolle ölçüldü, N=1024 ve N=2048'i yavaşlattığı için kaynak ağacından çıkarıldı.

## Durulan çekirdek

MASM 4×16. K yönü 8'er açılır. Biriktiriciler YMM0–YMM7. `kMr=4`, `kNr=16`, `kKc=512`. N en az 512 ise A, 64 satırlık blokta bir kez paketlenir ve B dilimi bu bloktaki bütün 4'lü gruplara uygulanır. N 512'den küçükse eski 4 satırlık döngü durur; 64 satırlık paket orada yavaşlatmıştı.

Zen 3 FMA gecikmesi yaklaşık 4 çevrim, çevrim başına 2 FMA. Sekiz biriktirici bu çarpımı tam doldurur. Daha geniş karo için yazmaç kalmaz: 6×16 on iki biriktirici ister, MSVC bunu C++ döngüsünde yığına döker. O yüzden 6×16 elle MASM ile denendi.

## Deney A — sütun dilimi, B L2'de

Düşünce: N=2048 ve Kc=512 iken paketlenmiş B yaklaşık 4 MB'dır, çekirdek L2'si 512 KB'dır. N, sekiz fiziksel çekirdeğe bölünürse her çekirdeğin B paneli L2'ye sığar. A, her K diliminde bir kez, paylaşılan tampona paketlenir. Bu, daha önce yavaşlatan yöntem değildir. Eski Nc deneyi A'yı her sütun diliminde yeniden paketliyordu. Burada A bir kez paketlendi, iş parçacığı ölçüm süresince bir fiziksel çekirdeğe iğnelendi, bitince iğne kalktı.

Doğruluk: `test_phase13_isa` geçti. 80×1024×128, 70×1032×160 ve 128×2048×256 referansla karşılaştırıldı.

Hız, aynı protokol:

| N | Bu deney | Aynı gün 4×16 |
|---:|---:|---:|
| 256 | 139 | 146–164 bandı, bu yol zaten 256'ya girmez |
| 512 | 211 | 235–295 |
| 1024 | 129 | 325–338 |
| 2048 | 299 | 337–362 |

N=1024 yaklaşık 2.5 kat yavaşladı. Sabit gider yaklaşık 10 ms: her K diliminde tek iş parçacığında A paketleme, sonra havuzu uyandırma, sonra ilginlik maskesini değiştirme. Büyük matrisin hesabı bu gideri ödemiyor. Yol silindi.

## Deney B — 6×16 mikroçekirdek

Düşünce: sekiz biriktirici gecikmenin tam sınırında. On iki biriktirici FMA zincirini rahatlatır. A, 8 genişliğe doldurulur; çekirdek ilk 6 satırı okur, son iki şerit sıfırdır. Paketleme 8×8 AVX2 devriğidir. Yol yalnız N≥1024 için açıldı. N=256 ve N=512 4×16'da kaldı.

Doğruluk geçti, buna kuyruk K=130 da dahildi.

Hız:

| N | 6×16 koşusu | Not |
|---:|---:|---|
| 256 | 52 | Kod bu boyuta girmedi. Aynı koşuda makine düşüktü |
| 512 | 272 | Yine 4×16 |
| 1024 | 300 | 4×16 turlarındaki 331–350'nin altında |
| 2048 | 259 | 4×16 turlarındaki 330–369'un altında |

N=2048'de 259, aynı akşam 4×16'nın 362'sinden düşük. Ek yayınlar ve 8'e doldurulmuş A, on iki biriktiricinin kazancını yedi. MASM yolu ve C++ paketlemesi silindi. `kernel_edge` genişlik parametresi duruyor; çağrılar hâlâ 4 gönderiyor.

## Geri alınmış ikilinin üç turu

Kaynak deneylerden sonraki 4×16. Ayrıntı `2026-10-21-nexus-cpu.json`.

| İş | Tur 1 | Tur 2, hemen ardından | Rakipten sonra |
|---|---:|---:|---:|
| GEMM 256 | 126 | 115 | 62 |
| GEMM 512 | 259 | 275 | 188 |
| GEMM 1024 | 331 | 270 | 350 |
| GEMM 2048 | 362 | 330 | 369 |
| Conv2d | 5.43 | 4.90 | 6.78 |
| Depthwise | 10.4 | 8.95 | 12.1 |

N=2048 üç turda 330, 362 ve 369. Kod aynı. Tur 2, bir önceki turun üstüne binince düştü. N=256'nın 62'si çekirdek değişimi değil; o boyutun yolu deneylerden beri aynı ve sabah 164 görmüştü.

## Bu akşam rakip GEMM, 8 iş parçacığı

| N | Nexus tur 1 / rakip sonrası | NumPy | PyTorch | TensorFlow | JAX |
|---:|---:|---:|---:|---:|---:|
| 256 | 126 / 62 | 126 | 156 | 39.3 | 85.0 |
| 512 | 259 / 188 | 175 | 251 | 133 | 265 |
| 1024 | 331 / 350 | 302 | 221 | 312 | 317 |
| 2048 | 362 / 369 | 353 | 336 | 389 | 292 |

Birim GFLOPS. TensorFlow N=2048 bu akşam 389. Nexus'un en yüksek turu 369, en düşük 330. Bu satırda TensorFlow önde. NumPy 353, PyTorch 336 ve JAX 292, Nexus'un 362 ve 369 turlarının gerisinde. N=1024'te rakip sonrası tur 350 ile bu akşamın en yüksek GEMM satırı Nexus'ta. N=256 sıralaması turdan tura değişiyor; 62'lik turu bir yenilgi olarak kilitlemiyorum, 126'lık tur da PyTorch'un 156'sının gerisinde.

Sabahki oturumda TensorFlow N=2048 383, JAX 366, PyTorch 346, Nexus 337 idi. Bu akşam JAX 292'ye indi, NumPy 221'den 353'e çıktı. Aynı protokol, aynı sürümler, farklı frekans. Tek turun sırası bir dünya rekoru değildir.
