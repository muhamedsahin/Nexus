# 21 Ekim 2026 — Aynı sütun karşılaştırması

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Sütun: CPU, iş parçacığı etiketi `"8"`. Nexus havuzu 16 işçidir; GEMM yaklaşık 8 karo koşar. Rakip JSON'ları bu akşam, Nexus "rakip sonrası" koşusu onların hemen ardından alındı. Nexus tur 1, rakipten önce, aynı ikilidir. İkisini de yazıyorum ki tek tur seçilmiş gibi durmasın.

## Tablo

| İş | Nexus tur 1 | Nexus rakip sonrası | NumPy | PyTorch | TensorFlow | JAX |
|---|---:|---:|---:|---:|---:|---:|
| GEMM 256 GFLOPS | 126 | 62.5 | 126 | 156 | 39.3 | 85.0 |
| GEMM 512 | 259 | 188 | 175 | 251 | 133 | 265 |
| GEMM 1024 | 331 | 350 | 302 | 221 | 312 | 317 |
| GEMM 2048 | 362 | 369 | 353 | 336 | 389 | 292 |
| Conv2d GFLOPS | 5.43 | 6.78 | — | 31.6 | 12.7 | 7.55 |
| Depthwise | 10.4 | 12.1 | — | 4.93 | 0.688 | — |
| Attention çağrı/s | 9653 | 6870 | — | 1192 | 1076 | — |
| LSTM | 4574 | 5667 | — | 1540 | 684 | — |
| GRU | 5103 | 7223 | — | 528 | 606 | — |
| MSE GB/s | 18.7 | 27.6 | — | 9.35 | 5.88 | — |
| Adam GB/s | 28.2 | 24.8 | — | 4.71 | 3.23 | 5.51 |

Tam kesir JSON dosyalarında. Tablodaki kısaltma okuma içindir.

## Bu akşam Nexus'un geride kaldığı satırlar

Aynı aygıt, aynı iş parçacığı etiketi.

1. GEMM N=2048 karşısında TensorFlow. TensorFlow 389 GFLOPS, 0.0442 s mertebesi. Nexus 330, 362 ve 369. Üç tur da 389'un altında.
2. Conv2d karşısında PyTorch 31.6, TensorFlow 12.7 ve JAX 7.55. Nexus 4.90–6.78. Sabahki PyTorch 8 sütunu 3.68 idi; o dosya da duruyor. Bu akşamın ölçüsü 31.6.
3. GEMM N=256 karşısında PyTorch 156. Nexus tur 1'de 126, rakip sonrası turda 62. Aynı kod sabah 164 görmüştü. Sıra sabit değil. N=512'de rakip sonrası tur 188, JAX 265 ve PyTorch 251'in gerisinde; tur 1 ise 259 ile ikisinin de önündeydi. Bu iki boyutu tek sıraya kilitlemiyorum.

## Bu akşam Nexus'un önde olduğu satırlar

Üç turda da, ya da rakip sonrası tur dahil en düşük turda:

- Depthwise, PyTorch 4.93 ve TensorFlow 0.688 karşısında 8.95–12.1.
- GEMM N=1024 rakip sonrası turda 350; TensorFlow 312, JAX 317, NumPy 302, PyTorch 221. Tur 2'de 270 vardı. 350'lik tur bu akşamın rakip satırlarının önünde, 270'lik tur değil.
- GEMM N=2048, TensorFlow hariç: NumPy 353, PyTorch 336, JAX 292. Nexus 362 ve 369 bunların önünde, 330'luk sıcak tur NumPy 353'ün gerisinde.
- Attention, LSTM, GRU, MSE, Adam. En düşük Nexus attention 6870, PyTorch 1192.

## Oran

Protokol oranı `rakip_süre / nexus_süre`. 1'in üstü bu koşuda Nexus'un daha kısa sürdüğünü gösterir. N=2048 için TensorFlow yaklaşık 0.0442 s, Nexus rakip sonrası 0.0466 s, oran yaklaşık 0.95. Depthwise için PyTorch 0.000239 s, Nexus 0.0000974 s, oran yaklaşık 2.45.
