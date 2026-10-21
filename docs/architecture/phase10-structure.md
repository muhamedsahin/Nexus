# Faz 10 yapi katmanlari

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`nexus::nn` su katmanlari `nexus_model` uzerinden sunar:

- `MaxPool2d(kernel, stride, padding)`: girdi `[N, C, H, W]`. Esit maksimumlarda ilk konum gradyani alir.
- `AvgPool2d(kernel, stride, padding)`: pencere paydasi dolguyu da sayar.
- `GlobalAvgPool2d`: uzamsal eksenleri kanal basina ortalar, cikti `[N, C]`.
- `Flatten`: `[N, ...]` girisini `[N, -1]` yapar. Geri yayilim eski sirayi geri yazar.
- `Dropout(p)`: egitimde ters dropout, degerlendirmede birim donusum. `p = 1` ciktisi sifirdir.
- `BatchNorm2d(features)`: kanal ekseni 1'dir. `eval` kosu istatistigini kullanir.
- `Residual(modul)`: `y = F(x) + x`. `F` eleman sayisini korur. Parametreler `F` uzerinden toplanir.

`train` ve `eval`, `Sequential` dahil, alt katmana iletilir. Dropout ve BatchNorm bu bayragi okur.

Release olcumu, CUDA agaci, isinma haric 8 tekrar. Bayt sayisi okunan girdi, yazilan cikti ve MaxPool indeks tamponudur. Dropout maskesi de float tampon olarak sayilir.

| katman | us | cagri/s | GB/s |
| --- | ---: | ---: | ---: |
| MaxPool2d 8x16x32x32, cekirdek 2 | 576 | 1736 | 1.37 |
| AvgPool2d 8x16x32x32, cekirdek 2 | 504 | 1986 | 1.30 |
| GlobalAvgPool2d 8x16x32x32 | 242 | 4134 | 2.17 |
| Flatten 8x16x32x32 | 22.2 | 44994 | 47.2 |
| Dropout 32x1024, p 0.1 | 267 | 3741 | 1.47 |
| BatchNorm2d 8x32x16x16 | 319 | 3138 | 1.65 |

Havuz ve normalizasyon bu boyutta skaler kurulumdadir. Flatten yalnizca bellek kopyalar. Sarmalayici ayri bir cekirdek calistirmaz.
