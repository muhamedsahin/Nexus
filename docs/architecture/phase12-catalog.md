# Faz 12 katalog

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Bu faz `nexus_model` katman katalogunun yuksek seviye kabugunu kapatir.

- `Conv1d(in, out, kernel, stride, padding, bias)`: girdi `[N, C, L]`.
- `Conv3d(in, out, kernel, stride, padding, bias)`: girdi `[N, C, D, H, W]`.
- `DepthwiseConv2d(channels, kernel, stride, padding)`: grup sayisi kanal sayisidir. Girdi `[N, C, H, W]`.
- `GroupNorm(groups, channels)`: kanal ekseni 1'dir. Kanal sayisi grup sayisina bolunur.
- `InstanceNorm(channels)`: her kanal kendi grubudur.

`padding = kernel/2` ve stride 1 iken uzamsal boyut korunur. Geri yayilim parametre gradyanini biriktirir.

Release olcumu, CUDA agaci, isinma haric. Konvolusyon 4 tekrar, normalizasyon 8 tekrar. FLOP sayisi carpan-toplamadir. Normalizasyon bayti giris ve cikistir.

| katman | us | cagri/s | olcu |
| --- | ---: | ---: | --- |
| Conv1d 8x16x128, cekirdek 3 | 453 | 2206 | 3.15e6 carpan-toplama, 6.94 GFLOPS |
| Conv3d 2x4x8x8x8, cekirdek 3 | 916 | 1091 | 1.77e6 carpan-toplama, 1.93 GFLOPS |
| DepthwiseConv2d 8x32x16x16, cekirdek 3 | 4865 | 206 | 1.18e6 carpan-toplama, 0.24 GFLOPS |
| GroupNorm 8x32x16x16, 8 grup | 303 | 3301 | 512 KiB, 1.73 GB/s |
| InstanceNorm 8x32x16x16 | 317 | 3156 | 512 KiB, 1.65 GB/s |

Depthwise bu boyutta im2col kurulumunda kalir: 32 grup ayri kolon tamponu kurar.
