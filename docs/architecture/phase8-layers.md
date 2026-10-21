# Faz 8 yuksek seviye katmanlar

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`nexus::nn` artik `Conv2d`, `MultiHeadAttention` ve `LSTM` katmanlarini `nexus_model` uygulamalarinin uzerine koyar. `Sequential::add` bu katmanlari da alir.

- `Conv2d` girdi duzeni `[N, C, H, W]`. `padding = kernel/2` ve stride 1 iken uzamsal boyut korunur.
- `MultiHeadAttention` girdi duzeni `[B, T, E]`. `E`, bas sayisina bolunur. Dropout kapali, nedensel maske kapali.
- `LSTM` girdi duzeni `[B, T, C]`, cikti `[B, T, H]`. Unutma kapisi bias'i 1 ile baslar.

Geri yayilim parametre gradyanini biriktirir. `zero_grad` bir sonraki adimdan once cagrilir.

Release olcumu, isinma haric 6 tekrar:

| katman | us | cagri/s | GFLOPS |
| --- | --- | --- | --- |
| Conv2d 8x3x32x32, cekirdek 3 | 1140 | 878 | 6.21 |
| Attention B4 T32 E64 H4 | 150 | 6673 |  |
| LSTM B4 T16 C32 H64 | 244 | 4091 |  |

Konvolusyon bu boyutta im2col kurulumunda kalir. GFLOPS yalnizca carpan-toplamalari sayar.
