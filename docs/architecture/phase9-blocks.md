# Faz 9 yuksek seviye bloklar

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`nexus::nn` su bloklari `nexus_model` uzerinden sunar:

- `TransformerEncoder(embed, heads, feedforward)`: dropout kapali post-norm kodlayici. Girdi ve cikti `[B, T, E]`.
- `GRU(input, hidden)`: girdi `[B, T, C]`, cikti `[B, T, H]`.
- `Embedding(vocab, dim, padding_idx)`: indeks tensoru tamsayi degerli float'tir. `padding_idx` satiri sifirdir ve gradyan almaz.
- `BatchNorm1d(features)`: kanal ekseni 1'dir. Egitimde batch istatistigi kullanilir.

Geri yayilim parametre gradyanini biriktirir. `zero_grad` bir sonraki adimdan once cagrilir.

Release olcumu, CUDA agaci, isinma haric 4 (Transformer, GRU) veya 8 (Embedding, BatchNorm) tekrar:

| katman | us | cagri/s | not |
| --- | --- | --- | --- |
| Transformer B4 T32 E64 H4 FF128 | 736 | 1359 | yaklasik 9.44e6 carpan-toplama, 12.8 GFLOPS |
| GRU B4 T16 C32 H64 | 305 | 3283 | yaklasik 2.36e6 carpan-toplama, 7.7 GFLOPS |
| Embedding 32x64, sozluk 1024, dim 64 | 38.3 | 26093 | indeks + cikis + her aramada tablo satiri 1.057 MB, 27.6 GB/s |
| BatchNorm1d 64x128 | 114 | 8803 | giris+cikis 64 KiB, 0.58 GB/s |

Bu boyutlar cekirdek kurulumunda kalir. FLOP sayisi matris carpimlarinin carpan-toplamalaridir. Embedding bayt sayisi her aramanin tablo satirini bir kez okudugunu varsayar; ardarda ayni satir onbellekten gelebilir.
