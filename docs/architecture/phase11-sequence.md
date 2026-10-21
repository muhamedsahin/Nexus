# Faz 11 dizi katmanlari

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`nexus::nn` su katmanlari `nexus_model` uzerinden sunar:

- `PositionalEncoding(embed, max_len, learnable)`: girdi `[B, T, E]`. Sabit tabloda konum 0'in cift kanallari 0, tek kanallari 1'dir. `learnable` acikken tablo parametredir.
- `TransformerDecoder(embed, heads, feedforward)`: dropout kapali. `forward(target, memory)` kullanilir. Tek tensörlük `forward` hata verir, bu yuzden `Sequential` icine konmaz.
- `RNN(input, hidden)`: girdi `[B, T, C]`, cikti `[B, T, H]`.
- `Reshape({...})`: eleman sayisi ayni kalir. Ileri gecis depolamayi paylasir. Geri yayilim eski sirayi geri yazar.

Release olcumu, CUDA agaci. Konum kodlama ve yeniden sekil 8 tekrar, cozucu ve RNN 4 tekrar. Isinma haric.

| katman | us | cagri/s | olcu |
| --- | ---: | ---: | --- |
| PositionalEncoding B4 T32 E64 | 19.9 | 50188 | girdi + tablo + cikis 72 KiB, 3.70 GB/s |
| TransformerDecoder B4 T32 E64 H4 FF128 | 947 | 1056 | yaklasik 1.47e7 carpan-toplama, 15.5 GFLOPS |
| RNN B4 T16 C32 H64 | 191 | 5242 | yaklasik 7.86e5 carpan-toplama, 4.12 GFLOPS |
| Reshape 4x32x64 -> 4x2048 | 0.125 | 8.00e6 | gorunum, kopya yok |

Cozucu FLOP sayisi oz-dikkat, capraz dikkat ve iki dogrusal katmanin carpan-toplamalaridir. RNN sayisi girdi ve gizli carpanlaridir.
