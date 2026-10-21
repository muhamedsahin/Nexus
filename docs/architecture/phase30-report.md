# Faz 30 kapanis

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Sayilar `bench_phase30_close` ciktisidir. Faz 24 sayilari ayni programin onceki turudur. PyTorch, Eigen ve OpenBLAS calistirilmadi.

| olcu | Faz 24 | Faz 30 | birim |
| --- | ---: | ---: | --- |
| GEMM N256 | 126.16 | 107.03 | GFLOPS |
| GEMM N512 | 300.62 | 234.82 | GFLOPS |
| GEMM N1024 | 342.20 | 335.16 | GFLOPS |
| GEMM N2048 | 365.56 | 353.71 | GFLOPS |
| Conv2d 8x3x32x32 k3 | 6.62 | 6.72 | GFLOPS |
| Depthwise 8x32x16x16 k3 | 2.06 | 2.00 | GFLOPS |
| Attention B4 T32 E64 H4 | 8451 | 9874 | cagri/s |
| LSTM B4 T16 C32 H64 | 5127 | 5482 | cagri/s |
| GRU B4 T16 C32 H64 | 6742 | 6792 | cagri/s |
| MSE driver 1M | 20.12 | 25.03 | GB/s |
| Adam CPU 1M | 18.73 | 22.82 | GB/s |

Dikkat artik `probs @ V` yapar; V devrik tamponu yok. Cagri hizi 8451'den 9874'e cikti.

Zaman-major LSTM/GRU duzeni bu turda LSTM'i 4416 cagri/s'ye dusurdu. Geri alindi. Paketleme duruyor. Geri alinan yol 5482 cagri/s.

Duz Conv2d icin dogrudan 3x3, 4.32 GFLOPS'ta kaldi. Im2col duruyor. Depthwise dogrudan yol duruyor.

GEMM onceden cekme N=512'yi 191 GFLOPS'a dusurdu. Geri alindi. Bu tablodaki GEMM, Faz 24 cekirdeginin bu saatteki olcumudur.

Havuz, cekirdek stride'a esit ve dolgu yokken sinir kontrolu yapmadan gider. LayerNorm geri ve GELU geri AVX2'ye baglandi. HWC 3 kanal LUT cozumu AVX2 biriminde. Resize LUT skaler kaldi.

PPO, DQN, SAC ve VectorEnv kararli adimda ayni tamponu yeniden kullanir.
