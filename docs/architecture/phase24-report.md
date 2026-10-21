# Faz 24 kapanis

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Ayni boyutlar Faz 13 tabaninda ve bu turda olculdu. Cekirdek adi `avx2-4x16`. PyTorch, Eigen ve OpenBLAS bu makinede calistirilmadi.

| olcu | Faz 13 | Faz 24 | birim |
| --- | ---: | ---: | --- |
| GEMM N256 | 88.76 | 126.16 | GFLOPS |
| GEMM N512 | 241.69 | 300.62 | GFLOPS |
| GEMM N1024 | 306.01 | 342.20 | GFLOPS |
| GEMM N2048 | 361.68 | 365.56 | GFLOPS |
| Conv2d 8x3x32x32 k3 | 7.39 | 6.62 | GFLOPS |
| Depthwise 8x32x16x16 k3 | 0.487 | 2.06 | GFLOPS |
| Attention B4 T32 E64 H4 | 9249 | 8451 | cagri/s |
| LSTM B4 T16 C32 H64 | 5845 | 5127 | cagri/s |
| GRU B4 T16 C32 H64 | 6311 | 6742 | cagri/s |
| MSE driver 1M | 2.76 | 20.12 | GB/s |
| Adam CPU 1M | 33.32 | 18.73 | GB/s |

Depthwise dogrudan 3x3 cekirdege gecti. Duz Conv2d im2col yolunda kaldi; bu turda tabanin altinda, saat dalgalanmasiyla ayni bantta. MSE surucusu artik tek gecis fused cekirdegi cagiriyor. Adam cekirdegi degismedi; 18.73 GB/s bu turun sayisidir, Faz 13'teki 33.32 ile ayni koddur. Dikkat ve LSTM bu turda tabanin altinda kaldi. GRU `tanh_fwd` ile 6742 cagri/s.

Hibrit Adam varsayilan degil. Her adim parametreyi hosta yazmak zorunda. Is parcacigi artik surec omrunde duruyor. GPU sutunu bu tabloda yok; master benchmark ayri kosar, cuBLAS FP32, TF32 kapali, H2D haric.
