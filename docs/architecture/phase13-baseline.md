# Faz 13 taban olcum

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`gemm_f32` AVX2 varken seri AVX-512 cekirdegini secmez. O cekirdek satirlari kendisi dagitmaz ve her N karosunda A'yi yeniden paketler. `gemm_kernel_name()` bu makinede `avx2-4x16` doner.

Sayilar `bench_phase13_baseline` ciktisidir. Sonraki fazlar yalniz bu tabloya gore konusur.

| olcu | us | deger | birim |
| --- | ---: | ---: | --- |
| GEMM N256 | 378.0 | 88.76 | GFLOPS |
| GEMM N512 | 1110.7 | 241.69 | GFLOPS |
| GEMM N1024 | 7017.8 | 306.01 | GFLOPS |
| GEMM N2048 | 47500.8 | 361.68 | GFLOPS |
| Conv2d 8x3x32x32 k3 | 958.3 | 7.39 | GFLOPS |
| Depthwise 8x32x16x16 k3 | 2423.8 | 0.487 | GFLOPS |
| Attention B4 T32 E64 H4 | 108.1 | 9249 | cagri/s |
| LSTM B4 T16 C32 H64 | 171.1 | 5845 | cagri/s |
| GRU B4 T16 C32 H64 | 158.5 | 6311 | cagri/s |
| MSE driver 1M | 4555 | 2.76 | GB/s |
| Adam CPU 1M | 503.5 | 33.32 | GB/s |
