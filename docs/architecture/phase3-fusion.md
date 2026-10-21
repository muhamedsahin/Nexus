# Faz 3 fuzyon

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Faz 3 uc yeri birbirine baglar: NexusOptim guncellemesi, NexusModel matmul politikasi ve NexusLoss tek gecis kaybi.

## Optimizator

AVX2 Adam ve duz SGD ana dongusu 64 float acilir. Adam'in ters karekok yaklasimi iki Newton adimi kullanir; AVX-512 yolu `_mm512_sqrt_ps` ile kalir.

`adam_update_hybrid` ve `sgd_update_hybrid`, n en az 4096 ve CUDA gorunurse vektorun son %75'ini (64 bayt hizali) cihaza koyar. On ek CPU cekirdeginde, sonek CUDA cekirdeginde ayni anda guncellenir. Cihaz tamponu adimlar arasinda durur. Ayni host isaretcileriyle gelen sonraki adim parametre ve momentleri yeniden yuklemez; yalnizca gradyan gider, guncellenmis degerler hosta doner. CUDA yoksa tum vektor CPU'da kalir.

Kok farki (AVX2 `rsqrt` ile `sqrtf`) bit esitligi vermez. Faz 3 testi mutlak hata 1e-5 altini bekler.

## Model

`MatmulPolicy` varsayilan olarak `CpuKernel` dir; mevcut `kernels::linear_forward` degismez. `Hybrid` ve bir `PackedGemmFn` kurulunca `Linear`, agirligi `[out, in]` biciminden satir-major `W^T` bicimine cevirir ve `Y = X * W^T` carpar, sonra bias ekler. Ceviri tamponu katmanda durur; her ileri gecis yeni vektor ayirmaz. Kanca, `HeterogeneousScheduler::gemm` olabilir. Satir, giris ve cikis 32'den kucukse CPU cekirdegi kullanilir.

## Kayip

`nexusloss` `src/fused_simd.cpp` iceren statik kutuphanedir. `MSELoss`, `MAELoss`, `HuberLoss` ve `BCEWithLogitsLoss` icin `forward_and_backward` kaybi ve gradyani tek geciste yazar. Mean ve BatchMean gradyani `1/N` ile olcekler. MSE, MAE ve Huber AVX2 acikken 8'li islenir. BCE logit formulu `log1p(exp)` icerdigi icin skaler kalir. Sifir varyans `rsqrt(0)` sonsuz dondurur; Adam cekirdegi bu seridi sifir karekok olarak birakir, boylece payda `eps` olur.

## Olcum (Release, bu makine)

n = 1 048 576 Adam adimi. Linear 128x256x256. Hibrit yol dogru sonucu verir; bu boyutta kopya ve zamanlayici maliyeti yuzunden CPU cekirdeginin gerisinde kalir.

| islem | sure | hiz |
| --- | --- | --- |
| Adam CPU | 611 µs | 27.4 GB/s |
| Adam hibrit | 2752 µs | 6.10 GB/s (262144 CPU + 786432 GPU) |
| Linear CPU | 195 µs | 86.2 GFLOPS |
| Linear hibrit | 406 µs | 41.3 GFLOPS |
