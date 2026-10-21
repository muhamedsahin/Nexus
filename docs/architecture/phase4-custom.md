# Faz 4 kullanici genisletmeleri

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Basliklar `include/nexus/custom/` altindadir. Kullanici kutuphane kaynaklarina dokunmadan aktivasyon, kayip, katman, guncelleme kurali, metrik ve adim gozlemcisi ekler.

## Aktivasyon

`make_activation` ileri lambdayi ve istege bagli geri lambdayi alir. Geri bos ise merkezil sonlu fark turevi uretir. Uzunluk 2048 ve uzerinde is, en fazla 8 thread'e bolunur.

`set_batch` span kancasi skaler yolu degistirir. AVX2 seridi icin `bind_avx2_elementwise` vardir; onu cagiran ceviri birimi `/arch:AVX2` ile derlenir.

N = 1 048 576 ileri gecis. Olcek cekirdegi bellek agirliklidir; Swish `exp` agirliklidir. Havuz yolu her cagrida thread actigi icin kisa olcekte skalerden yavas, Swish'te 1.17 kattir.

| yol | us | cagri/s | GB/s | hizlanma |
| --- | --- | --- | --- | --- |
| skaler olcek | 824 | 1214 | 10.2 | 1.00 |
| havuz olcek | 1325 | 754 | 6.33 | 0.62 |
| AVX2 olcek | 485 | 2064 | 17.3 | 1.70 |
| skaler swish | 6281 | 159 | 1.34 | 1.00 |
| havuz swish | 5353 | 187 | 1.57 | 1.17 |

## Kayip

`make_loss` gradyani kendisi yazar. `make_loss_autograd` yalnizca ileri fonksiyon ister. `make_loss_tape` `MicroTape` uzerinde toplama, carpma, kare, exp, log ve mutlak deger kurar; geri gecis ters moddadir.

## Katman, optimizator, metrik

`CustomModule::add_parameter` agirligi `parameters()` listesine koyar. `CustomOptimizer` her adimda kullanicinin kuralini cagirir. `CustomMetric` batch ortalamalarini uzunlukla birlestirir. `CallbackList` adim sonu kaybini gozlemcilere iletir. Akici `TrainerBuilder` Faz 6 kapsamina birakilmistir.
