# backward_into

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`LossBase<T>::backward()` sonuc icin `std::vector` ayirir. `backward_into(std::span<T>)` ayirmaz.

```cpp
nexusloss::MSELoss<float> loss;
loss.forward(prediction, target);

std::vector<float> gradient(prediction.size());
loss.backward_into(gradient);
// veya
nexus::backward_into(loss, std::span<float>(gradient));
```

Eleman turevi dogrudan `gradient` icine yazilir. Mean, her elemani `1/N` ile carpar. Sum olcek uygulamaz. Agirlikli Mean, paydada agirlik toplamini kullanir; sifira yakin payda `epsilon` ile sinirlanir. Formul, vektor donduren `backward()` ile aynidir cunku `backward()` artik kendi tamponunu ayirip `backward_into` cagirir.

`grad_out` uzunlugu tahminle ayni degilse `std::invalid_argument` atilir. `forward` cagrilmamissa `std::logic_error` atilir.

`LossBase` turevleri (`MSE`, `MAE`, `Huber`, `BCE`, `Focal`, ...) `gradient_element_wise_into` yazar. Varsayilan sanal yol, turevi henuz span'e tasinmamis bir sinifta gecici vektor ayirir ve `nexusloss::core::inplace_fallback_allocations()` sayacini artirir. Faz 1 sicak yolu bu sayaci artirmaz.

C++20 kavramlari `nexus::InplaceBackward<Loss, T>` ve `nexus::SizedGradient<Loss>` bu sozlesmeyi adlandirir. `nexus::backward_into(loss, span)` dogrudan yazar. `nexus::backward_into(loss, vector)` tamponu yalnizca boy farkliysa buyutur, sonra ayni yola girer.

VICReg (Bardes, Ponce, LeCun, 2022) `VICRegWorkspace` icinde ortalama, standart sapma ve kovaryansi saklar. `forward` istatistigi bir kez hesaplar; `backward_into` ayni sayilari kullanir, ikinci bir gecis veya yeni tampon ayirmaz.

Span'e yazan diger siniflar: `CrossEntropyLoss`, `LabelSmoothingCrossEntropyLoss`, `KLDivLoss`, regresyon ailesi, siniflandirma `LossBase` turevleri, `PolicyGradientLoss`, `PPOClipLoss`, `ValueLoss`, `EntropyBonusLoss`, `DiceLoss`, `IoULoss`, `TverskyLoss`, `FocalTverskyLoss`.

Cross-entropy, one-hot vektor ayirmaz. Sinif indeksi dongunun icinde hedefe cevrilir:

```text
y_c = (1 - eps) + eps / C
y_i = eps / C          (i != c)
dL/dz_i = softmax(z)_i - y_i
```

`log_sum_exp` maksimumu cikararak `exp` tasmasi yapmaz. Softmax eksi hedefin toplami, eps = 0 iken sifirdir.
