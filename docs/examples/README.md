# Ornekler

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Ornekler kok derlemede `NEXUS_BUILD_EXAMPLES=ON` iken uretilir. Her biri basarisiz olursa sifirdan farkli kodla cikar.

| Dosya | Ne gosterir |
| --- | --- |
| `01_owning_tensor.cpp` | `Tensor::zeros`, 64 bayt hiza, eleman yazma |
| `02_zero_copy_model.cpp` | `nexus_model::Tensor` ile ayni adres |
| `03_ndarray_and_batch.cpp` | `Batch` girdi ve etiket gorunumu |
| `04_backward_into.cpp` | MSE turevinin cagiran tamponuna yazilmasi |
| `05_share_lifetime.cpp` | `share` sonrasi NDArray olunce bellegin yasamasi |
| `06_custom_activation.cpp` | Lambda Swish ve sonlu fark turetivi |

Derlenmis adlar ayni koktendir: `01_owning_tensor`, `02_zero_copy_model`, ...

Asgari include, tum cerceveyi cekmez:

```cpp
#include "nexus/core/tensor.hpp"          // sahip olunan tensör
#include "nexus/core/tensor_bridge.hpp"   // view / share
#include "nexus/loss/backward_into.hpp"   // nexus::backward_into
```

`#include <nexus/nexus.hpp>` alti kutuphanenin semsiye basligini da acar. Ornekler bunu bilerek kullanmaz; bir ceviri biriminin derlenme suresi kucuk kalsin diye dar baslik yeter.
