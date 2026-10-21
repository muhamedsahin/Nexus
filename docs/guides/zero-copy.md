# Kopya var mi, yok mu

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

| Cagri | Float dizisi | Ne ayirir |
| --- | --- | --- |
| `nexus::view(model)` | kopyalamaz | yalnizca yigindaki baslik |
| `nexus::view(array)` | kopyalamaz | baslik |
| `nexus::view_inputs(batch)` | kopyalamaz | baslik |
| `nexus::view(matrix_pro_tensor)` | kopyalamaz | baslik; host veya cihaz isaretcisi |
| `nexus::share(array)` | kopyalamaz | `shared_ptr` sayaci |
| `Tensor::alias(ptr, shape)` | kopyalamaz | sahip degil |
| `Tensor::zeros` | yeni blok | 64 bayt hizali host |
| `Tensor::pinned` | yeni blok | sayfa kilitli host |
| `Tensor` kopya kurucusu | kopyalar | ayni bellek alaninda yeni blok |
| `loss.backward()` | sonucu yeni vektore yazar | bir `std::vector` |
| `loss.backward_into(span)` | cagiranin tamponuna yazar | basarili yolda hic |

`backward_into` icinde atilan hata (`std::invalid_argument`) istisna nesnesi ayirabilir. Bu, hatali cagri yoludur. Dogru uzunluktaki span'de ayirma olmaz; test bunu `operator new` sayaciyla olcer.
