# Monorepo hedefleri

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Kok `nexus` statik kutuphanesi yalnizca bellek kapisini derler (`src/core/memory.cpp`). Geri kalan API basliktadir. `nexus::nexus` takma adi tuketiciye su hedefleri `PUBLIC` tasir:

| Hedef | Baslik | Isim alani |
| --- | --- | --- |
| `nexusloss` | `nexusloss/nexusloss.hpp` | `nexusloss` |
| `nexus_optim` | `nexus_optim/nexus_optim.hpp` | `nexus_optim` |
| `nexusdata` | `nexusdata/nexusdata.hpp` | `nexusdata` |
| `nexus_model` | `nexus_model/nexus_model.hpp` | `nexus_model` |
| `nexus_train` | `nexus_train/nexus_train.hpp` | `nexus_train` |
| `matrix_pro` | `matrix_pro/matrix_pro.hpp` | `matrix_pro` |

`matrix_pro` her kok derlemeye girer. CUDA toolkit ve `NEXUS_ENABLE_CUDA` varsa cihaz cekirdekleri de derlenir; yoksa ayni hedef CPU GEMM, elementwise ve is parcacigi havuzuyla kalir. Tuketici her iki durumda `NEXUS_HAS_MATRIX_PRO=1` gorur. Kopru, CUDA acikken `matrix_pro::Tensor` gorunumlerini de acar.

Alt projelerin kendi testleri kok derlemede kapali tutulur. Boylece ayni test adi iki kez tanimlanmaz ve FetchContent ile doktrin indirilmez. Bir alt kutuphaneyi tek basina derlemek isteyen, o kutuphanenin dizininde kendi `CMakeLists.txt` seceneklerini kullanir.
