# Hizli baslangic

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Asagidaki program uc seyi yapar: sahip olunan bir tensör ayirir, NexusModel tensörune kopyasiz bakar, MSE turevini hazir bir tampona yazar.

```cpp
#include "nexus/core/tensor_bridge.hpp"
#include "nexus/loss/backward_into.hpp"
#include "nexusloss/losses/regression.hpp"

#include <vector>

int main() {
    nexus_model::Tensor layer = nexus_model::Tensor::from_values({1, 2}, {1.0f, 3.0f});
    nexus::TensorView<float> weights = nexus::view(layer);
    // weights.data() ile layer.data() ayni adrestir.

    std::vector<float> prediction{1.0f, 3.0f};
    std::vector<float> target{0.0f, 1.0f};
    nexusloss::MSELoss<float> loss;
    loss.forward(prediction, target);

    std::vector<float> gradient(prediction.size());
    nexus::backward_into(loss, std::span<float>(gradient));
    // gradient == {1, 2}  cunku d/dx (x-y)^2 = 2(x-y) ve Mean bunu 1/2 ile carpar.
    return gradient[0] == 1.0f && gradient[1] == 2.0f ? 0 : 1;
}
```

CMake:

```cmake
find_package(Nexus) # kurulum sonrasi; agac icinde dogrudan hedef:
target_link_libraries(app PRIVATE nexus::nexus)
```

Agac icinde ayri bir `find_package` dosyasi Faz 1'de uretilmez. Tuketici kok projeye `add_subdirectory` eder ya da ayni derlemede `nexus::nexus` hedefini kullanir.
