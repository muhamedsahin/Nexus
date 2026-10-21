# NexusTrain kullanım kılavuzu

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Bu sayfa `include/nexus/train/trainer.hpp` içindeki `TrainerBuilder` ile bir eğitim kurar. Örnek AdamW, `CrossEntropy`, CPU yürütme politikası, gradyan kırpma ve `EarlyStopping` kullanır. Ayrıntılı döngü `README.md` dosyasındadır.

`Sequential` kopyası katman listesini paylaşır. `TrainerBuilder` modeli değerle alır; `model->parameters()` eğitilen ağırlıkları göstermeye devam eder.

## Örnek

```cpp
#include "nexus/nexus.hpp"

#include <iostream>
#include <vector>

int main() {
    constexpr int ozellik = 2;
    constexpr int ornek = 64;

    std::vector<float> egitim;
    std::vector<int> etiket;
    std::vector<float> dogrulama;
    std::vector<int> dogrulama_etiket;
    egitim.reserve(static_cast<std::size_t>(ornek * ozellik));
    dogrulama.reserve(static_cast<std::size_t>(ornek * ozellik));

    for (int i = 0; i < ornek; ++i) {
        const float x = (i % 2 == 0) ? 1.0f : -1.0f;
        egitim.push_back(x);
        egitim.push_back(0.1f * static_cast<float>(i % 5));
        etiket.push_back(x > 0.0f ? 1 : 0);
        dogrulama.push_back(-x);
        dogrulama.push_back(0.05f);
        dogrulama_etiket.push_back(-x > 0.0f ? 1 : 0);
    }

    auto model = nexus::nn::Sequential()
                     .add<nexus::nn::Linear>(2, 8, true)
                     .add(nexus::nn::swish())
                     .add<nexus::nn::Linear>(8, 2, true);

    auto trainer = nexus::TrainerBuilder(model)
                       .optimizer(nexus::optim::AdamW(
                           model->parameters(),
                           nexus::optim::AdamWOptions{.lr = 0.05f, .weight_decay = 0.0f}))
                       .loss(nexus::loss::CrossEntropy(0.0f))
                       .execution_policy(nexus::ExecutionPolicy::cpu())
                       .gradient_clip_norm(1.0f)
                       .metric<nexus::metrics::Accuracy>()
                       .callback<nexus::callbacks::EarlyStopping>(5)
                       .build();

    auto train = nexus::data::DataLoader(std::move(egitim), std::move(etiket), ozellik, 16, true, 1);
    auto validation =
        nexus::data::DataLoader(std::move(dogrulama), std::move(dogrulama_etiket), ozellik, 16, false, 1);

    trainer.fit(train, validation, 12);

    for (const nexus::EpochStats& tur : trainer.history()) {
        std::cout << tur.epoch << " " << tur.train_loss << " " << tur.val_loss << " " << tur.accuracy << "\n";
    }
    return 0;
}
```

Üst Nexus hedefi `nexus::nexus` bu başlığı ve döngüyü bağlar. `fit` eğitim ve doğrulama yükleyicilerini referansla alır; ikisi de adlı nesne olmalıdır.

## Kurucu satırları

`optimizer(AdamW)` parametre işaretçilerini ve `AdamWOptions` alanlarını alır. Örnekteki öğrenme hızı `0.05`, ağırlık sönümü `0` dir. Yazılmayan `beta1`, `beta2` ve `eps` yapı içindeki varsayılanlarda kalır: `0.9`, `0.999`, `1e-8`. Aynı kurucunun `optimizer(SGD)` yüklemesi de vardır.

`loss(CrossEntropy(0.0f))` etiket yumuşatmasını kapatır. Argüman `[0, 1)` dışında ise kurucu hata verir. `Trainer` kaybı logit üzerinden, batch ortalaması olarak hesaplar.

`execution_policy(ExecutionPolicy::cpu())` varsayılan politikayı açıkça yazar. `fit` bu politikada matmul için CPU çekirdeğini seçer. `ExecutionPolicy::hybrid_cpu_multi_gpu()` yalnızca kullanıcı istediğinde kurulur ve MatrixFlash zamanlayıcısına bağlanır. `ExecutionPolicy::cuda` aynı CPU matmul çekirdeğine düşer. Matmul FP32 kalır.

`gradient_clip_norm(1.0f)` adım atılacak batch'te global normu 1 ile sınırlar. Norm sıfırdan büyük değilse `fit` kırpma çağırmaz.

`metric<Accuracy>()` doğrulama geçişinde argmax doğruluğunu biriktirir. `metric<F1Score>()` aynı geçişte makro F1 tutar. `EpochStats::accuracy` ve `EpochStats::f1` bu doğrulama değerleridir.

`callback<EarlyStopping>(5)` sabır 5, `min_delta` 0 kurar. Doğrulama kaybı `best - min_delta` kadar iyileşmezse kötü epoch artar. Beş epoch sonunda `stop()` döner ve `fit` döngüden çıkar. İkinci argüman `callback<EarlyStopping>(5, 1.0e-4f)` gibi `min_delta` verir. `patience < 1` hata verir.

`build()` optimizatör ve kayıp yoksa hata verir. Dönüş `Trainer` dir.

## Epoch sonu

İsteğe bağlı zamanlayıcı her epochun sonunda bir kez çalışır. C++ bağlama `learning_rate_schedule` dir; Python adları `cosine` ve `warmup` tir. `fit` dönüş değerini `optimizer->set_learning_rate` ile yazar. Kurulmazsa hız, optimizatördeki değerde kalır. Tek yuva vardır: ikinci zamanlayıcı birincinin yerini alır.

```cpp
.learning_rate_schedule(nexus::optim::CosineAnnealing(0.05f, 12))
```

`CosineAnnealing(base_lr, t_max, eta_min)` ve `LinearWarmup(base_lr, warmup_steps)` `step()` ile yeni hız üretir. `fit` bu çağrıyı epoch bittikten sonra, geri çağrılardan önce yapar.

Karma hassasiyet bu örnekte kapalıdır. `mixed_precision(true)` kayıp gradyanını 128 ile başlayan bir ölçekle çarpar; matmul FP32 kalır. Sonlu olmayan parametre gradyanı o batch'te kırpmayı ve `step` çağrısını atlar, ölçeği yarıya indirir ve tabanı 1 de tutar.

## Lisans

Lisans Apache-2.0'dır. Metin üst Nexus `LICENSE` dosyasında ve kardeş kütüphanelerin lisans dosyalarında durur.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
