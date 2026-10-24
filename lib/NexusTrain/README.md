# NexusTrain

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Eğitim döngüsü `nexus::Trainer::fit` içindedir. Bildirim `include/nexus/train/trainer.hpp` dosyasındadır. Kurulum `nexus::TrainerBuilder` ile yapılır. Python tarafı aynı kurucuyu `nexus.train.TrainerBuilder` adıyla açar.

Bu klasörün ayrı bir GitHub uzak deposu yoktur. Kaynak, üst Nexus ağacının içindedir.

## Döngü

`fit(egitim, dogrulama, epoch_sayisi)` `epoch_sayisi < 1` ise hata verir. Döngü süresince matmul politikası seçilen yürütme politikasına çekilir; çıkışta önceki politika geri yazılır.

Her epoch sırası:

1. Eğitim geçişi. Her batch için `zero_grad`, `forward`, `CrossEntropy::forward_backward`.
2. Karma hassasiyet açıksa kayıp gradyanı ölçeklenir, ardından `backward`.
3. Parametre gradyanları sonluysa, istenmişse `clip_grad_norm`, sonra `optimizer.step`. Sonlu değilse adım atlanır.
4. Doğrulama geçişi. Yalnızca `forward` ve kayıp. Metrikler bu geçişin başında sıfırlanır ve batch sonunda güncellenir. Optimizatör adımı yoktur.
5. `EpochStats` yazılır: `train_loss`, `val_loss`, doğrulama `accuracy`, doğrulama `f1`.
6. Zamanlayıcı kurulmuşsa bir kez çağrılır ve sonuç `set_learning_rate` ile yazılır.
7. Geri çağrılar `on_epoch_end` alır. Biri `stop()` döndürürse epoch döngüsü biter.

Raporlanan kayıp, batch kaybının örnek sayısıyla ağırlıklı ortalamasıdır. `CrossEntropy` batch ortalamasını döndürür; ölçek çarpanı bu sayıya yazılmaz. Ölçek yalnızca geri yayılan kayıp gradyanındadır.

`gradient_clip_norm` sıfırdan büyükse ve adım atlanmadıysa kırpma çalışır. Sıfır ve altı kırpmayı kapalı bırakır.

`build()` optimizatör ve kayıp yoksa `std::logic_error` fırlatır. Kayıp tipi `loss::CrossEntropy` dir. Etiket yumuşatma `[0, 1)` aralığındadır.

## Yürütme politikası

Varsayılan `ExecutionPolicy::cpu()` dir. `apply_execution_policy` bu durumda `MatmulPolicy::CpuKernel` yazar.

`execution_policy_hybrid` ancak kullanıcı isterse `ExecutionPolicy::hybrid_cpu_multi_gpu()` çağırır. O politika `hybrid()` doğru olduğunda packed GEMM kancasını MatrixFlash `HeterogeneousScheduler` üzerine alır ve `MatmulPolicy::Hybrid` seçer.

`ExecutionPolicy::cuda` türü vardır. `apply_execution_policy` hibrit olmayan her politikada, bu tür dahil, CPU matmul çekirdeğine düşer. Matmul FP32 kalır.

## Karma hassasiyet

`mixed_precision(true)` kayıp gradyanını `LossScaler` ile çarpar. Başlangıç ölçeği 128 dir. Geri yayılımdan sonra parametre gradyanları aynı ölçeğin tersiyle bölünür. Matmul FP32 kalır.

Sonlu olmayan bir parametre gradyanı adımı atlar. Ölçek `max(1, olcek * 0.5)` olur. Kırpma ve `step` o batch için çalışmaz. Sonraki eğitim batch'i başında `zero_grad` gradyanı temizler.

## Python kurucu

`nexus.train.TrainerBuilder` üzerinde kullanılan yöntemler:

| Yöntem | Bağladığı C++ çağrısı |
| --- | --- |
| `adamw(model, lr=1e-3, weight_decay=0)` | `AdamW`, beta1 `0.9`, beta2 `0.999`, eps `1e-8` |
| `sgd(model, lr=1e-2, momentum=0, weight_decay=0)` | `SGD` |
| `loss(label_smoothing=0)` | `CrossEntropy` |
| `mixed_precision(enabled=True)` | kayıp gradyanı ölçeği |
| `cosine(CosineAnnealing)` | `learning_rate_schedule` |
| `warmup(LinearWarmup)` | `learning_rate_schedule` |
| `callback(Callback)` | epoch sonu gözlemcisi |
| `execution_policy_cpu()` | `ExecutionPolicy::cpu()` |
| `execution_policy_hybrid()` | `ExecutionPolicy::hybrid_cpu_multi_gpu()` |
| `early_stopping(patience)` | `EarlyStopping`, `min_delta` `0` |
| `accuracy()` | `metrics::Accuracy` |
| `f1()` | `metrics::F1Score` |
| `gradient_clip_norm(norm)` | global norm kırpma |
| `build()` | `Trainer` |

`cosine` ve `warmup` tek zamanlayıcı yuvasına yazar. Son çağrı öncekini değiştirir. `fit` bu işlevi her epochun sonunda bir kez çağırır.

```python
import nexus

model = nexus.nn.Sequential()
model.add(nexus.nn.Linear(2, 8, True))
model.add(nexus.nn.swish())
model.add(nexus.nn.Linear(8, 2, True))

trainer = (
    nexus.train.TrainerBuilder(model)
    .adamw(model, lr=0.05)
    .loss(0.0)
    .execution_policy_cpu()
    .gradient_clip_norm(1.0)
    .accuracy()
    .early_stopping(5)
    .build()
)
```

## C++ kurucu

C++ adları Python sarmalayıcısından ayrıdır: `optimizer`, `loss`, `execution_policy`, `mixed_precision`, `gradient_clip_norm`, `metric`, `callback`, `learning_rate_schedule`, `build`. Tam örnek `docs/kilavuz.md` dosyasındadır. Başlık `nexus/nexus.hpp` üzerinden gelir; `Sequential` kopyası katmanları paylaşır, `model->parameters()` aynı ağırlıkları gösterir.

`EarlyStopping` doğrulama kaybını izler. `patience` en az 1 dir. Kayıp `best - min_delta` altına inmezse kötü epoch sayılır; eşik dolunca `stop()` epoch döngüsünü keser. O epoch geçmişe yazılmış ve zamanlayıcı, varsa, o epoch için çalışmış olur.

## Bu klasördeki statik kütüphane

`nexus_train` hedefi checkpoint, günlük ve prefetch parçalarını derler. `examples/train_mnist_full_example.cpp` bu hedefin `nexus_train::Trainer` tipini sentetik etiketle çalıştırır. Python ve `TrainerBuilder`, yukarıdaki `nexus::Trainer::fit` sırasını kullanır.

```text
cmake -S lib/NexusTrain -B lib/NexusTrain/build
cmake --build lib/NexusTrain/build --config Release
ctest --test-dir lib/NexusTrain/build -C Release --output-on-failure
```

Release derlemede MSVC `/O2`, diğer derleyiciler `-O3 -march=native` kullanır. `NEXUS_TRAIN_WITH_DISTRIBUTED` veri-paralel adım yardımcısını derler. Akıcı `nexus::Trainer` üst Nexus hedefi `nexus::nexus` ile bağlanır.

## Lisans

Lisans Apache-2.0'dır. Metin üst ağaçtaki [LICENSE](../../LICENSE) dosyasında ve kardeş kütüphanelerde durur: [NexusData](../NexusData/LICENSE), [NexusLoss](../NexusLoss/LICENSE), [NexusModel](../NexusModel/LICENSE), [NexusOptim](../NexusOptim/LICENSE), [MatrixFlash-Pro](../MatrixFlash-Pro/LICENSE).

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

---

<div align="center">

**🎓 Bu proje BTÜ (Bursa Teknik Üniversitesi) Bilgisayar Mühendisliği 1. sınıf öğrencisi tarafından yapay zeka destekli olarak yapıldı.**

<sub>Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.</sub>

<sub>⚡ [Nexus](https://github.com/muhamedsahin/Nexus) ekosisteminin bir parçasıdır.</sub>

</div>
