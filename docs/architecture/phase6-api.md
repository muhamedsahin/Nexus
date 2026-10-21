# Faz 6 yuksek seviye API

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`#include <nexus/nexus.hpp>` model, kayip, optimizer, veri ve egitmeni ayni ad alanina getirir.

```cpp
auto model = nexus::nn::Sequential()
    .add<nexus::nn::Linear>(16, 32)
    .add(nexus::nn::swish())
    .add<nexus::nn::LayerNorm>(32)
    .add<nexus::nn::Linear>(32, 3);

auto trainer = nexus::TrainerBuilder(model)
    .optimizer(nexus::optim::AdamW(model->parameters(), {.lr = 0.05f, .weight_decay = 0.0f}))
    .loss(nexus::loss::CrossEntropy(0.1f))
    .execution_policy(nexus::ExecutionPolicy::cpu())
    .mixed_precision(true)
    .gradient_clip_norm(1.0f)
    .metric<nexus::metrics::Accuracy>()
    .metric<nexus::metrics::F1Score>()
    .callback<nexus::callbacks::EarlyStopping>(5)
    .build();

trainer.fit(train_loader, val_loader, /*epochs=*/12);
```

`ExecutionPolicy::hybrid_cpu_multi_gpu` Linear carpimini MatrixFlash zamanlayicisina baglar. `mixed_precision(true)` kayip olcegi uygular; matmul FP32 kalir, sonsuz gradyanda adim atlanir ve olcek yarilanir.

`nexus::train::all_reduce_mean` rank tamponlarini halka algoritmasiyla ortalar. Zamanlayicilar `CosineAnnealing`, `LinearWarmup` ve `OneCycle` epoch basina hiz uretir.

Uygulamalar `apps/` altindadir. Siniflandirici dogrulama dogrulugu 1.0. Ozel Swish egitimi dogruluk 1.0 ve teyp MSE 2.5. Arena IPPO 1.33'ten 5.5'e, MAPPO 2'den 5.67'ye cikar; uc ajanli takim da kosar; ELO rekabetci oz-oyunu ayirir. N=256 GEMM, isinma disi 4 tekrar: CPU 370 µs / 90.7 GFLOPS, hibrit 296 µs / 113 GFLOPS.
