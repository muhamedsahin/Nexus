# Nexus

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Nexus, alti C++20 kutuphanesini tek CMake hedefine baglayan bir derin ogrenme ve pekistirmeli ogrenme cercevesidir. Bellek kopyasini gereksiz yere yapmaz. Matris carpimi CPU AVX yolunda, cuBLAS FP32 yolunda ya da ikisinin zamanlayicisinda kosar. Kullanici, kutuphane kaynaklarina girmeden aktivasyon, kayip, katman ve optimizator ekler. Egitim dongusu `TrainerBuilder` ile kurulur.

## Ne var

| Parca | Is |
| --- | --- |
| MatrixFlash-Pro | GEMM, kolektifler, heterojen zamanlayici |
| NexusData | Batch ve CSV yukleyici |
| NexusLoss | Kayip ve tek gecis gradyan |
| NexusOptim | SGD, Adam, AdamW ve diger guncelleyiciler |
| NexusModel | Linear, normalizasyon, aktivasyon, geri yayilim |
| NexusTrain | Callback ve dagitik ortalama |
| `include/nexus` | Akici API, ozel genisletme, RL |

`nexus::nexus` bu parcalari tek link satirina toplar.

```cmake
target_link_libraries(my_app PRIVATE nexus::nexus)
```

## Kurulum

Windows, Visual Studio 2022, CMake 3.24 ve Ninja gerekir. CUDA 13 varsa GPU yolu acilir; yoksa derleme CPU'da kalir.

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-phase1 -G Ninja -DCMAKE_BUILD_TYPE=Release
"C:\Program Files\CMake\bin\cmake.exe" --build build-phase1 --parallel
"C:\Program Files\CMake\bin\cmake.exe" -E chdir build-phase1 ctest --output-on-failure -C Release
```

Onemli anahtarlar `cmake/NexusOptions.cmake` icindedir: `NEXUS_ENABLE_CUDA`, `NEXUS_BUILD_TESTS`, `NEXUS_BUILD_BENCHMARKS`, `NEXUS_BUILD_EXAMPLES`.

## Egitim

```cpp
#include <nexus/nexus.hpp>

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

trainer.fit(train_loader, val_loader, 12);
```

`mixed_precision(true)` kayip olcegi uygular. Matmul FP32 kalir. Sonsuz gradyanda adim atlanir ve olcek yarilanir.

`ExecutionPolicy::hybrid_cpu_multi_gpu()` Linear carpimini MatrixFlash zamanlayicisina verir. Kucuk matrislerde kopya maliyeti saf GPU'yu gecer. Buyuk ve cihaz icinde kalan carpimda cuBLAS yolu kullanilir.

## Ozel fonksiyon

```cpp
auto swish = nexus::nn::make_activation(
    "Swish",
    [](float x) { return x / (1.0f + std::exp(-x)); });
```

Geri turev verilmezse merkezil sonlu fark kullanilir. `make_loss`, `make_loss_autograd` ve `make_loss_tape` kaybi ya acik gradyanla, ya sayisal turevle, ya da `MicroTape` ile kurar. `CustomModule::add_parameter` agirligi `parameters()` listesine koyar.

## Pekistirmeli ogrenme

Tek ajan: `LineWorld`, `GridWorld`, `CartPole`, `PointMass`. Algoritmalar: DQN (Double, Dueling, PER), PPO, A2C, SAC.

Cok ajan: `ButtonWorld` ve N ajanli `TeamButton` uzerinde IPPO ve MAPPO. `League` ELO ve agirlik anlik goruntusu tutar. `VectorEnv` sekiz ve daha cok kopyada kalici isci kullanir; her adim thread acmaz.

```cpp
nexus::rl::ButtonWorld env;
nexus::rl::MultiAgentTrainer trainer;
auto report = trainer.train_independent(env, 10, 36, 11);
```

## Donanim

`Device::cpu()`, `Device::cuda(0)`, `Device::tpu(0)` ve `Device::hybrid()` etiket uretir. Gercek hesap su an CPU AVX ve CUDA cuBLAS icindir. ROCm, bir `hipblas` kutuphanesi yuklenirse GEMM yapar; DLL yoksa cihaz sayisi sifirdir. NPU, Vulkan, DirectML ve Metal bir eklenti girisi bekler. Olmayan bir aygita bellek ayirma istegi hata verir.

Kolektifler halka ve agactir. `nexus::train::all_reduce_mean` iki veya daha fazla host tamponunu ortalar.

## Python

Anaconda 3.11 kullanilir. ESP-IDF Python 3.14 kullanilmaz.

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-phase1 -DNEXUS_BUILD_PYTHON=ON -DPython3_EXECUTABLE=C:\ProgramData\anaconda3\python.exe
"C:\Program Files\CMake\bin\cmake.exe" --build build-phase1 --target _core --parallel
set PYTHONPATH=python\src
```

Cagri adlari C++ yuzeyinin kendisidir. Model dosyasi `NXS2` sihirli `.nxs` dir. Egitmen durumunun `NXS1` dosyasi model diye acilmaz. Ayrinti `python/README.md` ve `web/` dokumantasyon sitesindedir.

## Bu makinede olculen hiz

Rakip protokol 21 Ekim 2026 aksamidir. CPU, is parcacigi etiketi "8", Ryzen 7 5800H, RTX 3070 Laptop. Ham sayilar `docs/benchmarks/2026-10-21-*.json` icindedir. Dunya siralamasi yoktur.

GEMM N=2048: Nexus turlari 330, 362 ve 369 GFLOPS. TensorFlow 389. NumPy 353, PyTorch 336, JAX 292. Nexus uc turun hepsinde TensorFlow'un altindadir; 362 ve 369 NumPy, PyTorch ve JAX'in onundedir.

Conv2d ayni aksam: Nexus 4.90–6.78 GFLOPS, PyTorch 31.6, TensorFlow 12.7, JAX 7.55. Depthwise: Nexus 8.95–12.1, PyTorch 4.93, TensorFlow 0.688.

Attention, LSTM, GRU, MSE ve Adam bu aksamki 8 is parcacigi sutununda Nexus'un olculen en dusuk turu rakiplerin onundedir. Attention en dusuk Nexus 6870 cagri/s, PyTorch 1192.

CUDA conv kapisi Nexus'un kendi CPU conv'una karstir: protokol sekli 8x3x32x32, 16 filtre, 3x3, pad 1. CPU 6.39 GFLOPS, aygitta kalan cekirdek 48.7, host gidis-donus 19.0. Bu sayi PyTorch satirina yazilmaz. cuDNN bu kurulumda yoktur.

`ExecutionPolicy::cuda` matmul'u CPU cekirdeginde birakir. Hibrit politika yalniz acikca istenirse baglanir.

## Fazlar

1. Kok CMake hedefi, sifir kopya kopru, `backward_into`, `MirroredTensor`.
2. Cihaz soyutlamasi, halka/agac kolektifler, hibrit CPU+GPU GEMM. AVX2 cekirdegi 4x16 MASM.
3. Optimizator SIMD, Linear politikasi, tek gecis kayip.
4. Kullanici aktivasyonu, kaybi, katmani, optimizatoru ve metrigi.
5. Ortam, tampon, DQN, PPO, SAC ve N ajanli IPPO/MAPPO.
6. `TrainerBuilder` ve dort uygulama.
7. Bu sayfadaki olcum tablosu ve iki `.txt` rapor.
8. `nexus::nn::Conv2d`, `MultiHeadAttention` ve `LSTM`.
9. `TransformerEncoder`, `GRU`, `Embedding` ve `BatchNorm1d`.
10. `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`, `Flatten`, `Dropout`, `BatchNorm2d` ve `Residual`. `train` ve `eval` alt katmana iner.
11. `PositionalEncoding`, `TransformerDecoder`, `RNN` ve `Reshape`.
12. `Conv1d`, `Conv3d`, `DepthwiseConv2d`, `GroupNorm` ve `InstanceNorm`. Katalog burada kapanir.
13. GEMM cekirdek adi. Seri AVX-512, AVX2 varken secilmez. Taban olcum.
14. GEMM: K acilimi 8, Nc 128, AVX2 paket ve kenar.
15. 3x3 depthwise ve kucuk gruplar dogrudan konvolusyon.
16. Softmax geri, log-softmax ve erf-GELU AVX2.
17. GRU adayi `tanh_fwd`. LSTM kapisi AVX2 carpma.
18. Batch/GroupNorm ic eksen, dropout maskesi, embedding `memcpy`.
19. MSE surucusu `forward_and_backward`. Seyrek CE one-hot ayirmaz.
20. Hibrit Adam/SGD surec omurlu is parcacigi. Varsayilan adim CPU'da kalir.
21. `persistent_workers` epoch boyunca iscileri tutar.
22. MLP cekirdek carpimi. IndexPool ic ice cagriyi seri yapar.
23. Ozel aktivasyon kalici isci. WorkStealingPool yok.
24. Ayni boyutlarin kapanis olcumu.
25. Dikkat `probs @ V`. V devrigi yok.
26. Zaman-major tekrarlayan duzen olculdu, LSTM yavasladi, geri alindi.
27. Havuz hizali pencere, LayerNorm geri, GELU geri.
28. Duz Conv2d dogrudan 3x3 im2col'dan yavas, im2col kaldi.
29. GEMM onceden cekme N=512'yi dusurdu, geri alindi.
30. PPO, DQN, VectorEnv tamponu ve kapanis olcumu.

## Uygulamalar

| Dizin | Ne yapar |
| --- | --- |
| `apps/dl_vision_classifier` | 8x8 tablo siniflandirma |
| `apps/custom_ops_showcase` | Swish ve teyp MSE |
| `apps/multi_agent_rl_arena` | IPPO, MAPPO, uc ajan ve ELO |
| `apps/hybrid_cluster_benchmark` | CPU, hibrit GEMM ve all-reduce |

## Belgeler

Okuma sirasi `docs/README.md` icindedir. Faz notlari `docs/architecture/` altindadir. Kisa ornekler `docs/examples/` altindadir.

## Test

`ctest` Release agacinda faz koprusu, donanim, fuzyon, ozel genisletme, RL, egitmen ve olcum programlarini kosar. Son kosu CPU ve CUDA agaclarinda 15/15 gecti. Olcum programinin cikis kodu 0'dir ve `sink` 0.1024'tur.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.
