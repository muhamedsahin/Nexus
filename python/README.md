# Nexus Python kabugu

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Cagri adlari C++ yuzeyinin kendisidir. PyTorch taklidi yoktur.

## Derleme

Anaconda Python 3.11 kullanin. ESP-IDF Python 3.14 kullanilmaz.

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build-phase1 -DNEXUS_BUILD_PYTHON=ON -DPython3_EXECUTABLE=C:\ProgramData\anaconda3\python.exe
"C:\Program Files\CMake\bin\cmake.exe" --build build-phase1 --target _core --parallel
```

Modul `python/src/nexus/_core.pyd` olarak yazilir.

```bat
set PYTHONPATH=python\src
C:\ProgramData\anaconda3\python.exe -m pytest python\tests
```

## Model dosyasi

`model.save_nxs("ag.nxs")` tek dosya yazar. Sihir `NXS2` dir. `nexus.load` agi tur kimliginden kurar. Egitmen durumunun eski `NXS1` dosyasi model diye acilmaz.

`Swish` adi C++ formuludur. Baska bir adla kurulan aktivasyonun govdesi ayni dosyaya yazilir ve `nexus.load` onu geri kurar. Kapali hucre yalniz float, int veya bool ise yazilir. `data.save_image` PNG, JPEG ve BMP yazar. `data.load_image` kanal ve hedef boy alir.

## Graf ve dil modeli

`nn.Graph` adli dugum tutar. `add(name, layer)` onceki dugumu okur. `concat(name, layer, inputs)` son eksende birlestirir. `add(name, layer, inputs, "add")` eleman sayilari esitse toplar. Besinci katman `["katman4", "katman2"]` alabilir.

`nn.CausalTransformerBlock(embed, heads, feedforward)` tek tensor alir. Dikkat nedenseldir. `train.LanguageTrainer(model, learning_rate).fit(tokens, epochs)` kaydirilmis capraz entropi uygular. `tokens` sekli `[batch, time]` olan tamsayi satirlardir. `TrainerBuilder` duz satir siniflandirmasi olarak kalir.

RoPE, uzman karisimi ve KV onbellegi bu turda yoktur.

`nn.Conv2d` CPU'da kalir. `nn.conv2d_device` aygit cekirdegidir. Bu makinede protokol sekli (8, 3, 32, 32) -> 16 filtre, 3x3, pad 1: CPU 6.39 GFLOPS, aygitta kalan cekirdek 48.7 GFLOPS, host gidis-donus 19.0 GFLOPS. cuDNN bu kurulumda yok, kapali kalir. `execution_policy_hybrid` acikca istenir; varsayilan CPU'dur.
