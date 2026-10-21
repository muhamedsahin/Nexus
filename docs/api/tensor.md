# Tensor ve TensorView

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

## Sahip olan tensör

```cpp
#include "nexus/core/tensor.hpp"

nexus::Tensor w = nexus::Tensor::zeros(nexus::Shape{128, 64});
nexus::Tensor pinned = nexus::Tensor::pinned(nexus::Shape{1024});
nexus::Tensor gpu = nexus::Tensor::zeros(
    nexus::Shape{1024}, nexus::MemorySpace::Device, nexus::DeviceRef::cuda(0));
```

`Tensor(shape)` bellegi ayirir ama doldurmaz. `zeros` ayirir ve sifirlar. `operator[]` yalnizca host ve pinned icin gecerlidir; cihaz isaretcisi host'ta okunamaz.

`alias(ptr, shape, space, device)` dis isaretciyi sarar ve yikicide serbest birakmaz. `ptr` bos olmayan bir sekil icin null olamaz.

## Gorunum

```cpp
nexus::TensorView<float> view = w.view();
std::span<float> flat = view.elements();  // yalnizca bitisik
float value = view.at({0, 3});
```

Bitisik olmayan bir gorunumde `elements()` ve dogrusal `operator[]` hata verir. Eleman `at({i, j})` adimlarla hesaplanir.

Sekil 8 ekseni asarsa `TensorError` atilir. Boyut sessizce kesilmez.
