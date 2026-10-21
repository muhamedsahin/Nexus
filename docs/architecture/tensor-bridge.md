# Tensör koprusu

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`nexus::view` kaynak tamponun isaretcisini dondurur. Float degerleri kopyalanmaz, sekil yiginda durur (en fazla 8 eksen).

```text
nexus_model::Tensor.data()  ==  view(tensor).data()
NDArray.data<float>()        ==  view(array).data()
Batch.inputs.data<float>()   ==  view_inputs(batch).data()
matrix_pro host_ptr()        ==  view(tensor).data()     (host aynasi doluysa)
device_data() + offset       ==  view(slice).data()      (narrow / device_only)
```

`nexus_model::Tensor` her zaman satir-major host bellegidir. `NDArray` Float32 olmak zorundadir. CUDA uzerindeki bir NDArray'de `data<float>()` bilerek hata verdigi icin kopru ham `void*` isaretcisini kullanir; adres yine aynidir, tur kontrolu `DType::Float32` ile yapilir.

`share(array)` veri kopyalamaz. `NDArray` icindeki `shared_ptr` tensöre kopyalanir (sayac artar). Kaynak nesne yok olsa da float dizisi durur.

`Tensor` kopya kurucusu bilerek derindir. Sifir kopya icin `view()`, `alias()` veya `share()` kullanilir.

MatrixFlash tensöru aygit indeksini saklamaz. Host aynasi yoksa gorunum `cuda(0)` etiketi tasir. Baska aygit icin:

```cpp
auto tagged = nexus::view(slice).with_device(nexus::DeviceRef::cuda(1));
```

Bu cagri da bayt kopyalamaz.
