# Bellek modeli

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Uc bellek alani vardir.

| `MemorySpace` | Ne | Kim ayirir |
| --- | --- | --- |
| `Host` | 64 bayt hizali, sayfalanabilir CPU | `allocate_aligned` |
| `PinnedHost` | Sayfa kilitli host | Once `cudaMallocHost`, olmazsa `VirtualLock` / `mlock` |
| `Device` | Hizlandirici | Faz 1'de yalnizca CUDA `cudaMalloc` |

`DeviceRef` aileyi ve indeksi tasir: `cpu()`, `cuda(0)`, `rocm(0)`, `tpu(0)`, `npu(0)`, `vulkan(0)`, `directml(0)`, `metal(0)`.

ROCm, TPU, NPU, Vulkan, DirectML ve Metal icin **ayirma** Faz 1'de yoktur. `allocate_device` bu ailelerde acik `TensorError` firlatir. Elinde hazir bir aygit isaretcisi olan cagiran `Tensor::alias` ile o adresi goruntuler; kopya yapilmaz.

Ayirma ve serbest birakma bir kilitli tabloya yazilir. `allocation_count()` basarili ayirma sayisidir, canli blok sayisi degildir. `alias` ve `share` bu sayaci artirmaz.

Host ve pinned arasindaki kopya `memcpy` dir. Iki ucundan biri cihazsa kopya `cudaMemcpy` veya farkli aygitlarda `cudaMemcpyPeer` olur. Cihaz kopyasindan once secilen aygit, cagri bitince eski aygita geri alinir.

Tablo kilitlidir: ayirma ve serbest birakma eszamanli cagrilabilir. Tensör elemanlarina eszamanli yazmak veri yarisidir; kopru bunu gizlemez.

## MirroredTensor

`MirroredTensor` ayni elemanlarin host ve cihaz kopyasini tutar. Durum `Synced`, `HostAhead` veya `DeviceAhead` olur. Yazmadan once `modify_host` / `modify_device`, okumadan once `sync_device` / `sync_host` cagrilir. Iki taraf birden guncelse `TensorError` atilir.

Kopya, nesnenin kendi engellemeyen akisindadir. `device_stream()` cekirdeklerin akisidir; bos birakilirsa CUDA'nin eski varsayilan akisidir. Siralama olayladir:

1. H2D yazmadan once cihaz akisi bu olayi bekler (okuma-sonrasi-yazma).
2. H2D bittikten sonra cihaz akisi kopyayi bekler (yazma-sonrasi-okuma).
3. D2H, cihaz akisindaki yazmadan sonra baslar ve hosta donmeden biter.

`sync_device(false)` kopyayi kuyruga koyar. `pending_` true iken `modify_*` once `synchronize()` cagirir; aksi halde host, suren DMA ile yarisir. CPU cihazinda tek tampon vardir, senkron bir seyi kopyalamaz.

`wrap_host` baska bir kutuphanenin sayfalanabilir bellegini sarar. `pin_host()` bu adresi `cudaHostRegister` ile kilitler. Yapici, hizlandirici cihazda dogrudan sayfa kilitli host ayirir.
