# Rakip benchmark raporu

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Sayılar bu klasördeki JSON dosyalarından üretildi.

## Makine

- Python: `3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]`
- Yorumlayıcı: `C:\Users\muham\Desktop\Nexus\.bench-venv\Scripts\python.exe`
- Platform: `Windows-10-10.0.26200-SP0`
- İşlemci: `AMD Ryzen 7 5800H with Radeon Graphics`
- numpy: `2.4.6`
- torch: `2.14.1+cpu`
- tensorflow: `2.21.0`
- torch.cuda.is_available(): `False`
- tf GPU: `[]`
- Ortam: py -3.12 yok. Sanal ortam Anaconda Python 3.11. Anaconda python.exe yanındaki VCRUNTIME140 resmi CPU tekerlerinin DLL başlatmasını düşürdü. .bench-venv/host aynı yorumlayıcıyı sistem CRT ile çalıştırır.

- Nexus çekirdek: `avx2-4x16`
- Nexus havuz iş parçacığı: `16`
- Derleyici: `MSVC 194434823`
- GPU adı: `NVIDIA GeForce RTX 3070 Laptop GPU`
- NumPy BLAS (8): `scipy-openblas 0.3.31.188.0 OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24`
- NumPy BLAS (default): `scipy-openblas 0.3.31.188.0 OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24`
- PyTorch mkldnn (cpu / 8): `True`
- PyTorch get_num_threads (8): `8`
- torch: Ileri cagri varsayilan eager moddadir. mkldnn kapatilmadi. TF32 kapali.
- PyTorch mkldnn (cpu / default): `True`
- PyTorch get_num_threads (default): `8`
- TensorFlow: tf.function izi isinma cagrisindadir, tekrar ortalamasina girmez. Nexus her cagrida iz cikarmaz. oneDNN varsayilan birakildi.
- TensorFlow intra_op (8): `8`
- TensorFlow conv düzeni (8): `['conv2d:channels_first', 'depthwise:channels_first']`
- TensorFlow intra_op (default): `0`
- TensorFlow conv düzeni (default): `['conv2d:channels_first', 'depthwise:channels_first']`
- jax: GEMM, Conv2d ve Adam. Bias duzeltmesi t=1 sabit, momentler adimlar arasinda tasinir.
- JAX: `0.10.2`
- JAX cihazları: `['cpu:0']`

## Sözleşme

- FP32. GEMM beta `0`. FLOP `2 N³`.
- GEMM N = `[256, 512, 1024, 2048]`. Tekrar `3`, ısınma `1`.
- Conv2d NCHW 8×3×32×32, çekirdek 3, stride 1, padding 1, 16 çıkış.
- Depthwise 8×32×16×16, çekirdek 3, padding 1.
- Attention B4 T32 E64, 4 baş, dropout 0.
- LSTM ve GRU: B4 T16, giriş 32, gizli 64, çıktı dizisi.
- MSE 1048576 eleman, ortalama, gradyan dahil.
- Adam 1048576 parametre, bir adım, momentler dahil.
- Katman ve optimizasyon tekrarı `4`.
- İş parçacığı kolonları `8` ve `default`. GPU kolonu `device`. Kolonlar karışmaz.
- GPU'da ağırlık ve girdi ölçümden önce cihazda. TF32 kapalı. CPU sayısı GPU hücresine yazılmaz.

## CPU, 8 iş parçacığı

Oran = rakip süre / Nexus süre. 1'in üstü, Nexus'un daha kısa sürdüğüdür. Nexus havuz sayısı makine satırındadır; kolon etiketi protokoldeki 8'dir.

| iş | Nexus | NumPy | PyTorch | TensorFlow | NumPy/Nexus | PyTorch/Nexus | TensorFlow/Nexus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMM N=256 | 0.0002298 s · 146.01581 GFLOPS | 0.0003282 s · 102.23776 GFLOPS | 0.0002166 s · 154.91428 GFLOPS | 0.0011976333 s · 28.017283 GFLOPS | 1.4281984 | 0.94255875 | 5.2116333 |
| GEMM N=512 | 0.0011424333 s · 234.96816 GFLOPS | 0.0015757 s · 170.3595 GFLOPS | 0.0012365 s · 217.09297 GFLOPS | 0.0022268333 s · 120.54582 GFLOPS | 1.379249 | 1.0823389 | 1.949202 |
| GEMM N=1024 | 0.0066017333 s · 325.29088 GFLOPS | 0.0083874 s · 256.03687 GFLOPS | 0.0073918667 s · 290.5198 GFLOPS | 0.0068704333 s · 312.56888 GFLOPS | 1.2704845 | 1.1196857 | 1.0407014 |
| GEMM N=2048 | 0.0509663 s · 337.08292 GFLOPS | 0.077635133 s · 221.28988 GFLOPS | 0.0496815 s · 345.80013 GFLOPS | 0.044811067 s · 383.38452 GFLOPS | 1.5232641 | 0.97479119 | 0.87922935 |
| Conv2d | 0.001077325 s · 6.5698726 GFLOPS | koşulmadı | 0.0019237 s · 3.6793097 GFLOPS | 0.001086625 s · 6.5136436 GFLOPS | koşulmadı | 1.7856264 | 1.0086325 |
| Depthwise | 0.00056155 s · 2.1006998 GFLOPS | koşulmadı | 0.0001898 s · 6.2152161 GFLOPS | 0.00265345 s · 0.44457141 GFLOPS | koşulmadı | 0.33799305 | 4.7252248 |
| Attention | 9.565e-05 s · 10454.783 cagri/s | koşulmadı | 0.0004871 s · 2052.9665 cagri/s | 0.000965925 s · 1035.2771 cagri/s | koşulmadı | 5.0925248 | 10.098536 |
| LSTM | 0.000219375 s · 4558.4046 cagri/s | koşulmadı | 0.00057125 s · 1750.547 cagri/s | 0.00160545 s · 622.87832 cagri/s | koşulmadı | 2.6039886 | 7.3182906 |
| GRU | 0.0001773 s · 5640.1579 cagri/s | koşulmadı | 0.00244365 s · 409.22391 cagri/s | 0.00194595 s · 513.88782 cagri/s | koşulmadı | 13.782572 | 10.975465 |
| MSE | 0.0006426 s · 19.581251 GB/s | koşulmadı | 0.001690125 s · 7.4449594 GB/s | 0.00230765 s · 5.4526952 GB/s | koşulmadı | 2.6301354 | 3.5911142 |
| Adam | 0.0007093 s · 23.653202 GB/s | koşulmadı | 0.003626425 s · 4.6263789 GB/s | 0.004222525 s · 3.9732662 GB/s | koşulmadı | 5.1126815 | 5.9530876 |

## CPU, kütüphane varsayılanı

Nexus'un ayrı bir varsayılan iş parçacığı koşusu yoktur. O hücre `koşulmadı` kalır. Oran, Nexus bu kolonda yoksa hesaplanmaz.

| iş | Nexus | NumPy | PyTorch | TensorFlow | NumPy/Nexus | PyTorch/Nexus | TensorFlow/Nexus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMM N=256 | koşulmadı | 0.00034133333 s · 98.304 GFLOPS | 0.00049069999 s · 68.380747 GFLOPS | 0.0011020667 s · 30.446826 GFLOPS | koşulmadı | koşulmadı | koşulmadı |
| GEMM N=512 | koşulmadı | 0.0016775333 s · 160.01796 GFLOPS | 0.00214 s · 125.43713 GFLOPS | 0.0029556667 s · 90.820612 GFLOPS | koşulmadı | koşulmadı | koşulmadı |
| GEMM N=1024 | koşulmadı | 0.0081311333 s · 264.10631 GFLOPS | 0.0085137333 s · 252.2376 GFLOPS | 0.0074182333 s · 289.48721 GFLOPS | koşulmadı | koşulmadı | koşulmadı |
| GEMM N=2048 | koşulmadı | 0.056795833 s · 302.48467 GFLOPS | 0.056910367 s · 301.87592 GFLOPS | 0.044472733 s · 386.30118 GFLOPS | koşulmadı | koşulmadı | koşulmadı |
| Conv2d | koşulmadı | koşulmadı | 0.0002814 s · 25.15241 GFLOPS | 0.00080785 s · 8.7613888 GFLOPS | koşulmadı | koşulmadı | koşulmadı |
| Depthwise | koşulmadı | koşulmadı | 0.000229125 s · 5.148491 GFLOPS | 0.001959775 s · 0.60193032 GFLOPS | koşulmadı | koşulmadı | koşulmadı |
| Attention | koşulmadı | koşulmadı | 0.00069645 s · 1435.8533 cagri/s | 0.0011714 s · 853.67936 cagri/s | koşulmadı | koşulmadı | koşulmadı |
| LSTM | koşulmadı | koşulmadı | 0.001487275 s · 672.37061 cagri/s | 0.00165725 s · 603.40926 cagri/s | koşulmadı | koşulmadı | koşulmadı |
| GRU | koşulmadı | koşulmadı | 0.006523025 s · 153.30311 cagri/s | 0.0018095 s · 552.63885 cagri/s | koşulmadı | koşulmadı | koşulmadı |
| MSE | koşulmadı | koşulmadı | 0.001543775 s · 8.1507422 GB/s | 0.002335 s · 5.3888274 GB/s | koşulmadı | koşulmadı | koşulmadı |
| Adam | koşulmadı | koşulmadı | 0.004135625 s · 4.0567547 GB/s | 0.004925575 s · 3.4061436 GB/s | koşulmadı | koşulmadı | koşulmadı |

## GPU

| iş | Nexus | NumPy | PyTorch | TensorFlow | NumPy/Nexus | PyTorch/Nexus | TensorFlow/Nexus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMM N=256 | 3.64e-05 s · 921.82505 GFLOPS | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| GEMM N=512 | 0.00035343333 s · 759.50803 GFLOPS | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| GEMM N=1024 | 0.00049926667 s · 4301.2758 GFLOPS | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| GEMM N=2048 | 0.0022751333 s · 7551.1483 GFLOPS | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| Conv2d | Nexus GPU yolu yok | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| Depthwise | Nexus GPU yolu yok | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| Attention | Nexus GPU yolu yok | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| LSTM | Nexus GPU yolu yok | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| GRU | Nexus GPU yolu yok | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| MSE | Nexus GPU yolu yok | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |
| Adam | Nexus GPU yolu yok | koşulmadı | GPU tekeri yok, koşulmadı | GPU tekeri yok, koşulmadı | koşulmadı | koşulmadı | koşulmadı |

Durum notları:

- `torch` / `gpu`: GPU tekeri yok, koşulmadı
- `tensorflow` / `gpu`: GPU tekeri yok, koşulmadı

## İkinci ekosistem

| iş | iş parçacığı | JAX | Nexus | JAX/Nexus |
| --- | --- | --- | --- | --- |
| GEMM N=256 | 8 | 0.0003601 s · 93.180873 GFLOPS | 0.0002298 s · 146.01581 GFLOPS | 1.5670148 |
| GEMM N=512 | 8 | 0.00096433333 s · 278.36376 GFLOPS | 0.0011424333 s · 234.96816 GFLOPS | 0.84410469 |
| GEMM N=1024 | 8 | 0.0074972667 s · 286.43554 GFLOPS | 0.0066017333 s · 325.29088 GFLOPS | 1.1356512 |
| GEMM N=2048 | 8 | 0.046925867 s · 366.10659 GFLOPS | 0.0509663 s · 337.08292 GFLOPS | 0.92072343 |
| Conv2d | 8 | 0.0013024 s · 5.4344963 GFLOPS | 0.001077325 s · 6.5698726 GFLOPS | 1.2089202 |
| Adam | 8 | 0.002894 s · 5.7972412 GB/s | 0.0007093 s · 23.653202 GB/s | 4.0800789 |
| GEMM N=256 | default | 0.00027016667 s · 124.19901 GFLOPS | koşulmadı | koşulmadı |
| GEMM N=512 | default | 0.0009194 s · 291.96808 GFLOPS | koşulmadı | koşulmadı |
| GEMM N=1024 | default | 0.0067299333 s · 319.09434 GFLOPS | koşulmadı | koşulmadı |
| GEMM N=2048 | default | 0.054229633 s · 316.79855 GFLOPS | koşulmadı | koşulmadı |
| Conv2d | default | 0.0012513 s · 5.6564277 GFLOPS | koşulmadı | koşulmadı |
| Adam | default | 0.00281095 s · 5.9685217 GB/s | koşulmadı | koşulmadı |

## Nexus'un kazanmadığı satırlar

Aynı cihaz ve aynı iş parçacığı kolonu. Rakip süresi Nexus süresinden kısa ya da eşit.

| iş | cihaz | iş parçacığı | rakip | Nexus saniye | rakip saniye | Nexus değer | rakip değer | oran |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GEMM N=256 | cpu | 8 | torch | 0.0002298 | 0.0002166 | 146.01581 GFLOPS | 154.91428 GFLOPS | 0.94255875 |
| GEMM N=512 | cpu | 8 | jax | 0.0011424333 | 0.00096433333 | 234.96816 GFLOPS | 278.36376 GFLOPS | 0.84410469 |
| GEMM N=2048 | cpu | 8 | jax | 0.0509663 | 0.046925867 | 337.08292 GFLOPS | 366.10659 GFLOPS | 0.92072343 |
| GEMM N=2048 | cpu | 8 | tensorflow | 0.0509663 | 0.044811067 | 337.08292 GFLOPS | 383.38452 GFLOPS | 0.87922935 |
| GEMM N=2048 | cpu | 8 | torch | 0.0509663 | 0.0496815 | 337.08292 GFLOPS | 345.80013 GFLOPS | 0.97479119 |
| Depthwise | cpu | 8 | torch | 0.00056155 | 0.0001898 | 2.1006998 GFLOPS | 6.2152161 GFLOPS | 0.33799305 |

## Ham JSON

- `env`: `C:/Users/muham/Desktop/Nexus/docs/benchmarks/phase-1-env.json`
- `nexus`: `C:/Users/muham/Desktop/Nexus/docs/benchmarks/phase-2-nexus.json`
- `numpy`: `C:/Users/muham/Desktop/Nexus/docs/benchmarks/phase-3-numpy.json`
- `torch`: `C:/Users/muham/Desktop/Nexus/docs/benchmarks/phase-4-pytorch-cpu.json`
- `tensorflow`: `C:/Users/muham/Desktop/Nexus/docs/benchmarks/phase-5-tensorflow-cpu.json`
- `gpu`: `C:/Users/muham/Desktop/Nexus/docs/benchmarks/phase-6-gpu.json`
- `extra`: `C:/Users/muham/Desktop/Nexus/docs/benchmarks/phase-7-extra.json`
- sözleşme: `C:/Users/muham/Desktop/Nexus/benchmarks/competitors/protocol.json`
