# Faz 2 donanim katmani

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

`matrix_pro` artik CUDA olmadan da derlenir. `MATRIX_PRO_ENABLE_CUDA=OFF` yalnizca CPU GEMM, halka/agac kolektifleri, HLO dot derleyicisi ve zamanlayiciyi birakir. Toolkit acikken ayni kod cuBLAS FP32 (pedantic, TF32 kapali) ve P2P halka all-reduce ekler.

## Cihazlar

`DeviceManager` CPU'yu her zaman gorur. CUDA aygitlari sayilir. ROCm, `hipblas.dll` yuklenirse hipBLAS GEMM calistirir; DLL yoksa cihaz sayisi 0'dir ve zorlanan cagri hata firlatir. NPU, Vulkan, DirectML ve Metal `CreateMatrixProBackend` eklentisi ile kaydolur.

## GEMM

`gemm_f32` AVX-512, AVX2 FMA veya skaler cekirdegi secer. AVX2 mikrocekirdegi 4x16'dır. MSVC ic fonksiyonu biriktiricileri yigina doktugu icin sicak dongu MASM'dedir; YMM0-YMM7 dongu boyunca yazmacda kalir. B, K dilimi (512) boyunca bir kez paketlenir; satir karolari bu paneli paylaşır. Skaler yol buyuk M'de havuzu kullanır. Referans `gemm_reference` skalerdir.

## Kolektifler

`all_reduce` Sum, Mean ve Max icin halka ve ikili agac uygular. Uzunluk rank sayisina bolunmuyorsa kuyruk ayri toplanir. CUDA rankleri ayni algoritmayi `cudaMemcpyPeerAsync` ve cihaz toplama cekirdegi ile kosar.

## Hibrit

`HeterogeneousScheduler::gemm` A'nin satirlarini olculen GFLOPS ve PCIe maliyetine gore boler. GPU payi bir CUDA akisinda, CPU payi kalici yardimci thread uzerinden ayni anda biter. B'nin host icerigi (isaretci, uzunluk ve FNV ozeti) onceki cagriyla ayniysa H2D atlanir; bir bayt degisirse yeniden kopyalanir. Satir araliklari ayrık oldugu icin sonuc yerinde birlesir.

## HLO

`compile_hlo` yalnizca su dot bicimini kabul eder:

```text
a = f32[M,K] parameter(0)
b = f32[K,N] parameter(1)
ROOT c = f32[M,N] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
```

Baska bir op `MatrixProError` firlatir.
