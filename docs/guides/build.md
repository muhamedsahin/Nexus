# Derleme

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Kok dizin `c:\Users\muham\Desktop\Nexus`. CMake 3.24 ve bir C++20 derleyicisi gerekir. MatrixFlash-Pro her agaca girer. CUDA toolkit yoksa veya `NEXUS_ENABLE_CUDA=OFF` ise cihaz cekirdekleri derlenmez; CPU GEMM ve bellek koprusu kalir.

```bat
cmake -S . -B build-phase1 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-phase1
ctest --test-dir build-phase1 --output-on-failure
```

## Secenekler

| Secenek | Varsayilan | Anlam |
| --- | --- | --- |
| `NEXUS_ENABLE_CUDA` | ON | nvcc varsa MatrixFlash-Pro baglanir |
| `NEXUS_ENABLE_AVX2` | ON | NexusData AVX2 ceviri birimleri |
| `NEXUS_ENABLE_AVX512` | ON | NexusOptim ve NexusModel AVX-512 birimleri |
| `NEXUS_ENABLE_MULTI_GPU` | OFF | Derleme tanimi. Kolektif motor Faz 2 |
| `NEXUS_ENABLE_TPU_XLA` | OFF | Derleme tanimi. OpenXLA calistirma Faz 2 |
| `NEXUS_ENABLE_ROCM` | OFF | Derleme tanimi. HIP calistirma Faz 2 |
| `NEXUS_BUILD_TESTS` | ON | `test_phase1_*` ve sonraki faz testleri |
| `NEXUS_BUILD_BENCHMARKS` | ON | `bench_phase1_bridge` ve sonraki olcumler |
| `NEXUS_BUILD_EXAMPLES` | ON | `apps/` ve `docs/examples/` |
| `NEXUS_WARNINGS_AS_ERRORS` | OFF | Kok kutuphane, test, bench ve ornekte `/WX` |
| `NEXUS_ENABLE_ASAN` | OFF | MSVC `/fsanitize=address`; CUDA birimleri haric |

AddressSanitizer acikken MSVC'de `/RTC` bayraklari dusurulur, baglama `/INCREMENTAL:NO /DEBUG` olur. Calistirma:

```bat
set ASAN_OPTIONS=detect_leaks=0:halt_on_error=1
ctest --test-dir build-asan --output-on-failure -C RelWithDebInfo
```

`NEXUS_ENABLE_CUDA` acik olsa bile NexusData'nin kendi CUDA cekirdekleri ayri kapidadir: `NEXUSDATA_WITH_CUDA`. Kok derleme onu zorlamaz.

Alt kutuphaneler `if(NOT TARGET ...)` ile korunur. Kok once yapraklari ekler, `NexusTrain` ayni hedefleri ikinci kez `add_subdirectory` etmez.
