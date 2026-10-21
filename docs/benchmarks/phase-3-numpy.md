# Faz 3 — NumPy

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Yalnız GEMM ölçüldü. Conv, dikkat, LSTM, GRU, MSE ve Adam bu fazda yok.

## İş parçacığı: 8

BLAS (`numpy.show_config`):

```
{
  "Compilers": {
    "c": {
      "name": "msvc",
      "linker": "link",
      "version": "19.44.35226",
      "commands": "cl"
    },
    "cython": {
      "name": "cython",
      "linker": "cython",
      "version": "3.2.4",
      "commands": "cython"
    },
    "c++": {
      "name": "msvc",
      "linker": "link",
      "version": "19.44.35226",
      "commands": "cl"
    }
  },
  "Machine Information": {
    "host": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "windows"
    },
    "build": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "windows"
    }
  },
  "Build Dependencies": {
    "blas": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.31.188.0",
      "detection method": "pkgconfig",
      "include directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/include",
      "lib directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24",
      "pc file directory": "D:/a/numpy-release/numpy-release/.openblas"
    },
    "lapack": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.31.188.0",
      "detection method": "pkgconfig",
      "include directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/include",
      "lib directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24",
      "pc file directory": "D:/a/numpy-release/numpy-release/.openblas"
    }
  },
  "Python Information": {
    "path": "C:\\Users\\runneradmin\\AppData\\Local\\Temp\\build-env-jxn7bo0g\\Scripts\\python.exe",
    "version": "3.11"
  },
  "SIMD Extensions": {
    "baseline": [
      "X86_V2"
    ],
    "found": [
      "X86_V3"
    ]
  }
}

```

| iş | saniye | değer | birim |
| --- | --- | --- | --- |
| gemm_n256 | 0.000328199996 | 102.237759 | GFLOPS |
| gemm_n512 | 0.0015757 | 170.359495 | GFLOPS |
| gemm_n1024 | 0.0083874 | 256.03687 | GFLOPS |
| gemm_n2048 | 0.0776351333 | 221.289878 | GFLOPS |

## İş parçacığı: default

BLAS (`numpy.show_config`):

```
{
  "Compilers": {
    "c": {
      "name": "msvc",
      "linker": "link",
      "version": "19.44.35226",
      "commands": "cl"
    },
    "cython": {
      "name": "cython",
      "linker": "cython",
      "version": "3.2.4",
      "commands": "cython"
    },
    "c++": {
      "name": "msvc",
      "linker": "link",
      "version": "19.44.35226",
      "commands": "cl"
    }
  },
  "Machine Information": {
    "host": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "windows"
    },
    "build": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "windows"
    }
  },
  "Build Dependencies": {
    "blas": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.31.188.0",
      "detection method": "pkgconfig",
      "include directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/include",
      "lib directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24",
      "pc file directory": "D:/a/numpy-release/numpy-release/.openblas"
    },
    "lapack": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.31.188.0",
      "detection method": "pkgconfig",
      "include directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/include",
      "lib directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24",
      "pc file directory": "D:/a/numpy-release/numpy-release/.openblas"
    }
  },
  "Python Information": {
    "path": "C:\\Users\\runneradmin\\AppData\\Local\\Temp\\build-env-jxn7bo0g\\Scripts\\python.exe",
    "version": "3.11"
  },
  "SIMD Extensions": {
    "baseline": [
      "X86_V2"
    ],
    "found": [
      "X86_V3"
    ]
  }
}

```

| iş | saniye | değer | birim |
| --- | --- | --- | --- |
| gemm_n256 | 0.000341333332 | 98.3040002 | GFLOPS |
| gemm_n512 | 0.00167753334 | 160.017956 | GFLOPS |
| gemm_n1024 | 0.00813113333 | 264.106313 | GFLOPS |
| gemm_n2048 | 0.0567958333 | 302.484675 | GFLOPS |
