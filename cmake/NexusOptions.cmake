# Nexus monorepo secenekleri.
#
# Bu dosya kok CMakeLists.txt tarafindan, herhangi bir add_subdirectory
# cagrisindan once okunur. Alt kutuphanelerin test/ornek hedefleri burada
# kapatilir; aksi halde kok derleme her alt projenin kendi testini de uretir
# ve ayni hedef adlari cakisir.
#
# Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
# 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
# olarak gelistirilmistir.

include_guard(GLOBAL)

option(NEXUS_ENABLE_CUDA "MatrixFlash-Pro ve CUDA bellek koprusunu derle (toolkit yoksa otomatik kapanir)" ON)
option(NEXUS_ENABLE_AVX2 "Alt kutuphanelerde AVX2 cekirdeklerini ac" ON)
option(NEXUS_ENABLE_AVX512 "Alt kutuphanelerde AVX-512 cekirdeklerini ac" ON)
option(NEXUS_ENABLE_MULTI_GPU "Coklu GPU kolektif bayragini derleme tanimina yaz (motor Faz 2)" OFF)
option(NEXUS_ENABLE_TPU_XLA "TPU/OpenXLA cihaz etiketini derleme tanimina yaz (calistirma motoru Faz 2)" OFF)
option(NEXUS_ENABLE_ROCM "ROCm cihaz etiketini derleme tanimina yaz (calistirma motoru Faz 2)" OFF)
option(NEXUS_BUILD_TESTS "Birlesik Nexus testlerini derle" ON)
option(NEXUS_BUILD_BENCHMARKS "Birlesik Nexus benchmarklarini derle" ON)
option(NEXUS_BUILD_EXAMPLES "Birlesik ornekleri ve demo uygulamasini derle" ON)
option(NEXUS_WARNINGS_AS_ERRORS "Kok Nexus hedeflerinde (kutuphane, test, bench, ornek) uyarilari hata say" OFF)
option(NEXUS_ENABLE_ASAN "Tum C++ ceviri birimlerini AddressSanitizer ile derle" OFF)

# --- AddressSanitizer -------------------------------------------------------
# add_subdirectory'lerden once kurulur; boylece alt kutuphaneler de ayni
# enstrumantasyonla derlenir. Yalniz bir kutuphanenin ASan'li olmasi
# std::vector aciklama (annotation) uyumsuzlugu yuzunden baglanmaz.
# CUDA ceviri birimleri nvcc ile derlenir ve enstrumante edilmez.
if(NEXUS_ENABLE_ASAN)
    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address> $<$<COMPILE_LANGUAGE:CXX>:/Zi>)
        add_link_options(/INCREMENTAL:NO /DEBUG)
        # /RTC calisma zamani denetimleri MSVC ASan ile birlikte kullanilamaz.
        foreach(flag_var CMAKE_CXX_FLAGS_DEBUG CMAKE_C_FLAGS_DEBUG)
            string(REGEX REPLACE "/RTC[1csu]+" "" ${flag_var} "${${flag_var}}")
        endforeach()
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address> $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
        add_link_options(-fsanitize=address)
    endif()
    message(STATUS "Nexus: AddressSanitizer acik")
endif()

# Eski kok secenegi. Yeni ad NEXUS_BUILD_EXAMPLES. Ikisi de ayni kapidir.
if(NOT DEFINED NEXUS_BUILD_APPS)
    set(NEXUS_BUILD_APPS ${NEXUS_BUILD_EXAMPLES} CACHE BOOL "NEXUS_BUILD_EXAMPLES ile ayni" FORCE)
endif()

# --- CUDA toolkit tespiti ---------------------------------------------------
# nvcc yoksa tum monorepo CPU olarak derlenir. MatrixFlash-Pro su an CUDA
# ceviri birimleri tasidigi icin toolkit olmadan hedefe eklenmez.
set(NEXUS_CUDA_AVAILABLE OFF)
if(NEXUS_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA sanal mimarileri" FORCE)
    endif()
    find_program(NEXUS_NVCC_EXECUTABLE nvcc
        HINTS "$ENV{CUDA_PATH}/bin" "$ENV{CUDA_PATH}/bin/x64"
        DOC "NVIDIA CUDA derleyicisi")
    if(NEXUS_NVCC_EXECUTABLE)
        set(NEXUS_CUDA_AVAILABLE ON)
        message(STATUS "Nexus: CUDA toolkit bulundu (${NEXUS_NVCC_EXECUTABLE})")
    else()
        message(STATUS "Nexus: CUDA toolkit bulunamadi. NEXUS_ENABLE_CUDA kapatildi.")
        set(NEXUS_ENABLE_CUDA OFF CACHE BOOL "MatrixFlash-Pro ve CUDA bellek koprusunu derle" FORCE)
    endif()
endif()

# --- Alt projelerin kendi test/ornek/bench hedeflerini kokte kapat ----------
set(NEXUSLOSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUSLOSS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(NEXUS_OPTIM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUS_OPTIM_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(NEXUS_OPTIM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NEXUS_OPTIM_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
if(NEXUS_ENABLE_CUDA)
    set(NEXUS_OPTIM_WITH_CUDA ON CACHE BOOL "" FORCE)
else()
    set(NEXUS_OPTIM_WITH_CUDA OFF CACHE BOOL "" FORCE)
endif()
if(NEXUS_ENABLE_AVX512)
    set(NEXUS_OPTIM_ENABLE_AVX512 ON CACHE BOOL "" FORCE)
else()
    set(NEXUS_OPTIM_ENABLE_AVX512 OFF CACHE BOOL "" FORCE)
endif()

set(NEXUS_MODEL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUS_MODEL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(NEXUS_MODEL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NEXUS_MODEL_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
if(NEXUS_ENABLE_AVX512)
    set(NEXUS_MODEL_ENABLE_AVX512 ON CACHE BOOL "" FORCE)
else()
    set(NEXUS_MODEL_ENABLE_AVX512 OFF CACHE BOOL "" FORCE)
endif()

set(NEXUSDATA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_BUILD_FUZZ OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_FETCH_DEPS OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_WITH_WEBP OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_WITH_ZSTD OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_WITH_LZ4 OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_WITH_BZIP2 OFF CACHE BOOL "" FORCE)
set(NEXUSDATA_WITH_LZMA OFF CACHE BOOL "" FORCE)
if(NEXUS_ENABLE_AVX2)
    set(NEXUSDATA_ENABLE_AVX2 ON CACHE BOOL "" FORCE)
else()
    set(NEXUSDATA_ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
endif()
# NexusData'nin kendi CUDA cekirdekleri ayri bir kapidir (NEXUSDATA_WITH_CUDA).
# Kokteki NEXUS_ENABLE_CUDA MatrixFlash-Pro'yu baglar. Iki kapı birbirini
# zorlamaz; aksi halde kok derleme iki ayri CUDA agaci uretir.

set(NEXUS_TRAIN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUS_TRAIN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(NEXUS_TRAIN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NEXUS_TRAIN_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

set(MATRIX_PRO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MATRIX_PRO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MATRIX_PRO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MATRIX_PRO_ENABLE_CUDNN OFF CACHE BOOL "" FORCE)
