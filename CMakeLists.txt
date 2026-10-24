cmake_minimum_required(VERSION 3.24)

project(NexusFramework
    VERSION 1.0.0
    DESCRIPTION "Birlesik Nexus C++20 AI cercevesi"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(NexusOptions)
include(NexusTargets)

# Yaprak kutuphaneler once gelir. NexusTrain ayni hedefleri tekrar eklemez.
if(NOT TARGET nexusloss)
    add_subdirectory(lib/NexusLoss)
endif()
if(NOT TARGET nexus_optim)
    add_subdirectory(lib/NexusOptim)
endif()
if(NOT TARGET nexusdata)
    add_subdirectory(lib/NexusData)
endif()
if(NOT TARGET nexus_model)
    add_subdirectory(lib/NexusModel)
endif()
if(NOT TARGET matrix_pro)
    set(MATRIX_PRO_ENABLE_CUDA ${NEXUS_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(MATRIX_PRO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MATRIX_PRO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(MATRIX_PRO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MATRIX_PRO_ENABLE_CUDNN OFF CACHE BOOL "" FORCE)
    add_subdirectory(lib/MatrixFlash-Pro)
endif()
if(NOT TARGET nexus_train)
    add_subdirectory(lib/NexusTrain)
endif()

add_library(nexus STATIC
    src/core/allocator.cpp
    src/core/memory.cpp
    src/core/mirrored_tensor.cpp
    src/autograd/engine.cpp
    src/autograd/ops.cpp
    src/fused/fused.cpp
    src/fused/training.cpp
    src/attention/cpu.cpp
    src/lm/config.cpp
    src/lm/rope.cpp
    src/lm/ops.cpp
    src/lm/model.cpp
    src/prof/prof.cpp
    src/model/nxs.cpp)
add_library(nexus::nexus ALIAS nexus)

target_compile_features(nexus PUBLIC cxx_std_20)
target_include_directories(nexus
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(nexus
    PUBLIC
        nexus_train
        nexus_model
        nexus_optim
        nexusloss
        nexusdata)

target_compile_definitions(nexus PUBLIC NEXUS_FRAMEWORK_VERSION_MAJOR=${PROJECT_VERSION_MAJOR})

if(NEXUS_ENABLE_AVX2)
    target_compile_definitions(nexus PUBLIC NEXUS_ENABLE_AVX2=1)
    # FAZ 5.8: CPU FlashAttention doseme donguleri AVX2/FMA ile vektorlesir.
    if(MSVC)
        set_source_files_properties(src/attention/cpu.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/attention/cpu.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()
if(NEXUS_ENABLE_AVX512)
    target_compile_definitions(nexus PUBLIC NEXUS_ENABLE_AVX512=1)
endif()
if(NEXUS_ENABLE_MULTI_GPU)
    target_compile_definitions(nexus PUBLIC NEXUS_ENABLE_MULTI_GPU=1)
endif()
if(NEXUS_ENABLE_TPU_XLA)
    target_compile_definitions(nexus PUBLIC NEXUS_ENABLE_TPU_XLA=1)
endif()
if(NEXUS_ENABLE_ROCM)
    target_compile_definitions(nexus PUBLIC NEXUS_ENABLE_ROCM=1)
endif()

if(TARGET matrix_pro)
    target_link_libraries(nexus PUBLIC matrix_pro)
    target_compile_definitions(nexus PUBLIC NEXUS_HAS_MATRIX_PRO=1)
    if(NEXUS_ENABLE_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(nexus PUBLIC CUDA::cudart)
        target_compile_definitions(nexus PUBLIC NEXUS_ENABLE_CUDA=1)
        # FAZ 1.3/1.4: akis/olay ve onbellekli aygit + pinned ayiricisi.
        target_sources(nexus PRIVATE src/cuda/runtime.cpp src/cuda/caching_allocator.cpp)
    endif()
    message(STATUS "Nexus: nexus::nexus MatrixFlash-Pro ile baglandi (CUDA=${NEXUS_ENABLE_CUDA})")
else()
    message(STATUS "Nexus: nexus::nexus CPU bellek koprusu ile baglandi")
endif()

nexus_configure_target(nexus)

# FAZ 3 GPU yolu: cuBLASLt GEMM + CUDA egitim cekirdekleri + GPU GPT egitmeni.
if(NEXUS_ENABLE_CUDA AND TARGET matrix_pro)
    enable_language(CUDA)
    add_library(nexus_cuda STATIC
        src/cuda/gemm_lt.cpp
        src/cuda/kernels_norm.cu
        src/cuda/kernels_elementwise.cu
        src/cuda/kernels_train.cu
        src/cuda/kernels_fp8.cu
        src/cuda/flash_attention.cu
        src/cuda/kernels_lm.cu
        src/cuda/fp8_native.cpp
        src/cuda/cast_cache.cpp
        src/cuda/gpt.cpp)
    add_library(nexus::cuda ALIAS nexus_cuda)
    # nexus'un kullanim gereksinimleri (or. NexusLoss'un INTERFACE /W4) nvcc'ye gecmesin:
    # yalniz baglama icin nexus, basliklar ve tanimlar acikca verilir.
    target_link_libraries(nexus_cuda PUBLIC $<LINK_ONLY:nexus> CUDA::cudart CUDA::cublasLt)
    target_include_directories(nexus_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/NexusTrain/include)
    target_compile_definitions(nexus_cuda PUBLIC NEXUS_ENABLE_CUDA=1)
    set_target_properties(nexus_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(nexus_cuda PUBLIC NEXUS_HAS_CUDA_KERNELS=1)
    if(MSVC)
        target_compile_options(nexus_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /EHsc /utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/EHsc,/Zc:preprocessor -Xcudafe=--diag_suppress=20012>)
        target_compile_definitions(nexus_cuda PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
endif()

include(CTest)
enable_testing()

add_subdirectory(tools)

if(NEXUS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(NEXUS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
if(NEXUS_BUILD_EXAMPLES OR NEXUS_BUILD_APPS)
    add_subdirectory(apps)
    add_subdirectory(docs/examples)
endif()

option(NEXUS_BUILD_PYTHON "Python kabugunu pybind11 ile derle" OFF)
if(NEXUS_BUILD_PYTHON)
    add_subdirectory(python)
endif()
