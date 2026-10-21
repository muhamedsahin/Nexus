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
    src/core/memory.cpp
    src/core/mirrored_tensor.cpp
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
    endif()
    message(STATUS "Nexus: nexus::nexus MatrixFlash-Pro ile baglandi (CUDA=${NEXUS_ENABLE_CUDA})")
else()
    message(STATUS "Nexus: nexus::nexus CPU bellek koprusu ile baglandi")
endif()

nexus_configure_target(nexus)

include(CTest)
enable_testing()

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
