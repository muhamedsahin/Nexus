cmake_minimum_required(VERSION 3.20)

# Ayni yapida ikinci kez add_subdirectory edilirse hedefler yeniden tanimlanmaz.
if(TARGET nexus_train)
    return()
endif()

project(NexusTrain
  VERSION 1.0.0
  DESCRIPTION "Training-loop orchestration for the Nexus stack"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NEXUS_TRAIN_BUILD_TESTS "Build NexusTrain tests" ON)
option(NEXUS_TRAIN_BUILD_BENCHMARKS "Build NexusTrain benchmarks" ON)
option(NEXUS_TRAIN_BUILD_EXAMPLES "Build NexusTrain examples" ON)
option(NEXUS_TRAIN_WITH_DISTRIBUTED "Enable data-parallel step helpers" ON)
option(NEXUS_TRAIN_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

set(NEXUS_MODEL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUS_MODEL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(NEXUS_MODEL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NEXUS_MODEL_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(NEXUSLOSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUSLOSS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NEXUS_OPTIM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUS_OPTIM_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(NEXUS_OPTIM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NEXUS_OPTIM_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
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

# Kok CMake ayni hedefleri daha once eklediyse ikinci add_subdirectory CMake hatasidir.
if(NOT TARGET nexusloss)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../NexusLoss ${CMAKE_BINARY_DIR}/_deps/nexusloss)
endif()
if(NOT TARGET nexus_optim)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../NexusOptim ${CMAKE_BINARY_DIR}/_deps/nexusoptim)
endif()
if(NOT TARGET nexusdata)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../NexusData ${CMAKE_BINARY_DIR}/_deps/nexusdata)
endif()
if(NOT TARGET nexus_model)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../NexusModel ${CMAKE_BINARY_DIR}/_deps/nexusmodel)
endif()

find_package(Threads REQUIRED)

add_library(nexus_train STATIC
  src/core/trainer.cpp
  src/checkpoint/checkpoint_manager.cpp
  src/prefetch/async_prefetcher.cpp)
add_library(nexus_train::nexus_train ALIAS nexus_train)

target_include_directories(nexus_train PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(nexus_train PUBLIC cxx_std_20)
target_link_libraries(nexus_train PUBLIC nexus_model nexusloss nexus_optim nexusdata Threads::Threads)

if(NEXUS_TRAIN_WITH_DISTRIBUTED)
  target_compile_definitions(nexus_train PUBLIC NEXUS_TRAIN_WITH_DISTRIBUTED=1)
endif()

if(MSVC)
  target_compile_definitions(nexus_train PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(nexus_train PRIVATE /W4 /permissive- /EHsc /utf-8)
  if(NEXUS_TRAIN_WARNINGS_AS_ERRORS)
    target_compile_options(nexus_train PRIVATE /WX)
  endif()
  target_compile_options(nexus_train PRIVATE $<$<CONFIG:Release>:/O2>)
else()
  target_compile_options(nexus_train PRIVATE -Wall -Wextra -Wpedantic)
  if(NEXUS_TRAIN_WARNINGS_AS_ERRORS)
    target_compile_options(nexus_train PRIVATE -Werror)
  endif()
  target_compile_options(nexus_train PRIVATE $<$<CONFIG:Release>:-O3;-march=native>)
endif()

function(nexus_train_flags target)
  if(MSVC)
    target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(${target} PRIVATE /W4 /permissive- /EHsc /utf-8)
    if(NEXUS_TRAIN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(NEXUS_TRAIN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

include(CTest)

if(NEXUS_TRAIN_BUILD_TESTS)
  add_executable(nexus_train_tests
    tests/test_main.cpp
    tests/test_trainer_basic_loop.cpp
    tests/test_callbacks.cpp
    tests/test_checkpoint.cpp
    tests/test_metrics.cpp
    tests/test_gradient_accumulation.cpp
    tests/test_early_stopping.cpp
    tests/test_prefetcher.cpp
    tests/test_mixed_precision.cpp
    tests/test_reproducibility.cpp
    tests/test_distributed.cpp)
  target_link_libraries(nexus_train_tests PRIVATE nexus_train)
  nexus_train_flags(nexus_train_tests)
  add_test(NAME nexus_train_tests COMMAND nexus_train_tests)
endif()

if(NEXUS_TRAIN_BUILD_BENCHMARKS)
  add_executable(bench_training_loop_overhead benchmarks/bench_training_loop_overhead.cpp)
  target_link_libraries(bench_training_loop_overhead PRIVATE nexus_train)
  nexus_train_flags(bench_training_loop_overhead)

  add_executable(bench_prefetch_vs_sync benchmarks/bench_prefetch_vs_sync.cpp)
  target_link_libraries(bench_prefetch_vs_sync PRIVATE nexus_train)
  nexus_train_flags(bench_prefetch_vs_sync)
endif()

if(NEXUS_TRAIN_BUILD_EXAMPLES)
  add_executable(train_mnist_full_example examples/train_mnist_full_example.cpp)
  target_link_libraries(train_mnist_full_example PRIVATE nexus_train)
  nexus_train_flags(train_mnist_full_example)

  add_executable(train_with_custom_callback_example examples/train_with_custom_callback_example.cpp)
  target_link_libraries(train_with_custom_callback_example PRIVATE nexus_train)
  nexus_train_flags(train_with_custom_callback_example)
endif()

install(TARGETS nexus_train ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
