# Nexus/benchmarks/runner: FAZ 0 olcum altyapisi ve FAZ 1 k64 ayirici benchmarklari.
# Sonuclar docs/benchmarks/<faz>/bench_phase*.json'a makinayla okunur yazilir.
add_executable(bench_phase0 bench_phase0.cpp)
target_link_libraries(bench_phase0 PRIVATE nexus::nexus)
target_compile_definitions(bench_phase0 PRIVATE
    NEXUS_FAZ0_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-23-faz0")
nexus_configure_target(bench_phase0)
add_test(NAME bench_phase0 COMMAND bench_phase0)
set_tests_properties(bench_phase0 PROPERTIES TIMEOUT 300)

add_executable(bench_phase1 bench_phase1.cpp)
target_link_libraries(bench_phase1 PRIVATE nexus::nexus)
target_compile_definitions(bench_phase1 PRIVATE
    NEXUS_FAZ1_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-23-faz1")
nexus_configure_target(bench_phase1)
add_test(NAME bench_phase1 COMMAND bench_phase1)
set_tests_properties(bench_phase1 PROPERTIES TIMEOUT 300)

add_executable(bench_phase2 bench_phase2.cpp)
target_link_libraries(bench_phase2 PRIVATE nexus::nexus)
target_compile_definitions(bench_phase2 PRIVATE
    NEXUS_FAZ2_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-23-faz2")
nexus_configure_target(bench_phase2)
add_test(NAME bench_phase2 COMMAND bench_phase2)
set_tests_properties(bench_phase2 PROPERTIES TIMEOUT 300)

add_executable(bench_phase3_fused bench_phase3_fused.cpp)
target_link_libraries(bench_phase3_fused PRIVATE nexus::nexus)
target_compile_definitions(bench_phase3_fused PRIVATE
    NEXUS_FAZ3_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-23-faz3")
nexus_configure_target(bench_phase3_fused)
add_test(NAME bench_phase3_fused COMMAND bench_phase3_fused)
set_tests_properties(bench_phase3_fused PROPERTIES TIMEOUT 300)

# FAZ 3 egitim yolu (CPU): kaydirilmis CE, foreach AdamW 124M, RMSNorm/LayerNorm. BOLUM G protokolu.
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE NEXUS_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT NEXUS_GIT_COMMIT)
    set(NEXUS_GIT_COMMIT "unknown")
endif()
add_executable(bench_phase3_training bench_phase3_training.cpp)
target_link_libraries(bench_phase3_training PRIVATE nexus::nexus)
target_compile_definitions(bench_phase3_training PRIVATE
    NEXUS_FAZ3_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-23-faz3"
    NEXUS_GIT_COMMIT="${NEXUS_GIT_COMMIT}")
nexus_configure_target(bench_phase3_training)
add_test(NAME bench_phase3_training COMMAND bench_phase3_training)
set_tests_properties(bench_phase3_training PROPERTIES TIMEOUT 600)

# FAZ 1 + FAZ 3 GPU (ayirici, eleman eleman, cuBLASLt Linear, norm, CE, AdamW, GPT-2 124M).
if(TARGET nexus_cuda)
    add_executable(bench_cuda_phase3 bench_cuda_phase3.cpp)
    target_link_libraries(bench_cuda_phase3 PRIVATE nexus::nexus nexus::cuda CUDA::cublas)
    target_compile_definitions(bench_cuda_phase3 PRIVATE
        NEXUS_FAZ3_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-24-faz1-3-gpu"
        NEXUS_GIT_COMMIT="${NEXUS_GIT_COMMIT}")
    nexus_configure_target(bench_cuda_phase3)
    add_test(NAME bench_cuda_phase3 COMMAND bench_cuda_phase3)
    set_tests_properties(bench_cuda_phase3 PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)
endif()
