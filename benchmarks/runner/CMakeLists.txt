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

# FAZ 5.G: CPU FlashAttention (FP32); PyTorch CPU SDPA karsiligi bench_sdpa_torch_cpu.py.
add_executable(bench_attention_cpu bench_attention_cpu.cpp)
target_link_libraries(bench_attention_cpu PRIVATE nexus::nexus)
target_compile_definitions(bench_attention_cpu PRIVATE
    NEXUS_FAZ5_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-24-faz5")
nexus_configure_target(bench_attention_cpu)
add_test(NAME bench_attention_cpu COMMAND bench_attention_cpu)
set_tests_properties(bench_attention_cpu PROPERTIES TIMEOUT 600)

# FAZ 1 + FAZ 3 GPU (ayirici, eleman eleman, cuBLASLt Linear, norm, CE, AdamW, GPT-2 124M).
if(TARGET nexus_cuda)
    add_executable(bench_cuda_phase20 bench_cuda_phase20.cpp)
    target_link_libraries(bench_cuda_phase20 PRIVATE nexus::nexus nexus::cuda)
    nexus_configure_target(bench_cuda_phase20)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME test_phase20_resume_short
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/parity/test_phase20_resume.py"
                --exe $<TARGET_FILE:bench_cuda_phase20> --work "${CMAKE_CURRENT_BINARY_DIR}/phase20-tests")
        set_tests_properties(test_phase20_resume_short PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()

    add_executable(bench_cuda_phase3 bench_cuda_phase3.cpp)
    target_link_libraries(bench_cuda_phase3 PRIVATE nexus::nexus nexus::cuda CUDA::cublas)
    target_compile_definitions(bench_cuda_phase3 PRIVATE
        NEXUS_FAZ3_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-24-faz1-3-gpu"
        NEXUS_GIT_COMMIT="${NEXUS_GIT_COMMIT}")
    nexus_configure_target(bench_cuda_phase3)
    add_test(NAME bench_cuda_phase3 COMMAND bench_cuda_phase3)
    set_tests_properties(bench_cuda_phase3 PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)
    add_executable(bench_cuda_phase4 bench_cuda_phase4.cpp)
    target_link_libraries(bench_cuda_phase4 PRIVATE nexus::nexus nexus::cuda)
    target_compile_definitions(bench_cuda_phase4 PRIVATE
        NEXUS_FAZ4_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-24-faz4"
        NEXUS_GIT_COMMIT="${NEXUS_GIT_COMMIT}")
    nexus_configure_target(bench_cuda_phase4)
    add_test(NAME bench_cuda_phase4 COMMAND bench_cuda_phase4)
    set_tests_properties(bench_cuda_phase4 PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)

    add_executable(bench_cuda_phase5 bench_cuda_phase5.cpp)
    target_link_libraries(bench_cuda_phase5 PRIVATE nexus::nexus nexus::cuda)
    target_compile_definitions(bench_cuda_phase5 PRIVATE
        NEXUS_FAZ5_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-24-faz5")
    nexus_configure_target(bench_cuda_phase5)
    add_test(NAME bench_cuda_phase5 COMMAND bench_cuda_phase5)
    set_tests_properties(bench_cuda_phase5 PROPERTIES TIMEOUT 3600 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

    add_executable(bench_cuda_phase6 bench_cuda_phase6.cpp)
    target_link_libraries(bench_cuda_phase6 PRIVATE nexus::nexus nexus::cuda)
    target_compile_definitions(bench_cuda_phase6 PRIVATE
        NEXUS_FAZ6_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-24-faz6")
    nexus_configure_target(bench_cuda_phase6)
    add_test(NAME bench_cuda_phase6 COMMAND bench_cuda_phase6)
    set_tests_properties(bench_cuda_phase6 PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

    add_executable(bench_cuda_phase7 bench_cuda_phase7.cpp)
    target_link_libraries(bench_cuda_phase7 PRIVATE nexus::nexus nexus::cuda nexus_estimate_core)
    target_include_directories(bench_cuda_phase7 PRIVATE ${CMAKE_SOURCE_DIR}/tools/nexus_estimate)
    target_compile_definitions(bench_cuda_phase7 PRIVATE
        NEXUS_FAZ7_DOCS_DIR="${CMAKE_SOURCE_DIR}/docs/benchmarks/2026-10-24-faz7")
    nexus_configure_target(bench_cuda_phase7)
    add_test(NAME bench_cuda_phase7 COMMAND bench_cuda_phase7)
    set_tests_properties(bench_cuda_phase7 PROPERTIES TIMEOUT 7200 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

endif()
