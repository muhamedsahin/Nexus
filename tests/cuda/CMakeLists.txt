# Nexus/tests/cuda: FAZ 1 (akis, onbellekli ayirici) ve FAZ 3 (cuBLASLt + CUDA cekirdekleri,
# GPU GPT) testleri. Yalniz NEXUS_ENABLE_CUDA ile derlenir; aygit yoksa 77 = SKIPPED.
function(nexus_add_cuda_test name)
    cmake_parse_arguments(ARG "" "TIMEOUT" "" ${ARGN})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE nexus::nexus nexus::cuda nexus_test_support)
    nexus_configure_target(${name})
    add_test(NAME ${name} COMMAND ${name})
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 300)
    endif()
    set_tests_properties(${name} PROPERTIES TIMEOUT ${ARG_TIMEOUT} SKIP_RETURN_CODE 77)
endfunction()

nexus_add_cuda_test(test_cuda_allocator)
nexus_add_cuda_test(test_cuda_kernels TIMEOUT 600)

add_executable(test_phase4_cuda_kernels test_phase4_cuda_kernels.cpp)
target_link_libraries(test_phase4_cuda_kernels PRIVATE nexus::nexus nexus::cuda nexus_test_support)
nexus_configure_target(test_phase4_cuda_kernels)
add_test(NAME test_phase4_narrow_gemm COMMAND test_phase4_cuda_kernels gemm)
add_test(NAME test_phase4_cast_cache COMMAND test_phase4_cuda_kernels cache)
add_test(NAME test_fp8_emulation COMMAND test_phase4_cuda_kernels fp8)
set_tests_properties(test_phase4_narrow_gemm test_phase4_cast_cache test_fp8_emulation
    PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)


# FAZ 3.I: GPU GPT-2 vs PyTorch (FP32). PyTorch'lu python: NEXUS_PARITY_PYTHON veya .bench-venv.
add_executable(test_phase4_cuda_training test_phase4_cuda_training.cpp)
target_link_libraries(test_phase4_cuda_training PRIVATE nexus::nexus nexus::cuda nexus_test_support)
nexus_configure_target(test_phase4_cuda_training)
add_test(NAME test_fp16_loss_scale COMMAND test_phase4_cuda_training loss)
add_test(NAME test_master_weight_modes COMMAND test_phase4_cuda_training master)
add_test(NAME test_bf16_training_converges COMMAND test_phase4_cuda_training converge)
add_test(NAME test_gpt_flash_vs_math COMMAND test_phase4_cuda_training flash)
set_tests_properties(test_fp16_loss_scale PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
set_tests_properties(test_master_weight_modes test_bf16_training_converges test_gpt_flash_vs_math
    PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)


if(NEXUS_PARITY_PYTHON)
    set(_nexus_gpt_python "${NEXUS_PARITY_PYTHON}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.bench-venv/Scripts/python.exe")
    set(_nexus_gpt_python "${CMAKE_SOURCE_DIR}/.bench-venv/Scripts/python.exe")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.bench-venv/bin/python")
    set(_nexus_gpt_python "${CMAKE_SOURCE_DIR}/.bench-venv/bin/python")
else()
    set(_nexus_gpt_python "")
endif()
add_executable(test_cuda_gpt test_cuda_gpt.cpp)
target_link_libraries(test_cuda_gpt PRIVATE nexus::nexus nexus::cuda)
target_compile_definitions(test_cuda_gpt PRIVATE
    NEXUS_PARITY_PYTHON="${_nexus_gpt_python}"
    NEXUS_GPT_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/reference_gpt2_torch.py")
nexus_configure_target(test_cuda_gpt)
add_test(NAME test_cuda_gpt_small COMMAND test_cuda_gpt small 10)
set_tests_properties(test_cuda_gpt_small PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
add_test(NAME test_gpt2s_parity_fp32 COMMAND test_cuda_gpt gpt2s 10)
set_tests_properties(test_gpt2s_parity_fp32 PROPERTIES TIMEOUT 3600 SKIP_RETURN_CODE 77)

# Agir GPU egitim testleri paralel kosuda (WDDM) birbirini dakikalarca yavaslatir: tek basina kos.
set_tests_properties(test_master_weight_modes test_bf16_training_converges test_cuda_gpt_small
    test_gpt2s_parity_fp32 test_gpt_flash_vs_math PROPERTIES RUN_SERIAL TRUE)

# FAZ 5: FlashAttention-2 (CUDA) ve CPU flash, FP64 math kahinine karsi.
add_executable(test_flash_attention test_flash_attention.cpp)
target_link_libraries(test_flash_attention PRIVATE nexus::nexus nexus::cuda nexus_test_support)
nexus_configure_target(test_flash_attention)
foreach(_mode grid decode varlen dropout determinism future)
    set(_name test_flash_${_mode})
    if(_mode STREQUAL "grid")
        set(_name test_flash_vs_math)
    elseif(_mode STREQUAL "decode")
        set(_name test_flash_tq1_decode)
    elseif(_mode STREQUAL "varlen")
        set(_name test_flash_varlen_equals_separate)
    elseif(_mode STREQUAL "dropout")
        set(_name test_flash_dropout_stats)
    elseif(_mode STREQUAL "future")
        set(_name test_causal_future_invariance)
    endif()
    add_test(NAME ${_name} COMMAND test_flash_attention ${_mode})
    set_tests_properties(${_name} PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)
endforeach()

# FAZ 6: RoPE ve SwiGLU/GeGLU GPU cekirdekleri (CPU nexus::lm referansina karsi).
add_executable(test_lm_kernels test_lm_kernels.cpp)
target_link_libraries(test_lm_kernels PRIVATE nexus::nexus nexus::cuda nexus_test_support)
nexus_configure_target(test_lm_kernels)
add_test(NAME test_rope_swiglu_cuda COMMAND test_lm_kernels)
set_tests_properties(test_rope_swiglu_cuda PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
