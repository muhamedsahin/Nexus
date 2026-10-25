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
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.bench-venv-py314/Scripts/python.exe")
    set(_nexus_gpt_python "${CMAKE_SOURCE_DIR}/.bench-venv-py314/Scripts/python.exe")
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

# FAZ 7 GPU: yeniden hesaplama, parcali CE, ZeRO-Offload, 8-bit AdamW.
add_executable(test_phase7_memory_cuda test_phase7_memory_cuda.cpp)
target_link_libraries(test_phase7_memory_cuda PRIVATE nexus::nexus nexus::cuda nexus_test_support)
nexus_configure_target(test_phase7_memory_cuda)
add_test(NAME test_checkpoint_grad_equal_cuda COMMAND test_phase7_memory_cuda recompute)
add_test(NAME test_chunked_ce_equals_full COMMAND test_phase7_memory_cuda chunked)
add_test(NAME test_zero_offload_equals_gpu_adam COMMAND test_phase7_memory_cuda offload)
add_test(NAME test_8bit_adam_converges COMMAND test_phase7_memory_cuda adam8)
add_test(NAME test_fused_8bit_update COMMAND test_phase7_memory_cuda fused)
set_tests_properties(test_checkpoint_grad_equal_cuda test_chunked_ce_equals_full test_zero_offload_equals_gpu_adam
    test_8bit_adam_converges test_fused_8bit_update PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

# FAZ 8.F: asenkron token yukleyici (pinned halka + kopya akisi).
add_executable(test_phase8_loader test_phase8_loader.cpp)
target_link_libraries(test_phase8_loader PRIVATE nexus::nexus nexus::cuda nexus_test_support)
nexus_configure_target(test_phase8_loader)
add_test(NAME test_loader_order COMMAND test_phase8_loader order)
add_test(NAME test_loader_error COMMAND test_phase8_loader error)
set_tests_properties(test_loader_order test_loader_error PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)

# FAZ 9: dil modeli egitmeni v2, GPU arka ucu (+ uctan uca egitim ve benchmark modlari).
add_executable(test_phase9_trainer_cuda test_phase9_trainer_cuda.cpp)
target_link_libraries(test_phase9_trainer_cuda PRIVATE nexus::nexus nexus::cuda nexus_test_support)
nexus_configure_target(test_phase9_trainer_cuda)
add_test(NAME test_accum_equivalence_cuda COMMAND test_phase9_trainer_cuda accum)
add_test(NAME test_param_groups_wd_cuda COMMAND test_phase9_trainer_cuda groups)
add_test(NAME test_nan_policy_cuda COMMAND test_phase9_trainer_cuda nan)
add_test(NAME test_eval_no_grad_memory COMMAND test_phase9_trainer_cuda eval_mem)
add_test(NAME test_trainer_resume_cuda COMMAND test_phase9_trainer_cuda resume)
set_tests_properties(test_accum_equivalence_cuda test_param_groups_wd_cuda test_nan_policy_cuda
    test_eval_no_grad_memory test_trainer_resume_cuda PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

# FAZ 10: GPU DDP (tek GPU'yu paylasan surecler, tcp ProcessGroup; nexus-run ile).
add_executable(test_phase10_ddp_cuda test_phase10_ddp_cuda.cpp)
target_link_libraries(test_phase10_ddp_cuda PRIVATE nexus::nexus nexus::cuda nexus_test_support)
nexus_configure_target(test_phase10_ddp_cuda)
add_test(NAME test_ddp_equivalence_cuda COMMAND $<TARGET_FILE:nexus_run> --nproc-per-node 2 --master-port 29651
    -- $<TARGET_FILE:test_phase10_ddp_cuda> equiv)
add_test(NAME test_overlap_occurs_cuda COMMAND $<TARGET_FILE:nexus_run> --nproc-per-node 2 --master-port 29652
    -- $<TARGET_FILE:test_phase10_ddp_cuda> overlap)
set_tests_properties(test_ddp_equivalence_cuda test_overlap_occurs_cuda PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)
