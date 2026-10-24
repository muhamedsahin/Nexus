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

# FAZ 3.I: GPU GPT-2 vs PyTorch (FP32). PyTorch'lu python: NEXUS_PARITY_PYTHON veya .bench-venv.
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
