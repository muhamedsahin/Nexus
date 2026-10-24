# Nexus/tests/parity: FAZ 0 olcum altyapisi, FAZ 1 k64 ayirici ve tahmin araci parite testleri.
add_executable(test_phase0_prof test_phase0_prof.cpp)
target_link_libraries(test_phase0_prof PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase0_prof)
add_test(NAME test_phase0_prof COMMAND test_phase0_prof)
set_tests_properties(test_phase0_prof PROPERTIES TIMEOUT 180)

add_executable(test_estimate_parity test_estimate_parity.cpp)
target_link_libraries(test_estimate_parity PRIVATE nexus_estimate_core nexus_test_support)
target_compile_definitions(test_estimate_parity PRIVATE
    NEXUS_ESTIMATE_EXE="$<TARGET_FILE:nexus_estimate>"
    NEXUS_ESTIMATE_PY="${CMAKE_SOURCE_DIR}/benchmarks/runner/estimate.py")
nexus_configure_target(test_estimate_parity)
add_test(NAME test_estimate_parity COMMAND test_estimate_parity)
set_tests_properties(test_estimate_parity PROPERTIES TIMEOUT 180)

add_executable(test_phase1_allocator test_phase1_allocator.cpp)
target_link_libraries(test_phase1_allocator PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase1_allocator)
add_test(NAME test_phase1_allocator COMMAND test_phase1_allocator)
set_tests_properties(test_phase1_allocator PROPERTIES TIMEOUT 180)

add_executable(test_phase2_autograd test_phase2_autograd.cpp)
target_link_libraries(test_phase2_autograd PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase2_autograd)
add_test(NAME test_phase2_autograd COMMAND test_phase2_autograd)
set_tests_properties(test_phase2_autograd PROPERTIES TIMEOUT 180)

add_executable(test_phase3_fused test_phase3_fused.cpp)
target_link_libraries(test_phase3_fused PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase3_fused)
add_test(NAME test_phase3_fused COMMAND test_phase3_fused)
set_tests_properties(test_phase3_fused PROPERTIES TIMEOUT 180)

add_executable(test_phase2_engine test_phase2_engine.cpp)
target_link_libraries(test_phase2_engine PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase2_engine)
add_test(NAME test_phase2_engine COMMAND test_phase2_engine)
set_tests_properties(test_phase2_engine PROPERTIES TIMEOUT 180)

add_executable(test_phase1_dtype test_phase1_dtype.cpp)
target_link_libraries(test_phase1_dtype PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase1_dtype)
add_test(NAME test_phase1_dtype COMMAND test_phase1_dtype)
set_tests_properties(test_phase1_dtype PROPERTIES TIMEOUT 300)

add_executable(test_phase3_training test_phase3_training.cpp)
target_link_libraries(test_phase3_training PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase3_training)
add_test(NAME test_phase3_training COMMAND test_phase3_training)
set_tests_properties(test_phase3_training PROPERTIES TIMEOUT 300)

# FAZ 0.E parity: Nexus vs PyTorch (FP32). PyTorch yoksa test SKIPPED (77) olur.
set(NEXUS_PARITY_PYTHON "" CACHE FILEPATH "PyTorch kurulu python (bos: .bench-venv denenir)")
if(NOT NEXUS_PARITY_PYTHON)
    if(EXISTS "${CMAKE_SOURCE_DIR}/.bench-venv/Scripts/python.exe")
        set(_nexus_parity_python "${CMAKE_SOURCE_DIR}/.bench-venv/Scripts/python.exe")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/.bench-venv/bin/python")
        set(_nexus_parity_python "${CMAKE_SOURCE_DIR}/.bench-venv/bin/python")
    else()
        set(_nexus_parity_python "")
    endif()
else()
    set(_nexus_parity_python "${NEXUS_PARITY_PYTHON}")
endif()
add_executable(test_parity_nano test_parity_nano.cpp)
target_link_libraries(test_parity_nano PRIVATE nexus::nexus nexus_estimate_core)
target_compile_definitions(test_parity_nano PRIVATE
    NEXUS_PARITY_PYTHON="${_nexus_parity_python}"
    NEXUS_PARITY_SCRIPTS="${CMAKE_CURRENT_SOURCE_DIR}")
nexus_configure_target(test_parity_nano)
add_test(NAME test_parity_nano COMMAND test_parity_nano "${CMAKE_SOURCE_DIR}/configs/nano.json")
set_tests_properties(test_parity_nano PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
# tiny: tam sekil (L=6, d=384, V=8192); B=1, T=64 ile (otograd CPU naif GEMM'de sure sinirli).
add_test(NAME test_parity_tiny COMMAND test_parity_nano "${CMAKE_SOURCE_DIR}/configs/tiny.json" 1 64)
set_tests_properties(test_parity_tiny PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)
