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

add_executable(test_phase4_mixed_precision test_phase4_mixed_precision.cpp)
target_link_libraries(test_phase4_mixed_precision PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase4_mixed_precision)
add_test(NAME test_autocast_policy COMMAND test_phase4_mixed_precision autocast)
add_test(NAME test_phase4_host_fp8 COMMAND test_phase4_mixed_precision fp8)
add_test(NAME test_phase4_loss_scale_policy COMMAND test_phase4_mixed_precision loss)
set_tests_properties(test_autocast_policy test_phase4_host_fp8 test_phase4_loss_scale_policy PROPERTIES TIMEOUT 180)


# FAZ 0.E parity: Nexus vs PyTorch (FP32). PyTorch yoksa test SKIPPED (77) olur.
set(NEXUS_PARITY_PYTHON "" CACHE FILEPATH "PyTorch kurulu python (bos: .bench-venv denenir)")
if(NOT NEXUS_PARITY_PYTHON)
    # .bench-venv-py314 once: eski .bench-venv'in tabani (Anaconda 3.11) 2026-10-25'te kaldirildi.
    if(EXISTS "${CMAKE_SOURCE_DIR}/.bench-venv-py314/Scripts/python.exe")
        set(_nexus_parity_python "${CMAKE_SOURCE_DIR}/.bench-venv-py314/Scripts/python.exe")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/.bench-venv/Scripts/python.exe")
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

# FAZ 6: nexus::lm (RoPE + olcekleme, SwiGLU, GQA, DecoderBlock, kurucu, NXS) vs PyTorch (HF formulleri).
add_executable(test_phase6_lm test_phase6_lm.cpp)
target_link_libraries(test_phase6_lm PRIVATE nexus::nexus nexus_test_support)
target_compile_definitions(test_phase6_lm PRIVATE
    NEXUS_PARITY_PYTHON="${_nexus_parity_python}"
    NEXUS_PARITY_SCRIPTS="${CMAKE_CURRENT_SOURCE_DIR}")
nexus_configure_target(test_phase6_lm)
add_test(NAME test_rope_vs_reference COMMAND test_phase6_lm rope)
add_test(NAME test_swiglu_gradcheck COMMAND test_phase6_lm gradcheck)
add_test(NAME test_decoder_block_parity COMMAND test_phase6_lm parity)
add_test(NAME test_tied_embeddings_single_copy COMMAND test_phase6_lm tied)
add_test(NAME test_build_param_counts COMMAND test_phase6_lm counts)
add_test(NAME test_lm_nxs_roundtrip COMMAND test_phase6_lm nxs)
add_test(NAME test_lm_mup_multipliers COMMAND test_phase6_lm mup)
add_test(NAME test_deep_stack_stability COMMAND test_phase6_lm deep 500)
set_tests_properties(test_rope_vs_reference test_swiglu_gradcheck test_decoder_block_parity
    test_tied_embeddings_single_copy test_build_param_counts test_lm_nxs_roundtrip test_lm_mup_multipliers
    PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
set_tests_properties(test_deep_stack_stability PROPERTIES TIMEOUT 3600)

# FAZ 7 (CPU): non-reentrant checkpoint + bellek planlayici.
add_executable(test_phase7_memory test_phase7_memory.cpp)
target_link_libraries(test_phase7_memory PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase7_memory)
add_test(NAME test_checkpoint_grad_equal COMMAND test_phase7_memory checkpoint)
add_test(NAME test_planner_no_overlap COMMAND test_phase7_memory planner)
set_tests_properties(test_checkpoint_grad_equal test_planner_no_overlap PROPERTIES TIMEOUT 300)

# FAZ 8: veri hatti (BPE vs tiktoken / HF tokenizers, NXTK, ornekleyici).
add_executable(test_phase8_data test_phase8_data.cpp)
target_link_libraries(test_phase8_data PRIVATE nexus::nexus nexus_test_support)
target_compile_definitions(test_phase8_data PRIVATE
    NEXUS_PARITY_PYTHON="${_nexus_parity_python}"
    NEXUS_PARITY_SCRIPTS="${CMAKE_CURRENT_SOURCE_DIR}"
    NEXUS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
nexus_configure_target(test_phase8_data)
if(WIN32)
    target_link_libraries(test_phase8_data PRIVATE psapi)
endif()
add_test(NAME test_bpe_encode_matches_tiktoken COMMAND test_phase8_data tiktoken)
add_test(NAME test_decode_roundtrip COMMAND test_phase8_data roundtrip)
add_test(NAME test_bpe_train_matches_hf COMMAND test_phase8_data train)
add_test(NAME test_shard_roundtrip COMMAND test_phase8_data shard)
add_test(NAME test_sampler_resume_exact COMMAND test_phase8_data resume)
add_test(NAME test_pack_mask COMMAND test_phase8_data pack)
add_test(NAME test_no_overlap_across_ranks COMMAND test_phase8_data ranks)
set_tests_properties(test_bpe_encode_matches_tiktoken test_decode_roundtrip test_bpe_train_matches_hf
    test_shard_roundtrip test_sampler_resume_exact test_pack_mask test_no_overlap_across_ranks
    PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)

# FAZ 9: dil modeli egitmeni v2 (CPU yolu).
add_executable(test_phase9_trainer test_phase9_trainer.cpp)
target_link_libraries(test_phase9_trainer PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase9_trainer)
add_test(NAME test_old_api_still_works COMMAND test_phase9_trainer old_api)
add_test(NAME test_accum_equivalence COMMAND test_phase9_trainer accum)
add_test(NAME test_schedule_values COMMAND test_phase9_trainer schedule)
add_test(NAME test_param_groups_wd COMMAND test_phase9_trainer groups)
add_test(NAME test_nan_policy COMMAND test_phase9_trainer nan)
add_test(NAME test_loss_spike COMMAND test_phase9_trainer spike)
add_test(NAME test_trainer_resume COMMAND test_phase9_trainer resume)
add_test(NAME test_eval_fixed_set COMMAND test_phase9_trainer eval)
add_test(NAME test_trainer_loggers COMMAND test_phase9_trainer loggers)
add_test(NAME test_classifier_shared_infra COMMAND test_phase9_trainer classifier)
set_tests_properties(test_old_api_still_works test_accum_equivalence test_schedule_values test_param_groups_wd
    test_nan_policy test_loss_spike test_trainer_resume test_eval_fixed_set test_trainer_loggers
    test_classifier_shared_infra PROPERTIES TIMEOUT 300)

# FAZ 10: ProcessGroup (thread + tcp), dogrulama, rank olumu.
add_executable(test_phase10_dist test_phase10_dist.cpp)
target_link_libraries(test_phase10_dist PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase10_dist)
add_test(NAME test_collectives_thread COMMAND test_phase10_dist collectives_thread)
foreach(_n 2 3 4 8)
    math(EXPR _port "29620 + ${_n}")
    add_test(NAME test_collectives_tcp_${_n} COMMAND $<TARGET_FILE:nexus_run> --nproc-per-node ${_n} --master-port ${_port}
        -- $<TARGET_FILE:test_phase10_dist> collectives_tcp)
endforeach()
add_test(NAME test_mismatched_collective_detected COMMAND test_phase10_dist mismatch_thread)
add_test(NAME test_mismatched_collective_detected_tcp COMMAND $<TARGET_FILE:nexus_run> --nproc-per-node 2 --master-port 29631
    -- $<TARGET_FILE:test_phase10_dist> mismatch_tcp)
add_test(NAME test_rank_failure_clean_exit COMMAND test_phase10_dist failure $<TARGET_FILE:nexus_run> $<TARGET_FILE:test_phase10_dist>)
add_test(NAME test_collectives_nccl COMMAND test_phase10_dist nccl)
add_test(NAME test_topology_and_lr_scaling COMMAND test_phase10_dist topology)
add_test(NAME test_ddp_equivalence COMMAND test_phase10_dist ddp_thread)
add_test(NAME test_ddp_equivalence_tcp COMMAND $<TARGET_FILE:nexus_run> --nproc-per-node 4 --master-port 29641
    -- $<TARGET_FILE:test_phase10_dist> ddp_tcp)
add_test(NAME test_overlap_occurs COMMAND test_phase10_dist overlap_thread)
set_tests_properties(test_collectives_thread test_collectives_tcp_2 test_collectives_tcp_3 test_collectives_tcp_4
    test_collectives_tcp_8 test_mismatched_collective_detected test_mismatched_collective_detected_tcp
    test_rank_failure_clean_exit test_collectives_nccl test_ddp_equivalence test_ddp_equivalence_tcp test_overlap_occurs test_topology_and_lr_scaling
    PROPERTIES TIMEOUT 600 RUN_SERIAL TRUE)
