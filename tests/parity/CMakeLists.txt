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

# FAZ 11: ZeRO-0/1/2/3, FSDP, HSDP, onceden getirme, NVMe takasi, parcali kayit.
add_executable(test_phase11_fsdp test_phase11_fsdp.cpp)
target_link_libraries(test_phase11_fsdp PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase11_fsdp)
add_test(NAME test_zero_stages_equivalence COMMAND test_phase11_fsdp stages)
add_test(NAME test_zero_stages_equivalence_tcp COMMAND $<TARGET_FILE:nexus_run> --nproc-per-node 4 --master-port 29671
    -- $<TARGET_FILE:test_phase11_fsdp> stages_tcp)
add_test(NAME test_fsdp_memory_scaling COMMAND test_phase11_fsdp memory)
add_test(NAME test_fsdp_prefetch_order COMMAND test_phase11_fsdp prefetch)
add_test(NAME test_nvme_offload_roundtrip COMMAND test_phase11_fsdp nvme)
add_test(NAME test_sharded_checkpoint_reshard COMMAND test_phase11_fsdp reshard)
add_test(NAME test_hsdp_and_bf16_comm COMMAND test_phase11_fsdp hybrid)
set_tests_properties(test_zero_stages_equivalence test_zero_stages_equivalence_tcp test_fsdp_memory_scaling
    test_fsdp_prefetch_order test_nvme_offload_roundtrip test_sharded_checkpoint_reshard test_hsdp_and_bf16_comm
    PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

# FAZ 12: DeviceMesh/DTensor, TP/SP/async TP, PP zamanlamalari, CP (ring/Ulysses), 3B/4B esdegerlik.
add_executable(test_phase12_parallel test_phase12_parallel.cpp)
target_link_libraries(test_phase12_parallel PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase12_parallel)
add_test(NAME test_dtensor_redistribute_all_pairs COMMAND test_phase12_parallel dtensor)
add_test(NAME test_tp_equivalence COMMAND test_phase12_parallel tp)
add_test(NAME test_vocab_parallel_ce_equals_full COMMAND test_phase12_parallel vocab)
add_test(NAME test_pp_schedules_equivalence COMMAND test_phase12_parallel pp)
add_test(NAME test_ring_attention_equals_flash COMMAND test_phase12_parallel ring)
add_test(NAME test_parallel_zero1_equivalence COMMAND test_phase12_parallel zero)
add_test(NAME test_3d_equivalence COMMAND test_phase12_parallel 3d)
add_test(NAME test_4d_equivalence COMMAND test_phase12_parallel 4d)
add_test(NAME test_parallel_trainer_api COMMAND test_phase12_parallel trainer)
add_test(NAME test_tp_equivalence_tcp COMMAND $<TARGET_FILE:nexus_run> --nproc-per-node 4 --master-port 29681
    -- $<TARGET_FILE:test_phase12_parallel> tcp)
set_tests_properties(test_dtensor_redistribute_all_pairs test_tp_equivalence test_vocab_parallel_ce_equals_full
    test_pp_schedules_equivalence test_ring_attention_equals_flash test_parallel_zero1_equivalence test_3d_equivalence
    test_4d_equivalence test_parallel_trainer_api test_tp_equivalence_tcp PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

# FAZ 12.G: otomatik paralellik onericisi (nexus_estimate auto_parallel).
add_executable(test_phase12_autoconfig test_phase12_autoconfig.cpp)
target_link_libraries(test_phase12_autoconfig PRIVATE nexus_estimate_core nexus_test_support)
nexus_configure_target(test_phase12_autoconfig)
add_test(NAME test_auto_parallel_config COMMAND test_phase12_autoconfig)

# FAZ 13 (CPU): IR, izleme, gecisler, kod uretimi (cl.exe JIT), yurutucu, StableHLO.
add_executable(test_phase13_compile test_phase13_compile.cpp)
target_link_libraries(test_phase13_compile PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase13_compile)
add_test(NAME test_compile_numerics COMMAND test_phase13_compile numerics)
add_test(NAME test_fusion_kernel_count COMMAND test_phase13_compile fusion)
add_test(NAME test_dynamic_shape_no_recompile COMMAND test_phase13_compile dynamic)
add_test(NAME test_cache_hit_zero_compile COMMAND test_phase13_compile cache)
add_test(NAME test_compile_min_cut_memory COMMAND test_phase13_compile memory)
add_test(NAME test_compile_comm_reorder COMMAND test_phase13_compile comm)
add_test(NAME test_compile_export_stablehlo COMMAND test_phase13_compile export)
set_tests_properties(test_compile_numerics test_fusion_kernel_count test_dynamic_shape_no_recompile test_cache_hit_zero_compile
    test_compile_min_cut_memory test_compile_comm_reorder test_compile_export_stablehlo PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

# FAZ 14 (CPU): MoE â€” yonlendirici, kapasite, gruplu GEMM, yuk dengeleme, uzman paralelligi, dar uzman agirliklari.
add_executable(test_phase14_moe test_phase14_moe.cpp)
target_link_libraries(test_phase14_moe PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase14_moe)
add_test(NAME test_moe_router_deterministic COMMAND test_phase14_moe router)
add_test(NAME test_moe_k1_equals_dense COMMAND test_phase14_moe k1)
add_test(NAME test_moe_grad_numerics COMMAND test_phase14_moe grad)
add_test(NAME test_grouped_gemm_vs_loop COMMAND test_phase14_moe grouped)
add_test(NAME test_load_balance_in_range COMMAND test_phase14_moe balance)
add_test(NAME test_ep_equivalence COMMAND test_phase14_moe ep)
add_test(NAME test_moe_quantized_experts COMMAND test_phase14_moe quant)
add_test(NAME test_moe_config_roundtrip COMMAND test_phase14_moe config)
set_tests_properties(test_load_balance_in_range PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(test_moe_router_deterministic test_moe_k1_equals_dense test_moe_grad_numerics test_grouped_gemm_vs_loop
    test_load_balance_in_range test_ep_equivalence test_moe_quantized_experts test_moe_config_roundtrip PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

# FAZ 15 (CPU): dayaniklilik â€” NXS3 tam durum kaydi, RNG, parcali kayit, yeniden parcalama, cokme, bekci, elastik.
add_executable(test_phase15_durability test_phase15_durability.cpp)
target_link_libraries(test_phase15_durability PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase15_durability)
add_test(NAME test_save_load_exact COMMAND test_phase15_durability exact)
add_test(NAME test_rng_restore COMMAND test_phase15_durability rng)
add_test(NAME test_sharded_save_load COMMAND test_phase15_durability sharded)
add_test(NAME test_reshard_2_to_4 COMMAND test_phase15_durability reshard)
add_test(NAME test_crash_injection COMMAND test_phase15_durability crash)
add_test(NAME test_watchdog_timeout COMMAND test_phase15_durability watchdog)
add_test(NAME test_elastic_restart COMMAND test_phase15_durability elastic $<TARGET_FILE:nexus_run>)
set_tests_properties(test_save_load_exact test_rng_restore test_sharded_save_load test_reshard_2_to_4 test_crash_injection
    test_watchdog_timeout test_elastic_restart PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

# FAZ 16 (CPU): lm::LanguageModel artimli cozme (KV onbellegi), tum mimari varyantlari.
add_executable(test_phase16_infer test_phase16_infer.cpp)
target_link_libraries(test_phase16_infer PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase16_infer)
add_test(NAME test_kv_cache_matches_full_cpu COMMAND test_phase16_infer kvcache)
add_test(NAME test_kv_cache_generate_cpu COMMAND test_phase16_infer generate)
set_tests_properties(test_kv_cache_matches_full_cpu test_kv_cache_generate_cpu PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

# FAZ 17 (CPU): birlikte calisabilirlik â€” safetensors, HF ice/disa aktarma (llama/mistral/qwen2), GGUF, ONNX.
add_executable(test_phase17_interop test_phase17_interop.cpp)
target_link_libraries(test_phase17_interop PRIVATE nexus::nexus nexus_test_support)
target_compile_definitions(test_phase17_interop PRIVATE
    NEXUS_PARITY_PYTHON="${_nexus_parity_python}"
    NEXUS_FAZ17_SCRIPT="${CMAKE_SOURCE_DIR}/benchmarks/runner/faz17_ref.py")
nexus_configure_target(test_phase17_interop)
add_test(NAME test_safetensors_roundtrip COMMAND test_phase17_interop safetensors)
add_test(NAME test_hf_templates COMMAND test_phase17_interop hf_tiny)
add_test(NAME test_hf_import_logits COMMAND test_phase17_interop hf_tinyllama)
add_test(NAME test_gguf_roundtrip COMMAND test_phase17_interop gguf)
add_test(NAME test_onnx_roundtrip COMMAND test_phase17_interop onnx)
set_tests_properties(test_safetensors_roundtrip test_hf_templates test_hf_import_logits test_gguf_roundtrip test_onnx_roundtrip PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
# Python: DLPack sifir kopya (torch CPU + CUDA), nexus.lm / nexus.interop. _core.cp314 build-py'de derlenir
# (NEXUS_BUILD_PYTHON); yoksa 77 (SKIP). torch'lu yorumlayici: .bench-venv-cuda (CUDA testleri icin).
if(EXISTS "${CMAKE_SOURCE_DIR}/.bench-venv-cuda/Scripts/python.exe")
    add_test(NAME test_dlpack_torch COMMAND "${CMAKE_SOURCE_DIR}/.bench-venv-cuda/Scripts/python.exe"
        "${CMAKE_SOURCE_DIR}/python/tests/test_faz17.py")
    set_tests_properties(test_dlpack_torch PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python/src"
        TIMEOUT 600 SKIP_RETURN_CODE 77)
endif()

# FAZ 18 (CPU): ozellik algilama, AVX2/AVX512/VNNI GEMM (FP32/BF16/FP16/INT8/INT4), sabitli havuz / NUMA, CPU FlashAttention, egitim, kuantize cikarim, nexus::compile fusion.
add_executable(test_phase18_cpu test_phase18_cpu.cpp)
target_link_libraries(test_phase18_cpu PRIVATE nexus::nexus nexus_test_support)
nexus_configure_target(test_phase18_cpu)
add_test(NAME test_cpu_features COMMAND test_phase18_cpu features)
add_test(NAME test_cpu_gemm_vs_ref COMMAND test_phase18_cpu gemm)
add_test(NAME test_cpu_gemm_avx2_vs_ref COMMAND test_phase18_cpu gemm_avx2)
add_test(NAME test_cpu_gemm_avx512_vs_ref COMMAND test_phase18_cpu gemm_avx512)
add_test(NAME test_numa_affinity COMMAND test_phase18_cpu numa)
add_test(NAME test_cpu_numa_and_pinning COMMAND test_phase18_cpu numa)
add_test(NAME test_offload_adam_bitwise COMMAND test_phase18_cpu adam)
add_test(NAME test_cpu_adam_step COMMAND test_phase18_cpu adam)
add_test(NAME test_zero_offload_step COMMAND test_phase18_cpu offload)
add_test(NAME test_cpu_flash_vs_math COMMAND test_phase18_cpu flash)
add_test(NAME test_cpu_flash_attn_vs_ref COMMAND test_phase18_cpu flash)
add_test(NAME test_cpu_train_parity COMMAND test_phase18_cpu train)
add_test(NAME test_cpu_quantized_inference COMMAND test_phase18_cpu quant)
add_test(NAME test_cpu_compile_fusion COMMAND test_phase18_cpu compile)
set_tests_properties(
    test_cpu_features
    test_cpu_gemm_vs_ref
    test_cpu_gemm_avx2_vs_ref
    test_cpu_gemm_avx512_vs_ref
    test_numa_affinity
    test_cpu_numa_and_pinning
    test_offload_adam_bitwise
    test_cpu_adam_step
    test_zero_offload_step
    test_cpu_flash_vs_math
    test_cpu_flash_attn_vs_ref
    test_cpu_train_parity
    test_cpu_quantized_inference
    test_cpu_compile_fusion
    PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)


# FAZ 19: Diger hizlandiricilar (ROCm/HIP, Vulkan SPIR-V, Metal MSL, DirectML DX12, OpenXLA PJRT).
add_executable(test_phase19_backends test_phase19_backends.cpp)
target_link_libraries(test_phase19_backends PRIVATE nexus::nexus nexus_test_support)
target_compile_definitions(test_phase19_backends PRIVATE NEXUS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
nexus_configure_target(test_phase19_backends)
add_test(NAME test_backend_detection COMMAND test_phase19_backends detection)
add_test(NAME test_hip_parity COMMAND test_phase19_backends hip)
add_test(NAME test_hip_source_compat COMMAND test_phase19_backends hip-source)
set_tests_properties(test_hip_source_compat PROPERTIES TIMEOUT 180)
add_test(NAME test_vulkan_parity COMMAND test_phase19_backends vulkan)
add_test(NAME test_directml_parity COMMAND test_phase19_backends directml)
add_test(NAME test_pjrt_export COMMAND test_phase19_backends pjrt)
add_test(NAME test_backend_fallback COMMAND test_phase19_backends fallback)
add_library(nexus_pjrt_test_plugin SHARED pjrt_test_plugin.cpp)
target_include_directories(nexus_pjrt_test_plugin PRIVATE "${CMAKE_SOURCE_DIR}/include")
set_target_properties(nexus_pjrt_test_plugin PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
nexus_configure_target(nexus_pjrt_test_plugin)
add_executable(test_pjrt_plugin test_pjrt_plugin.cpp)
target_link_libraries(test_pjrt_plugin PRIVATE nexus::nexus nexus_test_support nexus_pjrt_test_plugin)
nexus_configure_target(test_pjrt_plugin)
add_test(NAME test_pjrt_plugin_abi COMMAND test_pjrt_plugin $<TARGET_FILE:nexus_pjrt_test_plugin>)
set_tests_properties(test_pjrt_plugin_abi PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
set(NEXUS_PJRT_TEST_PLUGIN_PATH "" CACHE FILEPATH "Trusted vendor PJRT library for real execution verification")
add_test(NAME test_pjrt_vendor COMMAND test_phase19_backends vendor "${NEXUS_PJRT_TEST_PLUGIN_PATH}")
set_tests_properties(test_pjrt_vendor PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
set_tests_properties(
    test_backend_detection
    test_hip_parity
    test_vulkan_parity
    test_directml_parity
    test_pjrt_export
    test_backend_fallback
    PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

set(NEXUS_XLA_PYTHON "" CACHE FILEPATH "Optional Python with jax/jaxlib for independent StableHLO verification")
if(NEXUS_XLA_PYTHON)
    add_test(NAME test_pjrt_xla_cpu
        COMMAND "${NEXUS_XLA_PYTHON}" "${CMAKE_SOURCE_DIR}/benchmarks/runner/faz19_pjrt.py"
            --producer $<TARGET_FILE:test_phase19_backends>)
    set_tests_properties(test_pjrt_xla_cpu PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
endif()
