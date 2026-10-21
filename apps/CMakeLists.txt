add_executable(nexus_demo demo/main.cpp)
target_link_libraries(nexus_demo PRIVATE nexus::nexus)
nexus_configure_target(nexus_demo)

function(nexus_app folder)
    add_executable(${folder} ${folder}/main.cpp)
    target_link_libraries(${folder} PRIVATE nexus::nexus)
    nexus_configure_target(${folder})
endfunction()

nexus_app(dl_vision_classifier)
nexus_app(custom_ops_showcase)
nexus_app(multi_agent_rl_arena)
nexus_app(hybrid_cluster_benchmark)
