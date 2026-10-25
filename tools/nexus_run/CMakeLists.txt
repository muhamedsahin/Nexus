# FAZ 10.2: nexus-run baslatici (torchrun esdegeri).
add_executable(nexus_run main.cpp)
set_target_properties(nexus_run PROPERTIES OUTPUT_NAME nexus-run)
nexus_configure_target(nexus_run)
