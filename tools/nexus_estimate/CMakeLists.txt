# Nexus/tools/nexus_estimate: kapali bicim tahmin araci (FAZ 0).
# Formuller nexus_estimate_core'da; CLI main.cpp'de. Python aynasi
# benchmarks/runner/estimate.py ile parite tests/parity altinda dogrulanir.
add_library(nexus_estimate_core STATIC
    estimate.cpp
    json_mini.cpp)

target_include_directories(nexus_estimate_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(nexus_estimate_core PUBLIC cxx_std_20)
nexus_configure_target(nexus_estimate_core)

add_executable(nexus_estimate main.cpp)
target_link_libraries(nexus_estimate PRIVATE nexus_estimate_core)
nexus_configure_target(nexus_estimate)
