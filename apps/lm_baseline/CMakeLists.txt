# Nexus/apps/lm_baseline: FAZ 0 taban cizgisi dil modeli egitimi.
# Olcum: nexus prof altyapisi + surec doruk bellegi; tahmin: nexus_estimate formulleri.
add_executable(lm_baseline main.cpp)
target_link_libraries(lm_baseline PRIVATE nexus::nexus nexus_estimate_core)
nexus_configure_target(lm_baseline)
