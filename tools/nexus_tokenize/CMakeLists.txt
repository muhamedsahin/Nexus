# FAZ 8.2: ham metin/JSONL -> NXTK parcalari + manifest.
add_executable(nexus_tokenize main.cpp)
target_link_libraries(nexus_tokenize PRIVATE nexusdata::nexusdata)
nexus_configure_target(nexus_tokenize)
