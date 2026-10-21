# Nexus kok hedefleri icin ortak derleme ayarlari.
#
# Testler, benchmarklar, ornekler ve uygulamalar ayni uyari seviyesini,
# ayni Windows tanimlarini ve ayni (istege bagli) uyari=hata kapisini kullanir.
# Her hedef bu yardimcilarla tanimlanir; boylece bir bayrak degisikligi tek
# yerde yapilir ve hicbir hedef eksik ayarla derlenmez.
#
# Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
# 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
# olarak gelistirilmistir.

include_guard(GLOBAL)

# nexus_configure_target(<target> [AVX2])
#   /W4 (MSVC) veya -Wall -Wextra -Wpedantic uygular. AVX2 verilirse hedef
#   AVX2 komut kumesiyle derlenir. NEXUS_WARNINGS_AS_ERRORS acikken uyarilar
#   hata sayilir.
function(nexus_configure_target target)
    cmake_parse_arguments(ARG "AVX2" "" "" ${ARGN})
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /EHsc /utf-8 /bigobj)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        if(ARG_AVX2)
            target_compile_options(${target} PRIVATE /arch:AVX2)
        endif()
        if(NEXUS_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(ARG_AVX2)
            target_compile_options(${target} PRIVATE -mavx2 -mfma)
        endif()
        if(NEXUS_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

# nexus_add_test(<name> [AVX2] [SUPPORT] [TIMEOUT <saniye>])
#   <name>.cpp dosyasindan calistirilabilir uretir, nexus::nexus'a baglar ve
#   CTest'e kaydeder. SUPPORT, tests/support altindaki ortak denetim ve
#   operator new sayacini (nexus_test_support OBJECT kutuphanesi) ekler.
function(nexus_add_test name)
    cmake_parse_arguments(ARG "AVX2;SUPPORT" "TIMEOUT" "" ${ARGN})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE nexus::nexus)
    if(ARG_SUPPORT)
        target_link_libraries(${name} PRIVATE nexus_test_support)
    endif()
    if(ARG_AVX2)
        nexus_configure_target(${name} AVX2)
    else()
        nexus_configure_target(${name})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 180)
    endif()
    set_tests_properties(${name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
endfunction()
