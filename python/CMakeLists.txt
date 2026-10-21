# Nexus Python kabugu. Yorumlayici Anaconda 3.11 olmalidir.
#
# Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
# 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
# olarak gelistirilmistir.

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core
    bindings/module.cpp
    bindings/image_write.cpp)
target_include_directories(_core PRIVATE "${CMAKE_SOURCE_DIR}/lib/NexusData/third_party/stb")
target_link_libraries(_core PRIVATE nexus::nexus)
target_compile_features(_core PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(_core PRIVATE /bigobj /EHsc /utf-8)
    target_compile_definitions(_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(_nexus_py_out "${CMAKE_CURRENT_SOURCE_DIR}/src/nexus")
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_nexus_py_out}"
    RUNTIME_OUTPUT_DIRECTORY "${_nexus_py_out}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${_nexus_py_out}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_nexus_py_out}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${_nexus_py_out}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${_nexus_py_out}")
