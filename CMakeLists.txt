cmake_minimum_required(VERSION 3.20)
project(VapourSynth-EEDI2CUDA CUDA)

include(FetchContent)
FetchContent_Declare(VapourSynth
    URL             https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R54.zip
    URL_HASH    MD5=B869FFB96D423B086C0E19E2210229DF)
FetchContent_Declare(GSL
    GIT_REPOSITORY  https://github.com/microsoft/GSL
    GIT_TAG         8a4b9ed0bf643726ce625678a17b1fc40d90870c)
FetchContent_MakeAvailable(VapourSynth GSL)

set(CMAKE_CUDA_STANDARD 14 REQUIRED)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
add_library(EEDI2CUDA SHARED
    "EEDI2.cu")
target_include_directories(EEDI2CUDA PRIVATE
    "${vapoursynth_SOURCE_DIR}/include"
    "${gsl_SOURCE_DIR}/include"
    ${CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(EEDI2CUDA PRIVATE
    --ptxas-options=-v
    --restrict
    --use_fast_math
    --extra-device-vectorization
    --Wdefault-stream-launch
    --Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr)
if(MSVC)
    target_compile_options(EEDI2CUDA PRIVATE -Xcompiler=/W4)
else()
    target_compile_options(EEDI2CUDA PRIVATE -Xcompiler=-Wall)
endif()
set_target_properties(EEDI2CUDA PROPERTIES CUDA_ARCHITECTURES "75-real")
