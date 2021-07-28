cmake_minimum_required(VERSION 3.20)
project(VapourSynth-EEDI2CUDA CUDA)

include(FetchContent)
FetchContent_Declare(VapourSynth
    URL             https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R54.zip
    URL_HASH    MD5=B869FFB96D423B086C0E19E2210229DF)
FetchContent_MakeAvailable(VapourSynth)

set(CMAKE_CUDA_STANDARD 17 REQUIRED)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
add_library(EEDI2CUDA SHARED
    "EEDI2.cu")
target_include_directories(EEDI2CUDA PRIVATE
    "${vapoursynth_SOURCE_DIR}/include"
    ${Boost_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(EEDI2CUDA PRIVATE
    --ptxas-options=-v
    --restrict
    --ftz=true
    --fmad=true
    --Wdefault-stream-launch
    --Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr
    --display-error-number
    --diag-suppress 69,177,186,830
    -lineinfo
    -src-in-ptx)
if(MSVC)
    target_compile_options(EEDI2CUDA PRIVATE "-Xcompiler=/W4 /wd4100 /wd4297 /wd4458 /wd4291")
else()
    target_compile_options(EEDI2CUDA PRIVATE "-Xcompiler=-Wall -Wno-terminate")
endif()
set_target_properties(EEDI2CUDA PROPERTIES CUDA_ARCHITECTURES "50-virtual")
