# VapourSynth-EEDI2CUDA

Copyright (C) 2005-2006 Kevin Stone

Copyright (C) 2014-2019 HolyWu

Copyright (C) 2021 Misaki Kasumi

EEDI2 filter using CUDA

Ported from [HolyWu's plugin](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI2). See usage there.

## Arguments
`eedi2cuda.EEDI2(clip clip, int field[, int mthresh=10, int lthresh=20, int vthresh=20, int estr=2, int dstr=4, int maxd=24, int map=0, int nt=50, int pp=1, int num_streams=1])`

Only `pp=1` is implemented.

The additional argument `int num_streams` specify the number of CUDA streams it will use. The value must less than or equal to `core.num_threads`.
A larger value increases the concurrency and also increases the GPU memory usage. The default value `num_streams=1` is already fast enough.

## Requirements
- CPU with AVX support.
- CUDA-enabled GPU(s) of compute capability 5.0 or higher (Maxwell+).
- GPU driver **471.11** or newer. (If you see `the provided PTX was compiled with an unsupported toolchain`, please upgrade your driver.)

## Compilation on Windows
### Requirements
- MSVC v142 - VS 2019 C++ x64/x86 build tools (v14.29-16.10). Older compilers may not be able to compile!
- CUDA Toolchain 11.4
- Boost C++ Libraries [dcea408](https://github.com/boostorg/boost/tree/dcea408). Right that commit!
- CMake 3.20

### Commands
```cmd
:: Before you start, make sure cl.exe and nvcc.exe are in %Path%
mkdir build
cd build
:: I use Ninja but you can use what you like
:: You must specify the path to Boost in `Boost_INCLUDE_DIRS`
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DBoost_INCLUDE_DIRS=W:\path\to\boost
ninja
```

## Performance

|EEDI2CUDA   |Holy's EEDI2|
|------------|------------|
|115.72 fps  |17.58 fps   |

Tested on i7-10875H and RTX 2070 with the script below:
```python
import vapoursynth as vs
from vapoursynth import core

core.num_threads = 16
core.max_cache_size = 16000

a = r"W:\BRMM-103621BD\BDMV\STREAM\00008.m2ts"

src8 = core.lsmas.LWLibavSource(a)
src16 = src8.resize.Point(format=vs.YUV420P16)[4000:4500] * 20
a = src16.eedi2.EEDI2(1)
b = src16.eedi2cuda.EEDI2(1, num_streams=16)
a.set_output(1)
b.set_output(0)
```
