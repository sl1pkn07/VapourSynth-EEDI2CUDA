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
- A CUDA-enabled GPU of compute capability 5.0 or higher (Maxwell+).
- GPU driver **461.33** or newer.

## Compilation
Please refer to [build.yml](https://github.com/AmusementClub/VapourSynth-EEDI2CUDA/blob/main/.github/workflows/build.yml).

## Performance

|EEDI2CUDA   |Holy's EEDI2|
|------------|------------|
|115.72 fps  |17.58 fps   |

Doubling 1080P YUV420P16 clip.
Tested on i7-10875H and RTX 2070.
