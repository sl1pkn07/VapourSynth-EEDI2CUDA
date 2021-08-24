#pragma once

#define setup_xy int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y

struct VideoInfo {
  int width, height, subSampling;
};
