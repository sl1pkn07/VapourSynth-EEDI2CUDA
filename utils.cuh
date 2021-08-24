#pragma once

#include "common.h"

__constant__ float spline36_offset00_weights12[] = {9.8684211e-03f,  0.0000000e+00f, -5.9210526e-02f, 0.0000000e+00f,
                                                    2.9934211e-01f,  5.0000000e-01f, 2.9934211e-01f,  0.0000000e+00f,
                                                    -5.9210526e-02f, 0.0000000e+00f, 9.8684211e-03f,  -6.9388939e-18f};

__constant__ float spline36_offset05_weights6[] = {0.019736842f, -0.118421053f, 0.598684211f, 0.598684211f, -0.118421053f, 0.019736842f};

template <typename T>
__global__ void transpose(const T *src, T *dst, const int width, const int height, const int src_stride, const int dst_stride) {
  constexpr int TILE_DIM = 64;
  constexpr int BLOCK_ROWS = 8;

  __shared__ T tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (y + j < height)
      tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * src_stride + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (y + j < width)
      dst[(y + j) * dst_stride + x] = tile[threadIdx.x][threadIdx.y + j];
}

template <typename T> __device__ T value_bound(int val) {
  constexpr int peak = std::numeric_limits<T>::max();
  return static_cast<T>(mmax(mmin(val, peak), 0));
}

template <typename T> __global__ void resample12(const T *src, T *dst, int width, int height, unsigned d_pitch_src, unsigned d_pitch_dst) {
  setup_xy;

  auto pitch = d_pitch_src;
  auto srcp = line(src);
  pitch = d_pitch_dst;
  auto &out = point(dst);

  auto c = 0.f;

  for (int i = -5; i <= 6; ++i)
    c += spline36_offset00_weights12[i + 5] * srcp[mmin(mmax(i + x * 2, 0), width * 2 - 1)];

  out = value_bound<T>(__float2int_rn(c));
}

template <typename T> __global__ void resample6(const T *src, T *dst, int width, int height, unsigned d_pitch_src, unsigned d_pitch_dst) {
  setup_xy;

  auto pitch = d_pitch_src;
  auto srcp = line(src);
  pitch = d_pitch_dst;
  auto &out = point(dst);

  auto c = 0.f;

  for (int i = -3; i < 3; ++i)
    c += spline36_offset05_weights6[i + 3] * srcp[mmin(mmax(i + x, 0), width - 1)];

  out = value_bound<T>(__float2int_rn(c));
}

template <typename T> struct TransposePass final : public BridgePass<T> {
  using BridgePass<T>::BridgePass;

  [[nodiscard]] Pass<T> *dup() const override { return new TransposePass(*this); }

  void process(int, int plane, cudaStream_t stream) override {
    auto ss = !!plane * vi.subSampling;
    auto width = vi.width >> ss;
    auto height = vi.height >> ss;
    dim3 blocks = dim3(64, 8);
    dim3 grids = dim3((width - 1) / blocks.x + 1, (height - 1) / blocks.x + 1);
    transpose<<<grids, blocks, 0, stream>>>(src, dst, width, height, d_pitch_src / sizeof(T) >> ss, d_pitch_dst / sizeof(T) >> ss);
  }
};

template <typename T> struct ScaleDownWPass final : public BridgePass<T> {
  using BridgePass<T>::BridgePass;

  [[nodiscard]] Pass<T> *dup() const override { return new ScaleDownWPass(*this); }

  void process(int, int plane, cudaStream_t stream) override {
    auto ss = !!plane * vi.subSampling;
    auto width = vi2.width >> ss;
    auto height = vi2.height >> ss;
    dim3 blocks = dim3(64, 8);
    dim3 grids = dim3((width - 1) / blocks.x + 1, (height - 1) / blocks.y + 1);
    resample12<<<grids, blocks, 0, stream>>>(src, dst, width, height, d_pitch_src >> ss, d_pitch_dst >> ss);
  }
};

template <typename T> struct ShiftWPass final : public BridgePass<T> {
  using BridgePass<T>::BridgePass;

  [[nodiscard]] Pass<T> *dup() const override { return new ShiftWPass(*this); }

  void process(int, int plane, cudaStream_t stream) override {
    auto ss = !!plane * vi.subSampling;
    auto width = vi.width >> ss;
    auto height = vi.height >> ss;
    dim3 blocks = dim3(64, 8);
    dim3 grids = dim3((width - 1) / blocks.x + 1, (height - 1) / blocks.y + 1);
    resample6<<<grids, blocks, 0, stream>>>(src, dst, width, height, d_pitch_src >> ss, d_pitch_dst >> ss);
  }
};
