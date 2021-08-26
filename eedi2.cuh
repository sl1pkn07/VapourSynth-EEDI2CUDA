/*
 * EEDI2CUDA: EEDI2 filter using CUDA
 *
 * Copyright (C) 2005-2006 Kevin Stone
 * Copyright (C) 2014-2019 HolyWu
 * Copyright (C) 2021 Misaki Kasumi
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#pragma once

#include <limits>
#include <type_traits>

#include "common.h"

struct EEDI2Param : public VideoDimension {
  unsigned d_pitch, field;
  unsigned nt4, nt7, nt8, nt13, nt19;
  unsigned mthresh, lthresh, vthresh;
  unsigned estr, dstr, maxd;
};

#define KERNEL __global__ __launch_bounds__(64)

#define setup_kernel                                                                                                                       \
  setup_xy;                                                                                                                                \
  int width = d.width, height = d.height;                                                                                                  \
  auto pitch = d.d_pitch;                                                                                                                  \
  __builtin_assume(width > 0);                                                                                                             \
  __builtin_assume(height > 0);                                                                                                            \
  __builtin_assume(x >= 0);                                                                                                                \
  __builtin_assume(y >= 0);                                                                                                                \
  constexpr T shift = sizeof(T) * 8 - 8, peak = std::numeric_limits<T>::max(), ten = 10 << shift, twleve = 12 << shift,                    \
              eight = 8 << shift, twenty = 20 << shift, three = 3 << shift, nine = 9 << shift;                                             \
  constexpr T shift2 = shift + 2, neutral = peak / 2;                                                                                      \
  constexpr int intmax = std::numeric_limits<int>::max()

__device__ int limlut[33]{6,  6,  7,  7,  8,  8,  9,  9,  9,  10, 10, 11, 11, 12, 12, 12, 12,
                          12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, -1, -1};

#define setup_kernel2x                                                                                                                     \
  setup_kernel;                                                                                                                            \
  height *= 2;                                                                                                                             \
  y = d.field ? 2 * y + 1 : 2 * y

#define bounds_check3(value, lower, upper)                                                                                                 \
  if ((value) < (lower) || (value) >= (upper))                                                                                             \
  return

namespace bose {
template <typename T, size_t I, size_t J> __device__ __forceinline__ void P(T *arr) {
  T &a = arr[I - 1], &b = arr[J - 1];
  const T c = a;
  a = a < b ? a : b;
  b = c < b ? b : c;
}

template <typename T, size_t I, size_t X, size_t J, size_t Y> __device__ __forceinline__ void Pbracket(T *arr) {
  constexpr size_t A = X / 2, B = (X & 1) ? (Y / 2) : ((Y + 1) / 2);

  if constexpr (X == 1 && Y == 1)
    P<T, I, J>(arr);
  else if constexpr (X == 1 && Y == 2) {
    P<T, I, J + 1>(arr);
    P<T, I, J>(arr);
  } else if constexpr (X == 2 && Y == 1) {
    P<T, I, J>(arr);
    P<T, I + 1, J>(arr);
  } else {
    Pbracket<T, I, A, J, B>(arr);
    Pbracket<T, I + A, X - A, J + B, Y - B>(arr);
    Pbracket<T, I + A, X - A, J, B>(arr);
  }
}

template <typename T, size_t I, size_t M> __device__ __forceinline__ void Pstar(T *arr) {
  constexpr size_t A = M / 2;

  if constexpr (M > 1) {
    Pstar<T, I, A>(arr);
    Pstar<T, I + A, M - A>(arr);
    Pbracket<T, I, A, I + A, M - A>(arr);
  }
}
} // namespace bose

template <typename T, size_t N> __device__ __forceinline__ void boseSortArray(T (&arr)[N]) { bose::Pstar<T, 1, N>(arr); }

template <typename T> __device__ __forceinline__ T mmax(T last) { return last; }

template <typename T, typename... Args> __device__ __forceinline__ T mmax(T first, Args... remaining) {
  auto candidate = mmax(remaining...);
  static_assert(std::is_same<decltype(candidate), T>::value, "arguments must have the same type");
  return first < candidate ? candidate : first;
}

template <typename T> __device__ __forceinline__ T mmin(T last) { return last; }

template <typename T, typename... Args> __device__ __forceinline__ T mmin(T first, Args... remaining) {
  auto candidate = mmin(remaining...);
  static_assert(std::is_same<decltype(candidate), T>::value, "arguments must have the same type");
  return first < candidate ? first : candidate;
}

template <typename T> __device__ T round_div(T a, T b) {
  // XXX: only for same sign
  return (a + (b / 2)) / b;
}

template <typename T> KERNEL void buildEdgeMask(const EEDI2Param d, const T *src, T *dst) {
  setup_kernel;

  auto srcp = line(src);
  auto srcpp = lineOff(src, -1);
  auto srcpn = lineOff(src, 1);
  auto &out = point(dst);

  out = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if ((abs(srcpp[x] - srcp[x]) < ten && abs(srcp[x] - srcpn[x]) < ten && abs(srcpp[x] - srcpn[x]) < ten) ||
      (abs(srcpp[x - 1] - srcp[x - 1]) < ten && abs(srcp[x - 1] - srcpn[x - 1]) < ten && abs(srcpp[x - 1] - srcpn[x - 1]) < ten &&
       abs(srcpp[x + 1] - srcp[x + 1]) < ten && abs(srcp[x + 1] - srcpn[x + 1]) < ten && abs(srcpp[x + 1] - srcpn[x + 1]) < ten))
    return;

  const unsigned sum =
      (srcpp[x - 1] + srcpp[x] + srcpp[x + 1] + srcp[x - 1] + srcp[x] + srcp[x + 1] + srcpn[x - 1] + srcpn[x] + srcpn[x + 1]) >> shift;
  const unsigned sumsq = (srcpp[x - 1] >> shift) * (srcpp[x - 1] >> shift) + (srcpp[x] >> shift) * (srcpp[x] >> shift) +
                         (srcpp[x + 1] >> shift) * (srcpp[x + 1] >> shift) + (srcp[x - 1] >> shift) * (srcp[x - 1] >> shift) +
                         (srcp[x] >> shift) * (srcp[x] >> shift) + (srcp[x + 1] >> shift) * (srcp[x + 1] >> shift) +
                         (srcpn[x - 1] >> shift) * (srcpn[x - 1] >> shift) + (srcpn[x] >> shift) * (srcpn[x] >> shift) +
                         (srcpn[x + 1] >> shift) * (srcpn[x + 1] >> shift);
  if (9 * sumsq - sum * sum < d.vthresh)
    return;

  const unsigned Ix = abs(srcp[x + 1] - srcp[x - 1]) >> shift;
  const unsigned Iy = mmax(abs(srcpp[x] - srcpn[x]), abs(srcpp[x] - srcp[x]), abs(srcp[x] - srcpn[x])) >> shift;
  if (Ix * Ix + Iy * Iy >= d.mthresh)
    out = peak;

  const unsigned Ixx = abs(srcp[x - 1] - 2 * srcp[x] + srcp[x + 1]) >> shift;
  const unsigned Iyy = abs(srcpp[x] - 2 * srcp[x] + srcpn[x]) >> shift;
  if (Ixx + Iyy >= d.lthresh)
    out = peak;
}

template <typename T> KERNEL void erode(const EEDI2Param d, const T *msk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  unsigned count = 0;

  count += mskpp[x - 1] == peak;
  count += mskpp[x] == peak;
  count += mskpp[x + 1] == peak;
  count += mskp[x - 1] == peak;
  count += mskp[x + 1] == peak;
  count += mskpn[x - 1] == peak;
  count += mskpn[x] == peak;
  count += mskpn[x + 1] == peak;

  out = mskp[x] == peak && count < d.estr ? 0 : mskp[x];
}

template <typename T> KERNEL void dilate(const EEDI2Param d, const T *msk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  unsigned count = 0;

  count += mskpp[x - 1] == peak;
  count += mskpp[x] == peak;
  count += mskpp[x + 1] == peak;
  count += mskp[x - 1] == peak;
  count += mskp[x + 1] == peak;
  count += mskpn[x - 1] == peak;
  count += mskpn[x] == peak;
  count += mskpn[x + 1] == peak;

  out = mskp[x] == 0 && count >= d.dstr ? peak : mskp[x];
}

template <typename T> KERNEL void removeSmallHorzGaps(const EEDI2Param d, const T *msk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto &out = point(dst);

  auto orig = mskp[x];
  out = orig;

  bounds_check3(x, 3, width - 3);
  bounds_check3(y, 1, height - 1);

  auto a = mskp[x - 3] | mskp[x - 2] | mskp[x - 1] | mskp[x + 1] | mskp[x + 2] | mskp[x + 3] ? orig : 0;
  auto b =
      (mskp[x + 1] & (mskp[x - 1] | mskp[x - 2] | mskp[x - 3])) | (mskp[x + 2] & (mskp[x - 1] | mskp[x - 2])) | (mskp[x + 3] & mskp[x - 1])
          ? peak
          : orig;

  out = mskp[x] ? a : b;
}

template <typename T> KERNEL void calcDirections(const EEDI2Param d, const T *src, const T *msk, T *dst) {
  setup_kernel;

  auto srcp = line(src);
  auto mskp = line(msk);
  auto src2p = lineOff(src, -2);
  auto srcpp = lineOff(src, -1);
  auto srcpn = lineOff(src, 1);
  auto src2n = lineOff(src, 2);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = peak;

  constexpr int block_w = 64;
  constexpr int off_w = block_w / 2;
  constexpr int buf_w = block_w * 2;
  __shared__ int s2p[buf_w], s1p[buf_w], s[buf_w], s1n[buf_w], s2n[buf_w];
  __shared__ int m1p[buf_w], m1n[buf_w];

  // XXX: we are safe because they won't be the first or last plane in pool
  s2p[x % block_w] = src2p[x - off_w];
  s2p[x % block_w + block_w] = src2p[x + off_w];
  s1p[x % block_w] = srcpp[x - off_w];
  s1p[x % block_w + block_w] = srcpp[x + off_w];
  s[x % block_w] = srcp[x - off_w];
  s[x % block_w + block_w] = srcp[x + off_w];
  s1n[x % block_w] = srcpn[x - off_w];
  s1n[x % block_w + block_w] = srcpn[x + off_w];
  s2n[x % block_w] = src2n[x - off_w];
  s2n[x % block_w + block_w] = src2n[x + off_w];
  m1p[x % block_w] = mskpp[x - off_w];
  m1p[x % block_w + block_w] = mskpp[x + off_w];
  m1n[x % block_w] = mskpn[x - off_w];
  m1n[x % block_w + block_w] = mskpn[x + off_w];
  __syncthreads();
  auto X = x % block_w + off_w;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  const int maxd = d.maxd >> d.subSampling;

  if (mskp[x] != peak || (mskp[x - 1] != peak && mskp[x + 1] != peak))
    return;

  const int uStartBound = -x + 1;
  const int uStopBound = width - 2 - x;
  const unsigned min0 = abs(s[X] - s1n[X]) + abs(s[X] - s1p[X]);
  unsigned minA = mmin(d.nt19, min0 * 9);
  unsigned minB = mmin(d.nt13, min0 * 6);
  unsigned minC = minA;
  unsigned minD = minB;
  unsigned minE = minB;
  int dirA = -5000, dirB = -5000, dirC = -5000, dirD = -5000, dirE = -5000;

  for (int u = -maxd; u <= maxd; u++) {
    if (u < uStartBound || u > uStopBound)
      continue;
    if ((y == 1 || m1p[X - 1 + u] == peak || m1p[X + u] == peak || m1p[X + 1 + u] == peak) &&
        (y == height - 2 || m1n[X - 1 - u] == peak || m1n[X - u] == peak || m1n[X + 1 - u] == peak)) {
      const unsigned diffsn = abs(s[X - 1] - s1n[X - 1 - u]) + abs(s[X] - s1n[X - u]) + abs(s[X + 1] - s1n[X + 1 - u]);
      const unsigned diffsp = abs(s[X - 1] - s1p[X - 1 + u]) + abs(s[X] - s1p[X + u]) + abs(s[X + 1] - s1p[X + 1 + u]);
      const unsigned diffps = abs(s1p[X - 1] - s[X - 1 - u]) + abs(s1p[X] - s[X - u]) + abs(s1p[X + 1] - s[X + 1 - u]);
      const unsigned diffns = abs(s1n[X - 1] - s[X - 1 + u]) + abs(s1n[X] - s[X + u]) + abs(s1n[X + 1] - s[X + 1 + u]);
      const unsigned diff = diffsn + diffsp + diffps + diffns;
      unsigned diffD = diffsp + diffns;
      unsigned diffE = diffsn + diffps;

      if (diff < minB) {
        dirB = u;
        minB = diff;
      }

      if (y > 1) {
        const unsigned diff2pp = abs(s2p[X - 1] - s1p[X - 1 - u]) + abs(s2p[X] - s1p[X - u]) + abs(s2p[X + 1] - s1p[X + 1 - u]);
        const unsigned diffp2p = abs(s1p[X - 1] - s2p[X - 1 + u]) + abs(s1p[X] - s2p[X + u]) + abs(s1p[X + 1] - s2p[X + 1 + u]);
        const unsigned diffA = diff + diff2pp + diffp2p;
        diffD += diffp2p;
        diffE += diff2pp;

        if (diffA < minA) {
          dirA = u;
          minA = diffA;
        }
      }

      if (y < height - 2) {
        const unsigned diff2nn = abs(s2n[X - 1] - s1n[X - 1 + u]) + abs(s2n[X] - s1n[X + u]) + abs(s2n[X + 1] - s1n[X + 1 + u]);
        const unsigned diffn2n = abs(s1n[X - 1] - s2n[X - 1 - u]) + abs(s1n[X] - s2n[X - u]) + abs(s1n[X + 1] - s2n[X + 1 - u]);
        const unsigned diffC = diff + diff2nn + diffn2n;
        diffD += diff2nn;
        diffE += diffn2n;

        if (diffC < minC) {
          dirC = u;
          minC = diffC;
        }
      }

      if (diffD < minD) {
        dirD = u;
        minD = diffD;
      }

      if (diffE < minE) {
        dirE = u;
        minE = diffE;
      }
    }
  }

  auto okA = dirA != -5000, okB = dirB != -5000, okC = dirC != -5000, okD = dirD != -5000, okE = dirE != -5000;
  unsigned k = okA + okB + okC + okD + okE;
  int order[] = {
      okA ? dirA : intmax, okB ? dirB : intmax, okC ? dirC : intmax, okD ? dirD : intmax, okE ? dirE : intmax,
  };

  if (k > 1) {
    boseSortArray(order);

    const int mid = (k & 1) ? order[k / 2] : (order[(k - 1) / 2] + order[k / 2] + 1) / 2;
    const int lim = mmax(limlut[abs(mid)] / 4, 2);
    int sum = 0;
    int count = 0;

    for (unsigned i = 0; i < 5; i++) {
      auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
      sum += cond * order[i];
      count += cond;
    }

    out = (count > 1) ? neutral + ((sum / count) << shift2) : neutral;
  } else {
    out = neutral;
  }
}

template <typename T> KERNEL void filterDirMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto dmskp = line(dmsk);
  auto dmskpp = lineOff(dmsk, -1);
  auto dmskpn = lineOff(dmsk, 1);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (mskp[x] != peak)
    return;

  int val0 = dmskpp[x - 1], val1 = dmskpp[x], val2 = dmskpp[x + 1], val3 = dmskp[x - 1], val4 = dmskp[x], val5 = dmskp[x + 1],
      val6 = dmskpn[x - 1], val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak, cond3 = val3 != peak, cond4 = val4 != peak, cond5 = val5 != peak,
       cond6 = val6 != peak, cond7 = val7 != peak, cond8 = val8 != peak;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax, cond3 ? val3 : intmax, cond4 ? val4 : intmax,
      cond5 ? val5 : intmax, cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned u = cond0 + cond1 + cond2 + cond3 + cond4 + cond5 + cond6 + cond7 + cond8;

  if (u < 4) {
    out = peak;
    return;
  }

  boseSortArray(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  for (unsigned i = 0; i < 9; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < 4 || (count < 5 && dmskp[x] == peak)) {
    out = peak;
    return;
  }

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void expandDirMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto dmskp = line(dmsk);
  auto dmskpp = lineOff(dmsk, -1);
  auto dmskpn = lineOff(dmsk, 1);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (dmskp[x] != peak || mskp[x] != peak)
    return;

  int val0 = dmskpp[x - 1], val1 = dmskpp[x], val2 = dmskpp[x + 1], val3 = dmskp[x - 1], val5 = dmskp[x + 1], val6 = dmskpn[x - 1],
      val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak, cond3 = val3 != peak, cond5 = val5 != peak, cond6 = val6 != peak,
       cond7 = val7 != peak, cond8 = val8 != peak;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax, cond3 ? val3 : intmax,
      cond5 ? val5 : intmax, cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned u = cond0 + cond1 + cond2 + cond3 + cond5 + cond6 + cond7 + cond8;

  if (u < 5)
    return;

  boseSortArray(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  for (unsigned i = 0; i < 8; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < 5)
    return;

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void filterMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto dmskp = line(dmsk);
  auto dmskpp = lineOff(dmsk, -1);
  auto dmskpn = lineOff(dmsk, 1);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (dmskp[x] == peak || mskp[x] != peak)
    return;

  int dir = (dmskp[x] - neutral) / 4;
  const int lim = mmax(abs(dir) * 2, twleve + 0);
  dir >>= shift;
  bool ict = false, icb = false;

  auto cond = dir < 0;
  auto l = cond ? mmax(-x, dir) : 0;
  auto r = cond ? 0 : mmin(width - x - 1, dir);
  for (int j = l; j <= r; j++) {
    if ((abs(dmskpp[x + j] - dmskp[x]) > lim && dmskpp[x + j] != peak) || (dmskp[x + j] == peak && dmskpp[x + j] == peak) ||
        (abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
      ict = true;
      break;
    }
  }

  if (ict) {
    auto l = cond ? 0 : mmax(-x, -dir);
    auto r = cond ? mmin(width - x - 1, -dir) : 0;
    for (int j = l; j <= r; j++) {
      if ((abs(dmskpn[x + j] - dmskp[x]) > lim && dmskpn[x + j] != peak) || (dmskpn[x + j] == peak && dmskp[x + j] == peak) ||
          (abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
        icb = true;
        break;
      }
    }

    if (icb)
      out = peak;
  }
}

template <typename T> KERNEL void markDirections2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel2x;

  auto mskp = lineOff(msk, -1);
  auto dmskp = lineOff(dmsk, -1);
  auto mskpn = lineOff(msk, 1);
  auto dmskpn = lineOff(dmsk, 1);
  auto &out = point(dst);

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (mskp[x] != peak && mskpn[x] != peak)
    return;

  int val0 = dmskp[x - 1], val1 = dmskp[x], val2 = dmskp[x + 1], val6 = dmskpn[x - 1], val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak, cond6 = val6 != peak, cond7 = val7 != peak, cond8 = val8 != peak;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax,
      cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned v = cond0 + cond1 + cond2 + cond6 + cond7 + cond8;

  if (v < 3)
    return;

  boseSortArray(order);

  const int mid = (v & 1) ? order[v / 2] : (order[(v - 1) / 2] + order[v / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  unsigned u = 0;
  u += (abs(dmskp[x - 1] - dmskpn[x - 1]) <= lim || dmskp[x - 1] == peak || dmskpn[x - 1] == peak);
  u += (abs(dmskp[x] - dmskpn[x]) <= lim || dmskp[x] == peak || dmskpn[x] == peak);
  u += (abs(dmskp[x + 1] - dmskpn[x - 1]) <= lim || dmskp[x + 1] == peak || dmskpn[x + 1] == peak);
  if (u < 2)
    return;

  for (unsigned i = 0; i < 6; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < v - 2 || count < 2)
    return;

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void filterDirMap2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel2x;

  auto mskp = lineOff(msk, -1);
  auto dmskp = line(dmsk);
  auto mskpn = lineOff(msk, 1);
  auto dmskpn = lineOff(dmsk, 2);
  auto dmskpp = lineOff(dmsk, -2);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (mskp[x] != peak && mskpn[x] != peak)
    return;

  // XXX: we are safe because dmsk won't be the first or last plane in pool
  int val0 = dmskpp[x - 1], val1 = dmskpp[x], val2 = dmskpp[x + 1], val3 = dmskp[x - 1], val4 = dmskp[x], val5 = dmskp[x + 1],
      val6 = dmskpn[x - 1], val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak && y > 1, cond1 = val1 != peak && y > 1, cond2 = val2 != peak && y > 1, cond3 = val3 != peak,
       cond4 = val4 != peak, cond5 = val5 != peak, cond6 = val6 != peak && y < height - 2, cond7 = val7 != peak && y < height - 2,
       cond8 = val8 != peak && y < height - 2;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax, cond3 ? val3 : intmax, cond4 ? val4 : intmax,
      cond5 ? val5 : intmax, cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned u = cond0 + cond1 + cond2 + cond3 + cond4 + cond5 + cond6 + cond7 + cond8;

  if (u < 4) {
    out = peak;
    return;
  }

  boseSortArray(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  for (unsigned i = 0; i < 9; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < 4 || (count < 5 && dmskp[x] == peak)) {
    out = peak;
    return;
  }

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void expandDirMap2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel2x;

  auto mskp = lineOff(msk, -1);
  auto dmskp = line(dmsk);
  auto mskpn = lineOff(msk, 1);
  auto dmskpn = lineOff(dmsk, 2);
  auto dmskpp = lineOff(dmsk, -2);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (dmskp[x] != peak || (mskp[x] != peak && mskpn[x] != peak))
    return;

  // XXX: we are safe because dmsk won't be the first or last plane in pool
  int val0 = dmskpp[x - 1], val1 = dmskpp[x], val2 = dmskpp[x + 1], val3 = dmskp[x - 1], val5 = dmskp[x + 1], val6 = dmskpn[x - 1],
      val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak && y > 1, cond1 = val1 != peak && y > 1, cond2 = val2 != peak && y > 1, cond3 = val3 != peak,
       cond5 = val5 != peak, cond6 = val6 != peak && y < height - 2, cond7 = val7 != peak && y < height - 2,
       cond8 = val8 != peak && y < height - 2;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax, cond3 ? val3 : intmax,
      cond5 ? val5 : intmax, cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned u = cond0 + cond1 + cond2 + cond3 + cond5 + cond6 + cond7 + cond8;

  if (u < 5)
    return;

  boseSortArray(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  for (unsigned i = 0; i < 8; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < 5)
    return;

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void fillGaps2X(const EEDI2Param d, const T *msk, const T *dmsk, T *tmp) {
  setup_kernel2x;
  constexpr int fiveHundred = 500 << shift;

  auto mskp = lineOff(msk, -1);
  auto dmskp = line(dmsk);
  auto mskpn = lineOff(msk, 1);
  auto mskpp = lineOff(msk, -3);
  auto mskpnn = lineOff(msk, 3);
  auto dmskpn = lineOff(dmsk, 2);
  auto dmskpp = lineOff(dmsk, -2);
  auto &out_tmp = point(tmp);

  out_tmp = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (dmskp[x] != peak || (mskp[x] != peak && mskpn[x] != peak))
    return;

  unsigned u = x - 1, v = x + 1;
  int back = fiveHundred, forward = -fiveHundred;

  while (u && x - u < 16) {
    if (dmskp[u] != peak) {
      back = dmskp[u];
      break;
    }
    if (mskp[u] != peak && mskpn[u] != peak)
      break;
    u--;
  }

  while (v < width && v - x < 16) {
    if (dmskp[v] != peak) {
      forward = dmskp[v];
      break;
    }
    if (mskp[v] != peak && mskpn[v] != peak)
      break;
    v++;
  }

  bool tc = true, bc = true;
  int mint = fiveHundred, maxt = -twenty;
  int minb = fiveHundred, maxb = -twenty;

  for (unsigned j = u; j <= v && tc; j++) {
    tc = !(y <= 2 || dmskpp[j] == peak || (mskpp[j] != peak && mskp[j] != peak));
    mint = tc ? mmin(mint, dmskpp[j] + 0) : twenty;
    maxt = tc ? mmax(maxt, dmskpp[j] + 0) : twenty;
  }

  for (unsigned j = u; j <= v && bc; j++) {
    bc = !(y >= height - 3 || dmskpn[j] == peak || (mskpn[j] != peak && mskpnn[j] != peak));
    minb = bc ? mmin(minb, dmskpn[j] + 0) : twenty;
    maxb = bc ? mmax(maxb, dmskpn[j] + 0) : twenty;
  }

  if (maxt == -twenty)
    maxt = mint = twenty;
  if (maxb == -twenty)
    maxb = minb = twenty;

  const int thresh = mmax(mmax(abs(forward - neutral), abs(back - neutral)) / 4, eight + 0, abs(mint - maxt), abs(minb - maxb));
  const unsigned lim = mmin(mmax(abs(forward - neutral), abs(back - neutral)) >> shift2, 6);
  if (abs(forward - back) <= thresh && (v - u - 1 <= lim || tc || bc)) {
    out_tmp = (x - u) | (v - x) << 8u;
  }
}

template <typename T> KERNEL void fillGaps2XStep2(const EEDI2Param d, const T *msk, const T *dmsk, const T *tmp, T *dst) {
  setup_kernel2x;

  auto mskp = lineOff(msk, -1);
  auto dmskp = line(dmsk);
  auto mskpn = lineOff(msk, 1);
  auto tmpp = line(tmp);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  unsigned uv = 0, pos;

  for (unsigned i = max(x - 16, 1); i < x; ++i) {
    bool cond = i + (tmpp[i] >> 8) > x;
    uv = cond ? tmpp[i] : uv;
    pos = cond ? i : pos;
  }
  uv = tmpp[x] ? tmpp[x] : uv;
  pos = tmpp[x] ? x : pos;
  for (unsigned i = x + 1; i - x < 16 && i < width - 1; ++i) {
    bool cond = i - (tmpp[i] & 255u) < x;
    uv = cond ? tmpp[i] : uv;
    pos = cond ? i : pos;
  }

  if (!uv)
    return;

  int u = pos - (uv & 255);
  int v = pos + (uv >> 8);
  int back = dmskp[u];
  int forward = dmskp[v];

  out = back + round_div((forward - back) * (x - 1 - u), (v - u));
}

template <typename T> KERNEL void interpolateLattice(const EEDI2Param d, const T *omsk, const T *dmsk, T *dst, T *dmsk_2) {
  setup_kernel2x;

  auto omskp = lineOff(omsk, -1);
  auto omskn = lineOff(omsk, 1);
  auto dmskp = line(dmsk);
  auto dstp = lineOff(dst, -1);
  auto dstpn = line(dst);
  auto dstpnn = lineOff(dst, 1);
  auto &out = point(dst);
  auto &mout = point(dmsk_2);

  bounds_check3(x, 0, width);
  bounds_check3(y, 1, height - 1);

  int dir = dmskp[x];
  const int lim = limlut[abs(dir - neutral) >> shift2] << shift;

  if (dir == peak || (abs(dmskp[x] - dmskp[x - 1]) > lim && abs(dmskp[x] - dmskp[x + 1]) > lim)) {
    out = (dstp[x] + dstpnn[x] + 1) / 2;
    if (dir != peak)
      mout = neutral;
    return;
  }

  if (lim < nine) {
    const unsigned sum = (dstp[x - 1] + dstp[x] + dstp[x + 1] + dstpnn[x - 1] + dstpnn[x] + dstpnn[x + 1]) >> shift;
    const unsigned sumsq = (dstp[x - 1] >> shift) * (dstp[x - 1] >> shift) + (dstp[x] >> shift) * (dstp[x] >> shift) +
                           (dstp[x + 1] >> shift) * (dstp[x + 1] >> shift) + (dstpnn[x - 1] >> shift) * (dstpnn[x - 1] >> shift) +
                           (dstpnn[x] >> shift) * (dstpnn[x] >> shift) + (dstpnn[x + 1] >> shift) * (dstpnn[x + 1] >> shift);
    if (6 * sumsq - sum * sum < 576) {
      out = (dstp[x] + dstpnn[x] + 1) / 2;
      mout = peak;
      return;
    }
  }

  if (x > 1 && x < width - 2 &&
      ((dstp[x] < mmax(dstp[x - 2], dstp[x - 1]) - three && dstp[x] < mmax(dstp[x + 2], dstp[x + 1]) - three &&
        dstpnn[x] < mmax(dstpnn[x - 2], dstpnn[x - 1]) - three && dstpnn[x] < mmax(dstpnn[x + 2], dstpnn[x + 1]) - three) ||
       (dstp[x] > mmin(dstp[x - 2], dstp[x - 1]) + three && dstp[x] > mmin(dstp[x + 2], dstp[x + 1]) + three &&
        dstpnn[x] > mmin(dstpnn[x - 2], dstpnn[x - 1]) + three && dstpnn[x] > mmin(dstpnn[x + 2], dstpnn[x + 1]) + three))) {
    out = (dstp[x] + dstpnn[x] + 1) / 2;
    mout = neutral;
    return;
  }

  dir = (dir - neutral + (1 << (shift2 - 1))) >> shift2;
  const int uStart = (dir - 2 < 0) ? mmax(-x + 1, dir - 2, -width + 2 + x) : mmin(x - 1, dir - 2, width - 2 - x);
  const int uStop = (dir + 2 < 0) ? mmax(-x + 1, dir + 2, -width + 2 + x) : mmin(x - 1, dir + 2, width - 2 - x);
  unsigned min = d.nt8;
  unsigned val = (dstp[x] + dstpnn[x] + 1) / 2;

  for (int u = uStart; u <= uStop; u++) {
    const unsigned diff = abs(dstp[x - 1] - dstpnn[x - u - 1]) + abs(dstp[x] - dstpnn[x - u]) + abs(dstp[x + 1] - dstpnn[x - u + 1]) +
                          abs(dstpnn[x - 1] - dstp[x + u - 1]) + abs(dstpnn[x] - dstp[x + u]) + abs(dstpnn[x + 1] - dstp[x + u + 1]);
    if (diff < min &&
        ((omskp[x - 1 + u] != peak && abs(omskp[x - 1 + u] - dmskp[x]) <= lim) ||
         (omskp[x + u] != peak && abs(omskp[x + u] - dmskp[x]) <= lim) ||
         (omskp[x + 1 + u] != peak && abs(omskp[x + 1 + u] - dmskp[x]) <= lim)) &&
        ((omskn[x - 1 - u] != peak && abs(omskn[x - 1 - u] - dmskp[x]) <= lim) ||
         (omskn[x - u] != peak && abs(omskn[x - u] - dmskp[x]) <= lim) ||
         (omskn[x + 1 - u] != peak && abs(omskn[x + 1 - u] - dmskp[x]) <= lim))) {
      const unsigned diff2 = abs(dstp[x + u / 2 - 1] - dstpnn[x - u / 2 - 1]) + abs(dstp[x + u / 2] - dstpnn[x - u / 2]) +
                             abs(dstp[x + u / 2 + 1] - dstpnn[x - u / 2 + 1]);
      if (diff2 < d.nt4 &&
          (((abs(omskp[x + u / 2] - omskn[x - u / 2]) <= lim || abs(omskp[x + u / 2] - omskn[x - ((u + 1) / 2)]) <= lim) &&
            omskp[x + u / 2] != peak) ||
           ((abs(omskp[x + ((u + 1) / 2)] - omskn[x - u / 2]) <= lim || abs(omskp[x + ((u + 1) / 2)] - omskn[x - ((u + 1) / 2)]) <= lim) &&
            omskp[x + ((u + 1) / 2)] != peak))) {
        if ((abs(dmskp[x] - omskp[x + u / 2]) <= lim || abs(dmskp[x] - omskp[x + ((u + 1) / 2)]) <= lim) &&
            (abs(dmskp[x] - omskn[x - u / 2]) <= lim || abs(dmskp[x] - omskn[x - ((u + 1) / 2)]) <= lim)) {
          val = (dstp[x + u / 2] + dstp[x + ((u + 1) / 2)] + dstpnn[x - u / 2] + dstpnn[x - ((u + 1) / 2)] + 2) / 4;
          min = diff;
          dir = u;
        }
      }
    }
  }

  if (min != d.nt8) {
    out = val;
    mout = neutral + (dir << shift2);
  } else {
    const int dt = 4 >> d.subSampling;
    const int uStart2 = mmax(-x + 1, -dt);
    const int uStop2 = mmin(width - 2 - x, dt);
    const unsigned minm = mmin(dstp[x], dstpnn[x]);
    const unsigned maxm = mmax(dstp[x], dstpnn[x]);
    min = d.nt7;

    for (int u = uStart2; u <= uStop2; u++) {
      const int p1 = dstp[x + u / 2] + dstp[x + ((u + 1) / 2)];
      const int p2 = dstpnn[x - u / 2] + dstpnn[x - ((u + 1) / 2)];
      const unsigned diff = abs(dstp[x - 1] - dstpnn[x - u - 1]) + abs(dstp[x] - dstpnn[x - u]) + abs(dstp[x + 1] - dstpnn[x - u + 1]) +
                            abs(dstpnn[x - 1] - dstp[x + u - 1]) + abs(dstpnn[x] - dstp[x + u]) + abs(dstpnn[x + 1] - dstp[x + u + 1]) +
                            abs(p1 - p2);
      if (diff < min) {
        const unsigned valt = (p1 + p2 + 2) / 4;
        if (valt >= minm && valt <= maxm) {
          val = valt;
          min = diff;
          dir = u;
        }
      }
    }

    out = val;
    mout = (min == d.nt7) ? neutral : neutral + (dir << shift2);
  }
}

template <typename T> KERNEL void postProcess(const EEDI2Param d, const T *nmsk, const T *omsk, T *dst) {
  setup_kernel2x;

  auto nmskp = line(nmsk);
  auto omskp = line(omsk);
  auto &out = point(dst);
  auto dstpp = lineOff(dst, -1);
  auto dstpn = lineOff(dst, 1);

  bounds_check3(x, 0, width);
  bounds_check3(y, 1, height - 1);

  const int lim = limlut[abs(nmskp[x] - neutral) >> shift2] << shift;
  if (abs(nmskp[x] - omskp[x]) > lim && omskp[x] != peak && omskp[x] != neutral)
    out = (dstpp[x] + dstpn[x] + 1) / 2;
}

template <typename T> class EEDI2Pass final : public Pass<T> {
  EEDI2Param d;

  T *dst, *msk, *src, *tmp;
  T *dst2, *dst2M, *tmp2, *tmp2_2, *tmp2_3, *msk2;

  unsigned map, pp, fieldS;
  unsigned d_pitch;

public:
  EEDI2Pass(const EEDI2Pass &other) : Pass<T>(other), d(other.d), map(other.map), pp(other.pp), fieldS(other.fieldS) { initCuda(); }

  EEDI2Pass(VideoDimension vi, VideoDimension vi2, const EEDI2Param &d, unsigned map, unsigned pp, unsigned fieldS)
      : Pass<T>(vi, vi2), d(d), map(map), pp(pp), fieldS(fieldS) {
    initCuda();
  }

  [[nodiscard]] Pass<T> *dup() const override { return new EEDI2Pass(*this); }

  ~EEDI2Pass() override {
    try_cuda(cudaFree(dst));
    try_cuda(cudaFree(dst2));
  }

private:
  void initCuda() {
    constexpr size_t numMem = 4;
    constexpr size_t numMem2x = 6;
    T **mem = &dst;
    size_t pitch;
    auto width = this->vi.width;
    auto height = this->vi.height;
    auto height2x = height * 2;
    try_cuda(cudaMallocPitch(&mem[0], &pitch, width * sizeof(T), height * numMem));
    narrow_cast_to(d_pitch, pitch);
    for (size_t i = 1; i < numMem; ++i)
      mem[i] = reinterpret_cast<T *>(reinterpret_cast<char *>(mem[i - 1]) + d_pitch * height);

    if (map == 0 || map == 3) {
      try_cuda(cudaMalloc(&dst2, d_pitch * height * numMem2x * 2));
      mem = &dst2;
      for (size_t i = 1; i < numMem2x; ++i)
        mem[i] = reinterpret_cast<T *>(reinterpret_cast<char *>(mem[i - 1]) + d_pitch * height2x);
    } else {
      dst2 = nullptr;
    }
  }

public:
  T *getSrcDevPtr() override { return src; }
  unsigned getSrcPitch() override { return d_pitch; }
  const T *getDstDevPtr() const override {
    switch (map) {
    case 0:
      return dst2;
    case 1:
      return msk;
    case 3:
      return tmp2;
    default:
      return dst;
    }
  }
  unsigned getDstPitch() override { return d_pitch; }

  void process(int n, int plane, cudaStream_t stream) override {
    auto field = fieldS;
    if (field > 1)
      field = (n & 1) ? (field == 2 ? 1 : 0) : (field == 2 ? 0 : 1);

    auto subSampling = plane ? this->vi.subSampling : 0u;

    auto width = this->vi.width >> subSampling;
    auto height = this->vi.height >> subSampling;
    auto height2x = height * 2;
    auto width_bytes = width * sizeof(T);
    auto d_pitch = this->d_pitch >> subSampling;

    d.field = field;
    d.width = width;
    d.height = height;
    d.subSampling = subSampling;
    d.d_pitch = d_pitch;

    dim3 blocks = dim3(64, 1);
    dim3 grids = dim3((width - 1) / blocks.x + 1, (height - 1) / blocks.y + 1);

    buildEdgeMask<<<grids, blocks, 0, stream>>>(d, src, msk);
    erode<<<grids, blocks, 0, stream>>>(d, msk, tmp);
    dilate<<<grids, blocks, 0, stream>>>(d, tmp, msk);
    erode<<<grids, blocks, 0, stream>>>(d, msk, tmp);
    removeSmallHorzGaps<<<grids, blocks, 0, stream>>>(d, tmp, msk);

    if (map != 1) {
      calcDirections<<<grids, blocks, 0, stream>>>(d, src, msk, tmp);
      filterDirMap<<<grids, blocks, 0, stream>>>(d, msk, tmp, dst);
      expandDirMap<<<grids, blocks, 0, stream>>>(d, msk, dst, tmp);
      filterMap<<<grids, blocks, 0, stream>>>(d, msk, tmp, dst);

      if (map != 2) {
        auto upscaleBy2 = [&](const T *src, T *dst) {
          try_cuda(cudaMemcpy2DAsync(dst + d_pitch * (1 - field), d_pitch * 2, src, d_pitch, width_bytes, height, cudaMemcpyDeviceToDevice,
                                     stream));
        };
        try_cuda(cudaMemset2DAsync(dst2, d_pitch, 0, width_bytes, height2x, stream));
        try_cuda(cudaMemset2DAsync(tmp2, d_pitch, 255, width_bytes, height2x, stream));
        upscaleBy2(src, dst2);
        upscaleBy2(dst, tmp2_2);
        upscaleBy2(msk, msk2);

        markDirections2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2_2, tmp2);
        filterDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2, dst2M);
        expandDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2);
        fillGaps2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2, tmp2_3);
        fillGaps2XStep2<<<grids, blocks, 0, stream>>>(d, msk2, tmp2, tmp2_3, dst2M);
        fillGaps2X<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2_3);
        fillGaps2XStep2<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2_3, tmp2);

        if (map != 3) {
          if (field)
            try_cuda(cudaMemcpyAsync(dst2 + d_pitch / sizeof(T) * (height2x - 1), dst2 + d_pitch / sizeof(T) * (height2x - 2), width_bytes,
                                     cudaMemcpyDeviceToDevice, stream));
          else
            try_cuda(cudaMemcpyAsync(dst2, dst2 + d_pitch / sizeof(T), width_bytes, cudaMemcpyDeviceToDevice, stream));
          try_cuda(cudaMemcpy2DAsync(tmp2_3, d_pitch, tmp2, d_pitch, width_bytes, height2x, cudaMemcpyDeviceToDevice, stream));

          interpolateLattice<<<grids, blocks, 0, stream>>>(d, tmp2_2, tmp2, dst2, tmp2_3);

          if (pp == 1) {
            filterDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2_3, dst2M);
            expandDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2);
            postProcess<<<grids, blocks, 0, stream>>>(d, tmp2, tmp2_3, dst2);
          }
        }
      }
    }
  }
};

#undef KERNEL
#undef setup_kernel
#undef setup_kernel2x
#undef bounds_check3
