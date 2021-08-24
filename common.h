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

#include <stdexcept>
#include <string>

#include <assert.h>

#include <boost/numeric/conversion/cast.hpp>

using namespace std::literals::string_literals;

#define setup_xy int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y
#define line(p) ((p) + (pitch / sizeof(T)) * y)
#define lineOff(p, off) ((p) + int(pitch / sizeof(T)) * (y + (off)))
#define point(p) ((p)[(pitch / sizeof(T)) * y + x])

struct VideoInfo {
  int width, height, subSampling;
};

class CUDAError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
#define try_cuda(expr)                                                                                                                     \
  do {                                                                                                                                     \
    cudaError_t __err = expr;                                                                                                              \
    if (__err != cudaSuccess) {                                                                                                            \
      throw CUDAError("'"s + #expr + " failed: " + cudaGetErrorString(__err));                                                             \
    }                                                                                                                                      \
  } while (0)

static [[noreturn]] void unreachable() { assert(false); }

template <typename Td, typename Ts> void numeric_cast_to(Td &dst, Ts src) { dst = boost::numeric_cast<Td>(src); }
template <typename Td, typename Ts> void narrow_cast_to(Td &dst, Ts src) { dst = static_cast<Td>(src); }

template <typename T> struct Pass {
  virtual ~Pass() = default;
  virtual T *getSrcDevPtr() { throw std::logic_error("not implemented"); }
  virtual unsigned getSrcPitch() { throw std::logic_error("not implemented"); }
  virtual const T *getDstDevPtr() const { throw std::logic_error("not implemented"); }
  virtual unsigned getDstPitch() { throw std::logic_error("not implemented"); }
  virtual void process(int n, int plane, cudaStream_t stream) = 0;
  [[nodiscard]] virtual Pass *dup() const = 0;

  virtual void setSrcDevPtr(T *) { throw std::logic_error("this variable is readonly"); }
  virtual void setSrcPitch(unsigned) { throw std::logic_error("this variable is readonly"); }
  virtual void setDstDevPtr(T *) { throw std::logic_error("this variable is readonly"); };
  virtual void setDstPitch(unsigned) { throw std::logic_error("this variable is readonly"); }

  VideoInfo getOutputVI() const { return vi2; };

  Pass(VideoInfo vi, VideoInfo vi2) : vi(vi), vi2(vi2) {}

protected:
  VideoInfo vi, vi2;
};

template <typename T> struct BridgePass : public Pass<T> {
  using Pass<T>::Pass;

  BridgePass(const BridgePass &other) : Pass<T>(other) {}

protected:
  T *src = nullptr, *dst = nullptr;
  unsigned d_pitch_src, d_pitch_dst;

public:
  void setSrcDevPtr(T *p) final { src = p; }
  void setSrcPitch(unsigned p) final { d_pitch_src = p; }
  void setDstDevPtr(T *p) final { dst = p; }
  void setDstPitch(unsigned p) final { d_pitch_dst = p; }
  T *getSrcDevPtr() final { return src; }
  unsigned getSrcPitch() final { return d_pitch_src; }
  const T *getDstDevPtr() const final { return dst; }
  unsigned getDstPitch() final { return d_pitch_dst; }
};
