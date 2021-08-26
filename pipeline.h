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

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "eedi2.cuh"
#include "utils.cuh"

static inline void bitblt(void *dstp, int dst_stride, const void *srcp, int src_stride, size_t row_size, size_t height) {
  // Copyright (C) 2012-2015 Fredrik Mellbin
  if (height) {
    if (src_stride == dst_stride && src_stride == (int)row_size) {
      memcpy(dstp, srcp, row_size * height);
    } else {
      const uint8_t *srcp8 = (const uint8_t *)srcp;
      uint8_t *dstp8 = (uint8_t *)dstp;
      size_t i;
      for (i = 0; i < height; i++) {
        memcpy(dstp8, srcp8, row_size);
        srcp8 += src_stride;
        dstp8 += dst_stride;
      }
    }
  }
}

struct PropsMap : public std::multimap<std::string_view, int64_t> {
  using std::multimap<std::string_view, int64_t>::multimap;

  std::optional<mapped_type> get(const key_type &key, const size_type idx = static_cast<size_type>(-1)) const {
    auto casual = idx == static_cast<size_type>(-1);
    auto [bg, ed] = casual ? std::make_pair(find(key), end()) : equal_range(key);
    if (bg == ed)
      return std::nullopt;
    if (!casual)
      for (size_type i = 0; i < idx; ++i)
        if (++bg == ed)
          return std::nullopt;
    return std::make_optional(bg->second);
  }
};

template <typename T> class BasePipeline {
  std::vector<std::unique_ptr<Pass<T>>> passes;
  VideoDimension ivd;
  int device_id;
  cudaStream_t stream;
  T *h_src, *h_dst;
  std::vector<T *> fbs;
  unsigned plane_mask = 0;

protected:
  VideoDimension getOVD() const { return passes.back()->getOVD(); }

  BasePipeline(std::string_view filterName, const PropsMap &props, VideoDimension vd) : ivd(vd) {
    using invalid_arg = std::invalid_argument;

    auto vd2 = vd;
    EEDI2Param d;
    unsigned map, pp, fieldS;

    if (vd.width < 8 || vd.height < 7)
      throw invalid_arg("clip resolution too low");

    if (filterName == "EEDI2")
      numeric_cast_to(fieldS, props.get("field").value());
    else
      fieldS = 1;

    numeric_cast_to(d.mthresh, props.get("mthresh").value_or(10));
    numeric_cast_to(d.lthresh, props.get("lthresh").value_or(20));
    numeric_cast_to(d.vthresh, props.get("vthresh").value_or(20));

    numeric_cast_to(d.estr, props.get("estr").value_or(2));
    numeric_cast_to(d.dstr, props.get("dstr").value_or(4));
    numeric_cast_to(d.maxd, props.get("maxd").value_or(24));

    numeric_cast_to(map, props.get("map").value_or(0));
    numeric_cast_to(pp, props.get("pp").value_or(1));

    unsigned nt;
    numeric_cast_to(nt, props.get("nt").value_or(50));

    numeric_cast_to(device_id, props.get("device_id").value_or(-1));

    if (fieldS > 3)
      throw invalid_arg("field must be 0, 1, 2 or 3");
    if (d.maxd < 1 || d.maxd > 29)
      throw invalid_arg("maxd must be between 1 and 29 (inclusive)");
    if (map > 3)
      throw invalid_arg("map must be 0, 1, 2 or 3");
    if (pp > 1)
      throw invalid_arg("only pp=0 or 1 is implemented");

    if (map == 0 || map == 3)
      vd2.height *= 2;

    d.mthresh *= d.mthresh;
    d.vthresh *= 81;

    nt <<= sizeof(T) * 8 - 8;
    d.nt4 = nt * 4;
    d.nt7 = nt * 7;
    d.nt8 = nt * 8;
    d.nt13 = nt * 13;
    d.nt19 = nt * 19;

    passes.emplace_back(new EEDI2Pass<T>(vd, vd2, d, map, pp, fieldS));

    if (filterName != "EEDI2") {
      auto vd3 = vd2;
      std::swap(vd3.width, vd3.height); // XXX: this is correct for 420 & 444 only
      passes.emplace_back(new TransposePass<T>(vd2, vd3));
      auto vd4 = vd3;
      if (filterName == "AA2") {
        vd4.width /= 2;
        passes.emplace_back(new ScaleDownWPass<T>(vd3, vd4));
      } else {
        passes.emplace_back(new ShiftWPass<T>(vd3, vd4));
      }
      auto vd5 = vd4;
      vd5.height *= 2;
      passes.emplace_back(new EEDI2Pass<T>(vd4, vd5, d, map, pp, fieldS));
      auto vd6 = vd5;
      std::swap(vd6.width, vd6.height);
      passes.emplace_back(new TransposePass<T>(vd5, vd6));
      auto vd7 = vd6;
      if (filterName == "AA2") {
        vd7.width /= 2;
        passes.emplace_back(new ScaleDownWPass<T>(vd6, vd7));
      } else {
        passes.emplace_back(new ShiftWPass<T>(vd6, vd7));
      }
    }

    PropsMap::size_type i = 0;
    try {
      for (;; ++i) {
        auto plane = props.get("planes", i).value();
        plane_mask |= 1 << plane;
      }
    } catch (const std::bad_optional_access &) {
      if (i == 0)
        plane_mask = 7;
    }

    passes.shrink_to_fit();

    initCuda();
  }

  BasePipeline(const BasePipeline &other) : ivd(other.ivd), device_id(other.device_id), plane_mask(other.plane_mask) {
    passes.reserve(other.passes.size());
    for (const auto &step : other.passes)
      passes.emplace_back(step->dup());

    initCuda();
  }

public:
  ~BasePipeline() {
    try_cuda(cudaFreeHost(h_src));
    try_cuda(cudaFreeHost(h_dst));
    for (auto fb : fbs)
      try_cuda(cudaFree(fb));
  }

private:
  void initCuda() {
    try {
      try_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    } catch (const CUDAError &exc) {
      throw CUDAError(exc.what() + " Please upgrade your driver."s);
    }

    if (auto &firstStep = *passes.front(); !firstStep.getSrcDevPtr()) {
      size_t pitch;
      T *fb_d_src;
      try_cuda(cudaMallocPitch(&fb_d_src, &pitch, ivd.width * sizeof(T), ivd.height));
      firstStep.setSrcDevPtr(fb_d_src);
      firstStep.setSrcPitch(static_cast<unsigned>(pitch));
      fbs.push_back(fb_d_src);
    }

    if (auto &lastStep = *passes.back(); !lastStep.getDstDevPtr()) {
      auto ovd = getOVD();
      size_t pitch;
      T *fb_d_dst;
      try_cuda(cudaMallocPitch(&fb_d_dst, &pitch, ovd.width * sizeof(T), ovd.height));
      lastStep.setDstDevPtr(fb_d_dst);
      lastStep.setDstPitch(static_cast<unsigned>(pitch));
      fbs.push_back(fb_d_dst);
    }

    auto d_pitch_src = passes.front()->getSrcPitch();
    auto d_pitch_dst = passes.back()->getDstPitch();
    auto src_height = ivd.height;
    auto dst_height = getOVD().height;
    try_cuda(cudaHostAlloc(&h_src, d_pitch_src * src_height, cudaHostAllocWriteCombined));
    try_cuda(cudaHostAlloc(&h_dst, d_pitch_dst * dst_height, cudaHostAllocDefault));
  }

protected:
  void processPlane(int n, int plane, int src_width, int src_height, int dst_width, int dst_height, int s_pitch_src, int s_pitch_dst,
                    const void *s_src, void *s_dst) {
    auto src_width_bytes = src_width * sizeof(T);
    auto dst_width_bytes = dst_width * sizeof(T);
    auto d_src = passes.front()->getSrcDevPtr();
    auto d_dst = passes.back()->getDstDevPtr();
    auto d_pitch_src = passes.front()->getSrcPitch() >> !!plane * ivd.subSampling;
    auto d_pitch_dst = passes.back()->getDstPitch() >> !!plane * getOVD().subSampling;

    if (!((1u << plane) & plane_mask))
      return;

    // upload
    bitblt(h_src, d_pitch_src, s_src, s_pitch_src, src_width_bytes, src_height);
    try_cuda(cudaMemcpy2DAsync(d_src, d_pitch_src, h_src, d_pitch_src, src_width_bytes, src_height, cudaMemcpyHostToDevice, stream));

    // process
    for (unsigned i = 0; i < passes.size(); ++i) {
      auto &cur = *passes[i];
      if (i) {
        auto &last = *passes[i - 1];
        auto &next = *passes[i + 1];
        auto last_vi = last.getOVD();
        auto ss = !!plane * last_vi.subSampling;
        if (!cur.getSrcDevPtr()) {
          cur.setSrcDevPtr(const_cast<T *>(last.getDstDevPtr()));
          cur.setSrcPitch(last.getDstPitch());
        }
        if (!cur.getDstDevPtr()) {
          cur.setDstDevPtr(next.getSrcDevPtr());
          cur.setDstPitch(next.getSrcPitch());
        }
        if (!cur.getDstDevPtr()) {
          auto vi = cur.getOVD();
          size_t pitch;
          T *fb;
          try_cuda(cudaMallocPitch(&fb, &pitch, vi.width * sizeof(T), vi.height));
          cur.setDstDevPtr(fb);
          next.setSrcDevPtr(fb);
          cur.setDstPitch(static_cast<unsigned>(pitch));
          next.setSrcPitch(static_cast<unsigned>(pitch));
          fbs.push_back(fb);
        }
        auto curPtr = cur.getSrcDevPtr();
        auto lastPtr = last.getDstDevPtr();
        if (curPtr != lastPtr)
          try_cuda(cudaMemcpy2DAsync(curPtr, cur.getSrcPitch() >> ss, lastPtr, last.getDstPitch() >> ss, last_vi.width * sizeof(T) >> ss,
                                     last_vi.height >> ss, cudaMemcpyDeviceToDevice, stream));
      }
      cur.process(n, plane, stream);
    }

    // download
    try_cuda(cudaMemcpy2DAsync(h_dst, d_pitch_dst, d_dst, d_pitch_dst, dst_width_bytes, dst_height, cudaMemcpyDeviceToHost, stream));
    try_cuda(cudaStreamSynchronize(stream));
    bitblt(s_dst, s_pitch_dst, h_dst, d_pitch_dst, dst_width_bytes, dst_height);
  }

  void prepare() {
    if (device_id != -1)
      try_cuda(cudaSetDevice(device_id));
  }

  unsigned getPlaneBypassMask() const {
    const auto vi2 = getOVD();
    if (ivd != vi2)
      return 0;
    else
      return ~plane_mask & 7;
  }
};
