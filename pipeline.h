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
  VideoInfo vi;
  int device_id;
  cudaStream_t stream;
  T *h_src, *h_dst;
  std::vector<T *> fbs;

protected:
  VideoInfo getOutputVI() const { return passes.back()->getOutputVI(); }

  BasePipeline(std::string_view filterName, const PropsMap &props, VideoInfo vi) : vi(vi) {
    using invalid_arg = std::invalid_argument;

    auto vi2 = vi;
    EEDI2Param d;
    unsigned map, pp, fieldS;

    if (vi.width < 8 || vi.height < 7)
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
      vi2.height *= 2;

    d.mthresh *= d.mthresh;
    d.vthresh *= 81;

    nt <<= sizeof(T) * 8 - 8;
    d.nt4 = nt * 4;
    d.nt7 = nt * 7;
    d.nt8 = nt * 8;
    d.nt13 = nt * 13;
    d.nt19 = nt * 19;

    passes.emplace_back(new EEDI2Pass<T>(vi, vi2, d, map, pp, fieldS));

    if (filterName != "EEDI2") {
      auto vi3 = vi2;
      std::swap(vi3.width, vi3.height); // XXX: this is correct for 420 & 444 only
      passes.emplace_back(new TransposePass<T>(vi2, vi3));
      auto vi4 = vi3;
      if (filterName == "AA2") {
        vi4.width /= 2;
        passes.emplace_back(new ScaleDownWPass<T>(vi3, vi4));
      } else {
        passes.emplace_back(new ShiftWPass<T>(vi3, vi4));
      }
      auto vi5 = vi4;
      vi5.height *= 2;
      passes.emplace_back(new EEDI2Pass<T>(vi4, vi5, d, map, pp, fieldS));
      auto vi6 = vi5;
      std::swap(vi6.width, vi6.height);
      passes.emplace_back(new TransposePass<T>(vi5, vi6));
      auto vi7 = vi6;
      if (filterName == "AA2") {
        vi7.width /= 2;
        passes.emplace_back(new ScaleDownWPass<T>(vi6, vi7));
      } else {
        passes.emplace_back(new ShiftWPass<T>(vi6, vi7));
      }
    }

    passes.shrink_to_fit();

    initCuda();
  }

  BasePipeline(const BasePipeline &other) : vi(other.vi), device_id(other.device_id) {
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
      try_cuda(cudaMallocPitch(&fb_d_src, &pitch, vi.width * sizeof(T), vi.height));
      firstStep.setSrcDevPtr(fb_d_src);
      firstStep.setSrcPitch(static_cast<unsigned>(pitch));
      fbs.push_back(fb_d_src);
    }

    if (auto &lastStep = *passes.back(); !lastStep.getDstDevPtr()) {
      auto vi2 = lastStep.getOutputVI();
      size_t pitch;
      T *fb_d_dst;
      try_cuda(cudaMallocPitch(&fb_d_dst, &pitch, vi2.width * sizeof(T), vi2.height));
      lastStep.setDstDevPtr(fb_d_dst);
      lastStep.setDstPitch(static_cast<unsigned>(pitch));
      fbs.push_back(fb_d_dst);
    }

    auto d_pitch_src = passes.front()->getSrcPitch();
    auto d_pitch_dst = passes.back()->getDstPitch();
    auto src_height = vi.height;
    auto dst_height = passes.back()->getOutputVI().height;
    try_cuda(cudaHostAlloc(&h_src, d_pitch_src * src_height, cudaHostAllocWriteCombined));
    try_cuda(cudaHostAlloc(&h_dst, d_pitch_dst * dst_height, cudaHostAllocDefault));
  }

protected:
  void getPlane(int n, int plane, int src_width, int src_height, int dst_width, int dst_height, int s_pitch_src, int s_pitch_dst,
                const void *s_src, void *s_dst) {
    auto src_width_bytes = src_width * sizeof(T);
    auto dst_width_bytes = dst_width * sizeof(T);
    auto d_src = passes.front()->getSrcDevPtr();
    auto d_dst = passes.back()->getDstDevPtr();
    auto d_pitch_src = passes.front()->getSrcPitch() >> !!plane * vi.subSampling;
    auto d_pitch_dst = passes.back()->getDstPitch() >> !!plane * getOutputVI().subSampling;

    // upload
    vs_bitblt(h_src, d_pitch_src, s_src, s_pitch_src, src_width_bytes, src_height);
    try_cuda(cudaMemcpy2DAsync(d_src, d_pitch_src, h_src, d_pitch_src, src_width_bytes, src_height, cudaMemcpyHostToDevice, stream));

    // process
    for (unsigned i = 0; i < passes.size(); ++i) {
      auto &cur = *passes[i];
      if (i) {
        auto &last = *passes[i - 1];
        auto &next = *passes[i + 1];
        auto last_vi = last.getOutputVI();
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
          auto vi = cur.getOutputVI();
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
    vs_bitblt(s_dst, s_pitch_dst, h_dst, d_pitch_dst, dst_width_bytes, dst_height);
  }

  void prepare() {
    if (device_id != -1)
      try_cuda(cudaSetDevice(device_id));
  }
};
