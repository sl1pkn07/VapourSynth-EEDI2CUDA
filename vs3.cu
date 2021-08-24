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

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/sync/semaphore.hpp>

#include <VSHelper.h>
#include <VapourSynth.h>

#include "config.h"

#include "eedi2.cuh"
#include "utils.cuh"

using namespace std::literals::string_literals;

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
protected:
  std::vector<std::unique_ptr<Pass<T>>> passes;
  VideoInfo vi;
  int device_id;
  cudaStream_t stream;
  T *h_src, *h_dst;
  std::vector<T *> fbs;

protected:
  VideoInfo getOutputVI() const { return passes.back()->getOutputVI(); }

public:
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
};

VideoInfo get_vi(const VSMap *in, const VSAPI *vsapi) {
  auto node = vsapi->propGetNode(in, "clip", 0, nullptr);
  auto vi = vsapi->getVideoInfo(node);
  vsapi->freeNode(node);
  VideoInfo vvi{vi->width, vi->height, vi->format->subSamplingW};
  return vvi;
}

PropsMap mapize(const VSMap *in, const VSAPI *vsapi) {
  PropsMap m;
  for (auto i = 0, num_keys = vsapi->propNumKeys(in); i < num_keys; ++i) {
    auto key = vsapi->propGetKey(in, i);
    if (vsapi->propGetType(in, key) != ptInt)
      continue;
    auto num_el = vsapi->propNumElements(in, key);
    for (auto j = 0; j < num_el; ++j) {
      auto val = vsapi->propGetInt(in, key, j, nullptr);
      m.emplace(key, val);
    }
  }
  return m;
}

template <typename T> class Pipeline : public BasePipeline<T> {
  std::unique_ptr<VSNodeRef, void(VS_CC *const)(VSNodeRef *)> node;
  VSVideoInfo vi2;

public:
  Pipeline(std::string_view filterName, const VSMap *in, const VSAPI *vsapi)
      : BasePipeline<T>(filterName, mapize(in, vsapi), get_vi(in, vsapi)),
        node(vsapi->propGetNode(in, "clip", 0, nullptr), vsapi->freeNode) {
    vi2 = *vsapi->getVideoInfo(node.get());
    auto ovi = BasePipeline<T>::getOutputVI();
    vi2.width = ovi.width;
    vi2.height = ovi.height;
  }

  Pipeline(const Pipeline &other, const VSAPI *vsapi)
      : BasePipeline<T>(other), node(vsapi->cloneNodeRef(other.node.get()), vsapi->freeNode), vi2(other.vi2) {}

  const VSVideoInfo &getOutputVI() const { return vi2; }

  VSFrameRef *getFrame(int n, int activationReason, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    if (activationReason == arInitial) {
      vsapi->requestFrameFilter(n, node.get(), frameCtx);
      return nullptr;
    } else if (activationReason != arAllFramesReady)
      return nullptr;

    if (device_id != -1)
      try_cuda(cudaSetDevice(device_id));

    std::unique_ptr<const VSFrameRef, void(VS_CC *const)(const VSFrameRef *)> src_frame{vsapi->getFrameFilter(n, node.get(), frameCtx),
                                                                                        vsapi->freeFrame};
    std::unique_ptr<VSFrameRef, void(VS_CC *const)(const VSFrameRef *)> dst_frame{
        vsapi->newVideoFrame(vi2.format, vi2.width, vi2.height, src_frame.get(), core), vsapi->freeFrame};

    for (int plane = 0; plane < vi2.format->numPlanes; ++plane) {
      auto src_width = vsapi->getFrameWidth(src_frame.get(), plane);
      auto src_height = vsapi->getFrameHeight(src_frame.get(), plane);
      auto dst_width = vsapi->getFrameWidth(dst_frame.get(), plane);
      auto dst_height = vsapi->getFrameHeight(dst_frame.get(), plane);
      auto s_pitch_src = vsapi->getStride(src_frame.get(), plane);
      auto s_pitch_dst = vsapi->getStride(dst_frame.get(), plane);
      auto src_width_bytes = src_width * sizeof(T);
      auto dst_width_bytes = dst_width * sizeof(T);
      auto s_src = vsapi->getReadPtr(src_frame.get(), plane);
      auto s_dst = vsapi->getWritePtr(dst_frame.get(), plane);
      auto d_src = passes.front()->getSrcDevPtr();
      auto d_dst = passes.back()->getDstDevPtr();
      auto d_pitch_src = passes.front()->getSrcPitch() >> !!plane * vi2.format->subSamplingW;
      auto d_pitch_dst = passes.back()->getDstPitch() >> !!plane * vi2.format->subSamplingW;

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

    return dst_frame.release();
  }
};

template <typename T> class Instance {
  using Item = std::pair<Pipeline<T>, std::atomic_flag>;
  boost::sync::semaphore semaphore;

  inline Item *items() noexcept { return reinterpret_cast<Item *>(reinterpret_cast<unsigned *>(this + 1) + 1); }
  inline unsigned num_streams() const noexcept { return *reinterpret_cast<const unsigned *>(this + 1); }

public:
  Instance(std::string_view filterName, const VSMap *in, const VSAPI *vsapi) : semaphore(num_streams()) {
    auto items = this->items();
    new (items) Item(std::piecewise_construct, std::forward_as_tuple(filterName, in, vsapi), std::forward_as_tuple());
    items[0].second.clear();
    for (unsigned i = 1; i < num_streams(); ++i) {
      new (items + i) Item(std::piecewise_construct, std::forward_as_tuple(firstReactor(), vsapi), std::forward_as_tuple());
      items[i].second.clear();
    }
  }

  ~Instance() {
    auto items = this->items();
    for (unsigned i = 0; i < num_streams(); ++i)
      items[i].~Item();
  }

  Pipeline<T> &firstReactor() { return items()[0].first; }

  Pipeline<T> &acquireReactor() {
    if (num_streams() == 1)
      return firstReactor();
    semaphore.wait();
    auto items = this->items();
    for (unsigned i = 0; i < num_streams(); ++i) {
      if (!items[i].second.test_and_set())
        return items[i].first;
    }
    unreachable();
  }

  void releaseReactor(const Pipeline<T> &instance) {
    if (num_streams() == 1)
      return;
    auto items = this->items();
    for (unsigned i = 0; i < num_streams(); ++i) {
      if (&instance == &items[i].first) {
        items[i].second.clear();
        break;
      }
    }
    semaphore.post();
  }

  static void *operator new(size_t sz, unsigned num_streams) {
    auto p = static_cast<Instance *>(::operator new(sz + sizeof(unsigned) + sizeof(Item) * num_streams));
    *reinterpret_cast<unsigned *>(p + 1) = num_streams;
    return p;
  }

  static void operator delete(void *p, unsigned) { ::operator delete(p); }

  static void operator delete(void *p) { ::operator delete(p); }
};

template <typename T> void VS_CC eedi2Init(VSMap *, VSMap *, void **instanceData, VSNode *node, VSCore *, const VSAPI *vsapi) {
  auto data = static_cast<Instance<T> *>(*instanceData);
  vsapi->setVideoInfo(&data->firstReactor().getOutputVI(), 1, node);
}

template <typename T>
const VSFrameRef *VS_CC eedi2GetFrame(int n, int activationReason, void **instanceData, void **, VSFrameContext *frameCtx, VSCore *core,
                                      const VSAPI *vsapi) {

  auto data = static_cast<Instance<T> *>(*instanceData);
  const VSFrameRef *out = nullptr;

  if (activationReason == arInitial) {
    out = data->firstReactor().getFrame(n, activationReason, frameCtx, core, vsapi);
  } else {
    auto &d = data->acquireReactor();
    try {
      out = d.getFrame(n, activationReason, frameCtx, core, vsapi);
    } catch (const std::exception &exc) {
      vsapi->setFilterError(("EEDI2CUDA: "s + exc.what()).c_str(), frameCtx);
    }
    data->releaseReactor(d);
  }

  return out;
}

template <typename T> void VS_CC eedi2Free(void *instanceData, VSCore *, const VSAPI *) {
  auto data = static_cast<Instance<T> *>(instanceData);
  delete data;
}

template <typename T> void eedi2CreateInner(std::string_view filterName, const VSMap *in, VSMap *out, const VSAPI *vsapi, VSCore *core) {
  try {
    int err;
    unsigned num_streams;
    numeric_cast_to(num_streams, vsapi->propGetInt(in, "num_streams", 0, &err));
    if (err)
      num_streams = 1;
    auto data = new (num_streams) Instance<T>(filterName, in, vsapi);
    vsapi->createFilter(in, out, filterName.data(), eedi2Init<T>, eedi2GetFrame<T>, eedi2Free<T>,
                        num_streams > 1 ? fmParallel : fmParallelRequests, 0, data, core);
  } catch (const std::exception &exc) {
    vsapi->setError(out, ("EEDI2CUDA: "s + exc.what()).c_str());
    return;
  }
}

VS_CC void eedi2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
  std::string_view filterName{static_cast<const char *>(userData)};
  VSNodeRef *node = vsapi->propGetNode(in, "clip", 0, nullptr);
  const VSVideoInfo *vi = vsapi->getVideoInfo(node);
  vsapi->freeNode(node);
  if (vi->format->bytesPerSample == 1)
    eedi2CreateInner<uint8_t>(filterName, in, out, vsapi, core);
  else
    eedi2CreateInner<uint16_t>(filterName, in, out, vsapi, core);
}

void VS_CC BuildConfigCreate(const VSMap *, VSMap *out, void *, VSCore *, const VSAPI *vsapi) {
  vsapi->propSetData(out, "version", VERSION, -1, paAppend);
  vsapi->propSetData(out, "options", BUILD_OPTIONS, -1, paAppend);
  vsapi->propSetData(out, "timestamp", CONFIGURE_TIME, -1, paAppend);
  vsapi->propSetInt(out, "vsapi_version", VAPOURSYNTH_API_VERSION, paAppend);
}

#define eedi2_common_params                                                                                                                \
  "mthresh:int:opt;"                                                                                                                       \
  "lthresh:int:opt;"                                                                                                                       \
  "vthresh:int:opt;"                                                                                                                       \
  "estr:int:opt;"                                                                                                                          \
  "dstr:int:opt;"                                                                                                                          \
  "maxd:int:opt;"                                                                                                                          \
  "map:int:opt;"                                                                                                                           \
  "nt:int:opt;"                                                                                                                            \
  "pp:int:opt;"                                                                                                                            \
  "num_streams:int:opt;"                                                                                                                   \
  "device_id:int:opt"

VS_EXTERNAL_API(void)
VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
  auto to_voidp = [](auto *p) { return const_cast<void *>(static_cast<const void *>(p)); };

  configFunc("club.amusement.eedi2cuda", "eedi2cuda", "EEDI2 filter using CUDA", VAPOURSYNTH_API_VERSION, 1, plugin);
  registerFunc("EEDI2",
               "clip:clip;"
               "field:int;" eedi2_common_params,
               eedi2Create, to_voidp("EEDI2"), plugin);
  registerFunc("Enlarge2", "clip:clip;" eedi2_common_params, eedi2Create, to_voidp("Enlarge2"), plugin);
  registerFunc("AA2", "clip:clip;" eedi2_common_params, eedi2Create, to_voidp("AA2"), plugin);
  registerFunc("BuildConfig", "", BuildConfigCreate, nullptr, plugin);
}
