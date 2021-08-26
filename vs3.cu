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

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include <VSHelper.h>
#include <VapourSynth.h>

#include "config.h"

#include "instance.h"
#include "pipeline.h"

using namespace std::literals::string_literals;

static VideoDimension create_vd(const VSMap *in, const VSAPI *vsapi) {
  auto node = vsapi->propGetNode(in, "clip", 0, nullptr);
  auto vi = vsapi->getVideoInfo(node);
  vsapi->freeNode(node);
  return VideoDimension{vi->width, vi->height, vi->format->subSamplingW};
}

static PropsMap create_props_map(const VSMap *in, const VSAPI *vsapi) {
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

namespace {
template <typename T> class Pipeline : public BasePipeline<T> {
  std::unique_ptr<VSNodeRef, void(VS_CC *const)(VSNodeRef *)> node;
  VSVideoInfo vi2;

public:
  Pipeline(std::string_view filterName, const VSMap *in, const VSAPI *vsapi)
      : BasePipeline<T>(filterName, create_props_map(in, vsapi), create_vd(in, vsapi)),
        node(vsapi->propGetNode(in, "clip", 0, nullptr), vsapi->freeNode) {
    auto vi = vsapi->getVideoInfo(node.get());
    vi2 = *vi;
    auto ovi = this->getOVD();
    vi2.width = ovi.width;
    vi2.height = ovi.height;

    if (!isConstantFormat(vi) || vi->format->sampleType != stInteger || vi->format->bytesPerSample > 2)
      throw std::invalid_argument("only constant format 8-16 bits integer input supported");
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

    this->prepare();

    std::unique_ptr<const VSFrameRef, void(VS_CC *const)(const VSFrameRef *)> src_frame{vsapi->getFrameFilter(n, node.get(), frameCtx),
                                                                                        vsapi->freeFrame};

    auto bypass_mask = this->getPlaneBypassMask();
    const VSFrameRef *plane_src[]{bypass_mask & 1 ? src_frame.get() : nullptr, bypass_mask & 2 ? src_frame.get() : nullptr,
                                  bypass_mask & 4 ? src_frame.get() : nullptr};
    int planes[] = {0, 1, 2};

    std::unique_ptr<VSFrameRef, void(VS_CC *const)(const VSFrameRef *)> dst_frame{
        vsapi->newVideoFrame2(vi2.format, vi2.width, vi2.height, plane_src, planes, src_frame.get(), core), vsapi->freeFrame};

    for (int plane = 0; plane < vi2.format->numPlanes; ++plane) {
      if (bypass_mask & (1u << plane))
        continue;

      auto src_width = vsapi->getFrameWidth(src_frame.get(), plane);
      auto src_height = vsapi->getFrameHeight(src_frame.get(), plane);
      auto dst_width = vsapi->getFrameWidth(dst_frame.get(), plane);
      auto dst_height = vsapi->getFrameHeight(dst_frame.get(), plane);
      auto s_pitch_src = vsapi->getStride(src_frame.get(), plane);
      auto s_pitch_dst = vsapi->getStride(dst_frame.get(), plane);
      auto s_src = vsapi->getReadPtr(src_frame.get(), plane);
      auto s_dst = vsapi->getWritePtr(dst_frame.get(), plane);

      this->processPlane(n, plane, src_width, src_height, dst_width, dst_height, s_pitch_src, s_pitch_dst, s_src, s_dst);
    }

    return dst_frame.release();
  }
};

template <typename T> struct Instance : public BaseInstance<T> {
  Instance(std::string_view filterName, const VSMap *in, const VSAPI *vsapi)
      : BaseInstance<T>(std::forward_as_tuple(filterName, in, vsapi), std::forward_as_tuple(vsapi)) {}
};
} // namespace

template <typename T> static void VS_CC eedi2Init(VSMap *, VSMap *, void **instanceData, VSNode *node, VSCore *, const VSAPI *vsapi) {
  auto data = static_cast<Instance<T> *>(*instanceData);
  vsapi->setVideoInfo(&data->firstReactor().getOutputVI(), 1, node);
}

template <typename T>
static const VSFrameRef *VS_CC eedi2GetFrame(int n, int activationReason, void **instanceData, void **, VSFrameContext *frameCtx,
                                             VSCore *core, const VSAPI *vsapi) {

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

template <typename T> static void VS_CC eedi2Free(void *instanceData, VSCore *, const VSAPI *) {
  auto data = static_cast<Instance<T> *>(instanceData);
  delete data;
}

template <typename T>
static void eedi2CreateInner(std::string_view filterName, const VSMap *in, VSMap *out, const VSAPI *vsapi, VSCore *core) {
  try {
    int err;
    std::size_t num_streams;
    numeric_cast_to(num_streams, vsapi->propGetInt(in, "num_streams", 0, &err));
    if (err)
      num_streams = 1;
    auto data = allocInstance<T>(num_streams);
    new (data) Instance<T>(filterName, in, vsapi);
    vsapi->createFilter(in, out, filterName.data(), eedi2Init<T>, eedi2GetFrame<T>, eedi2Free<T>,
                        num_streams > 1 ? fmParallel : fmParallelRequests, 0, data, core);
  } catch (const std::exception &exc) {
    vsapi->setError(out, ("EEDI2CUDA: "s + exc.what()).c_str());
    return;
  }
}

static VS_CC void eedi2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
  std::string_view filterName{static_cast<const char *>(userData)};
  VSNodeRef *node = vsapi->propGetNode(in, "clip", 0, nullptr);
  const VSVideoInfo *vi = vsapi->getVideoInfo(node);
  vsapi->freeNode(node);
  if (vi->format->bytesPerSample == 1)
    eedi2CreateInner<uint8_t>(filterName, in, out, vsapi, core);
  else
    eedi2CreateInner<uint16_t>(filterName, in, out, vsapi, core);
}

static void VS_CC BuildConfigCreate(const VSMap *, VSMap *out, void *, VSCore *, const VSAPI *vsapi) {
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
  "planes:int[]:opt;"                                                                                                                      \
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
