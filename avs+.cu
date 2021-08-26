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

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include <avisynth.h>

#include "config.h"

#include "instance.h"
#include "pipeline.h"

static inline int getPlaneId(const VideoInfo &vi, int plane) {
  if (!vi.IsPlanar()) {
    throw std::runtime_error("only planar format is supported");
  }
  if (vi.IsY()) {
    assert(plane == 0);
    return PLANAR_Y;
  } else if (vi.IsRGB()) {
    assert(0 <= plane && plane < 3);
    int planes_id[]{PLANAR_R, PLANAR_G, PLANAR_B};
    return planes_id[plane];
  } else if (vi.IsYUV()) {
    assert(0 <= plane && plane < 3);
    int planes_id[]{PLANAR_Y, PLANAR_U, PLANAR_V};
    return planes_id[plane];
  } else {
    throw std::runtime_error("unsupported format");
  }
}

static VideoDimension get_vi(AVSValue args) {
  auto node = args[0].AsClip();
  const auto &vi = node->GetVideoInfo();
  return VideoDimension{vi.width, vi.height, vi.IsYUV() ? vi.GetPlaneWidthSubsampling(PLANAR_U) : 0};
}

static PropsMap mapize(AVSValue args) {
  PropsMap m;

  auto num_args = args.ArraySize();
  if (num_args == 14) { // EEDI2_CUDA exclusive
    m.emplace("field", args[num_args - 13].AsInt());
  }

  auto load_param = [&](const char *name, AVSValue arg) {
    if (arg.Defined()) {
      m.emplace(name, arg.AsInt());
    }
  };
  load_param("mthresh", args[num_args - 12]);
  load_param("lthresh", args[num_args - 11]);
  load_param("vthresh", args[num_args - 10]);
  load_param("estr", args[num_args - 9]);
  load_param("dstr", args[num_args - 8]);
  load_param("maxd", args[num_args - 7]);
  load_param("map", args[num_args - 6]);
  load_param("nt", args[num_args - 5]);
  load_param("pp", args[num_args - 4]);
  if (const auto &arg = args[num_args - 3]; arg.Defined()) {
    if (arg.IsInt()) {
      m.emplace("planes", arg.AsInt());
    } else {
      auto num = arg.ArraySize();
      for (auto i = 0; i < num; ++i) {
        m.emplace("planes", arg[i].AsInt());
      }
    }
  }
  load_param("device_id", args[num_args - 1]);

  return m;
}

namespace {
template <typename T> class Pipeline : public BasePipeline<T> {
  PClip node;
  VideoInfo vi2;

public:
  Pipeline(std::string_view filterName, AVSValue args) : BasePipeline<T>(filterName, mapize(args), get_vi(args)), node(args[0].AsClip()) {
    auto vi = node->GetVideoInfo();
    vi2 = vi;
    auto ovi = BasePipeline<T>::getOutputVI();
    vi2.width = ovi.width;
    vi2.height = ovi.height;

    if (vi.BitsPerComponent() < 8 || vi.BitsPerComponent() > 16)
      throw std::invalid_argument("only constant format 8-16 bits integer input supported");
  }

  Pipeline(const Pipeline &other) : BasePipeline<T>(other), node(other.node), vi2(other.vi2) {}

  const VideoInfo &getOutputVI() const { return vi2; }

  PVideoFrame getFrame(int n, IScriptEnvironment *env) {
    this->prepare();

    auto src_frame = node->GetFrame(n, env);

    auto bypass_mask = this->getPlaneBypassMask();
    PVideoFrame dst_frame = env->NewVideoFrameP(vi2, &src_frame);

    for (int plane = 0; plane < vi2.NumComponents(); ++plane) {
      if (bypass_mask & (1u << plane))
        continue;

      auto src_width_bytes = src_frame->GetRowSize(getPlaneId(vi2, plane));
      auto dst_width_bytes = dst_frame->GetRowSize(getPlaneId(vi2, plane));
      auto src_width = src_width_bytes / sizeof(T);
      auto src_height = src_frame->GetHeight(getPlaneId(vi2, plane));
      auto dst_width = dst_width_bytes / sizeof(T);
      auto dst_height = dst_frame->GetHeight(getPlaneId(vi2, plane));
      auto s_pitch_src = src_frame->GetPitch(getPlaneId(vi2, plane));
      auto s_pitch_dst = dst_frame->GetPitch(getPlaneId(vi2, plane));
      auto s_src = src_frame->GetReadPtr(getPlaneId(vi2, plane));
      auto s_dst = dst_frame->GetWritePtr(getPlaneId(vi2, plane));

      this->processPlane(n, plane, src_width, src_height, dst_width, dst_height, s_pitch_src, s_pitch_dst, s_src, s_dst);
    }

    return dst_frame;
  }
};

template <typename T> struct Instance : public BaseInstance<T> {
  Instance(std::string_view filterName, AVSValue args, IScriptEnvironment *env)
      : BaseInstance<T>(std::forward_as_tuple(filterName, args), std::forward_as_tuple()) {}
};
} // namespace

class EEDI2Filter : public GenericVideoFilter {
  union {
    Instance<uint8_t> *u8;
    Instance<uint16_t> *u16;
  } data;

public:
  EEDI2Filter(std::string_view filterName, AVSValue args, IScriptEnvironment *env);

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override;

  static AVSValue __cdecl Create(AVSValue args, void *user_data, IScriptEnvironment *env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override { return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0; }
};

PVideoFrame __stdcall EEDI2Filter::GetFrame(int n, IScriptEnvironment *env) {
  PVideoFrame out;
  if (vi.BitsPerComponent() == 8) {
    auto &d = data.u8->acquireReactor();
    try {
      out = d.getFrame(n, env);
    } catch (const std::exception &exc) {
      env->ThrowError(("EEDI2CUDA: "s + exc.what()).c_str());
    }
    data.u8->releaseReactor(d);
  } else {
    auto &d = data.u16->acquireReactor();
    try {
      out = d.getFrame(n, env);
    } catch (const std::exception &exc) {
      env->ThrowError(("EEDI2CUDA: "s + exc.what()).c_str());
    }
    data.u16->releaseReactor(d);
  }
  return out;
}

EEDI2Filter::EEDI2Filter(std::string_view filterName, AVSValue args, IScriptEnvironment *env) : GenericVideoFilter(args[0].AsClip()) {

  env->CheckVersion(8);

  try {
    int num_streams;
    numeric_cast_to(num_streams, args[args.ArraySize() - 2].AsInt(1));
    if (vi.BitsPerComponent() == 8) {
      data.u8 = allocInstance<uint8_t>(num_streams);
      new (data.u8) Instance<uint8_t>(filterName, args, env);
    } else {
      data.u16 = allocInstance<uint16_t>(num_streams);
      new (data.u16) Instance<uint16_t>(filterName, args, env);
    }
  } catch (const std::exception &exc) {
    env->ThrowError(("EEDI2CUDA: "s + exc.what()).c_str());
  }

  if (vi.BitsPerComponent() == 8) {
    vi.width = data.u8->firstReactor().getOutputVI().width;
    vi.height = data.u8->firstReactor().getOutputVI().height;
  } else {
    vi.width = data.u16->firstReactor().getOutputVI().width;
    vi.height = data.u16->firstReactor().getOutputVI().height;
  }
}

AVSValue __cdecl EEDI2Filter::Create(AVSValue args, void *user_data, IScriptEnvironment *env) {

  std::string_view filterName{static_cast<const char *>(user_data)};

  return new EEDI2Filter(filterName, args, env);
}

const AVS_Linkage *AVS_linkage{};

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors) {

  auto to_voidp = [](auto *p) { return const_cast<void *>(static_cast<const void *>(p)); };

  AVS_linkage = vectors;

#define eedi2_common_params                                                                                                                \
  "[mthresh]i"                                                                                                                             \
  "[lthresh]i"                                                                                                                             \
  "[vthresh]i"                                                                                                                             \
  "[estr]i"                                                                                                                                \
  "[dstr]i"                                                                                                                                \
  "[maxd]i"                                                                                                                                \
  "[map]i"                                                                                                                                 \
  "[nt]i"                                                                                                                                  \
  "[pp]i"                                                                                                                                  \
  "[planes]i"                                                                                                                              \
  "[num_streams]i"                                                                                                                         \
  "[device_id]i"

  env->AddFunction("EEDI2_CUDA", "ci" eedi2_common_params, EEDI2Filter::Create, to_voidp("EEDI2"));
  env->AddFunction("EEDI2_CUDA_Enlarge2", "c" eedi2_common_params, EEDI2Filter::Create, to_voidp("Enlarge2"));
  env->AddFunction("EEDI2_CUDA_AA2", "c" eedi2_common_params, EEDI2Filter::Create, to_voidp("AA2"));
  env->AddFunction(
      "EEDI2_CUDA_Version", "", [](AVSValue, void *, IScriptEnvironment *) { return AVSValue(VERSION); }, nullptr);
  env->AddFunction(
      "EEDI2_CUDA_Options", "", [](AVSValue, void *, IScriptEnvironment *) { return AVSValue(BUILD_OPTIONS); }, nullptr);
  env->AddFunction(
      "EEDI2_CUDA_TimeStamp", "", [](AVSValue, void *, IScriptEnvironment *) { return AVSValue(CONFIGURE_TIME); }, nullptr);
  env->AddFunction(
      "EEDI2_CUDA_APIVersion", "", [](AVSValue, void *, IScriptEnvironment *) { return AVSValue(AVISYNTH_INTERFACE_VERSION); }, nullptr);

  return "EEDI2 filter using CUDA";
}
