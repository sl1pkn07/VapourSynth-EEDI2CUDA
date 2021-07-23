#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include <stdint.h>

#include <gsl/narrow>

#include <VSHelper.h>
#include <VapourSynth.h>

using namespace std::literals::string_literals;

class CUDAError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
#define try_cuda(expr)                                                         \
  do {                                                                         \
    cudaError_t __err = expr;                                                  \
    if (__err != cudaSuccess) {                                                \
      throw CUDAError(cudaGetErrorString(__err));                              \
    }                                                                          \
  } while (0)

struct EEDI2Param {
  uint32_t d_pitch;
  uint16_t mthresh, lthresh, vthresh;
  uint16_t nt4, nt7, nt8, nt13, nt19;
  uint16_t width, height;
  uint8_t field, fieldS;
  uint8_t estr, dstr, maxd;
  uint8_t map, pp, plane;
};
__constant__ char d_buf[sizeof(EEDI2Param)];

template <typename T> struct EEDI2Instance {
  VSNodeRef *node;
  const VSVideoInfo *vi; // input
  VSVideoInfo *vi2;      // output
  cudaStream_t stream;
  EEDI2Param param;
  T *dst, *msk, *tmp, *src;

  void freeResources(const VSAPI *vsapi) noexcept {
    // free VS related resource
    if (node) {
      vsapi->freeNode(node);
      node = nullptr;
      delete vi2;
    }

    // free CUDA related resource
    if (stream) {
      try_cuda(cudaFree(dst));
      try_cuda(cudaStreamDestroy(stream));
      stream = nullptr;
    }
  }

  void initParams(const VSMap *in, const VSAPI *vsapi) {
    using invalid_arg = std::invalid_argument;
    using gsl::narrow;

    node = vsapi->propGetNode(in, "clip", 0, nullptr);
    vi = vsapi->getVideoInfo(node);
    auto vi2 = std::make_unique<VSVideoInfo>(*vi);
    const auto &fmt = *vi->format;

    if (!isConstantFormat(vi) || fmt.sampleType != stInteger ||
        fmt.bytesPerSample > 2)
      throw invalid_arg(
          "only constant format 8-16 bits integer input supported");
    if (vi->width < 8 || vi->height < 7)
      throw invalid_arg("clip resolution too low");

    auto propGetIntDefault = [&](const char *key, int64_t def) {
      int err;
      auto ret = vsapi->propGetInt(in, key, 0, &err);
      return err ? def : ret;
    };

    param.field = narrow<uint8_t>(vsapi->propGetInt(in, "field", 0, nullptr));

    param.mthresh = narrow<uint8_t>(propGetIntDefault("mthresh", 10));
    param.lthresh = narrow<uint8_t>(propGetIntDefault("lthresh", 20));
    param.vthresh = narrow<uint8_t>(propGetIntDefault("vthresh", 20));

    param.estr = narrow<uint8_t>(propGetIntDefault("estr", 2));
    param.dstr = narrow<uint8_t>(propGetIntDefault("dstr", 4));
    param.maxd = narrow<uint8_t>(propGetIntDefault("maxd", 24));

    param.map = narrow<uint8_t>(propGetIntDefault("map", 0));
    param.pp = narrow<uint8_t>(propGetIntDefault("pp", 1));

    auto nt = narrow<uint16_t>(propGetIntDefault("nt", 50));

    if (param.field > 3)
      throw invalid_arg("field must be 0, 1, 2 or 3");
    if (param.maxd < 1 || param.maxd > 29)
      throw invalid_arg("maxd must be between 1 and 29 (inclusive)");
    if (param.map > 3)
      throw invalid_arg("map must be 0, 1, 2 or 3");
    if (param.pp > 3)
      throw invalid_arg("pp must be 0, 1, 2 or 3");

    param.fieldS = param.field;
    if (param.fieldS == 2)
      param.field = 0;
    else if (param.fieldS == 3)
      param.field = 1;

    if (param.map == 0 || param.map == 3)
      vi2->height *= 2;

    param.mthresh *= param.mthresh;
    param.vthresh *= 81;

    param.nt4 = nt * 4;
    param.nt7 = nt * 7;
    param.nt8 = nt * 8;
    param.nt13 = nt * 13;
    param.nt19 = nt * 19;

    this->vi2 = vi2.release();
  }

  void initCuda() {
    // create stream
    try_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // alloc mem
    constexpr size_t numMem = 4;
    T **mem = &dst;
    size_t pitch;
    try_cuda(cudaMallocPitch(&mem[0], &pitch, vi->width * sizeof(T),
                             vi->height * numMem));
    auto d_pitch = param.d_pitch = static_cast<uint32_t>(pitch);
    for (size_t i = 1; i < numMem; ++i)
      mem[i] = mem[i - 1] + d_pitch * vi->height;
  }

  VSFrameRef *getFrame(int n, VSFrameContext *frameCtx, VSCore *core,
                       const VSAPI *vsapi) {
    auto field = this->param.field;
    if (param.fieldS > 1)
      field =
          (n & 1) ? (param.fieldS == 2 ? 1 : 0) : (param.fieldS == 2 ? 0 : 1);

    auto src_frame = vsapi->getFrameFilter(n, node, frameCtx);
    auto dst_frame = vsapi->newVideoFrame(vi2->format, vi2->width, vi2->height,
                                          src_frame, core);

    auto d_src = src;
    auto d_dst = msk;
    auto d_pitch = param.d_pitch;

    for (int plane = 0; plane < vi->format->numPlanes; ++plane) {
      auto width = vsapi->getFrameWidth(src_frame, plane);
      auto height = vsapi->getFrameHeight(src_frame, plane);
      auto h_pitch = vsapi->getStride(src_frame, plane);
      auto width_bytes = width * vi->format->bytesPerSample;
      auto h_src = vsapi->getReadPtr(src_frame, plane);
      auto h_dst = vsapi->getWritePtr(dst_frame, plane);

      param.field = field;
      param.plane = plane;
      param.width = width;
      param.height = height;

      try_cuda(cudaMemcpy2DAsync(d_src, d_pitch, h_src, h_pitch, width_bytes,
                                 height, cudaMemcpyHostToDevice, stream));
      try_cuda(cudaMemcpyToSymbolAsync(d_buf, &param, sizeof(EEDI2Param), 0,
                                       cudaMemcpyHostToDevice, stream));

      dim3 blocks = dim3(16, 8);
      dim3 grids =
          dim3((width - 1) / blocks.x + 1, (height - 1) / blocks.y + 1);
      buildEdgeMask<<<grids, blocks, 0, stream>>>(src, msk);
      erode<<<grids, blocks, 0, stream>>>(msk, tmp);
      dilate<<<grids, blocks, 0, stream>>>(tmp, msk);
      erode<<<grids, blocks, 0, stream>>>(msk, tmp);
      removeSmallHorzGaps<<<grids, blocks, 0, stream>>>(tmp, msk);

      try_cuda(cudaMemcpy2DAsync(h_dst, h_pitch, d_dst, d_pitch, width_bytes,
                                 height, cudaMemcpyDeviceToHost, stream));
      try_cuda(cudaStreamSynchronize(stream));
    }

    return dst_frame;
  }
};

#define setup_kernel                                                           \
  const EEDI2Param *d = reinterpret_cast<const EEDI2Param *>(d_buf);           \
  const uint16_t width = d->width, height = d->height,                         \
                 x = threadIdx.x + blockIdx.x * blockDim.x,                    \
                 y = threadIdx.y + blockIdx.y * blockDim.y;                    \
  constexpr T shift = sizeof(T) * 8 - 8, peak = std::numeric_limits<T>::max(), \
              ten = 10 << shift

#define bounds_check3(value, lower, upper)                                     \
  if ((value) < (lower) || (value) >= (upper))                                 \
  return

#define stride (d->d_pitch / sizeof(T))
#define line(p) ((p) + stride * y)
#define lineOff(p, off) ((p) + stride * (y + (off)))
#define point(p) ((p)[stride * y + x])

template <typename T> __global__ void buildEdgeMask(const T *src, T *dst) {
  setup_kernel;
  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  auto srcp = line(src);
  auto srcpp = lineOff(src, -1);
  auto srcpn = lineOff(src, 1);
  auto &out = point(dst);

  out = 0;

  if ((std::abs(srcpp[x] - srcp[x]) < ten &&
       std::abs(srcp[x] - srcpn[x]) < ten &&
       std::abs(srcpp[x] - srcpn[x]) < ten) ||
      (std::abs(srcpp[x - 1] - srcp[x - 1]) < ten &&
       std::abs(srcp[x - 1] - srcpn[x - 1]) < ten &&
       std::abs(srcpp[x - 1] - srcpn[x - 1]) < ten &&
       std::abs(srcpp[x + 1] - srcp[x + 1]) < ten &&
       std::abs(srcp[x + 1] - srcpn[x + 1]) < ten &&
       std::abs(srcpp[x + 1] - srcpn[x + 1]) < ten))
    return;

  const unsigned sum =
      (srcpp[x - 1] + srcpp[x] + srcpp[x + 1] + srcp[x - 1] + srcp[x] +
       srcp[x + 1] + srcpn[x - 1] + srcpn[x] + srcpn[x + 1]) >>
      shift;
  const unsigned sumsq = (srcpp[x - 1] >> shift) * (srcpp[x - 1] >> shift) +
                         (srcpp[x] >> shift) * (srcpp[x] >> shift) +
                         (srcpp[x + 1] >> shift) * (srcpp[x + 1] >> shift) +
                         (srcp[x - 1] >> shift) * (srcp[x - 1] >> shift) +
                         (srcp[x] >> shift) * (srcp[x] >> shift) +
                         (srcp[x + 1] >> shift) * (srcp[x + 1] >> shift) +
                         (srcpn[x - 1] >> shift) * (srcpn[x - 1] >> shift) +
                         (srcpn[x] >> shift) * (srcpn[x] >> shift) +
                         (srcpn[x + 1] >> shift) * (srcpn[x + 1] >> shift);
  if (9 * sumsq - sum * sum < d->vthresh)
    return;

  const unsigned Ix = std::abs(srcp[x + 1] - srcp[x - 1]) >> shift;
  const unsigned Iy =
      std::max({std::abs(srcpp[x] - srcpn[x]), std::abs(srcpp[x] - srcp[x]),
                std::abs(srcp[x] - srcpn[x])}) >>
      shift;
  if (Ix * Ix + Iy * Iy >= d->mthresh) {
    out = peak;
    return;
  }

  const unsigned Ixx =
      std::abs(srcp[x - 1] - 2 * srcp[x] + srcp[x + 1]) >> shift;
  const unsigned Iyy = std::abs(srcpp[x] - 2 * srcp[x] + srcpn[x]) >> shift;
  if (Ixx + Iyy >= d->lthresh)
    out = peak;
}

template <typename T> __global__ void erode(const T *msk, T *dst) {
  setup_kernel;
  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  auto mskp = line(msk);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = mskp[x];

  if (mskp[x] != peak)
    return;

  unsigned count = 0;

  if (mskpp[x - 1] == peak)
    count++;
  if (mskpp[x] == peak)
    count++;
  if (mskpp[x + 1] == peak)
    count++;
  if (mskp[x - 1] == peak)
    count++;
  if (mskp[x + 1] == peak)
    count++;
  if (mskpn[x - 1] == peak)
    count++;
  if (mskpn[x] == peak)
    count++;
  if (mskpn[x + 1] == peak)
    count++;

  if (count < d->estr)
    out = 0;
}

template <typename T> __global__ void dilate(const T *msk, T *dst) {
  setup_kernel;
  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  auto mskp = line(msk);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = mskp[x];

  if (mskp[x] != 0)
    return;

  unsigned count = 0;

  if (mskpp[x - 1] == peak)
    count++;
  if (mskpp[x] == peak)
    count++;
  if (mskpp[x + 1] == peak)
    count++;
  if (mskp[x - 1] == peak)
    count++;
  if (mskp[x + 1] == peak)
    count++;
  if (mskpn[x - 1] == peak)
    count++;
  if (mskpn[x] == peak)
    count++;
  if (mskpn[x + 1] == peak)
    count++;

  if (count >= d->dstr)
    out = peak;
}

template <typename T>
__global__ void removeSmallHorzGaps(const T *msk, T *dst) {
  setup_kernel;
  bounds_check3(x, 3, width - 3);
  bounds_check3(y, 1, height - 1);

  auto mskp = line(msk);
  auto &out = point(dst);

  out = mskp[x];

  if (mskp[x]) {
    if (mskp[x - 3] || mskp[x - 2] || mskp[x - 1] || mskp[x + 1] ||
        mskp[x + 2] || mskp[x + 3])
      return;
    out = 0;
  } else {
    if ((mskp[x + 1] && (mskp[x - 1] || mskp[x - 2] || mskp[x - 3])) ||
        (mskp[x + 2] && (mskp[x - 1] || mskp[x - 2])) ||
        (mskp[x + 3] && mskp[x - 1]))
      out = peak;
  }
}

template <typename T>
void VS_CC eedi2Init(VSMap *_in, VSMap *_out, void **instanceData, VSNode *node,
                     VSCore *_core, const VSAPI *vsapi) {
  auto d = static_cast<EEDI2Instance<T> *>(*instanceData);
  vsapi->setVideoInfo(d->vi2, 1, node);
}

template <typename T>
const VSFrameRef *VS_CC eedi2GetFrame(int n, int activationReason,
                                      void **instanceData, void **_frameData,
                                      VSFrameContext *frameCtx, VSCore *core,
                                      const VSAPI *vsapi) {
  auto d = static_cast<EEDI2Instance<T> *>(*instanceData);
  if (activationReason == arInitial)
    vsapi->requestFrameFilter(n, d->node, frameCtx);
  else if (activationReason == arAllFramesReady)
    try {
      return d->getFrame(n, frameCtx, core, vsapi);
    } catch (const std::exception &exc) {
      vsapi->setFilterError(("EEDI2CUDA: "s + exc.what()).c_str(), frameCtx);
    }
  return nullptr;
}

template <typename T>
void VS_CC eedi2Free(void *instanceData, VSCore *_core, const VSAPI *vsapi) {
  auto d = static_cast<EEDI2Instance<T> *>(instanceData);
  d->freeResources(vsapi);
  delete d;
}

template <typename T>
void eedi2CreateInner(const VSMap *in, VSMap *out, const VSAPI *vsapi,
                      VSCore *core) {
  std::unique_ptr<EEDI2Instance<T>> d{new EEDI2Instance<T>()};
  try {
    d->initParams(in, vsapi);
    d->initCuda();
  } catch (const std::exception &exc) {
    d->freeResources(vsapi);
    vsapi->setError(out, ("EEDI2CUDA: "s + exc.what()).c_str());
    return;
  }
  vsapi->createFilter(in, out, "EEDI2", eedi2Init<T>, eedi2GetFrame<T>,
                      eedi2Free<T>, fmParallelRequests, 0, d.release(), core);
}

void VS_CC eedi2Create(const VSMap *in, VSMap *out, void *_userData,
                       VSCore *core, const VSAPI *vsapi) {
  VSNodeRef *node = vsapi->propGetNode(in, "clip", 0, nullptr);
  const VSVideoInfo *vi = vsapi->getVideoInfo(node);
  vsapi->freeNode(node);
  if (vi->format->bytesPerSample == 1)
    eedi2CreateInner<uint8_t>(in, out, vsapi, core);
  else
    eedi2CreateInner<uint16_t>(in, out, vsapi, core);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit(VSConfigPlugin configFunc,
                      VSRegisterFunction registerFunc, VSPlugin *plugin) {
  configFunc("club.amusement.eedi2cuda", "eedi2cuda", "EEDI2 filter using CUDA",
             VAPOURSYNTH_API_VERSION, 1, plugin);
  registerFunc("EEDI2",
               "clip:clip;"
               "field:int;"
               "mthresh:int:opt;"
               "lthresh:int:opt;"
               "vthresh:int:opt;"
               "estr:int:opt;"
               "dstr:int:opt;"
               "maxd:int:opt;"
               "map:int:opt;"
               "nt:int:opt;"
               "pp:int:opt;",
               eedi2Create, nullptr, plugin);
}
