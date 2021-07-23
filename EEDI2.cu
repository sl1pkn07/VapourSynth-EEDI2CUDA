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

#define bounds_check3(value, lower, upper)                                     \
  if (value < lower || value >= upper)                                         \
  return

#define bounds_check4(lx, ly, rx, ry)                                          \
  do {                                                                         \
    static_assert(offsetof(EEDI2Instance<T>, x) % 4 == 0, "unaligned");        \
    static_assert(offsetof(EEDI2Instance<T>, width) % 4 == 0, "unaligned");    \
    static_assert(offsetof(EEDI2Instance<T>, x) + 2 ==                         \
                      offsetof(EEDI2Instance<T>, y),                           \
                  "not compact");                                              \
    static_assert(offsetof(EEDI2Instance<T>, width) + 2 ==                     \
                      offsetof(EEDI2Instance<T>, height),                      \
                  "not compact");                                              \
    constexpr auto lx16 = static_cast<uint16_t>(lx),                           \
                   ly16 = static_cast<uint16_t>(ly),                           \
                   rx16 = static_cast<uint16_t>(rx),                           \
                   ry16 = static_cast<uint16_t>(ry);                           \
    constexpr auto lower = lx16 | ly16 << 16u, upper = rx16 | ry16 << 16u;     \
    const auto co = *reinterpret_cast<uint32_t *>(&this->x),                   \
               size = *reinterpret_cast<uint32_t *>(&this->width);             \
    if (__vcmpltu2(co, lower) | __vcmpgeu2(co, __vadd2(size, upper)))          \
      return;                                                                  \
  } while (0)

#define bounds_check2(l, r) bounds_check4(l, l, r, r)

template <typename T> struct EEDI2Instance {
  // host-only variables
  /// VS related
  VSNodeRef *node;
  const VSVideoInfo *vi; // input
  VSVideoInfo *vi2;      // output
  /// CUDA related
  cudaStream_t stream;
  EEDI2Instance<T> *shadow;

  // host-device parameters
  uint8_t field, fieldS;
  uint16_t mthresh, lthresh, vthresh;
  uint8_t estr, dstr, maxd;
  uint8_t map, pp;
  uint16_t nt4, nt7, nt8, nt13, nt19;
  uint32_t d_pitch;

  // device thread-local variables
  uint16_t x, y, width, height;
  uint8_t plane;

  // device memory pointers
  T *mem[6];

  // constexpr
  static constexpr T shift = sizeof(T) * 8 - 8;
  static constexpr T peak = std::numeric_limits<T>::max();
  static constexpr T ten = 10 << shift;
  static constexpr int8_t limlut[33]{
      6,  6,  7,  7,  8,  8,  9,  9,  9,  10, 10, 11, 11, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, -1, -1};
  static constexpr int16_t limlut2[33]{
      6 << shift,  6 << shift,  7 << shift,  7 << shift,  8 << shift,
      8 << shift,  9 << shift,  9 << shift,  9 << shift,  10 << shift,
      10 << shift, 11 << shift, 11 << shift, 12 << shift, 12 << shift,
      12 << shift, 12 << shift, 12 << shift, 12 << shift, 12 << shift,
      12 << shift, 12 << shift, 12 << shift, 12 << shift, 12 << shift,
      12 << shift, 12 << shift, 12 << shift, 12 << shift, 12 << shift,
      12 << shift, -1 << shift, -1 << shift};

  void freeResources(const VSAPI *vsapi) noexcept {
    // free VS related resource
    if (node) {
      vsapi->freeNode(node);
      node = nullptr;
      delete vi2;
    }

    // free CUDA related resource
    if (stream) {
      try_cuda(cudaFreeAsync(shadow, stream));
      try_cuda(cudaFreeAsync(mem[0], stream));
      try_cuda(cudaStreamSynchronize(stream));
      try_cuda(cudaStreamDestroy(stream));
      stream = nullptr;
    }
  }

  void initProps(const VSMap *in, const VSAPI *vsapi) {
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

    field = narrow<uint8_t>(vsapi->propGetInt(in, "field", 0, nullptr));

    mthresh = narrow<uint8_t>(propGetIntDefault("mthresh", 10));
    lthresh = narrow<uint8_t>(propGetIntDefault("lthresh", 20));
    vthresh = narrow<uint8_t>(propGetIntDefault("vthresh", 20));

    estr = narrow<uint8_t>(propGetIntDefault("estr", 2));
    dstr = narrow<uint8_t>(propGetIntDefault("dstr", 4));
    maxd = narrow<uint8_t>(propGetIntDefault("maxd", 24));

    map = narrow<uint8_t>(propGetIntDefault("map", 0));
    pp = narrow<uint8_t>(propGetIntDefault("pp", 1));

    auto nt = narrow<uint16_t>(propGetIntDefault("nt", 50));

    if (field > 3)
      throw invalid_arg("field must be 0, 1, 2 or 3");
    if (maxd < 1 || maxd > 29)
      throw invalid_arg("maxd must be between 1 and 29 (inclusive)");
    if (map > 3)
      throw invalid_arg("map must be 0, 1, 2 or 3");
    if (pp > 3)
      throw invalid_arg("pp must be 0, 1, 2 or 3");

    fieldS = field;
    if (fieldS == 2)
      field = 0;
    else if (fieldS == 3)
      field = 1;

    if (map == 0 || map == 3)
      vi2->height *= 2;

    mthresh *= mthresh;
    vthresh *= 81;

    nt4 = nt * 4;
    nt7 = nt * 7;
    nt8 = nt * 8;
    nt13 = nt * 13;
    nt19 = nt * 19;

    width = vi->width;
    height = vi->height;

    this->vi2 = vi2.release();
  }

  void initCuda() {
    using gsl::narrow;

    // create stream
    try_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // alloc mem
    constexpr size_t numMem = sizeof(mem) / sizeof(T *);
    size_t pitch;
    try_cuda(
        cudaMallocPitch(&mem[0], &pitch, width * sizeof(T), height * numMem));
    d_pitch = narrow<uint32_t>(pitch);
    for (size_t i = 1; i < numMem; ++i)
      mem[i] = mem[i - 1] + d_pitch * height;

    // upload shadow
    EEDI2Instance<T> *shadow;
    try_cuda(cudaMalloc(&shadow, sizeof(*this)));
    try_cuda(cudaMemcpy(shadow, this, sizeof(*this), cudaMemcpyHostToDevice));
    this->shadow = shadow;
  }

  VSFrameRef *getFrame(int n, VSFrameContext *frameCtx, VSCore *core,
                       const VSAPI *vsapi) {
    auto field = this->field;
    if (fieldS > 1)
      field = (n & 1) ? (fieldS == 2 ? 1 : 0) : (fieldS == 2 ? 0 : 1);

    auto src_frame = vsapi->getFrameFilter(n, node, frameCtx);
    auto dst_frame = vsapi->newVideoFrame(vi2->format, vi2->width, vi2->height,
                                          src_frame, core);

    auto d_src = mem[0];
    auto d_dst = mem[5];

    for (int plane = 0; plane < vi->format->numPlanes; ++plane) {
      auto width = vsapi->getFrameWidth(src_frame, plane);
      auto height = vsapi->getFrameHeight(src_frame, plane);
      auto h_pitch = vsapi->getStride(src_frame, plane);
      auto width_bytes = width * vi->format->bytesPerSample;
      auto h_src = vsapi->getReadPtr(src_frame, plane);
      auto h_dst = vsapi->getWritePtr(dst_frame, plane);

      try_cuda(cudaMemcpy2DAsync(d_src, d_pitch, h_src, h_pitch, width_bytes,
                                 height, cudaMemcpyHostToDevice, stream));

      dim3 blocks = dim3(16, 8);
      dim3 grids =
          dim3((width - 1) / blocks.x + 1, (height - 1) / blocks.y + 1);
      kernel<<<grids, blocks, 0, stream>>>(shadow, field, plane, width, height);

      try_cuda(cudaMemcpy2DAsync(h_dst, h_pitch, d_dst, d_pitch, width_bytes,
                                 height, cudaMemcpyDeviceToHost, stream));
      try_cuda(cudaStreamSynchronize(stream));
    }

    return dst_frame;
  }

  __device__ void process() {
    auto src = mem[0];
    buildEdgeMask(src, mem[1]);
    erode(mem[1], mem[2]);
    dilate(mem[2], mem[3]);
    erode(mem[3], mem[4]);
    removeSmallHorzGaps(mem[4], mem[5]);
  }

  __device__ void buildEdgeMask(const T *src, T *dst) {
    bounds_check2(1, -1);

    auto srcp = line(src);
    auto srcpp = line(src, -1);
    auto srcpn = line(src, 1);
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
    if (9 * sumsq - sum * sum < vthresh)
      return;

    const unsigned Ix = std::abs(srcp[x + 1] - srcp[x - 1]) >> shift;
    const unsigned Iy =
        std::max({std::abs(srcpp[x] - srcpn[x]), std::abs(srcpp[x] - srcp[x]),
                  std::abs(srcp[x] - srcpn[x])}) >>
        shift;
    if (Ix * Ix + Iy * Iy >= mthresh) {
      out = peak;
      return;
    }

    const unsigned Ixx =
        std::abs(srcp[x - 1] - 2 * srcp[x] + srcp[x + 1]) >> shift;
    const unsigned Iyy = std::abs(srcpp[x] - 2 * srcp[x] + srcpn[x]) >> shift;
    if (Ixx + Iyy >= lthresh)
      out = peak;
  }

  __device__ void erode(const T *msk, T *dst) {
    bounds_check2(1, -1);

    auto mskp = line(msk);
    auto mskpp = line(msk, -1);
    auto mskpn = line(msk, 1);
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

    if (count < estr)
      out = 0;
  }

  __device__ void dilate(const T *msk, T *dst) {
    bounds_check2(1, -1);

    auto mskp = line(msk);
    auto mskpp = line(msk, -1);
    auto mskpn = line(msk, 1);
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

    if (count >= dstr)
      out = peak;
  }

  __device__ void removeSmallHorzGaps(const T *msk, T *dst) {
    bounds_check4(3, 1, -3, -1);

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

  // helpers
  __device__ uint32_t stride() const { return d_pitch / sizeof(T); }
  __device__ T *line(T *p, int8_t off = 0) const {
    return p + stride() * (y + off);
  }
  __device__ const T *line(const T *p, int8_t off = 0) const {
    return p + stride() * (y + off);
  }
  __device__ T &point(T *p) const { return p[stride() * y + x]; }
};

template <typename T>
__global__ void kernel(EEDI2Instance<T> *d_g, uint8_t field, uint8_t plane,
                       uint16_t width, uint16_t height) {
  EEDI2Instance<T> d_l = *d_g;
  d_l.x = threadIdx.x + blockIdx.x * blockDim.x;
  d_l.y = threadIdx.y + blockIdx.y * blockDim.y;
  d_l.field = field;
  d_l.plane = plane;
  d_l.width = width;
  d_l.height = height;
  d_l.process();
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
      vsapi->setFilterError(("EEDI2CUDA"s + exc.what()).c_str(), frameCtx);
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
    d->initProps(in, vsapi);
    d->initCuda();
  } catch (const std::exception &exc) {
    d->freeResources(vsapi);
    vsapi->setError(out, ("EEDI2CUDA"s + exc.what()).c_str());
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
