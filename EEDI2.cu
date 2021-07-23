#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>

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
      delete vi2;
    }

    // free CUDA related resource
    if (stream) {
      try_cuda(cudaFreeAsync(shadow, stream));
      try_cuda(cudaFreeAsync(mem[0], stream));
      try_cuda(cudaStreamSynchronize(stream));
      try_cuda(cudaStreamDestroy(stream));
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
};

template <typename T>
void VS_CC eedi2Init(VSMap *_in, VSMap *_out, void **instanceData, VSNode *node,
                     VSCore *_core, const VSAPI *vsapi) {
  auto d = static_cast<EEDI2Instance<T> *>(*instanceData);
  vsapi->setVideoInfo(d->vi2, 1, node);
}

template <typename T>
const VSFrameRef *VS_CC eedi2GetFrame(int n, int activationReason,
                                      void **instanceData, void **frameData,
                                      VSFrameContext *frameCtx, VSCore *core,
                                      const VSAPI *vsapi) {
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
