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
  uint32_t nt4, nt7, nt8, nt13, nt19;
  uint16_t mthresh, lthresh, vthresh;
  uint16_t width, height;
  uint8_t field;
  uint8_t estr, dstr, maxd;
  uint8_t subSampling;
};
__constant__ char d_buf[sizeof(EEDI2Param)];
__constant__ int8_t limlut[33]{6,  6,  7,  7,  8,  8,  9,  9,  9,  10, 10,
                               11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, -1, -1};

template <typename T> class EEDI2Instance {
  std::unique_ptr<VSNodeRef, void (*const)(VSNodeRef *)> node;
  const VSVideoInfo *vi;
  std::unique_ptr<VSVideoInfo> vi2;
  cudaStream_t stream;
  EEDI2Param param;
  T *dst, *msk, *tmp, *src;

  uint8_t map, pp, field, fieldS;

public:
  EEDI2Instance(const VSMap *in, const VSAPI *vsapi)
      : node(vsapi->propGetNode(in, "clip", 0, nullptr), vsapi->freeNode) {
    initParams(in, vsapi);
    initCuda();
  }

  ~EEDI2Instance() {
    try_cuda(cudaFree(dst));
    try_cuda(cudaStreamDestroy(stream));
  }

private:
  void initParams(const VSMap *in, const VSAPI *vsapi) {
    using invalid_arg = std::invalid_argument;
    using gsl::narrow;

    vi = vsapi->getVideoInfo(node.get());
    vi2 = std::make_unique<VSVideoInfo>(*vi);
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

    param.mthresh = narrow<uint8_t>(propGetIntDefault("mthresh", 10));
    param.lthresh = narrow<uint8_t>(propGetIntDefault("lthresh", 20));
    param.vthresh = narrow<uint8_t>(propGetIntDefault("vthresh", 20));

    param.estr = narrow<uint8_t>(propGetIntDefault("estr", 2));
    param.dstr = narrow<uint8_t>(propGetIntDefault("dstr", 4));
    param.maxd = narrow<uint8_t>(propGetIntDefault("maxd", 24));

    map = narrow<uint8_t>(propGetIntDefault("map", 0));
    pp = narrow<uint8_t>(propGetIntDefault("pp", 1));

    uint16_t nt = narrow<uint8_t>(propGetIntDefault("nt", 50));

    if (field > 3)
      throw invalid_arg("field must be 0, 1, 2 or 3");
    if (param.maxd < 1 || param.maxd > 29)
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

    param.mthresh *= param.mthresh;
    param.vthresh *= 81;

    nt <<= sizeof(T) * 8 - 8;
    param.nt4 = nt * 4;
    param.nt7 = nt * 7;
    param.nt8 = nt * 8;
    param.nt13 = nt * 13;
    param.nt19 = nt * 19;
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
      mem[i] = reinterpret_cast<T *>(reinterpret_cast<char *>(mem[i - 1]) +
                                     d_pitch * vi->height);
  }

public:
  void initNode(VSNode *node, const VSAPI *vsapi) {
    vsapi->setVideoInfo(vi2.get(), 1, node);
  }

  VSFrameRef *getFrame(int n, int activationReason, VSFrameContext *frameCtx,
                       VSCore *core, const VSAPI *vsapi) {
    if (activationReason == arInitial) {
      vsapi->requestFrameFilter(n, node.get(), frameCtx);
      return nullptr;
    } else if (activationReason != arAllFramesReady)
      return nullptr;

    auto field = this->field;
    if (fieldS > 1)
      field = (n & 1) ? (fieldS == 2 ? 1 : 0) : (fieldS == 2 ? 0 : 1);

    auto src_frame = vsapi->getFrameFilter(n, node.get(), frameCtx);
    auto dst_frame = vsapi->newVideoFrame(vi2->format, vi2->width, vi2->height,
                                          src_frame, core);

    auto d_src = src;
    auto d_dst = dst;
    auto d_pitch = param.d_pitch;

    for (int plane = 0; plane < vi->format->numPlanes; ++plane) {
      auto width = vsapi->getFrameWidth(src_frame, plane);
      auto height = vsapi->getFrameHeight(src_frame, plane);
      auto h_pitch = vsapi->getStride(src_frame, plane);
      auto width_bytes = width * vi->format->bytesPerSample;
      auto h_src = vsapi->getReadPtr(src_frame, plane);
      auto h_dst = vsapi->getWritePtr(dst_frame, plane);

      param.field = static_cast<uint8_t>(field);
      param.width = static_cast<uint16_t>(width);
      param.height = static_cast<uint16_t>(height);
      param.subSampling =
          static_cast<uint8_t>(plane ? vi->format->subSamplingW : 0);

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
      if (map != 1) {
        calcDirections<<<grids, blocks, 0, stream>>>(src, msk, tmp);
        filterDirMap<<<grids, blocks, 0, stream>>>(msk, tmp, dst);
        expandDirMap<<<grids, blocks, 0, stream>>>(msk, dst, tmp);
        filterMap<<<grids, blocks, 0, stream>>>(msk, tmp, dst);
      }

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
              ten = 10 << shift, twleve = 12 << shift;                         \
  constexpr T shift2 = shift + 2, neutral = peak / 2;                          \
  if (x >= width || y >= height)                                               \
  return

#define bounds_check3(value, lower, upper)                                     \
  if ((value) < (lower) || (value) >= (upper))                                 \
  return

#define stride (d->d_pitch / sizeof(T))
#define line(p) ((p) + stride * y)
#define lineOff(p, off) ((p) + stride * (y + (off)))
#define point(p) ((p)[stride * y + x])

#define CMPSWAP(arr, i, j)                                                     \
  if (arr[i] > arr[j]) {                                                       \
    auto t = arr[i];                                                           \
    arr[i] = arr[j];                                                           \
    arr[j] = t;                                                                \
  }

template <typename T> __global__ void buildEdgeMask(const T *src, T *dst) {
  setup_kernel;

  auto srcp = line(src);
  auto srcpp = lineOff(src, -1);
  auto srcpn = lineOff(src, 1);
  auto &out = point(dst);

  out = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

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

  auto mskp = line(msk);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = mskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

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

  auto mskp = line(msk);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = mskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

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

  auto mskp = line(msk);
  auto &out = point(dst);

  out = mskp[x];

  bounds_check3(x, 3, width - 3);
  bounds_check3(y, 1, height - 1);

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
__global__ void calcDirections(const T *src, const T *msk, T *dst) {
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

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  const int maxd = d->maxd >> d->subSampling;

  if (mskp[x] != peak || (mskp[x - 1] != peak && mskp[x + 1] != peak))
    return;

  const int uStart = std::max(-x + 1, -maxd);
  const int uStop = std::min(width - 2 - x, maxd);
  const unsigned min0 =
      std::abs(srcp[x] - srcpn[x]) + std::abs(srcp[x] - srcpp[x]);
  unsigned minA = std::min(d->nt19, min0 * 9);
  unsigned minB = std::min(d->nt13, min0 * 6);
  unsigned minC = minA;
  unsigned minD = minB;
  unsigned minE = minB;
  int dirA = -5000, dirB = -5000, dirC = -5000, dirD = -5000, dirE = -5000;

  for (int u = uStart; u <= uStop; u++) {
    if ((y == 1 || mskpp[x - 1 + u] == peak || mskpp[x + u] == peak ||
         mskpp[x + 1 + u] == peak) &&
        (y == height - 2 || mskpn[x - 1 - u] == peak || mskpn[x - u] == peak ||
         mskpn[x + 1 - u] == peak)) {
      const unsigned diffsn = std::abs(srcp[x - 1] - srcpn[x - 1 - u]) +
                              std::abs(srcp[x] - srcpn[x - u]) +
                              std::abs(srcp[x + 1] - srcpn[x + 1 - u]);
      const unsigned diffsp = std::abs(srcp[x - 1] - srcpp[x - 1 + u]) +
                              std::abs(srcp[x] - srcpp[x + u]) +
                              std::abs(srcp[x + 1] - srcpp[x + 1 + u]);
      const unsigned diffps = std::abs(srcpp[x - 1] - srcp[x - 1 - u]) +
                              std::abs(srcpp[x] - srcp[x - u]) +
                              std::abs(srcpp[x + 1] - srcp[x + 1 - u]);
      const unsigned diffns = std::abs(srcpn[x - 1] - srcp[x - 1 + u]) +
                              std::abs(srcpn[x] - srcp[x + u]) +
                              std::abs(srcpn[x + 1] - srcp[x + 1 + u]);
      const unsigned diff = diffsn + diffsp + diffps + diffns;
      unsigned diffD = diffsp + diffns;
      unsigned diffE = diffsn + diffps;

      if (diff < minB) {
        dirB = u;
        minB = diff;
      }

      if (y > 1) {
        const unsigned diff2pp = std::abs(src2p[x - 1] - srcpp[x - 1 - u]) +
                                 std::abs(src2p[x] - srcpp[x - u]) +
                                 std::abs(src2p[x + 1] - srcpp[x + 1 - u]);
        const unsigned diffp2p = std::abs(srcpp[x - 1] - src2p[x - 1 + u]) +
                                 std::abs(srcpp[x] - src2p[x + u]) +
                                 std::abs(srcpp[x + 1] - src2p[x + 1 + u]);
        const unsigned diffA = diff + diff2pp + diffp2p;
        diffD += diffp2p;
        diffE += diff2pp;

        if (diffA < minA) {
          dirA = u;
          minA = diffA;
        }
      }

      if (y < height - 2) {
        const unsigned diff2nn = std::abs(src2n[x - 1] - srcpn[x - 1 + u]) +
                                 std::abs(src2n[x] - srcpn[x + u]) +
                                 std::abs(src2n[x + 1] - srcpn[x + 1 + u]);
        const unsigned diffn2n = std::abs(srcpn[x - 1] - src2n[x - 1 - u]) +
                                 std::abs(srcpn[x] - src2n[x - u]) +
                                 std::abs(srcpn[x + 1] - src2n[x + 1 - u]);
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

  int order[5];
  unsigned k = 0;

  if (dirA != -5000)
    order[k++] = dirA;
  if (dirB != -5000)
    order[k++] = dirB;
  if (dirC != -5000)
    order[k++] = dirC;
  if (dirD != -5000)
    order[k++] = dirD;
  if (dirE != -5000)
    order[k++] = dirE;

  for (auto t = k; t < 5; ++t)
    order[t] = std::numeric_limits<int>::max();

  if (k > 1) {
    CMPSWAP(order, 0, 1);
    CMPSWAP(order, 3, 4);
    CMPSWAP(order, 2, 4);
    CMPSWAP(order, 2, 3);
    CMPSWAP(order, 0, 3);
    CMPSWAP(order, 0, 2);
    CMPSWAP(order, 1, 4);
    CMPSWAP(order, 1, 3);
    CMPSWAP(order, 1, 2);

    const int mid =
        (k & 1) ? order[k / 2] : (order[(k - 1) / 2] + order[k / 2] + 1) / 2;
    const int lim = std::max(limlut[std::abs(mid)] / 4, 2);
    int sum = 0;
    unsigned count = 0;

    for (unsigned i = 0; i < k; i++) {
      if (std::abs(order[i] - mid) <= lim) {
        sum += order[i];
        count++;
      }
    }

    out = (count > 1)
              ? neutral + (static_cast<int>(static_cast<float>(sum) / count)
                           << shift2)
              : neutral;
  } else {
    out = neutral;
  }
}

template <typename T>
__global__ void filterDirMap(const T *msk, const T *dmsk, T *dst) {
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

  int order[9];
  unsigned u = 0;

  if (dmskpp[x - 1] != peak)
    order[u++] = dmskpp[x - 1];
  if (dmskpp[x] != peak)
    order[u++] = dmskpp[x];
  if (dmskpp[x + 1] != peak)
    order[u++] = dmskpp[x + 1];
  if (dmskp[x - 1] != peak)
    order[u++] = dmskp[x - 1];
  if (dmskp[x] != peak)
    order[u++] = dmskp[x];
  if (dmskp[x + 1] != peak)
    order[u++] = dmskp[x + 1];
  if (dmskpn[x - 1] != peak)
    order[u++] = dmskpn[x - 1];
  if (dmskpn[x] != peak)
    order[u++] = dmskpn[x];
  if (dmskpn[x + 1] != peak)
    order[u++] = dmskpn[x + 1];

  if (u < 4) {
    out = peak;
    return;
  }

  for (auto t = u; t < 9; ++t)
    order[t] = std::numeric_limits<int>::max();

  CMPSWAP(order, 0, 1);
  CMPSWAP(order, 2, 3);
  CMPSWAP(order, 0, 2);
  CMPSWAP(order, 1, 3);
  CMPSWAP(order, 1, 2);
  CMPSWAP(order, 4, 5);
  CMPSWAP(order, 7, 8);
  CMPSWAP(order, 6, 8);
  CMPSWAP(order, 6, 7);
  CMPSWAP(order, 4, 7);
  CMPSWAP(order, 4, 6);
  CMPSWAP(order, 5, 8);
  CMPSWAP(order, 5, 7);
  CMPSWAP(order, 5, 6);
  CMPSWAP(order, 0, 5);
  CMPSWAP(order, 0, 4);
  CMPSWAP(order, 1, 6);
  CMPSWAP(order, 1, 5);
  CMPSWAP(order, 1, 4);
  CMPSWAP(order, 2, 7);
  CMPSWAP(order, 3, 8);
  CMPSWAP(order, 3, 7);
  CMPSWAP(order, 2, 5);
  CMPSWAP(order, 2, 4);
  CMPSWAP(order, 3, 6);
  CMPSWAP(order, 3, 5);
  CMPSWAP(order, 3, 4);

  const int mid =
      (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[std::abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  unsigned count = 0;

  for (unsigned i = 0; i < u; i++) {
    if (std::abs(order[i] - mid) <= lim) {
      sum += order[i];
      count++;
    }
  }

  if (count < 4 || (count < 5 && dmskp[x] == peak)) {
    out = peak;
    return;
  }

  out = static_cast<int>(static_cast<float>(sum + mid) / (count + 1) + 0.5f);
}

template <typename T>
__global__ void expandDirMap(const T *msk, const T *dmsk, T *dst) {
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

  int order[8];
  unsigned u = 0;

  if (dmskpp[x - 1] != peak)
    order[u++] = dmskpp[x - 1];
  if (dmskpp[x] != peak)
    order[u++] = dmskpp[x];
  if (dmskpp[x + 1] != peak)
    order[u++] = dmskpp[x + 1];
  if (dmskp[x - 1] != peak)
    order[u++] = dmskp[x - 1];
  if (dmskp[x + 1] != peak)
    order[u++] = dmskp[x + 1];
  if (dmskpn[x - 1] != peak)
    order[u++] = dmskpn[x - 1];
  if (dmskpn[x] != peak)
    order[u++] = dmskpn[x];
  if (dmskpn[x + 1] != peak)
    order[u++] = dmskpn[x + 1];

  if (u < 5)
    return;

  for (auto t = u; t < 8; ++t)
    order[t] = std::numeric_limits<int>::max();

  CMPSWAP(order, 0, 1);
  CMPSWAP(order, 2, 3);
  CMPSWAP(order, 0, 2);
  CMPSWAP(order, 1, 3);
  CMPSWAP(order, 1, 2);
  CMPSWAP(order, 4, 5);
  CMPSWAP(order, 6, 7);
  CMPSWAP(order, 4, 6);
  CMPSWAP(order, 5, 7);
  CMPSWAP(order, 5, 6);
  CMPSWAP(order, 0, 4);
  CMPSWAP(order, 1, 5);
  CMPSWAP(order, 1, 4);
  CMPSWAP(order, 2, 6);
  CMPSWAP(order, 3, 7);
  CMPSWAP(order, 3, 6);
  CMPSWAP(order, 2, 4);
  CMPSWAP(order, 3, 5);
  CMPSWAP(order, 3, 4);

  const int mid =
      (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[std::abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  unsigned count = 0;

  for (unsigned i = 0; i < u; i++) {
    if (std::abs(order[i] - mid) <= lim) {
      sum += order[i];
      count++;
    }
  }

  if (count < 5)
    return;

  out = static_cast<int>(static_cast<float>(sum + mid) / (count + 1) + 0.5f);
}

template <typename T>
__global__ void filterMap(const T *msk, const T *dmsk, T *dst) {
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
  const int lim = std::max(std::abs(dir) * 2, twleve + 0);
  dir >>= shift;
  bool ict = false, icb = false;

  if (dir < 0) {
    for (int j = std::max(-(int)x, dir); j <= 0; j++) {
      if ((std::abs(dmskpp[x + j] - dmskp[x]) > lim && dmskpp[x + j] != peak) ||
          (dmskp[x + j] == peak && dmskpp[x + j] == peak) ||
          (std::abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
        ict = true;
        break;
      }
    }
  } else {
    for (int j = 0; j <= std::min((int)width - (int)x - 1, dir); j++) {
      if ((std::abs(dmskpp[x + j] - dmskp[x]) > lim && dmskpp[x + j] != peak) ||
          (dmskp[x + j] == peak && dmskpp[x + j] == peak) ||
          (std::abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
        ict = true;
        break;
      }
    }
  }

  if (ict) {
    if (dir < 0) {
      for (int j = 0; j <= std::min((int)width - (int)x - 1, std::abs(dir));
           j++) {
        if ((std::abs(dmskpn[x + j] - dmskp[x]) > lim &&
             dmskpn[x + j] != peak) ||
            (dmskpn[x + j] == peak && dmskp[x + j] == peak) ||
            (std::abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
          icb = true;
          break;
        }
      }
    } else {
      for (int j = std::max(-(int)x, -dir); j <= 0; j++) {
        if ((std::abs(dmskpn[x + j] - dmskp[x]) > lim &&
             dmskpn[x + j] != peak) ||
            (dmskpn[x + j] == peak && dmskp[x + j] == peak) ||
            (std::abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
          icb = true;
          break;
        }
      }
    }

    if (icb)
      out = peak;
  }
}

template <typename T>
void VS_CC eedi2Init(VSMap *_in, VSMap *_out, void **instanceData, VSNode *node,
                     VSCore *_core, const VSAPI *vsapi) {
  auto d = static_cast<EEDI2Instance<T> *>(*instanceData);
  d->initNode(node, vsapi);
}

template <typename T>
const VSFrameRef *VS_CC eedi2GetFrame(int n, int activationReason,
                                      void **instanceData, void **_frameData,
                                      VSFrameContext *frameCtx, VSCore *core,
                                      const VSAPI *vsapi) {
  auto d = static_cast<EEDI2Instance<T> *>(*instanceData);
  try {
    return d->getFrame(n, activationReason, frameCtx, core, vsapi);
  } catch (const std::exception &exc) {
    vsapi->setFilterError(("EEDI2CUDA: "s + exc.what()).c_str(), frameCtx);
    return nullptr;
  }
}

template <typename T>
void VS_CC eedi2Free(void *instanceData, VSCore *_core, const VSAPI *vsapi) {
  auto d = static_cast<EEDI2Instance<T> *>(instanceData);
  delete d;
}

template <typename T>
void eedi2CreateInner(const VSMap *in, VSMap *out, const VSAPI *vsapi,
                      VSCore *core) {
  try {
    vsapi->createFilter(in, out, "EEDI2", eedi2Init<T>, eedi2GetFrame<T>,
                        eedi2Free<T>, fmParallelRequests, 0,
                        new EEDI2Instance<T>(in, vsapi), core);
  } catch (const std::exception &exc) {
    vsapi->setError(out, ("EEDI2CUDA: "s + exc.what()).c_str());
    return;
  }
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
