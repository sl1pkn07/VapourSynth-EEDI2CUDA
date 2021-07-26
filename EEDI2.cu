#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include <stdint.h>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/sync/semaphore.hpp>

#include <VSHelper.h>
#include <VapourSynth.h>

using namespace std::literals::string_literals;

class CUDAError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
#define try_cuda(expr)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t __err = expr;                                                                                          \
    if (__err != cudaSuccess) {                                                                                        \
      throw CUDAError("'"s + #expr + " failed: " + cudaGetErrorString(__err));                                         \
    }                                                                                                                  \
  } while (0)

[[noreturn]] void unreachable() { assert(false); }

struct EEDI2Param {
  uint32_t d_pitch;
  uint32_t nt4, nt7, nt8, nt13, nt19;
  uint16_t mthresh, lthresh, vthresh;
  uint16_t width, height;
  uint8_t field;
  uint8_t estr, dstr, maxd;
  uint8_t subSampling;
};

template <typename T> class EEDI2Instance {
  std::unique_ptr<VSNodeRef, void (*const)(VSNodeRef *)> node;
  const VSVideoInfo *vi;
  std::unique_ptr<VSVideoInfo> vi2;
  EEDI2Param d;
  cudaStream_t stream;

  T *dst, *msk, *tmp, *src;
  T *dst2, *dst2M, *tmp2, *tmp2_2, *tmp2_3, *msk2;

  T *h_src, *h_dst;

  uint8_t map, pp, field, fieldS;
  uint32_t d_pitch;

public:
  EEDI2Instance(const VSMap *in, const VSAPI *vsapi)
      : node(vsapi->propGetNode(in, "clip", 0, nullptr), vsapi->freeNode) {
    initParams(in, vsapi);
    initCuda();
  }

  EEDI2Instance(const EEDI2Instance &other, const VSAPI *vsapi)
      : node(vsapi->cloneNodeRef(other.node.get()), vsapi->freeNode), vi(other.vi),
        vi2(std::make_unique<VSVideoInfo>(*other.vi2)), d(other.d), map(other.map), pp(other.pp), field(other.field),
        fieldS(other.fieldS), d_pitch(other.d_pitch) {
    initCuda();
  }

  ~EEDI2Instance() {
    try_cuda(cudaFree(dst));
    try_cuda(cudaFree(dst2));
    try_cuda(cudaFreeHost(h_src));
    try_cuda(cudaFreeHost(h_dst));
    try_cuda(cudaStreamDestroy(stream));
  }

private:
  void initParams(const VSMap *in, const VSAPI *vsapi) {
    using invalid_arg = std::invalid_argument;
    using boost::numeric_cast;

    vi = vsapi->getVideoInfo(node.get());
    vi2 = std::make_unique<VSVideoInfo>(*vi);
    const auto &fmt = *vi->format;

    if (!isConstantFormat(vi) || fmt.sampleType != stInteger || fmt.bytesPerSample > 2)
      throw invalid_arg("only constant format 8-16 bits integer input supported");
    if (vi->width < 8 || vi->height < 7)
      throw invalid_arg("clip resolution too low");

    auto propGetIntDefault = [&](const char *key, int64_t def) {
      int err;
      auto ret = vsapi->propGetInt(in, key, 0, &err);
      return err ? def : ret;
    };

    field = numeric_cast<uint8_t>(vsapi->propGetInt(in, "field", 0, nullptr));

    d.mthresh = numeric_cast<uint8_t>(propGetIntDefault("mthresh", 10));
    d.lthresh = numeric_cast<uint8_t>(propGetIntDefault("lthresh", 20));
    d.vthresh = numeric_cast<uint8_t>(propGetIntDefault("vthresh", 20));

    d.estr = numeric_cast<uint8_t>(propGetIntDefault("estr", 2));
    d.dstr = numeric_cast<uint8_t>(propGetIntDefault("dstr", 4));
    d.maxd = numeric_cast<uint8_t>(propGetIntDefault("maxd", 24));

    map = numeric_cast<uint8_t>(propGetIntDefault("map", 0));
    pp = numeric_cast<uint8_t>(propGetIntDefault("pp", 1));

    uint16_t nt = numeric_cast<uint8_t>(propGetIntDefault("nt", 50));

    if (field > 3)
      throw invalid_arg("field must be 0, 1, 2 or 3");
    if (d.maxd < 1 || d.maxd > 29)
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

    d.mthresh *= d.mthresh;
    d.vthresh *= 81;

    nt <<= sizeof(T) * 8 - 8;
    d.nt4 = nt * 4;
    d.nt7 = nt * 7;
    d.nt8 = nt * 8;
    d.nt13 = nt * 13;
    d.nt19 = nt * 19;
  }

  void initCuda() {
    // create stream
    try_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // alloc mem
    constexpr size_t numMem = 4;
    constexpr size_t numMem2x = 6;
    T **mem = &dst;
    size_t pitch;
    auto width = vi->width;
    auto height = vi->height;
    try_cuda(cudaMallocPitch(&mem[0], &pitch, width * sizeof(T), height * numMem));
    d_pitch = static_cast<uint32_t>(pitch);
    for (size_t i = 1; i < numMem; ++i)
      mem[i] = reinterpret_cast<T *>(reinterpret_cast<char *>(mem[i - 1]) + d_pitch * height);

    if (map == 0 || map == 3) {
      try_cuda(cudaMalloc(&dst2, d_pitch * height * numMem2x * 2));
      mem = &dst2;
      for (size_t i = 1; i < numMem2x; ++i)
        mem[i] = reinterpret_cast<T *>(reinterpret_cast<char *>(mem[i - 1]) + d_pitch * height * 2);
    }

    try_cuda(cudaHostAlloc(&h_src, d_pitch * height, cudaHostAllocWriteCombined));
    try_cuda(cudaHostAlloc(&h_dst, d_pitch * height * (map == 0 || map == 3 ? 2 : 1), cudaHostAllocDefault));
  }

public:
  const VSVideoInfo *getOutputVI() const { return vi2.get(); }

  VSFrameRef *getFrame(int n, int activationReason, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    if (activationReason == arInitial) {
      vsapi->requestFrameFilter(n, node.get(), frameCtx);
      return nullptr;
    } else if (activationReason != arAllFramesReady)
      return nullptr;

    auto field = this->field;
    if (fieldS > 1)
      field = (n & 1) ? (fieldS == 2 ? 1 : 0) : (fieldS == 2 ? 0 : 1);

    std::unique_ptr<const VSFrameRef, void (*const)(const VSFrameRef *)> src_frame{
        vsapi->getFrameFilter(n, node.get(), frameCtx), vsapi->freeFrame};
    std::unique_ptr<VSFrameRef, void (*const)(const VSFrameRef *)> dst_frame{
        vsapi->newVideoFrame(vi2->format, vi2->width, vi2->height, src_frame.get(), core), vsapi->freeFrame};

    for (int plane = 0; plane < vi->format->numPlanes; ++plane) {
      auto width = vsapi->getFrameWidth(src_frame.get(), plane);
      auto height = vsapi->getFrameHeight(src_frame.get(), plane);
      auto height2x = height * 2;
      auto s_pitch = vsapi->getStride(src_frame.get(), plane);
      auto width_bytes = width * vi->format->bytesPerSample;
      auto s_src = vsapi->getReadPtr(src_frame.get(), plane);
      auto s_dst = vsapi->getWritePtr(dst_frame.get(), plane);
      auto d_src = src;
      T *d_dst;
      auto d_pitch = this->d_pitch;
      int dst_height;

      d.field = static_cast<uint8_t>(field);
      d.width = static_cast<uint16_t>(width);
      d.height = static_cast<uint16_t>(height);
      d.subSampling = static_cast<uint8_t>(plane ? vi->format->subSamplingW : 0);
      d_pitch >>= d.subSampling;
      d.d_pitch = d_pitch;

      try_cuda(cudaMemcpy2DAsync(h_src, d_pitch, s_src, s_pitch, width_bytes, height, cudaMemcpyHostToHost, stream));
      try_cuda(cudaMemcpy2DAsync(d_src, d_pitch, h_src, d_pitch, width_bytes, height, cudaMemcpyHostToDevice, stream));

      dim3 blocks = dim3(64, 1);
      dim3 grids = dim3((width - 1) / blocks.x + 1, (height - 1) / blocks.y + 1);

      buildEdgeMask<<<grids, blocks, 0, stream>>>(d, src, msk);
      erode<<<grids, blocks, 0, stream>>>(d, msk, tmp);
      dilate<<<grids, blocks, 0, stream>>>(d, tmp, msk);
      erode<<<grids, blocks, 0, stream>>>(d, msk, tmp);
      removeSmallHorzGaps<<<grids, blocks, 0, stream>>>(d, tmp, msk);

      if (map != 1) {
        calcDirections<<<grids, blocks, 0, stream>>>(d, src, msk, tmp);
        filterDirMap<<<grids, blocks, 0, stream>>>(d, msk, tmp, dst);
        expandDirMap<<<grids, blocks, 0, stream>>>(d, msk, dst, tmp);
        filterMap<<<grids, blocks, 0, stream>>>(d, msk, tmp, dst);

        if (map != 2) {
          auto upscaleBy2 = [&](const T *src, T *dst) {
            try_cuda(cudaMemcpy2DAsync(dst + d_pitch * (1 - field), d_pitch * 2, src, d_pitch, width_bytes, height,
                                       cudaMemcpyDeviceToDevice, stream));
          };
          try_cuda(cudaMemset2DAsync(dst2, d_pitch, 0, width_bytes, height2x, stream));
          try_cuda(cudaMemset2DAsync(tmp2, d_pitch, 255, width_bytes, height2x, stream));
          upscaleBy2(src, dst2);
          upscaleBy2(dst, tmp2_2);
          upscaleBy2(msk, msk2);

          markDirections2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2_2, tmp2);
          filterDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2, dst2M);
          expandDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2);
          fillGaps2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2, tmp2_3);
          fillGaps2XStep2<<<grids, blocks, 0, stream>>>(d, msk2, tmp2, tmp2_3, dst2M);
          fillGaps2X<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2_3);
          fillGaps2XStep2<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2_3, tmp2);

          if (map != 3) {
            if (field)
              try_cuda(cudaMemcpyAsync(dst2 + d_pitch / sizeof(T) * (height2x - 1),
                                       dst2 + d_pitch / sizeof(T) * (height2x - 2), width_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            else
              try_cuda(
                  cudaMemcpyAsync(dst2, dst2 + d_pitch / sizeof(T), width_bytes, cudaMemcpyDeviceToDevice, stream));
            interpolateLattice<<<grids, blocks, 0, stream>>>(d, tmp2_2, tmp2, dst2);

            if (pp == 1) {
              try_cuda(cudaMemcpy2DAsync(tmp2_2, d_pitch, tmp2, d_pitch, width_bytes, height2x,
                                         cudaMemcpyDeviceToDevice, stream));
              filterDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2, dst2M);
              expandDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2);
              postProcess<<<grids, blocks, 0, stream>>>(d, tmp2, tmp2_2, dst2);
            } else if (pp != 0) {
              throw std::runtime_error("currently only pp == 1 is supported");
            }
          }
        }
      }

      if (map == 0) {
        d_dst = dst2;
        dst_height = height2x;
      } else if (map == 1) {
        d_dst = msk;
        dst_height = height;
      } else if (map == 3) {
        d_dst = tmp2;
        dst_height = height2x;
      } else {
        d_dst = dst;
        dst_height = height;
      }
      try_cuda(
          cudaMemcpy2DAsync(h_dst, d_pitch, d_dst, d_pitch, width_bytes, dst_height, cudaMemcpyDeviceToHost, stream));
      try_cuda(
          cudaMemcpy2DAsync(s_dst, s_pitch, h_dst, d_pitch, width_bytes, dst_height, cudaMemcpyHostToHost, stream));
      try_cuda(cudaStreamSynchronize(stream));
    }

    return dst_frame.release();
  }
};

template <typename T> struct EEDI2Data {
  using EEDI2Item = std::pair<EEDI2Instance<T>, std::atomic_flag>;

  unsigned num_streams;

private:
  boost::sync::semaphore semaphore;

  EEDI2Item *items() { return reinterpret_cast<EEDI2Item *>(this + 1); }

public:
  EEDI2Data() : semaphore(num_streams) {}

  EEDI2Instance<T> &firstInstance() { return items()[0].first; }

  EEDI2Instance<T> &acquireInstance() {
    if (num_streams == 1)
      return firstInstance();
    semaphore.wait();
    auto items = this->items();
    for (unsigned i = 0; i < num_streams; ++i) {
      if (!items[i].second.test_and_set())
        return items[i].first;
    }
    unreachable();
  }

  void releaseInstance(const EEDI2Instance<T> &instance) {
    if (num_streams == 1)
      return;
    auto items = this->items();
    for (unsigned i = 0; i < num_streams; ++i) {
      if (&instance == &items[i].first) {
        items[i].second.clear();
        break;
      }
    }
    semaphore.post();
  }

  static void *operator new(size_t sz, const VSMap *in, const VSAPI *vsapi) {
    int err;
    auto num_streams = vsapi->propGetInt(in, "num_streams", 0, &err);
    if (err)
      num_streams = 1;
    if (num_streams <= 0 || num_streams > 32)
      throw std::invalid_argument("num_streams must greater than 0 and less or equal to 32");
    auto *data = static_cast<EEDI2Data *>(::operator new(sizeof(EEDI2Data) + sizeof(EEDI2Item) * num_streams));
    try {
      data->num_streams = static_cast<unsigned>(num_streams);
      auto items = data->items();
      new (items) EEDI2Item(std::piecewise_construct, std::forward_as_tuple(in, vsapi), std::forward_as_tuple());
      items[0].second.clear();
      for (unsigned i = 1; i < num_streams; ++i) {
        new (items + i) EEDI2Item(std::piecewise_construct, std::forward_as_tuple(data->firstInstance(), vsapi),
                                  std::forward_as_tuple());
        items[i].second.clear();
      }
    } catch (...) {
      ::operator delete(data);
    }
    return data;
  }

  static void operator delete(void *p) {
    auto *data = static_cast<EEDI2Data *>(p);
    auto items = data->items();
    for (unsigned i = 0; i < data->num_streams; ++i)
      items[i].~EEDI2Item();
    ::operator delete(p);
  }
};

#define setup_kernel                                                                                                   \
  uint16_t width = d.width, height = d.height, x = threadIdx.x + blockIdx.x * blockDim.x,                              \
           y = threadIdx.y + blockIdx.y * blockDim.y;                                                                  \
  constexpr T shift = sizeof(T) * 8 - 8, peak = std::numeric_limits<T>::max(), ten = 10 << shift,                      \
              twleve = 12 << shift, eight = 8 << shift, twenty = 20 << shift, three = 3 << shift, nine = 9 << shift;   \
  constexpr T shift2 = shift + 2, neutral = peak / 2

__constant__ int8_t limlut[33]{6,  6,  7,  7,  8,  8,  9,  9,  9,  10, 10, 11, 11, 12, 12, 12, 12,
                               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, -1, -1};

#define setup_kernel2x                                                                                                 \
  setup_kernel;                                                                                                        \
  height *= 2;                                                                                                         \
  y = d.field ? 2 * y + 1 : 2 * y

#define bounds_check3(value, lower, upper)                                                                             \
  if ((value) < (lower) || (value) >= (upper))                                                                         \
  return

#define stride (d.d_pitch / sizeof(T))
#define line(p) ((p) + stride * y)
#define lineOff(p, off) ((p) + stride * (y + (off)))
#define point(p) ((p)[stride * y + x])

namespace bose {
template <typename T, size_t I, size_t J> __device__ __forceinline__ void P(T *arr) {
  T &a = arr[I - 1], &b = arr[J - 1];
  const T c = a;
  a = a < b ? a : b;
  b = c < b ? b : c;
}

template <typename T, size_t I, size_t X, size_t J, size_t Y> __device__ __forceinline__ void Pbracket(T *arr) {
  constexpr size_t A = X / 2, B = (X & 1) ? (Y / 2) : ((Y + 1) / 2);

  if constexpr (X == 1 && Y == 1)
    P<T, I, J>(arr);
  else if constexpr (X == 1 && Y == 2) {
    P<T, I, J + 1>(arr);
    P<T, I, J>(arr);
  } else if constexpr (X == 2 && Y == 1) {
    P<T, I, J>(arr);
    P<T, I + 1, J>(arr);
  } else {
    Pbracket<T, I, A, J, B>(arr);
    Pbracket<T, I + A, X - A, J + B, Y - B>(arr);
    Pbracket<T, I + A, X - A, J, B>(arr);
  }
}

template <typename T, size_t I, size_t M> __device__ __forceinline__ void Pstar(T *arr) {
  constexpr size_t A = M / 2;

  if constexpr (M > 1) {
    Pstar<T, I, A>(arr);
    Pstar<T, I + A, M - A>(arr);
    Pbracket<T, I, A, I + A, M - A>(arr);
  }
}
} // namespace bose

template <size_t N, typename T> __device__ __forceinline__ void BoseSort(T *arr) { bose::Pstar<T, 1, N>(arr); }

#define bose_sort_array(arr) BoseSort<sizeof(arr) / sizeof((arr)[0])>(arr)

template <typename T> __global__ void buildEdgeMask(const EEDI2Param d, const T *src, T *dst) {
  setup_kernel;

  auto srcp = line(src);
  auto srcpp = lineOff(src, -1);
  auto srcpn = lineOff(src, 1);
  auto &out = point(dst);

  out = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if ((std::abs(srcpp[x] - srcp[x]) < ten && std::abs(srcp[x] - srcpn[x]) < ten &&
       std::abs(srcpp[x] - srcpn[x]) < ten) ||
      (std::abs(srcpp[x - 1] - srcp[x - 1]) < ten && std::abs(srcp[x - 1] - srcpn[x - 1]) < ten &&
       std::abs(srcpp[x - 1] - srcpn[x - 1]) < ten && std::abs(srcpp[x + 1] - srcp[x + 1]) < ten &&
       std::abs(srcp[x + 1] - srcpn[x + 1]) < ten && std::abs(srcpp[x + 1] - srcpn[x + 1]) < ten))
    return;

  const unsigned sum = (srcpp[x - 1] + srcpp[x] + srcpp[x + 1] + srcp[x - 1] + srcp[x] + srcp[x + 1] + srcpn[x - 1] +
                        srcpn[x] + srcpn[x + 1]) >>
                       shift;
  const unsigned sumsq = (srcpp[x - 1] >> shift) * (srcpp[x - 1] >> shift) + (srcpp[x] >> shift) * (srcpp[x] >> shift) +
                         (srcpp[x + 1] >> shift) * (srcpp[x + 1] >> shift) +
                         (srcp[x - 1] >> shift) * (srcp[x - 1] >> shift) + (srcp[x] >> shift) * (srcp[x] >> shift) +
                         (srcp[x + 1] >> shift) * (srcp[x + 1] >> shift) +
                         (srcpn[x - 1] >> shift) * (srcpn[x - 1] >> shift) + (srcpn[x] >> shift) * (srcpn[x] >> shift) +
                         (srcpn[x + 1] >> shift) * (srcpn[x + 1] >> shift);
  if (9 * sumsq - sum * sum < d.vthresh)
    return;

  const unsigned Ix = std::abs(srcp[x + 1] - srcp[x - 1]) >> shift;
  const unsigned Iy =
      std::max({std::abs(srcpp[x] - srcpn[x]), std::abs(srcpp[x] - srcp[x]), std::abs(srcp[x] - srcpn[x])}) >> shift;
  if (Ix * Ix + Iy * Iy >= d.mthresh) {
    out = peak;
    return;
  }

  const unsigned Ixx = std::abs(srcp[x - 1] - 2 * srcp[x] + srcp[x + 1]) >> shift;
  const unsigned Iyy = std::abs(srcpp[x] - 2 * srcp[x] + srcpn[x]) >> shift;
  if (Ixx + Iyy >= d.lthresh)
    out = peak;
}

template <typename T> __global__ void erode(const EEDI2Param d, const T *msk, T *dst) {
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

  if (count < d.estr)
    out = 0;
}

template <typename T> __global__ void dilate(const EEDI2Param d, const T *msk, T *dst) {
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

  if (count >= d.dstr)
    out = peak;
}

template <typename T> __global__ void removeSmallHorzGaps(const EEDI2Param d, const T *msk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto &out = point(dst);

  out = mskp[x];

  bounds_check3(x, 3, width - 3);
  bounds_check3(y, 1, height - 1);

  if (mskp[x]) {
    if (mskp[x - 3] || mskp[x - 2] || mskp[x - 1] || mskp[x + 1] || mskp[x + 2] || mskp[x + 3])
      return;
    out = 0;
  } else {
    if ((mskp[x + 1] && (mskp[x - 1] || mskp[x - 2] || mskp[x - 3])) || (mskp[x + 2] && (mskp[x - 1] || mskp[x - 2])) ||
        (mskp[x + 3] && mskp[x - 1]))
      out = peak;
  }
}

template <typename T> __global__ void calcDirections(const EEDI2Param d, const T *src, const T *msk, T *dst) {
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

  const int maxd = d.maxd >> d.subSampling;

  if (mskp[x] != peak || (mskp[x - 1] != peak && mskp[x + 1] != peak))
    return;

  const int uStart = std::max(-x + 1, -maxd);
  const int uStop = std::min(width - 2 - x, maxd);
  const unsigned min0 = std::abs(srcp[x] - srcpn[x]) + std::abs(srcp[x] - srcpp[x]);
  unsigned minA = std::min(d.nt19, min0 * 9);
  unsigned minB = std::min(d.nt13, min0 * 6);
  unsigned minC = minA;
  unsigned minD = minB;
  unsigned minE = minB;
  int dirA = -5000, dirB = -5000, dirC = -5000, dirD = -5000, dirE = -5000;

  for (int u = uStart; u <= uStop; u++) {
    if ((y == 1 || mskpp[x - 1 + u] == peak || mskpp[x + u] == peak || mskpp[x + 1 + u] == peak) &&
        (y == height - 2 || mskpn[x - 1 - u] == peak || mskpn[x - u] == peak || mskpn[x + 1 - u] == peak)) {
      const unsigned diffsn = std::abs(srcp[x - 1] - srcpn[x - 1 - u]) + std::abs(srcp[x] - srcpn[x - u]) +
                              std::abs(srcp[x + 1] - srcpn[x + 1 - u]);
      const unsigned diffsp = std::abs(srcp[x - 1] - srcpp[x - 1 + u]) + std::abs(srcp[x] - srcpp[x + u]) +
                              std::abs(srcp[x + 1] - srcpp[x + 1 + u]);
      const unsigned diffps = std::abs(srcpp[x - 1] - srcp[x - 1 - u]) + std::abs(srcpp[x] - srcp[x - u]) +
                              std::abs(srcpp[x + 1] - srcp[x + 1 - u]);
      const unsigned diffns = std::abs(srcpn[x - 1] - srcp[x - 1 + u]) + std::abs(srcpn[x] - srcp[x + u]) +
                              std::abs(srcpn[x + 1] - srcp[x + 1 + u]);
      const unsigned diff = diffsn + diffsp + diffps + diffns;
      unsigned diffD = diffsp + diffns;
      unsigned diffE = diffsn + diffps;

      if (diff < minB) {
        dirB = u;
        minB = diff;
      }

      if (y > 1) {
        const unsigned diff2pp = std::abs(src2p[x - 1] - srcpp[x - 1 - u]) + std::abs(src2p[x] - srcpp[x - u]) +
                                 std::abs(src2p[x + 1] - srcpp[x + 1 - u]);
        const unsigned diffp2p = std::abs(srcpp[x - 1] - src2p[x - 1 + u]) + std::abs(srcpp[x] - src2p[x + u]) +
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
        const unsigned diff2nn = std::abs(src2n[x - 1] - srcpn[x - 1 + u]) + std::abs(src2n[x] - srcpn[x + u]) +
                                 std::abs(src2n[x + 1] - srcpn[x + 1 + u]);
        const unsigned diffn2n = std::abs(srcpn[x - 1] - src2n[x - 1 - u]) + std::abs(srcpn[x] - src2n[x - u]) +
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
    bose_sort_array(order);

    const int mid = (k & 1) ? order[k / 2] : (order[(k - 1) / 2] + order[k / 2] + 1) / 2;
    const int lim = std::max(limlut[std::abs(mid)] / 4, 2);
    int sum = 0;
    unsigned count = 0;

    for (unsigned i = 0; i < k; i++) {
      if (std::abs(order[i] - mid) <= lim) {
        sum += order[i];
        count++;
      }
    }

    out = (count > 1) ? neutral + (static_cast<int>(static_cast<float>(sum) / count) << shift2) : neutral;
  } else {
    out = neutral;
  }
}

template <typename T> __global__ void filterDirMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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

  bose_sort_array(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
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

template <typename T> __global__ void expandDirMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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

  bose_sort_array(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
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

template <typename T> __global__ void filterMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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
      for (int j = 0; j <= std::min((int)width - (int)x - 1, std::abs(dir)); j++) {
        if ((std::abs(dmskpn[x + j] - dmskp[x]) > lim && dmskpn[x + j] != peak) ||
            (dmskpn[x + j] == peak && dmskp[x + j] == peak) ||
            (std::abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
          icb = true;
          break;
        }
      }
    } else {
      for (int j = std::max(-(int)x, -dir); j <= 0; j++) {
        if ((std::abs(dmskpn[x + j] - dmskp[x]) > lim && dmskpn[x + j] != peak) ||
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

template <typename T> __global__ void markDirections2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel2x;

  auto mskp = lineOff(msk, -1);
  auto dmskp = lineOff(dmsk, -1);
  auto mskpn = lineOff(msk, 1);
  auto dmskpn = lineOff(dmsk, 1);
  auto &out = point(dst);

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (mskp[x] != peak && mskpn[x] != peak)
    return;

  int order[6];
  unsigned v = 0;

  if (dmskp[x - 1] != peak)
    order[v++] = dmskp[x - 1];
  if (dmskp[x] != peak)
    order[v++] = dmskp[x];
  if (dmskp[x + 1] != peak)
    order[v++] = dmskp[x + 1];
  if (dmskpn[x - 1] != peak)
    order[v++] = dmskpn[x - 1];
  if (dmskpn[x] != peak)
    order[v++] = dmskpn[x];
  if (dmskpn[x + 1] != peak)
    order[v++] = dmskpn[x + 1];

  if (v < 3) {
    return;
  } else {
    for (auto t = v; t < 6; ++t)
      order[t] = std::numeric_limits<int>::max();
    bose_sort_array(order);

    const int mid = (v & 1) ? order[v / 2] : (order[(v - 1) / 2] + order[v / 2] + 1) / 2;
    const int lim = limlut[std::abs(mid - neutral) >> shift2] << shift;
    int sum = 0;
    unsigned count = 0;

    unsigned u = 0;
    if (std::abs(dmskp[x - 1] - dmskpn[x - 1]) <= lim || dmskp[x - 1] == peak || dmskpn[x - 1] == peak)
      u++;
    if (std::abs(dmskp[x] - dmskpn[x]) <= lim || dmskp[x] == peak || dmskpn[x] == peak)
      u++;
    if (std::abs(dmskp[x + 1] - dmskpn[x - 1]) <= lim || dmskp[x + 1] == peak || dmskpn[x + 1] == peak)
      u++;
    if (u < 2)
      return;

    for (unsigned i = 0; i < v; i++) {
      if (std::abs(order[i] - mid) <= lim) {
        sum += order[i];
        count++;
      }
    }

    if (count < v - 2 || count < 2)
      return;

    out = static_cast<int>(static_cast<float>(sum + mid) / (count + 1) + 0.5f);
  }
}

template <typename T> __global__ void filterDirMap2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel2x;

  auto mskp = lineOff(msk, -1);
  auto dmskp = line(dmsk);
  auto mskpn = lineOff(msk, 1);
  auto dmskpn = lineOff(dmsk, 2);
  auto dmskpp = lineOff(dmsk, -2);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (mskp[x] != peak && mskpn[x] != peak)
    return;

  int order[9];
  unsigned u = 0;

  if (y > 1) {
    if (dmskpp[x - 1] != peak)
      order[u++] = dmskpp[x - 1];
    if (dmskpp[x] != peak)
      order[u++] = dmskpp[x];
    if (dmskpp[x + 1] != peak)
      order[u++] = dmskpp[x + 1];
  }

  if (dmskp[x - 1] != peak)
    order[u++] = dmskp[x - 1];
  if (dmskp[x] != peak)
    order[u++] = dmskp[x];
  if (dmskp[x + 1] != peak)
    order[u++] = dmskp[x + 1];

  if (y < height - 2) {
    if (dmskpn[x - 1] != peak)
      order[u++] = dmskpn[x - 1];
    if (dmskpn[x] != peak)
      order[u++] = dmskpn[x];
    if (dmskpn[x + 1] != peak)
      order[u++] = dmskpn[x + 1];
  }

  if (u < 4) {
    out = peak;
    return;
  }

  for (auto t = u; t < 9; ++t)
    order[t] = std::numeric_limits<int>::max();
  bose_sort_array(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
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

template <typename T> __global__ void expandDirMap2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
  setup_kernel2x;

  auto mskp = lineOff(msk, -1);
  auto dmskp = line(dmsk);
  auto mskpn = lineOff(msk, 1);
  auto dmskpn = lineOff(dmsk, 2);
  auto dmskpp = lineOff(dmsk, -2);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (dmskp[x] != peak || (mskp[x] != peak && mskpn[x] != peak))
    return;

  int order[9];
  unsigned u = 0;

  if (y > 1) {
    if (dmskpp[x - 1] != peak)
      order[u++] = dmskpp[x - 1];
    if (dmskpp[x] != peak)
      order[u++] = dmskpp[x];
    if (dmskpp[x + 1] != peak)
      order[u++] = dmskpp[x + 1];
  }

  if (dmskp[x - 1] != peak)
    order[u++] = dmskp[x - 1];
  if (dmskp[x + 1] != peak)
    order[u++] = dmskp[x + 1];

  if (y < height - 2) {
    if (dmskpn[x - 1] != peak)
      order[u++] = dmskpn[x - 1];
    if (dmskpn[x] != peak)
      order[u++] = dmskpn[x];
    if (dmskpn[x + 1] != peak)
      order[u++] = dmskpn[x + 1];
  }

  if (u < 5)
    return;

  for (auto t = u; t < 9; ++t)
    order[t] = std::numeric_limits<int>::max();
  bose_sort_array(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
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

template <typename T> __global__ void fillGaps2X(const EEDI2Param d, const T *msk, const T *dmsk, T *tmp) {
  setup_kernel2x;
  constexpr int fiveHundred = 500 << shift;

  auto mskp = lineOff(msk, -1);
  auto dmskp = line(dmsk);
  auto mskpn = lineOff(msk, 1);
  auto mskpp = lineOff(msk, -3);
  auto mskpnn = lineOff(msk, 3);
  auto dmskpn = lineOff(dmsk, 2);
  auto dmskpp = lineOff(dmsk, -2);
  auto &out_tmp = point(tmp);

  out_tmp = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if (dmskp[x] != peak || (mskp[x] != peak && mskpn[x] != peak))
    return;

  unsigned u = x - 1, v = x + 1;
  int back = fiveHundred, forward = -fiveHundred;

  while (u && x - u < 255) {
    if (dmskp[u] != peak) {
      back = dmskp[u];
      break;
    }
    if (mskp[u] != peak && mskpn[u] != peak)
      break;
    u--;
  }

  while (v < width && v - x < 255) {
    if (dmskp[v] != peak) {
      forward = dmskp[v];
      break;
    }
    if (mskp[v] != peak && mskpn[v] != peak)
      break;
    v++;
  }

  bool tc = true, bc = true;
  int mint = fiveHundred, maxt = -twenty;
  int minb = fiveHundred, maxb = -twenty;

  for (unsigned j = u; j <= v; j++) {
    if (tc) {
      if (y <= 2 || dmskpp[j] == peak || (mskpp[j] != peak && mskp[j] != peak)) {
        tc = false;
        mint = maxt = twenty;
      } else {
        if (dmskpp[j] < mint)
          mint = dmskpp[j];
        if (dmskpp[j] > maxt)
          maxt = dmskpp[j];
      }
    }

    if (bc) {
      if (y >= height - 3 || dmskpn[j] == peak || (mskpn[j] != peak && mskpnn[j] != peak)) {
        bc = false;
        minb = maxb = twenty;
      } else {
        if (dmskpn[j] < minb)
          minb = dmskpn[j];
        if (dmskpn[j] > maxb)
          maxb = dmskpn[j];
      }
    }
  }

  if (maxt == -twenty)
    maxt = mint = twenty;
  if (maxb == -twenty)
    maxb = minb = twenty;

  const int thresh = std::max({std::max(std::abs(forward - neutral), std::abs(back - neutral)) / 4, eight + 0,
                               std::abs(mint - maxt), std::abs(minb - maxb)});
  const unsigned lim = std::min(std::max(std::abs(forward - neutral), std::abs(back - neutral)) >> shift2, 6);
  if (std::abs(forward - back) <= thresh && (v - u - 1 <= lim || tc || bc)) {
    //    const float step = static_cast<float>(forward - back) / (v - u);
    //    for (unsigned j = 0; j < v - u - 1; j++)
    //      dstp[u + j + 1] = back + static_cast<int>(j * step + 0.5);
    out_tmp = (x - u) | (v - x) << 8u;
  }
}

template <typename T>
__global__ void fillGaps2XStep2(const EEDI2Param d, const T *msk, const T *dmsk, const T *tmp, T *dst) {
  setup_kernel2x;

  auto mskp = lineOff(msk, -1);
  auto dmskp = line(dmsk);
  auto mskpn = lineOff(msk, 1);
  auto tmpp = line(tmp);
  auto &out = point(dst);

  out = dmskp[x];

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  unsigned uv = 0, pos;

  for (unsigned i = max(x - 255, 1); i < x; ++i) {
    bool cond = i + (tmpp[i] >> 8) > x;
    uv = cond ? tmpp[i] : uv;
    pos = cond ? i : pos;
  }
  uv = tmpp[x] ? tmpp[x] : uv;
  pos = tmpp[x] ? x : pos;
  for (unsigned i = x + 1; i - x < 255 && i < width - 1; ++i) {
    bool cond = i - (tmpp[i] & 255u) < x;
    uv = cond ? tmpp[i] : uv;
    pos = cond ? i : pos;
  }

  if (!uv)
    return;

  int u = pos - (uv & 255);
  int v = pos + (uv >> 8);
  int back = dmskp[u];
  int forward = dmskp[v];

  out = back + __float2int_ru((forward - back) * (x - 1 - u) * 1.f / (v - u));
}

template <typename T> __global__ void interpolateLattice(const EEDI2Param d, const T *omsk, T *dmsk, T *dst) {
  setup_kernel2x;

  auto omskp = lineOff(omsk, -1);
  auto omskn = lineOff(omsk, 1);
  auto dmskp = line(dmsk);
  auto dstp = lineOff(dst, -1);
  auto dstpn = line(dst);
  auto dstpnn = lineOff(dst, 1);

  bounds_check3(x, 0, width);
  bounds_check3(y, 1, height - 1);

  int dir = dmskp[x];
  const int lim = limlut[std::abs(dir - neutral) >> shift2] << shift;

  if (dir == peak || (std::abs(dmskp[x] - dmskp[x - 1]) > lim && std::abs(dmskp[x] - dmskp[x + 1]) > lim)) {
    dstpn[x] = (dstp[x] + dstpnn[x] + 1) / 2;
    if (dir != peak)
      dmskp[x] = neutral;
    return;
  }

  if (lim < nine) {
    const unsigned sum = (dstp[x - 1] + dstp[x] + dstp[x + 1] + dstpnn[x - 1] + dstpnn[x] + dstpnn[x + 1]) >> shift;
    const unsigned sumsq =
        (dstp[x - 1] >> shift) * (dstp[x - 1] >> shift) + (dstp[x] >> shift) * (dstp[x] >> shift) +
        (dstp[x + 1] >> shift) * (dstp[x + 1] >> shift) + (dstpnn[x - 1] >> shift) * (dstpnn[x - 1] >> shift) +
        (dstpnn[x] >> shift) * (dstpnn[x] >> shift) + (dstpnn[x + 1] >> shift) * (dstpnn[x + 1] >> shift);
    if (6 * sumsq - sum * sum < 576) {
      dstpn[x] = (dstp[x] + dstpnn[x] + 1) / 2;
      dmskp[x] = peak;
      return;
    }
  }

  if (x > 1 && x < width - 2 &&
      ((dstp[x] < std::max(dstp[x - 2], dstp[x - 1]) - three && dstp[x] < std::max(dstp[x + 2], dstp[x + 1]) - three &&
        dstpnn[x] < std::max(dstpnn[x - 2], dstpnn[x - 1]) - three &&
        dstpnn[x] < std::max(dstpnn[x + 2], dstpnn[x + 1]) - three) ||
       (dstp[x] > std::min(dstp[x - 2], dstp[x - 1]) + three && dstp[x] > std::min(dstp[x + 2], dstp[x + 1]) + three &&
        dstpnn[x] > std::min(dstpnn[x - 2], dstpnn[x - 1]) + three &&
        dstpnn[x] > std::min(dstpnn[x + 2], dstpnn[x + 1]) + three))) {
    dstpn[x] = (dstp[x] + dstpnn[x] + 1) / 2;
    dmskp[x] = neutral;
    return;
  }

  dir = (dir - neutral + (1 << (shift2 - 1))) >> shift2;
  const int uStart =
      (dir - 2 < 0) ? std::max({-x + 1, dir - 2, -width + 2 + x}) : std::min({x - 1, dir - 2, width - 2 - x});
  const int uStop =
      (dir + 2 < 0) ? std::max({-x + 1, dir + 2, -width + 2 + x}) : std::min({x - 1, dir + 2, width - 2 - x});
  unsigned min = d.nt8;
  unsigned val = (dstp[x] + dstpnn[x] + 1) / 2;

  for (int u = uStart; u <= uStop; u++) {
    const unsigned diff = std::abs(dstp[x - 1] - dstpnn[x - u - 1]) + std::abs(dstp[x] - dstpnn[x - u]) +
                          std::abs(dstp[x + 1] - dstpnn[x - u + 1]) + std::abs(dstpnn[x - 1] - dstp[x + u - 1]) +
                          std::abs(dstpnn[x] - dstp[x + u]) + std::abs(dstpnn[x + 1] - dstp[x + u + 1]);
    if (diff < min &&
        ((omskp[x - 1 + u] != peak && std::abs(omskp[x - 1 + u] - dmskp[x]) <= lim) ||
         (omskp[x + u] != peak && std::abs(omskp[x + u] - dmskp[x]) <= lim) ||
         (omskp[x + 1 + u] != peak && std::abs(omskp[x + 1 + u] - dmskp[x]) <= lim)) &&
        ((omskn[x - 1 - u] != peak && std::abs(omskn[x - 1 - u] - dmskp[x]) <= lim) ||
         (omskn[x - u] != peak && std::abs(omskn[x - u] - dmskp[x]) <= lim) ||
         (omskn[x + 1 - u] != peak && std::abs(omskn[x + 1 - u] - dmskp[x]) <= lim))) {
      const unsigned diff2 = std::abs(dstp[x + u / 2 - 1] - dstpnn[x - u / 2 - 1]) +
                             std::abs(dstp[x + u / 2] - dstpnn[x - u / 2]) +
                             std::abs(dstp[x + u / 2 + 1] - dstpnn[x - u / 2 + 1]);
      if (diff2 < d.nt4 && (((std::abs(omskp[x + u / 2] - omskn[x - u / 2]) <= lim ||
                              std::abs(omskp[x + u / 2] - omskn[x - ((u + 1) / 2)]) <= lim) &&
                             omskp[x + u / 2] != peak) ||
                            ((std::abs(omskp[x + ((u + 1) / 2)] - omskn[x - u / 2]) <= lim ||
                              std::abs(omskp[x + ((u + 1) / 2)] - omskn[x - ((u + 1) / 2)]) <= lim) &&
                             omskp[x + ((u + 1) / 2)] != peak))) {
        if ((std::abs(dmskp[x] - omskp[x + u / 2]) <= lim || std::abs(dmskp[x] - omskp[x + ((u + 1) / 2)]) <= lim) &&
            (std::abs(dmskp[x] - omskn[x - u / 2]) <= lim || std::abs(dmskp[x] - omskn[x - ((u + 1) / 2)]) <= lim)) {
          val = (dstp[x + u / 2] + dstp[x + ((u + 1) / 2)] + dstpnn[x - u / 2] + dstpnn[x - ((u + 1) / 2)] + 2) / 4;
          min = diff;
          dir = u;
        }
      }
    }
  }

  if (min != d.nt8) {
    dstpn[x] = val;
    dmskp[x] = neutral + (dir << shift2);
  } else {
    const int dt = 4 >> d.subSampling;
    const int uStart2 = std::max(-x + 1, -dt);
    const int uStop2 = std::min(width - 2 - x, dt);
    const unsigned minm = std::min(dstp[x], dstpnn[x]);
    const unsigned maxm = std::max(dstp[x], dstpnn[x]);
    min = d.nt7;

    for (int u = uStart2; u <= uStop2; u++) {
      const int p1 = dstp[x + u / 2] + dstp[x + ((u + 1) / 2)];
      const int p2 = dstpnn[x - u / 2] + dstpnn[x - ((u + 1) / 2)];
      const unsigned diff = std::abs(dstp[x - 1] - dstpnn[x - u - 1]) + std::abs(dstp[x] - dstpnn[x - u]) +
                            std::abs(dstp[x + 1] - dstpnn[x - u + 1]) + std::abs(dstpnn[x - 1] - dstp[x + u - 1]) +
                            std::abs(dstpnn[x] - dstp[x + u]) + std::abs(dstpnn[x + 1] - dstp[x + u + 1]) +
                            std::abs(p1 - p2);
      if (diff < min) {
        const unsigned valt = (p1 + p2 + 2) / 4;
        if (valt >= minm && valt <= maxm) {
          val = valt;
          min = diff;
          dir = u;
        }
      }
    }

    dstpn[x] = val;
    dmskp[x] = (min == d.nt7) ? neutral : neutral + (dir << shift2);
  }
}

template <typename T> __global__ void postProcess(const EEDI2Param d, const T *nmsk, const T *omsk, T *dst) {
  setup_kernel2x;

  auto nmskp = line(nmsk);
  auto omskp = line(omsk);
  auto &out = point(dst);
  auto dstpp = lineOff(dst, -1);
  auto dstpn = lineOff(dst, 1);

  bounds_check3(x, 0, width);
  bounds_check3(y, 1, height - 1);

  const int lim = limlut[std::abs(nmskp[x] - neutral) >> shift2] << shift;
  if (std::abs(nmskp[x] - omskp[x]) > lim && omskp[x] != peak && omskp[x] != neutral)
    out = (dstpp[x] + dstpn[x] + 1) / 2;
}

template <typename T>
void VS_CC eedi2Init(VSMap *_in, VSMap *_out, void **instanceData, VSNode *node, VSCore *_core, const VSAPI *vsapi) {
  auto data = static_cast<EEDI2Data<T> *>(*instanceData);
  vsapi->setVideoInfo(data->firstInstance().getOutputVI(), 1, node);
}

template <typename T>
const VSFrameRef *VS_CC eedi2GetFrame(int n, int activationReason, void **instanceData, void **_frameData,
                                      VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

  auto data = static_cast<EEDI2Data<T> *>(*instanceData);
  const VSFrameRef *out = nullptr;

  if (activationReason == arInitial) {
    out = data->firstInstance().getFrame(n, activationReason, frameCtx, core, vsapi);
  } else {
    auto &d = data->acquireInstance();
    try {
      out = d.getFrame(n, activationReason, frameCtx, core, vsapi);
    } catch (const std::exception &exc) {
      vsapi->setFilterError(("EEDI2CUDA: "s + exc.what()).c_str(), frameCtx);
    }
    data->releaseInstance(d);
  }

  return out;
}

template <typename T> void VS_CC eedi2Free(void *instanceData, VSCore *_core, const VSAPI *vsapi) {
  auto data = static_cast<EEDI2Data<T> *>(instanceData);
  delete data;
}

template <typename T> void eedi2CreateInner(const VSMap *in, VSMap *out, const VSAPI *vsapi, VSCore *core) {
  try {
    auto data = new (in, vsapi) EEDI2Data<T>;
    vsapi->createFilter(in, out, "EEDI2", eedi2Init<T>, eedi2GetFrame<T>, eedi2Free<T>,
                        data->num_streams > 1 ? fmParallel : fmParallelRequests, 0, data, core);
  } catch (const std::exception &exc) {
    vsapi->setError(out, ("EEDI2CUDA: "s + exc.what()).c_str());
    return;
  }
}

void VS_CC eedi2Create(const VSMap *in, VSMap *out, void *_userData, VSCore *core, const VSAPI *vsapi) {
  VSNodeRef *node = vsapi->propGetNode(in, "clip", 0, nullptr);
  const VSVideoInfo *vi = vsapi->getVideoInfo(node);
  vsapi->freeNode(node);
  if (vi->format->bytesPerSample == 1)
    eedi2CreateInner<uint8_t>(in, out, vsapi, core);
  else
    eedi2CreateInner<uint16_t>(in, out, vsapi, core);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
  configFunc("club.amusement.eedi2cuda", "eedi2cuda", "EEDI2 filter using CUDA", VAPOURSYNTH_API_VERSION, 1, plugin);
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
               "pp:int:opt;"
               "num_streams:int:opt",
               eedi2Create, nullptr, plugin);
}
