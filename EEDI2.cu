#include <atomic>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/sync/semaphore.hpp>

#include <VSHelper.h>
#include <VapourSynth.h>

using namespace std::literals::string_literals;

class CUDAError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
#define try_cuda(expr)                                                                                                                     \
  do {                                                                                                                                     \
    cudaError_t __err = expr;                                                                                                              \
    if (__err != cudaSuccess) {                                                                                                            \
      throw CUDAError("'"s + #expr + " failed: " + cudaGetErrorString(__err));                                                             \
    }                                                                                                                                      \
  } while (0)

[[noreturn]] void unreachable() { assert(false); }

template <typename Td, typename Ts> void numeric_cast_to(Td &dst, Ts src) { dst = boost::numeric_cast<Td>(src); }
template <typename Td, typename Ts> void narrow_cast_to(Td &dst, Ts src) { dst = static_cast<Td>(src); }

struct EEDI2Param {
  unsigned d_pitch;
  unsigned nt4, nt7, nt8, nt13, nt19;
  unsigned mthresh, lthresh, vthresh;
  unsigned field;
  unsigned estr, dstr;
  unsigned maxd, subSampling;
  int width, height;
};

template <typename T> class EEDI2Instance {
  std::unique_ptr<VSNodeRef, void (*const)(VSNodeRef *)> node;
  const VSVideoInfo *vi;
  std::unique_ptr<VSVideoInfo> vi2;
  EEDI2Param d;
  cudaStream_t stream;

  T *dst, *msk, *src, *tmp;
  T *dst2, *dst2M, *tmp2, *tmp2_2, *tmp2_3, *msk2;

  T *h_src, *h_dst;

  unsigned map, pp, field, fieldS;
  unsigned d_pitch;

public:
  EEDI2Instance(const VSMap *in, const VSAPI *vsapi) : node(vsapi->propGetNode(in, "clip", 0, nullptr), vsapi->freeNode) {
    initParams(in, vsapi);
    initCuda();
    tuneKernels();
  }

  EEDI2Instance(const EEDI2Instance &other, const VSAPI *vsapi)
      : node(vsapi->cloneNodeRef(other.node.get()), vsapi->freeNode), vi(other.vi), vi2(std::make_unique<VSVideoInfo>(*other.vi2)),
        d(other.d), map(other.map), pp(other.pp), field(other.field), fieldS(other.fieldS), d_pitch(other.d_pitch) {
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

    numeric_cast_to(field, vsapi->propGetInt(in, "field", 0, nullptr));

    numeric_cast_to(d.mthresh, propGetIntDefault("mthresh", 10));
    numeric_cast_to(d.lthresh, propGetIntDefault("lthresh", 20));
    numeric_cast_to(d.vthresh, propGetIntDefault("vthresh", 20));

    numeric_cast_to(d.estr, propGetIntDefault("estr", 2));
    numeric_cast_to(d.dstr, propGetIntDefault("dstr", 4));
    numeric_cast_to(d.maxd, propGetIntDefault("maxd", 24));

    numeric_cast_to(map, propGetIntDefault("map", 0));
    numeric_cast_to(pp, propGetIntDefault("pp", 1));

    unsigned nt;
    numeric_cast_to(nt, propGetIntDefault("nt", 50));

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
    try {
      try_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    } catch (const CUDAError &exc) {
      throw CUDAError(exc.what() + " Please upgrade your driver."s);
    }

    // alloc mem
    constexpr size_t numMem = 4;
    constexpr size_t numMem2x = 6;
    T **mem = &dst;
    size_t pitch;
    auto width = vi->width;
    auto height = vi->height;
    try_cuda(cudaMallocPitch(&mem[0], &pitch, width * sizeof(T), height * numMem));
    narrow_cast_to(d_pitch, pitch);
    for (size_t i = 1; i < numMem; ++i)
      mem[i] = reinterpret_cast<T *>(reinterpret_cast<char *>(mem[i - 1]) + d_pitch * height);

    if (map == 0 || map == 3) {
      try_cuda(cudaMalloc(&dst2, d_pitch * height * numMem2x * 2));
      mem = &dst2;
      for (size_t i = 1; i < numMem2x; ++i)
        mem[i] = reinterpret_cast<T *>(reinterpret_cast<char *>(mem[i - 1]) + d_pitch * height * 2);
    } else {
      dst2 = nullptr;
    }

    try_cuda(cudaHostAlloc(&h_src, d_pitch * height, cudaHostAllocWriteCombined));
    try_cuda(cudaHostAlloc(&h_dst, d_pitch * height * (map == 0 || map == 3 ? 2 : 1), cudaHostAllocDefault));
  }

  void tuneKernels();

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

    std::unique_ptr<const VSFrameRef, void (*const)(const VSFrameRef *)> src_frame{vsapi->getFrameFilter(n, node.get(), frameCtx),
                                                                                   vsapi->freeFrame};
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

      d.field = field;
      d.width = width;
      d.height = height;
      d.subSampling = plane ? vi->format->subSamplingW : 0u;
      d_pitch >>= d.subSampling;
      d.d_pitch = d_pitch;

      try_cuda(cudaMemcpy2D(h_src, d_pitch, s_src, s_pitch, width_bytes, height, cudaMemcpyHostToHost));
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
              try_cuda(cudaMemcpyAsync(dst2 + d_pitch / sizeof(T) * (height2x - 1), dst2 + d_pitch / sizeof(T) * (height2x - 2),
                                       width_bytes, cudaMemcpyDeviceToDevice, stream));
            else
              try_cuda(cudaMemcpyAsync(dst2, dst2 + d_pitch / sizeof(T), width_bytes, cudaMemcpyDeviceToDevice, stream));
            try_cuda(cudaMemcpy2DAsync(tmp2_3, d_pitch, tmp2, d_pitch, width_bytes, height2x, cudaMemcpyDeviceToDevice, stream));

            interpolateLattice<<<grids, blocks, 0, stream>>>(d, tmp2_2, tmp2, dst2, tmp2_3);

            if (pp == 1) {
              filterDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, tmp2_3, dst2M);
              expandDirMap2X<<<grids, blocks, 0, stream>>>(d, msk2, dst2M, tmp2);
              postProcess<<<grids, blocks, 0, stream>>>(d, tmp2, tmp2_3, dst2);
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
      try_cuda(cudaMemcpy2DAsync(h_dst, d_pitch, d_dst, d_pitch, width_bytes, dst_height, cudaMemcpyDeviceToHost, stream));
      try_cuda(cudaStreamSynchronize(stream));
      try_cuda(cudaMemcpy2D(s_dst, s_pitch, h_dst, d_pitch, width_bytes, dst_height, cudaMemcpyHostToHost));
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
  EEDI2Data(const VSMap *in, const VSAPI *vsapi) : semaphore(num_streams) {
    auto items = this->items();
    new (items) EEDI2Item(std::piecewise_construct, std::forward_as_tuple(in, vsapi), std::forward_as_tuple());
    items[0].second.clear();
    for (unsigned i = 1; i < num_streams; ++i) {
      new (items + i) EEDI2Item(std::piecewise_construct, std::forward_as_tuple(firstInstance(), vsapi), std::forward_as_tuple());
      items[i].second.clear();
    }
  }

  ~EEDI2Data() {
    auto items = this->items();
    for (unsigned i = 0; i < num_streams; ++i)
      items[i].~EEDI2Item();
  }

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

  static void *operator new(size_t sz, unsigned num_streams) {
    auto p = static_cast<EEDI2Data *>(::operator new(sz + sizeof(EEDI2Item) * num_streams));
    p->num_streams = num_streams;
    return p;
  }

  static void operator delete(void *p, unsigned num_streams) { ::operator delete(p); }

  static void operator delete(void *p) { ::operator delete(p); }
};

#define KERNEL __global__ __launch_bounds__(64)

#define setup_kernel                                                                                                                       \
  int width = d.width, height = d.height, x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;            \
  constexpr T shift = sizeof(T) * 8 - 8, peak = std::numeric_limits<T>::max(), ten = 10 << shift, twleve = 12 << shift,                    \
              eight = 8 << shift, twenty = 20 << shift, three = 3 << shift, nine = 9 << shift;                                             \
  constexpr T shift2 = shift + 2, neutral = peak / 2;                                                                                      \
  constexpr int intmax = std::numeric_limits<int>::max()

__device__ int limlut[33]{6,  6,  7,  7,  8,  8,  9,  9,  9,  10, 10, 11, 11, 12, 12, 12, 12,
                          12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, -1, -1};

#define setup_kernel2x                                                                                                                     \
  setup_kernel;                                                                                                                            \
  height *= 2;                                                                                                                             \
  y = d.field ? 2 * y + 1 : 2 * y

#define bounds_check3(value, lower, upper)                                                                                                 \
  if (static_cast<unsigned>(value) < (lower) || static_cast<unsigned>(value) >= (upper))                                                   \
  return

#define stride (d.d_pitch / sizeof(T))
#define line(p) ((p) + stride * y)
#define lineOff(p, off) ((p) + int(stride) * (y + (off)))
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

template <typename T> __device__ __forceinline__ T mmax(T last) { return last; }

template <typename T, typename... Args> __device__ __forceinline__ T mmax(T first, Args... remaining) {
  auto candidate = mmax(remaining...);
  static_assert(std::is_same<decltype(candidate), T>::value, "arguments must have the same type");
  return first < candidate ? candidate : first;
}

template <typename T> __device__ __forceinline__ T mmin(T last) { return last; }

template <typename T, typename... Args> __device__ __forceinline__ T mmin(T first, Args... remaining) {
  auto candidate = mmin(remaining...);
  static_assert(std::is_same<decltype(candidate), T>::value, "arguments must have the same type");
  return first < candidate ? first : candidate;
}

template <typename T> __device__ T round_div(T a, T b) {
  // XXX: only for same sign
  return (a + (b / 2)) / b;
}

template <typename T> KERNEL void buildEdgeMask(const EEDI2Param d, const T *src, T *dst) {
  setup_kernel;

  auto srcp = line(src);
  auto srcpp = lineOff(src, -1);
  auto srcpn = lineOff(src, 1);
  auto &out = point(dst);

  out = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  if ((abs(srcpp[x] - srcp[x]) < ten && abs(srcp[x] - srcpn[x]) < ten && abs(srcpp[x] - srcpn[x]) < ten) ||
      (abs(srcpp[x - 1] - srcp[x - 1]) < ten && abs(srcp[x - 1] - srcpn[x - 1]) < ten && abs(srcpp[x - 1] - srcpn[x - 1]) < ten &&
       abs(srcpp[x + 1] - srcp[x + 1]) < ten && abs(srcp[x + 1] - srcpn[x + 1]) < ten && abs(srcpp[x + 1] - srcpn[x + 1]) < ten))
    return;

  const unsigned sum =
      (srcpp[x - 1] + srcpp[x] + srcpp[x + 1] + srcp[x - 1] + srcp[x] + srcp[x + 1] + srcpn[x - 1] + srcpn[x] + srcpn[x + 1]) >> shift;
  const unsigned sumsq = (srcpp[x - 1] >> shift) * (srcpp[x - 1] >> shift) + (srcpp[x] >> shift) * (srcpp[x] >> shift) +
                         (srcpp[x + 1] >> shift) * (srcpp[x + 1] >> shift) + (srcp[x - 1] >> shift) * (srcp[x - 1] >> shift) +
                         (srcp[x] >> shift) * (srcp[x] >> shift) + (srcp[x + 1] >> shift) * (srcp[x + 1] >> shift) +
                         (srcpn[x - 1] >> shift) * (srcpn[x - 1] >> shift) + (srcpn[x] >> shift) * (srcpn[x] >> shift) +
                         (srcpn[x + 1] >> shift) * (srcpn[x + 1] >> shift);
  if (9 * sumsq - sum * sum < d.vthresh)
    return;

  const unsigned Ix = abs(srcp[x + 1] - srcp[x - 1]) >> shift;
  const unsigned Iy = mmax(abs(srcpp[x] - srcpn[x]), abs(srcpp[x] - srcp[x]), abs(srcp[x] - srcpn[x])) >> shift;
  if (Ix * Ix + Iy * Iy >= d.mthresh)
    out = peak;

  const unsigned Ixx = abs(srcp[x - 1] - 2 * srcp[x] + srcp[x + 1]) >> shift;
  const unsigned Iyy = abs(srcpp[x] - 2 * srcp[x] + srcpn[x]) >> shift;
  if (Ixx + Iyy >= d.lthresh)
    out = peak;
}

template <typename T> KERNEL void erode(const EEDI2Param d, const T *msk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  unsigned count = 0;

  count += mskpp[x - 1] == peak;
  count += mskpp[x] == peak;
  count += mskpp[x + 1] == peak;
  count += mskp[x - 1] == peak;
  count += mskp[x + 1] == peak;
  count += mskpn[x - 1] == peak;
  count += mskpn[x] == peak;
  count += mskpn[x + 1] == peak;

  out = mskp[x] == peak && count < d.estr ? 0 : mskp[x];
}

template <typename T> KERNEL void dilate(const EEDI2Param d, const T *msk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto mskpp = lineOff(msk, -1);
  auto mskpn = lineOff(msk, 1);
  auto &out = point(dst);

  out = 0;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  unsigned count = 0;

  count += mskpp[x - 1] == peak;
  count += mskpp[x] == peak;
  count += mskpp[x + 1] == peak;
  count += mskp[x - 1] == peak;
  count += mskp[x + 1] == peak;
  count += mskpn[x - 1] == peak;
  count += mskpn[x] == peak;
  count += mskpn[x + 1] == peak;

  out = mskp[x] == 0 && count >= d.dstr ? peak : mskp[x];
}

template <typename T> KERNEL void removeSmallHorzGaps(const EEDI2Param d, const T *msk, T *dst) {
  setup_kernel;

  auto mskp = line(msk);
  auto &out = point(dst);

  auto orig = mskp[x];
  out = orig;

  bounds_check3(x, 3, width - 3);
  bounds_check3(y, 1, height - 1);

  auto a = mskp[x - 3] | mskp[x - 2] | mskp[x - 1] | mskp[x + 1] | mskp[x + 2] | mskp[x + 3] ? orig : 0;
  auto b =
      (mskp[x + 1] & (mskp[x - 1] | mskp[x - 2] | mskp[x - 3])) | (mskp[x + 2] & (mskp[x - 1] | mskp[x - 2])) | (mskp[x + 3] & mskp[x - 1])
          ? peak
          : orig;

  out = mskp[x] ? a : b;
}

template <typename T> KERNEL void calcDirections(const EEDI2Param d, const T *src, const T *msk, T *dst) {
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

  constexpr int block_w = 64;
  constexpr int off_w = block_w / 2;
  constexpr int buf_w = block_w * 2;
  __shared__ int s2p[buf_w], s1p[buf_w], s[buf_w], s1n[buf_w], s2n[buf_w];
  __shared__ int m1p[buf_w], m1n[buf_w];

  // XXX: we are safe because they won't be the first or last plane in pool
  s2p[x % block_w] = src2p[x - off_w];
  s2p[x % block_w + block_w] = src2p[x + off_w];
  s1p[x % block_w] = srcpp[x - off_w];
  s1p[x % block_w + block_w] = srcpp[x + off_w];
  s[x % block_w] = srcp[x - off_w];
  s[x % block_w + block_w] = srcp[x + off_w];
  s1n[x % block_w] = srcpn[x - off_w];
  s1n[x % block_w + block_w] = srcpn[x + off_w];
  s2n[x % block_w] = src2n[x - off_w];
  s2n[x % block_w + block_w] = src2n[x + off_w];
  m1p[x % block_w] = mskpp[x - off_w];
  m1p[x % block_w + block_w] = mskpp[x + off_w];
  m1n[x % block_w] = mskpn[x - off_w];
  m1n[x % block_w + block_w] = mskpn[x + off_w];
  __syncthreads();
  auto X = x % block_w + off_w;

  bounds_check3(x, 1, width - 1);
  bounds_check3(y, 1, height - 1);

  const int maxd = d.maxd >> d.subSampling;

  if (mskp[x] != peak || (mskp[x - 1] != peak && mskp[x + 1] != peak))
    return;

  const int uStartBound = -x + 1;
  const int uStopBound = width - 2 - x;
  const unsigned min0 = abs(s[X] - s1n[X]) + abs(s[X] - s1p[X]);
  unsigned minA = mmin(d.nt19, min0 * 9);
  unsigned minB = mmin(d.nt13, min0 * 6);
  unsigned minC = minA;
  unsigned minD = minB;
  unsigned minE = minB;
  int dirA = -5000, dirB = -5000, dirC = -5000, dirD = -5000, dirE = -5000;

  for (int u = -maxd; u <= maxd; u++) {
    if (u < uStartBound || u > uStopBound)
      continue;
    if ((y == 1 || m1p[X - 1 + u] == peak || m1p[X + u] == peak || m1p[X + 1 + u] == peak) &&
        (y == height - 2 || m1n[X - 1 - u] == peak || m1n[X - u] == peak || m1n[X + 1 - u] == peak)) {
      const unsigned diffsn = abs(s[X - 1] - s1n[X - 1 - u]) + abs(s[X] - s1n[X - u]) + abs(s[X + 1] - s1n[X + 1 - u]);
      const unsigned diffsp = abs(s[X - 1] - s1p[X - 1 + u]) + abs(s[X] - s1p[X + u]) + abs(s[X + 1] - s1p[X + 1 + u]);
      const unsigned diffps = abs(s1p[X - 1] - s[X - 1 - u]) + abs(s1p[X] - s[X - u]) + abs(s1p[X + 1] - s[X + 1 - u]);
      const unsigned diffns = abs(s1n[X - 1] - s[X - 1 + u]) + abs(s1n[X] - s[X + u]) + abs(s1n[X + 1] - s[X + 1 + u]);
      const unsigned diff = diffsn + diffsp + diffps + diffns;
      unsigned diffD = diffsp + diffns;
      unsigned diffE = diffsn + diffps;

      if (diff < minB) {
        dirB = u;
        minB = diff;
      }

      if (y > 1) {
        const unsigned diff2pp = abs(s2p[X - 1] - s1p[X - 1 - u]) + abs(s2p[X] - s1p[X - u]) + abs(s2p[X + 1] - s1p[X + 1 - u]);
        const unsigned diffp2p = abs(s1p[X - 1] - s2p[X - 1 + u]) + abs(s1p[X] - s2p[X + u]) + abs(s1p[X + 1] - s2p[X + 1 + u]);
        const unsigned diffA = diff + diff2pp + diffp2p;
        diffD += diffp2p;
        diffE += diff2pp;

        if (diffA < minA) {
          dirA = u;
          minA = diffA;
        }
      }

      if (y < height - 2) {
        const unsigned diff2nn = abs(s2n[X - 1] - s1n[X - 1 + u]) + abs(s2n[X] - s1n[X + u]) + abs(s2n[X + 1] - s1n[X + 1 + u]);
        const unsigned diffn2n = abs(s1n[X - 1] - s2n[X - 1 - u]) + abs(s1n[X] - s2n[X - u]) + abs(s1n[X + 1] - s2n[X + 1 - u]);
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

  auto okA = dirA != -5000, okB = dirB != -5000, okC = dirC != -5000, okD = dirD != -5000, okE = dirE != -5000;
  unsigned k = okA + okB + okC + okD + okE;
  int order[] = {
      okA ? dirA : intmax, okB ? dirB : intmax, okC ? dirC : intmax, okD ? dirD : intmax, okE ? dirE : intmax,
  };

  if (k > 1) {
    bose_sort_array(order);

    const int mid = (k & 1) ? order[k / 2] : (order[(k - 1) / 2] + order[k / 2] + 1) / 2;
    const int lim = mmax(limlut[abs(mid)] / 4, 2);
    int sum = 0;
    int count = 0;

    for (unsigned i = 0; i < 5; i++) {
      auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
      sum += cond * order[i];
      count += cond;
    }

    out = (count > 1) ? neutral + ((sum / count) << shift2) : neutral;
  } else {
    out = neutral;
  }
}

template <typename T> KERNEL void filterDirMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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

  int val0 = dmskpp[x - 1], val1 = dmskpp[x], val2 = dmskpp[x + 1], val3 = dmskp[x - 1], val4 = dmskp[x], val5 = dmskp[x + 1],
      val6 = dmskpn[x - 1], val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak, cond3 = val3 != peak, cond4 = val4 != peak, cond5 = val5 != peak,
       cond6 = val6 != peak, cond7 = val7 != peak, cond8 = val8 != peak;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax, cond3 ? val3 : intmax, cond4 ? val4 : intmax,
      cond5 ? val5 : intmax, cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned u = cond0 + cond1 + cond2 + cond3 + cond4 + cond5 + cond6 + cond7 + cond8;

  if (u < 4) {
    out = peak;
    return;
  }

  bose_sort_array(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  for (unsigned i = 0; i < 9; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < 4 || (count < 5 && dmskp[x] == peak)) {
    out = peak;
    return;
  }

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void expandDirMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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

  int val0 = dmskpp[x - 1], val1 = dmskpp[x], val2 = dmskpp[x + 1], val3 = dmskp[x - 1], val5 = dmskp[x + 1], val6 = dmskpn[x - 1],
      val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak, cond3 = val3 != peak, cond5 = val5 != peak, cond6 = val6 != peak,
       cond7 = val7 != peak, cond8 = val8 != peak;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax, cond3 ? val3 : intmax,
      cond5 ? val5 : intmax, cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned u = cond0 + cond1 + cond2 + cond3 + cond5 + cond6 + cond7 + cond8;

  if (u < 5)
    return;

  bose_sort_array(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  for (unsigned i = 0; i < 8; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < 5)
    return;

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void filterMap(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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
  const int lim = mmax(abs(dir) * 2, twleve + 0);
  dir >>= shift;
  bool ict = false, icb = false;

  auto cond = dir < 0;
  auto l = cond ? mmax(-x, dir) : 0;
  auto r = cond ? 0 : mmin(width - x - 1, dir);
  for (int j = l; j <= r; j++) {
    if ((abs(dmskpp[x + j] - dmskp[x]) > lim && dmskpp[x + j] != peak) || (dmskp[x + j] == peak && dmskpp[x + j] == peak) ||
        (abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
      ict = true;
      break;
    }
  }

  if (ict) {
    auto l = cond ? 0 : mmax(-x, -dir);
    auto r = cond ? mmin(width - x - 1, -dir) : 0;
    for (int j = l; j <= r; j++) {
      if ((abs(dmskpn[x + j] - dmskp[x]) > lim && dmskpn[x + j] != peak) || (dmskpn[x + j] == peak && dmskp[x + j] == peak) ||
          (abs(dmskp[x + j] - dmskp[x]) > lim && dmskp[x + j] != peak)) {
        icb = true;
        break;
      }
    }

    if (icb)
      out = peak;
  }
}

template <typename T> KERNEL void markDirections2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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

  int val0 = dmskp[x - 1], val1 = dmskp[x], val2 = dmskp[x + 1], val6 = dmskpn[x - 1], val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak, cond1 = val1 != peak, cond2 = val2 != peak, cond6 = val6 != peak, cond7 = val7 != peak, cond8 = val8 != peak;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax,
      cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned v = cond0 + cond1 + cond2 + cond6 + cond7 + cond8;

  if (v < 3)
    return;

  bose_sort_array(order);

  const int mid = (v & 1) ? order[v / 2] : (order[(v - 1) / 2] + order[v / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  unsigned u = 0;
  u += (abs(dmskp[x - 1] - dmskpn[x - 1]) <= lim || dmskp[x - 1] == peak || dmskpn[x - 1] == peak);
  u += (abs(dmskp[x] - dmskpn[x]) <= lim || dmskp[x] == peak || dmskpn[x] == peak);
  u += (abs(dmskp[x + 1] - dmskpn[x - 1]) <= lim || dmskp[x + 1] == peak || dmskpn[x + 1] == peak);
  if (u < 2)
    return;

  for (unsigned i = 0; i < 6; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < v - 2 || count < 2)
    return;

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void filterDirMap2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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

  // XXX: we are safe because dmsk won't be the first or last plane in pool
  int val0 = dmskpp[x - 1], val1 = dmskpp[x], val2 = dmskpp[x + 1], val3 = dmskp[x - 1], val4 = dmskp[x], val5 = dmskp[x + 1],
      val6 = dmskpn[x - 1], val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak && y > 1, cond1 = val1 != peak && y > 1, cond2 = val2 != peak && y > 1, cond3 = val3 != peak,
       cond4 = val4 != peak, cond5 = val5 != peak, cond6 = val6 != peak && y < height - 2, cond7 = val7 != peak && y < height - 2,
       cond8 = val8 != peak && y < height - 2;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax, cond3 ? val3 : intmax, cond4 ? val4 : intmax,
      cond5 ? val5 : intmax, cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned u = cond0 + cond1 + cond2 + cond3 + cond4 + cond5 + cond6 + cond7 + cond8;

  if (u < 4) {
    out = peak;
    return;
  }

  bose_sort_array(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  for (unsigned i = 0; i < 9; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < 4 || (count < 5 && dmskp[x] == peak)) {
    out = peak;
    return;
  }

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void expandDirMap2X(const EEDI2Param d, const T *msk, const T *dmsk, T *dst) {
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

  // XXX: we are safe because dmsk won't be the first or last plane in pool
  int val0 = dmskpp[x - 1], val1 = dmskpp[x], val2 = dmskpp[x + 1], val3 = dmskp[x - 1], val5 = dmskp[x + 1], val6 = dmskpn[x - 1],
      val7 = dmskpn[x], val8 = dmskpn[x + 1];
  auto cond0 = val0 != peak && y > 1, cond1 = val1 != peak && y > 1, cond2 = val2 != peak && y > 1, cond3 = val3 != peak,
       cond5 = val5 != peak, cond6 = val6 != peak && y < height - 2, cond7 = val7 != peak && y < height - 2,
       cond8 = val8 != peak && y < height - 2;
  int order[] = {
      cond0 ? val0 : intmax, cond1 ? val1 : intmax, cond2 ? val2 : intmax, cond3 ? val3 : intmax,
      cond5 ? val5 : intmax, cond6 ? val6 : intmax, cond7 ? val7 : intmax, cond8 ? val8 : intmax,
  };
  unsigned u = cond0 + cond1 + cond2 + cond3 + cond5 + cond6 + cond7 + cond8;

  if (u < 5)
    return;

  bose_sort_array(order);

  const int mid = (u & 1) ? order[u / 2] : (order[(u - 1) / 2] + order[u / 2] + 1) / 2;
  const int lim = limlut[abs(mid - neutral) >> shift2] << shift;
  int sum = 0;
  int count = 0;

  for (unsigned i = 0; i < 8; i++) {
    auto cond = order[i] != intmax && abs(order[i] - mid) <= lim;
    sum += cond * order[i];
    count += cond;
  }

  if (count < 5)
    return;

  out = round_div(sum + mid, count + 1);
}

template <typename T> KERNEL void fillGaps2X(const EEDI2Param d, const T *msk, const T *dmsk, T *tmp) {
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

  while (u && x - u < 16) {
    if (dmskp[u] != peak) {
      back = dmskp[u];
      break;
    }
    if (mskp[u] != peak && mskpn[u] != peak)
      break;
    u--;
  }

  while (v < width && v - x < 16) {
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

  for (unsigned j = u; j <= v && tc; j++) {
    tc = !(y <= 2 || dmskpp[j] == peak || (mskpp[j] != peak && mskp[j] != peak));
    mint = tc ? mmin(mint, dmskpp[j] + 0) : twenty;
    maxt = tc ? mmax(maxt, dmskpp[j] + 0) : twenty;
  }

  for (unsigned j = u; j <= v && bc; j++) {
    bc = !(y >= height - 3 || dmskpn[j] == peak || (mskpn[j] != peak && mskpnn[j] != peak));
    minb = bc ? mmin(minb, dmskpn[j] + 0) : twenty;
    maxb = bc ? mmax(maxb, dmskpn[j] + 0) : twenty;
  }

  if (maxt == -twenty)
    maxt = mint = twenty;
  if (maxb == -twenty)
    maxb = minb = twenty;

  const int thresh = mmax(mmax(abs(forward - neutral), abs(back - neutral)) / 4, eight + 0, abs(mint - maxt), abs(minb - maxb));
  const unsigned lim = mmin(mmax(abs(forward - neutral), abs(back - neutral)) >> shift2, 6);
  if (abs(forward - back) <= thresh && (v - u - 1 <= lim || tc || bc)) {
    out_tmp = (x - u) | (v - x) << 8u;
  }
}

template <typename T> KERNEL void fillGaps2XStep2(const EEDI2Param d, const T *msk, const T *dmsk, const T *tmp, T *dst) {
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

  for (unsigned i = max(x - 16, 1); i < x; ++i) {
    bool cond = i + (tmpp[i] >> 8) > x;
    uv = cond ? tmpp[i] : uv;
    pos = cond ? i : pos;
  }
  uv = tmpp[x] ? tmpp[x] : uv;
  pos = tmpp[x] ? x : pos;
  for (unsigned i = x + 1; i - x < 16 && i < width - 1; ++i) {
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

  out = back + round_div((forward - back) * (x - 1 - u), (v - u));
}

template <typename T> KERNEL void interpolateLattice(const EEDI2Param d, const T *omsk, const T *dmsk, T *dst, T *dmsk_2) {
  setup_kernel2x;

  auto omskp = lineOff(omsk, -1);
  auto omskn = lineOff(omsk, 1);
  auto dmskp = line(dmsk);
  auto dstp = lineOff(dst, -1);
  auto dstpn = line(dst);
  auto dstpnn = lineOff(dst, 1);
  auto &out = point(dst);
  auto &mout = point(dmsk_2);

  bounds_check3(x, 0, width);
  bounds_check3(y, 1, height - 1);

  int dir = dmskp[x];
  const int lim = limlut[abs(dir - neutral) >> shift2] << shift;

  if (dir == peak || (abs(dmskp[x] - dmskp[x - 1]) > lim && abs(dmskp[x] - dmskp[x + 1]) > lim)) {
    out = (dstp[x] + dstpnn[x] + 1) / 2;
    if (dir != peak)
      mout = neutral;
    return;
  }

  if (lim < nine) {
    const unsigned sum = (dstp[x - 1] + dstp[x] + dstp[x + 1] + dstpnn[x - 1] + dstpnn[x] + dstpnn[x + 1]) >> shift;
    const unsigned sumsq = (dstp[x - 1] >> shift) * (dstp[x - 1] >> shift) + (dstp[x] >> shift) * (dstp[x] >> shift) +
                           (dstp[x + 1] >> shift) * (dstp[x + 1] >> shift) + (dstpnn[x - 1] >> shift) * (dstpnn[x - 1] >> shift) +
                           (dstpnn[x] >> shift) * (dstpnn[x] >> shift) + (dstpnn[x + 1] >> shift) * (dstpnn[x + 1] >> shift);
    if (6 * sumsq - sum * sum < 576) {
      out = (dstp[x] + dstpnn[x] + 1) / 2;
      mout = peak;
      return;
    }
  }

  if (x > 1 && x < width - 2 &&
      ((dstp[x] < mmax(dstp[x - 2], dstp[x - 1]) - three && dstp[x] < mmax(dstp[x + 2], dstp[x + 1]) - three &&
        dstpnn[x] < mmax(dstpnn[x - 2], dstpnn[x - 1]) - three && dstpnn[x] < mmax(dstpnn[x + 2], dstpnn[x + 1]) - three) ||
       (dstp[x] > mmin(dstp[x - 2], dstp[x - 1]) + three && dstp[x] > mmin(dstp[x + 2], dstp[x + 1]) + three &&
        dstpnn[x] > mmin(dstpnn[x - 2], dstpnn[x - 1]) + three && dstpnn[x] > mmin(dstpnn[x + 2], dstpnn[x + 1]) + three))) {
    out = (dstp[x] + dstpnn[x] + 1) / 2;
    mout = neutral;
    return;
  }

  dir = (dir - neutral + (1 << (shift2 - 1))) >> shift2;
  const int uStart = (dir - 2 < 0) ? mmax(-x + 1, dir - 2, -width + 2 + x) : mmin(x - 1, dir - 2, width - 2 - x);
  const int uStop = (dir + 2 < 0) ? mmax(-x + 1, dir + 2, -width + 2 + x) : mmin(x - 1, dir + 2, width - 2 - x);
  unsigned min = d.nt8;
  unsigned val = (dstp[x] + dstpnn[x] + 1) / 2;

  for (int u = uStart; u <= uStop; u++) {
    const unsigned diff = abs(dstp[x - 1] - dstpnn[x - u - 1]) + abs(dstp[x] - dstpnn[x - u]) + abs(dstp[x + 1] - dstpnn[x - u + 1]) +
                          abs(dstpnn[x - 1] - dstp[x + u - 1]) + abs(dstpnn[x] - dstp[x + u]) + abs(dstpnn[x + 1] - dstp[x + u + 1]);
    if (diff < min &&
        ((omskp[x - 1 + u] != peak && abs(omskp[x - 1 + u] - dmskp[x]) <= lim) ||
         (omskp[x + u] != peak && abs(omskp[x + u] - dmskp[x]) <= lim) ||
         (omskp[x + 1 + u] != peak && abs(omskp[x + 1 + u] - dmskp[x]) <= lim)) &&
        ((omskn[x - 1 - u] != peak && abs(omskn[x - 1 - u] - dmskp[x]) <= lim) ||
         (omskn[x - u] != peak && abs(omskn[x - u] - dmskp[x]) <= lim) ||
         (omskn[x + 1 - u] != peak && abs(omskn[x + 1 - u] - dmskp[x]) <= lim))) {
      const unsigned diff2 = abs(dstp[x + u / 2 - 1] - dstpnn[x - u / 2 - 1]) + abs(dstp[x + u / 2] - dstpnn[x - u / 2]) +
                             abs(dstp[x + u / 2 + 1] - dstpnn[x - u / 2 + 1]);
      if (diff2 < d.nt4 &&
          (((abs(omskp[x + u / 2] - omskn[x - u / 2]) <= lim || abs(omskp[x + u / 2] - omskn[x - ((u + 1) / 2)]) <= lim) &&
            omskp[x + u / 2] != peak) ||
           ((abs(omskp[x + ((u + 1) / 2)] - omskn[x - u / 2]) <= lim || abs(omskp[x + ((u + 1) / 2)] - omskn[x - ((u + 1) / 2)]) <= lim) &&
            omskp[x + ((u + 1) / 2)] != peak))) {
        if ((abs(dmskp[x] - omskp[x + u / 2]) <= lim || abs(dmskp[x] - omskp[x + ((u + 1) / 2)]) <= lim) &&
            (abs(dmskp[x] - omskn[x - u / 2]) <= lim || abs(dmskp[x] - omskn[x - ((u + 1) / 2)]) <= lim)) {
          val = (dstp[x + u / 2] + dstp[x + ((u + 1) / 2)] + dstpnn[x - u / 2] + dstpnn[x - ((u + 1) / 2)] + 2) / 4;
          min = diff;
          dir = u;
        }
      }
    }
  }

  if (min != d.nt8) {
    out = val;
    mout = neutral + (dir << shift2);
  } else {
    const int dt = 4 >> d.subSampling;
    const int uStart2 = mmax(-x + 1, -dt);
    const int uStop2 = mmin(width - 2 - x, dt);
    const unsigned minm = mmin(dstp[x], dstpnn[x]);
    const unsigned maxm = mmax(dstp[x], dstpnn[x]);
    min = d.nt7;

    for (int u = uStart2; u <= uStop2; u++) {
      const int p1 = dstp[x + u / 2] + dstp[x + ((u + 1) / 2)];
      const int p2 = dstpnn[x - u / 2] + dstpnn[x - ((u + 1) / 2)];
      const unsigned diff = abs(dstp[x - 1] - dstpnn[x - u - 1]) + abs(dstp[x] - dstpnn[x - u]) + abs(dstp[x + 1] - dstpnn[x - u + 1]) +
                            abs(dstpnn[x - 1] - dstp[x + u - 1]) + abs(dstpnn[x] - dstp[x + u]) + abs(dstpnn[x + 1] - dstp[x + u + 1]) +
                            abs(p1 - p2);
      if (diff < min) {
        const unsigned valt = (p1 + p2 + 2) / 4;
        if (valt >= minm && valt <= maxm) {
          val = valt;
          min = diff;
          dir = u;
        }
      }
    }

    out = val;
    mout = (min == d.nt7) ? neutral : neutral + (dir << shift2);
  }
}

template <typename T> KERNEL void postProcess(const EEDI2Param d, const T *nmsk, const T *omsk, T *dst) {
  setup_kernel2x;

  auto nmskp = line(nmsk);
  auto omskp = line(omsk);
  auto &out = point(dst);
  auto dstpp = lineOff(dst, -1);
  auto dstpn = lineOff(dst, 1);

  bounds_check3(x, 0, width);
  bounds_check3(y, 1, height - 1);

  const int lim = limlut[abs(nmskp[x] - neutral) >> shift2] << shift;
  if (abs(nmskp[x] - omskp[x]) > lim && omskp[x] != peak && omskp[x] != neutral)
    out = (dstpp[x] + dstpn[x] + 1) / 2;
}

template <typename T> void EEDI2Instance<T>::tuneKernels() {
  int totalSmem;
  try_cuda(cudaDeviceGetAttribute(&totalSmem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
  auto carveout65536 = 65536 * 100 / totalSmem;
  try_cuda(cudaFuncSetAttribute(buildEdgeMask<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(erode<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(dilate<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(removeSmallHorzGaps<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(calcDirections<T>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout65536));
  try_cuda(cudaFuncSetAttribute(filterDirMap<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(expandDirMap<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(filterMap<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(markDirections2X<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(filterDirMap2X<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(expandDirMap2X<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(fillGaps2X<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(fillGaps2XStep2<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(interpolateLattice<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  try_cuda(cudaFuncSetAttribute(postProcess<T>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
}

template <typename T> void VS_CC eedi2Init(VSMap *_in, VSMap *_out, void **instanceData, VSNode *node, VSCore *_core, const VSAPI *vsapi) {
  auto data = static_cast<EEDI2Data<T> *>(*instanceData);
  vsapi->setVideoInfo(data->firstInstance().getOutputVI(), 1, node);
}

template <typename T>
const VSFrameRef *VS_CC eedi2GetFrame(int n, int activationReason, void **instanceData, void **_frameData, VSFrameContext *frameCtx,
                                      VSCore *core, const VSAPI *vsapi) {

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
    int err;
    unsigned num_streams;
    numeric_cast_to(num_streams, vsapi->propGetInt(in, "num_streams", 0, &err));
    if (err)
      num_streams = 1;
    auto data = new (num_streams) EEDI2Data<T>(in, vsapi);
    vsapi->createFilter(in, out, "EEDI2", eedi2Init<T>, eedi2GetFrame<T>, eedi2Free<T>, num_streams > 1 ? fmParallel : fmParallelRequests,
                        0, data, core);
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
