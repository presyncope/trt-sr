
#include "resize.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector_types.h>

// ------------------------------------------------------------------
// 1. Type Traits & Helpers
// ------------------------------------------------------------------

template <typename T> struct PixelTraits { static const int channels = 1; using BaseType = T; };
template <> struct PixelTraits<uchar2> { static const int channels = 2; using BaseType = uint8_t; };
template <> struct PixelTraits<ushort2> { static const int channels = 2; using BaseType = uint16_t; };

// Safe casting with clamping (Saturate)
template <typename OutT>
__device__ inline OutT saturate_cast(float val) {
  if (sizeof(OutT) == 1) { // 8-bit
    val = fminf(fmaxf(val, 0.0f), 255.0f);
  } else { // 16-bit
    val = fminf(fmaxf(val, 0.0f), 65535.0f);
  }
  return (OutT)(val + 0.5f);
}

// Component extraction
template <typename T> __device__ inline float get_component(T val, int c) { return (float)val; }
template <> __device__ inline float get_component<uchar2>(uchar2 val, int c) {
  return (c == 0) ? (float)val.x : (float)val.y;
}
template <> __device__ inline float get_component<ushort2>(ushort2 val, int c) {
  return (c == 0) ? (float)val.x : (float)val.y;
}

// Pixel maker
template <typename T> __device__ inline T make_pixel(float v0, float v1);

template <> __device__ inline uint8_t make_pixel<uint8_t>(float v0, float v1) { return saturate_cast<uint8_t>(v0); }
template <> __device__ inline uint16_t make_pixel<uint16_t>(float v0, float v1) { return saturate_cast<uint16_t>(v0); }
template <> __device__ inline uchar2 make_pixel<uchar2>(float v0, float v1) {
  return make_uchar2(saturate_cast<uint8_t>(v0), saturate_cast<uint8_t>(v1));
}
template <> __device__ inline ushort2 make_pixel<ushort2>(float v0, float v1) {
  return make_ushort2(saturate_cast<uint16_t>(v0), saturate_cast<uint16_t>(v1));
}

// Memory Access Helper (Uses Read-Only Cache)
template <typename T>
__device__ inline T load_pixel_clamped(const T* __restrict__ src, int x, int y, int w, int h, size_t stride_bytes) {
  x = max(0, min(x, w - 1));
  y = max(0, min(y, h - 1));
  const uint8_t* row = (const uint8_t*)src + y * stride_bytes;
  // __ldg optimizes global memory reads by using the read-only cache
  return __ldg((const T*)row + x);
}

// ------------------------------------------------------------------
// 2. Interpolation Logic
// ------------------------------------------------------------------

// Bilinear Interpolation
template <typename T>
__device__ T get_pixel_bilinear(const T *src, int srcW, int srcH, size_t srcStride, float u, float v) {
  int x0 = floorf(u);
  int y0 = floorf(v);
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float wu = u - x0;
  float wv = v - y0;

  T p00 = load_pixel_clamped(src, x0, y0, srcW, srcH, srcStride);
  T p10 = load_pixel_clamped(src, x1, y0, srcW, srcH, srcStride);
  T p01 = load_pixel_clamped(src, x0, y1, srcW, srcH, srcStride);
  T p11 = load_pixel_clamped(src, x1, y1, srcW, srcH, srcStride);

  float res[2] = {0.0f, 0.0f};

#pragma unroll
  for (int c = 0; c < PixelTraits<T>::channels; ++c) {
    float v00 = get_component(p00, c);
    float v10 = get_component(p10, c);
    float v01 = get_component(p01, c);
    float v11 = get_component(p11, c);

    float val0 = v00 * (1.0f - wu) + v10 * wu;
    float val1 = v01 * (1.0f - wu) + v11 * wu;
    res[c] = val0 * (1.0f - wv) + val1 * wv;
  }

  return make_pixel<T>(res[0], res[1]);
}

// Cubic Weight Function
__device__ inline float cubic_weight(float x) {
  const float A = -0.75f;
  x = fabsf(x);
  if (x <= 1.0f) return (A + 2.0f) * x * x * x - (A + 3.0f) * x * x + 1.0f;
  if (x < 2.0f) return A * x * x * x - 5.0f * A * x * x + 8.0f * A * x - 4.0f * A;
  return 0.0f;
}

// Bicubic Interpolation
template <typename T>
__device__ T get_pixel_bicubic(const T *src, int srcW, int srcH, size_t srcStride, float u, float v) {
  int x_int = floorf(u);
  int y_int = floorf(v);
  float dx = u - x_int;
  float dy = v - y_int;

  float wx[4], wy[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) wx[i] = cubic_weight(dx - (i - 1));
#pragma unroll
  for (int j = 0; j < 4; ++j) wy[j] = cubic_weight(dy - (j - 1));

  float res[2] = {0.0f, 0.0f};

#pragma unroll
  for (int c = 0; c < PixelTraits<T>::channels; ++c) {
    float val = 0.0f;
    // Optimized 4x4 loop
#pragma unroll
    for (int j = -1; j <= 2; ++j) {
      float row_val = 0.0f;
#pragma unroll
      for (int i = -1; i <= 2; ++i) {
        T p = load_pixel_clamped(src, x_int + i, y_int + j, srcW, srcH, srcStride);
        row_val += get_component(p, c) * wx[i + 1];
      }
      val += row_val * wy[j + 1];
    }
    res[c] = val;
  }

  return make_pixel<T>(res[0], res[1]);
}

// ------------------------------------------------------------------
// 3. Kernel
// ------------------------------------------------------------------

template <typename T, bool USE_BICUBIC>
__global__ void resize_plane_kernel(const T *__restrict__ src, int srcW, int srcH, size_t srcStride,
                                    T *__restrict__ dst, int dstW, int dstH, size_t dstStride,
                                    float scaleX, float scaleY) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstW || y >= dstH) return;

  // Center-aligned mapping
  float u = (x + 0.5f) * scaleX - 0.5f;
  float v = (y + 0.5f) * scaleY - 0.5f;

  T val;
  if (USE_BICUBIC) {
    val = get_pixel_bicubic(src, srcW, srcH, srcStride, u, v);
  } else {
    val = get_pixel_bilinear(src, srcW, srcH, srcStride, u, v);
  }

  // Output write
  uint8_t *dst_row = (uint8_t *)dst + y * dstStride;
  ((T *)dst_row)[x] = val;
}

// ------------------------------------------------------------------
// 4. Host Dispatcher
// ------------------------------------------------------------------

template <typename T>
void launch_kernel(const FramePlane &srcPlane, FramePlane &dstPlane, ResizeMethod method, cudaStream_t stream) {
  int srcW = srcPlane.width_bytes / sizeof(T);
  int srcH = srcPlane.height;
  int dstW = dstPlane.width_bytes / sizeof(T);
  int dstH = dstPlane.height;

  // Pre-calculate scales
  float scaleX = (float)srcW / dstW;
  float scaleY = (float)srcH / dstH;

  dim3 block(32, 16);
  dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

  if (method == ResizeMethod::BICUBIC) {
    resize_plane_kernel<T, true><<<grid, block, 0, stream>>>(
      (const T *)srcPlane.data, srcW, srcH, srcPlane.stride,
      (T *)dstPlane.data, dstW, dstH, dstPlane.stride, scaleX, scaleY);
  } else {
    resize_plane_kernel<T, false><<<grid, block, 0, stream>>>(
      (const T *)srcPlane.data, srcW, srcH, srcPlane.stride,
      (T *)dstPlane.data, dstW, dstH, dstPlane.stride, scaleX, scaleY);
  }
}

void resize_yuv(const FrameBuffer &src, FrameBuffer &dst, ResizeMethod method, cudaStream_t stream) {
  if (src.format != dst.format) {
    fprintf(stderr, "[Error] resize_yuv: Formats mismatch.\n");
    return;
  }

  for (int i = 0; i < src.num_planes; ++i) {
    if (!src.planes[i].data || !dst.planes[i].data) continue;

    bool is_16bit = src.bitdepth > 8;
    bool is_interleaved = is_semi_planar(src.format) && (i > 0);

    if (is_16bit) {
      if (is_interleaved) launch_kernel<ushort2>(src.planes[i], dst.planes[i], method, stream);
      else launch_kernel<uint16_t>(src.planes[i], dst.planes[i], method, stream);
    } else {
      if (is_interleaved) launch_kernel<uchar2>(src.planes[i], dst.planes[i], method, stream);
      else launch_kernel<uint8_t>(src.planes[i], dst.planes[i], method, stream);
    }
  }
}
