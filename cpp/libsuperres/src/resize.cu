
#include "resize.h"

template <typename T>
__global__ void resize_plane_linear_kernel(const T *__restrict__ src, int srcW,
                                           int srcH, size_t srcStride,
                                           T *__restrict__ dst, int dstW,
                                           int dstH, size_t dstStride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstW || y >= dstH)
    return;

  // Map to src coordinates
  float u = (x + 0.5f) * ((float)srcW / dstW) - 0.5f;
  float v = (y + 0.5f) * ((float)srcH / dstH) - 0.5f;

  int x0 = floorf(u);
  int y0 = floorf(v);
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float wu = u - x0;
  float wv = v - y0;

  // Clamp coordinates
  x0 = max(0, min(x0, srcW - 1));
  x1 = max(0, min(x1, srcW - 1));
  y0 = max(0, min(y0, srcH - 1));
  y1 = max(0, min(y1, srcH - 1));

  // Bilinear interpolation
  // Use stride for access
  auto get_val = [&](const T *base, int stride, int xx, int yy) -> float {
    const uint8_t *row = (const uint8_t *)base + yy * stride;
    return static_cast<float>(((const T *)row)[xx]);
  };

  float val00 = get_val(src, srcStride, x0, y0);
  float val10 = get_val(src, srcStride, x1, y0);
  float val01 = get_val(src, srcStride, x0, y1);
  float val11 = get_val(src, srcStride, x1, y1);

  float val0 = val00 * (1.0f - wu) + val10 * wu;
  float val1 = val01 * (1.0f - wu) + val11 * wu;
  float val = val0 * (1.0f - wv) + val1 * wv;

  // Output write
  uint8_t *dst_row = (uint8_t *)dst + y * dstStride;
  ((T *)dst_row)[x] = static_cast<T>(val + 0.5f);
}

void resize_yuv(const ResizeParams &params, cudaStream_t stream) {
  // Plane 0 (Y)
  dim3 block(32, 32);
  dim3 gridY((params.dst_width[0] + 31) / 32, (params.dst_height[0] + 31) / 32);

  const uint8_t *src_p0 = (const uint8_t *)params.src_ptr;
  uint8_t *dst_p0 = (uint8_t *)params.dst_ptr;

  if (params.bitdepth == 8) {
    resize_plane_linear_kernel<uint8_t><<<gridY, block, 0, stream>>>(
        (const uint8_t *)src_p0, params.src_width[0], params.src_height[0],
        params.src_pitch, (uint8_t *)dst_p0, params.dst_width[0],
        params.dst_height[0], params.dst_pitch);
  } else {
    resize_plane_linear_kernel<uint16_t><<<gridY, block, 0, stream>>>(
        (const uint16_t *)src_p0, params.src_width[0], params.src_height[0],
        params.src_pitch, (uint16_t *)dst_p0, params.dst_width[0],
        params.dst_height[0], params.dst_pitch);
  }

  if (params.num_comps > 1) {
    // Plane 1 (U or UV)
    dim3 gridC((params.dst_width[1] + 31) / 32,
               (params.dst_height[1] + 31) / 32);

    // Calculate offsets based on pitch * height of previous plane
    const uint8_t *src_p1 = src_p0 + params.src_pitch * params.src_height[0];
    uint8_t *dst_p1 = dst_p0 + params.dst_pitch * params.dst_height[0];

    if (params.bitdepth == 8) {
      resize_plane_linear_kernel<uint8_t><<<gridC, block, 0, stream>>>(
          (const uint8_t *)src_p1, params.src_width[1], params.src_height[1],
          params.src_pitch, (uint8_t *)dst_p1, params.dst_width[1],
          params.dst_height[1], params.dst_pitch);
    } else {
      resize_plane_linear_kernel<uint16_t><<<gridC, block, 0, stream>>>(
          (const uint16_t *)src_p1, params.src_width[1], params.src_height[1],
          params.src_pitch, (uint16_t *)dst_p1, params.dst_width[1],
          params.dst_height[1], params.dst_pitch);
    }

    if (params.num_comps > 2) {
      // Plane 2 (V) - Same dimensions as Plane 1 usually
      const uint8_t *src_p2 = src_p1 + params.src_pitch * params.src_height[1];
      uint8_t *dst_p2 = dst_p1 + params.dst_pitch * params.dst_height[1];

      if (params.bitdepth == 8) {
        resize_plane_linear_kernel<uint8_t><<<gridC, block, 0, stream>>>(
            (const uint8_t *)src_p2, params.src_width[1], params.src_height[1],
            params.src_pitch, (uint8_t *)dst_p2, params.dst_width[1],
            params.dst_height[1], params.dst_pitch);
      } else {
        resize_plane_linear_kernel<uint16_t><<<gridC, block, 0, stream>>>(
            (const uint16_t *)src_p2, params.src_width[1], params.src_height[1],
            params.src_pitch, (uint16_t *)dst_p2, params.dst_width[1],
            params.dst_height[1], params.dst_pitch);
      }
    }
  }
}
