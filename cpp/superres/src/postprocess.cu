#include "common.h"
#include "postprocess.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

struct ColorConvParams {
  // Coefficients pre-scaled by 255 (Full) or 219/224 (Limited)
  float3 y_coef;
  float y_off;
  float3 u_coef;
  float u_off;
  float3 v_coef;
  float v_off;
};

static void populate_ccparams(ColorConvParams &params, bool full_range) {
  if (full_range) {
    // BT.709 Full Range (0-255)
    // Y = (0.2126 R + 0.7152 G + 0.0722 B) * 255
    params.y_coef =
        make_float3(0.2126f * 255.0f, 0.7152f * 255.0f, 0.0722f * 255.0f);
    params.y_off = 0.0f;
    // U = (-0.1146 R - 0.3854 G + 0.5000 B) * 255 + 128
    params.u_coef =
        make_float3(-0.1146f * 255.0f, -0.3854f * 255.0f, 0.5000f * 255.0f);
    params.u_off = 128.0f;
    // V = (0.5000 R - 0.4542 G - 0.0458 B) * 255 + 128
    params.v_coef =
        make_float3(0.5000f * 255.0f, -0.4542f * 255.0f, -0.0458f * 255.0f);
    params.v_off = 128.0f;
  } else {
    // BT.709 Limited Range (Y: 16-235, UV: 16-240)
    // Y = (0.2126 R + 0.7152 G + 0.0722 B) * 219 + 16
    params.y_coef =
        make_float3(0.2126f * 219.0f, 0.7152f * 219.0f, 0.0722f * 219.0f);
    params.y_off = 16.0f;
    // U = (-0.1146 R - 0.3854 G + 0.5000 B) * 224 + 128
    params.u_coef =
        make_float3(-0.1146f * 224.0f, -0.3854f * 224.0f, 0.5000f * 224.0f);
    params.u_off = 128.0f;
    // V = (0.5000 R - 0.4542 G - 0.0458 B) * 224 + 128
    params.v_coef =
        make_float3(0.5000f * 224.0f, -0.4542f * 224.0f, -0.0458f * 224.0f);
    params.v_off = 128.0f;
  }
}

// Helper for safe saturation and casting
__device__ __forceinline__ uint8_t clip_u8(float v) {
  return static_cast<uint8_t>(__float2int_rn(fminf(fmaxf(v, 0.0f), 255.0f)));
}

__global__ void postprocess_iyuv_kernel(const half *__restrict__ srcTensor,
                                        uint8_t *__restrict__ dstY,
                                        uint8_t *__restrict__ dstU,
                                        uint8_t *__restrict__ dstV,
                                        int dstWidth, int dstHeight,
                                        int dstOffsetY, int overlap_pixels,
                                        const ColorConvParams cpar) {
  const int local_x = blockIdx.x * blockDim.x + threadIdx.x; // 0~tile_size-1
  const int local_y = blockIdx.y * blockDim.y + threadIdx.y; // 0~tile_size-1
  const int n = blockIdx.z; // tile index in batch

  const int tile_size = TILE_SIZE - 2 * overlap_pixels;
  const int dst_x = local_x + n * tile_size;
  const int dst_y = dstOffsetY + local_y;

  if (dst_x >= dstWidth || dst_y >= dstHeight)
    return;

  if (local_x >= tile_size || local_y >= tile_size)
    return;

  const int src_x = local_x + overlap_pixels;
  const int src_y = local_y + overlap_pixels;
  constexpr size_t plane_size = TILE_SIZE * TILE_SIZE;
  const half *src_tile = srcTensor + n * plane_size * 3;
  const size_t tensor_idx = src_y * TILE_SIZE + src_x;

  float red = __half2float(src_tile[tensor_idx]);
  float green = __half2float(src_tile[plane_size + tensor_idx]);
  float blue = __half2float(src_tile[2 * plane_size + tensor_idx]);

  // rgb to yuv
  float y_comp =
      fmaf(cpar.y_coef.x, red,
           fmaf(cpar.y_coef.y, green, fmaf(cpar.y_coef.z, blue, cpar.y_off)));
  float u_comp =
      fmaf(cpar.u_coef.x, red,
           fmaf(cpar.u_coef.y, green, fmaf(cpar.u_coef.z, blue, cpar.u_off)));
  float v_comp =
      fmaf(cpar.v_coef.x, red,
           fmaf(cpar.v_coef.y, green, fmaf(cpar.v_coef.z, blue, cpar.v_off)));

  dstY[dst_y * dstWidth + dst_x] = clip_u8(y_comp);

  if (!(dst_x & 1) && !(dst_y & 1)) {
    int uv_idx = (dst_y >> 1) * (dstWidth >> 1) + (dst_x >> 1);
    dstU[uv_idx] = clip_u8(u_comp);
    dstV[uv_idx] = clip_u8(v_comp);
  }
}

void postprocess(const PostprocessParams &params, cudaStream_t stream) {
  const int tile_size = TILE_SIZE - 2 * params.overlap_pixels;
  const int batch = (params.dst_width + tile_size - 1) / tile_size;

  dim3 block(32, 32, 1);
  dim3 grid;
  grid.x = (tile_size + 31) / 32;
  grid.y = (tile_size + 31) / 32;
  grid.z = batch;

  ColorConvParams cc_params{};
  populate_ccparams(cc_params, params.video_full_range_flag);

  postprocess_iyuv_kernel<<<grid, block, 0, stream>>>(
      params.d_srcTensor, params.d_dstY, params.d_dstU, params.d_dstV,
      params.dst_width, params.dst_height, params.dst_start_y,
      params.overlap_pixels, cc_params);
}