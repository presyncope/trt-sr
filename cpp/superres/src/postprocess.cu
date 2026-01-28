#include "common.h"
#include "postprocess.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

struct ColorConvParams {
  float3 y_coef;
  float y_off;
  float3 u_coef;
  float u_off;
  float3 v_coef;
  float v_off;
};

static void populate_ccparams(ColorConvParams &params, bool full_range, int bitdepth) {
  float max_val = (float)((1 << bitdepth) - 1);
  float scale = max_val / 255.0f; // Scale factor from 8-bit spec to target bitdepth

  if (full_range) {
    // BT.709 Full Range
    params.y_coef =
        make_float3(0.2126f * 255.0f * scale, 0.7152f * 255.0f * scale, 0.0722f * 255.0f * scale);
    params.y_off = 0.0f;
    
    params.u_coef =
        make_float3(-0.1146f * 255.0f * scale, -0.3854f * 255.0f * scale, 0.5000f * 255.0f * scale);
    params.u_off = 128.0f * scale;
    
    params.v_coef =
        make_float3(0.5000f * 255.0f * scale, -0.4542f * 255.0f * scale, -0.0458f * 255.0f * scale);
    params.v_off = 128.0f * scale;
  } else {
    // BT.709 Limited Range
    params.y_coef =
        make_float3(0.2126f * 219.0f * scale, 0.7152f * 219.0f * scale, 0.0722f * 219.0f * scale);
    params.y_off = 16.0f * scale;
    
    params.u_coef =
        make_float3(-0.1146f * 224.0f * scale, -0.3854f * 224.0f * scale, 0.5000f * 224.0f * scale);
    params.u_off = 128.0f * scale;
    
    params.v_coef =
        make_float3(0.5000f * 224.0f * scale, -0.4542f * 224.0f * scale, -0.0458f * 224.0f * scale);
    params.v_off = 128.0f * scale;
  }
}

// Helper for safe saturation and casting
template <typename T>
__device__ __forceinline__ T clip_val(float v, int bitdepth) {
  float max_v = (float)((1 << bitdepth) - 1);
  return static_cast<T>(__float2int_rn(fminf(fmaxf(v, 0.0f), max_v)));
}

template <typename T>
__global__ void postprocess_iyuv_kernel(const half *__restrict__ srcTensor,
                                        T *__restrict__ dstY,
                                        T *__restrict__ dstU,
                                        T *__restrict__ dstV,
                                        int dstWidth, int dstHeight,
                                        int dstOffsetY, int overlap_pixels,
                                        const ColorConvParams cpar,
                                        int bitdepth, int chfmt, int scale_factor) {
  const int local_x = blockIdx.x * blockDim.x + threadIdx.x; 
  const int local_y = blockIdx.y * blockDim.y + threadIdx.y; 
  const int n = blockIdx.z; // tile index in batch

  // TILE_SIZE is Model Input Tile. Output/Effective Tile is scaled.
  // We assume 'overlap_pixels' passed here is ALREADY SCALED by process().
  // And we iterate over the Scaled Tile.
  const int effective_tile_dim = TILE_SIZE * scale_factor;
  const int tile_size = effective_tile_dim - 2 * overlap_pixels;
  
  const int dst_x = local_x + n * tile_size;
  const int dst_y = dstOffsetY + local_y;

  if (dst_x >= dstWidth || dst_y >= dstHeight)
    return;

  if (local_x >= tile_size || local_y >= tile_size)
    return;

  const int src_x = local_x + overlap_pixels;
  const int src_y = local_y + overlap_pixels;
  
  // Tensor Plane Size is based on Output Tensor Dim (Scaled)
  size_t plane_size = (size_t)effective_tile_dim * effective_tile_dim;
  const half *src_tile = srcTensor + n * plane_size * 3;
  const size_t tensor_idx = src_y * effective_tile_dim + src_x;

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

  dstY[dst_y * dstWidth + dst_x] = clip_val<T>(y_comp, bitdepth);

  if (chfmt == 400) return;

  // Chroma Store
  bool store_chroma = false;
  int uv_x = 0, uv_y = 0, uv_stride = 0;

  switch(chfmt) {
      case 420: // Subsampled 2x2. Store only on even coordinates?
          if (!(dst_x & 1) && !(dst_y & 1)) {
              store_chroma = true;
              uv_x = dst_x >> 1; uv_y = dst_y >> 1; uv_stride = dstWidth >> 1;
          }
          break;
      case 422: // Subsampled 2x1.
          if (!(dst_x & 1)) {
              store_chroma = true;
              uv_x = dst_x >> 1; uv_y = dst_y; uv_stride = dstWidth >> 1;
          }
          break;
      case 444:
          store_chroma = true;
          uv_x = dst_x; uv_y = dst_y; uv_stride = dstWidth;
          break;
  }

  if (store_chroma) {
    int uv_idx = uv_y * uv_stride + uv_x;
    dstU[uv_idx] = clip_val<T>(u_comp, bitdepth);
    dstV[uv_idx] = clip_val<T>(v_comp, bitdepth);
  }
}

void postprocess(const PostprocessParams &params, cudaStream_t stream) {
  const int effective_tile_dim = TILE_SIZE * params.scale_factor;
  const int tile_size = effective_tile_dim - 2 * params.overlap_pixels;
  const int batch = (params.dst_width + tile_size - 1) / tile_size;

  dim3 block(32, 32, 1);
  dim3 grid;
  grid.x = (tile_size + 31) / 32;
  grid.y = (tile_size + 31) / 32;
  grid.z = batch;

  ColorConvParams cc_params{};
  populate_ccparams(cc_params, params.video_full_range_flag, params.bitdepth);

  if (params.bitdepth == 8) {
      postprocess_iyuv_kernel<uint8_t><<<grid, block, 0, stream>>>(
          params.d_srcTensor, (uint8_t*)params.d_dstY, (uint8_t*)params.d_dstU, (uint8_t*)params.d_dstV,
          params.dst_width, params.dst_height, params.dst_start_y,
          params.overlap_pixels, cc_params, params.bitdepth, params.chfmt, params.scale_factor);
  } else {
      postprocess_iyuv_kernel<uint16_t><<<grid, block, 0, stream>>>(
          params.d_srcTensor, (uint16_t*)params.d_dstY, (uint16_t*)params.d_dstU, (uint16_t*)params.d_dstV,
          params.dst_width, params.dst_height, params.dst_start_y,
          params.overlap_pixels, cc_params, params.bitdepth, params.chfmt, params.scale_factor);
  }
}