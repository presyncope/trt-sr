#include "common.h"
#include "postprocess.h"
#include <cmath>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

struct ColorConvParams {
  float3 y_coef;
  float y_off;
  float3 u_coef;
  float u_off;
  float3 v_coef;
  float v_off;
};

static void populate_ccparams(ColorConvParams &params, bool full_range,
                              int bitdepth) {
  float max_val = (float)((1 << bitdepth) - 1);
  float scale =
      max_val / 255.0f; // Scale factor from 8-bit spec to target bitdepth

  if (full_range) {
    // BT.709 Full Range
    params.y_coef =
        make_float3(0.2126f * 255.0f * scale, 0.7152f * 255.0f * scale,
                    0.0722f * 255.0f * scale);
    params.y_off = 0.0f;

    params.u_coef =
        make_float3(-0.1146f * 255.0f * scale, -0.3854f * 255.0f * scale,
                    0.5000f * 255.0f * scale);
    params.u_off = 128.0f * scale;

    params.v_coef =
        make_float3(0.5000f * 255.0f * scale, -0.4542f * 255.0f * scale,
                    -0.0458f * 255.0f * scale);
    params.v_off = 128.0f * scale;
  } else {
    // BT.709 Limited Range
    params.y_coef =
        make_float3(0.2126f * 219.0f * scale, 0.7152f * 219.0f * scale,
                    0.0722f * 219.0f * scale);
    params.y_off = 16.0f * scale;

    params.u_coef =
        make_float3(-0.1146f * 224.0f * scale, -0.3854f * 224.0f * scale,
                    0.5000f * 224.0f * scale);
    params.u_off = 128.0f * scale;

    params.v_coef =
        make_float3(0.5000f * 224.0f * scale, -0.4542f * 224.0f * scale,
                    -0.0458f * 224.0f * scale);
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
__global__ void
postprocess_yuv_kernel(const half *__restrict__ srcTensor,
                       T *__restrict__ dstFrame, int dstWidth, int dstHeight,
                       int dstHeightC, int dstOffsetY, int dstBitdepth,
                       size_t dstPitch, int overlap_pixels,
                       const ColorConvParams ccp, int num_comps) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2, "T must be uint8 or uint16");
  const int local_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z; // tile index in batch

  // TILE_SIZE is Model Input Tile. Output/Effective Tile is scaled.
  // We assume 'overlap_pixels' passed here is ALREADY SCALED by process().
  // And we iterate over the Scaled Tile.
  const int effective_tile_dim = TILE_SIZE * 4;
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
      fmaf(ccp.y_coef.x, red,
           fmaf(ccp.y_coef.y, green, fmaf(ccp.y_coef.z, blue, ccp.y_off)));
  float u_comp =
      fmaf(ccp.u_coef.x, red,
           fmaf(ccp.u_coef.y, green, fmaf(ccp.u_coef.z, blue, ccp.u_off)));
  float v_comp =
      fmaf(ccp.v_coef.x, red,
           fmaf(ccp.v_coef.y, green, fmaf(ccp.v_coef.z, blue, ccp.v_off)));

  uint8_t *dstY_bytes = reinterpret_cast<uint8_t *>(dstFrame);
  T *rowY = reinterpret_cast<T *>(dstY_bytes + dst_y * dstPitch);
  rowY[dst_x] = clip_val<T>(y_comp, dstBitdepth);

  // Chroma Store
  // Determine if we need to store chroma for this pixel.
  // Assuming 4:2:0 if dstHeight != dstHeightC
  // only considers 4:2:0 and 4:2:2
  bool is_420 = (dstHeight != dstHeightC);

  if ((dst_x & 1) || (is_420 && (dst_y & 1)))
    return;

  // Calculate UV coordinates
  int uv_x = dst_x >> 1;
  int uv_y = is_420 ? (dst_y >> 1) : dst_y;

  // Base pointers
  // U is at Y + H * Pitch
  uint8_t *dstU_bytes = dstY_bytes + dstHeight * dstPitch;
  T *rowU = reinterpret_cast<T *>(dstU_bytes + uv_y * dstPitch);

  if (num_comps == 2) {
    // Interleaved: U V U V
    // U at 2*uv_x, V at 2*uv_x+1
    rowU[2 * uv_x] = clip_val<T>(u_comp, dstBitdepth);
    rowU[2 * uv_x + 1] = clip_val<T>(v_comp, dstBitdepth);
  } else {
    // Planar: V is at U + HC * Pitch
    uint8_t *dstV_bytes = dstU_bytes + dstHeightC * dstPitch;
    T *rowV = reinterpret_cast<T *>(dstV_bytes + uv_y * dstPitch);

    rowU[uv_x] = clip_val<T>(u_comp, dstBitdepth);
    rowV[uv_x] = clip_val<T>(v_comp, dstBitdepth);
  }
}

void postprocess(const PostprocessParams &params, cudaStream_t stream) {
  // TILE_SIZE is input tile size. Output is 4x.
  const int effective_tile_dim = TILE_SIZE * 4;
  const int tile_size = effective_tile_dim - 2 * params.overlap_pixels;
  const int batch = (params.dst_width[0] + tile_size - 1) / tile_size;

  dim3 block(32, 32, 1);
  dim3 grid;
  grid.x = (tile_size + 31) / 32;
  grid.y = (tile_size + 31) / 32;
  grid.z = batch;

  ColorConvParams cc_params{};
  // Full range flag and bitdepth needed for conversion coeffs
  populate_ccparams(cc_params, params.video_full_range_flag,
                    params.dst_bitdepth);

  int dstHeightC = params.dst_height[1];

  if (params.dst_bitdepth == 8) {
    postprocess_yuv_kernel<uint8_t><<<grid, block, 0, stream>>>(
        params.d_srcTensor, (uint8_t *)params.d_dst, params.dst_width[0],
        params.dst_height[0], dstHeightC, params.dst_start_y,
        params.dst_bitdepth, params.dst_pitch, params.overlap_pixels, cc_params,
        params.dst_num_comps);
  } else {
    postprocess_yuv_kernel<uint16_t><<<grid, block, 0, stream>>>(
        params.d_srcTensor, (uint16_t *)params.d_dst, params.dst_width[0],
        params.dst_height[0], dstHeightC, params.dst_start_y,
        params.dst_bitdepth, params.dst_pitch, params.overlap_pixels, cc_params,
        params.dst_num_comps);
  }
}