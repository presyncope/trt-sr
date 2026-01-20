#include "common.h"
#include "preprocess.h"
#include <cuda_runtime.h>

struct ColorConvParams {
  float y_offset;
  float y_mult;
  float m1;
  float m21;
  float m22;
  float m3;
};

static void populate_ccparams(ColorConvParams &params, bool full_range) {
  if (full_range) {
    params.y_offset = 0.0f;
    params.y_mult = 1.0f;
    params.m1 = 1.5748f;
    params.m21 = -0.1873f;
    params.m22 = -0.4681f;
    params.m3 = 1.8556f;
  } else {
    params.y_offset = -16.0f;
    params.y_mult = 1.164f;
    params.m1 = 1.793f;
    params.m21 = -0.213f;
    params.m22 = -0.533f;
    params.m3 = 2.112f;
  }
}

__global__ void preprocess_iyuv_kernel(const uint8_t *__restrict__ srcY,
                                       const uint8_t *__restrict__ srcU,
                                       const uint8_t *__restrict__ srcV,
                                       half *__restrict__ dstTensor,
                                       int srcWidth, int srcHeight,
                                       int srcOffsetY, int overlap_pixels,
                                       const ColorConvParams color_params) {
  const int local_x = blockIdx.x * blockDim.x + threadIdx.x; // 0~TILE_SIZE-1
  const int local_y = blockIdx.y * blockDim.y + threadIdx.y; // 0~TILE_SIZE-1
  const int tile_size = TILE_SIZE - 2 * overlap_pixels;
  const int n = blockIdx.z;

  if (local_x >= TILE_SIZE || local_y >= TILE_SIZE)
    return;

  // mirror padding
  int global_x = abs(n * tile_size + local_x - overlap_pixels);
  int global_y = abs(srcOffsetY + local_y - overlap_pixels); 
  global_x = (srcWidth - 1) - abs(global_x - (srcWidth - 1));
  global_y = (srcHeight - 1) - abs(global_y - (srcHeight - 1));

  const int uv_offset = (global_y >> 1) * (srcWidth >> 1) + (global_x >> 1);
  float y_comp = static_cast<float>(srcY[global_y * srcWidth + global_x]);
  float u_comp = static_cast<float>(srcU[uv_offset]);
  float v_comp = static_cast<float>(srcV[uv_offset]);

  // yuv to rgb
  y_comp = fmaf(color_params.y_mult, y_comp, color_params.y_offset);
  u_comp -= 128.0f;
  v_comp -= 128.0f;

  float red = fmaf(color_params.m1, v_comp, y_comp);
  float green = fmaf(color_params.m21, u_comp, y_comp);
  green = fmaf(color_params.m22, v_comp, green);
  float blue = fmaf(color_params.m3, u_comp, y_comp);

  // normalize
  constexpr float inv_255 = 1.0f / 255.0f;
  red = __saturatef(red * inv_255);
  green = __saturatef(green * inv_255);
  blue = __saturatef(blue * inv_255);

  // write to output tensor in NCHW format
  constexpr size_t plane_size = TILE_SIZE * TILE_SIZE;
  const size_t pixel_idx = local_y * TILE_SIZE + local_x;
  half *dst_tile = dstTensor + (n * 3 * plane_size);

  dst_tile[pixel_idx] = __float2half(red);
  dst_tile[plane_size + pixel_idx] = __float2half(green);
  dst_tile[2 * plane_size + pixel_idx] = __float2half(blue);
}

void preprocess(const PreprocessParams &params, cudaStream_t stream) {
  const int tile_size = TILE_SIZE - 2 * params.overlap_pixels;
  const int batch = (params.src_width + tile_size - 1) / tile_size;

  dim3 block(32, 32, 1); // 1024 threads
  dim3 grid;
  grid.x = (TILE_SIZE / 32);
  grid.y = (TILE_SIZE / 32);
  grid.z = batch;

  ColorConvParams cc_params{};
  populate_ccparams(cc_params, params.video_full_range_flag);

  preprocess_iyuv_kernel<<<grid, block, 0, stream>>>(
      params.d_srcY, params.d_srcU, params.d_srcV, params.d_dstTensor,
      params.src_width, params.src_height, params.src_start_y,
      params.overlap_pixels, cc_params);
}