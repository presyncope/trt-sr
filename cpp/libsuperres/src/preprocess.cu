#include "preprocess.h"
#include <cmath>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

// ------------------------------------------------------------------
// 1. Helpers & Context
// ------------------------------------------------------------------

struct ColorConvParams {
  float y_offset, y_mult;
  float r_v, g_u, g_v, b_u; // Simplified coefficients
};

struct KernelContext {
  int src_width, src_height;
  int src_stride_y, src_stride_uv; // stride in bytes
  int start_tile_index;
  int num_tiles_x;
  int tile_valid_size;
  int overlap;
  float norm_scale; // Pre-calculated: 255.0f / ((1 << bitdepth) - 1)
  ColorConvParams ccp;
  bool is_interleaved;
  int chroma_shift_y;
};

// Reflect 101 padding (gfedcb|abcdefgh|gfedcba)
__device__ __forceinline__ int reflect_coord(int x, int limit) {
  x = ::abs(x);
  return (limit - 1) - ::abs(x - (limit - 1));
}

// Optimized memory load using Read-Only Cache (__ldg)
template <typename T>
__device__ __forceinline__ T load_val(const uint8_t* ptr, int offset) {
  const T* typed_ptr = reinterpret_cast<const T*>(ptr);
  return __ldg(&typed_ptr[offset]);
}


// ------------------------------------------------------------------
// 2. Kernel
// ------------------------------------------------------------------

template <typename T>
__global__ void preprocess_kernel_optimized(
    const uint8_t *__restrict__ src_y,
    const uint8_t *__restrict__ src_u,
    const uint8_t *__restrict__ src_v,
    half *__restrict__ dst_tensor,
    KernelContext ctx)
{
  // Output Coordinates (within a tile)
  const int local_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int local_y = threadIdx.y + blockIdx.y * blockDim.y;
  const int tile_idx = blockIdx.z; // Process one tile per Z-block

  if (local_x >= TILE_SIZE || local_y >= TILE_SIZE)
    return;

  // 1. Calculate Global Tile Position
  int global_tile_idx = ctx.start_tile_index + tile_idx;
  int tx = global_tile_idx % ctx.num_tiles_x;
  int ty = global_tile_idx / ctx.num_tiles_x;

  int src_start_x = tx * ctx.tile_valid_size;
  int src_start_y = ty * ctx.tile_valid_size;

  // 2. Map to Source Coordinates with Reflection
  int global_x = src_start_x + local_x - ctx.overlap;
  int global_y = src_start_y + local_y - ctx.overlap;

  global_x = reflect_coord(global_x, ctx.src_width);
  global_y = reflect_coord(global_y, ctx.src_height);

  // 3. Load YUV Data
  float y_val = 0.0f, u_val = 0.0f, v_val = 0.0f;

  // Load Y
  // Note: stride is in bytes. T* access assumes properly aligned pixel offset.
  const uint8_t *row_y = src_y + global_y * ctx.src_stride_y;
  y_val = (float)load_val<T>(row_y, global_x) * ctx.norm_scale;

  // Chroma Subsampling
  int uv_x = global_x >> 1;
  int uv_y = global_y >> ctx.chroma_shift_y;

  if (ctx.is_interleaved)
  {
    // NV12/NV21 (Interleaved UV)
    // Stride is in bytes. offset 2*uv_x puts us at the UV pair.
    const uint8_t *row_uv = src_u + uv_y * ctx.src_stride_uv;
    // Read 2 consecutive values (U and V)
    // Optimization: Could read uchar2/ushort2 for better bandwidth
    u_val = (float)load_val<T>(row_uv, 2 * uv_x) * ctx.norm_scale;
    v_val = (float)load_val<T>(row_uv, 2 * uv_x + 1) * ctx.norm_scale;
  }
  else
  {
    // I420 (Planar)
    u_val = (float)load_val<T>(src_u + uv_y * ctx.src_stride_uv, uv_x) * ctx.norm_scale;
    v_val = (float)load_val<T>(src_v + uv_y * ctx.src_stride_uv, uv_x) * ctx.norm_scale; // stride U=V usually
  }

  // 4. Color Conversion (YUV -> RGB)
  // Offset correction: u/v are 0..255 (normalized from bitdepth), subtract 128
  float y_comp = fmaf(ctx.ccp.y_mult, y_val, ctx.ccp.y_offset);
  float u_comp = u_val - 128.0f;
  float v_comp = v_val - 128.0f;

  float red = y_comp + ctx.ccp.r_v * v_comp;
  float green = y_comp - (ctx.ccp.g_u * u_comp + ctx.ccp.g_v * v_comp);
  float blue = y_comp + ctx.ccp.b_u * u_comp;

  // 5. Write Output (NCHW Planar)
  constexpr float inv_255 = 1.0f / 255.0f;
  constexpr int plane_pixels = TILE_SIZE * TILE_SIZE;

  // Output ptr offset: Batch(n) * 3 channels * plane size
  half *dst_tile_base = dst_tensor + (tile_idx * 3 * plane_pixels);
  int px_idx = local_y * TILE_SIZE + local_x;

  dst_tile_base[px_idx] = __float2half(__saturatef(red * inv_255));
  dst_tile_base[plane_pixels + px_idx] = __float2half(__saturatef(green * inv_255));
  dst_tile_base[2 * plane_pixels + px_idx] = __float2half(__saturatef(blue * inv_255));
}

// ------------------------------------------------------------------
// 3. Host Dispatcher
// ------------------------------------------------------------------

void preprocess(const PreprocessParams &params, cudaStream_t stream) {
  // 1. Setup Context
  KernelContext ctx;
  ctx.src_width = params.src_frame->width;
  ctx.src_height = params.src_frame->height;
  ctx.src_stride_y = params.src_frame->planes[0].stride;
  ctx.src_stride_uv = params.src_frame->planes[1].stride;

  ctx.start_tile_index = params.start_tile_index;
  ctx.tile_valid_size = TILE_SIZE - 2 * params.overlap_pixels;
  ctx.overlap = params.overlap_pixels;

  // Calculate Grid Layout
  ctx.num_tiles_x = (ctx.src_width + ctx.tile_valid_size - 1) / ctx.tile_valid_size;

  // Setup Color Conversion Params
  bool full_range = params.video_full_range_flag;
  if (full_range) {
    ctx.ccp = {0.0f, 1.0f, 1.5748f, 0.1873f, 0.4681f, 1.8556f}; // Coefficients simplified signs handled in kernel
  } else {
    ctx.ccp = {-16.0f, 1.164f, 1.793f, 0.213f, 0.533f, 2.112f};
  }

  int bitdepth = get_bit_depth(params.src_frame->format);
  ctx.norm_scale = 255.0f / (float)((1 << bitdepth) - 1);
  ctx.is_interleaved = is_semi_planar(params.src_frame->format);
  ctx.chroma_shift_y = get_chroma_shift_y(params.src_frame->format);

  // 2. Kernel Launch Config
  // Block: 32x32 covers a standard tile. If TILE_SIZE > 32, we need grid expansion.
  dim3 block(32, 32, 1);
  dim3 grid;
  grid.x = (TILE_SIZE + block.x - 1) / block.x;
  grid.y = (TILE_SIZE + block.y - 1) / block.y;
  grid.z = params.batch_size; // Z dimension handles the batch/tiles

  const uint8_t *src_y = params.src_frame->planes[0].data;
  const uint8_t *src_u = params.src_frame->planes[1].data;
  const uint8_t *src_v = (params.src_frame->num_planes > 2) ? params.src_frame->planes[2].data : nullptr;
  half *dst_ptr = static_cast<half *>(params.dst_tensor->data.get());

  if (bitdepth == 8)
  {
    preprocess_kernel_optimized<uint8_t><<<grid, block, 0, stream>>>(
        src_y, src_u, src_v, dst_ptr, ctx);
  }
  else
  {
    preprocess_kernel_optimized<uint16_t><<<grid, block, 0, stream>>>(
        src_y, src_u, src_v, dst_ptr, ctx);
  }
}