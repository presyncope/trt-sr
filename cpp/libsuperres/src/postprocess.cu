#include "postprocess.h"
#include <cuda_fp16.h>

// ------------------------------------------------------------------
// 1. Context & Helpers
// ------------------------------------------------------------------

struct ColorConvParams {
  float3 y_coef; float y_off;
  float3 u_coef; float u_off;
  float3 v_coef; float v_off;
};

struct KernelContext {
    // Destination Info
  int dst_width;
  int dst_height;
  int dst_stride_y;  // Bytes
  int dst_stride_uv; // Bytes
  bool is_nv12;      // True if UV interleaved
  int chroma_sub_x;  // Shift amount (e.g., 1 for 4:2:0/4:2:2)
  int chroma_sub_y;

  // Source & Tiling Info
  int src_valid_w;   // Tensor width - 2 * overlap
  int src_valid_h;
  int overlap;
  int tensor_stride; // Full tensor width (e.g. 512)

  int start_tile_index;
  int num_tiles_x;

  float scale_x;     // dst_width / src_virtual_width
  float scale_y;

  ColorConvParams ccp;
};

struct BilinearCoords {
  int x0, y0, x1, y1;
  float wx, wy;
};

__device__ __forceinline__ void compute_bilinear_coords(
    float u, float v, int limit_w, int limit_h, BilinearCoords* coords) 
{
  coords->x0 = __float2int_rd(u); // floor
  coords->y0 = __float2int_rd(v);
  
  // Boundary checkh
  coords->x1 = min(coords->x0 + 1, limit_w - 1);
  coords->y1 = min(coords->y0 + 1, limit_h - 1);
  coords->x0 = max(0, min(coords->x0, limit_w - 1));
  coords->y0 = max(0, min(coords->y0, limit_h - 1));

  coords->wx = u - coords->x0;
  coords->wy = v - coords->y0;
}

__device__ __forceinline__ float sample_plane_fast(
    const half *__restrict__ plane_ptr, 
    int stride, 
    const BilinearCoords& c) 
{
  // __ldg for Read-Only Cache
  const half* row0 = plane_ptr + c.y0 * stride;
  const half* row1 = plane_ptr + c.y1 * stride;

  float v00 = __half2float(__ldg(&row0[c.x0]));
  float v10 = __half2float(__ldg(&row0[c.x1]));
  float v01 = __half2float(__ldg(&row1[c.x0]));
  float v11 = __half2float(__ldg(&row1[c.x1]));

  // Interpolation
  float val_top = v00 + c.wx * (v10 - v00); 
  float val_bot = v01 + c.wx * (v11 - v01); 
  return val_top + c.wy * (val_bot - val_top);
}

// ------------------------------------------------------------------
// 2. Kernel
// ------------------------------------------------------------------

template <typename T>
__global__ void k_postprocess(
    const half *__restrict__ src_tensor,
    uint8_t *__restrict__ ptr_dst_y,   
    uint8_t *__restrict__ ptr_dst_uv1,
    uint8_t *__restrict__ ptr_dst_uv2, 
    KernelContext ctx)
{
  int batch_idx = blockIdx.z; // Batch Index = Tile Index in this grid setup

  int global_tile_idx = ctx.start_tile_index + batch_idx;
  int tx = global_tile_idx % ctx.num_tiles_x;
  int ty = global_tile_idx / ctx.num_tiles_x;

  // 1. Output Region Calculation
  int dst_start_x = (int)((long long)tx * ctx.src_valid_w * ctx.scale_x);
  int dst_end_x = (int)((long long)(tx + 1) * ctx.src_valid_w * ctx.scale_x);
  int dst_start_y = (int)((long long)ty * ctx.src_valid_h * ctx.scale_y);
  int dst_end_y = (int)((long long)(ty + 1) * ctx.src_valid_h * ctx.scale_y);

  dst_end_x = min(dst_end_x, ctx.dst_width);
  dst_end_y = min(dst_end_y, ctx.dst_height);

  // 2. Thread Local Mapping
  int dst_x = dst_start_x + blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = dst_start_y + blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_end_x || dst_y >= dst_end_y)
    return;

  // 3. Inverse Mapping (Dst -> Src Tensor)
  float global_src_x = (dst_x + 0.5f) / ctx.scale_x;
  float global_src_y = (dst_y + 0.5f) / ctx.scale_y;

  float tensor_u = global_src_x - (tx * ctx.src_valid_w) + ctx.overlap;
  float tensor_v = global_src_y - (ty * ctx.src_valid_h) + ctx.overlap;

  BilinearCoords coords;
  compute_bilinear_coords(tensor_u, tensor_v, ctx.tensor_stride, ctx.tensor_stride, &coords);

  // 4. Sampling RGB
  // Tensor Layout: [Batch, 3, H, W]
  long plane_pixels = (long)ctx.tensor_stride * ctx.tensor_stride;
  const half *ptr_r = src_tensor + (long)batch_idx * 3 * plane_pixels;
  const half *ptr_g = ptr_r + plane_pixels;
  const half *ptr_b = ptr_g + plane_pixels;

  float r = sample_plane_fast(ptr_r, ctx.tensor_stride, coords) * 255.0f;
  float g = sample_plane_fast(ptr_g, ctx.tensor_stride, coords) * 255.0f;
  float b = sample_plane_fast(ptr_b, ctx.tensor_stride, coords) * 255.0f;

  // 5. Color Conversion & Store
  constexpr float max_val = (float)((1 << (sizeof(T) * 8)) - 1);

  // Y Plane Write
  float y_val = ctx.ccp.y_off + ctx.ccp.y_coef.x * r + ctx.ccp.y_coef.y * g + ctx.ccp.y_coef.z * b;
  T *row_y = (T *)(ptr_dst_y + dst_y * ctx.dst_stride_y);
  row_y[dst_x] = (T)fminf(fmaxf(y_val, 0.0f), max_val);

  // Chroma Processing (Subsampled)
  bool is_chroma_site = ((dst_x & ((1 << ctx.chroma_sub_x) - 1)) == 0) &&
                        ((dst_y & ((1 << ctx.chroma_sub_y) - 1)) == 0);

  if (is_chroma_site)
  {
    float u_val = ctx.ccp.u_off + ctx.ccp.u_coef.x * r + ctx.ccp.u_coef.y * g + ctx.ccp.u_coef.z * b;
    float v_val = ctx.ccp.v_off + ctx.ccp.v_coef.x * r + ctx.ccp.v_coef.y * g + ctx.ccp.v_coef.z * b;

    int cx = dst_x >> ctx.chroma_sub_x;
    int cy = dst_y >> ctx.chroma_sub_y;

    T *row_u_base = (T *)(ptr_dst_uv1 + cy * ctx.dst_stride_uv);

    if (ctx.is_nv12)
    {
      // Interleaved (UVUV...)
      row_u_base[2 * cx + 0] = (T)fminf(fmaxf(u_val, 0.0f), max_val);
      row_u_base[2 * cx + 1] = (T)fminf(fmaxf(v_val, 0.0f), max_val);
    }
    else
    {
      // Planar (I420 etc.)
      T *row_v_base = (T *)(ptr_dst_uv2 + cy * ctx.dst_stride_uv);
      row_u_base[cx] = (T)fminf(fmaxf(u_val, 0.0f), max_val);
      row_v_base[cx] = (T)fminf(fmaxf(v_val, 0.0f), max_val);
    }
  }
}

// ------------------------------------------------------------------
// 3. Host Dispatcher
// ------------------------------------------------------------------

static void populate_ccparams(ColorConvParams &ccp, bool full_range, int bitdepth)
{
  float max_v = (float)((1 << bitdepth) - 1);
  float scale = max_v / 255.0f;

  if (full_range) {
    ccp.y_coef = make_float3(0.2126f, 0.7152f, 0.0722f); ccp.y_off = 0.0f;
    ccp.u_coef = make_float3(-0.1146f, -0.3854f, 0.5000f); ccp.u_off = 128.0f;
    ccp.v_coef = make_float3(0.5000f, -0.4542f, -0.0458f); ccp.v_off = 128.0f;
  } else {
    ccp.y_coef = make_float3(0.1826f, 0.6142f, 0.0620f); ccp.y_off = 16.0f;
    ccp.u_coef = make_float3(-0.1006f, -0.3386f, 0.4392f); ccp.u_off = 128.0f;
    ccp.v_coef = make_float3(0.4392f, -0.3989f, -0.0403f); ccp.v_off = 128.0f;
  }

  // Scale apply
  auto apply_scale = [&](float3 &c, float &off) {
    c.x *= scale; c.y *= scale; c.z *= scale; off *= scale;
  };
  apply_scale(ccp.y_coef, ccp.y_off);
  apply_scale(ccp.u_coef, ccp.u_off);
  apply_scale(ccp.v_coef, ccp.v_off);
}

void postprocess(const PostprocessParams &params, cudaStream_t stream) {
  KernelContext ctx;

  // Geometry Setup
  // params.src_tensor가 NCHW이고 4x super resolution이라고 가정하면 
  // tensor width는 input tile size * 4. 여기서는 TILE_SIZE 상수에 의존.
  ctx.tensor_stride = TILE_SIZE * 4;                                   

  // overlap_pixels는 원본 해상도 기준일 수 있으므로 scale factor(4)를 곱함
  ctx.overlap = params.overlap_pixels * 4;

  // 유효 영역(Valid Area) 계산: 양쪽 padding을 뺀 영역
  ctx.src_valid_w = ctx.tensor_stride - 2 * ctx.overlap; 
  ctx.src_valid_h = ctx.tensor_stride - 2 * ctx.overlap;

  ctx.dst_width = params.dst_frame->width;
  ctx.dst_height = params.dst_frame->height;
  ctx.start_tile_index = params.start_tile_index;

  // Global Tiling Info
  ctx.num_tiles_x = (params.src_virtual_width + ctx.src_valid_w - 1) / ctx.src_valid_w;

  // Scale Factors
  ctx.scale_x = (float)ctx.dst_width / params.src_virtual_width;
  ctx.scale_y = (float)ctx.dst_height / params.src_virtual_height;

  // Output Format Info
  ctx.dst_stride_y = params.dst_frame->planes[0].stride;
  ctx.dst_stride_uv = params.dst_frame->planes[1].stride;
  ctx.is_nv12 = is_semi_planar(params.dst_frame->format);
  ctx.chroma_sub_x = 1; // 4:2:0/4:2:2 assumed
  ctx.chroma_sub_y = get_chroma_shift_y(params.dst_frame->format);

  populate_ccparams(ctx.ccp, params.video_full_range_flag, params.dst_frame->bitdepth);

  // Launch Configuration
  // Max size of a tile in destination pixels
  float max_dst_tile_w = ctx.src_valid_w * ctx.scale_x;
  float max_dst_tile_h = ctx.src_valid_h * ctx.scale_y;

  dim3 block(32, 16);
  dim3 grid;
  grid.x = (int)((max_dst_tile_w + block.x - 1) / block.x);
  grid.y = (int)((max_dst_tile_h + block.y - 1) / block.y);
  grid.z = params.batch_size;

  // Safety check
  grid.x = max(1, grid.x);
  grid.y = max(1, grid.y);

  const half *src = static_cast<const half *>(params.src_tensor->data.get());
  uint8_t *ptr_dst_y = params.dst_frame->planes[0].data;
  uint8_t *ptr_dst_u = params.dst_frame->planes[1].data;
  uint8_t *ptr_dst_v = (params.dst_frame->num_planes > 2) ? params.dst_frame->planes[2].data : nullptr;

  if (params.dst_frame->bitdepth == 8) {
    k_postprocess<uint8_t><<<grid, block, 0, stream>>>(src, ptr_dst_y, ptr_dst_u, ptr_dst_v, ctx);
  } else {
    k_postprocess<uint16_t><<<grid, block, 0, stream>>>(src, ptr_dst_y, ptr_dst_u, ptr_dst_v, ctx);
  }
}
