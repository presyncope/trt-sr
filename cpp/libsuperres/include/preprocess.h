#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct PreprocessParams {
  const void *d_src;
  int src_pitch;
  int src_width[2];
  int src_height[2];
  int src_bitdepth;  // 8 or 10 or 16
  int src_num_comps; // 2 if interleaved, 3 if planar
  int src_start_y;
  half *d_dstTensor; // shape=(n, 3, TILE_SIZE, TILE_SIZE)
  int overlap_pixels;
  bool video_full_range_flag;
};

void preprocess(const PreprocessParams &params, cudaStream_t stream);