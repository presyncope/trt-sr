#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct PostprocessParams {
  const half *d_srcTensor;
  void *d_dst;
  size_t dst_pitch;
  int dst_width[2];
  int dst_height[2];
  int dst_bitdepth;  // 8 or 10 or 16
  int dst_num_comps; // 2 if interleaved, 3 if planar
  int dst_start_y;
  int overlap_pixels;
  bool video_full_range_flag;
};

void postprocess(const PostprocessParams &params, cudaStream_t stream);