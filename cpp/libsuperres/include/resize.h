#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

struct ResizeParams {
  const void *src_ptr;
  size_t src_pitch;
  void *dst_ptr;
  size_t dst_pitch;
  int src_width[2];
  int src_height[2];
  int dst_width[2];
  int dst_height[2];
  int bitdepth;  // 8 or 10 or 16
  int num_comps; // 2 if interleaved, 3 if planar
};

// Resizes all planes of a YUV image based on params
void resize_yuv(const ResizeParams &params, cudaStream_t stream);
