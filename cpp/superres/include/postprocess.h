#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct PostprocessParams {
  const half *d_srcTensor;
  uint8_t *d_dstY;
  uint8_t *d_dstU;
  uint8_t *d_dstV;
  int dst_width;
  int dst_height;
  int dst_start_y;
  int overlap_pixels;
  bool video_full_range_flag;
};

void postprocess(const PostprocessParams &params, cudaStream_t stream);