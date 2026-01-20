#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct PreprocessParams {
  const uint8_t *d_srcY;
  const uint8_t *d_srcU;
  const uint8_t *d_srcV;
  half *d_dstTensor;
  int src_width;
  int src_height;
  int src_start_y;
  int overlap_pixels;
  bool video_full_range_flag;
};

void preprocess(const PreprocessParams &params, cudaStream_t stream);