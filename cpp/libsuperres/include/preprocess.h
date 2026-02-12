#pragma once
#include "common.h"

struct PreprocessParams {
  const FrameBuffer *src_frame;
  TensorBuffer *dst_tensor;
  int start_tile_index;
  int batch_size;
  int overlap_pixels;
  bool video_full_range_flag;
};

void preprocess(const PreprocessParams &params, cudaStream_t stream);