#pragma once
#include "common.h"

struct PostprocessParams {
  // Src Partial Info
  const TensorBuffer *src_tensor; // [Batch, 3, 512, 512] NCHW layout assumed
  int batch_size;
  int overlap_pixels;

  // Src Global Geometrics
  int src_virtual_width; // e.g., 7680
  int src_virtual_height; // e.g., 4320
  int start_tile_index;

  // Dst Frame Info
  FrameBuffer *dst_frame;
  bool video_full_range_flag;
};

void postprocess(const PostprocessParams &params, cudaStream_t stream);