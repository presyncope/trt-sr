#pragma once
#include "common.h"

enum class ResizeMethod {
  BILINEAR,
  BICUBIC
};

// Resizes all planes of a YUV image
void resize_yuv(const FrameBuffer &src, FrameBuffer& dst, ResizeMethod method, cudaStream_t stream);
