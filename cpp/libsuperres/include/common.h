#pragma once
#include <cuda_runtime.h>
#include <memory>
#include "superres.h"

static constexpr int TILE_SIZE = 128;

struct CudaDeleter {
  void operator()(void *p) const { if (p) (void)cudaFree(p); }
};

struct StreamDeleter {
  void operator()(cudaStream_t s) const { if (s) (void)cudaStreamDestroy(s); }
};

struct FramePlane {
  uint8_t *data;   // Pointer to the start of this plane
  int width_bytes; // Valid data width in bytes
  int height;      // Number of rows
  size_t stride;   // Pitch (stride) in bytes
};

struct FrameBuffer {
  std::unique_ptr<void, CudaDeleter> raw_data; // Owns the memory
  sr_pixel_format format;
  int width;  // Logical Image Width (pixels)
  int height; // Logical Image Height (pixels)
  int bitdepth;
  int num_planes;

  // Explicit descriptors for up to 3 planes (Y, U, V or Y, UV)
  FramePlane planes[3];
};

struct TensorBuffer {
  std::unique_ptr<void, CudaDeleter> data;
  size_t total_size;
};

int get_bit_depth(sr_pixel_format fmt);
bool is_semi_planar(sr_pixel_format fmt);
int get_chroma_shift_y(sr_pixel_format fmt);