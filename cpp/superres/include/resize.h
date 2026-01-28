#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

struct ResizeParams {
    const void* src; // uint8_t* or uint16_t*
    void* dst;       // uint8_t* or uint16_t*
    int src_width;
    int src_height;
    int dst_width;
    int dst_height;
    int bitdepth; // 8 or 16
    int chfmt; // 420, 422, 444, 400
};

// Resizes all planes of a YUV image based on params
void resize_yuv(const ResizeParams& params, cudaStream_t stream);
