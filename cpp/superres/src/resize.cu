#include "resize.h"
#include <iostream>

template <typename T>
__global__ void resize_plane_linear_kernel(const T* __restrict__ src, int srcW, int srcH,
                                           T* __restrict__ dst, int dstW, int dstH) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    // Map to src coordinates
    float u = (x + 0.5f) * ((float)srcW / dstW) - 0.5f;
    float v = (y + 0.5f) * ((float)srcH / dstH) - 0.5f;

    int x0 = floorf(u);
    int y0 = floorf(v);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wu = u - x0;
    float wv = v - y0;

    // Clamp coordinates
    x0 = max(0, min(x0, srcW - 1));
    x1 = max(0, min(x1, srcW - 1));
    y0 = max(0, min(y0, srcH - 1));
    y1 = max(0, min(y1, srcH - 1));

    // Bilinear interpolation
    float val00 = static_cast<float>(src[y0 * srcW + x0]);
    float val10 = static_cast<float>(src[y0 * srcW + x1]);
    float val01 = static_cast<float>(src[y1 * srcW + x0]);
    float val11 = static_cast<float>(src[y1 * srcW + x1]);

    float val0 = val00 * (1.0f - wu) + val10 * wu;
    float val1 = val01 * (1.0f - wu) + val11 * wu;
    float val = val0 * (1.0f - wv) + val1 * wv;

    dst[y * dstW + x] = static_cast<T>(val + 0.5f);
}

void resize_yuv(const ResizeParams& params, cudaStream_t stream) {
    int bytes_per_pixel = (params.bitdepth > 8) ? 2 : 1;
    
    // Y Plane
    dim3 block(32, 32);
    dim3 gridY((params.dst_width + 31) / 32, (params.dst_height + 31) / 32);

    const void* srcY = params.src;
    void* dstY = params.dst;

    if (params.bitdepth == 8) {
        resize_plane_linear_kernel<uint8_t><<<gridY, block, 0, stream>>>(
            (const uint8_t*)srcY, params.src_width, params.src_height,
            (uint8_t*)dstY, params.dst_width, params.dst_height);
    } else {
        resize_plane_linear_kernel<uint16_t><<<gridY, block, 0, stream>>>(
            (const uint16_t*)srcY, params.src_width, params.src_height,
            (uint16_t*)dstY, params.dst_width, params.dst_height);
    }

    if (params.chfmt == 400) return;

    // Chroma Planes
    int srcCW = params.src_width;
    int srcCH = params.src_height;
    int dstCW = params.dst_width;
    int dstCH = params.dst_height;

    switch(params.chfmt) {
        case 420: // W/2, H/2
            srcCW /= 2; srcCH /= 2;
            dstCW /= 2; dstCH /= 2;
            break;
        case 422: // W/2, H
            srcCW /= 2;
            dstCW /= 2;
            break;
        case 444: // W, H
            break;
    }

    dim3 gridC((dstCW + 31) / 32, (dstCH + 31) / 32);
    
    size_t srcYSize = params.src_width * params.src_height * bytes_per_pixel;
    size_t dstYSize = params.dst_width * params.dst_height * bytes_per_pixel;
    size_t srcCSize = srcCW * srcCH * bytes_per_pixel;
    size_t dstCSize = dstCW * dstCH * bytes_per_pixel;

    const uint8_t* srcU_ptr = (const uint8_t*)params.src + srcYSize;
    const uint8_t* srcV_ptr = srcU_ptr + srcCSize;
    uint8_t* dstU_ptr = (uint8_t*)params.dst + dstYSize;
    uint8_t* dstV_ptr = dstU_ptr + dstCSize;

    if (params.bitdepth == 8) {
        resize_plane_linear_kernel<uint8_t><<<gridC, block, 0, stream>>>(
            (const uint8_t*)srcU_ptr, srcCW, srcCH,
            (uint8_t*)dstU_ptr, dstCW, dstCH);
        resize_plane_linear_kernel<uint8_t><<<gridC, block, 0, stream>>>(
            (const uint8_t*)srcV_ptr, srcCW, srcCH,
            (uint8_t*)dstV_ptr, dstCW, dstCH);
    } else {
        resize_plane_linear_kernel<uint16_t><<<gridC, block, 0, stream>>>(
            (const uint16_t*)srcU_ptr, srcCW, srcCH,
            (uint16_t*)dstU_ptr, dstCW, dstCH);
        resize_plane_linear_kernel<uint16_t><<<gridC, block, 0, stream>>>(
            (const uint16_t*)srcV_ptr, srcCW, srcCH,
            (uint16_t*)dstV_ptr, dstCW, dstCH);
    }
}
