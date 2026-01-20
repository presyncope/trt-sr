#pragma once

#include <stdint.h>

typedef struct real_esrgan_context *real_esrgan_handle;

typedef struct real_esrgan_init_params {
  const char *plan_file;
  int width;
  int height;
  int overlap_pixels;
} real_esrgan_init_params;

#if __cplusplus
extern "C" {
#endif

int real_esrgan_build(const char *model_onnx, const char *plan_file);

real_esrgan_handle real_esrgan_create();

int real_esrgan_destroy(real_esrgan_handle handle);

int real_esrgan_init(real_esrgan_handle handle,
                     const real_esrgan_init_params *params);

int real_esrgan_process(real_esrgan_handle handle, const uint8_t *srcYuv,
                        uint8_t *dstYuv);

#if __cplusplus
}
#endif
