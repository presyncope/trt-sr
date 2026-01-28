#pragma once

#include <stdint.h>

typedef struct real_esrgan_context *real_esrgan_handle;

typedef struct real_esrgan_init_params {
  const char *plan_file;
  int input_width;
  int input_height;
  int input_bitdepth; // range: 8~16
  int input_chfmt;    // {420,422,444,400}
  int output_width;
  int output_height;
  int output_bitdepth; // range: 8~16
  int output_chfmt;    // {420,422,444,400}
  float prescale; // range: 0.25~1.0 , used for performance increment
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
