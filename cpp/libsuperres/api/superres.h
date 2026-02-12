#pragma once
#define ENABLE_NVONNXPARSER 1

#include <stddef.h>
#include <stdint.h>

typedef struct sr_context *sr_handle;

typedef enum sr_pixel_format {
  /* =================================================================
   * 8-BIT FORMATS
   * ================================================================= */

  /* --- 4:2:0 (8-bit) --- */
  SR_PIXEL_FMT_I420 = 0, // [Planar] Y, U, V
  SR_PIXEL_FMT_NV12 = 1, // [Semi-Planar] Y, UV interleaved

  /* --- 4:2:2 (8-bit) --- */
  SR_PIXEL_FMT_I422 = 2, // [Planar] Y, U, V
  SR_PIXEL_FMT_NV16 = 3, // [Semi-Planar] Y, UV interleaved

  /* =================================================================
   * 10-BIT FORMATS (Stored in uint16_t, 0~1023 range)
   * ================================================================= */

  /* --- 4:2:0 --- */
  SR_PIXEL_FMT_I010 = 8, // [Planar] 10-bit I420
  SR_PIXEL_FMT_P010 = 9, // [Semi-Planar] 10-bit NV12 (Most Common for HDR)

  /* --- 4:2:2 --- */
  SR_PIXEL_FMT_I210 = 10, // [Planar] 10-bit I422
  SR_PIXEL_FMT_P210 = 11, // [Semi-Planar] 10-bit NV16

  /* =================================================================
   * 16-BIT FORMATS (Little Endian)
   * ================================================================= */

  /* --- 4:2:0 (16-bit) --- */
  // [Planar 16-bit] 3 Planes: Y(16b), U(16b), V(16b)
  SR_PIXEL_FMT_I016 = 16,

  // [Semi-Planar 16-bit] 2 Planes: Y(16b), UV(16b interleaved)
  SR_PIXEL_FMT_P016 = 17,

  /* --- 4:2:2 (16-bit) --- */
  // [Planar 16-bit] 3 Planes: Y(16b), U(16b), V(16b)
  SR_PIXEL_FMT_I216 = 18,

  // [Semi-Planar 16-bit] 2 Planes: Y(16b), UV(16b interleaved)
  SR_PIXEL_FMT_P216 = 19,

  SR_PIXEL_FMT_UNKNOWN = -1

} sr_pixel_format;

typedef struct sr_init_params {
  const char *plan_file;
  int input_width;
  int input_height;
  sr_pixel_format input_format;
  int input_color_fullrange; // default 0 (limited range)
  int output_width;
  int output_height;
  sr_pixel_format output_format;
  int output_color_fullrange; // default 0 (limited range)
  float prescale;     // range: 0.25 ~ 1.0 , used for performance increment
  int overlap_pixels; // 4 or 2 is recommended
  int concurrent_batches;
} sr_init_params;

typedef struct sr_build_params {
  const char *model_onnx; // [src, required]
  const char *plan_file;  // [dst, required]
  uint64_t max_workspace_size;
  uint64_t max_shared_memory;
  int min_batch_size;
  int max_batch_size;
  int optimal_batch_size;
  bool strongly_typed;
} sr_build_params;

typedef struct sr_frame {
  void *data[3];
  size_t stride[3];
} sr_frame;

#if __cplusplus
extern "C" {
#endif

/**
 * @brief Build a TensorRT engine plan from an ONNX model.
 *
 * @param model_onnx Path to the source ONNX model file.
 * @param plan_file Path to the destination TensorRT plan file.
 * @return int 0 on success, negative value on failure.
 */
int sr_build(const sr_build_params *params);

/**
 * @brief Create a new super resolution context handle.
 *
 * @return sr_handle Detailed opaque handle to the context, or NULL if creation
 * failed.
 */
sr_handle sr_create();

/**
 * @brief Destroy the super resolution context and release resources.
 *
 * @param handle The handle to the context to be destroyed.
 * @return int 0 on success, negative value on failure.
 */
int sr_destroy(sr_handle handle);

/**
 * @brief Initialize the super resolution context with parameters.
 *
 * @param handle The context handle.
 * @param params Initialization parameters including model path and dimensions.
 * @return int 0 on success, negative value on failure.
 */
int sr_init(sr_handle handle, const sr_init_params *params);

/**
 * @brief Process a frame (perform super resolution).
 *
 * @param handle The context handle.
 * @param src_frame Pointer to the input image data.
 * @param dst_frame Pointer to the output image buffer.
 * @return int 0 on success, negative value on failure.
 */
int sr_process(sr_handle handle, const sr_frame *src_frame,
               sr_frame *dst_frame);

#if __cplusplus
}
#endif
