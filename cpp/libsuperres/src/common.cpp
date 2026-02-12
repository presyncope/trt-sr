#include "common.h"

int get_bit_depth(sr_pixel_format fmt) {
  switch (fmt) {
    case SR_PIXEL_FMT_I016: case SR_PIXEL_FMT_P016: 
    case SR_PIXEL_FMT_I216: case SR_PIXEL_FMT_P216: return 16;
    case SR_PIXEL_FMT_I010: case SR_PIXEL_FMT_P010: 
    case SR_PIXEL_FMT_I210: case SR_PIXEL_FMT_P210: return 10;
    default: return 8;
  }
}

bool is_semi_planar(sr_pixel_format fmt) {
  switch (fmt) {
    case SR_PIXEL_FMT_NV12: case SR_PIXEL_FMT_NV16:
    case SR_PIXEL_FMT_P010: case SR_PIXEL_FMT_P210:
    case SR_PIXEL_FMT_P016: case SR_PIXEL_FMT_P216: return true;
    default: return false;
  }
}

int get_chroma_shift_y(sr_pixel_format fmt) {
  // Returns the divisor for chroma height (e.g., 2 for 4:2:0, 1 for 4:2:2)
  switch (fmt) {
    case SR_PIXEL_FMT_I420: case SR_PIXEL_FMT_NV12:
    case SR_PIXEL_FMT_I010: case SR_PIXEL_FMT_P010:
    case SR_PIXEL_FMT_I016: case SR_PIXEL_FMT_P016: return 1;
    default: return 0; // 4:2:2 formats
  }
}
