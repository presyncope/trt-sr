#include "superres.h"
#include "common.h"
#include "postprocess.h"
#include "preprocess.h"
#include "resize.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;
using namespace nvonnxparser;

#define CUDA_CHECK(callstr)                                                    \
  {                                                                            \
    cudaError_t ecode = callstr;                                               \
    if (ecode != cudaSuccess) {                                                \
      std::cerr << "CUDA error " << ecode << " at " << __FILE__ << ":"         \
                << __LINE__;                                                   \
      return -1;                                                               \
    }                                                                          \
  }

namespace {
class Logger : public ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT " << getSeverityString(severity) << "] " << msg
                << std::endl;
    }
  }

  const char *getSeverityString(Severity severity) {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case Severity::kERROR:
      return "ERROR";
    case Severity::kWARNING:
      return "WARNING";
    case Severity::kINFO:
      return "INFO";
    case Severity::kVERBOSE:
      return "VERBOSE";
    default:
      return "UNKNOWN";
    }
  }
} gLogger;
} // namespace

struct CudaDeleter {
  void operator()(void *p) const {
    if (p)
      (void)cudaFree(p);
  }
};

struct StreamDeleter {
  void operator()(cudaStream_t s) const {
    if (s)
      (void)cudaStreamDestroy(s);
  }
};

struct frame_buffer {
  std::unique_ptr<void, CudaDeleter> data;
  int num_comps;
  int bitdepth;
  int width[2];
  int height[2];
  size_t stride;
};

struct tensor_buffer {
  std::unique_ptr<void, CudaDeleter> data;
  size_t total_size;
};

struct sr_context {
  std::unique_ptr<IRuntime> trt_runtime;
  std::unique_ptr<ICudaEngine> trt_engine;
  std::unique_ptr<IExecutionContext> trt_context;
  std::unique_ptr<std::remove_pointer<cudaStream_t>::type, StreamDeleter>
      cuda_stream;

  // Buffers
  frame_buffer input_frame;  // Model Input
  frame_buffer output_frame; // Model Output (Pre-resize)
  tensor_buffer input_tensor;
  tensor_buffer output_tensor;

  // Extra buffers for Resizing
  frame_buffer src_original; // Original Input
  frame_buffer dst_final;    // Final Output

  std::string input_name;
  std::string output_name;

  sr_init_params ipar;

  bool need_pre_resize;
  bool need_post_resize;
  int num_tiles_in_x;
  int num_tiles_in_y;
  int model_input_width;
  int model_input_height;
  int model_output_width;
  int model_output_height;
};

static int create_frame_buffer(frame_buffer &buf, int w, int h,
                               sr_pixel_format fmt) {
  buf.width[0] = w;
  buf.height[0] = h;
  buf.bitdepth = (fmt >= SR_PIXEL_FMT_I016)   ? 16
                 : (fmt >= SR_PIXEL_FMT_I010) ? 10
                                              : 8;

  int bytes_per_pixel = (buf.bitdepth > 8) ? 2 : 1;
  int total_height = h;

  switch (fmt) {
  case SR_PIXEL_FMT_I420:
  case SR_PIXEL_FMT_I010:
  case SR_PIXEL_FMT_I016:
    buf.num_comps = 3;
    buf.width[1] = w / 2;
    buf.height[1] = h / 2;
    total_height += h;
    break;
  case SR_PIXEL_FMT_NV12:
  case SR_PIXEL_FMT_P010:
  case SR_PIXEL_FMT_P016:
    buf.num_comps = 2;
    buf.width[1] = w; // UV interleaved
    buf.height[1] = h / 2;
    total_height += h / 2;
    break;
  case SR_PIXEL_FMT_I422:
  case SR_PIXEL_FMT_I210:
  case SR_PIXEL_FMT_I216:
    buf.num_comps = 3;
    buf.width[1] = w / 2;
    buf.height[1] = h;
    total_height += 2 * h;
    break;
  case SR_PIXEL_FMT_NV16:
  case SR_PIXEL_FMT_P210:
  case SR_PIXEL_FMT_P216:
    buf.num_comps = 2;
    buf.width[1] = w; // UV interleaved
    buf.height[1] = h;
    total_height += h;
    break;

  default:
    return -1;
  }

  size_t pitch = 0;
  void *dev_ptr = nullptr;

  size_t max_row_bytes = w * bytes_per_pixel;
  CUDA_CHECK(cudaMallocPitch(&dev_ptr, &pitch, max_row_bytes, total_height));
  buf.data.reset(dev_ptr);
  buf.stride = pitch;
  return 0;
}

static int copy_frame_to_gpu(const sr_frame &src, const frame_buffer &dst,
                             cudaStream_t stream) {
  int bpp = (dst.bitdepth > 8) ? 2 : 1;

  // Plane 0 (Y)
  int w_bytes_0 = dst.width[0] * bpp;
  CUDA_CHECK(cudaMemcpy2DAsync(dst.data.get(), dst.stride, src.data[0],
                               src.stride[0], w_bytes_0, dst.height[0],
                               cudaMemcpyHostToDevice, stream));

  if (dst.num_comps > 1) {
    // Plane 1 (U or UV)
    uint8_t *dst_p1 = (uint8_t *)dst.data.get() + dst.stride * dst.height[0];
    int w_bytes_1 = dst.width[1] * bpp;
    CUDA_CHECK(cudaMemcpy2DAsync(dst_p1, dst.stride, src.data[1], src.stride[1],
                                 w_bytes_1, dst.height[1],
                                 cudaMemcpyHostToDevice, stream));

    if (dst.num_comps > 2) {
      // Plane 2 (V)
      uint8_t *dst_p2 = dst_p1 + dst.stride * dst.height[1];
      // V dimensions same as U for supported 3-plane formats
      CUDA_CHECK(cudaMemcpy2DAsync(dst_p2, dst.stride, src.data[2],
                                   src.stride[2], w_bytes_1, dst.height[1],
                                   cudaMemcpyHostToDevice, stream));
    }
  }
  return 0;
}

static int copy_frame_to_host(const frame_buffer &src, sr_frame &dst,
                              cudaStream_t stream) {
  int bpp = (src.bitdepth > 8) ? 2 : 1;

  // Plane 0
  int w_bytes_0 = src.width[0] * bpp;
  CUDA_CHECK(cudaMemcpy2DAsync(dst.data[0], dst.stride[0], src.data.get(),
                               src.stride, w_bytes_0, src.height[0],
                               cudaMemcpyDeviceToHost, stream));

  if (src.num_comps > 1) {
    // Plane 1
    uint8_t *src_p1 = (uint8_t *)src.data.get() + src.stride * src.height[0];
    int w_bytes_1 = src.width[1] * bpp;
    CUDA_CHECK(cudaMemcpy2DAsync(dst.data[1], dst.stride[1], src_p1, src.stride,
                                 w_bytes_1, src.height[1],
                                 cudaMemcpyDeviceToHost, stream));

    if (src.num_comps > 2) {
      // Plane 2
      uint8_t *src_p2 = src_p1 + src.stride * src.height[1];
      CUDA_CHECK(cudaMemcpy2DAsync(dst.data[2], dst.stride[2], src_p2,
                                   src.stride, w_bytes_1, src.height[1],
                                   cudaMemcpyDeviceToHost, stream));
    }
  }
  return 0;
}

extern "C" {

int sr_build(const char *model_onnx, const char *plan_file) {
  if (!model_onnx || !plan_file) {
    std::cerr << "[sr_build] Error: Invalid arguments to sr_build. model_onnx "
                 "and plan_file must be provided."
              << std::endl;
    return -1;
  }

  std::cerr << "[sr_build] Starting build for model: " << model_onnx
            << std::endl;

  auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
  if (!builder) {
    std::cerr << "[sr_build] Error: Failed to create IBuilder." << std::endl;
    return -1;
  }
  uint32_t flags = 1u << static_cast<uint32_t>(
                       NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  auto network =
      std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
  if (!network) {
    std::cerr << "[sr_build] Error: Failed to create INetworkDefinition."
              << std::endl;
    return -1;
  }

  auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    std::cerr << "[sr_build] Error: Failed to create IBuilderConfig."
              << std::endl;
    return -1;
  }

  auto profile = builder->createOptimizationProfile();
  if (!profile) {
    std::cerr << "[sr_build] Error: Failed to create IOptimizationProfile."
              << std::endl;
    return -1;
  }

  auto parser = std::unique_ptr<IParser>(createParser(*network, gLogger));
  if (!parser) {
    std::cerr << "[sr_build] Error: Failed to create IParser." << std::endl;
    return -1;
  }

  if (!parser->parseFromFile(model_onnx,
                             static_cast<int>(ILogger::Severity::kWARNING))) {
    std::cerr << "[sr_build] Error: Failed to parse ONNX file: " << model_onnx
              << std::endl;
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << "  " << parser->getError(i)->desc() << std::endl;
    }
    return -1;
  }

  ITensor *srOutput = network->getOutput(0);

  /*
  IResizeLayer *resizeLayer = network->addResize(*srOutput);
  if (!resizeLayer) {
    std::cerr << "Error: Failed to add Resize layer." << std::endl;
    return -1;
  }

  float scales[] = {1.0f, 1.0f, 0.25f, 0.25f};
  resizeLayer->setScales(scales, 4);
  resizeLayer->setResizeMode(InterpolationMode::kCUBIC);

  network->unmarkOutput(*srOutput);
  network->markOutput(*resizeLayer->getOutput(0));
  */

  const char *inputName = network->getInput(0)->getName();

  profile->setDimensions(inputName, OptProfileSelector::kMIN,
                         Dims4{1, 3, TILE_SIZE, TILE_SIZE});
  profile->setDimensions(inputName, OptProfileSelector::kOPT,
                         Dims4{16, 3, TILE_SIZE, TILE_SIZE});
  profile->setDimensions(inputName, OptProfileSelector::kMAX,
                         Dims4{64, 3, TILE_SIZE, TILE_SIZE});

  config->addOptimizationProfile(profile);
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2ULL << 30);

  std::cerr << "[sr_build] Building serialized network..." << std::endl;
  auto plan = std::unique_ptr<IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    std::cerr << "[sr_build] Error: Failed to build serialized network."
              << std::endl;
    return -1;
  }

  std::ofstream ofs(plan_file, std::ios::binary);
  if (!ofs) {
    std::cerr << "[sr_build] Error: Failed to open output file: " << plan_file
              << std::endl;
    return -1;
  }
  ofs.write(reinterpret_cast<const char *>(plan->data()), plan->size());
  std::cerr << "[sr_build] Plan file successfully saved to: " << plan_file
            << std::endl;

  return 0;
}

sr_handle sr_create() {
  sr_handle c = new sr_context{};
  return c;
}

int sr_destroy(sr_handle handle) {
  if (handle) {
    delete handle;
  }
  return 0;
}

int sr_init(sr_handle handle, const sr_init_params *params) {
  if (!params) {
    std::cerr << "[sr_init] params is null" << std::endl;
    return -1;
  }

  if (!params->plan_file) {
    std::cerr << "[sr_init] plan_file is null" << std::endl;
    return -1;
  }

  if (params->prescale < 0.25f || params->prescale > 1.0f) {
    std::cerr << "[sr_init] prescale is out of range" << std::endl;
    return -1;
  }

  std::ifstream f(params->plan_file, std::ios::binary);
  if (!f.good()) {
    std::cerr << "[sr_init] Failed to open plan file." << std::endl;
    return -1;
  }

  f.seekg(0, f.end);
  size_t file_size = f.tellg();
  f.seekg(0, f.beg);

  std::vector<char> engine_data(file_size);
  f.read(engine_data.data(), file_size);

  handle->trt_runtime.reset(createInferRuntime(gLogger));
  if (!handle->trt_runtime) {
    std::cerr << "[sr_init] Failed to create TRT runtime." << std::endl;
    return -1;
  }

  handle->trt_engine.reset(handle->trt_runtime->deserializeCudaEngine(
      engine_data.data(), file_size));
  if (!handle->trt_engine) {
    std::cerr << "[sr_init] Failed to deserialize TRT engine." << std::endl;
    return -1;
  }

  handle->trt_context.reset(handle->trt_engine->createExecutionContext());
  if (!handle->trt_context) {
    std::cerr << "[sr_init] Failed to create TRT execution context."
              << std::endl;
    return -1;
  }

  cudaStream_t cuStream{};
  CUDA_CHECK(cudaStreamCreate(&cuStream));
  handle->cuda_stream.reset(cuStream);

  handle->input_name = handle->trt_engine->getIOTensorName(0);
  handle->output_name = handle->trt_engine->getIOTensorName(1);

  handle->ipar = *params;

  auto &ipar = handle->ipar;
  ipar.input_color_fullrange = ipar.input_color_fullrange ? 1 : 0;
  ipar.output_color_fullrange = ipar.output_color_fullrange ? 1 : 0;
  ipar.overlap_pixels = ipar.overlap_pixels < 0    ? 0
                        : ipar.overlap_pixels > 32 ? 32
                                                   : ipar.overlap_pixels;

  handle->model_input_width =
      static_cast<int>((ipar.input_width + 0.5) * ipar.prescale);
  handle->model_input_height =
      static_cast<int>((ipar.input_height + 0.5) * ipar.prescale);
  handle->model_output_width = handle->model_input_width * 4;
  handle->model_output_height = handle->model_input_height * 4;

  // Tile Logic (Based on Model Input)
  int tile_size = TILE_SIZE - 2 * ipar.overlap_pixels;
  handle->num_tiles_in_x =
      (handle->model_input_width + tile_size - 1) / tile_size;
  handle->num_tiles_in_y =
      (handle->model_input_height + tile_size - 1) / tile_size;

  const int batch_size = handle->num_tiles_in_x;
  if (batch_size > 64) {
    std::cerr << "[sr_init] Batch size exceeds MAX limit (64)" << std::endl;
    return -1;
  }

  Dims4 input_dims{batch_size, 3, TILE_SIZE, TILE_SIZE};
  handle->trt_context->setInputShape(handle->input_name.c_str(), input_dims);

  auto out_shape =
      handle->trt_context->getTensorShape(handle->output_name.c_str());
  if (out_shape.nbDims != 4 || out_shape.d[1] != 3 ||
      out_shape.d[2] != TILE_SIZE * 4 || out_shape.d[3] != TILE_SIZE * 4) {
    std::cerr << "[sr_init] Unexpected output tensor shape." << std::endl;
    return -1;
  }

  // 2. Create frame buffers
  handle->need_pre_resize = handle->model_input_width != ipar.input_width ||
                            handle->model_input_height != ipar.input_height;

  handle->need_post_resize = handle->model_output_width != ipar.output_width ||
                             handle->model_output_height != ipar.output_height;

  create_frame_buffer(handle->src_original, ipar.input_width, ipar.input_height,
                      ipar.input_format);

  create_frame_buffer(handle->dst_final, ipar.output_width, ipar.output_height,
                      ipar.output_format);

  if (handle->need_pre_resize) {
    create_frame_buffer(handle->input_frame, handle->model_input_width,
                        handle->model_input_height, ipar.input_format);
  }

  if (handle->need_post_resize) {
    create_frame_buffer(handle->output_frame, handle->model_output_width,
                        handle->model_output_height, ipar.output_format);
  }

  // 3. Tensors (Input is same as before, Output might be x4 larger now)
  void *dev_ptr = nullptr;
  handle->input_tensor.total_size =
      batch_size * 3 * TILE_SIZE * TILE_SIZE * sizeof(half);
  CUDA_CHECK(cudaMalloc(&dev_ptr, handle->input_tensor.total_size));
  handle->input_tensor.data.reset(dev_ptr);

  dev_ptr = nullptr;
  handle->output_tensor.total_size =
      batch_size * (3 * 4 * TILE_SIZE * 4 * TILE_SIZE * sizeof(half));
  CUDA_CHECK(cudaMalloc(&dev_ptr, handle->output_tensor.total_size));
  handle->output_tensor.data.reset(dev_ptr);

  // tensor binding
  handle->trt_context->setTensorAddress(handle->input_name.c_str(),
                                        handle->input_tensor.data.get());
  handle->trt_context->setTensorAddress(handle->output_name.c_str(),
                                        handle->output_tensor.data.get());

  return 0;
}

int sr_process(sr_handle handle, const sr_frame *src_frame,
               sr_frame *dst_frame) {
  // 1. Input Handling
  if (!handle->src_original.data) {
    std::cerr << "[sr_process] Error: internal input buffer (src_original) not "
                 "allocated."
              << std::endl;
    return -1;
  }

  if (copy_frame_to_gpu(*src_frame, handle->src_original,
                        handle->cuda_stream.get()) != 0) {
    return -1;
  }

  // Determine Source for Preprocess
  frame_buffer *proc_src = &handle->src_original;

  if (handle->need_pre_resize) {
    ResizeParams rparams{};
    rparams.src_ptr = handle->src_original.data.get();
    rparams.dst_ptr = handle->input_frame.data.get();

    rparams.src_pitch = handle->src_original.stride;
    rparams.dst_pitch = handle->input_frame.stride;

    rparams.num_comps = handle->src_original.num_comps;
    rparams.bitdepth = handle->src_original.bitdepth;

    for (int i = 0; i < 2; ++i) {
      rparams.src_width[i] = handle->src_original.width[i];
      rparams.src_height[i] = handle->src_original.height[i];
      rparams.dst_width[i] = handle->input_frame.width[i];
      rparams.dst_height[i] = handle->input_frame.height[i];
    }

    resize_yuv(rparams, handle->cuda_stream.get());
    proc_src = &handle->input_frame;
  }

  // 2. Preprocess: proc_src -> input_tensor
  PreprocessParams par1{};
  par1.d_dstTensor = (half *)handle->input_tensor.data.get();
  par1.d_src = (const void *)proc_src->data.get();
  par1.src_pitch = proc_src->stride;

  for (int i = 0; i < 2; ++i) {
    par1.src_width[i] = proc_src->width[i];
    par1.src_height[i] = proc_src->height[i];
  }

  par1.src_bitdepth = proc_src->bitdepth;
  par1.src_num_comps = proc_src->num_comps;

  par1.overlap_pixels = handle->ipar.overlap_pixels;
  par1.video_full_range_flag = (handle->ipar.input_color_fullrange == 1);

  // Tiling loop
  const int tile_size = TILE_SIZE - 2 * handle->ipar.overlap_pixels;

  for (int i = 0; i < handle->num_tiles_in_y; ++i) {
    par1.src_start_y = i * tile_size;
    preprocess(par1, handle->cuda_stream.get());

    handle->trt_context->enqueueV3(handle->cuda_stream.get());

    // Postprocess parameters
    PostprocessParams par2{};
    par2.d_srcTensor = (half *)handle->output_tensor.data.get();

    // Determine Destination for Postprocess
    frame_buffer *proc_dst =
        handle->need_post_resize ? &handle->output_frame : &handle->dst_final;

    par2.d_dst = (void *)proc_dst->data.get();
    par2.dst_pitch = proc_dst->stride;

    for (int k = 0; k < 2; ++k) {
      par2.dst_width[k] = proc_dst->width[k];
      par2.dst_height[k] = proc_dst->height[k];
    }

    par2.dst_bitdepth = proc_dst->bitdepth;
    par2.dst_num_comps = proc_dst->num_comps;
    par2.overlap_pixels = handle->ipar.overlap_pixels * 4; // x4 scale
    par2.video_full_range_flag = (handle->ipar.output_color_fullrange == 1);

    par2.dst_start_y = i * tile_size * 4;
    postprocess(par2, handle->cuda_stream.get());
  }

  // 4. Output Handling (Resize + Download)
  if (handle->need_post_resize) {
    ResizeParams rparams{};
    rparams.src_ptr = handle->output_frame.data.get();
    rparams.dst_ptr = handle->dst_final.data.get();

    rparams.src_pitch = handle->output_frame.stride;
    rparams.dst_pitch = handle->dst_final.stride;

    rparams.num_comps = handle->dst_final.num_comps;
    rparams.bitdepth = handle->dst_final.bitdepth;

    for (int i = 0; i < 2; ++i) {
      rparams.src_width[i] = handle->output_frame.width[i];
      rparams.src_height[i] = handle->output_frame.height[i];
      rparams.dst_width[i] = handle->dst_final.width[i];
      rparams.dst_height[i] = handle->dst_final.height[i];
    }

    resize_yuv(rparams, handle->cuda_stream.get());
  }

  if (copy_frame_to_host(handle->dst_final, *dst_frame,
                         handle->cuda_stream.get()) != 0) {
    return -1;
  }

  CUDA_CHECK(cudaStreamSynchronize(handle->cuda_stream.get()));
  return 0;
}
}