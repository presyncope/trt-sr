#include "real-esrgan.h"
#include "common.h"
#include "postprocess.h"
#include "preprocess.h"
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

struct real_esrgan_context {
  std::unique_ptr<IRuntime> trt_runtime;
  std::unique_ptr<ICudaEngine> trt_engine;
  std::unique_ptr<IExecutionContext> trt_context;
  std::unique_ptr<std::remove_pointer<cudaStream_t>::type, StreamDeleter>
      cuda_stream;
  std::unique_ptr<void, CudaDeleter> input_yuvframe;
  std::unique_ptr<void, CudaDeleter> output_yuvframe;
  std::unique_ptr<void, CudaDeleter> input_tensor;
  std::unique_ptr<void, CudaDeleter> output_tensor;
  std::string input_name;
  std::string output_name;

  int pic_width;
  int pic_height;
  int overlap_pixels;
  int num_tiles_in_x;
  int num_tiles_in_y;
};

extern "C" {

int real_esrgan_build(const char *model_onnx, const char *plan_file) {
  if (!model_onnx || !plan_file) {
    std::cerr << "Error: Invalid arguments to real_esrgan_build. model_onnx "
                 "and plan_file must be provided."
              << std::endl;
    return -1;
  }

  std::cerr << "Starting build for model: " << model_onnx << std::endl;

  auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
  if (!builder) {
    std::cerr << "Error: Failed to create IBuilder." << std::endl;
    return -1;
  }
  uint32_t flags = 1u << static_cast<uint32_t>(
                       NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  auto network =
      std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
  if (!network) {
    std::cerr << "Error: Failed to create INetworkDefinition." << std::endl;
    return -1;
  }

  auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    std::cerr << "Error: Failed to create IBuilderConfig." << std::endl;
    return -1;
  }

  auto profile = builder->createOptimizationProfile();
  if (!profile) {
    std::cerr << "Error: Failed to create IOptimizationProfile." << std::endl;
    return -1;
  }

  auto parser = std::unique_ptr<IParser>(createParser(*network, gLogger));
  if (!parser) {
    std::cerr << "Error: Failed to create IParser." << std::endl;
    return -1;
  }

  if (!parser->parseFromFile(model_onnx,
                             static_cast<int>(ILogger::Severity::kWARNING))) {
    std::cerr << "Error: Failed to parse ONNX file: " << model_onnx
              << std::endl;
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << "  " << parser->getError(i)->desc() << std::endl;
    }
    return -1;
  }

  ITensor *srOutput = network->getOutput(0);

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

  const char *inputName = network->getInput(0)->getName();

  profile->setDimensions(inputName, OptProfileSelector::kMIN,
                         Dims4{1, 3, TILE_SIZE, TILE_SIZE});
  profile->setDimensions(inputName, OptProfileSelector::kOPT,
                         Dims4{16, 3, TILE_SIZE, TILE_SIZE});
  profile->setDimensions(inputName, OptProfileSelector::kMAX,
                         Dims4{64, 3, TILE_SIZE, TILE_SIZE});

  config->addOptimizationProfile(profile);
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2ULL << 30);

  std::cerr << "Building serialized network..." << std::endl;
  auto plan = std::unique_ptr<IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    std::cerr << "Error: Failed to build serialized network." << std::endl;
    return -1;
  }

  std::ofstream ofs(plan_file, std::ios::binary);
  if (!ofs) {
    std::cerr << "Error: Failed to open output file: " << plan_file
              << std::endl;
    return -1;
  }
  ofs.write(reinterpret_cast<const char *>(plan->data()), plan->size());
  std::cerr << "Plan file successfully saved to: " << plan_file << std::endl;

  return 0;
}

real_esrgan_handle real_esrgan_create() {
  real_esrgan_handle c = new real_esrgan_context{};
  return c;
}

int real_esrgan_destroy(real_esrgan_handle handle) {
  if (handle) {
    delete handle;
  }
  return 0;
}

int real_esrgan_init(real_esrgan_handle handle,
                     const real_esrgan_init_params *params) {
  if (!params) {
    std::cerr << "params is null" << std::endl;
    return -1;
  }

  if (!params->plan_file) {
    std::cerr << "plan_file is none" << std::endl;
  }

  std::ifstream f(params->plan_file, std::ios::binary);
  if (!f.good()) {
    std::cerr << "Failed to open plan file." << std::endl;
    return -1;
  }

  f.seekg(0, f.end);
  size_t file_size = f.tellg();
  f.seekg(0, f.beg);

  std::vector<char> engine_data(file_size);
  f.read(engine_data.data(), file_size);

  handle->trt_runtime.reset(createInferRuntime(gLogger));
  if (!handle->trt_runtime) {
    std::cerr << "Failed to create TRT runtime." << std::endl;
    return -1;
  }

  handle->trt_engine.reset(handle->trt_runtime->deserializeCudaEngine(
      engine_data.data(), file_size));
  if (!handle->trt_engine) {
    std::cerr << "Failed to deserialize TRT engine." << std::endl;
    return -1;
  }

  handle->trt_context.reset(handle->trt_engine->createExecutionContext());
  if (!handle->trt_context) {
    std::cerr << "Failed to create TRT execution context." << std::endl;
    return -1;
  }

  cudaStream_t cuStream{};
  cudaStreamCreate(&cuStream);
  handle->cuda_stream.reset(cuStream);

  handle->input_name = handle->trt_engine->getIOTensorName(0);
  handle->output_name = handle->trt_engine->getIOTensorName(1);

  handle->pic_width = params->width;
  handle->pic_height = params->height;
  handle->overlap_pixels = params->overlap_pixels;

  int tile_size = TILE_SIZE - 2 * params->overlap_pixels;
  handle->num_tiles_in_x = (params->width + tile_size - 1) / tile_size;
  handle->num_tiles_in_y = (params->height + tile_size - 1) / tile_size;

  const int batch_size = handle->num_tiles_in_x;
  if (batch_size > 64) {
    std::cerr << "Batch size exceeds MAX limit (64)" << std::endl;
    return -1;
  }

  Dims4 input_dims{batch_size, 3, TILE_SIZE, TILE_SIZE};
  handle->trt_context->setInputShape(handle->input_name.c_str(), input_dims);

  auto out_shape = handle->trt_context->getTensorShape(handle->output_name.c_str());
  if(out_shape.nbDims != 4 || out_shape.d[1] != 3 || out_shape.d[2] != TILE_SIZE || out_shape.d[3] != TILE_SIZE){
    std::cerr << "Unexpected output tensor shape." << std::endl;
    return -1;
  }

  // memory allocation
  size_t yuv_size = params->width * params->height * 3 / 2 * sizeof(uint8_t);
  size_t tensor_size = batch_size * 3 * TILE_SIZE * TILE_SIZE * sizeof(half);

  void *dev_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_ptr, yuv_size));
  handle->input_yuvframe.reset(dev_ptr);

  dev_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_ptr, yuv_size));
  handle->output_yuvframe.reset(dev_ptr);

  dev_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_ptr, tensor_size));
  handle->input_tensor.reset(dev_ptr);

  dev_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_ptr, tensor_size));
  handle->output_tensor.reset(dev_ptr);

  // tensor binding
  handle->trt_context->setTensorAddress(handle->input_name.c_str(),
                                        handle->input_tensor.get());
  handle->trt_context->setTensorAddress(handle->output_name.c_str(),
                                        handle->output_tensor.get());

  return 0;
}

int real_esrgan_process(real_esrgan_handle handle, const uint8_t *srcYuv,
                        uint8_t *dstYuv) {
  const int batch_size = handle->num_tiles_in_x;
  const int luma_size = handle->pic_width * handle->pic_height;
  const int chroma_size = luma_size / 4;
  const int yuv_size = luma_size + 2 * chroma_size;
  const int tile_size = TILE_SIZE - 2 * handle->overlap_pixels;

  CUDA_CHECK(cudaMemcpyAsync(handle->input_yuvframe.get(), srcYuv, yuv_size,
                             cudaMemcpyHostToDevice,
                             handle->cuda_stream.get()));

  PreprocessParams par1{};
  par1.d_dstTensor = (half *)handle->input_tensor.get();
  par1.d_srcY = (uint8_t *)handle->input_yuvframe.get();
  par1.d_srcU = par1.d_srcY + luma_size;
  par1.d_srcV = par1.d_srcU + chroma_size;
  par1.src_width = handle->pic_width;
  par1.src_height = handle->pic_height;
  par1.overlap_pixels = handle->overlap_pixels;
  par1.video_full_range_flag = true;

  PostprocessParams par2{};
  par2.d_srcTensor = (half *)handle->output_tensor.get();
  par2.d_dstY = (uint8_t *)handle->output_yuvframe.get();
  par2.d_dstU = par2.d_dstY + luma_size;
  par2.d_dstV = par2.d_dstU + chroma_size;
  par2.dst_width = handle->pic_width;
  par2.dst_height = handle->pic_height;
  par2.overlap_pixels = handle->overlap_pixels;
  par2.video_full_range_flag = true;

  for (int i = 0; i < handle->num_tiles_in_y; ++i) {
    par1.src_start_y = i * tile_size;
    preprocess(par1, handle->cuda_stream.get());

    handle->trt_context->enqueueV3(handle->cuda_stream.get());

    par2.dst_start_y = i * tile_size;
    postprocess(par2, handle->cuda_stream.get());
  }

  CUDA_CHECK(cudaMemcpyAsync(dstYuv, handle->output_yuvframe.get(), yuv_size,
                             cudaMemcpyDeviceToHost,
                             handle->cuda_stream.get()));

  CUDA_CHECK(cudaStreamSynchronize(handle->cuda_stream.get()));

  return 0;
}
}