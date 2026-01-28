#include "real-esrgan.h"
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

static size_t get_frame_size(int w, int h, int bitdepth, int chfmt) {
    int bytes_per_pixel = (bitdepth > 8) ? 2 : 1;
    size_t luma = (size_t)w * h;
    size_t chroma = 0;
    switch(chfmt) {
        case 420: chroma = luma / 2; break;
        case 422: chroma = luma; break;
        case 444: chroma = luma * 2; break;
        case 400: chroma = 0; break;
    }
    return (luma + chroma) * bytes_per_pixel;
}

struct real_esrgan_context {
  std::unique_ptr<IRuntime> trt_runtime;
  std::unique_ptr<ICudaEngine> trt_engine;
  std::unique_ptr<IExecutionContext> trt_context;
  std::unique_ptr<std::remove_pointer<cudaStream_t>::type, StreamDeleter>
      cuda_stream;
  
  // Buffers
  std::unique_ptr<void, CudaDeleter> input_yuvframe; // Model Input (Pre-processed if prescale < 1)
  std::unique_ptr<void, CudaDeleter> output_yuvframe; // Model Output (Pre-resize)
  std::unique_ptr<void, CudaDeleter> input_tensor;
  std::unique_ptr<void, CudaDeleter> output_tensor;

  // Extra buffers for Resizing
  std::unique_ptr<void, CudaDeleter> src_yuv_original; // Original Input if prescale < 1
  std::unique_ptr<void, CudaDeleter> dst_yuv_final;    // Final Output if size != Model Output

  std::string input_name;
  std::string output_name;

  // Params
  int input_width, input_height, input_bitdepth, input_chfmt;
  int output_width, output_height, output_bitdepth, output_chfmt;
  
  // Model Dimensions (Effective)
  int model_input_width, model_input_height; 
  int model_output_width, model_output_height;

  float prescale;
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
  // Use raw output (x4)
  // already marked as output by default parsing?
  // parser->parseFromFile usually marks outputs.
  // If we don't unmark, it stays marked.
  // The original code did: ITensor *srOutput = network->getOutput(0); ... unmarkOutput(*srOutput);
  // So we just need to NOT do that.
  // actually, let's check if we need to do anything.
  // If we just remove the block, srOutput remains the output.


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
    return -1;
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

  // Copy Params
  handle->input_width = params->input_width;
  handle->input_height = params->input_height;
  handle->input_bitdepth = params->input_bitdepth;
  handle->input_chfmt = params->input_chfmt;
  handle->output_width = params->output_width;
  handle->output_height = params->output_height;
  handle->output_bitdepth = params->output_bitdepth;
  handle->output_chfmt = params->output_chfmt;
  handle->prescale = (params->prescale > 0.0f) ? params->prescale : 1.0f;
  handle->overlap_pixels = params->overlap_pixels;

  // Calculate Effective Model Input Dimensions
  handle->model_input_width = static_cast<int>(handle->input_width * handle->prescale);
  handle->model_input_height = static_cast<int>(handle->input_height * handle->prescale);
  
  // Model Output Dimensions (x4)
  handle->model_output_width = handle->model_input_width * 4;
  handle->model_output_height = handle->model_input_height * 4;

  // Tile Logic (Based on Model Input)
  int tile_size = TILE_SIZE - 2 * params->overlap_pixels;
  handle->num_tiles_in_x = (handle->model_input_width + tile_size - 1) / tile_size;
  handle->num_tiles_in_y = (handle->model_input_height + tile_size - 1) / tile_size;

  const int batch_size = handle->num_tiles_in_x;
  if (batch_size > 64) {
    std::cerr << "Batch size exceeds MAX limit (64)" << std::endl;
    return -1;
  }

  Dims4 input_dims{batch_size, 3, TILE_SIZE, TILE_SIZE};
  handle->trt_context->setInputShape(handle->input_name.c_str(), input_dims);

  auto out_shape = handle->trt_context->getTensorShape(handle->output_name.c_str());
  if(out_shape.nbDims != 4 || out_shape.d[1] != 3 || out_shape.d[2] != TILE_SIZE * 4 || out_shape.d[3] != TILE_SIZE * 4){
     // Note: Output tile size is typically Input Tile * 4 for this model
     // But wait, the TRT engine output shape depends on the model.
     // If the model is 4x SR, output should be 4x input. 
     // Input is (B, 3, TILE_SIZE, TILE_SIZE).
     // Output should be (B, 3, TILE_SIZE*4, TILE_SIZE*4).
     // My previous read of real-esrgan.cpp checked for TILE_SIZE on output.
     // "if(out_shape.nbDims != 4 || out_shape.d[1] != 3 || out_shape.d[2] != TILE_SIZE || out_shape.d[3] != TILE_SIZE)"
     // That check assumes output size == input size (because of the Resize layer 0.25x).
     // Since I REMOVED the resize layer, the output should now be x4.
     // I should update this check.
    std::cerr << "Unexpected output tensor shape." << std::endl;
    return -1;
  }

  // Memory Allocation Helper
  auto get_frame_size = [](int w, int h, int bitdepth, int chfmt) -> size_t {
      int bytes_per_pixel = (bitdepth > 8) ? 2 : 1;
      size_t luma = (size_t)w * h;
      size_t chroma = 0;
      switch(chfmt) {
          case 420: chroma = luma / 2; break; // U=L/4, V=L/4
          case 422: chroma = luma; break;     // U=L/2, V=L/2
          case 444: chroma = luma * 2; break; // U=L, V=L
          case 400: chroma = 0; break;
      }
      return (luma + chroma) * bytes_per_pixel;
  };

  // 1. Model Input YUV (Pre-processed)
  // This is what proper preprocess kernel expects.
  // It should match what the kernel expects.
  // Actually, if we use a resize kernel, it resizes `src_original` to `input_yuvframe`.
  // `input_yuvframe` is then consumed by `preprocess` to make Tensor.
  // `preprocess` typically takes YUV (usually 8-bit or same bitdepth).
  // Let's assume the resize kernel produces the same format/bitdepth as input, just resized.
  // So `input_yuvframe` has `model_input_width/height` and `input_bitdepth`.
  size_t model_input_size = get_frame_size(handle->model_input_width, handle->model_input_height, handle->input_bitdepth, handle->input_chfmt);
  void *dev_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_ptr, model_input_size));
  handle->input_yuvframe.reset(dev_ptr);

  // 2. Model Output YUV
  // `postprocess` converts Tensor to this.
  // Tensor is FP16. `postprocess` output should be `output_bitdepth`?
  // Or should it be an intermediate `input_bitdepth`?
  // Let's assume `postprocess` converts to `output_bitdepth`.
  // Size = `model_output_width/height` and `output_bitdepth`.
  size_t model_output_size = get_frame_size(handle->model_output_width, handle->model_output_height, handle->output_bitdepth, handle->output_chfmt);
  dev_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_ptr, model_output_size));
  handle->output_yuvframe.reset(dev_ptr);

  // 3. Tensors (Input is same as before, Output might be x4 larger now)
  size_t input_tensor_size = batch_size * 3 * TILE_SIZE * TILE_SIZE * sizeof(half);
  // Output element size depends on model x4
  size_t output_tile_size = TILE_SIZE * 4;
  size_t output_tensor_size = batch_size * 3 * output_tile_size * output_tile_size * sizeof(half);

  dev_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_ptr, input_tensor_size));
  handle->input_tensor.reset(dev_ptr);

  dev_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_ptr, output_tensor_size));
  handle->output_tensor.reset(dev_ptr);

  // 4. Extra Buffers
  if (std::abs(handle->prescale - 1.0f) > 1e-6) {
      size_t src_size = get_frame_size(handle->input_width, handle->input_height, handle->input_bitdepth, handle->input_chfmt);
      dev_ptr = nullptr;
      CUDA_CHECK(cudaMalloc(&dev_ptr, src_size));
      handle->src_yuv_original.reset(dev_ptr);
  }

  // If output dimensions differ from model output, alloc final buffer
  if (handle->model_output_width != handle->output_width || handle->model_output_height != handle->output_height) {
      // Actually, we might need this if we want to support strict output resolution.
      // Assuming we do.
      size_t final_size = get_frame_size(handle->output_width, handle->output_height, handle->output_bitdepth, handle->output_chfmt);
      dev_ptr = nullptr;
      CUDA_CHECK(cudaMalloc(&dev_ptr, final_size));
      handle->dst_yuv_final.reset(dev_ptr);
  }

  // tensor binding
  handle->trt_context->setTensorAddress(handle->input_name.c_str(),
                                        handle->input_tensor.get());
  handle->trt_context->setTensorAddress(handle->output_name.c_str(),
                                        handle->output_tensor.get());

  return 0;
}

int real_esrgan_process(real_esrgan_handle handle, const uint8_t *srcYuv,
                        uint8_t *dstYuv) {
  // Input Pointers
  const void* src_ptr = srcYuv;
  void* d_model_input = handle->input_yuvframe.get();

  // 1. Input Handling (Upload + Resize if needed)
  if (handle->src_yuv_original) { // Prescale active
      size_t src_size = get_frame_size(handle->input_width, handle->input_height, handle->input_bitdepth, handle->input_chfmt);
      CUDA_CHECK(cudaMemcpyAsync(handle->src_yuv_original.get(), src_ptr, src_size,
                                 cudaMemcpyHostToDevice, handle->cuda_stream.get()));
      
      // Perform Resize
      ResizeParams rparams{};
      rparams.src = handle->src_yuv_original.get();
      rparams.dst = d_model_input;
      rparams.src_width = handle->input_width;
      rparams.src_height = handle->input_height;
      rparams.dst_width = handle->model_input_width;
      rparams.dst_height = handle->model_input_height;
      rparams.bitdepth = handle->input_bitdepth;
      rparams.chfmt = handle->input_chfmt;
      resize_yuv(rparams, handle->cuda_stream.get());

  } else {
      size_t src_size = get_frame_size(handle->input_width, handle->input_height, handle->input_bitdepth, handle->input_chfmt);
      CUDA_CHECK(cudaMemcpyAsync(d_model_input, src_ptr, src_size,
                                 cudaMemcpyHostToDevice, handle->cuda_stream.get()));
  }

  // 2. Preprocess
  const int luma_size = handle->model_input_width * handle->model_input_height;
  
  // Calculates offsets
  int bytes_per_pixel = (handle->input_bitdepth > 8) ? 2 : 1;
  uint8_t* base_in = (uint8_t*)d_model_input;
  uint8_t* d_srcY = base_in;
  uint8_t* d_srcU = nullptr;
  uint8_t* d_srcV = nullptr;
  
  size_t luma_bytes = (size_t)luma_size * bytes_per_pixel;
  if(handle->input_chfmt != 400) {
      d_srcU = base_in + luma_bytes;
      size_t u_size = 0;
      switch(handle->input_chfmt) {
          case 420: u_size = luma_bytes / 4; break;
          case 422: u_size = luma_bytes / 2; break;
          case 444: u_size = luma_bytes; break;
      }
      d_srcV = d_srcU + u_size;
  }

  PreprocessParams par1{};
  par1.d_dstTensor = (half *)handle->input_tensor.get();
  par1.d_srcY = d_srcY;
  par1.d_srcU = d_srcU;
  par1.d_srcV = d_srcV;
  par1.src_width = handle->model_input_width;
  par1.src_height = handle->model_input_height;
  par1.overlap_pixels = handle->overlap_pixels;
  par1.video_full_range_flag = true; 
  par1.bitdepth = handle->input_bitdepth;
  par1.chfmt = handle->input_chfmt;
  
  // Tiling loop
  const int tile_size = TILE_SIZE - 2 * handle->overlap_pixels;
  // NOTE: Postprocess output tile is x4 larger.
  // We need postprocess to handle the scaling factor.
  // We will pass the 'scale' or derive it. For now, we manually adjust params for postprocess.
  
  for (int i = 0; i < handle->num_tiles_in_y; ++i) {
    par1.src_start_y = i * tile_size;
    preprocess(par1, handle->cuda_stream.get());

    handle->trt_context->enqueueV3(handle->cuda_stream.get());

    // Postprocess loop
    PostprocessParams par2{};
    par2.d_srcTensor = (half *)handle->output_tensor.get();
    
    // Calculate Output Pointers (Model Output)
    void* d_model_output = handle->output_yuvframe.get();
    uint8_t* base_out = (uint8_t*)d_model_output;
    int out_bpp = (handle->output_bitdepth > 8) ? 2 : 1;
    size_t out_luma_bytes = (size_t)handle->model_output_width * handle->model_output_height * out_bpp;

    par2.d_dstY = base_out;
    if(handle->output_chfmt != 400) {
        par2.d_dstU = base_out + out_luma_bytes;
        size_t u_size = 0;
        switch(handle->output_chfmt) {
            case 420: u_size = out_luma_bytes / 4; break;
            case 422: u_size = out_luma_bytes / 2; break;
            case 444: u_size = out_luma_bytes; break;
        }
        par2.d_dstV = par2.d_dstU + u_size;
    } else {
        par2.d_dstU = nullptr;
        par2.d_dstV = nullptr;
    }

    par2.dst_width = handle->model_output_width;
    par2.dst_height = handle->model_output_height;
    par2.overlap_pixels = handle->overlap_pixels * 4; // x4 scale
    par2.scale_factor = 4; // Passing scale factor explicitly to help postprocess kernel
    
    par2.video_full_range_flag = true;
    par2.bitdepth = handle->output_bitdepth;
    par2.chfmt = handle->output_chfmt;
    
    par2.dst_start_y = i * tile_size * 4; 
    postprocess(par2, handle->cuda_stream.get());
  }
  
  // 3. Output Handling (Resize if needed + Download)
  void* d_final = handle->output_yuvframe.get();
  size_t final_size = get_frame_size(handle->model_output_width, handle->model_output_height, handle->output_bitdepth, handle->output_chfmt);

  if (handle->dst_yuv_final) {
      d_final = handle->dst_yuv_final.get();
      // Resize
      ResizeParams rparams{};
      rparams.src = handle->output_yuvframe.get();
      rparams.dst = d_final;
      rparams.src_width = handle->model_output_width;
      rparams.src_height = handle->model_output_height;
      rparams.dst_width = handle->output_width;
      rparams.dst_height = handle->output_height;
      rparams.bitdepth = handle->output_bitdepth;
      rparams.chfmt = handle->output_chfmt;
      resize_yuv(rparams, handle->cuda_stream.get());
      
      final_size = get_frame_size(handle->output_width, handle->output_height, handle->output_bitdepth, handle->output_chfmt);
  }
  
  CUDA_CHECK(cudaMemcpyAsync(dstYuv, d_final, final_size,
                             cudaMemcpyDeviceToHost,
                             handle->cuda_stream.get()));

  CUDA_CHECK(cudaStreamSynchronize(handle->cuda_stream.get()));
  return 0;
}
}