#include "superres.h"
#include "common.h"
#include "postprocess.h"
#include "preprocess.h"
#include "resize.h"
#include <NvInfer.h>
#if ENABLE_NVONNXPARSER
#include <NvOnnxParser.h>
#endif
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#ifndef NDEBUG
#define DEBUG_LOG(x) std::cerr << x << std::endl
#else
#define DEBUG_LOG(x) ((void)0)
#endif

using namespace nvinfer1;
#if ENABLE_NVONNXPARSER
using namespace nvonnxparser;
#endif

#define CUDA_CHECK(callstr)                                                    \
  {                                                                            \
    cudaError_t ecode = callstr;                                               \
    if (ecode != cudaSuccess) {                                                \
      std::cerr << "CUDA error " << ecode << " at " << __FILE__ << ":"         \
                << __LINE__ << std::endl;                                      \
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

struct sr_context {
  std::unique_ptr<IRuntime> trt_runtime;
  std::unique_ptr<ICudaEngine> trt_engine;
  std::unique_ptr<IExecutionContext> trt_context;
  std::unique_ptr<std::remove_pointer<cudaStream_t>::type, StreamDeleter>
      cuda_stream;

  // Buffers (GPU)
  FrameBuffer src_yuvframe;       // Input YUV frame
  FrameBuffer dst_yuvframe;       // Output YUV frame
  FrameBuffer prescaled_yuvframe; // After Pre-Resize YUV frame (optional)
  TensorBuffer input_tensor;  // Input RGB tiled tensor (shape: [B,3,128,128])
  TensorBuffer output_tensor; // Output RGB tiled tensor (shape: [B,3,512,512])

  std::string input_name;
  std::string output_name;

  sr_init_params ipar;

  bool need_pre_resize;
  int num_tiles_in_x;
  int num_tiles_in_y;
  int model_input_width;
  int model_input_height;
  int model_output_width;
  int model_output_height;
};

static int create_frame_buffer(FrameBuffer &buf, int w, int h,
                               sr_pixel_format fmt) {
  buf.width = w;
  buf.height = h;
  buf.format = fmt;
  buf.bitdepth = get_bit_depth(fmt);

  bool semi_planar = is_semi_planar(fmt);
  buf.num_planes = semi_planar ? 2 : 3;

  int bpp_shift = (buf.bitdepth > 8) ? 1 : 0;
  int bytes_per_pixel = 1 << bpp_shift;

  // 1. Calculate Dimensions
  int csy = get_chroma_shift_y(fmt);
  int luma_w_bytes = w * bytes_per_pixel;
  int luma_h = h;

  int chroma_w_bytes;
  int chroma_h = h >> csy;

  if (semi_planar) {
    chroma_w_bytes = luma_w_bytes;
  } else {
    chroma_w_bytes = luma_w_bytes / 2;
  }

  // 2. Allocate Single Surface (Pitch Linear)
  // Note: We use the widest requirement (Luma) for the pitch.
  // For Planar formats (I420), this wastes stride on Chroma planes,
  // but allows a single allocation.
  size_t pitch = 0;
  void *dev_ptr = nullptr;

  int total_rows = luma_h + (chroma_h * (buf.num_planes - 1));

  // Allocate based on Luma width to ensure alignment
  CUDA_CHECK(cudaMallocPitch(&dev_ptr, &pitch, luma_w_bytes, total_rows));

  buf.raw_data.reset(dev_ptr);
  uint8_t *base_ptr = static_cast<uint8_t *>(dev_ptr);

  // 3. Setup Plane Pointers
  // Plane 0: Y
  buf.planes[0] = {base_ptr, luma_w_bytes, luma_h, pitch};

  size_t current_offset = pitch * luma_h;

  // Plane 1: U or UV
  buf.planes[1] = {base_ptr + current_offset, chroma_w_bytes, chroma_h, pitch};
  current_offset += pitch * chroma_h;

  // Plane 2: V (Only for fully planar)
  if (!semi_planar) {
    buf.planes[2] = {base_ptr + current_offset, chroma_w_bytes, chroma_h,
                     pitch};
  }

  return 0;
}

static int copy_frame_to_gpu(const sr_frame &src, const FrameBuffer &dst,
                             cudaStream_t stream) {
  if (!dst.raw_data)
    return -1;

  for (int i = 0; i < dst.num_planes; ++i) {

    const void *src_ptr = src.data[i];
    if (!src_ptr) {
      std::cerr << "[superres] Error: Source plane " << i << " is null."
                << std::endl;
      return -1;
    }

    void *dst_ptr = dst.planes[i].data;
    size_t dst_pitch = dst.planes[i].stride;
    size_t src_pitch = src.stride[i];
    size_t width_bytes = dst.planes[i].width_bytes;
    size_t height = dst.planes[i].height;

    if (src_pitch < width_bytes) {
      std::cerr << "[superres] Error: Source stride (" << src_pitch
                << ") is smaller than source width bytes (" << width_bytes
                << ") for plane " << i << std::endl;
      return -2; // Buffer overflow 방지
    }

    CUDA_CHECK(
        cudaMemcpy2DAsync(dst_ptr,     // Destination (Device)
                          dst_pitch,   // Dest Pitch
                          src_ptr,     // Source (Host)
                          src_pitch,   // Source Pitch
                          width_bytes, // Width in Bytes (Valid data per row)
                          height,      // Height (Number of rows)
                          cudaMemcpyHostToDevice, stream));
  }

  return 0;
}

static int copy_frame_to_host(const FrameBuffer &src, sr_frame &dst,
                              cudaStream_t stream) {
  if (!src.raw_data)
    return -1;

  for (int i = 0; i < src.num_planes; ++i) {
    void *dst_ptr = dst.data[i];
    if (!dst_ptr) {
      std::cerr << "[superres] Error: Destination plane " << i << " is null."
                << std::endl;
      return -1;
    }

    void *src_ptr = src.planes[i].data;
    size_t src_pitch = src.planes[i].stride;
    size_t dst_pitch = dst.stride[i];
    size_t width_bytes = src.planes[i].width_bytes;
    size_t height = src.planes[i].height;

    if (dst_pitch < width_bytes) {
      std::cerr << "[superres] Error: Destination stride (" << dst_pitch
                << ") is smaller than source width bytes (" << width_bytes
                << ") for plane " << i << std::endl;
      return -2; // Buffer overflow 방지
    }

    CUDA_CHECK(
        cudaMemcpy2DAsync(dst_ptr,     // Destination (Host)
                          dst_pitch,   // Dest Pitch
                          src_ptr,     // Source (Device)
                          src_pitch,   // Source Pitch
                          width_bytes, // Width in Bytes (Valid data per row)
                          height,      // Height (Number of rows)
                          cudaMemcpyDeviceToHost, stream));
  }

  return 0;
}

// [Helper] TensorRT Data Type Size
static size_t get_dtype_size(DataType type) {
  switch (type) {
  case DataType::kFLOAT:
    return 4;
  case DataType::kHALF:
    return 2;
  case DataType::kINT8:
    return 1;
  default:
    return 0;
  }
}

// [Helper] Read Binary File
static std::vector<char> read_file(const char *path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.good())
    return {};
  size_t size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  f.read(buffer.data(), size);
  return buffer;
}

extern "C" {

int sr_build(const sr_build_params *params) {
#if ENABLE_NVONNXPARSER
  // 1. Validate Arguments
  if (!params || !params->model_onnx || !params->plan_file) {
    std::cerr << "[superres] Error: Invalid arguments to sr_build. model_onnx "
                 "and plan_file must be provided."
              << std::endl;
    return -1;
  }

  // Local copy for modification
  sr_build_params p = *params;

  // 2. Validate & Fix Batch Sizes (Safety: min <= opt <= max)
  if (p.min_batch_size <= 0)
    p.min_batch_size = 1;
  if (p.optimal_batch_size <= 0)
    p.optimal_batch_size = std::max(p.min_batch_size, 8);
  if (p.max_batch_size <= 0)
    p.max_batch_size = std::max(p.optimal_batch_size, 8);

  // Force constraints
  p.optimal_batch_size = std::max(p.min_batch_size, p.optimal_batch_size);
  p.max_batch_size = std::max(p.optimal_batch_size, p.max_batch_size);

  std::cout << "[superres] Building Model: " << p.model_onnx << "\n"
            << "           Batch Sizes: [" << p.min_batch_size << ", "
            << p.optimal_batch_size << ", " << p.max_batch_size << "]"
            << std::endl;

  // 3. Create Builder
  auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
  if (!builder)
    return -1;

  // 4. Create Network
  uint32_t flags = 0u;
  if (p.strongly_typed) {
    flags |= 1u << static_cast<uint32_t>(
                 NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  }

  auto network =
      std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
  if (!network)
    return -1;

  // 5. Create Config & Parser
  auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
  auto parser = std::unique_ptr<IParser>(createParser(*network, gLogger));
  if (!config || !parser)
    return -1;

  // 6. Parse ONNX
  if (!parser->parseFromFile(p.model_onnx,
                             static_cast<int>(ILogger::Severity::kWARNING))) {
    std::cerr << "[superres] Error: Failed to parse ONNX file." << std::endl;
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << "  " << parser->getError(i)->desc() << std::endl;
    }
    return -1;
  }

  // =================================================================================
  // [Constraint Enforcement]
  // Input [-1, 3, 128, 128], Output [-1, 3, 512, 512]
  // =================================================================================

  // 7. Validate & Configure Input Dimensions
  ITensor *input = network->getInput(0);
  const char *inputName = input->getName();

  constexpr int INPUT_C = 3;
  constexpr int INPUT_H = TILE_SIZE;
  constexpr int INPUT_W = TILE_SIZE;

  auto profile = builder->createOptimizationProfile();

  profile->setDimensions(inputName, OptProfileSelector::kMIN,
                         Dims4{p.min_batch_size, INPUT_C, INPUT_H, INPUT_W});
  profile->setDimensions(
      inputName, OptProfileSelector::kOPT,
      Dims4{p.optimal_batch_size, INPUT_C, INPUT_H, INPUT_W});
  profile->setDimensions(inputName, OptProfileSelector::kMAX,
                         Dims4{p.max_batch_size, INPUT_C, INPUT_H, INPUT_W});

  config->addOptimizationProfile(profile);

  // 8. Validate Output Dimensions (Sanity Check)
  ITensor *output = network->getOutput(0);
  Dims outputDims = output->getDimensions();

  constexpr int EXPECTED_OUT_C = 3;
  constexpr int EXPECTED_OUT_H = TILE_SIZE * 4;
  constexpr int EXPECTED_OUT_W = TILE_SIZE * 4;

  bool dims_match = (outputDims.nbDims == 4) &&
                    (outputDims.d[1] == EXPECTED_OUT_C) &&
                    (outputDims.d[2] == EXPECTED_OUT_H) &&
                    (outputDims.d[3] == EXPECTED_OUT_W);

  if (!dims_match) {
    std::cerr << "[superres] Error: Output dimension mismatch!" << std::endl;
    std::cerr << "  Expected: [-1, " << EXPECTED_OUT_C << ", " << EXPECTED_OUT_H
              << ", " << EXPECTED_OUT_W << "]" << std::endl;
    std::cerr << "  Actual:   [";
    for (int i = 0; i < outputDims.nbDims; ++i)
      std::cerr << outputDims.d[i] << (i == outputDims.nbDims - 1 ? "" : ", ");
    std::cerr << "]" << std::endl;
    return -1;
  } else {
    std::cout << "[superres] Output dimension verified: [-1, 3, 512, 512]"
              << std::endl;
  }

  // 9. Set Memory Limits (TensorRT 10.x Update)
  if (p.max_workspace_size == 0)
    p.max_workspace_size = 240ULL << 20; // 256 MB default

  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, p.max_workspace_size);

  if (p.max_shared_memory > 0) {
    config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY,
                               p.max_shared_memory);
  }

  if (p.version_compatible) {
    config->setFlag(BuilderFlag::kVERSION_COMPATIBLE);
  }

  if (p.exclude_lean_runtime) {
    config->setFlag(BuilderFlag::kEXCLUDE_LEAN_RUNTIME);
  }

  // 10. Build Serialized Network (Plan)
  std::cerr << "[superres] Building serialized network..." << std::endl;
  auto plan = std::unique_ptr<IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    std::cerr << "[superres] Error: Failed to build serialized network."
              << std::endl;
    return -1;
  }

  // 11. Save to Disk
  std::ofstream ofs(p.plan_file, std::ios::binary);
  if (!ofs) {
    std::cerr << "[superres] Error: Failed to open output file: " << p.plan_file
              << std::endl;
    return -1;
  }
  ofs.write(reinterpret_cast<const char *>(plan->data()), plan->size());
  std::cerr << "[superres] Success: " << p.plan_file << std::endl;

  return 0;
#else
  std::cerr << "[superres] Error: onnx-parser is not enabled." << std::endl;
  return -1;
#endif
}

sr_handle sr_create() {
  sr_context *ctx = new (std::nothrow) sr_context{};

  if (!ctx) {
    return nullptr;
  }

  return ctx;
}

int sr_destroy(sr_handle handle) {
  if (handle) {
    delete handle;
  }
  return 0;
}

int sr_init(sr_handle handle, const sr_init_params *params) {
  // 1. Validation
  if (!handle) {
    std::cerr << "[sr_init] Handle is null" << std::endl;
    return -1;
  }
  if (!params || !params->plan_file) {
    std::cerr << "[sr_init] Invalid parameters." << std::endl;
    return -1;
  }
  if (params->prescale < 0.25f || params->prescale > 1.0f) {
    std::cerr << "[sr_init] Prescale out of range (0.25 ~ 1.0)." << std::endl;
    return -1;
  }

  // 2. Load Engine (Read File & Deserialize)
  auto engine_data = read_file(params->plan_file);
  if (engine_data.empty()) {
    std::cerr << "[sr_init] Failed to read plan file: " << params->plan_file
              << std::endl;
    return -1;
  }

  DEBUG_LOG("[sr_init] Debug: Calling createInferRuntime...");
  handle->trt_runtime.reset(createInferRuntime(gLogger));
  if (!handle->trt_runtime) {
    std::cerr << "[sr_init] Error: createInferRuntime failed." << std::endl;
    return -1;
  }
  DEBUG_LOG("[sr_init] Debug: Runtime created.");

  DEBUG_LOG("[sr_init] Debug: Calling deserializeCudaEngine");
  handle->trt_engine.reset(handle->trt_runtime->deserializeCudaEngine(
      engine_data.data(), engine_data.size()));
  if (!handle->trt_engine) {
    std::cerr << "[sr_init] Error: Failed to deserialize Engine." << std::endl;
    return -1;
  }
  DEBUG_LOG("[sr_init] Debug: Deserialize Engine Success");

  handle->trt_context.reset(handle->trt_engine->createExecutionContext());
  if (!handle->trt_context) {
    std::cerr << "[sr_init] Error: Failed to create ExecutionContext."
              << std::endl;
    return -1;
  }
  DEBUG_LOG("[sr_init] Debug: ExecutionContext created.");

  // 3. Setup CUDA Stream
  DEBUG_LOG("[sr_init] Debug: Setting up CUDA stream");
  cudaStream_t stream = {};
  CUDA_CHECK(cudaStreamCreate(&stream));
  DEBUG_LOG("[sr_init] Debug: CUDA stream created successfully.");

  handle->cuda_stream.reset(stream);
  DEBUG_LOG("[sr_init] Debug: Smart pointer reset done.");

  // 4. Discover IO Tensors (Dynamic & Safe)
  DEBUG_LOG("[sr_init] Debug: Discovering IO Tensors");
  int nb_io = handle->trt_engine->getNbIOTensors();
  size_t input_elem_size = 0;
  size_t output_elem_size = 0;

  for (int i = 0; i < nb_io; ++i) {
    const char *name = handle->trt_engine->getIOTensorName(i);
    TensorIOMode mode = handle->trt_engine->getTensorIOMode(name);
    DataType type = handle->trt_engine->getTensorDataType(name);
    Dims dims = handle->trt_engine->getTensorShape(name);

    if (mode == TensorIOMode::kINPUT) {
      // Requirement: [x, 3, 128, 128]
      bool is_valid = (dims.nbDims == 4) && (dims.d[1] == 3) &&
                      (dims.d[2] == TILE_SIZE) && (dims.d[3] == TILE_SIZE);
      if (!is_valid) {
        std::cerr << "[sr_init] Error: Input tensor shape mismatch!"
                  << std::endl;
        std::cerr << "  Expected: [x, 3, " << TILE_SIZE << ", " << TILE_SIZE
                  << "]" << std::endl;
        std::cerr << "  Actual:   [";
        for (int d = 0; d < dims.nbDims; ++d)
          std::cerr << dims.d[d] << (d < dims.nbDims - 1 ? ", " : "");
        std::cerr << "]" << std::endl;
        return -1;
      }

      handle->input_name = name;
      input_elem_size = get_dtype_size(type);

      std::cerr << "[sr_init] Found Input Tensor: " << name
                << " (Shape: [x, 3, " << TILE_SIZE << ", " << TILE_SIZE << "])"
                << std::endl;
    } else if (mode == TensorIOMode::kOUTPUT) {
      // Requirement: [x, 3, 512, 512]
      bool is_valid = (dims.nbDims == 4) && (dims.d[1] == 3) &&
                      (dims.d[2] == TILE_SIZE * 4) &&
                      (dims.d[3] == TILE_SIZE * 4);
      if (!is_valid) {
        std::cerr << "[sr_init] Error: Output tensor shape mismatch!"
                  << std::endl;
        std::cerr << "  Expected: [x, 3, " << TILE_SIZE * 4 << ", "
                  << TILE_SIZE * 4 << "]" << std::endl;
        std::cerr << "  Actual:   [";
        for (int d = 0; d < dims.nbDims; ++d)
          std::cerr << dims.d[d] << (d < dims.nbDims - 1 ? ", " : "");
        std::cerr << "]" << std::endl;
        return -1;
      }
      handle->output_name = name;
      output_elem_size = get_dtype_size(type);

      std::cerr << "[sr_init] Found Output Tensor: " << name
                << " (Shape: [x, 3, " << TILE_SIZE * 4 << ", " << TILE_SIZE * 4
                << "])" << std::endl;
    }
  }

  // Safety Check: Data Type
  DEBUG_LOG("[sr_init] Debug: Checking Data Type");
  if (handle->input_name.empty() || handle->output_name.empty()) {
    std::cerr << "[sr_init] Failed to find Input/Output tensors." << std::endl;
    return -1;
  }
  if (input_elem_size != 2 || output_elem_size != 2) {
    std::cerr << "[sr_init] Unsupported Tensor Data Type." << std::endl;
    return -1;
  }

  // 5. Logic & Dimensions
  DEBUG_LOG("[sr_init] Debug: Setting up Logic & Dimensions");
  handle->ipar = *params;
  auto &ipar = handle->ipar;

  // Clamp overlap
  ipar.overlap_pixels = std::max(0, std::min(ipar.overlap_pixels, 16));
  ipar.input_color_fullrange = ipar.input_color_fullrange ? 1 : 0;
  ipar.output_color_fullrange = ipar.output_color_fullrange ? 1 : 0;

  // Calculate Model Dimensions
  handle->model_input_width =
      static_cast<int>((ipar.input_width + 0.5) * ipar.prescale);
  handle->model_input_height =
      static_cast<int>((ipar.input_height + 0.5) * ipar.prescale);

  // Assumption: SR Model is x4 upscaling
  handle->model_output_width =
      handle->model_input_width * 4; // x4 SR assumption
  handle->model_output_height = handle->model_input_height * 4;

  // Calculate Tile Grid
  int tile_size_valid = TILE_SIZE - 2 * ipar.overlap_pixels;
  handle->num_tiles_in_x =
      (handle->model_input_width + tile_size_valid - 1) / tile_size_valid;
  handle->num_tiles_in_y =
      (handle->model_input_height + tile_size_valid - 1) / tile_size_valid;

  // 6. Set Input Shape (Dynamic Batch)
  DEBUG_LOG("[sr_init] Debug: Setting up Input Shape");
  int batch_size = (ipar.concurrent_batches > 0) ? ipar.concurrent_batches : 1;
  if (batch_size > 64) {
    std::cerr << "[sr_init] Batch size too large (>64)." << std::endl;
    return -1;
  }

  // Set Binding Dimension: [Batch, 3, TILE, TILE]
  handle->trt_context->setInputShape(
      handle->input_name.c_str(),
      nvinfer1::Dims4{batch_size, 3, TILE_SIZE, TILE_SIZE});

  // 7. Allocate Buffers
  // =========================================================================
  DEBUG_LOG("[sr_init] Debug: Allocating buffers");
  // A. FrameBuffer: User Input/Output (Host <-> Device Interface)
  create_frame_buffer(handle->src_yuvframe, ipar.input_width, ipar.input_height,
                      ipar.input_format);
  create_frame_buffer(handle->dst_yuvframe, ipar.output_width,
                      ipar.output_height, ipar.output_format);

  // B. FrameBuffer: Pre-Scaled Input
  // (Used if User Input Size != Model Input Size)
  handle->need_pre_resize = (handle->model_input_width != ipar.input_width) ||
                            (handle->model_input_height != ipar.input_height);

  if (handle->need_pre_resize) {
    // Must be a FrameBuffer because we resize YUV -> YUV
    create_frame_buffer(handle->prescaled_yuvframe, handle->model_input_width,
                        handle->model_input_height, ipar.input_format);
  }

  // C. TensorBuffer: Inference Batch Tiles
  void *dev_ptr = nullptr;
  handle->input_tensor.total_size =
      batch_size * 3 * TILE_SIZE * TILE_SIZE * input_elem_size;
  CUDA_CHECK(cudaMalloc(&dev_ptr, handle->input_tensor.total_size));
  handle->input_tensor.data.reset(dev_ptr);

  dev_ptr = nullptr;
  handle->output_tensor.total_size =
      batch_size * 3 * (TILE_SIZE * 4) * (TILE_SIZE * 4) * output_elem_size;
  CUDA_CHECK(cudaMalloc(&dev_ptr, handle->output_tensor.total_size));
  handle->output_tensor.data.reset(dev_ptr);

  // Bind Tensors
  handle->trt_context->setTensorAddress(handle->input_name.c_str(),
                                        handle->input_tensor.data.get());
  handle->trt_context->setTensorAddress(handle->output_name.c_str(),
                                        handle->output_tensor.data.get());

  DEBUG_LOG("[sr_init] Debug: All successfully finished");
  return 0;
}

int sr_process(sr_handle handle, const sr_frame *src_frame,
               sr_frame *dst_frame) {
  // 1. Validation & Copy
  if (!handle) {
    std::cerr << "[sr_process] Error: Handle is null" << std::endl;
    return -1;
  }
  if (!src_frame || !dst_frame) {
    std::cerr << "[sr_process] Error: Invalid frame parameters." << std::endl;
    return -1;
  }

  if (copy_frame_to_gpu(*src_frame, handle->src_yuvframe,
                        handle->cuda_stream.get()) != 0) {
    return -1;
  }

  // 2. Determine Processing Source (Original vs Resized)
  const FrameBuffer *proc_src = &handle->src_yuvframe;

  if (handle->need_pre_resize) {
    resize_yuv(handle->src_yuvframe, handle->prescaled_yuvframe,
               ResizeMethod::BILINEAR, handle->cuda_stream.get());
    proc_src = &handle->prescaled_yuvframe;
  }

  // 3. Setup Loop Constants
  int batch_limit = (handle->ipar.concurrent_batches > 0)
                        ? handle->ipar.concurrent_batches
                        : 1;
  int total_tiles = handle->num_tiles_in_x * handle->num_tiles_in_y;

  // 4. Prepare Parameters (Invariant parts)
  PreprocessParams pre_params{};
  pre_params.src_frame = proc_src;
  pre_params.dst_tensor = &handle->input_tensor;
  pre_params.overlap_pixels = handle->ipar.overlap_pixels;
  pre_params.video_full_range_flag = (handle->ipar.input_color_fullrange == 1);

  PostprocessParams post_params{};
  post_params.src_tensor = &handle->output_tensor;
  post_params.dst_frame = &handle->dst_yuvframe;
  post_params.overlap_pixels = handle->ipar.overlap_pixels;
  post_params.src_virtual_width = handle->model_output_width;
  post_params.src_virtual_height = handle->model_output_height;
  post_params.video_full_range_flag =
      (handle->ipar.output_color_fullrange == 1);

  // 5. Tiling Loop
  for (int i = 0; i < total_tiles; i += batch_limit) {
    // Calculate actual tiles for this iteration (handle remainder)
    int current_batch = std::min(batch_limit, total_tiles - i);

    // A. Preprocess
    pre_params.start_tile_index = i;
    pre_params.batch_size = current_batch;
    preprocess(pre_params, handle->cuda_stream.get());

    // B. Inference
    handle->trt_context->enqueueV3(handle->cuda_stream.get());

    // C. Postprocess
    post_params.start_tile_index = i;
    post_params.batch_size = current_batch;
    postprocess(post_params, handle->cuda_stream.get());
  }

  // 6. Copy Back & Sync
  if (copy_frame_to_host(handle->dst_yuvframe, *dst_frame,
                         handle->cuda_stream.get()) != 0) {
    return -1;
  }

  CUDA_CHECK(cudaStreamSynchronize(handle->cuda_stream.get()));
  return 0;
}
}