#include "CLI11.hpp"
#include "superres.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct AppConfig {
  int mode; // 1: build, 2: process, 3: both
  std::string onnx_file;
  std::string plan_file;
  std::string input_yuv_file;
  std::string output_yuv_file;
  int input_width;
  int input_height;
  std::string input_format_str = "i420";
  sr_pixel_format input_format;
  int input_full_range = 0;

  int output_width = 0; // 0 means same as input
  int output_height = 0;
  std::string output_format_str = "i420";
  sr_pixel_format output_format;
  int output_full_range = 0;

  float prescale = 1.0f;
  int overlap = 4;

  // New fields for updated API
  int concurrent_batches = 1;
  uint64_t max_workspace_size = 0;
  int min_batch_size = 1;
  int max_batch_size = 8;
  int opt_batch_size = 8;
  bool strongly_typed = false;
  bool version_compatible = false;
  bool exclude_lean_runtime = false;
};

// Helper to map string to sr_pixel_format
sr_pixel_format parse_format(const std::string &fmt_str) {
  static const std::map<std::string, sr_pixel_format> fmt_map = {
      {"i420", SR_PIXEL_FMT_I420}, {"nv12", SR_PIXEL_FMT_NV12},
      {"i422", SR_PIXEL_FMT_I422}, {"nv16", SR_PIXEL_FMT_NV16},
      {"i010", SR_PIXEL_FMT_I010}, {"p010", SR_PIXEL_FMT_P010},
      {"i210", SR_PIXEL_FMT_I210}, {"p210", SR_PIXEL_FMT_P210},
      {"i016", SR_PIXEL_FMT_I016}, {"p016", SR_PIXEL_FMT_P016},
      {"i216", SR_PIXEL_FMT_I216}, {"p216", SR_PIXEL_FMT_P216}};

  std::string lower_fmt = fmt_str;
  std::transform(lower_fmt.begin(), lower_fmt.end(), lower_fmt.begin(),
                 ::tolower);

  auto it = fmt_map.find(lower_fmt);
  if (it != fmt_map.end()) {
    return it->second;
  }
  std::cerr << "Unknown format: " << fmt_str << ". Defaulting to I420."
            << std::endl;
  return SR_PIXEL_FMT_I420;
}

AppConfig parse_args(int argc, char **argv) {
  AppConfig config{};

  CLI::App app("Superres");

  // Config file support: automatically read config.ini if it exists in CWD
  app.set_config("--config");
  if (std::ifstream("config.ini").good()) {
    app.set_config("--config", "config.ini");
  }

  app.add_option("-m,--mode", config.mode,
                 "Mode (1: build, 2: process, 3: both)")
      ->required();
  app.add_option("--onnx-file", config.onnx_file, "ONNX file");
  app.add_option("--plan-file", config.plan_file, "Plan file")->required();
  app.add_option("--input-yuv-file", config.input_yuv_file, "Input YUV file");
  app.add_option("--output-yuv-file", config.output_yuv_file,
                 "Output YUV file");
  app.add_option("--width", config.input_width, "Input Width");
  app.add_option("--height", config.input_height, "Input Height");
  app.add_option("--input-format", config.input_format_str,
                 "Input Pixel Format (i420, nv12, etc.)");
  app.add_option("--input-full-range", config.input_full_range,
                 "Input Full Range (0 or 1)");

  app.add_option("--output-width", config.output_width,
                 "Output Width (default: input width)");
  app.add_option("--output-height", config.output_height,
                 "Output Height (default: input height)");
  app.add_option("--output-format", config.output_format_str,
                 "Output Pixel Format (default: i420)");
  app.add_option("--output-full-range", config.output_full_range,
                 "Output Full Range (0 or 1)");

  app.add_option("--prescale", config.prescale,
                 "Prescale factor (0.25 to 1.0)");
  app.add_option("--overlap", config.overlap, "Overlap pixels");

  // New options for updated API
  app.add_option("--batches", config.concurrent_batches,
                 "Concurrent batches (1-64)");
  app.add_option("--workspace", config.max_workspace_size,
                 "Max workspace size (bytes)");
  app.add_option("--min-batch", config.min_batch_size,
                 "Min batch size for engine");
  app.add_option("--max-batch", config.max_batch_size,
                 "Max batch size for engine");
  app.add_option("--opt-batch", config.opt_batch_size,
                 "Optimal batch size for engine");
  app.add_flag("--strongly-typed", config.strongly_typed,
               "Enable strongly typed network");
  app.add_flag("--version-compatible", config.version_compatible,
               "Enable version compatibility");
  app.add_flag("--exclude-lean-runtime", config.exclude_lean_runtime,
               "Exclude lean runtime from plan");

  try {
    app.parse(argc, argv);
  } catch (const CLI::CallForHelp &e) {
    app.exit(e, std::cerr,
             std::cerr); // Use cerr for help to assume user error if needed?
                         // No, user requested. But standard says cout for help.
                         // CLI11 allows passing stream.
  } catch (const CLI::ParseError &e) {
    app.exit(e);
  }

  config.input_format = parse_format(config.input_format_str);
  config.output_format = parse_format(config.output_format_str);

  if (config.output_width == 0)
    config.output_width = config.input_width;
  if (config.output_height == 0)
    config.output_height = config.input_height;

  return config;
}

void print_config(const AppConfig &config) {
  std::cerr << "==========================================" << std::endl;
  std::cerr << "SuperRes App Configuration:" << std::endl;
  std::cerr << "  Mode:                " << config.mode
            << " (1:build, 2:process, 3:both)" << std::endl;
  if (!config.onnx_file.empty())
    std::cerr << "  ONNX File:           " << config.onnx_file << std::endl;
  std::cerr << "  Plan File:           " << config.plan_file << std::endl;
  if (!config.input_yuv_file.empty())
    std::cerr << "  Input YUV:           " << config.input_yuv_file
              << std::endl;
  if (!config.output_yuv_file.empty())
    std::cerr << "  Output YUV:          " << config.output_yuv_file
              << std::endl;
  std::cerr << "  Input Size:          " << config.input_width << "x"
            << config.input_height << " (" << config.input_format_str << ", "
            << (config.input_full_range ? "Full" : "Limited") << ")"
            << std::endl;
  std::cerr << "  Output Size:         " << config.output_width << "x"
            << config.output_height << " (" << config.output_format_str << ", "
            << (config.output_full_range ? "Full" : "Limited") << ")"
            << std::endl;
  std::cerr << "  Prescale:            " << config.prescale << std::endl;
  std::cerr << "  Overlap:             " << config.overlap << std::endl;
  std::cerr << "  Concurrent Batches:  " << config.concurrent_batches
            << std::endl;
  if (config.mode & 1) {
    std::cerr << "  TRT Build Params:" << std::endl;
    std::cerr << "    Workspace:         " << config.max_workspace_size
              << std::endl;
    std::cerr << "    Batch [min,opt,max]: [" << config.min_batch_size << ", "
              << config.opt_batch_size << ", " << config.max_batch_size << "]"
              << std::endl;
    std::cerr << "    Strongly Typed:    "
              << (config.strongly_typed ? "true" : "false") << std::endl;
    std::cerr << "    Version Compatible: "
              << (config.version_compatible ? "true" : "false") << std::endl;
    std::cerr << "    Exclude Lean Runtime: "
              << (config.exclude_lean_runtime ? "true" : "false") << std::endl;
  }
  std::cerr << "==========================================" << std::endl;
}

void build_plan(const AppConfig &config) {
  sr_build_params bpar{};
  bpar.model_onnx = config.onnx_file.c_str();
  bpar.plan_file = config.plan_file.c_str();
  bpar.max_workspace_size = config.max_workspace_size;
  bpar.min_batch_size = config.min_batch_size;
  bpar.max_batch_size = config.max_batch_size;
  bpar.optimal_batch_size = config.opt_batch_size;
  bpar.strongly_typed = config.strongly_typed;
  bpar.version_compatible = config.version_compatible;
  bpar.exclude_lean_runtime = config.exclude_lean_runtime;

  int ret = sr_build(&bpar);
  if (ret != 0) {
    std::cerr << "Build failed." << std::endl;
  }
}

// Calculate frame size in bytes
size_t get_frame_size(int width, int height, sr_pixel_format fmt) {
  // Simplified logic for common formats.
  // Assuming packed planes.
  int bpp_num = 0;
  int bpp_den = 0;

  // Check bit depth
  bool is_16bit = (fmt >= SR_PIXEL_FMT_I010); // 10, 16 bit formats are >= 8
  int bytes_per_comp = is_16bit ? 2 : 1;

  switch (fmt) {
  case SR_PIXEL_FMT_I420:
  case SR_PIXEL_FMT_NV12:
  case SR_PIXEL_FMT_I010:
  case SR_PIXEL_FMT_P010:
  case SR_PIXEL_FMT_I016:
  case SR_PIXEL_FMT_P016:
    // 4:2:0 -> 1.5 pixels per pixel
    bpp_num = 3;
    bpp_den = 2;
    break;
  case SR_PIXEL_FMT_I422:
  case SR_PIXEL_FMT_NV16:
  case SR_PIXEL_FMT_I210:
  case SR_PIXEL_FMT_P210:
  case SR_PIXEL_FMT_I216:
  case SR_PIXEL_FMT_P216:
    // 4:2:2 -> 2 pixels per pixel
    bpp_num = 2;
    bpp_den = 1;
    break;
  default:
    return 0;
  }

  return (size_t)width * height * bpp_num * bytes_per_comp / bpp_den;
}

void fill_frame_pointers(sr_frame &frame, void *buffer, int width, int height,
                         sr_pixel_format fmt) {
  uint8_t *base = static_cast<uint8_t *>(buffer);
  bool is_16bit = (fmt >= SR_PIXEL_FMT_I010);
  int bpc = is_16bit ? 2 : 1;

  // Y Plane always first
  frame.data[0] = base;
  frame.stride[0] = width * bpc;

  size_t y_size = width * height * bpc;

  // Chroma
  if (fmt == SR_PIXEL_FMT_NV12 || fmt == SR_PIXEL_FMT_P010 ||
      fmt == SR_PIXEL_FMT_P016) {
    // NV12/P010/P016: Y followed by UV interleaved
    // UV plane size = W * H/2 * bpc
    // Stride = W * bpc
    frame.data[1] = base + y_size;
    frame.stride[1] = width * bpc;
    frame.data[2] = nullptr;
    frame.stride[2] = 0;
  } else if (fmt == SR_PIXEL_FMT_NV16 || fmt == SR_PIXEL_FMT_P210 ||
             fmt == SR_PIXEL_FMT_P216) {
    // NV16/P210/P216: Y followed by UV interleaved
    // UV plane size = W * H * bpc
    // Stride = W * bpc
    frame.data[1] = base + y_size;
    frame.stride[1] = width * bpc;
    frame.data[2] = nullptr;
    frame.stride[2] = 0;
  } else if (fmt == SR_PIXEL_FMT_I420 || fmt == SR_PIXEL_FMT_I010 ||
             fmt == SR_PIXEL_FMT_I016) {
    // I420/I010/I016: Y, U, V
    // U plane size = W/2 * H/2 * bpc
    // V plane size = W/2 * H/2 * bpc
    // Stride = W/2 * bpc
    size_t u_size = (width / 2) * (height / 2) * bpc;
    frame.data[1] = base + y_size;
    frame.stride[1] = (width / 2) * bpc;
    frame.data[2] = base + y_size + u_size;
    frame.stride[2] = (width / 2) * bpc;
  } else if (fmt == SR_PIXEL_FMT_I422 || fmt == SR_PIXEL_FMT_I210 ||
             fmt == SR_PIXEL_FMT_I216) {
    // I422/I210/I216: Y, U, V
    // U plane size = W/2 * H * bpc
    // Stride = W/2 * bpc
    size_t u_size = (width / 2) * height * bpc;
    frame.data[1] = base + y_size;
    frame.stride[1] = (width / 2) * bpc;
    frame.data[2] = base + y_size + u_size;
    frame.stride[2] = (width / 2) * bpc;
  }
}

void process(const AppConfig &config) {
  sr_handle handle = sr_create();
  if (!handle) {
    std::cerr << "Failed to create sr handle." << std::endl;
    return;
  }
  std::unique_ptr<struct sr_context, decltype(&sr_destroy)> handle_holder(
      handle, sr_destroy);

  sr_init_params ipar{};
  ipar.plan_file = config.plan_file.c_str();
  ipar.input_width = config.input_width;
  ipar.input_height = config.input_height;
  ipar.input_format = config.input_format;
  ipar.input_color_fullrange = config.input_full_range;
  ipar.output_width = config.output_width;
  ipar.output_height = config.output_height;
  ipar.output_format = config.output_format;
  ipar.output_color_fullrange = config.output_full_range;
  ipar.prescale = config.prescale;
  ipar.overlap_pixels = config.overlap;
  ipar.concurrent_batches = config.concurrent_batches;

  if (sr_init(handle, &ipar) != 0) {
    std::cerr << "Failed to init superres" << std::endl;
    return;
  }
  std::cerr << "[DEBUG] Parsed input arg: [" << config.input_yuv_file << "]"
            << std::endl;

  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);

  // 1. 입력 스트림 설정 (stdin 분기)
  std::ifstream file_input;
  if (config.input_yuv_file != "-") {
    file_input.open(config.input_yuv_file, std::ios::binary);
    if (!file_input.is_open()) {
      std::cerr << "Failed to open input file: " << config.input_yuv_file
                << "\n";
      return;
    }
  }

  std::istream &input = (config.input_yuv_file == "-") ? std::cin : file_input;

  // 2. 출력 스트림 설정 (stdout 분기)
  std::ofstream file_output;
  if (config.output_yuv_file != "-") {
    file_output.open(config.output_yuv_file, std::ios::binary);
    if (!file_output.is_open()) {
      std::cerr << "Failed to open output file: " << config.output_yuv_file
                << "\n";
      return;
    }
  }
  std::ostream &output =
      (config.output_yuv_file == "-") ? std::cout : file_output;

  size_t input_size = get_frame_size(config.input_width, config.input_height,
                                     config.input_format);
  size_t output_size = get_frame_size(config.output_width, config.output_height,
                                      config.output_format);

  std::vector<uint8_t> input_buffer(input_size);
  std::vector<uint8_t> output_buffer(output_size);

  int frame_count = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  while (input.read(reinterpret_cast<char *>(input_buffer.data()),
                    input_buffer.size())) {

    sr_frame input_frame{};
    sr_frame output_frame{};
    fill_frame_pointers(input_frame, input_buffer.data(), config.input_width,
                        config.input_height, config.input_format);
    fill_frame_pointers(output_frame, output_buffer.data(), config.output_width,
                        config.output_height, config.output_format);

    if (sr_process(handle, &input_frame, &output_frame) != 0) {
      std::cerr << "Processing frame failed." << std::endl;
      break;
    }
    output.write(reinterpret_cast<char *>(output_buffer.data()),
                 output_buffer.size());
    frame_count++;

    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = current_time - start_time;
    if (frame_count % 10 == 0) { // Print every 10 frames
      std::cerr << "Processed " << frame_count
                << " frames. Average FPS: " << frame_count / elapsed.count()
                << "\r" << std::flush;
    }
  }
  if (input.eof()) {
    std::cerr << "Loop ended due to EOF. gcount: " << input.gcount()
              << std::endl;
  } else if (input.fail()) {
    std::cerr << "Loop ended due to FAIL. gcount: " << input.gcount()
              << std::endl;
  } else {
    std::cerr << "Loop ended unknown reason." << std::endl;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_elapsed = end_time - start_time;
  std::cerr << std::endl
            << "Total processed " << frame_count << " frames in "
            << total_elapsed.count()
            << " seconds. Average FPS: " << frame_count / total_elapsed.count()
            << std::endl;
}

int main(int argc, char **argv) {
  AppConfig config = parse_args(argc, argv);

  print_config(config);

  if (config.mode & 1)
    build_plan(config);

  if (config.mode & 2)
    process(config);

  return 0;
}