#include "CLI11.hpp"
#include "real-esrgan.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <chrono>

struct AppConfig {
  int mode; // 1: build, 2: process, 3: both
  std::string onnx_file;
  std::string plan_file;
  std::string input_yuv_file;
  std::string output_yuv_file;
  int width;
  int height;
};

AppConfig parse_args(int argc, char **argv) {
  AppConfig config{};

  CLI::App app("Superres");
  app.add_option("-m,--mode", config.mode,
                 "Mode (1: build, 2: process, 3: both)")
      ->required();
  app.add_option("--onnx-file", config.onnx_file, "ONNX file");
  app.add_option("--plan-file", config.plan_file, "Plan file")->required();
  app.add_option("--input-yuv-file", config.input_yuv_file, "Input YUV file");
  app.add_option("--output-yuv-file", config.output_yuv_file,
                 "Output YUV file");
  app.add_option("--width", config.width, "Width");
  app.add_option("--height", config.height, "Height");

  try {
    app.parse(argc, argv);
  } catch (const CLI::CallForHelp &e) {
    app.exit(e, std::cerr, std::cerr);
  } catch (const CLI::ParseError &e) {
    app.exit(e);
  }

  return config;
}

void build_plan(const AppConfig &config) {
  int ret =
      real_esrgan_build(config.onnx_file.c_str(), config.plan_file.c_str());
  if (ret != 0) {
    std::cerr << "Build failed." << std::endl;
  }
}

void process(const AppConfig &config) {
  real_esrgan_handle handle = real_esrgan_create();
  if (!handle) {
    std::cerr << "Failed to create real-esrgan handle." << std::endl;
    return;
  }
  std::unique_ptr<struct real_esrgan_context, decltype(&real_esrgan_destroy)>
      handle_holder(handle, real_esrgan_destroy);

  real_esrgan_init_params ipar{};
  ipar.width = config.width;
  ipar.height = config.height;
  ipar.overlap_pixels = 4;
  ipar.plan_file = config.plan_file.c_str();

  if (real_esrgan_init(handle, &ipar) != 0) {
    std::cerr << "Failed to init real_esrgan" << std::endl;
    return;
  }

  std::ifstream input(config.input_yuv_file, std::ios::binary);
  if (!input.is_open()) {
    std::cerr << "Failed to open input file: " << config.input_yuv_file
              << std::endl;
    return;
  }

  std::ofstream output(config.output_yuv_file, std::ios::binary);

  if (!output.is_open()) {
    std::cerr << "Failed to open output file: " << config.output_yuv_file
              << std::endl;
    return;
  }

  std::vector<uint8_t> input_buffer(config.width * config.height * 3 / 2);
  std::vector<uint8_t> output_buffer(config.width * config.height * 3 / 2);

  int frame_count = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  while (input.read(reinterpret_cast<char *>(input_buffer.data()),
                    input_buffer.size())) {
    if (real_esrgan_process(handle, input_buffer.data(),
                            output_buffer.data()) != 0) {
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

  if (config.mode & 1)
    build_plan(config);

  if (config.mode & 2)
    process(config);

  return 0;
}