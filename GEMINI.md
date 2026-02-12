# Gemini CLI Context: trt-sr

This repository contains a high-performance TensorRT-accelerated Super Resolution pipeline for video processing.

## Project Overview

The `trt-sr` project leverages NVIDIA TensorRT and CUDA to provide real-time or near-real-time super-resolution capabilities. It is designed to be run within a Docker container to ensure environment consistency and ease of deployment.

### Core Technologies
- **Inference Engine:** NVIDIA TensorRT (via C++ API)
- **Pre/Post-processing:** CUDA kernels (C++/CU)
- **Video I/O:** PyAV (Python wrapper for FFmpeg)
- **Containerization:** Docker with NVIDIA Container Toolkit
- **Communication:** Standard I/O pipes between Python and C++

### Architecture
1. **`libsuperres` (C++):** A library that encapsulates TensorRT logic, including engine building from ONNX, memory management, and CUDA kernels for resizing and color space conversion.
2. **`superres_app` (C++):** A CLI wrapper that reads raw YUV frames from `stdin`, processes them using `libsuperres`, and writes the output to `stdout`.
3. **`superres.py` (Python):** The pipeline orchestrator. It decodes input video, manages a thread to feed frames to the C++ process, and encodes the resulting frames into the final output video.

## Directory Structure
- `cpp/`: C++ source code for the inference engine and CLI app.
  - `libsuperres/`: Core library implementation.
  - `superres/`: CLI tool (`superres_app`).
- `scripts/`: Integration scripts.
  - `superres.py`: Python orchestration script.
  - `run_superres.sh`: Helper to build and run the pipeline inside Docker.
- `docker/`: Dockerfiles for build and runtime environments.
- `models/`: Optimized ONNX models for various SR architectures (Real-ESRGAN, QuickSRNet, SESR, etc.).

## Building and Running

### Prerequisites
- NVIDIA GPU with modern drivers.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
- Docker.

### 1. Build the Docker Image
```bash
docker build -t trt-build -f docker/trt-build.Dockerfile .
```

### 2. Run the Pipeline
Use the provided helper script to process a video. If the TensorRT engine (`.plan` file) doesn't exist for the selected model, it will be built automatically on the first run.
```bash
./scripts/run_superres.sh <input_video_path> <onnx_model_path> <output_video_path> [prescale]
```
- **Example:** `./scripts/run_superres.sh sample-5s.mp4 models/realesrgan_general_x4v3_f16.onnx output.mp4 0.5`

## Development Conventions
- **C++:** Follows C++11 standard. CUDA kernels are used for performance-critical image processing.
- **Python:** Uses `threading` for concurrent I/O feeding to the C++ backend.
- **Docker:** Development typically happens by mounting the local source into the `trt-build` container as seen in `run_superres.sh`.
