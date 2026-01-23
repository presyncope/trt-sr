# TRT-SR Pipeline via Docker

This repository contains a Dockerized setup for running the TensorRT Super Resolution pipeline.

## Usage

### 1. Build the Image

Build the Docker image using the following command:

```bash
docker build -t trt-sr-pipeline .
```

### 2. Run the Pipeline

You can use the helper script `run_docker.sh` to run the pipeline. This script handles mounting the necessary volumes (Input, Output, Models).

```bash
./run_docker.sh <input_video> <onnx_model> <output_video>
```

**Example:**

```bash
./run_docker.sh sample-5s.mp4 models/realesrgan_general_x4v3_f16.onnx output.mp4
```

This command will:

1. Mount the input video directory, output directory, and models directory to the container.
2. Run the container.
3. Execute the pipeline. If a TensorRT engine (plan) for the model does not exist, it will be built automatically.

### Models

Pre-built ONNX models (optimized for FP16) are available in the `models/` directory.

Common models include:

* `realesrgan_general_x4v3_f16.onnx`
* `realesrgan_x4plus_f16.onnx`
* `quicksrnetsmall_f16.onnx`
* `xlsr_f16.onnx`
* (and others as listed in the directory)
