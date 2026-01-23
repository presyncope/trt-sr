#!/bin/bash

# Helper script to run the dockerized trt-sr pipeline

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_video_path> <onnx_model_path> <output_video_path>"
    echo "Example: $0 ./sample.mp4 ./models/realesrgan.onnx ./output.mp4"
    exit 1
fi

INPUT_HOST_PATH=$(realpath "$1")
ONNX_HOST_PATH=$(realpath "$2")
OUTPUT_HOST_PATH=$(realpath "$3")

# We mount the directories containing the files to keep it simple, 
# or we can mount specific files. Mounting specific files is safer.
# Note: For output, we need to mount the directory so the file can be created.

INPUT_DIR=$(dirname "$INPUT_HOST_PATH")
INPUT_FILE=$(basename "$INPUT_HOST_PATH")

ONNX_DIR=$(dirname "$ONNX_HOST_PATH")
ONNX_FILE=$(basename "$ONNX_HOST_PATH")

OUTPUT_DIR=$(dirname "$OUTPUT_HOST_PATH")
OUTPUT_FILE=$(basename "$OUTPUT_HOST_PATH")

# Create output dir if not exists
mkdir -p "$OUTPUT_DIR"

# Docker run command
# We mount directories to /mnt/input, /mnt/onnx, /mnt/output
docker run --gpus all --rm \
    -v "$INPUT_DIR":/mnt/input \
    -v "$ONNX_DIR":/mnt/onnx \
    -v "$OUTPUT_DIR":/mnt/output \
    trt-sr-pipeline \
    "/mnt/input/$INPUT_FILE" \
    "/mnt/onnx/$ONNX_FILE" \
    "/mnt/output/$OUTPUT_FILE"
